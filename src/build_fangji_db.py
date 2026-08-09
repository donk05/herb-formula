"""
方剂书 RAG 本地知识库构建脚本
将 data/方剂_books/ 下的 12 本方剂书 TXT 文件向量化存储到独立的 ChromaDB

与古籍库分离，避免单个 chroma_db 文件过大：
  - 古籍：data/chroma_db/（collection: ancient_books）
  - 方剂：data/chroma_db_fangji/（collection: fangji_books）

技术栈（100% 本地 CPU）：
  - 文本切块：RecursiveCharacterTextSplitter (chunk=300, overlap=50)
  - 向量模型：BAAI/bge-small-zh-v1.5
  - 向量数据库：ChromaDB 持久化到 data/chroma_db_fangji/

用法：python src/build_fangji_db.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 强制离线：使用本地缓存的 BGE 模型，避免连 HuggingFace 检查更新卡住
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def main():
    books_dir = os.path.join(_project_root, "data", "方剂_books")
    persist_dir = os.path.join(_project_root, "data", "chroma_db_fangji")
    collection_name = "fangji_books"

    if not os.path.isdir(books_dir):
        print(f"❌ 方剂目录不存在: {books_dir}")
        sys.exit(1)

    txt_files = sorted(f for f in os.listdir(books_dir) if f.endswith(".txt"))
    if not txt_files:
        print(f"❌ 目录下未找到 .txt 文件: {books_dir}")
        sys.exit(1)

    print(f"📖 找到 {len(txt_files)} 本方剂书，开始处理...\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "；", " ", ""],
    )

    all_docs = []

    for fname in txt_files:
        book_name = os.path.splitext(fname)[0]
        fpath = os.path.join(books_dir, fname)

        raw_text = None
        for enc in ("utf-8", "gbk", "gb2312", "gb18030", "latin-1"):
            try:
                with open(fpath, "r", encoding=enc) as f:
                    raw_text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if raw_text is None:
            print(f"  ⚠️ 跳过（无法解码）: {fname}")
            continue

        chunks = splitter.split_text(raw_text)
        for chunk in chunks:
            all_docs.append(
                Document(page_content=chunk, metadata={"book_name": book_name})
            )
        print(f"  ✅ {book_name}  →  {len(chunks)} 个文本块")

    if not all_docs:
        print("\n❌ 未能提取任何有效文本，请检查文件编码。")
        sys.exit(1)

    print(f"\n📊 总计 {len(all_docs)} 个文本块，开始向量化...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 覆盖旧数据
    import shutil
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

    print(f"💾 写入 ChromaDB（{collection_name}）...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    print(f"\n🎉 方剂向量库构建成功！")
    print(f"   书籍数量: {len(txt_files)} 本")
    print(f"   文本块数: {len(all_docs)}")
    print(f"   存储路径: {persist_dir}")


if __name__ == "__main__":
    main()

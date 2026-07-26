"""
古籍 RAG 本地知识库构建脚本
将 data/ancient_books/ 下的 42 本古籍 TXT 文件向量化存储到 ChromaDB

技术栈（100% 本地 CPU，无需任何云端 API）：
  - 文本切块：RecursiveCharacterTextSplitter (chunk=300, overlap=50)
  - 向量模型：BAAI/bge-small-zh-v1.5（HuggingFace 开源，首次运行自动下载）
  - 向量数据库：ChromaDB 持久化到 data/chroma_db/

用法：python src/build_rag_db.py
"""

import os
import sys

# 确保项目根在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def main():
    books_dir = os.path.join(_project_root, "data", "ancient_books")
    persist_dir = os.path.join(_project_root, "data", "chroma_db")

    if not os.path.isdir(books_dir):
        print(f"❌ 古籍目录不存在: {books_dir}")
        sys.exit(1)

    # 收集所有 .txt 文件
    txt_files = sorted(
        f for f in os.listdir(books_dir) if f.endswith(".txt")
    )
    if not txt_files:
        print(f"❌ 目录下未找到 .txt 文件: {books_dir}")
        sys.exit(1)

    print(f"📖 找到 {len(txt_files)} 本古籍，开始处理...\n")

    # 文本切块器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "；", " ", ""],
    )

    all_docs = []

    for fname in txt_files:
        book_name = os.path.splitext(fname)[0]
        fpath = os.path.join(books_dir, fname)

        # 尝试多种编码读取古籍文件
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

        # 切块
        chunks = splitter.split_text(raw_text)
        for chunk in chunks:
            all_docs.append(
                Document(page_content=chunk, metadata={"book_name": book_name})
            )

        print(f"  ✅ {book_name}  →  {len(chunks)} 个文本块")

    if not all_docs:
        print("\n❌ 未能提取任何有效文本，请检查古籍文件编码。")
        sys.exit(1)

    print(f"\n📊 总计 {len(all_docs)} 个文本块，开始向量化...")
    print("🧠 加载模型 BAAI/bge-small-zh-v1.5（首次运行将自动下载到本地）...")

    # 本地 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 构建并持久化 ChromaDB（覆盖旧数据）
    print("💾 写入 ChromaDB ...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="ancient_books",
    )

    print(f"\n🎉 古籍向量库构建成功！")
    print(f"   书籍数量: {len(txt_files)} 本")
    print(f"   文本块数: {len(all_docs)}")
    print(f"   存储路径: {persist_dir}")


if __name__ == "__main__":
    main()

"""
饮料知识 RAG 本地知识库构建脚本
将 data/drink_pdfs/ 下的饮料 / 药食同源相关标准 PDF 向量化存储到独立 ChromaDB

与古籍、方剂、养生库分离，避免混库污染检索：
  - 古籍：data/chroma_db/          （collection: ancient_books）
  - 方剂：data/chroma_db_fangji/   （collection: fangji_books）
  - 养生：data/chroma_db_yangsheng/（collection: yangsheng_books）
  - 饮料：data/chroma_db_drink/    （collection: drink_books）

技术栈（100% 本地 CPU）：
  - PDF 抽取：pypdf（按页抽取，保留页码元数据）
  - 文本切块：RecursiveCharacterTextSplitter (chunk=500, overlap=80)
  - 向量模型：BAAI/bge-small-zh-v1.5
  - 向量数据库：ChromaDB 持久化到 data/chroma_db_drink/

用法：python src/build_drink_db.py
"""

import hashlib
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 强制离线：使用本地缓存的 BGE 模型，避免连 HuggingFace 检查更新卡住
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

_PAGE_NUM = re.compile(r"[-—–\s]*\d{1,4}[-—–\s]*")
# 用于判断文本是否可读：CJK 汉字 + 英文字母 + 数字
_USEFUL_CHAR = re.compile(r"[一-鿿A-Za-z0-9]")


def _usable_ratio(text: str) -> float:
    return len(_USEFUL_CHAR.findall(text)) / max(1, len(text))


def _clean_page(text: str) -> str:
    """清洗单页文本：去掉孤立代理字符/控制符、空行、首尾纯页码行，压缩多余空格。"""
    # 坏字体 PDF 会抽出孤立代理字符，会让 tokenizer 直接抛错
    text = (text or "").encode("utf-8", "ignore").decode("utf-8")
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    while lines and _PAGE_NUM.fullmatch(lines[0]):
        lines.pop(0)
    while lines and _PAGE_NUM.fullmatch(lines[-1]):
        lines.pop()
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def main():
    pdf_dir = os.path.join(_project_root, "data", "drink_pdfs")
    persist_dir = os.path.join(_project_root, "data", "chroma_db_drink")
    collection_name = "drink_books"

    if not os.path.isdir(pdf_dir):
        print(f"❌ PDF 目录不存在: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(pdf_dir)
        for f in files
        if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        print(f"❌ 目录下未找到 .pdf 文件: {pdf_dir}")
        sys.exit(1)

    print(f"📖 找到 {len(pdf_files)} 份 PDF，开始处理...\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    all_docs = []
    skipped = []
    seen_hashes = {}

    for fpath in pdf_files:
        book_name = os.path.splitext(os.path.basename(fpath))[0]
        # 分类取自文件所在目录名（如「饮料」「药食同源」）
        rel_parts = os.path.relpath(fpath, pdf_dir).split(os.sep)
        category = rel_parts[-2] if len(rel_parts) >= 2 else "饮料"

        try:
            reader = PdfReader(fpath)
            page_docs = []
            full_parts = []
            for page_no, page in enumerate(reader.pages, start=1):
                text = _clean_page(page.extract_text() or "")
                if len(text) < 20:
                    continue
                full_parts.append(text)
                for chunk in splitter.split_text(text):
                    chunk = chunk.strip()
                    # 过滤乱码块（字体编码损坏时会是韩文/符号乱码）
                    if not chunk or _usable_ratio(chunk) < 0.3:
                        continue
                    page_docs.append(
                        Document(
                            page_content=f"【{book_name}】{chunk}",
                            metadata={
                                "book_name": book_name,
                                "category": category,
                                "page": page_no,
                            },
                        )
                    )
        except Exception as e:
            print(f"  ⚠️ 跳过（读取失败 {type(e).__name__}）: {os.path.basename(fpath)}")
            skipped.append(os.path.basename(fpath))
            continue

        if not page_docs:
            print(f"  ⚠️ 跳过（无可抽取文字层）: {os.path.basename(fpath)}")
            skipped.append(os.path.basename(fpath))
            continue

        # 按内容哈希去重：同一标准的重复 PDF 只保留一份
        digest = hashlib.md5("".join(full_parts).encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            print(f"  ⚠️ 跳过（内容重复，与「{seen_hashes[digest]}」相同）: {os.path.basename(fpath)}")
            skipped.append(os.path.basename(fpath))
            continue
        seen_hashes[digest] = book_name

        all_docs.extend(page_docs)
        print(f"  ✅ [{category}] {book_name}  →  {len(page_docs)} 个文本块")

    if not all_docs:
        print("\n❌ 未能提取任何有效文本。")
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

    print(f"\n🎉 饮料向量库构建成功！")
    print(f"   PDF 数量: {len(pdf_files) - len(skipped)} 份（跳过 {len(skipped)} 份）")
    print(f"   文本块数: {len(all_docs)}")
    print(f"   存储路径: {persist_dir}")
    if skipped:
        print("   跳过文件:")
        for name in skipped:
            print(f"     - {name}")


if __name__ == "__main__":
    main()

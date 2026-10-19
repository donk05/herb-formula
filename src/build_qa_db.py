"""
问答数据增量建库脚本
将 data/qa_data/ 下的 question.csv + answer.csv 按 question_id 关联合并，
转换为与古籍完全同规格的 RAG Document，增量写入既有古籍向量库
（data/chroma_db，collection: ancient_books）。

技术栈与 src/build_rag_db.py 完全一致（100% 本地 CPU）：
  - 文本切块：RecursiveCharacterTextSplitter (chunk=300, overlap=50, 中文标点)
  - 向量模型：BAAI/bge-small-zh-v1.5
  - 向量数据库：写入既有 ChromaDB data/chroma_db/（纯追加，不新建、不清空）

用法：
  python src/build_qa_db.py [--limit N] [--batch-size 128]

约束：
  - 一次性离线建库工具；Streamlit 启动流程不包含任何 QA 数据处理
  - .rag_version 全程只读，写入前后各校验一次
  - 幂等：已入库的 question_id 自动跳过，支持断点续建
  - 批处理写入（batch embed + batch 落库），禁止逐条写入
"""

import argparse
import csv
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = os.path.join("data", "chroma_db")
QA_DIR = os.path.join("data", "qa_data")
QUESTION_CSV = os.path.join(QA_DIR, "question.csv")
ANSWER_CSV = os.path.join(QA_DIR, "answer.csv")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "ancient_books"
QA_BOOK_NAME = "问答知识库参考资料"
RAG_VERSION = "v2"
DEDUP_PAGE = 10000


def check_chroma_ready():
    """与 app.py _chroma_dir_ready() 同标准的前置检查：目录存在、
    chroma.sqlite3 >10KB、.rag_version == v2。任一不满足拒绝执行。"""
    if not os.path.isdir(CHROMA_DIR):
        return False, f"目录不存在: {CHROMA_DIR}"
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not (os.path.isfile(sqlite_path) and os.path.getsize(sqlite_path) > 10 * 1024):
        return False, "chroma.sqlite3 缺失或过小（<10KB）"
    marker = os.path.join(CHROMA_DIR, ".rag_version")
    try:
        with open(marker, "r", encoding="utf-8") as f:
            if f.read().strip() != RAG_VERSION:
                return False, f".rag_version 内容不是 {RAG_VERSION}"
    except OSError as e:
        return False, f".rag_version 不可读: {e}"
    return True, "OK"


def check_csv_files():
    problems = []
    for path, required in ((QUESTION_CSV, {"question_id", "content"}),
                           (ANSWER_CSV, {"ans_id", "question_id", "content"})):
        if not os.path.isfile(path):
            problems.append(f"缺失文件: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
        except (OSError, StopIteration, csv.Error) as e:
            problems.append(f"无法读取 {path}: {e}")
            continue
        missing = required - set(h.strip() for h in header if h)
        if missing:
            problems.append(f"{os.path.basename(path)} 缺少列: {sorted(missing)}")
    return problems


def load_questions(limit):
    """按文件顺序读取问题；--limit N 时仅取前 N 个有效 question_id。"""
    questions = []
    seen = set()
    empty_rows = 0
    dup_rows = 0
    with open(QUESTION_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = (row.get("question_id") or "").strip()
            content = (row.get("content") or "").strip()
            if not qid or not content:
                empty_rows += 1
                continue
            if qid in seen:
                dup_rows += 1
                continue
            seen.add(qid)
            questions.append((qid, content))
            if limit and len(questions) >= limit:
                break
    return questions, empty_rows, dup_rows


def load_answers(qid_set):
    """流式逐行读取回答并按 question_id 聚合（保持文件出现顺序）。"""
    answers = {}
    total = 0
    out_of_scope = 0
    empty_rows = 0
    with open(ANSWER_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            qid = (row.get("question_id") or "").strip()
            content = (row.get("content") or "").strip()
            if qid not in qid_set:
                out_of_scope += 1
                continue
            if not content:
                empty_rows += 1
                continue
            answers.setdefault(qid, []).append(content)
    return answers, total, out_of_scope, empty_rows


def build_merged_text(question, answers):
    if answers:
        return "问题：\n{}\n参考回答：\n{}".format(question, "\n".join(answers))
    return "问题：\n{}".format(question)


def make_docs(qid, text, splitter):
    return [
        Document(
            page_content=chunk,
            metadata={
                "book_name": QA_BOOK_NAME,
                "source": "qa",
                "question_id": str(qid),
            },
        )
        for chunk in splitter.split_text(text)
    ]


def load_existing_qa_qids(vectorstore):
    existing = set()
    offset = 0
    while True:
        res = vectorstore.get(
            where={"source": "qa"},
            include=["metadatas"],
            limit=DEDUP_PAGE,
            offset=offset,
        )
        metas = res.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            qid = m.get("question_id")
            if qid is not None:
                existing.add(str(qid))
        offset += len(metas)
        if len(metas) < DEDUP_PAGE:
            break
    return existing


def count_qa_entries(vectorstore):
    total = 0
    offset = 0
    while True:
        res = vectorstore.get(where={"source": "qa"}, include=[], limit=DEDUP_PAGE, offset=offset)
        ids = res.get("ids") or []
        if not ids:
            break
        total += len(ids)
        offset += len(ids)
        if len(ids) < DEDUP_PAGE:
            break
    return total


def retrieval_self_check(vectorstore, questions, answers):
    print("\n🔎 检索自检（只读）...")
    ok = True
    sample = questions[:5]
    qa_hit = 0
    for qid, qtext in sample:
        docs = vectorstore.similarity_search(qtext, k=3)
        hit = any(d.metadata.get("source") == "qa" for d in docs)
        qa_hit += 1 if hit else 0
        print(f"  问答命中: {'✅' if hit else '❌'} 「{qtext[:40]}」 -> "
              f"{[d.metadata.get('book_name') for d in docs]}")
    if qa_hit == 0:
        ok = False
        print("  ❌ 试点问题全部未命中 QA 片段")
    for kw in ("人参", "本草"):
        docs = vectorstore.similarity_search(kw, k=5)
        non_qa = [d for d in docs if d.metadata.get("source") != "qa"]
        names = [d.metadata.get("book_name") for d in docs]
        if non_qa:
            print(f"  古籍命中: ✅ 「{kw}」 -> {names}")
        else:
            print(f"  古籍命中: ⚠️ 「{kw}」前5均为QA片段（QA规模扩大后的预期现象） -> {names}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="问答数据增量写入既有古籍 Chroma 知识库")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅处理 CSV 文件顺序前 N 个 question_id（0=全量）")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="每批写入的 question_id 数（默认 128）")
    args = parser.parse_args()

    print("📖 QA 数据增量建库")
    print(f"   模式: {'试点（前 %d 个问题）' % args.limit if args.limit else '全量'}，批大小 {args.batch_size}")

    ok, reason = check_chroma_ready()
    if not ok:
        print(f"❌ 既有向量库未就绪: {reason}")
        print("   请先启动一次 Streamlit（streamlit run app.py）完成云端古籍库同步后重试。")
        sys.exit(1)
    print("✅ 既有向量库就绪: data/chroma_db / .rag_version=v2")

    problems = check_csv_files()
    if problems:
        for p in problems:
            print(f"❌ {p}")
        print("   未写入任何数据。")
        sys.exit(1)

    t0 = time.time()
    try:
        questions, q_empty, q_dup = load_questions(args.limit)
    except (OSError, csv.Error) as e:
        print(f"❌ question.csv 读取失败: {e}")
        print("   未写入任何数据。")
        sys.exit(1)
    try:
        answers, a_total, a_out, a_empty = load_answers({q for q, _ in questions})
    except (OSError, csv.Error) as e:
        print(f"❌ answer.csv 读取失败: {e}")
        print("   未写入任何数据。")
        sys.exit(1)

    no_answer = sum(1 for qid, _ in questions if qid not in answers)
    print(f"📊 数据统计: 有效问题 {len(questions)}（空行 {q_empty}，重复 {q_dup}）| "
          f"回答 {a_total}（本次范围外 {a_out}，空文本 {a_empty}）| 无回答问题 {no_answer}")

    print(f"🧠 加载 Embedding 模型 {EMBEDDING_MODEL}（cuda, normalize）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("💾 打开既有 Chroma 向量库（只读检查）...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    total_before = vectorstore._collection.count()
    qa_before = count_qa_entries(vectorstore)
    existing_qids = load_existing_qa_qids(vectorstore)
    pending = [(qid, content) for qid, content in questions if qid not in existing_qids]
    skipped = len(questions) - len(pending)
    print(f"   collection 总条数: {total_before}（QA 条数 {qa_before}）")
    print(f"   已入库问题 {len(existing_qids)}（本次跳过 {skipped}），待写入 {len(pending)} 个问题")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "；", " ", ""],
    )

    done_qids = 0
    done_docs = 0
    failed_batches = []
    t_write = time.time()
    for i in range(0, len(pending), args.batch_size):
        batch = pending[i:i + args.batch_size]
        docs = []
        for qid, qtext in batch:
            docs.extend(make_docs(qid, build_merged_text(qtext, answers.get(qid)), splitter))
        try:
            vectorstore.add_documents(docs)
        except Exception as e:
            failed_batches.append((batch[0][0], f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  ⚠️ 批次写入失败（起始 qid={batch[0][0]}），已记录并跳过: "
                  f"{type(e).__name__}: {str(e)[:150]}", flush=True)
            continue
        done_qids += len(batch)
        done_docs += len(docs)
        elapsed = time.time() - t_write
        rate = done_qids / elapsed if elapsed > 0 else 0.0
        remain = (len(pending) - done_qids) / rate if rate > 0 else 0.0
        pct = done_qids * 100 // len(pending) if pending else 100
        print(f"[{time.strftime('%H:%M:%S')}] 进度 {done_qids}/{len(pending)} ({pct}%) | "
              f"本批 {len(batch)} 问题 / {len(docs)} 文档 | 速率 {rate:.1f} qid/s | "
              f"预计剩余 {remain / 60:.1f} 分钟", flush=True)

    total_after = vectorstore._collection.count()
    qa_after = count_qa_entries(vectorstore)
    print(f"\n🧪 .rag_version 写入后二次校验（只读）...")
    try:
        with open(os.path.join(CHROMA_DIR, ".rag_version"), "r", encoding="utf-8") as f:
            after = f.read().strip()
        if after != RAG_VERSION:
            print(f"❌❌ 高优先级告警: .rag_version 变为 {after!r}（脚本从未写入该文件，请立即排查！）")
        else:
            print(f"✅ .rag_version 仍为 {RAG_VERSION}")
    except OSError as e:
        print(f"❌❌ 高优先级告警: .rag_version 不可读: {e}")

    elapsed_total = time.time() - t0
    print("\n🎉 QA 数据增量入库摘要")
    print(f"   入库问题数: {done_qids}（文档数 {done_docs}）")
    print(f"   跳过已入库: {skipped}")
    print(f"   collection 总条数: {total_before} -> {total_after}（QA 条数 {qa_before} -> {qa_after}，增量 +{qa_after - qa_before}）")
    print(f"   总耗时: {elapsed_total / 60:.1f} 分钟")
    print(f"   存储路径: {CHROMA_DIR}（collection: {COLLECTION_NAME}）")
    if failed_batches:
        print(f"   ⚠️ 失败批次 {len(failed_batches)} 个（起始 qid: "
              f"{[q for q, _ in failed_batches][:10]}），重跑本脚本可断点续建。")
    if pending and done_qids:
        retrieval_self_check(vectorstore, pending, answers)
    elif not pending:
        print("\nℹ️ 无待写入数据（全部已入库），跳过写入与自检。")

    sys.exit(2 if failed_batches else 0)


if __name__ == "__main__":
    main()

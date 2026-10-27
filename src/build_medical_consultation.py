"""
问诊数据集向量库构建脚本（FAISS + SQLite，不依赖 Chroma）

数据来源：
  data/answer_data/question.csv  (question_id, content)
  data/answer_data/answer.csv    (ans_id, question_id, content)

与 Chroma 方案相比：
  - 不建中文全文索引（省掉约一半体积）
  - 正文只在 SQLite 存一份，并按行 zlib 压缩
  - 向量用 FAISS int8 标量量化，约 61MB / 12 万条

输出（默认 data/chroma_db_medical_consultation/）：
  medical.sqlite  表 qa(id, question_id, question, answers[zlib], n_answers)
  faiss.index     512 维 int8 内积索引
  meta.json       模型/维度/条数等

用法：
  python src/build_medical_consultation.py            # 全量
  python src/build_medical_consultation.py --limit 2000  # 小样本验证
"""

import argparse
import json
import os
import sqlite3
import sys
import zlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

DIM = 512
EMBED_ANSWER_COUNT = 3      # 编码时最多用前 3 条回答
EMBED_ANSWER_CHARS = 300    # 每条回答编码时最多取 300 字
EMBED_TOTAL_CHARS = 800


def _build_embed_text(question: str, answers: list) -> str:
    parts = [f"问题：{question}"]
    for a in answers[:EMBED_ANSWER_COUNT]:
        parts.append(f"回答：{a[:EMBED_ANSWER_CHARS]}")
    return "\n".join(parts)[:EMBED_TOTAL_CHARS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全量）")
    parser.add_argument("--out", default=os.path.join(_project_root, "data", "chroma_db_medical_consultation"))
    args = parser.parse_args()

    csv_dir = os.path.join(_project_root, "data", "answer_data")
    q_path = os.path.join(csv_dir, "question.csv")
    a_path = os.path.join(csv_dir, "answer.csv")
    if not (os.path.isfile(q_path) and os.path.isfile(a_path)):
        print("❌ 未找到 data/answer_data/question.csv 或 answer.csv")
        sys.exit(1)

    print("📖 读取 CSV ...")
    q = pd.read_csv(q_path, encoding="utf-8")
    a = pd.read_csv(a_path, encoding="utf-8").sort_values("ans_id")
    if args.limit > 0:
        q = q.head(args.limit)
    print(f"   问题 {len(q):,} 条，回答 {len(a):,} 条")

    print("🔗 按 question_id 关联 ...")
    answer_map = a.groupby("question_id")["content"].apply(list).to_dict()

    rows = []          # (id, question_id, question, answers_compressed, n_answers)
    texts = []         # 用于编码
    for i, (_, r) in enumerate(q.iterrows()):
        qid = str(r["question_id"])
        question = str(r["content"])
        answers = [str(x) for x in answer_map.get(r["question_id"], [])]
        blob = zlib.compress(
            json.dumps(answers, ensure_ascii=False).encode("utf-8"), level=6
        )
        rows.append((i, qid, question, blob, len(answers)))
        texts.append(_build_embed_text(question, answers))

    n = len(texts)
    if n == 0:
        print("❌ 没有可处理的数据")
        sys.exit(1)

    print(f"🧠 编码 {n:,} 条文本（CPU，请耐心等待）...")
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )

    vectors = np.zeros((n, DIM), dtype="float32")
    batch = 5000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        vectors[start:end] = emb.embed_documents(texts[start:end])
        print(f"   {end:,}/{n:,}")

    print("📦 构建 FAISS int8 索引 ...")
    import faiss
    index = faiss.IndexScalarQuantizer(
        DIM, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    rng = np.random.default_rng(42)
    train_n = min(n, 50000)
    sample_idx = rng.choice(n, train_n, replace=False) if train_n < n else np.arange(n)
    index.train(np.ascontiguousarray(vectors[sample_idx]))
    index.add(np.ascontiguousarray(vectors))
    print(f"   索引条数: {index.ntotal:,}")

    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, "medical.sqlite")
    idx_path = os.path.join(args.out, "faiss.index")

    print("💾 写入 SQLite ...")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE qa ("
        "id INTEGER PRIMARY KEY, question_id TEXT, question TEXT, "
        "answers BLOB, n_answers INTEGER)"
    )
    conn.executemany("INSERT INTO qa VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    # 用 serialize + Python 写文件，避免 FAISS C++ 在非 ASCII 路径下报 Illegal byte sequence
    with open(idx_path, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())

    meta = {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": DIM,
        "metric": "inner_product",
        "index_type": "IndexScalarQuantizer(QT_8bit)",
        "count": n,
        "embed_answers": EMBED_ANSWER_COUNT,
        "version": 1,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n🎉 构建完成")
    for name in ("medical.sqlite", "faiss.index", "meta.json"):
        p = os.path.join(args.out, name)
        print(f"   {name:16s} {os.path.getsize(p)/1048576:8.2f} MB")


if __name__ == "__main__":
    main()

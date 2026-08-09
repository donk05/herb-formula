"""
将知网文献 papers_China.xlsx 追加合并进 literature.db
- 为 papers 表新增 source 列：已有记录标为 PubMed，知网新记录标为 CNKI
- 关键词分隔符统一为逗号

用法：python src/merge_cnki_papers.py
"""

import os
import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH = os.path.join(_project_root, "data", "papers_China.xlsx")
DB_PATH = os.path.join(_project_root, "data", "literature.db")


def main():
    if not os.path.exists(XLSX_PATH):
        print(f"❌ 找不到文件: {XLSX_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. 确保 source 列存在，并为旧数据标记 PubMed
    cols = [r[1] for r in cur.execute("PRAGMA table_info(papers)").fetchall()]
    if "source" not in cols:
        print("➕ 新增 source 列，旧数据标记为 PubMed ...")
        cur.execute("ALTER TABLE papers ADD COLUMN source TEXT DEFAULT 'PubMed'")
    else:
        print("ℹ️ source 列已存在")

    # 2. 读取知网数据
    print(f"📖 读取 {XLSX_PATH} ...")
    df = pd.read_excel(XLSX_PATH)

    required = {"中药", "疾病", "标题"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ xlsx 缺少必需列: {missing}")
        sys.exit(1)

    print(f"   共 {len(df):,} 行，开始导入...")

    # 3. 批量插入
    batch = []
    batch_size = 5000
    count = 0

    for _, row in df.iterrows():
        herb = str(row.get("中药") or "").strip()
        disease = str(row.get("疾病") or "").strip()
        title = str(row.get("标题") or "").strip()
        authors = str(row.get("作者") or "").strip()
        journal = str(row.get("期刊") or "").strip()
        year = row.get("年份")
        abstract = str(row.get("摘要") or "").strip()
        keywords = str(row.get("关键词") or "").strip()
        url = str(row.get("URL") or "").strip()

        if not herb or not title:
            continue

        # 年份转整数
        try:
            year = int(year) if year is not None and str(year) != "nan" else None
        except (ValueError, TypeError):
            year = None

        # 关键词分隔符统一（知网用分号/顿号 → 逗号）
        keywords = keywords.replace("；", ",").replace(";", ",").replace("、", ",").replace("，", ",")

        # 跳过重复（同 herb+disease+title）
        batch.append((herb, disease, title, authors, journal, year, abstract, keywords, url, "CNKI"))

        if len(batch) >= batch_size:
            count += _insert_batch(cur, batch)
            print(f"  已导入 {count:,} 条...")
            batch = []

    if batch:
        count += _insert_batch(cur, batch)

    conn.commit()
    conn.close()
    print(f"\n🎉 导入完成，共新增 {count:,} 条 CNKI 文献")


def _insert_batch(cur, rows):
    cur.executemany(
        "INSERT OR IGNORE INTO papers "
        "(herb, disease, title, authors, journal, year, abstract, keywords, url, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return cur.rowcount if cur.rowcount > 0 else len(rows)


if __name__ == "__main__":
    main()

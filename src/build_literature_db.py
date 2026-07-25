"""
文献数据清洗与建库脚本
将 literature_data.json（97 万行文献数据）转化为 SQLite 数据库

用法：python src/build_literature_db.py
"""

import json
import sqlite3
import os
import sys


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "literature_data.json")
    db_path = os.path.join(project_root, "data", "literature.db")

    if not os.path.exists(json_path):
        print(f"❌ 找不到文件: {json_path}")
        print("   请将 literature_data.json 放在项目根目录下再运行。")
        sys.exit(1)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 建表（先删旧表，确保全新重建）
    cursor.execute("DROP TABLE IF EXISTS papers")
    cursor.execute("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            herb TEXT NOT NULL,
            disease TEXT NOT NULL,
            title TEXT,
            authors TEXT,
            journal TEXT,
            year INTEGER,
            abstract TEXT,
            keywords TEXT,
            url TEXT
        )
    """)

    # 读取 JSON
    print(f"📖 读取 {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 适配不同 JSON 结构
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("data") or data.get("records") or data.get("papers") or []
    else:
        records = []

    if not records:
        print("⚠️  JSON 数据为空或格式无法识别，已创建空表。")
        conn.close()
        return

    print(f"   共 {len(records):,} 条记录，开始导入...")

    batch = []
    count = 0
    batch_size = 5000

    for item in records:
        herb = item.get("herb") or item.get("herb_name") or item.get("中药") or ""
        disease = item.get("disease") or item.get("disease_name") or item.get("疾病") or ""

        # paper 子对象（实际数据是嵌套结构）
        paper = item.get("paper") or {}
        if not isinstance(paper, dict):
            paper = {}

        title = paper.get("title") or item.get("title") or ""
        authors = paper.get("authors") or item.get("authors") or ""
        journal = paper.get("journal") or item.get("journal") or ""
        year = paper.get("year") or item.get("year") or None
        abstract = paper.get("abstract") or item.get("abstract") or ""
        keywords = paper.get("keywords") or item.get("keywords") or []
        url = paper.get("url") or paper.get("doi") or item.get("url") or ""

        # year 转为整数
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        # keywords 数组 → 逗号分隔字符串
        if isinstance(keywords, list):
            keywords = ", ".join(str(k) for k in keywords if k)
        keywords = keywords or ""

        batch.append((herb, disease, title, authors, journal, year, abstract, keywords, url))

        if len(batch) >= batch_size:
            cursor.executemany(
                "INSERT INTO papers (herb, disease, title, authors, journal, year, abstract, keywords, url) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            count += len(batch)
            print(f"  已插入 {count:,} 条...")
            batch = []

    # 处理剩余批次
    if batch:
        cursor.executemany(
            "INSERT INTO papers (herb, disease, title, authors, journal, year, abstract, keywords, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"✅ 数据导入完成，共 {count:,} 条记录")

    # 创建联合索引
    print("🔧 创建联合索引 idx_herb_disease ...")
    cursor.execute("DROP INDEX IF EXISTS idx_herb_disease")
    cursor.execute("CREATE INDEX idx_herb_disease ON papers(herb, disease);")
    conn.commit()

    conn.close()
    print(f"🎉 全部完成！数据库文件: {db_path}")


if __name__ == "__main__":
    main()

"""
清理 literature.db 中知网文献的脏数据
- keywords='nan' / 含 '相似文献' → 置空
- abstract 末尾的 ' 更多 还原 AbstractFilter(...)' → 移除

用法：python src/clean_cnki_data.py
"""

import os
import re
import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_project_root, "data", "literature.db")

# AbstractFilter 残留：... 更多 还原 AbstractFilter('ChDivSummary2',...);
_FILTER_RE = re.compile(r"更多\s*还原\s*AbstractFilter\([^)]*\);?\s*$")
_JUNK_KW = {"nan", "None", "相似文献"}


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. 清理 keywords
    cur.execute("SELECT id, keywords FROM papers WHERE source='CNKI'")
    rows = cur.fetchall()
    kw_fixed = 0
    for pid, kw in rows:
        if not kw:
            continue
        fixed = kw.strip()
        # 去 nan/相似文献
        if fixed.lower() in _JUNK_KW:
            fixed = ""
        else:
            # 移除字段里的 '相似文献' 片段
            parts = [p.strip() for p in fixed.split(",") if p.strip().lower() not in _JUNK_KW]
            fixed = ", ".join(parts)
        if fixed != kw:
            cur.execute("UPDATE papers SET keywords=? WHERE id=?", (fixed, pid))
            kw_fixed += 1

    # 2. 清理 abstract 末尾残留
    cur.execute("SELECT id, abstract FROM papers WHERE source='CNKI'")
    rows = cur.fetchall()
    ab_fixed = 0
    for pid, ab in rows:
        if not ab:
            continue
        fixed = _FILTER_RE.sub("", ab).rstrip()
        if fixed != ab:
            cur.execute("UPDATE papers SET abstract=? WHERE id=?", (fixed, pid))
            ab_fixed += 1

    conn.commit()
    conn.close()
    print(f"✅ 清理完成")
    print(f"   修复 keywords: {kw_fixed} 条")
    print(f"   修复 abstract: {ab_fixed} 条")


if __name__ == "__main__":
    main()

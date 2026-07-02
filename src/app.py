"""
药食同源饮品辅助配方推荐系统 —— Streamlit 入口兼容文件。

当前正式前端仍位于项目根目录 app.py。
如果希望按 src/app.py 的结构启动，可以执行：
    streamlit run src/app.py

本文件会转发执行根目录 app.py，避免破坏现有的：
    streamlit run app.py
"""

import os
import runpy


def main() -> None:
    """转发到项目根目录的 Streamlit 页面。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_app = os.path.join(project_root, "app.py")
    runpy.run_path(root_app, run_name="__main__")


if __name__ == "__main__":
    main()

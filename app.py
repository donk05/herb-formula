"""
药食同源饮品配方推荐系统 —— 知识图谱版 V2
基于 中药→化合物→靶点→疾病 网络药理学关系链
"""

import sys, os
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path: sys.path.insert(0, _project_root)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import json, time, urllib.request, urllib.error, base64, sqlite3, zipfile, shutil
from difflib import SequenceMatcher
from src.data_loader import GraphDataLoader, CN_TO_EN_DISEASE
from src.disease_advice import get_disease_advice

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="药食同源智能配方", page_icon="🌿", layout="wide")

# ==================== CSS ====================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* === 全局 === */
body, .stApp { background: linear-gradient(180deg, #FDFBF7 0%, #F5F0E6 100%); }
.main .block-container { padding-top: 1rem; max-width: 1200px; }
.stApp::before { content:""; position:fixed; top:0; left:0; right:0; height:4px;
  background: linear-gradient(90deg, #A5D6A7 0%, #2E7D32 25%, #43A047 50%, #1B5E20 75%, #A5D6A7 100%);
  z-index:9999; pointer-events:none; }
h1,h2,h3,h4,h5 { font-family: 'Inter', sans-serif; color: #1B5E20 !important; }
p,span,div { font-family: 'Inter', sans-serif; color: #333; }

/* === 分割线 === */
hr, [data-testid="stDivider"] { border:none!important; height:1px!important;
  background: linear-gradient(90deg, transparent 5%, #C8D6B8 30%, #8DA87B 50%, #C8D6B8 70%, transparent 95%)!important;
  margin: 2rem 0!important; }

/* === 侧边栏 === */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #F7F4EC 0%, #EDE7D8 100%);
  border-right: 1px solid #D8CFB8; }
section[data-testid="stSidebar"] input { border-radius: 10px!important; border: 1.5px solid #C8BFAA!important; }
section[data-testid="stSidebar"] input:focus { border-color: #2E7D32!important; box-shadow: 0 0 0 3px rgba(46,125,50,0.12)!important; }

/* === 按钮 === */
div.stButton > button { background: linear-gradient(135deg, #2E7D32, #43A047);
  color: #FFF!important; border: none; border-radius: 14px; padding: 0.7rem 1.6rem;
  font-size: 1rem; font-weight: 700; letter-spacing: 0.5px;
  box-shadow: 0 4px 16px rgba(46,125,50,0.32), 0 1px 4px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.18);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
div.stButton > button:hover { background: linear-gradient(135deg, #1B5E20, #388E3C);
  box-shadow: 0 6px 24px rgba(46,125,50,0.44), 0 2px 8px rgba(0,0,0,0.12);
  transform: translateY(-2px); }
div.stButton > button:disabled { background: #D4D4D4; box-shadow: 0 2px 6px rgba(0,0,0,0.04); color: #AAA!important; }

/* === KPI 卡片 === */
div[data-testid="stMetric"] { background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.4rem;
  box-shadow: 0 2px 16px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.03);
  border-top: 4px solid #2E7D32; transition: all 0.3s; position: relative; overflow: hidden; }
div[data-testid="stMetric"]:hover { box-shadow: 0 8px 28px rgba(0,0,0,0.10); transform: translateY(-3px); }
div[data-testid="stMetric"]:nth-child(2) { border-top-color: #E67E22; }
div[data-testid="stMetric"]:nth-child(3) { border-top-color: #2980B9; }
div[data-testid="stMetric"]:nth-child(4) { border-top-color: #8E44AD; }
div[data-testid="stMetric"] label { color: #888!important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.8px; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-weight: 800; font-size: 1.6rem; }

/* === Hero Banner === */
.hero-banner { background: linear-gradient(135deg, #E8F5E9 0%, #FDFBF7 30%, #F5F0E6 60%, #E8F5E9 100%);
  border-radius: 24px; padding: 2.5rem 2rem; margin-bottom: 2rem; text-align: center;
  box-shadow: 0 4px 30px rgba(46,125,50,0.08), inset 0 0 80px rgba(165,214,167,0.18);
  border: 1px solid rgba(165,214,167,0.3); position: relative; overflow: hidden; }
.hero-banner::before { content:""; position:absolute; top:-40px; left:-40px; width:140px; height:140px;
  background: radial-gradient(circle, rgba(46,125,50,0.08) 0%, transparent 70%); border-radius:50%; pointer-events:none; }
.hero-banner::after { content:""; position:absolute; bottom:-30px; right:-30px; width:160px; height:160px;
  background: radial-gradient(circle, rgba(139,195,74,0.06) 0%, transparent 70%); border-radius:50%; pointer-events:none; }
.hero-title { font-size: 2.8rem; font-weight: 800; color: #1B5E20; margin-bottom: 0.5rem;
  z-index: 1; position: relative; letter-spacing: 2px; }
.hero-subtitle { font-size: 1.08rem; color: #7A7A7A; z-index: 1; position: relative;
  line-height: 1.8; font-weight: 400; max-width: 600px; margin: 0 auto; }

/* === 中药卡片 === */
.herb-top3 { display: inline-block; width: 32px; height: 32px; line-height: 32px; text-align: center;
  border-radius: 50%; font-weight: 800; font-size: 1rem; color: #FFF; margin-right: 8px; }
.herb-card { background: #FFFFFF; border-radius: 18px; padding: 1.3rem 1.5rem; margin-bottom: 0.85rem;
  box-shadow: 0 3px 20px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.03);
  border-left: 6px solid #2E7D32; transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
  display: flex; align-items: center; gap: 1rem; position: relative; overflow: hidden; }
.herb-card::after { content:""; position:absolute; top:0; right:0; width:80px; height:80px;
  background: radial-gradient(circle at top right, rgba(46,125,50,0.03), transparent); }
.herb-card:hover { box-shadow: 0 10px 36px rgba(0,0,0,0.10), 0 0 0 1px rgba(46,125,50,0.1);
  transform: translateY(-2px); }
.herb-card .rank-num { font-size: 1.6rem; font-weight: 800; min-width: 40px; text-align: center; }
.herb-card .info { flex: 1; }
.herb-card .herb-name { font-size: 1.25rem; font-weight: 700; color: #1B5E20; }
.herb-card .herb-stats { font-size: 0.88rem; color: #777; margin-top: 0.2rem; display: flex; gap: 1rem; }
.herb-card .herb-stats span { display: inline-flex; align-items: center; gap: 4px; }
.herb-card .herb-score { font-size: 1.1rem; font-weight: 700; color: #E67E22; min-width: 50px; text-align: right; }

/* === 进度条 === */
.progress-bar { height: 4px; border-radius: 2px; background: #E0E0E0; margin-top: 0.35rem; overflow: hidden; }
.progress-bar .fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #2E7D32, #43A047);
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }

/* === 中药图片 === */
.herb-img-wrap { flex-shrink: 0; width: 72px; height: 72px; border-radius: 12px;
  overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 0 4px; }
.herb-img { width: 100%; height: 100%; object-fit: cover; display: block; }
.herb-img-placeholder { width: 72px; height: 72px; border-radius: 12px;
  background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; border: 1px dashed #C8E6C9; }

/* === 靶点 chips === */
.evidence-chip { display: inline-block; background: #E8F5E9; color: #2E7D32;
  padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; margin-right: 5px; margin-top: 4px;
  border: 1px solid #C8E6C9; font-weight: 500; }

/* === AI 建议 === */
.ai-card { background: linear-gradient(135deg, #F0F7EE 0%, #FDFBF7 100%); border-radius: 18px;
  padding: 1.6rem 1.8rem; border: 1.5px solid rgba(46,125,50,0.15);
  box-shadow: 0 2px 16px rgba(0,0,0,0.04); line-height: 1.9; }
.ai-badge { background: linear-gradient(135deg, #2E7D32, #43A047); color: #FFF;
  display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem; }

/* === 展开面板 === */
details[data-testid="stExpander"] { border-radius: 14px!important; border: 1.5px solid #E0D8C8!important;
  background: #FFF!important; box-shadow: 0 2px 10px rgba(0,0,0,0.03); }

/* === 搜索提示 === */
.search-hint { background: linear-gradient(135deg, #FFF8E1, #FFF3CD); border-radius: 12px;
  padding: 0.9rem 1.2rem; border: 1.5px solid #FFE082; font-size: 0.9rem; color: #795548; }

/* === 页脚 === */
.footer-note { text-align:center; color: #B0B0B0; padding: 1rem 0; font-size: 0.85rem; }

/* === 滚动条 === */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #F5F0E6; }
::-webkit-scrollbar-thumb { background: #C8D6B8; border-radius: 4px; }

/* === Selectbox === */
div[data-baseweb="select"] > div { border-radius: 10px!important; border-color: #C8BFAA!important; }

/* === 膳食助手聊天容器 === */
.chat-container {
  position: relative;
  overflow: hidden;
  z-index: 0;
  background: linear-gradient(160deg, rgba(255,255,255,0.72) 0%, rgba(245,242,235,0.58) 100%);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 24px;
  padding: 1.6rem 1.4rem 1.2rem;
  margin: 1.4rem 0;
  border: 1.5px solid rgba(129,199,132,0.35);
  box-shadow: 0 0 32px rgba(129,199,132,0.10), 0 6px 24px rgba(0,0,0,0.04),
              inset 0 1px 0 rgba(255,255,255,0.6);
  animation: dietChatGlow 4s ease-in-out infinite;
}
@keyframes dietChatGlow {
  0%, 100% { box-shadow: 0 0 32px rgba(129,199,132,0.08), 0 6px 24px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.6); border-color: rgba(129,199,132,0.28); }
  50% { box-shadow: 0 0 56px rgba(129,199,132,0.20), 0 6px 28px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.7); border-color: rgba(129,199,132,0.50); }
}

/* === 绿色荧光标题框（标题 + 搜索框） === */
.diet-header-box {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  padding: 1.4rem 1.4rem 1rem;
  margin: 1rem 0;
  background: radial-gradient(ellipse at 70% 30%, rgba(139,195,74,0.18) 0%, rgba(200,230,180,0.08) 40%, rgba(255,255,255,0.5) 100%);
  border: 1.5px solid rgba(129,199,132,0.35);
  box-shadow: 0 0 32px rgba(129,199,132,0.08), 0 4px 16px rgba(0,0,0,0.03);
}
.diet-header-box::before {
  content: "";
  position: absolute;
  top: -40px; right: -30px;
  width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(139,195,74,0.10) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
  z-index: 0;
}
.diet-header-box h3 {
  margin-top: 0;
}

/* === 顶部搜索框：绿色荧光椭圆背景 === */
.chat-search-wrap {
  position: relative;
  margin-bottom: 1.2rem;
  padding: 1rem 1.1rem;
  border-radius: 20px;
  background: radial-gradient(ellipse at 70% 30%, rgba(139,195,74,0.18) 0%, rgba(200,230,180,0.08) 40%, rgba(255,255,255,0.35) 100%);
  border: 1.5px solid rgba(129,199,132,0.30);
}
.chat-search-wrap::before {
  content: "";
  position: absolute;
  top: -30px; right: -20px;
  width: 140px; height: 140px;
  background: radial-gradient(circle, rgba(139,195,74,0.12) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
  z-index: -1;
}

/* === 微信风格用户消息气泡（右对齐，头像在右，绿色气泡） === */
.wechat-user-row {
  display: flex;
  justify-content: flex-end;
  align-items: flex-start;
  margin-bottom: 1rem;
  gap: 10px;
}
.wechat-user-bubble {
  max-width: 75%;
  background: #95ec69;
  color: #111;
  padding: 0.7rem 1rem;
  border-radius: 16px 4px 16px 16px;
  font-size: 0.93rem;
  line-height: 1.7;
  word-break: break-word;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.wechat-user-avatar {
  font-size: 1.5rem;
  line-height: 1;
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* === 内联输入区域 === */
[data-testid="stForm"] {
  margin-top: 0;
  border: none !important;
  padding: 0 !important;
}
.search-box [data-testid="stTextInput"] input,
.search-box input {
  border-radius: 14px !important;
  border: 3px solid #2E7D32 !important;
  outline: 2px solid rgba(46,125,50,0.6) !important;
  outline-offset: 2px !important;
  background: #FFFFFF !important;
  box-shadow: 0 0 24px rgba(46,125,50,0.28), 0 4px 16px rgba(0,0,0,0.10) !important;
  transition: all 0.3s ease !important;
  font-size: 1rem !important;
  padding: 0.7rem 1.2rem !important;
}
.search-box [data-testid="stTextInput"] input:hover,
.search-box input:hover {
  border-color: #1B5E20 !important;
  outline-color: rgba(27,94,32,0.8) !important;
  box-shadow: 0 0 32px rgba(46,125,50,0.38), 0 6px 20px rgba(0,0,0,0.14) !important;
  transform: translateY(-1px);
}
.search-box [data-testid="stTextInput"] input:focus,
.search-box input:focus {
  border-color: #1B5E20 !important;
  outline-color: rgba(27,94,32,0.9) !important;
  box-shadow: 0 0 40px rgba(46,125,50,0.48), 0 0 0 8px rgba(46,125,50,0.18) !important;
}
.chat-container button[kind="primary"] {
  min-height: 44px;
  font-size: 1.15rem;
}

</style>""", unsafe_allow_html=True)

# ==================== 数据加载 ====================
@st.cache_resource(ttl=3600)
def get_loader(): return GraphDataLoader()

loader = get_loader()
all_diseases = loader.all_diseases_cn
all_diseases_default = loader.all_diseases_cn_quality

def fuzzy_search(query, candidates, top_k=8):
    if not query: return candidates[:top_k]
    scored = [(c, SequenceMatcher(None, query.lower(), c.lower()).ratio() + (0.5 if query.lower() in c.lower() else 0)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_k] if s > 0]

# ==================== 文献知识检索 ====================
_LITERATURE_DB_URL = (
    "https://github.com/donk05/herb-formula/releases/download/V1.1/literature.db"
)
_LITERATURE_DB_PATH = os.path.join(_project_root, "data", "literature.db")


_LITERATURE_ATTEMPTED = False


def _ensure_literature_db():
    """确保 literature.db 存在，首次自动从 GitHub Releases 下载"""
    global _LITERATURE_ATTEMPTED, _RAG_ERROR_MSG
    if os.path.exists(_LITERATURE_DB_PATH):
        return True
    if _LITERATURE_ATTEMPTED:
        return False
    _LITERATURE_ATTEMPTED = True
    try:
        import requests
        with st.spinner("正在同步文献数据库（约 130MB），请稍候..."):
            resp = requests.get(
                _LITERATURE_DB_URL,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
                timeout=180,
            )
            resp.raise_for_status()
            with open(_LITERATURE_DB_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        # 校验下载的数据库有效
        if not os.path.exists(_LITERATURE_DB_PATH) or os.path.getsize(_LITERATURE_DB_PATH) < 5 * 1024 * 1024:
            raise RuntimeError("下载的文献库文件无效或过小")
        return True
    except Exception as e:
        _RAG_ERROR_MSG = f"❌ 文献库下载失败: {type(e).__name__} - {str(e)[:200]}"
        if os.path.exists(_LITERATURE_DB_PATH):
            try:
                os.remove(_LITERATURE_DB_PATH)
            except OSError:
                pass
        return False


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_paper_from_db(herb: str, disease: str):
    """从 SQLite 查询与指定中药-疾病相关的文献，最多返回 4 篇（PubMed/知网各 2 篇）"""
    _ensure_literature_db()
    db_path = _LITERATURE_DB_PATH
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        where_sql = "herb=? AND (disease=? OR ? LIKE '%' || disease || '%')"
        params = (herb, disease, disease)

        # 两来源各取 2 条，保证知网有同等展示机会
        pubmed = [dict(r) for r in conn.execute(
            f"SELECT title, abstract, keywords, url, source FROM papers "
            f"WHERE {where_sql} AND source='PubMed' LIMIT 2", params
        )]
        cnki = [dict(r) for r in conn.execute(
            f"SELECT title, abstract, keywords, url, source FROM papers "
            f"WHERE {where_sql} AND source='CNKI' LIMIT 2", params
        )]
        conn.close()

        # 交替混合
        rows = []
        for p, c in zip(pubmed, cnki):
            rows.append(p)
            rows.append(c)
        # 补齐剩余（若某个来源不足 2 条）
        for p in pubmed[len(cnki):]:
            rows.append(p)
        for c in cnki[len(pubmed):]:
            rows.append(c)
        return rows[:4]
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def has_literature(herb: str, disease: str) -> bool:
    """轻量级预检：该中药-疾病组合在文献库中是否有记录"""
    _ensure_literature_db()
    db_path = _LITERATURE_DB_PATH
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT 1 FROM papers WHERE herb=? "
            "AND (disease=? OR ? LIKE '%' || disease || '%') LIMIT 1",
            (herb, disease, disease),
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result
    except Exception:
        return False

# ==================== 古籍 RAG 知识库 ====================
_CHROMA_DOWNLOAD_URL = (
    "https://github.com/donk05/herb-formula/releases/download/v1.0/chroma_db.zip"
)
_DB_DIR = os.path.join(_project_root, "data", "chroma_db")
_ZIP_PATH = os.path.join(_project_root, "data", "chroma_db.zip")
_DATA_DIR = os.path.join(_project_root, "data")

# 全局变量：记录下载过程中的错误（主页面直接显示，不会错过）
_RAG_ERROR_MSG = None

# 版本标记：每次重新打包上传后递增，旧库无此标记会自动清理重下
_RAG_VERSION = "v2"
_RAG_MARKER = os.path.join(_DB_DIR, ".rag_version")


def _chroma_dir_ready() -> bool:
    """要求目录存在、chroma.sqlite3 非空，且带当前版本标记"""
    if not os.path.isdir(_DB_DIR):
        return False
    sqlite_path = os.path.join(_DB_DIR, "chroma.sqlite3")
    has_sqlite = os.path.isfile(sqlite_path)
    try:
        sqlite_ok = has_sqlite and os.path.getsize(sqlite_path) > 10 * 1024
    except OSError:
        sqlite_ok = False
    # 版本标记必须与当前一致
    marker_ok = False
    try:
        with open(_RAG_MARKER, "r") as mf:
            marker_ok = mf.read().strip() == _RAG_VERSION
    except OSError:
        marker_ok = False

    if sqlite_ok and marker_ok:
        return True

    # 无效或过期 → 清理后重新下载
    try:
        import shutil
        shutil.rmtree(_DB_DIR, ignore_errors=True)
    except OSError:
        pass
    return False


def _download_chroma_db():
    """分块下载，带 UA + 重定向 + 长超时"""
    try:
        import requests
    except ImportError:
        raise RuntimeError("缺少 requests 库，请确认 requirements.txt 已包含 requests")
    resp = requests.get(
        _CHROMA_DOWNLOAD_URL,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
        allow_redirects=True,
        timeout=180,
    )
    resp.raise_for_status()
    with open(_ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _fix_nested_chroma_dir():
    nested = os.path.join(_DB_DIR, "chroma_db")
    if not os.path.isdir(nested):
        return
    import shutil
    for entry in os.listdir(nested):
        src = os.path.join(nested, entry)
        dst = os.path.join(_DB_DIR, entry)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
    os.rmdir(nested)


def _ensure_rag_downloaded():
    """返回 True 表示下载成功或已存在，False 表示失败"""
    global _RAG_ERROR_MSG
    if _chroma_dir_ready():
        return True
    try:
        with st.spinner("正在首次同步云端古籍核心数据库（约 171MB），请稍候..."):
            _download_chroma_db()

        zip_size = os.path.getsize(_ZIP_PATH)
        if zip_size < 5 * 1024 * 1024:
            raise RuntimeError(
                f"下载的文件仅 {zip_size / 1024:.0f} KB，非有效压缩包。"
                f"请确认 GitHub Release 链接正确且文件已上传。"
            )

        with st.spinner("正在解压古籍数据库..."):
            with zipfile.ZipFile(_ZIP_PATH, "r") as zf:
                zf.extractall(_DATA_DIR)
            os.remove(_ZIP_PATH)

        _fix_nested_chroma_dir()

        # 解压成功后写入版本标记，避免下次误判为空库
        try:
            with open(_RAG_MARKER, "w") as mf:
                mf.write(_RAG_VERSION)
        except OSError:
            pass
        return True

    except Exception as e:
        _RAG_ERROR_MSG = f"❌ 古籍库同步失败: {type(e).__name__} - {str(e)[:300]}"
        if os.path.exists(_ZIP_PATH):
            try:
                os.remove(_ZIP_PATH)
            except OSError:
                pass
        return False


@st.cache_resource(show_spinner=False)
def _get_rag_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def _load_chroma_ready():
    """加载 ChromaDB（仅当 chroma_db 目录就绪后调用），结果常驻缓存"""
    from langchain_chroma import Chroma
    return Chroma(
        persist_directory=_DB_DIR,
        embedding_function=_get_rag_embeddings(),
        collection_name="ancient_books",
    )


def load_rag_db():
    """返回 Chroma 实例或 None；下载失败不会污染缓存，可重试"""
    global _RAG_ERROR_MSG
    if not _ensure_rag_downloaded():
        return None
    try:
        return _load_chroma_ready()
    except Exception as e:
        _RAG_ERROR_MSG = f"❌ ChromaDB 初始化失败: {type(e).__name__} - {str(e)[:300]}"
        return None


def retrieve_ancient_books(query: str, k: int = 3):
    db = load_rag_db()
    if db is None:
        return []
    try:
        docs = db.similarity_search(query, k=k)
    except Exception as e:
        _RAG_ERROR_MSG = f"❌ 古籍检索失败: {type(e).__name__} - {str(e)[:200]}"
        return []
    return [
        {"content": doc.page_content, "book_name": doc.metadata.get("book_name", "佚名")}
        for doc in docs
    ]

# ==================== Gemini 膳食助手 API ====================
DIET_SYSTEM_INSTRUCTION = (
    '你是一位专注于「药食同源」与「大众营养膳食」的温和科普助手。'
    '你的核心纪律：\n'
    '1. 严禁推荐任何处方药、非处方药、烟酒、或具有毒副作用的危险中药。\n'
    '2. 只提供温和、健康、日常的食疗建议（如：多吃膳食纤维、温水冲饮、多吃新鲜蔬果、保持规律作息）。\n'
    '3. 语言必须干净、阳光、积极向上、通俗易懂，适合包括青少年在内的全年龄段人群。\n'
    '4. 所有回答必须附带温馨提示：「本建议仅为日常膳食营养科普，不作为临床医疗诊断依据，如有身体不适请及时就医。」'
    '5. 优先结合中医「药食同源」理念，推荐山药、枸杞、红枣、薏米、桂圆、莲子、百合、茯苓等常见食材。'
)


# 中药图片文件夹
_HERB_IMG_DIR = os.path.join(_project_root, "（高清版）106种药食同源带介绍")
_HERB_IMG_FILES = None  # 延迟加载


def _load_img_files():
    """加载图片文件列表。"""
    global _HERB_IMG_FILES
    if _HERB_IMG_FILES is not None:
        return
    if not os.path.isdir(_HERB_IMG_DIR):
        _HERB_IMG_FILES = []
        return
    _HERB_IMG_FILES = [
        os.path.splitext(f)[0]
        for f in os.listdir(_HERB_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


def get_herb_image(herb_name):
    """查找中药对应图片，返回 base64 HTML 标签或占位符。"""
    _load_img_files()

    matched = None
    for fname in _HERB_IMG_FILES:
        if fname == herb_name:
            matched = fname
            break
    if not matched:
        for fname in _HERB_IMG_FILES:
            if fname.startswith(herb_name + "（"):
                matched = fname
                break
    if not matched:
        for fname in _HERB_IMG_FILES:
            if herb_name.startswith(fname):
                matched = fname
                break

    if not matched:
        return '<div class="herb-img-placeholder">🌿</div>'

    # 找到实际文件（处理扩展名）
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        img_path = os.path.join(_HERB_IMG_DIR, matched + ext)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext_mime = ext.replace(".", "")
                if ext_mime == "jpg":
                    ext_mime = "jpeg"
                return (
                    f'<img src="data:image/{ext_mime};base64,{b64}" '
                    f'class="herb-img" alt="{herb_name}" loading="lazy">'
                )
            except Exception:
                pass

    return '<div class="herb-img-placeholder">🌿</div>'


def generate_herb_circular_graph(herb_name, disease_name, chain_data):
    """使用 pyecharts 生成环形知识图谱：疾病←靶点←化合物←中药。"""
    from pyecharts.charts import Graph
    from pyecharts import options as opts

    compounds = chain_data.get("compounds", [])
    targets = chain_data.get("targets", [])
    comp_target_map = chain_data.get("compound_target_map", {})

    nodes = []
    links = []

    # --- 中心节点 ---
    nodes.append({"name": disease_name, "symbolSize": 30,
                  "itemStyle": {"color": "#FFB74D"}})   # 橙黄 — 疾病
    nodes.append({"name": herb_name, "symbolSize": 25,
                  "itemStyle": {"color": "#42A5F5"}})   # 蓝色 — 中药

    # --- 化合物节点（绿色）---
    for cid, cname in compounds:
        label = cname if cname and cname != cid else cid
        # 截断过长名称
        if len(label) > 18:
            label = label[:16] + "..."
        nodes.append({"name": cid, "symbolSize": 15,
                      "itemStyle": {"color": "#66BB6A"},
                      "label": {"formatter": label}})
        # 中药 → 化合物
        links.append({"source": herb_name, "target": cid})

    # --- 靶点节点（红色）---
    for tid, tname in targets:
        label = tname if tname and tname != tid else tid
        if len(label) > 14:
            label = label[:12] + "..."
        nodes.append({"name": tid, "symbolSize": 12,
                      "itemStyle": {"color": "#EF5350"},
                      "label": {"formatter": label}})
        # 靶点 → 疾病
        links.append({"source": tid, "target": disease_name})

    # --- 化合物 → 靶点 连线 ---
    for cid, tlist in comp_target_map.items():
        for tid, _ in tlist:
            links.append({"source": cid, "target": tid})

    graph = (
        Graph(init_opts=opts.InitOpts(
            width="100%", height="520px",
            bg_color="rgba(0,0,0,0)",  # 透明背景
        ))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=[
                {"name": disease_name},
                {"name": herb_name},
            ],
            layout="circular",
            is_rotate_label=True,
            is_draggable=True,
            edge_symbol=["none", "arrow"],
            edge_length=[50, 180],
            linestyle_opts=opts.LineStyleOpts(
                curve=0.3, width=1.2, opacity=0.65,
            ),
            label_opts=opts.LabelOpts(
                position="right", font_size=11,
                font_family="Microsoft YaHei, sans-serif",
            ),
            repulsion=600,
            gravity=0.15,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"🔬 {herb_name} ↔ {disease_name} 分子机制图谱",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=15, font_family="Microsoft YaHei, sans-serif",
                    color="#1B5E20",
                ),
                pos_left="center",
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    return graph


def _diet_fallback(disease_context=""):
    """当 Gemini API 不可用时的通用提示。"""
    disease_hint = f"当前查询疾病：「{disease_context}」。" if disease_context else ""

    return (
        f"{disease_hint}"
        "AI 膳食助手暂未配置 API 密钥，目前无法提供针对性的膳食建议。\n\n"
        "💡 **温馨提示**：请在环境变量或 `.streamlit/secrets.toml` 中配置 `GEMINI_API_KEY`，"
        "即可启用基于 Gemini 2.5 Flash 的智能膳食分析。\n\n"
        "---\n"
        "📝 本建议仅为日常膳食营养科普，不作为临床医疗诊断依据，如有身体不适请及时就医。"
    )


def ask_gemini_diet_assistant(messages, disease_context="", rag_context=""):
    """向 DeepSeek API 发送请求，可选结合古籍 RAG 上下文。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
        except Exception:
            api_key = ""

    if not api_key:
        return _diet_fallback(disease_context)

    system_instruction = DIET_SYSTEM_INSTRUCTION
    if rag_context:
        system_instruction += (
            "\n\n你是一位精通传统中医和现代健康的调理专家。请严格结合以下古籍原典，"
            "用通俗易懂的白话文回答用户的亚健康调理问题。如果古籍中未提及，"
            "请基于你自己的中医知识库进行补充，但要说明。\n\n"
            "【检索到的古籍记载】：\n" + rag_context
        )
    if disease_context:
        system_instruction += f"\n\n{disease_context}"

    if not api_key:
        return _diet_fallback(disease_context)

    # 构建 Groq API 请求体（OpenAI 兼容格式）
    api_messages = [{"role": "system", "content": system_instruction}]
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        api_messages.append({"role": role, "content": msg["content"]})

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": api_messages,
        "temperature": 0.7,
        "max_tokens": 800,
    }).encode("utf-8")

    url = "https://api.deepseek.com/v1/chat/completions"

    last_error = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_error = f"{e.code} {e.reason}"
            try:
                err_body = e.read().decode("utf-8")[:500]
                last_error += f" — {err_body}"
            except Exception:
                pass
            if e.code == 429 and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            last_error = str(e)[:500]
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue

    return f"❌ API 调用失败（已重试 5 次）\n\n错误信息：{last_error}\n\n---\n📝 请检查 API Key 是否有效或网络是否正常。"

    return _diet_fallback(disease_context)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0;'>🌿 知识图谱检索</h3>", unsafe_allow_html=True)
    st.caption("100 种药食同源 × 8300+ 疾病")
    st.markdown("---")

    search_query = st.text_input(
        "🔍 搜索疾病", placeholder="输入中文或英文疾病名...",
        label_visibility="collapsed",
    )
    if search_query.strip():
        matched = fuzzy_search(search_query.strip(), all_diseases, top_k=15)
        default_idx = 0
    else:
        matched = all_diseases_default
        default_idx = matched.index("高血压") if "高血压" in matched else 0

    if matched:
        selected_disease = st.selectbox("匹配结果（{}条）".format(len(matched)), options=matched, index=default_idx, label_visibility="collapsed")
    else:
        selected_disease = None
        st.markdown('<div class="search-hint">🔎 未找到匹配，试试其他关键词</div>', unsafe_allow_html=True)

    st.markdown("---")
    top_k = st.slider("📊 展示 Top N 中药", 5, 30, 15)
    generate_btn = st.button("🌿 查询知识图谱", type="primary", use_container_width=True, disabled=(selected_disease is None))

    # 将查询参数持久化到 session_state，防止 chat_input 重跑时丢失
    if generate_btn and selected_disease:
        st.session_state.query_disease = selected_disease
        st.session_state.query_top_k = top_k
        st.session_state.query_active = True

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 刷新缓存", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    with c2:
        st.caption(f"默认 {len(all_diseases_default)} 种疾病可查")

# ==================== 主页面 - Hero ====================
st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-title">🌿 药食同源智能配方推荐</div>'
    '<div class="hero-subtitle">基于网络药理学知识图谱，深度挖掘 中药 → 化合物 → 靶点 → 疾病 多层次关系链<br>'
    '为 1800+ 种疾病智能匹配最优药食同源中药组合，赋能精准健康决策</div>'
    '</div>', unsafe_allow_html=True,
)

# 初始化查询持久化状态
if "query_active" not in st.session_state:
    st.session_state.query_active = False

if not st.session_state.query_active:
    if selected_disease is None:
        st.info("👈 请在左侧搜索并选择一种疾病，然后点击「🌿 查询知识图谱」按钮")
    else:
        st.info("👈 请点击「🌿 查询知识图谱」按钮开始分析")
    st.stop()

# 从 session_state 取持久化的查询参数（chat_input 重跑时不会丢失）
selected_disease = st.session_state.query_disease
top_k = st.session_state.query_top_k

# ==================== 查询 ====================
with st.spinner("🌿 知识图谱检索中，深度分析疾病-靶点-化合物-中药关系链…"):
    stats = loader.get_graph_stats(selected_disease)
    ranked = loader.rank_herbs_for_disease(selected_disease, top_k=top_k)

if not ranked:
    st.warning(f"知识图谱中未找到与「{selected_disease}」直接关联的中药数据。")
    st.stop()

en_name = loader.cn_to_en.get(selected_disease) or CN_TO_EN_DISEASE.get(selected_disease, selected_disease)
max_targets = ranked[0]["关联靶点数"]

# ==================== KPI 卡片（彩色顶线） ====================
st.markdown("### 📊 图谱检索概览")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔬 关联靶点", f"{stats['关联靶点数']} 个")
c2.metric("🧪 关联化合物", f"{stats['关联化合物数']} 个")
c3.metric("🌱 相关中药", f"{stats['相关中药数']} 种")
c4.metric("📋 疾病英文名", en_name[:25])

# ==================== 中药排名 + 图表 ====================
st.markdown(f"### 🏆 「{selected_disease}」关联中药 Top {min(top_k, len(ranked))}")
st.caption(f"按靶点覆盖度排序，共 {stats['相关中药数']} 种药食同源中药与该疾病在分子层面存在关联")

# 在中药卡片渲染前确保文献库已下载，避免卡片查询缓存到空结果
_ensure_literature_db()

left_col, right_col = st.columns([5, 4], gap="large")

with left_col:
    for i, herb in enumerate(ranked):
        # 奖牌颜色
        if i == 0:
            medal_c, rank_bg, border_c = "🥇", "linear-gradient(135deg, #FFF8E1, #FFF3E0)", "#FF8F00"
        elif i == 1:
            medal_c, rank_bg, border_c = "🥈", "linear-gradient(135deg, #FAFAFA, #F5F5F5)", "#9E9E9E"
        elif i == 2:
            medal_c, rank_bg, border_c = "🥉", "linear-gradient(135deg, #FFF3E0, #FBE9E7)", "#BF360C"
        else:
            medal_c, rank_bg, border_c = f"<span style='color:#999;font-size:1.1rem'>{i+1}</span>", "#FFFFFF", "#E0E0E0"

        pct = round(herb["关联靶点数"] / max_targets * 100) if max_targets else 0
        evi_chips = "".join(f'<span class="evidence-chip">{t}</span>' for t, c in herb["证据链"][:4])

        herb_img = get_herb_image(herb["中药名"])

        st.markdown(
            f'<div class="herb-card" style="border-left-color:{border_c};background:{rank_bg};">'
            f'<div style="font-size:1.8rem;min-width:44px;text-align:center;">{medal_c}</div>'
            f'<div class="herb-img-wrap">{herb_img}</div>'
            f'<div class="info">'
            f'<div class="herb-name">{herb["中药名"]}</div>'
            f'<div class="herb-stats">'
            f'<span>🔬 靶点 <b>{herb["关联靶点数"]}</b></span>'
            f'<span>🧪 化合物 <b>{herb["关联化合物数"]}</b></span>'
            f'</div>'
            f'<div class="progress-bar"><div class="fill" style="width:{pct}%"></div></div>'
            f'<div style="margin-top:6px">{evi_chips}</div>'
            f'</div>'
            f'<div class="herb-score">{pct}%</div>'
            f'</div>', unsafe_allow_html=True,
        )

        # 环形知识图谱 Expander
        with st.expander(f"🧬 查看【{herb['中药名']}】专属机制图谱"):
            chain = loader.get_herb_disease_chain(
                herb["中药名"], selected_disease,
                max_ingredients=8, max_genes=12,
            )
            if chain["compounds"] and chain["targets"]:
                chart = generate_herb_circular_graph(
                    herb["中药名"], selected_disease, chain,
                )
                from streamlit_echarts import st_pyecharts
                st_pyecharts(chart, height="540px")
            else:
                st.caption("该中药暂无分子层面关联数据")

        # 文献知识 Expander
        herb_name = herb["中药名"]
        has_lit = has_literature(herb_name, selected_disease)
        expander_title = (
            f"📚 查看【{herb_name}】文献知识    🟢 已收录"
            if has_lit else
            f"📚 查看【{herb_name}】文献知识    🔴 暂无"
        )
        with st.expander(expander_title):
            if has_lit:
                papers = fetch_paper_from_db(herb_name, selected_disease)
                for idx, paper in enumerate(papers):
                    if idx > 0:
                        st.divider()
                    title = paper.get("title") or "无标题"
                    src = paper.get("source") or ""
                    if src == "CNKI":
                        src_badge = ('<span style="font-size:0.75rem;color:#fff;background:#E65100;'
                                     'padding:2px 8px;border-radius:10px;margin-left:6px;">知网</span>')
                    elif src == "PubMed":
                        src_badge = ('<span style="font-size:0.75rem;color:#fff;background:#1565C0;'
                                     'padding:2px 8px;border-radius:10px;margin-left:6px;">PubMed</span>')
                    else:
                        src_badge = ""
                    st.markdown(f"**{title}**{src_badge}", unsafe_allow_html=True)
                    kw = paper.get("keywords")
                    if kw:
                        tags = " | ".join(
                            f"🏷️ {t.strip()}" for t in kw.split(",") if t.strip()
                        )
                        st.markdown(tags)
                    abstract = paper.get("abstract")
                    if abstract:
                        escaped = abstract.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        st.markdown(
                            f'<div style="font-size:0.88rem;color:#666;line-height:1.7;'
                            f'white-space:pre-wrap;border-left:3px solid #C8E6C9;'
                            f'padding-left:12px;margin:8px 0;">{escaped}</div>',
                            unsafe_allow_html=True,
                        )
                    url = paper.get("url")
                    if url:
                        st.link_button("🌐 查看原文", url)
            else:
                st.info(
                    "💡 提示：该特定药-病组合在当前核心文献库中暂无直接收录。"
                    "系统基于其已知活性成分与靶点网络进行协同推理推荐。"
                )

with right_col:
    # 饼图：Top 8 中药占比
    st.markdown("#### 🍩 Top 8 靶点覆盖分布")
    top8 = ranked[:8]
    labels = [h["中药名"] for h in top8]
    values = [h["关联靶点数"] for h in top8]
    palette = ["#2E7D32", "#43A047", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9", "#FFB74D", "#FF9800"]

    fig1, ax1 = plt.subplots(figsize=(4.2, 4.2))
    fig1.patch.set_facecolor("none"); ax1.set_facecolor("none")
    wedges, texts, autotexts = ax1.pie(
        values, labels=None, autopct="%1.1f%%", colors=palette[:len(labels)],
        startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "antialiased": True},
    )
    for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("#333")
    ax1.legend(wedges, labels, title="中药名", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax1.set_title("靶点覆盖度分布", fontsize=12, fontweight="bold", color="#1B5E20", pad=12)
    st.pyplot(fig1)

    # 柱状图：Top 10
    st.markdown("#### 📊 Top 10 关联强度")
    top10 = ranked[:10]
    names = [h["中药名"] for h in reversed(top10)]
    tv = [h["关联靶点数"] for h in reversed(top10)]
    cv = [h["关联化合物数"] for h in reversed(top10)]

    fig2, ax2 = plt.subplots(figsize=(4.5, 4.2))
    fig2.patch.set_facecolor("none"); ax2.set_facecolor("none")
    y = range(len(names))
    ax2.barh([yi + 0.2 for yi in y], tv, 0.38, color="#2E7D32", alpha=0.9, label="靶点数", edgecolor="white", linewidth=0.5)
    ax2.barh([yi - 0.2 for yi in y], cv, 0.38, color="#A5D6A7", alpha=0.85, label="化合物数", edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=9)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax2.set_xlabel("数量", fontsize=9, color="#777")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.15, color="#999")
    st.pyplot(fig2)

# ==================== AI 健康建议 ====================
st.markdown("---")
st.markdown("### 🤖 AI 智能健康建议")
st.caption("内置知识库 + Groq 免费 AI 实时生成，覆盖 1800+ 种疾病")

graph_ctx = ""
if stats and ranked:
    herbs_top = ranked[:5]
    graph_ctx = (
        f"知识图谱数据：该疾病关联 {stats['关联靶点数']} 个蛋白质靶点、"
        f"{stats['关联化合物数']} 种活性化合物，涉及 {stats['相关中药数']} 种药食同源中药。"
        f"图谱推荐前五：{'、'.join(h['中药名'] for h in herbs_top)}。"
    )

advice = get_disease_advice(selected_disease, graph_ctx)
if advice:
    if "AI建议" in advice:
        st.markdown(f'<div class="ai-card"><span class="ai-badge">🤖 {advice.get("来源", "AI")}</span><br>{advice["AI建议"]}</div>', unsafe_allow_html=True)
    else:
        if "概述" in advice:
            st.markdown(f'<div class="ai-card" style="margin-bottom:1rem"><b>📖 {advice["概述"]}</b></div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### ⚠️ 注意事项")
            for item in advice.get("注意事项", []):
                st.markdown(f'<div style="background:#FFF8F0;border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;font-size:0.9rem;color:#555;border:1.5px solid #F0E0D0;">⚠️ {item}</div>', unsafe_allow_html=True)
        with cb:
            st.markdown("#### 🥗 推荐饮食")
            for item in advice.get("推荐饮食", []):
                st.markdown(f'<div style="background:#F0F8F0;border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;font-size:0.9rem;color:#555;border:1.5px solid #D0E8D0;">🥬 {item}</div>', unsafe_allow_html=True)
        if advice.get("生活建议"):
            st.markdown("#### 🏃 生活建议")
            for i, item in enumerate(advice["生活建议"]):
                st.markdown(f'<div style="background:#F5F0FA;border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.3rem;font-size:0.88rem;color:#555;border:1.5px solid #E0D5F0;display:inline-block;margin-right:8px;">💡 {item}</div>', unsafe_allow_html=True)
else:
    st.info("该疾病暂无健康建议数据。Groq AI 密钥未配置或调用失败。")

# ==================== 药食同源 · AI 健康膳食助手 ====================
st.markdown("---")

# 初始化聊天历史
if "diet_messages" not in st.session_state:
    st.session_state.diet_messages = []

st.markdown("### 🍽️ 亚健康调理建议")
st.caption("基于知识图谱 + AI 大模型，为您提供个性化的药食同源膳食方案")

# RAG 状态指示器
_rag_db_instance = load_rag_db()
_rag_loaded = _rag_db_instance is not None
if _rag_loaded:
    try:
        _rag_count = _rag_db_instance._collection.count()
    except Exception:
        _rag_count = -1
    st.markdown(
        f'<span style="font-size:0.82rem;color:#2E7D32;background:#E8F5E9;'
        f'padding:3px 10px;border-radius:12px;">📚 古籍知识库已就绪'
        f'（{_rag_count} 条）</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span style="font-size:0.82rem;color:#888;background:#F5F5F5;'
        'padding:3px 10px;border-radius:12px;">📚 古籍知识库未加载（纯 AI 模式）</span>',
        unsafe_allow_html=True,
    )
    if _RAG_ERROR_MSG:
        st.error(_RAG_ERROR_MSG)

# 文献库状态指示器
_lit_ok = _ensure_literature_db()
if _lit_ok:
    st.markdown(
        '<span style="font-size:0.82rem;color:#1565C0;background:#E3F2FD;'
        'padding:3px 10px;border-radius:12px;margin-left:6px;">📄 文献库已就绪</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span style="font-size:0.82rem;color:#888;background:#F5F5F5;'
        'padding:3px 10px;border-radius:12px;margin-left:6px;">📄 文献库未加载</span>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="search-box">', unsafe_allow_html=True)
with st.form("diet_chat_form", clear_on_submit=True, border=False):
    cols = st.columns([10, 2], gap="small")
    with cols[0]:
        user_input = st.text_input(
            "输入",
            placeholder="💬 询问关于该亚健康状态的日常膳食调理建议...",
            label_visibility="collapsed",
            key="diet_chat_input",
        )
    with cols[1]:
        submitted = st.form_submit_button("🔍 搜索", use_container_width=True, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input.strip():
    prompt = user_input.strip()
    st.session_state.diet_messages.append({"role": "user", "content": prompt})
    with st.spinner("🌿 AI 正在检索古籍并生成膳食建议..."):
        # RAG 检索古籍
        rag_docs = retrieve_ancient_books(prompt, k=3)
        st.session_state.diet_rag_docs = rag_docs

        # 构建古籍上下文
        rag_context = ""
        if rag_docs:
            lines = []
            for i, doc in enumerate(rag_docs, 1):
                lines.append(f"【古籍 {i}】《{doc['book_name']}》记载：{doc['content']}")
            rag_context = "\n\n".join(lines)

        herbs_top = ranked[:5] if ranked else []
        disease_ctx = (
            f"当前查询疾病：「{selected_disease}」\n"
            f"知识图谱：{stats['关联靶点数']} 个蛋白质靶点、{stats['关联化合物数']} 种化合物、{stats['相关中药数']} 种药食同源中药。\n"
            f"图谱 Top 5 推荐：{'、'.join(h['中药名'] for h in herbs_top)}。"
        ) if stats and ranked else f"当前查询疾病：「{selected_disease}」"
        response = ask_gemini_diet_assistant(
            st.session_state.diet_messages, disease_ctx, rag_context=rag_context,
        )
    st.session_state.diet_messages.append({"role": "assistant", "content": response})
    st.rerun()

# 显示完整对话记录
if st.session_state.diet_messages:
    st.markdown("---")
    st.markdown("### 💬 对话记录")
    for msg in st.session_state.diet_messages:
        if msg["role"] == "user":
            # 微信风格：右对齐，绿色气泡，头像在右侧
            safe_content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="wechat-user-row">'
                f'<div class="wechat-user-bubble">{safe_content}</div>'
                f'<div class="wechat-user-avatar">🧑</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant", avatar="🌿"):
                st.markdown(msg["content"])

# 古籍原文依据面板
if "diet_rag_docs" in st.session_state:
    rag_docs = st.session_state.diet_rag_docs
    if rag_docs:
        with st.expander("📜 查看 AI 引用的古籍原文依据", expanded=False):
            for i, doc in enumerate(rag_docs, 1):
                st.markdown(f"**📖 《{doc['book_name']}》**")
                st.info(doc["content"])
                if i < len(rag_docs):
                    st.divider()
    elif load_rag_db() is not None:
        # RAG 已加载但当前问题没检索到匹配古籍
        st.info("💡 本次问题未在古籍库中找到直接匹配的原文，AI 依据自身知识作答。")

# ==================== 页脚 ====================
st.markdown("---")
with st.expander("🔬 知识图谱检索路径（专业参考）", expanded=False):
    st.markdown(f"**疾病** → {en_name}")
    st.markdown(f"- 中药-化合物关系: {len(loader.herb_compound_df):,} 条")
    st.markdown(f"- 化合物-靶点关系: {len(loader.compound_target_df):,} 条")
    st.markdown(f"- 靶点-疾病关系: {len(loader.target_disease_df):,} 条")
    st.markdown(f"- 翻译词典: 400+ 医学词汇 | 高质量中文名: {len(loader.all_diseases_cn_quality)} 种")

st.markdown(
    '<div class="footer-note">🌱 基于网络药理学知识图谱 | 数据来源 TCMSP 等公共数据库 | '
    '中药→化合物→靶点→疾病 多层次关系链 | AI 驱动智能推荐</div>',
    unsafe_allow_html=True,
)

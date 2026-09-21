"""
药食同源饮品配方推荐系统 —— 知识图谱版 V2
基于 中药→化合物→靶点→疾病 网络药理学关系链
"""

import sys, os
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path: sys.path.insert(0, _project_root)

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import json, time, urllib.request, urllib.error, base64, sqlite3, zipfile, shutil
import threading
from difflib import SequenceMatcher
from src.data_loader import GraphDataLoader, CN_TO_EN_DISEASE
from src.disease_advice import get_disease_advice

# Bundled CJK font so matplotlib renders Chinese on servers without an apt fonts package
import matplotlib.font_manager as _fm
_BUNDLED_FONT_DIR = os.path.join(_project_root, "assets", "fonts")
if os.path.isdir(_BUNDLED_FONT_DIR):
    for _fn in sorted(os.listdir(_BUNDLED_FONT_DIR)):
        if _fn.lower().endswith((".ttf", ".otf", ".ttc")):
            try:
                _fm.fontManager.addfont(os.path.join(_BUNDLED_FONT_DIR, _fn))
            except Exception:
                pass
_CJK_FONTS = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
_available = {_f.name for _f in _fm.fontManager.ttflist}
for _name in _CJK_FONTS:
    if _name in _available:
        plt.rcParams["font.sans-serif"] = _CJK_FONTS[_CJK_FONTS.index(_name):]
        break
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

# ==================== Modern AI product visual layer ====================
# Keep the existing component class names because the rendering markup below
# relies on them; this layer replaces the legacy green dashboard appearance.
st.markdown("""<style>
:root {
  --primary: #7C3AED;
  --primary-2: #8B5CF6;
  --lavender: #C4B5FD;
  --lavender-bg: #EDE9FE;
  --ink: #111827;
  --body: #374151;
  --muted: #6B7280;
}

html, body, .stApp { background: #FAFAFC !important; color: var(--body); }
.stApp::before { display:none !important; }
[data-testid="stAppViewContainer"] { position: relative; overflow: hidden; }
[data-testid="stAppViewContainer"]::before {
  content: ""; position: fixed; inset: -220px 10% auto -15%; height: 520px;
  background: radial-gradient(circle, rgba(124,58,237,.16) 0%, rgba(196,181,253,.08) 38%, transparent 72%);
  pointer-events: none; z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
  content: ""; position: fixed; top: -120px; right: -120px; width: 520px; height: 520px;
  background: radial-gradient(circle, rgba(167,139,250,.14), transparent 70%);
  pointer-events: none; z-index: 0;
}
.main .block-container { max-width: 1240px; padding: 1.1rem 2rem 8rem; position: relative; z-index: 1; }
header[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
h1, h2, h3, h4, h5, h6 { color: var(--ink) !important; letter-spacing: -.02em; }
p, span, div, label { color: var(--body); }
hr, [data-testid="stDivider"] { border: 0 !important; height: 1px !important; background: linear-gradient(90deg, transparent, #DDD6FE, transparent) !important; margin: 1.7rem 0 !important; }

/* Product navigation and hero */
.product-nav { display:flex; align-items:center; gap:1rem; padding:.5rem 0 1.3rem; }
.brand-mark { color: var(--ink); font-size:1.18rem; font-weight:800; letter-spacing:-.03em; white-space:nowrap; }
.brand-mark span { color: var(--primary); }
.nav-links { display:flex; justify-content:center; gap:1.5rem; flex:1; }
.nav-links a { color:#6B7280 !important; text-decoration:none; font-size:.88rem; transition:color .2s ease; }
.nav-links a:hover { color:var(--primary) !important; }
.hero-banner { position:relative; overflow:hidden; background:rgba(255,255,255,.58); border:1px solid rgba(196,181,253,.35); border-radius:32px; padding:3.8rem 2rem 2.5rem; margin:.2rem auto 1.4rem; text-align:center; box-shadow:0 18px 60px rgba(76,29,149,.08); backdrop-filter:blur(18px); }
.hero-banner::before { content:""; position:absolute; width:340px; height:340px; top:-180px; left:8%; background:radial-gradient(circle,rgba(124,58,237,.16),transparent 70%); pointer-events:none; }
.hero-banner::after { content:""; position:absolute; width:360px; height:360px; right:-180px; bottom:-200px; background:radial-gradient(circle,rgba(196,181,253,.3),transparent 70%); pointer-events:none; }
.hero-kicker { position:relative; z-index:1; display:inline-flex; align-items:center; gap:6px; color:#6D28D9; background:#F5F3FF; border:1px solid #DDD6FE; padding:6px 14px; border-radius:999px; font-size:.78rem; font-weight:700; }
.hero-title { position:relative; z-index:1; color:var(--ink) !important; font-size:clamp(2.3rem,5vw,4rem); line-height:1.1; letter-spacing:-.06em; margin:1.1rem auto .8rem; }
.hero-title .gradient-word { background:linear-gradient(135deg,#6D28D9,#A78BFA); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.hero-subtitle { position:relative; z-index:1; color:var(--muted) !important; max-width:680px; font-size:1rem; line-height:1.8; margin:0 auto; }
.search-box { max-width:760px; margin:1.7rem auto .7rem; }
.search-box [data-testid="stTextInput"] input, .search-box input, [data-testid="stTextInput"] input { border-radius:9999px !important; border:1px solid #E5E7EB !important; background:#fff !important; color:var(--ink) !important; box-shadow:0 10px 30px rgba(17,24,39,.06) !important; outline:none !important; transition:all .25s ease !important; }
.search-box [data-testid="stTextInput"] input { padding:1rem 1.35rem !important; font-size:1.05rem !important; min-height:54px; }
.search-box [data-testid="stTextInput"] input:hover, .search-box input:hover, [data-testid="stTextInput"] input:hover { border-color:#C4B5FD !important; }
.search-box [data-testid="stTextInput"] input:focus, .search-box input:focus, [data-testid="stTextInput"] input:focus { border-color:var(--primary) !important; box-shadow:0 0 0 4px rgba(124,58,237,.15),0 10px 28px rgba(124,58,237,.16) !important; }
.hot-label { color:#9CA3AF; font-size:.8rem; margin:.3rem 0 .45rem; }

/* Controls, cards and metrics */
div[data-baseweb="select"] > div, [data-testid="stSlider"] { border-radius:999px !important; }
div[data-baseweb="select"] > div { border-color:#E5E7EB !important; background:#fff !important; box-shadow:0 5px 18px rgba(17,24,39,.04); }
div.stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:999px !important; border:1px solid #DDD6FE !important; background:#fff !important; color:#6D28D9 !important; font-weight:700 !important; box-shadow:0 5px 16px rgba(124,58,237,.06) !important; transition:all .22s ease !important; }
div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { border-color:#A78BFA !important; background:#F5F3FF !important; transform:translateY(-2px); }
div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { color:#fff !important; border:0 !important; background:linear-gradient(135deg,#7C3AED,#8B5CF6) !important; box-shadow:0 9px 22px rgba(124,58,237,.25) !important; }
div.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background:linear-gradient(135deg,#6D28D9,#7C3AED) !important; box-shadow:0 12px 28px rgba(124,58,237,.34) !important; }
div[data-testid="stMetric"], .herb-card, .ai-card, details[data-testid="stExpander"] { background:#fff !important; border-radius:20px !important; border:1px solid rgba(229,231,235,.85) !important; box-shadow:0 6px 24px rgba(17,24,39,.06) !important; transition:all .25s ease; }
div[data-testid="stMetric"] { padding:1.15rem 1.25rem; }
div[data-testid="stMetric"]:hover, .herb-card:hover, .ai-card:hover { transform:translateY(-3px); box-shadow:0 14px 34px rgba(124,58,237,.12) !important; }
div[data-testid="stMetric"] label { color:var(--muted) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:var(--ink) !important; font-weight:800; }
.herb-card { border-left:5px solid var(--primary) !important; padding:1.25rem 1.35rem; }
.herb-card .herb-name { color:#4C1D95; }
.herb-card .herb-stats, .herb-card .herb-stats span { color:var(--muted); }
.herb-card .herb-score { color:var(--primary); }
.progress-bar { background:#EDE9FE; height:5px; border-radius:999px; }
.progress-bar .fill { background:linear-gradient(90deg,#7C3AED,#A78BFA); border-radius:999px; }
.herb-img-wrap { border-radius:18px; box-shadow:0 5px 16px rgba(76,29,149,.12); }
.herb-img-placeholder { border-radius:18px; background:linear-gradient(135deg,#F5F3FF,#EDE9FE); border-color:#C4B5FD; }
.evidence-chip { color:#6D28D9; background:#F5F3FF; border-color:#DDD6FE; border-radius:999px; }
.ai-card { background:linear-gradient(135deg,#fff,#F5F3FF) !important; border-color:#EDE9FE !important; line-height:1.9; }
.ai-badge { background:linear-gradient(135deg,#7C3AED,#A78BFA); border-radius:999px; }
details[data-testid="stExpander"] { overflow:hidden; }
.search-hint { background:#F5F3FF; border-color:#DDD6FE; color:#6D28D9; border-radius:16px; }

/* Chat and status surfaces */
.chat-container, .diet-header-box, .chat-search-wrap { background:rgba(255,255,255,.68); border:1px solid rgba(196,181,253,.45); box-shadow:0 12px 36px rgba(76,29,149,.08); }
.wechat-user-bubble { background:linear-gradient(135deg,#7C3AED,#A78BFA); color:#fff; border-radius:20px 5px 20px 20px; box-shadow:0 5px 14px rgba(124,58,237,.18); }
.wechat-user-avatar { background:#F5F3FF; border:1px solid #DDD6FE; }
.footer-note { color:#9CA3AF; }
::-webkit-scrollbar-track { background:#FAFAFC; }
::-webkit-scrollbar-thumb { background:#C4B5FD; border-radius:999px; }

/* Functional portal buttons and the Streamlit-backed Spotlight form. */
div[data-testid="stButton"]:has(button[aria-label*="生活建议"]),
div[data-testid="stButton"]:has(button[aria-label*="药膳方案"]) { height:100%; }
div[data-testid="stButton"]:has(button[aria-label*="生活建议"]) button,
div[data-testid="stButton"]:has(button[aria-label*="药膳方案"]) button { min-height:150px !important; white-space:pre-wrap; text-align:left; padding:24px 26px !important; border-radius:24px !important; background:linear-gradient(135deg,#FFFFFF 0%,#F5F3FF 100%) !important; color:#4C1D95 !important; border:1px solid #EDE9FE !important; box-shadow:0 10px 28px rgba(76,29,149,.08) !important; font-size:1.05rem !important; line-height:1.7 !important; }
div[data-testid="stButton"]:has(button[aria-label*="生活建议"]) button:hover,
div[data-testid="stButton"]:has(button[aria-label*="药膳方案"]) button:hover { transform:translateY(-8px) scale(1.02); border-color:#C4B5FD !important; box-shadow:0 20px 42px rgba(124,58,237,.22) !important; background:linear-gradient(135deg,#FFFFFF 0%,#EDE9FE 100%) !important; }
[data-testid="stForm"]:has(input[aria-label*="食养灵感"]) { position:fixed !important; left:50% !important; bottom:40px !important; transform:translateX(-50%) !important; z-index:9999 !important; width:60% !important; min-width:280px !important; max-width:760px !important; height:62px !important; min-height:62px !important; max-height:62px !important; margin:0 !important; padding:8px 9px 8px 20px !important; overflow:hidden !important; border-radius:99px !important; background:rgba(255,255,255,.6) !important; backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,.8) !important; box-shadow:0 16px 40px rgba(124,58,237,.2), inset 0 1px 0 rgba(255,255,255,.9) !important; }
[data-testid="stForm"]:has(input[aria-label*="食养灵感"]) [data-testid="stTextInput"] input { border:0 !important; outline:0 !important; background:transparent !important; box-shadow:none !important; border-radius:99px !important; padding:11px 0 !important; }
[data-testid="stForm"]:has(input[aria-label*="食养灵感"]) [data-testid="stFormSubmitButton"] button { width:42px !important; height:42px !important; min-height:42px !important; padding:0 !important; border:0 !important; border-radius:50% !important; color:#fff !important; background:linear-gradient(135deg,#7C3AED,#8B5CF6) !important; box-shadow:0 8px 18px rgba(124,58,237,.3) !important; }
[data-testid="stForm"]:has(input[aria-label*="食养灵感"]) [data-testid="stFormSubmitButton"] button:hover { transform:scale(1.08) rotate(-5deg); }
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


_FANGJI_DB_DIR = os.path.join(_project_root, "data", "chroma_db_fangji")


@st.cache_resource(show_spinner=False)
def _load_fangji_chroma():
    """加载方剂书 ChromaDB（data/chroma_db_fangji，随仓库部署，无需下载）"""
    if not os.path.isdir(_FANGJI_DB_DIR) or not os.path.isfile(
        os.path.join(_FANGJI_DB_DIR, "chroma.sqlite3")
    ):
        return None
    try:
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=_FANGJI_DB_DIR,
            embedding_function=_get_rag_embeddings(),
            collection_name="fangji_books",
        )
    except Exception:
        return None


_YANGSHENG_DB_DIR = os.path.join(_project_root, "data", "chroma_db_yangsheng")


@st.cache_resource(show_spinner=False)
def _load_yangsheng_chroma():
    """加载养生书 ChromaDB（data/chroma_db_yangsheng，随仓库部署，无需下载）"""
    if not os.path.isdir(_YANGSHENG_DB_DIR) or not os.path.isfile(
        os.path.join(_YANGSHENG_DB_DIR, "chroma.sqlite3")
    ):
        return None
    try:
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=_YANGSHENG_DB_DIR,
            embedding_function=_get_rag_embeddings(),
            collection_name="yangsheng_books",
        )
    except Exception:
        return None


def retrieve_ancient_books(query: str, k: int = 3):
    """同时检索古籍库 + 方剂库 + 养生库，合并返回"""
    global _RAG_ERROR_MSG
    results = []

    # 古籍库
    db = load_rag_db()
    if db is not None:
        try:
            docs = db.similarity_search(query, k=k)
            results.extend(
                {
        "content": doc.page_content,
        "book_name": doc.metadata.get("book_name", "佚名"),
        "source": doc.metadata.get("source", "ancient_book"),
        "question_id": doc.metadata.get("question_id"),
                }
        except Exception as e:
            _RAG_ERROR_MSG = f"❌ 古籍检索失败: {type(e).__name__} - {str(e)[:200]}"

    # 方剂库
    fdb = _load_fangji_chroma()
    if fdb is not None:
        try:
            docs = fdb.similarity_search(query, k=k)
            results.extend(
    {
        "content": doc.page_content,
        "book_name": doc.metadata.get("book_name", "佚名"),
        "source": "fangji",
    }
                for doc in docs
)
        except Exception as e:
            _RAG_ERROR_MSG = f"❌ 方剂检索失败: {type(e).__name__} - {str(e)[:200]}"

    # 养生库
    ydb = _load_yangsheng_chroma()
    if ydb is not None:
        try:
            docs = ydb.similarity_search(query, k=k)
            results.extend(
    {
        "content": doc.page_content,
        "book_name": doc.metadata.get("book_name", "佚名"),
        "source": "yangsheng",
    }
                for doc in docs
)
        except Exception as e:
            _RAG_ERROR_MSG = f"❌ 养生库检索失败: {type(e).__name__} - {str(e)[:200]}"

    return results


def prewarm_rag_resources():
    """在图谱结果页提前初始化 RAG 资源，避免首次提交问题时才冷启动。"""
    # 先显式初始化 embedding 模型，再初始化三个 Chroma 实例；这些函数本身也有资源缓存。
    _get_rag_embeddings()
    ancient_db = load_rag_db()
    fangji_db = _load_fangji_chroma()
    yangsheng_db = _load_yangsheng_chroma()
    return {
        "ancient": ancient_db is not None,
        "fangji": fangji_db is not None,
        "yangsheng": yangsheng_db is not None,
    }


@st.cache_resource(show_spinner=False)
def start_rag_prewarm():
    """启动一次后台预热任务；同一 Streamlit 进程内的多个会话共享结果。"""
    state = {"done": False, "error": None, "thread": None}

    def _worker():
        try:
            prewarm_rag_resources()
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            state["done"] = True

    state["thread"] = threading.Thread(
        target=_worker,
        name="rag-prewarm",
        daemon=True,
    )
    state["thread"].start()
    return state

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
                  "itemStyle": {"color": "#7C3AED"}})   # 紫色 — 疾病
    nodes.append({"name": herb_name, "symbolSize": 25,
                  "itemStyle": {"color": "#8B5CF6"}})   # 紫色 — 中药

    # --- 化合物节点（浅紫）---
    for cid, cname in compounds:
        label = cname if cname and cname != cid else cid
        # 截断过长名称
        if len(label) > 18:
            label = label[:16] + "..."
        nodes.append({"name": cid, "symbolSize": 15,
                      "itemStyle": {"color": "#A78BFA"},
                      "label": {"formatter": label}})
        # 中药 → 化合物
        links.append({"source": herb_name, "target": cid})

    # --- 靶点节点（淡紫）---
    for tid, tname in targets:
        label = tname if tname and tname != tid else tid
        if len(label) > 14:
            label = label[:12] + "..."
        nodes.append({"name": tid, "symbolSize": 12,
                      "itemStyle": {"color": "#C4B5FD"},
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
            bg_color="transparent",
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
                    color="#4C1D95",
                ),
                pos_left="center",
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
                background_color="#FFFFFF", border_color="#EDE9FE", border_width=1,
                textstyle_opts=opts.TextStyleOpts(color="#111827"),
            ),
        )
        .set_colors(["#7C3AED", "#8B5CF6", "#A78BFA", "#C4B5FD", "#EDE9FE"])
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
            "【检索到的古籍记载】：\n" + rag_context +
            "\n\n其中标记为【问答知识库参考资料】的内容为检索到的网络问答参考资料（非古籍原典），"
            "仅供审慎参考；如与古籍记载冲突，请以古籍原典与专业知识为准。"
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

# ==================== 顶部导航与主搜索 ====================
if "hero_search" not in st.session_state:
    st.session_state.hero_search = ""
if "pending_disease_search" in st.session_state:
    st.session_state.hero_search = st.session_state.pop("pending_disease_search")

nav_brand, nav_links, nav_action = st.columns([2.2, 5.8, 1.5], vertical_alignment="center")
with nav_brand:
    st.markdown('<div class="brand-mark">🌿 TCM<span>KIAgent</span></div>', unsafe_allow_html=True)
with nav_links:
    st.markdown(
        '<div class="nav-links">'
        '<a href="#knowledge-graph">知识图谱检索</a>'
        '<a href="#literature-evidence">文献知识</a>'
        '<a href="#ancient-books">古籍问答</a>'
        '<a href="#health-advice">健康建议</a>'
        '</div>', unsafe_allow_html=True,
    )
with nav_action:
    if st.button("刷新缓存", key="refresh_cache_top", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-kicker">✦ 网络药理学知识图谱 · 药食同源智能推荐</div>'
    '<div class="hero-title">用 AI 读懂 <span class="gradient-word">中药与疾病</span> 的关联</div>'
    '<div class="hero-subtitle">从疾病、靶点、化合物到中药，沿着可解释的关系链探索日常健康科普线索。</div>'
    '</div>', unsafe_allow_html=True,
)

search_query = st.text_input(
    "搜索疾病", placeholder="输入中文或英文疾病名，例如：高血压、糖尿病...",
    label_visibility="collapsed", key="hero_search",
)

st.markdown('<div class="hot-label">🔥 热门搜索</div>', unsafe_allow_html=True)
hot_diseases = ["高血压", "糖尿病", "高血脂", "失眠", "痛风", "肥胖症", "贫血", "慢性胃炎"]

def _set_hot_search(disease_name):
    """在下一次脚本重跑前更新 Hero 搜索框 widget 的值。"""
    st.session_state.pending_disease_search = disease_name

hot_cols = st.columns(len(hot_diseases))
for hot_col, hot_disease in zip(hot_cols, hot_diseases):
    with hot_col:
        st.button(
            hot_disease, key=f"hot_search_{hot_disease}", use_container_width=True,
            on_click=_set_hot_search, args=(hot_disease,),
        )

if search_query.strip():
    matched = fuzzy_search(search_query.strip(), all_diseases, top_k=15)
    default_idx = 0
else:
    matched = all_diseases_default
    default_idx = matched.index("高血压") if "高血压" in matched else 0

control_cols = st.columns([4.4, 2.2, 2.1, 1.8], gap="small", vertical_alignment="bottom")
with control_cols[0]:
    if matched:
        selected_disease = st.selectbox(
            "匹配结果（{}条）".format(len(matched)), options=matched,
            index=default_idx, label_visibility="collapsed",
        )
    else:
        selected_disease = None
        st.markdown('<div class="search-hint">🔎 未找到匹配，试试其他关键词</div>', unsafe_allow_html=True)
with control_cols[1]:
    top_k = st.slider("展示 Top N", 5, 30, 15, key="top_k_control")
with control_cols[2]:
    formula_size = st.select_slider(
        "组合候选味数", options=[3, 4, 5], value=3, key="formula_size_control"
    )
with control_cols[3]:
    generate_btn = st.button(
        "查询知识图谱", type="primary", use_container_width=True,
        disabled=(selected_disease is None), key="query_graph_top",
    )

# 将查询参数持久化到 session_state，防止 AI 问答重跑时丢失
if generate_btn and selected_disease:
    if st.session_state.get("query_disease") != selected_disease:
        for transient_key in (
            "portal_lifestyle_result",
            "spotlight_response",
            "spotlight_query",
            "spotlight_prefill",
        ):
            st.session_state.pop(transient_key, None)
        st.session_state.diet_messages = []
        st.session_state.diet_rag_docs = []
    st.session_state.query_disease = selected_disease
    st.session_state.query_top_k = top_k
    st.session_state.query_formula_size = int(formula_size)
    st.session_state.query_active = True

# 初始化查询持久化状态
if "query_active" not in st.session_state:
    st.session_state.query_active = False

if not st.session_state.query_active:
    st.markdown('<div id="knowledge-graph"></div>', unsafe_allow_html=True)
    st.markdown("### 从一个问题开始探索")
    st.caption("选择疾病后，系统会沿着真实数据中的中药、化合物和靶点关系生成可解释的检索结果。")
    empty_cols = st.columns(3, gap="large")
    empty_cards = [
        ("🧬", "知识图谱检索", "查看疾病关联靶点、化合物和药食同源中药的证据链。"),
        ("📚", "文献证据", "在中药卡片中查看 PubMed 与知网文献摘要和原文入口。"),
        ("📜", "古籍 RAG 问答", "结合古籍、方剂与养生知识库，获得日常膳食科普建议。"),
    ]
    for card_col, (icon, title, desc) in zip(empty_cols, empty_cards):
        with card_col:
            st.markdown(
                f'<div class="ai-card" style="min-height:145px">'
                f'<div style="font-size:1.6rem;margin-bottom:.45rem">{icon}</div>'
                f'<strong style="color:#4C1D95;font-size:1.05rem">{title}</strong>'
                f'<div style="color:#6B7280;font-size:.9rem;line-height:1.7;margin-top:.45rem">{desc}</div>'
                '</div>', unsafe_allow_html=True,
            )
    st.info("选择一个疾病并点击「查询知识图谱」，开始查看结果。")
    st.stop()

# 从 session_state 取持久化的查询参数（chat_input 重跑时不会丢失）
selected_disease = st.session_state.query_disease
top_k = st.session_state.query_top_k
formula_size = st.session_state.get("query_formula_size", 3)

# ==================== 查询 ====================
with st.spinner("🌿 知识图谱检索中，深度分析疾病-靶点-化合物-中药关系链…"):
    stats = loader.get_graph_stats(selected_disease)
    ranked = loader.rank_herbs_for_disease(selected_disease, top_k=top_k)

if not ranked:
    st.warning(f"知识图谱中未找到与「{selected_disease}」直接关联的中药数据。")
    st.stop()

en_name = loader.cn_to_en.get(selected_disease) or CN_TO_EN_DISEASE.get(selected_disease, selected_disease)
max_targets = ranked[0]["关联靶点数"]

# ==================== KPI 卡片 ====================
st.markdown('<div id="knowledge-graph"></div>', unsafe_allow_html=True)
st.markdown("### 📊 图谱检索概览")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔬 关联靶点", f"{stats['关联靶点数']} 个")
c2.metric("🧪 关联化合物", f"{stats['关联化合物数']} 个")
c3.metric("🌱 相关中药", f"{stats['相关中药数']} 种")
c4.metric("📋 疾病英文名", en_name[:25])

# ==================== 多味组合候选 ====================
# 组合候选使用至少 Top15 作为搜索池；页面上的 Top N 仍然只控制单味排名展示。
formula_recommendations = loader.recommend_herb_combinations(
    selected_disease,
    candidate_k=max(15, top_k),
    formula_size=formula_size,
    max_alternatives=3,
)

if formula_recommendations:
    primary_formula = formula_recommendations[0]
    st.markdown("### 🧩 疾病相关组合候选")
    st.caption(
        "基于疾病相关靶点的边际覆盖生成；组合中的‘互补’仅表示网络证据互补，"
        "不包含剂量、禁忌、君臣佐使或临床疗效判断。"
    )

    formula_cols = st.columns(4)
    formula_cols[0].metric("候选味数", f"{primary_formula['组合规模']} 味")
    formula_cols[1].metric("覆盖靶点", f"{primary_formula['覆盖靶点数']} 个")
    formula_cols[2].metric("靶点覆盖率", f"{primary_formula['靶点覆盖率']:.1%}")
    formula_cols[3].metric("关联化合物", f"{primary_formula['组合关联化合物数']} 个")

    st.markdown(
        "**主组合候选：** "
        + " ＋ ".join(primary_formula["中药名列表"])
    )
    with st.expander("查看主组合的新增靶点贡献", expanded=True):
        for member in primary_formula["组合成员"]:
            new_targets = "、".join(
                target_name for _, target_name in member["新增靶点"][:8]
            ) or "无新增疾病靶点"
            if len(member["新增靶点"]) > 8:
                new_targets += " 等"
            st.markdown(
                f"**{member['中药名']}**（原始排名 #{member['原始排名']}）  "
                f"新增靶点 **{member['新增靶点数']}** 个、"
                f"新增化合物 **{member['新增化合物数']}** 个  \n"
                f"新增靶点：{new_targets}"
            )
        st.caption(
            f"候选池为当前单味排序前 {primary_formula['候选池大小']} 味；"
            f"{primary_formula['停止原因']}。"
        )

    for alt_index, formula in enumerate(formula_recommendations[1:], start=1):
        with st.expander(
            f"备选组合 {alt_index}：{' ＋ '.join(formula['中药名列表'])}"
        ):
            st.markdown(
                f"覆盖靶点 **{formula['覆盖靶点数']} / {formula['疾病靶点数']}**，"
                f"覆盖率 **{formula['靶点覆盖率']:.1%}**，"
                f"关联化合物 **{formula['组合关联化合物数']}** 个。"
            )
            st.caption(formula["停止原因"])

# ==================== 中药排名 + 图表 ====================
st.markdown(f"### 🏆 「{selected_disease}」关联中药 Top {min(top_k, len(ranked))}")
st.caption(f"按靶点覆盖度排序，共 {stats['相关中药数']} 种药食同源中药与该疾病在分子层面存在关联")

# 在中药卡片渲染前确保文献库已下载，避免卡片查询缓存到空结果
_ensure_literature_db()

# 进入结果页后在后台预热向量模型与三个知识库，避免用户首次提交问题时才冷启动。
rag_prewarm_state = start_rag_prewarm()

left_col, right_col = st.columns([5, 4], gap="large")

with left_col:
    for i, herb in enumerate(ranked):
        # 奖牌颜色
        if i == 0:
            medal_c, rank_bg, border_c = "🥇", "linear-gradient(135deg, #FFFFFF, #F5F3FF)", "#7C3AED"
        elif i == 1:
            medal_c, rank_bg, border_c = "🥈", "linear-gradient(135deg, #FFFFFF, #FAFAFF)", "#A78BFA"
        elif i == 2:
            medal_c, rank_bg, border_c = "🥉", "linear-gradient(135deg, #FFFFFF, #F8F7FF)", "#C4B5FD"
        else:
            medal_c, rank_bg, border_c = f"<span style='color:#6B7280;font-size:1.1rem'>{i+1}</span>", "#FFFFFF", "#EDE9FE"

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
        st.markdown('<div id="literature-evidence"></div>', unsafe_allow_html=True)
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
                        src_badge = ('<span style="font-size:0.75rem;color:#5B21B6;background:#EDE9FE;'
                                     'padding:2px 8px;border-radius:10px;margin-left:6px;">知网</span>')
                    elif src == "PubMed":
                        src_badge = ('<span style="font-size:0.75rem;color:#6D28D9;background:#F5F3FF;'
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
                            f'<div style="font-size:0.88rem;color:#6B7280;line-height:1.7;'
                            f'white-space:pre-wrap;border-left:3px solid #C4B5FD;'
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
    # 将服务端 Matplotlib 图表转成 Base64，交给自定义前端 Tabs 切换。
    top8 = ranked[:8]
    labels = [h["中药名"] for h in top8]
    values = [h["关联靶点数"] for h in top8]
    palette = ["#7C3AED", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE", "#EDE9FE", "#F5F3FF", "#E9D5FF"]

    fig1, ax1 = plt.subplots(figsize=(4.2, 4.2))
    fig1.patch.set_facecolor("none"); ax1.set_facecolor("none")
    wedges, texts, autotexts = ax1.pie(
        values, labels=None, autopct="%1.1f%%", colors=palette[:len(labels)],
        startangle=140, pctdistance=0.78,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2, "antialiased": True},
    )
    for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("#6B7280")
    ax1.legend(wedges, labels, title="中药名", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, frameon=False)
    ax1.set_title("靶点覆盖度分布", fontsize=12, fontweight="bold", color="#4C1D95", pad=12)
    ax1.tick_params(colors="#6B7280")
    donut_buffer = BytesIO()
    fig1.savefig(donut_buffer, format="png", dpi=150, transparent=True, bbox_inches="tight")
    donut_b64 = base64.b64encode(donut_buffer.getvalue()).decode("ascii")
    plt.close(fig1)

    # 柱状图：Top 10
    top10 = ranked[:10]
    names = [h["中药名"] for h in reversed(top10)]
    tv = [h["关联靶点数"] for h in reversed(top10)]
    cv = [h["关联化合物数"] for h in reversed(top10)]

    fig2, ax2 = plt.subplots(figsize=(4.5, 4.2))
    fig2.patch.set_facecolor("none"); ax2.set_facecolor("none")
    y = range(len(names))
    ax2.barh([yi + 0.2 for yi in y], tv, 0.38, color="#7C3AED", alpha=0.9, label="靶点数", edgecolor="white", linewidth=0.5)
    ax2.barh([yi - 0.2 for yi in y], cv, 0.38, color="#C4B5FD", alpha=0.95, label="化合物数", edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=9)
    ax2.tick_params(axis="both", colors="#6B7280")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.8, facecolor="white", edgecolor="#EDE9FE")
    ax2.set_xlabel("数量", fontsize=9, color="#6B7280")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.9, color="#E5E7EB", linestyle="--")
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    bar_buffer = BytesIO()
    fig2.savefig(bar_buffer, format="png", dpi=150, transparent=True, bbox_inches="tight")
    bar_b64 = base64.b64encode(bar_buffer.getvalue()).decode("ascii")
    plt.close(fig2)

    chart_tabs_html = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin:0; padding:0; background:transparent; font-family:Inter,Arial,'Microsoft YaHei',sans-serif; }}
      .chart-shell {{ background:rgba(255,255,255,.72); border:1px solid #EDE9FE; border-radius:24px; padding:16px; box-shadow:0 12px 30px rgba(76,29,149,.08); }}
      .segment {{ display:flex; gap:4px; padding:4px; background:#F5F3FF; border:1px solid #EDE9FE; border-radius:999px; }}
      .segment button {{ flex:1; border:0; border-radius:999px; padding:10px 12px; background:transparent; color:#6B7280; cursor:pointer; font-size:13px; font-weight:700; transition:all .3s ease; }}
      .segment button:hover {{ color:#6D28D9; }}
      .segment button.active {{ background:#7C3AED; color:#fff; box-shadow:0 6px 16px rgba(124,58,237,.22); }}
      .chart-panel {{ display:none; min-height:430px; padding:10px 2px 0; align-items:center; justify-content:center; animation:fadeIn .3s ease; }}
      .chart-panel.active {{ display:flex; }}
      .chart-panel img {{ display:block; width:100%; height:420px; object-fit:contain; }}
      .chart-caption {{ text-align:center; color:#6B7280; font-size:12px; margin-top:2px; }}
      @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(4px); }} to {{ opacity:1; transform:translateY(0); }} }}
    </style></head><body>
      <div class="chart-shell">
        <div class="segment" role="tablist" aria-label="图表切换">
          <button class="active" data-tab="donut" role="tab">🍩 靶点覆盖分布</button>
          <button data-tab="bar" role="tab">📊 关联强度排行</button>
        </div>
        <div class="chart-panel active" id="panel-donut" role="tabpanel">
          <img src="data:image/png;base64,{donut_b64}" alt="靶点覆盖分布环形图">
        </div>
        <div class="chart-panel" id="panel-bar" role="tabpanel">
          <img src="data:image/png;base64,{bar_b64}" alt="Top 10 关联强度条形图">
        </div>
        <div class="chart-caption">基于当前疾病的知识图谱关联计数，仅作科普参考</div>
      </div>
      <script>
        (() => {{
          const buttons = document.querySelectorAll('[data-tab]');
          const panels = document.querySelectorAll('.chart-panel');
          buttons.forEach((button) => button.addEventListener('click', () => {{
            const tab = button.dataset.tab;
            buttons.forEach((item) => item.classList.toggle('active', item === button));
            panels.forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${{tab}}`));
          }}));
        }})();
      </script>
    </body></html>
    """
    components.html(chart_tabs_html, height=520, scrolling=False)

# ==================== 建议区：Cute Portals ====================
st.markdown("---")
st.markdown('<div id="health-advice"></div><div id="ancient-books"></div>', unsafe_allow_html=True)

# 保留既有会话键，避免刷新页面时丢失对话上下文；RAG 只在用户主动提交问题时触发。
if "diet_messages" not in st.session_state:
    st.session_state.diet_messages = []
if "diet_rag_docs" not in st.session_state:
    st.session_state.diet_rag_docs = []

# ==================== 项目功能入口：生活建议 / 药膳问答 ====================

graph_context = (
    f"当前疾病为「{selected_disease}」（英文名：{en_name}）。"
    f"知识图谱统计：关联靶点 {stats['关联靶点数']} 个、"
    f"关联化合物 {stats['关联化合物数']} 个、相关中药 {stats['相关中药数']} 种。"
    f"当前推荐中药包括：{'、'.join(item['中药名'] for item in ranked[:5])}。"
)
if formula_recommendations:
    graph_context += (
        "网络覆盖候选组合（仅为知识图谱推理，不是临床处方）："
        + "；".join("、".join(item["中药名列表"]) for item in formula_recommendations[:3])
        + "。"
    )

portal_lifestyle_clicked = False
portal_diet_clicked = False
portal_left, portal_right = st.columns(2, gap="large")
with portal_left:
    portal_lifestyle_clicked = st.button(
        "🧘‍♀️  开启 AI 生活建议\n\n结合当前疾病与图谱结果，查看日常作息、饮食和注意事项",
        key="portal_lifestyle",
        use_container_width=True,
    )
with portal_right:
    portal_diet_clicked = st.button(
        "🍵  获取专属药膳方案\n\n结合古籍、方剂与养生知识库，向食养助手提问",
        key="portal_diet",
        use_container_width=True,
    )

if portal_lifestyle_clicked:
    with st.spinner("正在结合疾病知识库与图谱结果整理建议…"):
        advice = get_disease_advice(selected_disease, graph_context)
    st.session_state.portal_lifestyle_result = advice or {
        "来源": "暂不可用",
        "AI建议": "当前未获取到生活建议，请稍后重试。",
    }

if portal_diet_clicked:
    st.session_state.spotlight_prefill = (
        f"请结合「{selected_disease}」和当前知识图谱推荐，给我一份日常药膳方案"
    )
    st.session_state.portal_action = "diet"

if st.session_state.get("portal_lifestyle_result"):
    advice = st.session_state.portal_lifestyle_result
    with st.expander(f"✨ {selected_disease} 的 AI 生活建议 · {advice.get('来源', '项目知识库')}", expanded=True):
        if advice.get("AI建议"):
            st.markdown(advice["AI建议"])
        else:
            st.markdown(f"**{advice.get('概述', '结合当前查询结果，为你整理以下日常科普建议。')}**")
            advice_col, diet_col = st.columns(2, gap="large")
            with advice_col:
                st.markdown("**生活节奏**")
                for item in advice.get("生活建议", [])[:3]:
                    st.markdown(f"- {item}")
            with diet_col:
                st.markdown("**饮食参考**")
                for item in advice.get("推荐饮食", [])[:3]:
                    st.markdown(f"- {item}")
            precautions = advice.get("注意事项", [])[:2]
            if precautions:
                st.markdown("**需要留意**：" + "；".join(precautions))
        st.caption("本内容仅为日常膳食与生活科普，不作为临床医疗诊断依据；如有不适请及时就医。")

# ==================== Spotlight 对话入口（接入现有三库 RAG + DeepSeek） ====================

if "spotlight_query" not in st.session_state:
    st.session_state.spotlight_query = ""
if "spotlight_prefill" in st.session_state:
    st.session_state.spotlight_query = st.session_state.pop("spotlight_prefill")

with st.form("spotlight_form", clear_on_submit=True, border=False):
    spotlight_input_col, spotlight_send_col = st.columns([12, 1], vertical_alignment="center")
    with spotlight_input_col:
        spotlight_query = st.text_input(
            "食养灵感",
            placeholder="输入一个问题，探索你的食养灵感…",
            label_visibility="collapsed",
            key="spotlight_query",
        )
    with spotlight_send_col:
        spotlight_submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

if spotlight_submitted and spotlight_query.strip():
    user_query = spotlight_query.strip()
    st.session_state.diet_messages.append({"role": "user", "content": user_query})
    rag_thread = rag_prewarm_state.get("thread")
    if rag_thread is not None and rag_thread.is_alive():
        with st.spinner("📚 正在完成食养知识库首次准备…"):
            rag_thread.join()
    with st.spinner("正在检索古籍、方剂与养生知识库…"):
        rag_docs = retrieve_ancient_books(user_query, k=3)
        st.session_state.diet_rag_docs = rag_docs
        def format_rag_doc(doc):
            if doc.get("source") == "qa":
                return f"【问答知识库参考资料】\n{doc.get('content', '')}"
            return f"【{doc.get('book_name', '知识库')}】{doc.get('content', '')}"


        rag_context = "\n\n".join(
            format_rag_doc(doc)
            for doc in rag_docs
)
        response = ask_gemini_diet_assistant(
            st.session_state.diet_messages,
            disease_context=graph_context,
            rag_context=rag_context,
        )
    st.session_state.diet_messages.append({"role": "assistant", "content": response})
    st.session_state.spotlight_response = response

if st.session_state.get("spotlight_response"):
    with st.expander("✨ Spotlight 最近回复", expanded=True):
        st.markdown(st.session_state.spotlight_response)
        if st.session_state.diet_rag_docs:
            st.markdown(f"**📚 本次参考了 {len(st.session_state.diet_rag_docs)} 条知识库片段**")
            for doc in st.session_state.diet_rag_docs:
                if doc.get("source") == "qa":
                    st.caption(f"问答知识库：{doc.get('content', '')[:280]}…")
                else:
                    st.caption(f"{doc.get('book_name', '知识库')}：{doc.get('content', '')[:280]}…")

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

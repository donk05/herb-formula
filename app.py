"""
药食同源饮品配方推荐系统 —— 知识图谱版
基于 中药→化合物→靶点→疾病 网络药理学关系链
"""

import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher

from src.data_loader import GraphDataLoader, CN_TO_EN_DISEASE
from src.disease_advice import get_disease_advice

# ==================== matplotlib 中文配置 ====================
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 页面配置 ====================
st.set_page_config(page_title="药食同源饮品配方推荐系统", page_icon="🌿", layout="wide")

# ==================== 自定义 CSS ====================
st.markdown("""<style>
body, .stApp { background-color: #FDFBF7; background-image: url("data:image/svg+xml,%3Csvg width='120' height='120' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M20 60 Q40 30 60 60 Q40 90 20 60' fill='none' stroke='%23D4CFC4' stroke-width='0.6' opacity='0.3'/%3E%3Ccircle cx='90' cy='30' r='1.5' fill='%23C8C0B0' opacity='0.25'/%3E%3Ccircle cx='100' cy='90' r='1.2' fill='%23C8C0B0' opacity='0.2'/%3E%3Ccircle cx='30' cy='100' r='1.8' fill='%23C8C0B0' opacity='0.22'/%3E%3C/svg%3E"); background-repeat: repeat; background-size: 120px 120px; }
.main .block-container { padding-top: 1.5rem; background: transparent; }
.stApp::before { content:""; position:fixed; top:0; left:0; right:0; height:4px; background:linear-gradient(90deg,#A5D6A7 0%,#2E7D32 30%,#43A047 50%,#2E7D32 70%,#A5D6A7 100%); z-index:9999; pointer-events:none; }
h1,h2,h3,h4 { color:#2E7D32 !important; }
p,span,div { color:#333333; }
hr,[data-testid="stDivider"] { border:none!important; height:2px!important; background:linear-gradient(90deg,transparent 0%,#C8D6B8 20%,#A5B896 50%,#C8D6B8 80%,transparent 100%)!important; margin:1.5rem 0!important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#F8F3E8 0%,#F0EAD6 50%,#F5F1E8 100%); border-right:1px solid #E0D8C8; }
section[data-testid="stSidebar"]::before { content:""; position:absolute; top:0; left:0; right:0; bottom:0; background-image:url("data:image/svg+xml,%3Csvg width='80' height='80' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M15 45 Q30 20 45 45 Q30 70 15 45' fill='none' stroke='%23D8CFB8' stroke-width='0.5' opacity='0.35'/%3E%3C/svg%3E"); background-repeat:repeat; background-size:80px 80px; pointer-events:none; z-index:0; }
section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stSlider label { color:#333333!important; font-weight:600; }
div.stButton > button { background:linear-gradient(135deg,#2E7D32 0%,#388E3C 40%,#43A047 100%); color:#FFFFFF!important; border:none; border-radius:12px; padding:0.6rem 1.5rem; font-size:1.05rem; font-weight:700; box-shadow:0 4px 14px rgba(46,125,50,0.35),0 1px 3px rgba(46,125,50,0.2),inset 0 1px 0 rgba(255,255,255,0.15); transition:all 0.25s ease; letter-spacing:0.5px; }
div.stButton > button:hover { background:linear-gradient(135deg,#1B5E20 0%,#2E7D32 40%,#388E3C 100%); box-shadow:0 6px 22px rgba(46,125,50,0.48),0 2px 5px rgba(46,125,50,0.25),inset 0 1px 0 rgba(255,255,255,0.12); transform:translateY(-2px); }
div.stButton > button:disabled { background:linear-gradient(135deg,#C8C8C8,#D4D4D4); box-shadow:0 2px 6px rgba(0,0,0,0.06); color:#999999!important; }
div[data-testid="stMetric"] { background:linear-gradient(135deg,#FFFFFF 0%,#FAF8F3 100%); border-radius:14px; padding:1rem 1.2rem; box-shadow:0 2px 12px rgba(0,0,0,0.05),0 1px 3px rgba(0,0,0,0.04); border-left:4px solid #2E7D32; transition:box-shadow 0.25s,transform 0.25s; }
div[data-testid="stMetric"]:hover { box-shadow:0 6px 20px rgba(0,0,0,0.09); transform:translateY(-2px); }
div[data-testid="stMetric"] label { color:#777777!important; font-size:0.85rem; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:#2E7D32!important; font-weight:800; }
.search-hint { background:linear-gradient(135deg,#FFF8E1,#FFF3CD); border-radius:10px; padding:0.8rem 1rem; border:1px solid #FFE082; font-size:0.9rem; color:#795548; box-shadow:0 1px 4px rgba(0,0,0,0.04); }
.hero-banner { background:linear-gradient(160deg,#F0F7EE 0%,#FDFBF7 30%,#F7F3E8 70%,#EEF5EA 100%); border-radius:20px; padding:2.2rem 2rem 1.8rem 2rem; margin-bottom:1.5rem; text-align:center; box-shadow:0 2px 20px rgba(46,125,50,0.06),inset 0 0 60px rgba(165,214,167,0.15); border:1px solid rgba(165,214,167,0.25); position:relative; overflow:hidden; }
.hero-banner::before { content:""; position:absolute; top:-30px; left:-30px; width:100px; height:100px; background:radial-gradient(circle,rgba(46,125,50,0.06) 0%,transparent 70%); border-radius:50%; pointer-events:none; }
.hero-banner::after { content:""; position:absolute; bottom:-20px; right:-20px; width:120px; height:120px; background:radial-gradient(circle,rgba(165,214,167,0.08) 0%,transparent 70%); border-radius:50%; pointer-events:none; }
.hero-title { font-size:2.5rem; font-weight:800; color:#2E7D32; text-align:center; margin-bottom:0.4rem; position:relative; z-index:1; text-shadow:0 1px 2px rgba(0,0,0,0.04); }
.hero-subtitle { font-size:1.05rem; color:#888888; text-align:center; margin-bottom:0; line-height:1.7; position:relative; z-index:1; }
.herb-card { background:linear-gradient(135deg,#FFFFFF 0%,#FAF8F2 100%); border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:0.7rem; box-shadow:0 3px 16px rgba(0,0,0,0.05),0 1px 4px rgba(0,0,0,0.04); border-left:5px solid #2E7D32; transition:all 0.25s ease; }
.herb-card:hover { box-shadow:0 8px 28px rgba(0,0,0,0.10),0 2px 6px rgba(0,0,0,0.06); transform:translateY(-2px); }
.herb-card .rank { font-size:1.8rem; font-weight:800; color:#2E7D32; }
.herb-card .herb-name { font-size:1.3rem; font-weight:700; color:#2E7D32; }
.herb-card .stats { font-size:0.9rem; color:#666; margin-top:0.3rem; }
.herb-card .evidence { font-size:0.82rem; color:#999; margin-top:0.4rem; line-height:1.5; }
.footer-note { text-align:center; color:#B0B0B0; padding:0.5rem 0; }
details[data-testid="stExpander"] { border-radius:12px!important; border:1px solid #E5DFD0!important; background:linear-gradient(180deg,#FDFCFA,#F9F6EF)!important; box-shadow:0 1px 6px rgba(0,0,0,0.03); }
</style>""", unsafe_allow_html=True)

# ==================== 数据加载（缓存） ====================
import hashlib

def _data_mtime() -> str:
    data_path = os.path.join(_project_root, "data", "药食同源_真实关系链（去重）(1).xlsx")
    if not os.path.exists(data_path):
        return "no_file"
    with open(data_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

@st.cache_resource
def get_loader(_hash: str):
    return GraphDataLoader()

loader = get_loader(_data_mtime())
all_diseases = loader.all_diseases_cn  # 全部中文名（搜索用）
all_diseases_default = loader.all_diseases_cn_quality  # 高质量中文名（默认展示）


def fuzzy_search(query: str, candidates: list, top_k: int = 8) -> list:
    if not query:
        return candidates[:top_k]
    scored = []
    for c in candidates:
        score = SequenceMatcher(None, query.lower(), c.lower()).ratio()
        if query.lower() in c.lower():
            score += 0.5
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_k] if s > 0]

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 🌿 参数设置")
    st.markdown("---")
    st.markdown("#### 🔍 搜索疾病")
    search_query = st.text_input(
        "输入疾病名称", placeholder="例如：糖尿病、高血压、失眠...",
        label_visibility="collapsed",
    )
    if search_query.strip():
        # 搜索时匹配全部疾病（含英文残留名，确保不遗漏）
        matched = fuzzy_search(search_query.strip(), all_diseases, top_k=15)
    else:
        # 默认展示高质量翻译中文名
        matched = all_diseases_default

    if matched:
        selected_disease = st.selectbox(
            "匹配的疾病（共 {} 个结果）".format(len(matched)),
            options=matched, index=0,
            label_visibility="collapsed",
        )
    else:
        selected_disease = None
        st.markdown(
            '<div class="search-hint">'
            "🔎 未找到匹配疾病，请尝试其他关键词（支持中英文）</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    top_k = st.slider("展示 Top N 中药", 5, 30, 15)
    generate_btn = st.button(
        "🌿 查询知识图谱", type="primary",
        use_container_width=True, disabled=(selected_disease is None),
    )
    st.markdown("---")
    if st.button("🔄 刷新缓存", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ==================== 主页面 ====================
st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-title">🌿 药食同源饮品智能配方系统</div>'
    '<div class="hero-subtitle">'
    "基于网络药理学知识图谱（中药 → 化合物 → 靶点 → 疾病）<br>"
    "为 12 种常见疾病智能推荐最优药食同源物质组合"
    "</div></div>",
    unsafe_allow_html=True,
)

if selected_disease is None:
    st.info("👈 请在左侧边栏搜索并选择一种疾病，然后点击「🌿 查询知识图谱」。")
    st.stop()

if not generate_btn:
    st.stop()

# ==================== 执行图谱查询 ====================
with st.spinner("🌿 正在知识图谱中检索最优中药组合…"):
    stats = loader.get_graph_stats(selected_disease)
    ranked = loader.rank_herbs_for_disease(selected_disease, top_k=top_k)

if not ranked:
    st.warning(f"知识图谱中未找到与「{selected_disease}」直接关联的中药数据。")
    st.stop()

en_name = CN_TO_EN_DISEASE.get(selected_disease, selected_disease)

# ==================== KPI 卡片 ====================
st.markdown("---")
st.markdown("### 📊 图谱检索概览")
c1, c2, c3, c4 = st.columns(4)
c1.metric("关联靶点", f"{stats['关联靶点数']} 个")
c2.metric("关联化合物", f"{stats['关联化合物数']} 个")
c3.metric("相关药食同源中药", f"{stats['相关中药数']} 种")
c4.metric("疾病英文名", en_name)

# ==================== 中药排名 ====================
st.markdown("---")
st.markdown(f"### 🏆 「{selected_disease}」推荐中药 Top {min(top_k, len(ranked))}")

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    for i, herb in enumerate(ranked):
        gold = i == 0
        border_color = "#FF8F00" if gold else "#2E7D32"
        bg = "linear-gradient(135deg,#FFF8E1,#FFF3E0)" if gold else "linear-gradient(135deg,#FFFFFF,#FAF8F2)"
        evi_text = " → ".join(
            f"{t}({c})" for t, c in herb["证据链"][:3]
        )
        st.markdown(
            f'<div class="herb-card" style="border-left-color:{border_color};background:{bg};">'
            f'{"🥇" if gold else ""} '
            f'<span class="herb-name">{i+1}. {herb["中药名"]}</span><br>'
            f'<span class="stats">'
            f'关联靶点: <b>{herb["关联靶点数"]}</b> 个 | '
            f'关联化合物: <b>{herb["关联化合物数"]}</b> 个'
            f'</span><br>'
            f'<span class="evidence">靶点→化合物: {evi_text}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ==================== 右栏：横向柱状图 ====================
with right_col:
    st.markdown("#### 📊 关联强度 Top 10")
    top10 = ranked[:10]
    names = [h["中药名"] for h in reversed(top10)]
    targets = [h["关联靶点数"] for h in reversed(top10)]
    compounds = [h["关联化合物数"] for h in reversed(top10)]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    y_pos = range(len(names))
    bar_h = 0.35
    ax.barh([y + bar_h/2 for y in y_pos], targets, bar_h,
            color="#2E7D32", alpha=0.85, label="关联靶点数", edgecolor="white")
    ax.barh([y - bar_h/2 for y in y_pos], compounds, bar_h,
            color="#A5D6A7", alpha=0.85, label="关联化合物数", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("数量", fontsize=9, color="#666")
    ax.set_title(f"「{selected_disease}」图谱关联强度", fontsize=12, fontweight="bold", color="#2E7D32")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    st.pyplot(fig)

# ==================== AI 疾病建议 ====================
st.markdown("---")
st.markdown("### 🤖 AI 健康建议")
# 构建知识图谱上下文，喂给 AI 提升建议质量
graph_ctx = ""
if stats:
    herbs = ranked[:5]
    herb_str = "、".join(h["中药名"] for h in herbs)
    graph_ctx = (
        f"该疾病关联{stats['关联靶点数']}个靶点、{stats['关联化合物数']}种化合物，"
        f"涉及{stats['相关中药数']}种药食同源中药，"
        f"图谱推荐前5名：{herb_str}。"
    )
advice = get_disease_advice(selected_disease, graph_ctx)
if advice:
    source = advice.get("来源", "")
    if "AI建议" in advice:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#F0F7EE,#FAF8F3);'
            f'border-radius:14px;padding:1.2rem 1.5rem;border-left:4px solid #2E7D32;'
            f'line-height:1.8;font-size:0.95rem;color:#333;">{advice["AI建议"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        if "概述" in advice:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#F0F7EE,#FAF8F3);'
                f'border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;'
                f'border-left:4px solid #2E7D32;font-size:0.95rem;color:#333;">'
                f'<b>📖 概述：</b>{advice["概述"]}</div>',
                unsafe_allow_html=True,
            )
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### ⚠️ 注意事项")
            for item in advice.get("注意事项", []):
                st.markdown(
                    f'<div style="background:#FFF8F0;border-radius:8px;padding:0.5rem 0.8rem;'
                    f'margin-bottom:0.4rem;font-size:0.9rem;color:#555;border:1px solid #F0E0D0;">'
                    f'• {item}</div>', unsafe_allow_html=True,
                )
        with cb:
            st.markdown("#### 🥗 推荐饮食")
            for item in advice.get("推荐饮食", []):
                st.markdown(
                    f'<div style="background:#F0F8F0;border-radius:8px;padding:0.5rem 0.8rem;'
                    f'margin-bottom:0.4rem;font-size:0.9rem;color:#555;border:1px solid #D0E8D0;">'
                    f'• {item}</div>', unsafe_allow_html=True,
                )
        if advice.get("生活建议"):
            st.markdown("#### 🏃 生活建议")
            cols_life = st.columns(3)
            for i, item in enumerate(advice["生活建议"]):
                with cols_life[i % 3]:
                    st.markdown(
                        f'<div style="background:#F5F0FA;border-radius:8px;padding:0.6rem 0.8rem;'
                        f'margin-bottom:0.4rem;font-size:0.88rem;color:#555;text-align:center;'
                        f'border:1px solid #E0D5F0;">{item}</div>',
                        unsafe_allow_html=True,
                    )
    st.caption(f"💡 建议来源: {source}")
else:
    st.info("暂未收录该疾病的健康建议数据。")

# ==================== 图谱参考 ====================
st.markdown("---")
with st.expander(f"🔬 「{selected_disease}」知识图谱检索路径（专业参考）", expanded=False):
    st.markdown(f"**疾病** → {en_name}")
    target_ids = loader.get_targets_by_disease(selected_disease)
    st.markdown(f"**关联靶点** ({len(target_ids)} 个): {', '.join(sorted(target_ids)[:20])}")
    st.markdown("**关系链**: 疾病 → 靶点 → 化合物 → 中药")
    st.markdown(f"- 中药-化合物: {len(loader.herb_compound_df):,} 条")
    st.markdown(f"- 化合物-靶点: {len(loader.compound_target_df):,} 条")
    st.markdown(f"- 靶点-疾病: {len(loader.target_disease_df):,} 条")

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    '<div class="footer-note">'
    "🌱 本系统基于网络药理学知识图谱，数据来源于 TCMSP 等公共数据库。"
    "通过 中药→化合物→靶点→疾病 关系链智能推荐药食同源物质。"
    "</div>", unsafe_allow_html=True,
)

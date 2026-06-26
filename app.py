"""
药食同源饮品辅助配方推荐系统 —— Streamlit 可视化界面（草本风格升级版）
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

from src.data_loader import DataLoader
from src.optimizer import FormulaOptimizer
from src.disease_advice import get_disease_advice

# ==================== matplotlib 中文配置 ====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 页面配置 ====================
st.set_page_config(page_title="药食同源饮品配方推荐系统", page_icon="🌿", layout="wide")

# ==================== 自定义 CSS ====================
st.markdown(
    """
<style>
/* ===== 全局背景：淡米色底 + 草本叶脉暗纹 SVG ===== */
body, .stApp {
    background-color: #FDFBF7;
    background-image: url("data:image/svg+xml,%3Csvg width='120' height='120' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M20 60 Q40 30 60 60 Q40 90 20 60' fill='none' stroke='%23D4CFC4' stroke-width='0.6' opacity='0.3'/%3E%3Ccircle cx='90' cy='30' r='1.5' fill='%23C8C0B0' opacity='0.25'/%3E%3Ccircle cx='100' cy='90' r='1.2' fill='%23C8C0B0' opacity='0.2'/%3E%3Ccircle cx='30' cy='100' r='1.8' fill='%23C8C0B0' opacity='0.22'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 120px 120px;
}
.main .block-container {
    padding-top: 1.5rem;
    background: transparent;
}

/* ===== 顶部渐变装饰条 ===== */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #A5D6A7 0%, #2E7D32 30%, #43A047 50%, #2E7D32 70%, #A5D6A7 100%);
    z-index: 9999;
    pointer-events: none;
}

/* ===== 标题与文字 ===== */
h1, h2, h3, h4 {
    color: #2E7D32 !important;
}
p, span, div {
    color: #333333;
}

/* ===== 水平分割线美化 ===== */
hr, [data-testid="stDivider"] {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent 0%, #C8D6B8 20%, #A5B896 50%, #C8D6B8 80%, transparent 100%) !important;
    margin: 1.5rem 0 !important;
}

/* ===== 侧边栏：纹理背景 + 毛玻璃 ===== */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #F8F3E8 0%, #F0EAD6 50%, #F5F1E8 100%);
    border-right: 1px solid #E0D8C8;
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg width='80' height='80' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M15 45 Q30 20 45 45 Q30 70 15 45' fill='none' stroke='%23D8CFB8' stroke-width='0.5' opacity='0.35'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 80px 80px;
    pointer-events: none;
    z-index: 0;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #333333 !important;
    font-weight: 600;
}

/* ===== 按钮：多层阴影 + 光泽高光 ===== */
div.stButton > button {
    background: linear-gradient(135deg, #2E7D32 0%, #388E3C 40%, #43A047 100%);
    color: #FFFFFF !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    box-shadow:
        0 4px 14px rgba(46, 125, 50, 0.35),
        0 1px 3px rgba(46, 125, 50, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.15);
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 40%, #388E3C 100%);
    box-shadow:
        0 6px 22px rgba(46, 125, 50, 0.48),
        0 2px 5px rgba(46, 125, 50, 0.25),
        inset 0 1px 0 rgba(255,255,255,0.12);
    transform: translateY(-2px);
}
div.stButton > button:disabled {
    background: linear-gradient(135deg, #C8C8C8, #D4D4D4);
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    color: #999999 !important;
}

/* ===== 指标卡片：微渐变底 + 左绿线 + 悬浮 ===== */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF8F3 100%);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0,0,0,0.04);
    border-left: 4px solid #2E7D32;
    transition: box-shadow 0.25s, transform 0.25s;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.09);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: #777777 !important;
    font-size: 0.85rem;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #2E7D32 !important;
    font-weight: 800;
}

/* ===== 配方结果卡片：渐变框 + 立体层次 ===== */
.recipe-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF8F2 100%);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.9rem;
    box-shadow:
        0 3px 16px rgba(0, 0, 0, 0.05),
        0 1px 4px rgba(0, 0, 0, 0.04);
    border-left: 5px solid #2E7D32;
    transition: all 0.25s ease;
    position: relative;
}
.recipe-card::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(46,125,50,0.04) 0%, transparent 70%);
    border-radius: 0 16px 0 0;
    pointer-events: none;
}
.recipe-card:hover {
    box-shadow:
        0 8px 28px rgba(0, 0, 0, 0.10),
        0 2px 6px rgba(0, 0, 0, 0.06);
    transform: translateY(-2px);
}
.recipe-card .substance-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #2E7D32;
}
.recipe-card .dose {
    font-size: 1.5rem;
    font-weight: 800;
    color: #D84315;
}
.recipe-card .detail {
    font-size: 0.88rem;
    color: #999999;
    margin-top: 0.25rem;
}

/* ===== 搜索建议卡片 ===== */
.search-hint {
    background: linear-gradient(135deg, #FFF8E1, #FFF3CD);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid #FFE082;
    font-size: 0.9rem;
    color: #795548;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ===== 标题区域：渐变 Hero Banner ===== */
.hero-banner {
    background: linear-gradient(160deg, #F0F7EE 0%, #FDFBF7 30%, #F7F3E8 70%, #EEF5EA 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow:
        0 2px 20px rgba(46, 125, 50, 0.06),
        inset 0 0 60px rgba(165, 214, 167, 0.15);
    border: 1px solid rgba(165, 214, 167, 0.25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -30px; left: -30px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, rgba(46,125,50,0.06) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -20px; right: -20px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(165,214,167,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 0.4rem;
    position: relative;
    z-index: 1;
    text-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #888888;
    text-align: center;
    margin-bottom: 0;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

/* ===== 绿色装饰角标 (章节标题左侧小叶子) ===== */
.section-icon {
    display: inline-block;
    margin-right: 0.35rem;
    font-size: 1.1em;
}

/* ===== 页脚装饰线 ===== */
.footer-note {
    text-align: center;
    color: #B0B0B0;
    padding: 0.5rem 0;
}

/* ===== info / success / warning 面板柔化 ===== */
div[data-testid="stInfo"], div[data-testid="stSuccess"] {
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ===== expander 展开面板 ===== */
details[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #E5DFD0 !important;
    background: linear-gradient(180deg, #FDFCFA, #F9F6EF) !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.03);
}

/* ===== Selectbox 下拉框优化 ===== */
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #D4CFC4 !important;
}

/* ===== Slider 滑条染色 ===== */
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    background: #2E7D32 !important;
    color: #FFFFFF !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==================== 数据加载（缓存：依赖文件修改时间自动刷新） ====================
import hashlib

def _data_mtime() -> str:
    """计算 data.xlsx 的 MD5，文件内容变化时缓存自动失效。"""
    data_path = os.path.join(_project_root, "data", "data.xlsx")
    if not os.path.exists(data_path):
        return "no_file"
    with open(data_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@st.cache_resource
def get_loader(_file_hash: str):
    return DataLoader()


@st.cache_resource
def get_optimizer(_file_hash: str):
    return FormulaOptimizer()


file_hash = _data_mtime()
loader = get_loader(file_hash)
optimizer = get_optimizer(file_hash)
all_diseases = sorted(loader.disease_to_targets.keys())


# ==================== 模糊匹配工具 ====================
def fuzzy_search(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    """对候选列表进行模糊匹配，返回相似度排序后的前 top_k 个结果。"""
    if not query:
        return candidates[:top_k]
    scored = []
    for c in candidates:
        score = SequenceMatcher(None, query.lower(), c.lower()).ratio()
        # 子串匹配加分
        if query.lower() in c.lower():
            score += 0.5
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_k] if s > 0]


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 🌿 配方参数设置")
    st.markdown("---")

    # 疾病搜索（输入框 + 模糊匹配下拉列表）
    st.markdown("#### 🔍 搜索疾病")
    search_query = st.text_input(
        "输入疾病名称或症状关键词",
        placeholder="例如：糖尿病、高血压、肥胖...",
        label_visibility="collapsed",
    )

    if search_query.strip():
        matched = fuzzy_search(search_query.strip(), all_diseases)
    else:
        matched = all_diseases

    if matched:
        selected_disease = st.selectbox(
            "匹配的疾病",
            options=matched,
            index=0,
            help="从模糊匹配结果中选择",
            label_visibility="collapsed",
        )
    else:
        selected_disease = None
        st.markdown(
            '<div class="search-hint">'
            "🔎 未找到匹配疾病，请尝试：糖尿病、高血压、高血脂、失眠 等"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 剂型
    st.markdown("#### 💊 选择剂型")
    formula_type = st.radio(
        "剂型",
        options=["代茶饮", "汤剂"],
        index=0,
        help="代茶饮：总重 ≦ 5g，模拟独立茶包；汤剂：无总重限制，模拟煎煮",
        label_visibility="collapsed",
    )
    st.caption(
        "代茶饮：总重 ≦ 5g（模拟独立茶包）" if formula_type == "代茶饮"
        else "汤剂：无总量限制（模拟传统煎煮）"
    )

    st.markdown("---")

    # 最低用量
    st.markdown("#### ⚖️ 最低用量")
    min_dose = st.slider(
        "单物质最低用量 (g)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="防止以极微量骗取协同奖励，建议 ≥ 0.5g",
    )

    st.markdown("---")

    # 生成按钮
    generate_btn = st.button(
        "🌿 生成智能配方",
        type="primary",
        use_container_width=True,
        disabled=(selected_disease is None),
    )

    st.markdown("---")
    st.caption("💡 基于 MILP 运筹优化模型 · PuLP 求解器")

    st.markdown("---")
    if st.button("🔄 刷新数据缓存", use_container_width=True,
                 help="更新 data.xlsx 后点击此按钮重新加载"):
        st.cache_resource.clear()
        st.rerun()

# ==================== 主页面 ====================
# --- 标题区域（Hero Banner）---
st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-title">🌿 药食同源饮品智能配方系统</div>'
    '<div class="hero-subtitle">'
    "基于运筹优化（MILP）模型，为不同疾病精准匹配最优药食同源物质组合。<br>"
    "综合考虑靶点起效剂量、消化吸收率与协同增效，使配方总重量最小化。"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# --- 无选择时的引导 ---
if selected_disease is None:
    st.info("👈 请在左侧边栏搜索并选择一种疾病，然后点击「🌿 生成智能配方」按钮。")
    st.stop()

if not generate_btn:
    st.stop()

# ==================== 执行求解 ====================
with st.spinner("🌿 正在通过运筹优化算法为您计算最优配方…"):
    result = optimizer.optimize(selected_disease, formula_type, min_dose=min_dose)

# --- 无解处理 ---
if result["状态"] != "Optimal":
    st.error(
        f"😔 很抱歉，当前条件下「{selected_disease}」的配方无可行解。\n\n"
        f"求解状态: **{result['状态']}**。\n\n"
        "建议尝试：\n"
        "- 切换为「汤剂」（无总量限制）\n"
        "- 降低单物质最低用量\n"
        "- 确认数据中已配置该疾病的靶点与候选物质"
    )
    st.stop()

# ==================== KPI 指标卡片 ====================
st.markdown("---")
st.markdown("### 📊 配方概览")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总重量", f"{result['总重量_g']:.2f} g")
with col2:
    st.metric("包含物质数", f"{len(result['配方'])} 种")
with col3:
    # 协同增效倍数：有协同奖励的物质数
    syn_count = sum(1 for item in result["配方"] if item["其中协同奖励_mg"] > 0)
    st.metric("协同增效物质数", f"{syn_count} 种")

# ==================== 结果 + 图表区 ====================
st.markdown("---")
st.markdown("### 🧪 推荐配方 & 可视化分析")

left_col, right_col = st.columns([1, 1], gap="large")

# ---- 左侧：配方清单 ----
with left_col:
    st.markdown("#### 📦 配方清单")
    if not result["配方"]:
        st.warning("未生成任何配方项。")
    for idx, item in enumerate(result["配方"]):
        synergy_html = ""
        if item["其中协同奖励_mg"] > 0:
            synergy_html = (
                f'<span class="detail">⚡ 协同奖励: '
                f'+{item["其中协同奖励_mg"]:.1f} mg</span>'
            )
        st.markdown(
            f'<div class="recipe-card">'
            f'<span class="substance-name">{idx + 1}. {item["物质"]}</span><br>'
            f'<span class="dose">{item["重量_g"]:.2f} g</span><br>'
            f'<span class="detail">有效吸收量: {item["有效吸收量_mg"]:.1f} mg</span>'
            f'{synergy_html}'
            f"</div>",
            unsafe_allow_html=True,
        )

# ---- 右侧：饼图 + 雷达图 ----
with right_col:
    # --- 饼图 ---
    st.markdown("#### 🍩 重量占比分布")
    labels = [item["物质"] for item in result["配方"]]
    weights = [item["重量_g"] for item in result["配方"]]
    # 专业调色盘（低饱和度大地色系）
    pie_colors = ["#C8B293", "#7BA686", "#D4A76A", "#6B8F71", "#BFA578", "#8BA88F"]

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    fig1.patch.set_facecolor("none")
    ax1.set_facecolor("none")
    wedges, texts, autotexts = ax1.pie(
        weights,
        labels=labels,
        autopct="%1.1f%%",
        colors=pie_colors[: len(labels)],
        startangle=90,
        textprops={"fontsize": 11, "color": "#333333"},
        pctdistance=0.58,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(12)
        at.set_color("#333333")
    ax1.set_title(
        f"「{selected_disease}」配方组成",
        fontsize=13,
        fontweight="bold",
        color="#2E7D32",
        pad=16,
    )
    st.pyplot(fig1)

    st.markdown("---")

    # --- 雷达图 ---
    st.markdown("#### 🎯 靶点起效剂量满足度")

    target_labels = [t["靶点"] for t in result["靶点满足情况"]]
    satisfaction_pct = [
        min(t["实际_mg"] / t["需要_mg"] * 100, 200)
        for t in result["靶点满足情况"]
    ]

    num_targets = len(target_labels)
    if num_targets >= 3:
        angles = np.linspace(0, 2 * np.pi, num_targets, endpoint=False).tolist()
        angles += angles[:1]
        values_closed = satisfaction_pct + satisfaction_pct[:1]

        fig2, ax2 = plt.subplots(
            figsize=(4.8, 4.8), subplot_kw={"projection": "polar"}
        )
        fig2.patch.set_facecolor("none")
        ax2.set_facecolor("#FAF8F3")

        # 浅色填充
        ax2.fill(angles, values_closed, alpha=0.20, color="#2E7D32")
        ax2.plot(
            angles, values_closed, "o-", linewidth=2.2,
            color="#2E7D32", markersize=8,
            markerfacecolor="#FFFFFF", markeredgewidth=2,
            markeredgecolor="#2E7D32",
        )
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(target_labels, fontsize=11, color="#333333")
        ax2.set_ylim(0, 220)
        ax2.set_yticks([0, 50, 100, 150, 200])
        ax2.set_yticklabels(["0%", "50%", "100%", "150%", "200%"], fontsize=8, color="#888888")
        ax2.axhline(y=100, color="#43A047", linestyle="--", linewidth=1.5, alpha=0.7, label="100% 满足线")
        ax2.set_title("靶点起效剂量满足度 (%)", fontsize=13, fontweight="bold", color="#2E7D32", pad=18)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
        ax2.grid(True, alpha=0.25, color="#AAAAAA")

        # 数值标注
        for i, (angle, val) in enumerate(zip(angles[:-1], satisfaction_pct)):
            ax2.annotate(
                f"{val:.0f}%",
                xy=(angle, val),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#2E7D32",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#CCCCCC", alpha=0.85),
            )

        st.pyplot(fig2)
    else:
        st.info(f"当前仅 {num_targets} 个靶点（雷达图需 ≥ 3 个维度），请在下方查看靶点明细。")

# ==================== AI 疾病建议 ====================
st.markdown("---")
st.markdown("### 🤖 AI 健康建议")

advice = get_disease_advice(selected_disease)
if advice:
    source = advice.get("来源", "")
    # Claude AI 原文直接展示
    if "AI建议" in advice:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#F0F7EE,#FAF8F3);'
            f'border-radius:14px;padding:1.2rem 1.5rem;'
            f'border-left:4px solid #2E7D32;line-height:1.8;font-size:0.95rem;'
            f'color:#333;">{advice["AI建议"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # 内置知识库卡片式展示
        if "概述" in advice:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#F0F7EE,#FAF8F3);'
                f'border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;'
                f'border-left:4px solid #2E7D32;font-size:0.95rem;color:#333;">'
                f'<b>📖 概述：</b>{advice["概述"]}</div>',
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ⚠️ 注意事项")
            for item in advice.get("注意事项", []):
                st.markdown(
                    f'<div style="background:#FFF8F0;border-radius:8px;padding:0.5rem 0.8rem;'
                    f'margin-bottom:0.4rem;font-size:0.9rem;color:#555;'
                    f'border:1px solid #F0E0D0;">• {item}</div>',
                    unsafe_allow_html=True,
                )
        with col_b:
            st.markdown("#### 🥗 推荐饮食")
            for item in advice.get("推荐饮食", []):
                st.markdown(
                    f'<div style="background:#F0F8F0;border-radius:8px;padding:0.5rem 0.8rem;'
                    f'margin-bottom:0.4rem;font-size:0.9rem;color:#555;'
                    f'border:1px solid #D0E8D0;">• {item}</div>',
                    unsafe_allow_html=True,
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
    st.info(f"暂未收录「{selected_disease}」的健康建议数据。可配置 ANTHROPIC_API_KEY 环境变量启用 AI 实时生成。")

# ==================== 靶点明细（折叠） ====================
st.markdown("---")
with st.expander("🎯 靶点满足度明细", expanded=False):
    cols = st.columns(len(result["靶点满足情况"]) or 1)
    for i, t in enumerate(result["靶点满足情况"]):
        ratio = t["实际_mg"] / t["需要_mg"] * 100
        with cols[i % len(cols)]:
            st.metric(
                label=f"{'✅' if t['满足'] else '❌'} {t['靶点']}",
                value=f"{ratio:.1f}%",
                delta=f"需 {t['需要_mg']} mg → 实 {t['实际_mg']:.1f} mg",
            )

# --- 疾病靶点参考（仅供专业查看）---
with st.expander(f"📋 「{selected_disease}」关联靶点与候选物质（专业参考）", expanded=False):
    resolved = loader.resolve_disease_to_substances(selected_disease)
    if resolved:
        targets = {}
        for r in resolved:
            t = r["靶点"]
            targets.setdefault(t, {"剂量": r["起效剂量mg"], "物质": []})
            targets[t]["物质"].append(r["物质"])
        for t, info in targets.items():
            st.markdown(
                f"- **{t}**（起效剂量 {info['剂量']} mg）"
                f"→ 候选: {', '.join(sorted(set(info['物质'])))}"
            )
    synergy_pairs = []
    for (a, b), coef in loader.synergy_map.items():
        if a < b:
            synergy_pairs.append((a, b, coef))
    if synergy_pairs:
        st.caption(
            "🤝 协同组合: " + " | ".join(
                f"{a} + {b} (×{coef})" for a, b, coef in synergy_pairs
            )
        )

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    '<div class="footer-note">'
    "🌱 本系统基于混合整数线性规划（MILP）模型，由 PuLP 求解器驱动。"
    "在满足所有靶点起效剂量的前提下，使配方总重量最小化，并自动识别协同增效机会。"
    "</div>",
    unsafe_allow_html=True,
)

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
import json, time, urllib.request, urllib.error
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
.progress-bar { height: 6px; border-radius: 3px; background: #E0E0E0; margin-top: 0.4rem; overflow: hidden; }
.progress-bar .fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #2E7D32, #43A047);
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }

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
  position: relative;
  overflow: hidden;
  z-index: 0;
}
.chat-container::before {
  content: "";
  position: absolute;
  top: -70px; right: -50px;
  width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(139,195,74,0.07) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
  z-index: -1;
}
@keyframes dietChatGlow {
  0%, 100% { box-shadow: 0 0 32px rgba(129,199,132,0.08), 0 6px 24px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.6); border-color: rgba(129,199,132,0.28); }
  50% { box-shadow: 0 0 56px rgba(129,199,132,0.20), 0 6px 28px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.7); border-color: rgba(129,199,132,0.50); }
}

/* === 聊天消息气泡 === */
.chat-msg-user, .chat-msg-ai {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 0.9rem;
  animation: msgFadeIn 0.35s ease-out;
}
@keyframes msgFadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.chat-avatar {
  font-size: 1.4rem;
  line-height: 1;
  flex-shrink: 0;
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%;
  background: rgba(255,255,255,0.7);
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.chat-bubble {
  max-width: 82%;
  padding: 0.75rem 1rem;
  border-radius: 16px;
  font-size: 0.93rem;
  line-height: 1.75;
  word-break: break-word;
}
.chat-msg-user .chat-bubble {
  background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
  border: 1px solid rgba(46,125,50,0.12);
  border-bottom-right-radius: 4px;
}
.chat-msg-ai .chat-bubble {
  background: linear-gradient(135deg, #FFFFFF, #FDFBF7);
  border: 1px solid rgba(0,0,0,0.06);
  border-bottom-left-radius: 4px;
}
.chat-msg-user { flex-direction: row-reverse; }
.chat-msg-user .chat-bubble { text-align: right; }

/* === 内联输入区域 === */
[data-testid="stForm"] {
  margin-top: 0.8rem;
  border: none !important;
  padding: 0 !important;
}
.chat-container textarea {
  border-radius: 14px !important;
  border: 1.5px solid rgba(129,199,132,0.35) !important;
  background: rgba(255,255,255,0.65) !important;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  font-size: 0.92rem !important;
  resize: none !important;
}
.chat-container textarea:hover {
  border-color: rgba(67,160,71,0.50) !important;
  box-shadow: 0 3px 16px rgba(46,125,50,0.08);
}
.chat-container textarea:focus {
  border-color: #43A047 !important;
  box-shadow: 0 0 0 4px rgba(67,160,71,0.10) !important;
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


def ask_gemini_diet_assistant(messages, disease_context=""):
    """向 Gemini 2.5 Flash API 发送请求，含指数退避重试（最多 5 次）。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
        except Exception:
            api_key = ""

    if not api_key:
        return _diet_fallback(disease_context)

    system_instruction = DIET_SYSTEM_INSTRUCTION
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
    matched = fuzzy_search(search_query.strip(), all_diseases, top_k=15) if search_query.strip() else all_diseases_default

    if matched:
        selected_disease = st.selectbox("匹配结果（{}条）".format(len(matched)), options=matched, index=0, label_visibility="collapsed")
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

        st.markdown(
            f'<div class="herb-card" style="border-left-color:{border_c};background:{rank_bg};">'
            f'<div style="font-size:1.8rem;min-width:44px;text-align:center;">{medal_c}</div>'
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
# 初始化聊天历史
if "diet_messages" not in st.session_state:
    st.session_state.diet_messages = []

# 对话容器
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# 渲染历史消息
for msg in st.session_state.diet_messages:
    role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
    avatar = "🧑" if msg["role"] == "user" else "🌿"
    st.markdown(
        f'<div class="{role_class}"><span class="chat-avatar">{avatar}</span>'
        f'<div class="chat-bubble">{msg["content"]}</div></div>',
        unsafe_allow_html=True,
    )

# 内联输入区域（不悬浮）
with st.form("diet_chat_form", clear_on_submit=True, border=False):
    cols = st.columns([9, 1], gap="small")
    with cols[0]:
        user_input = st.text_area(
            "输入",
            placeholder="💬 询问关于该亚健康状态的日常膳食调理建议...",
            label_visibility="collapsed",
            height=68,
            key="diet_chat_input",
        )
    with cols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("📨", use_container_width=True, type="primary")

if submitted and user_input.strip():
    prompt = user_input.strip()
    st.session_state.diet_messages.append({"role": "user", "content": prompt})
    with st.spinner("🌿 AI 正在为您生成膳食建议..."):
        herbs_top = ranked[:5] if ranked else []
        disease_ctx = (
            f"当前查询疾病：「{selected_disease}」\n"
            f"知识图谱：{stats['关联靶点数']} 个蛋白质靶点、{stats['关联化合物数']} 种化合物、{stats['相关中药数']} 种药食同源中药。\n"
            f"图谱 Top 5 推荐：{'、'.join(h['中药名'] for h in herbs_top)}。"
        ) if stats and ranked else f"当前查询疾病：「{selected_disease}」"
        response = ask_gemini_diet_assistant(st.session_state.diet_messages, disease_ctx)
    st.session_state.diet_messages.append({"role": "assistant", "content": response})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

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

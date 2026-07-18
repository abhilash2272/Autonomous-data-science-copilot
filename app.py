"""
app.py — Autonomous Data Science Co-Pilot
Groq-powered | Auto RAG | Session History | Dark UI
"""

import os
import sys
import time
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Load Groq key from .env (NEVER hardcode keys here) ───────────────────────
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
if not _GROQ_KEY:
    import streamlit as _st
    _st.set_page_config(page_title="Setup Required", page_icon="⚠️")
    _st.error(
        "### ⚠️ GROQ_API_KEY not found\n\n"
        "Create a `.env` file in the project root with:\n\n"
        "```\nGROQ_API_KEY=your-key-here\n```\n\n"
        "Get your free key at [console.groq.com](https://console.groq.com)"
    )
    _st.stop()

# ── Early imports (must happen before st.set_page_config) ────────────────────
import utils.token_tracker as tt
from utils.loader   import load_dataframe
from utils.charts   import display_chart, display_no_chart_placeholder
from rag.ingest     import ingest_documentation
from rag.vectordb   import collection_exists
from agent.executor import generate_dataset_questions, TokenLimitError
import agent.agent  as ds_agent

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Science Co-Pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state initialisation ────────────────────────────────────────────
tt.init()
for key, val in [
    ("df", None), ("metadata", None), ("result", None),
    ("status_log", []), ("dynamic_questions", []), ("questions_generated", False),
    ("history", []),           # list of past analysis entries
    ("rag_auto_built", False),  # auto-KB build attempted this session
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Auto-build RAG KB on first run (silent) ───────────────────────────────────
if not st.session_state["rag_auto_built"]:
    if not collection_exists():
        with st.spinner("🔄 Building knowledge base for the first time… (~1–2 min)"):
            ingest_documentation(force=False, progress_callback=None)
    st.session_state["rag_auto_built"] = True

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══ Design Tokens ═══════════════════════════════════════════════════ */
:root {
  /* Main canvas */
  --bg:#080B10; --canvas:#0E1321; --surface:#171E30; --surf-2:#202A42; --surf-hi:#2A3754;
  /* Borders */
  --border:rgba(255,255,255,0.07); --border-hi:rgba(255,255,255,0.15);
  /* Brand */
  --navy:#F1F5F9; --navy-2:#E2E8F0;
  --blue:#3B82F6; --blue-dim:rgba(59,130,246,0.08); --blue-ring:rgba(59,130,246,0.20); --blue-glow:rgba(59,130,246,0.24);
  /* Semantic — used sparingly */
  --success:#10B981; --success-dim:rgba(16,185,129,0.08);
  --warning:#F59E0B; --warning-dim:rgba(245,158,11,0.08);
  --danger:#EF4444;  --danger-dim:rgba(239,68,68,0.08);
  /* Text scale */
  --text:#E2E8F0; --text-2:#CBD5E1; --text-muted:#94A3B8; --text-faint:#475569;
  /* Sidebar (stays dark) */
  --sb-bg:#080B10; --sb-surf:#0E1321; --sb-surf-hi:#171E30;
  --sb-border:rgba(255,255,255,0.07); --sb-bhi:rgba(255,255,255,0.13);
  --sb-text:#E2E8F0; --sb-muted:#94A3B8; --sb-faint:#4B5E7A;
  /* Radius */
  --r-xs:4px; --r-sm:6px; --r:8px; --r-lg:12px; --r-xl:16px;
  /* Elevation — 3 levels */
  --el-flat:  0 1px 3px rgba(0,0,0,0.3),0 1px 2px rgba(0,0,0,0.2);
  --el-raised:0 4px 16px rgba(0,0,0,0.5),0 2px 4px rgba(0,0,0,0.3);
  --el-float: 0 12px 32px rgba(0,0,0,0.7),0 4px 8px rgba(0,0,0,0.4);
  --el-blue:  0 4px 16px rgba(59,130,246,0.24);
}


/* ══ Base ════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family:'Inter',sans-serif !important;
  background-color:var(--bg) !important;
  color:var(--text) !important;
  -webkit-font-smoothing:antialiased !important;
}
.stApp { background:var(--bg) !important; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent !important; }
/* Ensure collapsed sidebar toggle button is always prominently visible and interactive */
[data-testid="collapsedControl"] {
  visibility:visible !important; display:flex !important;
  z-index:100000 !important; background:var(--surface) !important;
  border:1px solid var(--border-hi) !important; border-radius:var(--r) !important;
  padding:4px !important; margin:10px !important; box-shadow:var(--el-raised) !important;
}
[data-testid="collapsedControl"] * { visibility:visible !important; color:var(--navy) !important; }
[data-testid="collapsedControl"]:hover { background:var(--blue) !important; border-color:var(--blue) !important; }
[data-testid="collapsedControl"]:hover * { color:#fff !important; }

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--surf-hi); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--border-hi); }

/* ══ Sidebar ═════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background:var(--sb-bg) !important;
  border-right:1px solid var(--sb-border) !important;
}
[data-testid="stSidebar"] * { color:var(--sb-text) !important; }
[data-testid="stSidebar"] .stMarkdown p { color:var(--sb-muted) !important; }

/* Sidebar buttons — dark themed, left-aligned chips */
[data-testid="stSidebar"] .stButton > button {
  background:var(--sb-surf) !important;
  color:var(--sb-text) !important;
  border:1px solid var(--sb-border) !important;
  border-radius:var(--r-sm) !important;
  padding:0.45rem 0.85rem !important;
  font-size:0.81rem !important;
  font-weight:500 !important;
  letter-spacing:0 !important;
  transition:all 0.15s ease !important;
  box-shadow:none !important;
  text-align:left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:var(--sb-surf-hi) !important;
  border-color:rgba(45,107,255,0.45) !important;
  color:#fff !important;
  transform:translateX(3px) translateY(0) !important;
  box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton > button:active {
  transform:translateX(2px) translateY(0) !important;
}

/* Download button in sidebar */
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
  background:var(--sb-surf) !important;
  color:var(--sb-muted) !important;
  border:1px solid var(--sb-border) !important;
  font-size:0.78rem !important;
  padding:0.4rem 0.9rem !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
  background:var(--sb-surf-hi) !important;
  border-color:var(--sb-bhi) !important;
  color:#fff !important;
  transform:translateX(3px) translateY(0) !important;
}

/* ══ Main area buttons ═══════════════════════════════════════════════ */
.stButton > button {
  width:100% !important;
  background:var(--canvas) !important;
  color:var(--text-2) !important;
  border:1px solid var(--border-hi) !important;
  border-radius:var(--r) !important;
  padding:0.55rem 1.1rem !important;
  font-weight:500 !important;
  font-size:0.86rem !important;
  letter-spacing:0.01em !important;
  transition:all 0.15s ease !important;
  box-shadow:var(--el-flat) !important;
}
.stButton > button:hover {
  background:var(--blue-dim) !important;
  border-color:var(--blue) !important;
  color:var(--blue) !important;
  box-shadow:var(--el-raised) !important;
  transform:translateY(-1px) !important;
}
.stButton > button:active {
  transform:translateY(0) !important;
  box-shadow:var(--el-flat) !important;
}



/* ══ Inputs / Textarea ═══════════════════════════════════════════════ */
.stTextArea textarea, .stTextInput input {
  background:var(--canvas) !important;
  border:1px solid var(--border-hi) !important;
  border-radius:var(--r) !important;
  color:var(--text) !important;
  font-family:'Inter',sans-serif !important;
  font-size:0.9rem !important;
  transition:border-color 0.15s ease, box-shadow 0.15s ease !important;
  box-shadow:var(--el-flat) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px var(--blue-ring) !important;
  outline:none !important;
}

/* ══ File uploader ═══════════════════════════════════════════════════ */
[data-testid="stFileUploadDropzone"] {
  background:var(--canvas) !important;
  border:2px dashed var(--border-hi) !important;
  border-radius:var(--r-lg) !important;
  transition:all 0.15s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color:var(--blue) !important;
  background:var(--blue-dim) !important;
}

/* ══ Metrics ══════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {
  color:var(--navy) !important;
  font-weight:700 !important;
  font-size:1.35rem !important;
  line-height:1.2 !important;
}
[data-testid="stMetricLabel"] {
  color:var(--text-2) !important;
  font-size:0.78rem !important;
  font-weight:600 !important;
  letter-spacing:0.07em !important;
  text-transform:uppercase !important;
}
[data-testid="metric-container"] {
  background:var(--canvas) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--r) !important;
  padding:0.85rem !important;
  box-shadow:var(--el-flat) !important;
  transition:box-shadow 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="metric-container"]:hover {
  box-shadow:var(--el-raised) !important;
  border-color:var(--border-hi) !important;
}

/* ══ DataFrames ═══════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border:1px solid var(--border) !important;
  border-radius:var(--r) !important;
  overflow:hidden !important;
  box-shadow:var(--el-flat) !important;
}

/* ══ Tabs — underline style ══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background:transparent !important;
  border-radius:0 !important;
  padding:0 !important;
  gap:0 !important;
  border:none !important;
  border-bottom:2px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius:0 !important;
  color:var(--text-muted) !important;
  font-weight:500 !important;
  font-size:0.86rem !important;
  padding:0.55rem 1.1rem !important;
  transition:color 0.15s ease !important;
  border-bottom:2px solid transparent !important;
  margin-bottom:-2px !important;
  background:transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--navy) !important; }
.stTabs [aria-selected="true"] {
  background:transparent !important;
  color:var(--blue) !important;
  border-bottom:2px solid var(--blue) !important;
  font-weight:600 !important;
  box-shadow:none !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background:var(--canvas) !important;
  border:1px solid var(--border) !important;
  border-top:none !important;
  border-radius:0 0 var(--r) var(--r) !important;
  padding:1.25rem !important;
  box-shadow:var(--el-flat) !important;
}

/* ══ Alerts ══════════════════════════════════════════════════════════ */
.stSuccess {
  background:var(--success-dim) !important;
  border:1px solid rgba(16,185,129,0.25) !important;
  border-left:3px solid var(--success) !important;
  border-radius:var(--r) !important;
}
.stError {
  background:var(--danger-dim) !important;
  border:1px solid rgba(239,68,68,0.25) !important;
  border-left:3px solid var(--danger) !important;
  border-radius:var(--r) !important;
}
.stInfo {
  background:var(--blue-dim) !important;
  border:1px solid rgba(45,107,255,0.22) !important;
  border-left:3px solid var(--blue) !important;
  border-radius:var(--r) !important;
}
.stWarning {
  background:var(--warning-dim) !important;
  border:1px solid rgba(245,158,11,0.25) !important;
  border-left:3px solid var(--warning) !important;
  border-radius:var(--r) !important;
}

/* ══ Selectbox ════════════════════════════════════════════════════════ */
.stSelectbox [data-baseweb="select"] > div {
  background:var(--canvas) !important;
  border-color:var(--border-hi) !important;
  border-radius:var(--r) !important;
  color:var(--text) !important;
  font-weight:500 !important;
  box-shadow:var(--el-flat) !important;
  transition:border-color 0.15s ease !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] [data-testid="stSelectboxVirtualFocusContainer"],
.stSelectbox [data-baseweb="select"] [role="button"],
.stSelectbox [data-baseweb="select"] div {
  color:var(--text) !important;
  background-color:transparent !important;
}
.stSelectbox [data-baseweb="select"] > div:hover { border-color:var(--blue) !important; }
/* Dropdown popover and option list styling */
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background:var(--surface) !important;
  border:1px solid var(--border-hi) !important;
  border-radius:var(--r) !important;
}
[role="option"] {
  color:var(--text) !important;
  background:var(--surface) !important;
  font-size:0.88rem !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
  background:var(--blue-dim) !important;
  color:var(--blue) !important;
}

/* ══ Slider ════════════════════════════════════════════════════════════ */
[data-testid="stSlider"] [role="slider"] {
  background:var(--blue) !important;
  box-shadow:0 0 0 3px var(--blue-ring) !important;
}

/* ══ Progress ══════════════════════════════════════════════════════════ */
.stProgress > div > div > div > div {
  background:linear-gradient(90deg,var(--blue),var(--success)) !important;
  border-radius:99px !important;
}
.stProgress > div > div > div { background:var(--surf-hi) !important; border-radius:99px !important; }

/* ══ Expanders ══════════════════════════════════════════════════════ */
details {
  background:var(--canvas) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--r) !important;
  transition:border-color 0.15s ease, box-shadow 0.15s ease !important;
  box-shadow:var(--el-flat) !important;
}
details:hover {
  border-color:var(--border-hi) !important;
  box-shadow:var(--el-raised) !important;
}
summary { color:var(--text-2) !important; font-weight:600 !important; font-size:0.87rem !important; cursor:pointer !important; }
hr { border-color:var(--border) !important; margin:0.7rem 0 !important; }

/* ══ Code blocks ══════════════════════════════════════════════════════ */
.stCode, [data-testid="stCode"] {
  background:#0D1117 !important;
  border:1px solid #21262D !important;
  border-radius:var(--r) !important;
  box-shadow:var(--el-flat) !important;
}
.stCode code { font-family:'JetBrains Mono',monospace !important; font-size:0.81rem !important; line-height:1.65 !important; color:#E6EDF3 !important; }

/* ══ Spinner ══════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] > div { border-top-color:var(--blue) !important; }

/* ════════ Reusable component classes ═════════════════════════════════ */

/* --- Sidebar brand header --- */
.brand-wrap { padding:0.9rem 0 0.5rem; display:flex; align-items:center; gap:0.6rem; }
.brand-icon {
  font-size:1.4rem; display:flex; align-items:center; justify-content:center;
  width:2.2rem; height:2.2rem; flex-shrink:0;
  background:rgba(45,107,255,0.18); border-radius:var(--r);
  border:1px solid rgba(45,107,255,0.3);
}
.brand-name { font-size:0.88rem; font-weight:700; color:var(--sb-text)!important; margin:0; line-height:1.25; }
.brand-sub  { font-size:0.72rem; color:var(--sb-muted)!important; margin:2px 0 0; }

/* --- Sidebar section label --- */
.sb-label {
  font-size:0.72rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--sb-muted)!important; display:block;
  padding:0.55rem 0 0.25rem; margin-top:0.1rem;
  border-top:1px solid var(--sb-border);
}

/* --- Sidebar KB status pill --- */
.kb-status {
  display:inline-flex; align-items:center; gap:5px; font-size:0.66rem;
  color:var(--sb-muted)!important; background:var(--sb-surf);
  border-radius:99px; padding:3px 10px; border:1px solid var(--sb-border);
  transition:border-color 0.15s ease;
}
.kb-dot { width:6px; height:6px; border-radius:50%; background:var(--success); display:inline-block; flex-shrink:0; }

/* --- Sidebar footer --- */
.sb-footer { padding:0.7rem 0 0.4rem; border-top:1px solid var(--sb-border); text-align:center; margin-top:0.3rem; }
.sb-footer p { font-size:0.7rem; color:var(--sb-muted)!important; margin:3px 0; }

/* --- History card --- */
.hist-card {
  background:var(--sb-surf); border:1px solid var(--sb-border); border-radius:var(--r-sm);
  padding:0.5rem 0.72rem; margin-bottom:0.28rem;
  transition:border-color 0.15s ease, background 0.15s ease;
}
.hist-card:hover { background:var(--sb-surf-hi); border-color:var(--sb-bhi); }
.hist-time { font-size:0.72rem; color:var(--sb-muted)!important; }
.hist-q    { font-size:0.78rem; color:var(--sb-text)!important; margin-top:2px; line-height:1.35; }

/* --- Page hero header --- */
.page-header { padding:0.25rem 0 0.9rem; border-bottom:1px solid var(--border); margin-bottom:0.5rem; }
.page-title {
  font-size:1.3rem; font-weight:800; color:var(--navy);
  letter-spacing:-0.02em; margin:0 0 0.2rem; line-height:1.2;
}
.page-sub { color:var(--text-muted); font-size:0.84rem; margin:0 0 0.65rem; }

/* --- Tech badge (uniform enterprise style) --- */
.tech-badge {
  display:inline-flex; align-items:center; gap:4px;
  padding:3px 10px; border-radius:var(--r-sm); font-size:0.71rem; font-weight:500;
  margin-right:5px; background:var(--surface); color:var(--text-muted);
  border:1px solid var(--border); letter-spacing:0.02em;
  transition:border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
.tech-badge:hover { border-color:var(--blue); color:var(--blue); background:var(--blue-dim); }

/* --- Section label (main area) --- */
.section-lbl {
  font-size:0.75rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--text-muted); display:block; margin-bottom:0.4rem;
}

/* --- Stdout / preformatted output --- */
.output-pre {
  background:#0D1117; border:1px solid #21262D; border-radius:var(--r);
  padding:1rem 1.2rem; font-family:'JetBrains Mono',monospace;
  font-size:0.8rem; color:#E6EDF3; white-space:pre-wrap;
  overflow-x:auto; line-height:1.65; margin-top:0.5rem;
  box-shadow:var(--el-flat);
}

/* --- Empty-state panel --- */
@keyframes float-up { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
.ready-wrap {
  background:var(--canvas); border:1px solid var(--border);
  border-radius:var(--r-xl); padding:3rem 2rem; text-align:center;
  box-shadow:var(--el-flat); margin-top:0.75rem;
}
.ready-icon-wrap {
  font-size:2.2rem; animation:float-up 3s ease-in-out infinite;
  display:inline-flex; align-items:center; justify-content:center;
  width:4rem; height:4rem;
  background:var(--blue-dim); border-radius:var(--r-lg);
  border:1px solid rgba(45,107,255,0.2); margin-bottom:0.75rem;
}

.ready-title { font-size:1.15rem; font-weight:700; color:var(--navy); margin:0 0 0.35rem; }
.ready-sub   { color:var(--text-muted); font-size:0.85rem; margin:0 0 1.6rem; line-height:1.5; }
.steps-row { display:flex; justify-content:center; gap:0.8rem; flex-wrap:wrap; }
.step-card {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
  padding:1.2rem 0.9rem; min-width:95px; flex:1; max-width:125px; text-align:center;
  transition:border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
}
.step-card:hover {
  border-color:var(--blue);
  box-shadow:0 8px 24px rgba(45, 107, 255, 0.08);
  transform:translateY(-2px);
}
.step-n    { font-size:0.57rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.4rem; }
.step-icon { font-size:1.3rem; margin-bottom:0.3rem; }
.step-lbl  { font-size:0.75rem; font-weight:600; color:var(--text-2); }

</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Branding ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="brand-wrap">'
        '<div class="brand-icon">🤖</div>'
        '<div>'
        '<p class="brand-name">Data Science Co-Pilot</p>'
        '<p class="brand-sub">AI-powered analytics</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Settings ─────────────────────────────────────────────────────
    st.markdown('<span class="sb-label">⚙️ Settings</span>', unsafe_allow_html=True)
    model = st.selectbox(
        "LLM Model",
        [
            "llama-3.3-70b-versatile",   # Best quality (default)
            "llama-3.1-8b-instant",      # Fast & lightweight (alternative)
        ],
        index=0, key="model_select",
    )
    os.environ["GROQ_MODEL"] = model

    max_retries = st.slider("Max Self-Heal Retries", 1, 10, 5, key="max_retries_slider")
    os.environ["MAX_RETRIES"] = str(max_retries)

    st.divider()

    # ── Token Usage ──────────────────────────────────────────────────
    st.markdown('<span class="sb-label">⚡ Token Usage</span>', unsafe_allow_html=True)
    totals = tt.get_session_totals()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prompt", f"{totals['prompt_tokens']:,}")
    with col2:
        st.metric("Completion", f"{totals['completion_tokens']:,}")
    
    st.metric("Total Session Tokens", f"{totals['total_tokens']:,}")

    if tt.has_history():
        st.download_button(
            label="⬇️ Download Token History (CSV)",
            data=tt.to_csv_bytes(),
            file_name="token_usage_history.csv",
            mime="text/csv",
            key="dl_tokens_btn",
            use_container_width=True,
        )

    st.divider()

    # ── Sample / Dynamic Questions ──────────────────────────────────────
    st.markdown('<span class="sb-label">💡 Sample Questions</span>', unsafe_allow_html=True)

    if st.session_state.df is not None and st.session_state.dynamic_questions:
        q_list  = st.session_state.dynamic_questions
        q_label = "🎯 Tailored to your dataset:"
    else:
        q_list = [
            "Show sales by region as a bar chart",
            "Find and visualize missing values",
            "Detect and count duplicate rows",
            "Show outliers using a box plot",
            "Plot monthly revenue trends",
            "Group customers by spending tier",
            "Average revenue by product category",
            "Correlation heatmap of all numeric columns",
            "Distribution of ages as histogram",
            "Which product has the highest sales?",
        ]
        q_label = "📋 Click a question to use it:"

    st.caption(q_label)
    with st.expander("💬 Browse questions", expanded=False):
        for _qi, _q in enumerate(q_list):
            if st.button(_q, key=f"sq_{_qi}"):
                st.session_state["question_input"] = _q
                st.rerun()

    if st.session_state.df is not None and not st.session_state.questions_generated:
        st.write("")
        if st.button("🔮 Generate Questions for My Dataset", key="gen_q_btn"):
            with st.spinner("Generating personalised questions…"):
                _meta   = st.session_state.metadata
                _sample = st.session_state.df.head(3).to_string(index=False)
                _qs     = generate_dataset_questions(
                    columns=_meta["columns"], dtypes=_meta["dtypes"],
                    sample_data=_sample, n=10,
                )
            if _qs:
                st.session_state.dynamic_questions   = _qs
                st.session_state.questions_generated = True
                st.rerun()
            else:
                st.warning("Could not generate questions — check your connection.")

    st.divider()

    # ── Analysis History ──────────────────────────────────────────
    st.markdown('<span class="sb-label">📋 History</span>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.caption("_No analyses yet. Run your first query._")
    else:
        for _entry in reversed(st.session_state.history[-20:]):
            _icon  = "✅" if _entry["success"] else "❌"
            _color = "#10B981" if _entry["success"] else "#EF4444"
            _short = _entry["question"][:55] + ("…" if len(_entry["question"]) > 55 else "")
            st.markdown(
                f'<div class="hist-card">'
                f'<div class="hist-time">{_entry["time"]} — '
                f'<span style="color:{_color};font-weight:600;">{_icon}</span></div>'
                f'<div class="hist-q">{_short}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        import io as _hio, pandas as _hpd
        _hist_df = _hpd.DataFrame(st.session_state.history)[["time","question","success","healed","attempts"]]
        st.download_button(
            label="⬇️ Download History (CSV)",
            data=_hist_df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_history.csv",
            mime="text/csv",
            key="dl_history_btn",
            use_container_width=True,
        )

    st.markdown(
        '<div class="sb-footer">'
        '<span class="kb-status"><span class="kb-dot"></span> Knowledge base ready</span>'
        '<p style="margin-top:0.45rem;">Accepts CSV · Excel · JSON</p>'
        '<p>Powered by Groq &amp; ChromaDB</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<p class="page-title">🤖 Autonomous Data Science Co-Pilot</p>'
    '<p class="page-sub">Upload data · Ask questions in plain English · Get professional charts, insights &amp; reports — zero code required</p>'
    '<span class="tech-badge">⚡ Groq LLM</span>'
    '<span class="tech-badge">🔗 LangChain</span>'
    '<span class="tech-badge">🗄 ChromaDB RAG</span>'
    '<span class="tech-badge">🔧 Self-Healing</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — upload + question
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.subheader("📂 Upload Dataset")
    st.caption("Supported: **CSV**, **Excel (.xlsx / .xls)**, **JSON** — PDF not supported")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["csv", "xlsx", "xls", "json"],
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file:
        df, metadata = load_dataframe(uploaded_file)
        if df is not None:
            if (st.session_state.metadata is None or
                    st.session_state.metadata.get("name") != metadata["name"]):
                st.session_state.dynamic_questions   = []
                st.session_state.questions_generated = False
            st.session_state.df       = df
            st.session_state.metadata = metadata

    if st.session_state.df is not None:
        meta = st.session_state.metadata
        st.divider()
        st.subheader("📊 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    f"{meta['rows']:,}")
        c2.metric("Columns", meta["cols"])
        c3.metric("Size",    f"{meta['size_kb']} KB")

        with st.expander("🔍 Preview (first 10 rows)"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)

        with st.expander("📋 Column types & missing values"):
            import pandas as pd
            summary = pd.DataFrame({
                "Column":   meta["columns"],
                "Type":     [meta["dtypes"][c] for c in meta["columns"]],
                "Missing":  [meta["missing"][c] for c in meta["columns"]],
                "Missing%": [f"{meta['missing_pct'][c]}%" for c in meta["columns"]],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Generate Custom Questions ──────────────────────────────────────────
        st.divider()
        st.subheader("🔮 AI-Generated Questions")
        st.caption("Get questions tailored specifically to your dataset's columns and content.")

        if not st.session_state.questions_generated:
            if st.button("✨ Generate Questions from My Dataset",
                         key="gen_q_main_btn", use_container_width=True):
                with st.spinner("🤖 Analysing your dataset and crafting questions…"):
                    _meta   = st.session_state.metadata
                    _sample = st.session_state.df.head(3).to_string(index=False)
                    _qs     = generate_dataset_questions(
                        columns=_meta["columns"], dtypes=_meta["dtypes"],
                        sample_data=_sample, n=10,
                    )
                if _qs:
                    st.session_state.dynamic_questions   = _qs
                    st.session_state.questions_generated = True
                    st.rerun()
                else:
                    st.warning("⚠️ Could not generate questions — check your connection.")
        else:
            st.success("✅ Questions generated! Click any to use it.")
            if st.button("🔄 Regenerate", key="regen_q_btn"):
                st.session_state.dynamic_questions   = []
                st.session_state.questions_generated = False
                st.rerun()

        if st.session_state.dynamic_questions:
            for _gi, _gq in enumerate(st.session_state.dynamic_questions):
                if st.button(
                    f"✦ {_gq}",
                    key=f"gq_{_gi}",
                    use_container_width=True,
                ):
                    st.session_state["question_input"] = _gq
                    st.rerun()


    st.subheader("💬 Ask Your Question")
    question = st.text_area(
        "question_box",
        placeholder="e.g.  Show total sales by region as a bar chart",
        height=110,
        label_visibility="collapsed",
        key="question_input",
    )

    st.write("")
    analyse_clicked = st.button("🚀 Analyse", key="analyse_btn", use_container_width=True)

    if analyse_clicked:
        if st.session_state.df is None:
            st.error("❌ Upload a dataset first.")
            analyse_clicked = False
        elif not question.strip():
            st.error("❌ Type a question first.")
            analyse_clicked = False


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — results + token info
# ══════════════════════════════════════════════════════════════════════════════
with right_col:

    if analyse_clicked and st.session_state.df is not None and question.strip():
        st.session_state.status_log = []
        st.session_state.result     = None

        st.subheader("⚡ Pipeline Execution")
        prog = st.progress(0, text="Initialising…")
        stage_pct = {"INSPECT": 10, "CODEGEN": 30, "EXECUTE": 55, "HEAL": 75, "INSIGHTS": 92}

        def _status(stage, msg):
            st.session_state.status_log.append(f"[{stage}] {msg}")
            prog.progress(stage_pct.get(stage, 50), text=f"{stage}: {msg[:70]}")

        with st.spinner("🤖 AI agent is working…"):
            result = ds_agent.run(
                df=st.session_state.df,
                question=question,
                status_callback=_status,
            )

        prog.progress(100, text="Done!")
        st.session_state.result = result
        time.sleep(0.3)
        prog.empty()

        # ── Record to session history ────────────────────────────────────
        st.session_state.history.append({
            "time":     datetime.datetime.now().strftime("%H:%M:%S"),
            "question": question.strip(),
            "success":  result.get("success", False),
            "healed":   result.get("healed", False),
            "attempts": result.get("attempts", 0),
        })
        # Internal token tracking (no UI shown)
        _usage = result.get("usage")
        if _usage and _usage.total_tokens > 0:
            tt.record(question=question.strip(), usage=_usage)

    # ── Display results ────────────────────────────────────────────────────────
    result = st.session_state.result

    if result:
        # Token-limit banner
        if result.get("token_limit"):
            st.error(
                "⚠️ **Out of Tokens / Rate Limit Reached**\n\n"
                + result.get("error", "Groq token limit exceeded.")
                + "\n\n**Try one of these:**\n"
                "- Wait 60 seconds and try again\n"
                "- Switch to **mixtral-8x7b-32768** in the sidebar\n"
                "- Ask a shorter question\n"
                "- Upgrade at [console.groq.com](https://console.groq.com)"
            )

        else:
            # Status banner
            if result["success"]:
                heal_note = f" — self-healed in {result['attempts']-1} attempt(s)" if result["healed"] else ""
                st.success(f"✅ Analysis complete{heal_note}!")
            else:
                st.error(f"❌ Analysis failed after {result['attempts']} attempt(s).")

            # Result tabs
            tab_chart, tab_insights, tab_code, tab_log = st.tabs([
                "📈 Chart", "💡 Insights", "🧑‍💻 Code", "📋 Log"
            ])

            with tab_chart:
                if result["success"]:
                    shown = display_chart(result.get("chart_path"))
                    if result.get("stdout"):
                        st.markdown("**📊 Data Output:**")
                        # Try to render as a styled table; fall back to code block
                        import io as _io
                        import pandas as _pd
                        _raw = result["stdout"]
                        _rendered = False
                        try:
                            _tdf = _pd.read_csv(_io.StringIO(_raw), sep=r"\s{2,}", engine="python")
                            if len(_tdf.columns) > 1:
                                st.dataframe(_tdf, use_container_width=True)
                                _rendered = True
                        except Exception:
                            pass
                        if not _rendered:
                            st.markdown(
                                f'<div class="output-pre">'
                                + _raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                + '</div>',
                                unsafe_allow_html=True,
                            )
                    if not shown:
                        display_no_chart_placeholder()
                else:
                    st.error("Execution failed — no chart available.")
                    if result.get("error"):
                        st.code(result["error"])

            with tab_insights:
                if result["success"] and result.get("insights"):
                    st.info(result["insights"])
                elif not result["success"]:
                    st.warning("Insights unavailable — execution failed.")
                else:
                    st.info("No textual output produced.")

            with tab_code:
                if result["healed"]:
                    st.warning("🔧 Auto-repaired via RAG over official Python & Pandas docs.")
                st.code(result.get("code", "# No code generated"), language="python")

            with tab_log:
                if st.session_state.status_log:
                    st.code("\n".join(st.session_state.status_log))
                else:
                    st.info("No log entries.")
                if result.get("error") and not result["success"]:
                    with st.expander("⚠️ Full Error Detail"):
                        st.code(result.get("error", ""))

    elif not analyse_clicked:
        st.markdown(
            '<div class="ready-wrap">'
            '<div class="ready-icon-wrap">🤖</div>'
            '<p class="ready-title">Ready to Analyse</p>'
            '<p class="ready-sub">Upload a dataset, ask a question in plain English,<br>then hit <strong style="color:#2D6BFF;">🚀 Analyse</strong> — AI does the rest.</p>'
            '<div class="steps-row">'

            '<div class="step-card">'
            '<div class="step-n" style="color:#2D6BFF;">Step 1</div>'
            '<div class="step-icon">📂</div>'
            '<div class="step-lbl">Upload Data</div>'
            '</div>'

            '<div class="step-card">'
            '<div class="step-n" style="color:#10B981;">Step 2</div>'
            '<div class="step-icon">💬</div>'
            '<div class="step-lbl">Ask a Question</div>'
            '</div>'

            '<div class="step-card">'
            '<div class="step-n" style="color:#F59E0B;">Step 3</div>'
            '<div class="step-icon">📈</div>'
            '<div class="step-lbl">Get Insights</div>'
            '</div>'

            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


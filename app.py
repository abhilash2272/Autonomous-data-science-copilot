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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0d14 !important;
    color: #e8eaf0 !important;
}
.stApp { background: #0a0d14 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1420 0%, #0a0d14 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c63ff 0%, #8b5cf6 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.5) !important;
}

.stTextArea textarea, .stTextInput input {
    background: #131929 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
}

[data-testid="stFileUploadDropzone"] {
    background: rgba(108,99,255,0.04) !important;
    border: 2px dashed rgba(108,99,255,0.4) !important;
    border-radius: 14px !important; transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00d4aa !important;
    background: rgba(0,212,170,0.04) !important;
}

[data-testid="stMetricValue"] { color: #00d4aa !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8892a4 !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] {
    background: #131929 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; padding: 0.8rem !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #131929 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #8892a4 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: #0f1420 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 0 0 12px 12px !important; padding: 1rem !important;
}

.stSuccess { background: rgba(0,212,170,0.08) !important; border-left-color: #00d4aa !important; }
.stError   { background: rgba(255,107,107,0.08) !important; border-left-color: #ff6b6b !important; }
.stInfo    { background: rgba(108,99,255,0.08) !important; border-left-color: #6c63ff !important; }
.stWarning { background: rgba(255,215,0,0.08)  !important; border-left-color: #ffd700 !important; }

.stSelectbox [data-baseweb="select"] > div {
    background: #131929 !important; border-color: rgba(255,255,255,0.08) !important; color: #e8eaf0 !important;
}
.stProgress > div > div > div { background: linear-gradient(90deg, #6c63ff, #00d4aa) !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
details { background: #131929 !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; }
summary { color: #e8eaf0 !important; font-weight: 600 !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Branding ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 0.2rem;">'
        '<span style="font-size:2rem;">🤖</span>'
        '<h2 style="margin:0.2rem 0 0;font-size:1.1rem;font-weight:700;">Data Science Co-Pilot</h2>'
        '<p style="margin:0;font-size:0.75rem;color:#8892a4;">Groq · LangChain · ChromaDB RAG</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Settings ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192",
         "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0, key="model_select",
    )
    os.environ["GROQ_MODEL"] = model

    max_retries = st.slider("Max Self-Heal Retries", 1, 10, 5, key="max_retries_slider")
    os.environ["MAX_RETRIES"] = str(max_retries)


    st.divider()

    # ── Sample / Dynamic Questions ──────────────────────────────────────
    st.markdown("### 💡 Sample Questions")

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
    st.markdown("### 📋 History")
    if not st.session_state.history:
        st.caption("_No analyses yet. Run your first query._")
    else:
        for _entry in reversed(st.session_state.history[-20:]):
            _icon  = "✅" if _entry["success"] else "❌"
            _color = "#00d4aa" if _entry["success"] else "#ff6b6b"
            _short = _entry["question"][:55] + ("…" if len(_entry["question"]) > 55 else "")
            st.markdown(
                f'<div style="background:#131929;border:1px solid rgba(255,255,255,0.06);'
                f'border-radius:10px;padding:0.55rem 0.75rem;margin-bottom:0.4rem;">'
                f'<span style="font-size:0.68rem;color:#8892a4;">{_entry["time"]}</span><br/>'
                f'<span style="color:{_color};font-weight:600;">{_icon} </span>'
                f'<span style="font-size:0.82rem;color:#e8eaf0;">{_short}</span>'
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

    st.divider()
    st.caption("📄 CSV  |  📊 Excel (.xlsx)  |  🗂 JSON")
    st.caption("🌐 Internet connection required")


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Autonomous Data Science Co-Pilot")
st.caption("Upload data · Ask questions in plain English · Get professional charts, insights & reports — **zero code required**")
st.markdown(
    '<span style="background:rgba(108,99,255,0.15);color:#6c63ff;border:1px solid rgba(108,99,255,0.35);border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;margin-right:6px;">Groq LLM</span>'
    '<span style="background:rgba(0,212,170,0.15);color:#00d4aa;border:1px solid rgba(0,212,170,0.35);border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;margin-right:6px;">LangChain</span>'
    '<span style="background:rgba(255,215,0,0.15);color:#ffd700;border:1px solid rgba(255,215,0,0.35);border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;margin-right:6px;">ChromaDB RAG</span>'
    '<span style="background:rgba(255,107,107,0.15);color:#ff6b6b;border:1px solid rgba(255,107,107,0.35);border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;">Self-Healing</span>',
    unsafe_allow_html=True,
)
st.divider()

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
        c1, c2 = st.columns(2)
        c1.metric("Rows",    f"{meta['rows']:,}")
        c2.metric("Columns", meta["cols"])
        st.markdown(
            f'<div style="background:#131929;border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:12px;padding:0.8rem 1rem;margin-top:0.4rem;">'
            f'<span style="color:#8892a4;font-size:0.8rem;">File Size</span><br/>'
            f'<span style="color:#00d4aa;font-size:1.3rem;font-weight:700;">{meta["size_kb"]} KB</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
            _colors = [
                ("#6c63ff", "rgba(108,99,255,0.12)", "rgba(108,99,255,0.35)"),
                ("#00d4aa", "rgba(0,212,170,0.10)",  "rgba(0,212,170,0.35)"),
                ("#ffd700", "rgba(255,215,0,0.10)",  "rgba(255,215,0,0.35)"),
                ("#ff6b6b", "rgba(255,107,107,0.10)","rgba(255,107,107,0.35)"),
                ("#a78bfa", "rgba(167,139,250,0.10)","rgba(167,139,250,0.35)"),
            ]
            for _gi, _gq in enumerate(st.session_state.dynamic_questions):
                _tc, _bg, _bd = _colors[_gi % len(_colors)]
                if st.button(
                    f"{'🔹' if _gi%5==0 else '🔸' if _gi%5==1 else '💛' if _gi%5==2 else '❤️' if _gi%5==3 else '💜'} {_gq}",
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
                                f'<div style="background:#131929;border:1px solid rgba(255,255,255,0.06);'
                                f'border-radius:10px;padding:1rem;font-family:JetBrains Mono,monospace;'
                                f'font-size:0.85rem;color:#e8eaf0;white-space:pre-wrap;overflow-x:auto;">'
                                + _raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                + "</div>",
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
            '<div style="background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(0,212,170,0.04));'
            'border:1px solid rgba(108,99,255,0.2);border-radius:16px;padding:2rem;text-align:center;margin-top:1rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.5rem;">🤖</div>'
            '<h3 style="color:#e8eaf0;margin:0 0 0.5rem;">Ready to Analyse</h3>'
            '<p style="color:#8892a4;margin:0 0 1.5rem;font-size:0.95rem;">'
            'Upload a dataset, type your question, then click <strong style="color:#6c63ff;">🚀 Analyse</strong></p>'
            '<div style="display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;">'
            '<div style="background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.25);border-radius:12px;padding:0.8rem 1.2rem;min-width:120px;">'
            '<div style="font-size:1.4rem;">📂</div><div style="color:#a78bfa;font-size:0.82rem;font-weight:600;margin-top:4px;">Upload Data</div></div>'
            '<div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.25);border-radius:12px;padding:0.8rem 1.2rem;min-width:120px;">'
            '<div style="font-size:1.4rem;">💬</div><div style="color:#00d4aa;font-size:0.82rem;font-weight:600;margin-top:4px;">Ask Question</div></div>'
            '<div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.25);border-radius:12px;padding:0.8rem 1.2rem;min-width:120px;">'
            '<div style="font-size:1.4rem;">⚡</div><div style="color:#ffd700;font-size:0.82rem;font-weight:600;margin-top:4px;">AI Analyses</div></div>'
            '<div style="background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.25);border-radius:12px;padding:0.8rem 1.2rem;min-width:120px;">'
            '<div style="font-size:1.4rem;">📈</div><div style="color:#ff6b6b;font-size:0.82rem;font-weight:600;margin-top:4px;">Get Insights</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

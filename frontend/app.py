"""AI Analytics Platform – Tableau-style dashboards with AI-powered Q&A."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd

from backend.data_loader import load_csv_from_file, DataLoadError
from backend.data_profiler import profile_dataset, detect_anomalies, compute_feature_importance
from backend.insights_engine import generate_insights
from backend.narrative_engine import (
    generate_key_insights, generate_performance_drivers,
    generate_weak_segments, generate_data_explanation,
    generate_smart_suggestions,
)
from backend.demo_dataset import generate_demo_dataset
from frontend.dashboard import (
    render_dataset_overview, render_column_profiles,
    render_numeric_kpis, render_categorical_kpis, render_anomaly_summary,
)
from frontend.charts_ui import render_auto_charts, render_custom_chart_builder
from frontend.chat_ui import render_chat_interface
from frontend.tableau_dashboard import render_ai_dashboard, render_auto_dashboard
from ai_agents.copilot_agent import copilot_chat
from ai.sql_generator import generate_sql
from utils.column_classifier import get_measure_columns, get_dimension_columns, classify_all_columns
from utils.file_utils import file_hash
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI-Generated Dashboards | AI Analytics Platform", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f1117; --bg-card: #1a1d29; --bg-card-hover: #22263a;
    --border: #2d3148; --accent: #6366f1; --accent-light: #818cf8;
    --accent-glow: rgba(99,102,241,0.15); --success: #22c55e;
    --warning: #f59e0b; --danger: #ef4444;
    --text-primary: #f1f5f9; --text-secondary: #94a3b8; --text-muted: #64748b;
    --radius: 12px; --radius-sm: 8px;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .main .block-container, [data-testid="stAppViewBlockContainer"] {
    background-color: #0f1117 !important;
    color: #f1f5f9 !important;
}
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1500px !important; }
header[data-testid="stHeader"] { background: #0f1117 !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #111528, #0d0f1a) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem 1.25rem;
}
[data-testid="stMetric"]:hover { border-color: var(--accent); box-shadow: 0 0 20px var(--accent-glow); }
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: .8rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .05em !important; }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: var(--radius); padding: 6px; gap: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: var(--radius-sm); padding: .55rem 1.25rem; font-weight: 600; font-size: .85rem; color: var(--text-secondary); }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.glass-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1rem; }
.glass-card:hover { border-color: var(--accent); box-shadow: 0 0 24px var(--accent-glow); }
.stButton > button { background: var(--accent) !important; color: #fff !important; border: none !important; border-radius: var(--radius-sm) !important; font-weight: 600 !important; padding: .5rem 1.5rem !important; }
.stButton > button:hover { background: var(--accent-light) !important; box-shadow: 0 4px 16px var(--accent-glow) !important; }

/* ── GLOBAL TEXT VISIBILITY ── */
p, span, li, td, th, div, label, .stMarkdown, [data-testid="stMarkdownContainer"],
[data-testid="stText"], .element-container {
    color: #f1f5f9 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: .5rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}

/* DataFrames / Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
.dvn-scroller {
    background: #1a1d29 !important;
}
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td,
[data-testid="stTable"] th,
[data-testid="stTable"] td,
.gdg-cell span,
.dvn-cell-content {
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}
.glideDataEditor [role="gridcell"],
.glideDataEditor [role="columnheader"] {
    color: #f1f5f9 !important;
}

/* Chat input */
.stChatInput textarea,
.stChatInput input {
    color: #f1f5f9 !important;
    caret-color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}

/* Text inputs, text areas, select boxes */
.stTextInput input,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] input,
.stMultiSelect [data-baseweb="select"] input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: #f1f5f9 !important;
    caret-color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}

/* Placeholders */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stChatInput textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

/* Dropdown selected values */
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}

[data-baseweb="tag"] { background: rgba(99,102,241,.2) !important; color: #f1f5f9 !important; }
[data-baseweb="tag"] span { color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; }

.stTextInput > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: #f1f5f9 !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stFileUploader label {
    color: var(--text-secondary) !important;
}

/* Dropdowns */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background: #1a1d29 !important;
    border: 1px solid #2d3148 !important;
}
[data-baseweb="menu"] li,
[data-baseweb="popover"] li {
    color: #f1f5f9 !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover {
    background: rgba(99,102,241,.15) !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #94a3b8 !important;
}

/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

/* Expander */
[data-testid="stExpander"] summary span {
    color: #f1f5f9 !important;
}

/* Toggle label */
.stToggle label span {
    color: #f1f5f9 !important;
}

.hero { text-align: center; padding: 5rem 2rem 4rem; }
.hero h1 { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-light), #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .8rem; }
.hero p { color: var(--text-secondary); font-size: 1.15rem; max-width: 660px; margin: 0 auto 2.5rem; line-height: 1.7; }

.stat-grid { display: flex; justify-content: center; gap: 3rem; margin-top: 2rem; }
.stat-item { text-align: center; }
.stat-item .stat-num { font-size: 1.6rem; font-weight: 800; color: var(--text-primary); }
.stat-item .stat-label { font-size: .78rem; color: var(--text-muted); margin-top: 2px; }

.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; max-width: 1100px; margin: 2rem auto 0; }
.feature-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; text-align: left; transition: all .2s; }
.feature-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feature-card h3 { font-size: .95rem; font-weight: 700; color: var(--text-primary); margin: .5rem 0 .3rem; }
.feature-card p { font-size: .82rem; color: var(--text-muted); margin: 0; line-height: 1.5; }

.insight-section { margin-bottom: 1.5rem; }
.insight-section-title { display: flex; align-items: center; gap: .5rem; font-size: 1rem; font-weight: 700; color: var(--text-primary); margin-bottom: .75rem; }
.insight-item { display: flex; align-items: flex-start; gap: .6rem; padding: .5rem 0; font-size: .9rem; color: var(--text-secondary); line-height: 1.6; }
.insight-icon { flex-shrink: 0; margin-top: 2px; }

.report-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.75rem; }
.report-section { margin-bottom: 1.5rem; }
.report-section-title { display: flex; align-items: center; gap: .5rem; font-size: .95rem; font-weight: 700; color: var(--text-primary); margin-bottom: .5rem; }
.report-item { padding: .3rem 0 .3rem 1.2rem; font-size: .88rem; color: var(--text-secondary); line-height: 1.6; position: relative; }
.report-item::before { content: "·"; position: absolute; left: 0; font-weight: 700; color: var(--text-muted); }

.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600; }
.badge-success { background: rgba(34,197,94,.12); color: var(--success); }
.badge-demo { background: rgba(99,102,241,.15); color: var(--accent-light); }

.qa-header { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }
.qa-title { font-size: 1.4rem; font-weight: 800; color: #f1f5f9; }
.qa-subtitle { font-size: .82rem; color: #64748b; }
.qa-suggestion { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: .75rem 1rem; cursor: pointer; transition: all .2s; }
.qa-suggestion:hover { border-color: var(--accent); background: var(--bg-card-hover); }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""<div style="padding:1.2rem .5rem .5rem; text-align:center;">
        <div style="font-size:2rem; margin-bottom:.25rem;">🤖</div>
        <div style="font-size:1.15rem; font-weight:800; color:#f1f5f9;">AI Analytics Platform</div>
        <div style="font-size:.72rem; font-weight:600; color:#22c55e; text-transform:uppercase; letter-spacing:.1em; margin-top:2px;">AI-Generated Dashboards</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], help="CSV or Excel (max 100 MB)")
    if "df" in st.session_state:
        p = st.session_state.profile
        measures = get_measure_columns(st.session_state.df)
        dims = get_dimension_columns(st.session_state.df)
        dt_cols = st.session_state.df.select_dtypes(include="datetime").columns.tolist()
        mode_label = "Demo" if st.session_state.get("is_demo") else "Live"
        badge_css = "badge-demo" if st.session_state.get("is_demo") else "badge-success"
        st.markdown("---")
        st.markdown(f"""<div class="glass-card" style="padding:1rem;">
            <span class="badge {badge_css}">{mode_label}</span>
            <div style="font-size:.92rem; font-weight:600; color:#f1f5f9; margin-top:.5rem;">
                {p.n_rows:,} rows  x  {p.n_cols} cols</div>
            <div style="font-size:.78rem; color:#64748b; margin-top:2px;">
                {len(measures)} measures  ·  {len(dims)} dimensions{f'  ·  {len(dt_cols)} dates' if dt_cols else ''}</div>
            <div style="font-size:.78rem; color:#64748b;">{p.memory_usage_mb:.1f} MB</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Clear Dataset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.markdown("---")
    st.caption("v3.0 · Tableau-Style · AI-Powered")

# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=5)
def _load_and_profile(file_bytes: bytes, filename: str):
    import io
    buf = io.BytesIO(file_bytes)
    cache_key = file_hash(buf)
    df = load_csv_from_file(buf, filename, cache_key=cache_key)
    profile = profile_dataset(df)
    return df, profile, cache_key


def _load_demo():
    df = generate_demo_dataset()
    profile = profile_dataset(df)
    st.session_state.df = df
    st.session_state.profile = profile
    st.session_state.cache_key = "demo"
    st.session_state.table_name = "demo_sales"
    st.session_state.is_demo = True


if uploaded_file is not None and "df" not in st.session_state:
    try:
        with st.spinner("Analyzing your dataset..."):
            raw = uploaded_file.read()
            df, profile, cache_key = _load_and_profile(raw, uploaded_file.name)
            st.session_state.df = df
            st.session_state.profile = profile
            st.session_state.cache_key = cache_key
            import os
            clean_name = os.path.splitext(uploaded_file.name)[0]
            clean_name = "".join(c if c.isalnum() or c == "_" else "_" for c in clean_name)
            st.session_state.table_name = clean_name or "dataset"
            st.session_state.is_demo = False
            st.rerun()
    except DataLoadError as exc:
        st.error(f"Upload failed: {exc}")
    except Exception as exc:
        st.error(f"Error: {exc}")

# ── HERO / WELCOME ────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.markdown("""<div class="hero">
        <div style="display:inline-flex; padding:6px 16px; border-radius:999px; background:var(--accent-glow);
                    font-size:.82rem; font-weight:600; color:var(--accent-light); margin-bottom:1.5rem;">
            AI-Native Analytics Platform
        </div>
        <h1>AI-Generated Dashboards<br>in Seconds</h1>
        <p>Upload any dataset — the AI analyzes your data, designs the perfect dashboard layout,
           picks the right charts, generates KPIs, and builds an interactive analytics experience.
           Ask questions in plain English and get instant answers with visualizations.</p>
    </div>""", unsafe_allow_html=True)

    bc1, bc2, bc3 = st.columns([1, 2, 1])
    with bc2:
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("Try Demo Dataset", use_container_width=True, type="primary"):
                _load_demo()
                st.rerun()
        with dc2:
            st.markdown("<div style='padding:.5rem; text-align:center; font-size:.9rem; color:#94a3b8;'>Upload CSV / Excel in sidebar</div>", unsafe_allow_html=True)

    st.markdown("""<div class="stat-grid">
        <div class="stat-item"><div class="stat-num">AI</div><div class="stat-label">Designed Dashboards</div></div>
        <div class="stat-item"><div class="stat-num">Llama 3.1</div><div class="stat-label">Powers Everything</div></div>
        <div class="stat-item"><div class="stat-num">8+</div><div class="stat-label">Chart Types</div></div>
        <div class="stat-item"><div class="stat-num">NL</div><div class="stat-label">Natural Language Query</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="feature-grid">
        <div class="feature-card"><div style="font-size:1.5rem">🤖</div><h3>AI-Designed Dashboards</h3>
            <p>The AI analyzes your data structure and designs the optimal dashboard — picks KPIs, chart types, and layout.</p></div>
        <div class="feature-card"><div style="font-size:1.5rem">🎛</div><h3>Interactive Filters</h3>
            <p>Click-to-filter by any dimension. All charts and KPIs update instantly.</p></div>
        <div class="feature-card"><div style="font-size:1.5rem">💬</div><h3>Ask Anything</h3>
            <p>Ask any question in plain English — AI generates SQL, runs code, and visualizes the answer.</p></div>
        <div class="feature-card"><div style="font-size:1.5rem">💡</div><h3>AI Insights</h3>
            <p>AI explains your data — performance drivers, trends, anomalies, and key findings.</p></div>
        <div class="feature-card"><div style="font-size:1.5rem">🔍</div><h3>Auto SQL Generator</h3>
            <p>Natural language to SQL + Pandas with instant execution on your dataset.</p></div>
        <div class="feature-card"><div style="font-size:1.5rem">🛡</div><h3>Data Governance</h3>
            <p>Quality scoring, validation rules, anomaly detection, and column health reports.</p></div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# TAB RENDER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _show_html_table(data: pd.DataFrame, max_rows: int = 50) -> None:
    """Render a DataFrame as a styled HTML table (bypasses canvas rendering issues)."""
    html = data.head(max_rows).to_html(index=False, classes="ht", border=0)
    st.markdown(f"""<div style="overflow-x:auto; max-height:450px; overflow-y:auto;
        border-radius:8px; border:1px solid #2d3148;">
        <style>
        .ht {{ width:100%; border-collapse:collapse; font-size:.82rem; font-family:'Inter',monospace; }}
        .ht th {{ background:#1e2235; color:#94a3b8; font-weight:600; padding:8px 12px; text-align:left;
                  position:sticky; top:0; border-bottom:2px solid #2d3148; font-size:.75rem;
                  text-transform:uppercase; letter-spacing:.03em; }}
        .ht td {{ padding:6px 12px; color:#e2e8f0; border-bottom:1px solid #1e2235; }}
        .ht tr:hover td {{ background:rgba(99,102,241,0.06); }}
        </style>{html}
    </div>""", unsafe_allow_html=True)


def _render_full_qa(df: pd.DataFrame) -> None:
    """Full-page AI Q&A experience — ask any question about the data."""
    import plotly.express as px

    st.markdown("""<div class="qa-header">
        <span style="font-size:1.6rem;">🤖</span>
        <div>
            <div class="qa-title">Ask Anything About Your Data</div>
            <div class="qa-subtitle">Powered by Llama 3.1 — get answers, tables, and auto-generated charts</div>
        </div>
    </div>""", unsafe_allow_html=True)

    measures = get_measure_columns(df)
    dims = get_dimension_columns(df)

    example_questions = [
        f"What is the total and average {measures[0]}?" if measures else "Summarize this dataset",
        f"Show {measures[0]} by {dims[0]}" if measures and dims else "What are the top categories?",
        f"Which {dims[0]} has the highest {measures[0]}?" if measures and dims else "Which category performs best?",
        f"Compare {measures[0]} across {dims[1] if len(dims) > 1 else dims[0]}" if measures and dims else "Show comparisons",
        f"Show the distribution of {measures[0]}" if measures else "Describe the data",
        f"What are the top 10 rows by {measures[0]}?" if measures else "Show top rows",
    ]

    if "ask_chat" not in st.session_state:
        st.session_state.ask_chat = []

    if not st.session_state.ask_chat:
        st.markdown("""<div style="text-align:center; padding:2rem 0 1rem;">
            <div style="font-size:1.5rem; margin-bottom:.5rem;">💬</div>
            <div style="font-size:1.05rem; font-weight:600; color:#f1f5f9; margin-bottom:.25rem;">What would you like to know?</div>
            <div style="font-size:.82rem; color:#64748b; margin-bottom:1.5rem;">Ask any question in plain English and get instant answers with data & charts</div>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, q in enumerate(example_questions[:6]):
            with cols[i % 3]:
                if st.button(q, key=f"eq_{i}", use_container_width=True):
                    _process_ask_question(df, q)
                    st.rerun()

    for msg in st.session_state.ask_chat:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("data") is not None:
                _show_html_table(msg["data"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    user_q = st.chat_input("Ask any question about your data...", key="ask_input")
    if user_q:
        _process_ask_question(df, user_q)
        st.rerun()


def _process_ask_question(df: pd.DataFrame, question: str) -> None:
    """Process Q&A question, generate answer + optional chart."""
    import plotly.express as px
    import plotly.graph_objects as go

    st.session_state.ask_chat.append({"role": "user", "content": question})

    from backend.ai_chat_engine import ask_question
    resp = ask_question(df, question)

    entry = {"role": "assistant", "content": resp.answer, "data": None, "chart": None}

    if resp.data is not None:
        result_df = resp.data.head(25)
        entry["data"] = result_df

        num_cols = result_df.select_dtypes(include="number").columns.tolist()
        other_cols = [c for c in result_df.columns if c not in num_cols]

        if other_cols and num_cols and len(result_df) >= 2:
            try:
                _LAYOUT = dict(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(26,29,41,0.8)",
                    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
                    margin=dict(l=40, r=20, t=40, b=40),
                )

                x_col = other_cols[0]
                y_col = num_cols[0]

                if len(result_df) <= 20:
                    fig = px.bar(result_df, x=x_col, y=y_col,
                                 color_discrete_sequence=["#6366f1"],
                                 text=result_df[y_col].apply(lambda v: f"{v:,.0f}" if abs(v) >= 1 else f"{v:.2f}"))
                    fig.update_traces(textposition="outside", textfont_size=10)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result_df[x_col], y=result_df[y_col],
                        mode="lines+markers", line=dict(color="#6366f1", width=2),
                        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
                    ))

                fig.update_layout(**_LAYOUT, height=350,
                                  xaxis_title=x_col, yaxis_title=y_col,
                                  xaxis_tickangle=-30)
                entry["chart"] = fig
            except Exception:
                pass

    st.session_state.ask_chat.append(entry)


def _render_intelligence_tab(df: pd.DataFrame, profile) -> None:
    """Render the intelligence/insights tab."""
    show_report = st.toggle("Show Full Data Explanation Report", value=False)

    if show_report:
        report = generate_data_explanation(df)
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("""<div style="display:flex; align-items:center; gap:.6rem; margin-bottom:1.25rem;">
            <span style="font-size:1.3rem;">📄</span>
            <div><div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">Data Explanation Report</div>
            <div style="font-size:.78rem; color:#64748b;">A comprehensive narrative breakdown of your dataset</div></div>
        </div>""", unsafe_allow_html=True)

        for section_key, icon, title in [
            ("dataset_overview", "📋", "Dataset Overview"),
            ("key_insights", "📊", "Key Insights"),
            ("performance_drivers", "🚀", "Performance Drivers"),
            ("weak_segments", "⚠️", "Weak Segments"),
            ("correlations", "🔗", "Correlations"),
            ("recommendations", "💡", "Recommendations"),
        ]:
            items = report.get(section_key, [])
            if items:
                st.markdown(f'<div class="report-section"><div class="report-section-title">{icon} {title}</div>', unsafe_allow_html=True)
                for item in items:
                    text = item["text"] if isinstance(item, dict) else item
                    st.markdown(f'<div class="report-item">{text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

    ic1, ic2 = st.columns([1, 1], gap="large")

    with ic1:
        st.markdown("""<div class="glass-card">
            <div style="display:flex; align-items:center; gap:.5rem; margin-bottom:1rem;">
                <span style="font-size:1.2rem;">💡</span>
                <div><div style="font-size:1rem; font-weight:700; color:#f1f5f9;">AI Insights</div>
                <div style="font-size:.75rem; color:#64748b;">Auto-generated from your data</div></div>
            </div>""", unsafe_allow_html=True)

        key_insights = generate_key_insights(df)
        if key_insights:
            st.markdown('<div class="insight-section"><div class="insight-section-title">📊 Key Insights</div>', unsafe_allow_html=True)
            for ins in key_insights:
                st.markdown(f'<div class="insight-item"><span class="insight-icon">{ins["icon"]}</span> {ins["text"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        drivers = generate_performance_drivers(df)
        if drivers:
            st.markdown('<div class="insight-section"><div class="insight-section-title">🚀 Performance Drivers</div>', unsafe_allow_html=True)
            for d in drivers:
                st.markdown(f'<div class="insight-item"><span class="insight-icon">●</span> {d["text"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        weak = generate_weak_segments(df)
        if weak:
            st.markdown('<div class="insight-section"><div class="insight-section-title">⚠️ Weak Segments</div>', unsafe_allow_html=True)
            for w in weak:
                st.markdown(f'<div class="insight-item"><span class="insight-icon">●</span> {w["text"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with ic2:
        st.markdown("""<div class="glass-card">
            <div style="display:flex; align-items:center; gap:.5rem; margin-bottom:1rem;">
                <span style="font-size:1.2rem;">💬</span>
                <div><div style="font-size:1rem; font-weight:700; color:#f1f5f9;">Quick Data Assistant</div>
                <div style="font-size:.75rem; color:#64748b;">Ask questions about your dataset</div></div>
            </div>""", unsafe_allow_html=True)

        suggestions = generate_smart_suggestions(df)
        if "quick_chat_history" not in st.session_state:
            st.session_state.quick_chat_history = []

        if not st.session_state.quick_chat_history:
            scols = st.columns(min(len(suggestions), 3))
            for i, col in enumerate(scols):
                if i < len(suggestions):
                    with col:
                        if st.button(f"✦ {suggestions[i]}", key=f"qs_{i}", use_container_width=True):
                            st.session_state.quick_chat_history.append({"role": "user", "content": suggestions[i]})
                            from backend.ai_chat_engine import ask_question
                            resp = ask_question(df, suggestions[i])
                            st.session_state.quick_chat_history.append({"role": "assistant", "content": resp.answer})
                            st.rerun()

        for msg in st.session_state.quick_chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📊"):
                st.markdown(msg["content"])

        qi = st.chat_input("Ask a question about your data...", key="quick_chat_input")
        if qi:
            st.session_state.quick_chat_history.append({"role": "user", "content": qi})
            with st.spinner("Analyzing..."):
                from backend.ai_chat_engine import ask_question
                resp = ask_question(df, qi)
            st.session_state.quick_chat_history.append({"role": "assistant", "content": resp.answer})
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("KPIs")
    render_numeric_kpis(profile)
    render_categorical_kpis(profile)

    numeric_cols = get_measure_columns(df)
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.subheader("Feature Importance")
        target = st.selectbox("Target column", numeric_cols, key="fi_target")
        if target:
            importance = compute_feature_importance(df, target)
            if importance:
                imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
                _show_html_table(imp_df)


def _render_profile_tab(df: pd.DataFrame, profile) -> None:
    """Render the data profile tab."""
    render_dataset_overview(df, profile)

    col_class = classify_all_columns(df)
    class_df = pd.DataFrame([{"Column": c, "Role": r.replace("_", " ").title()} for c, r in col_class.items()])
    with st.expander("Column Classification", expanded=False):
        _show_html_table(class_df)

    render_column_profiles(profile)
    anomalies = detect_anomalies(df)
    render_anomaly_summary(anomalies)

    st.markdown("---")
    _, dl_col = st.columns([3, 1])
    with dl_col:
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="dataset.csv", mime="text/csv", use_container_width=True)


def _render_copilot_tab(df: pd.DataFrame) -> None:
    """Render the AI Copilot tab."""
    st.markdown("""<div style="display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem;">
        <span style="font-size:1.3rem;">🤖</span>
        <div><div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">AI Data Copilot</div>
        <div style="font-size:.78rem; color:#64748b;">Your assistant for KPIs, dashboards, transformations, and anomaly analysis</div></div>
    </div>""", unsafe_allow_html=True)

    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    for msg in st.session_state.copilot_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    if not st.session_state.copilot_history:
        prompts = ["Explain this dataset", "Suggest KPIs", "Recommend dashboards", "Suggest transformations", "Find anomalies"]
        cols = st.columns(len(prompts))
        for i, col in enumerate(cols):
            with col:
                if st.button(prompts[i], key=f"cop_{i}", use_container_width=True):
                    st.session_state.copilot_history.append({"role": "user", "content": prompts[i]})
                    resp = copilot_chat(prompts[i], df, st.session_state.copilot_history)
                    st.session_state.copilot_history.append({"role": "assistant", "content": resp})
                    st.rerun()

    ci = st.chat_input("Ask the Copilot anything...", key="copilot_input")
    if ci:
        st.session_state.copilot_history.append({"role": "user", "content": ci})
        with st.chat_message("user", avatar="👤"):
            st.markdown(ci)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                resp = copilot_chat(ci, df, st.session_state.copilot_history)
            st.markdown(resp)
        st.session_state.copilot_history.append({"role": "assistant", "content": resp})


def _render_sql_tab(df: pd.DataFrame) -> None:
    """Render the SQL generator tab."""
    st.markdown("""<div style="display:flex; align-items:center; gap:.5rem; margin-bottom:1rem;">
        <span style="font-size:1.3rem;">🔍</span>
        <div><div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">Auto SQL Generator</div>
        <div style="font-size:.78rem; color:#64748b;">Natural language to SQL + Pandas with instant execution</div></div>
    </div>""", unsafe_allow_html=True)

    sql_q = st.text_input("Enter query", placeholder="Show top 10 customers by revenue", key="sql_input")
    if sql_q:
        with st.spinner("Generating..."):
            tbl_name = st.session_state.get("table_name", "dataset")
            sql_res = generate_sql(sql_q, df, tbl_name)
        if sql_res["sql"]:
            st.markdown("**Generated SQL:**")
            st.code(sql_res["sql"], language="sql")
        if sql_res["pandas_code"]:
            st.markdown("**Equivalent Pandas:**")
            st.code(sql_res["pandas_code"], language="python")
        if sql_res.get("result") is not None:
            st.markdown("**Result:**")
            if isinstance(sql_res["result"], pd.DataFrame):
                _show_html_table(sql_res["result"])
            elif isinstance(sql_res["result"], pd.Series):
                _show_html_table(sql_res["result"].to_frame())
            else:
                st.markdown(f"**{sql_res['result']}**")
        if sql_res.get("error"):
            st.error(sql_res["error"])
    else:
        st.markdown("""<div style="padding:2rem; text-align:center; color:#64748b;">
            <div style="font-size:2rem; margin-bottom:.5rem;">🔍</div>
            <div style="font-size:.85rem; margin-top:.5rem; color:#94a3b8;">
                "Show top customers by revenue" · "Average order value by region" · "Monthly sales trend"
            </div></div>""", unsafe_allow_html=True)


def _render_governance_tab(df: pd.DataFrame) -> None:
    """Render Data Governance with quality scoring, validation rules, and schema."""
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <span style="font-size:1.4rem;">🛡</span>
        <div><div style="font-size:1.2rem;font-weight:800;color:#f1f5f9;">Data Governance</div>
        <div style="font-size:.78rem;color:#64748b;">Quality scoring, validation rules, and data health</div></div>
    </div>""", unsafe_allow_html=True)

    g1, g2, g3 = st.tabs(["📊 Quality Score", "✅ Validation Rules", "🏥 Column Health"])

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing = int(df.isnull().sum().sum())
    completeness = (1 - missing / total_cells) * 100 if total_cells else 100
    duplicates = int(df.duplicated().sum())
    uniqueness = (1 - duplicates / n_rows) * 100 if n_rows else 100

    consistency_issues = 0
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().unique()
        lower_vals = [str(v).lower().strip() for v in vals]
        if len(set(lower_vals)) < len(vals):
            consistency_issues += 1
    consistency = max(0, 100 - consistency_issues * 10)

    validity = 100.0
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
        if outliers > n_rows * 0.05:
            validity -= 5

    overall = round(min(completeness * 0.35 + uniqueness * 0.25 + consistency * 0.2 + max(validity, 0) * 0.2, 100), 1)

    with g1:
        _sc = lambda s: "#22c55e" if s >= 80 else ("#f59e0b" if s >= 60 else "#ef4444")
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        with qc1:
            c = _sc(overall)
            st.markdown(f"""<div style="background:#1a1d29;border:2px solid {c};border-radius:12px;padding:1.25rem;text-align:center;">
                <div style="font-size:2rem;font-weight:800;color:{c};">{overall}%</div>
                <div style="font-size:.75rem;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-top:4px;">Overall</div>
            </div>""", unsafe_allow_html=True)
        for cw, val, lbl in [(qc2, completeness, "Completeness"), (qc3, uniqueness, "Uniqueness"),
                              (qc4, consistency, "Consistency"), (qc5, max(validity, 0), "Validity")]:
            with cw:
                st.metric(f"Gov {lbl}", f"{val:.1f}%")

        st.markdown("---")
        rc1, rc2 = st.columns(2)
        with rc1:
            labels = ["Completeness", "Uniqueness", "Consistency", "Validity"]
            values = [round(completeness, 1), round(uniqueness, 1), round(min(consistency, 100), 1), round(max(validity, 0), 1)]
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]], theta=labels + [labels[0]],
                fill="toself", fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=8, color="#818cf8"),
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(26,29,41,0.8)", font=dict(family="Inter", color="#94a3b8", size=11),
                              margin=dict(l=40, r=20, t=40, b=40), title="Quality Radar", height=350,
                              polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True,
                                                          tickfont=dict(size=9, color="#64748b")),
                                         angularaxis=dict(tickfont=dict(size=11, color="#f1f5f9")),
                                         bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True, key="gov_radar")

        with rc2:
            col_missing = df.isnull().sum().reset_index()
            col_missing.columns = ["Column", "Missing"]
            col_missing = col_missing[col_missing["Missing"] > 0].sort_values("Missing", ascending=True)
            if len(col_missing) > 0:
                fig2 = px.bar(col_missing, y="Column", x="Missing", orientation="h",
                              color_discrete_sequence=["#ef4444"],
                              text=col_missing["Missing"].apply(lambda x: f"{x:,}"))
                fig2.update_traces(textposition="outside", textfont_size=10)
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(26,29,41,0.8)", font=dict(family="Inter", color="#94a3b8", size=11),
                                   margin=dict(l=40, r=20, t=40, b=40), title="Missing Values by Column", height=350)
                st.plotly_chart(fig2, use_container_width=True, key="gov_missing")
            else:
                st.success("No missing values in any column.")

        st.markdown(f"""<div style="display:flex;gap:2rem;justify-content:center;padding:1rem;
                        background:#1a1d29;border:1px solid #2d3148;border-radius:12px;">
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{missing:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Missing Cells</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{duplicates:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Duplicate Rows</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{total_cells:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Total Cells</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{n_cols}</div>
                <div style="font-size:.75rem;color:#64748b;">Columns</div></div>
        </div>""", unsafe_allow_html=True)

    with g2:
        rules = []
        for col in df.columns:
            miss_pct = df[col].isnull().mean() * 100
            status = "FAIL" if miss_pct > 20 else ("WARN" if miss_pct > 5 else "PASS")
            rules.append({"Rule": f"Null check: {col}", "Status": status,
                          "Detail": f"{miss_pct:.1f}% missing" if miss_pct > 0 else "No missing values",
                          "Severity": "High" if miss_pct > 20 else ("Medium" if miss_pct > 5 else "Low")})

        dups = df.duplicated().sum()
        rules.append({"Rule": "Duplicate row check",
                      "Status": "FAIL" if dups > n_rows * 0.05 else ("WARN" if dups > 0 else "PASS"),
                      "Detail": f"{dups:,} duplicates ({dups/n_rows*100:.1f}%)" if dups else "No duplicates",
                      "Severity": "High" if dups > n_rows * 0.05 else "Low"})

        for col in df.select_dtypes(include="number").columns:
            clean = df[col].dropna()
            if len(clean) < 10:
                continue
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            outliers = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
            if outliers > 0:
                rules.append({"Rule": f"Outlier check: {col}", "Status": "WARN" if outliers > n_rows * 0.02 else "PASS",
                              "Detail": f"{outliers:,} outliers (IQR method)", "Severity": "Medium" if outliers > n_rows * 0.02 else "Low"})

        rules_df = pd.DataFrame(rules)
        pass_c = sum(1 for r in rules if r["Status"] == "PASS")
        warn_c = sum(1 for r in rules if r["Status"] == "WARN")
        fail_c = sum(1 for r in rules if r["Status"] == "FAIL")

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.metric("Validation PASS", pass_c, delta=f"{pass_c/len(rules)*100:.0f}%")
        with vc2:
            st.metric("Validation WARN", warn_c)
        with vc3:
            st.metric("Validation FAIL", fail_c)

        status_filter = st.multiselect("Filter by status", ["PASS", "WARN", "FAIL"],
                                        default=["WARN", "FAIL"], key="tab_gov_filter")
        filtered = rules_df[rules_df["Status"].isin(status_filter)] if status_filter else rules_df
        _show_html_table(filtered)

    with g3:
        col_health = []
        for col in df.columns:
            s = df[col]
            miss_pct = s.isnull().mean() * 100
            uniq_pct = s.nunique() / len(s) * 100 if len(s) > 0 else 0
            health = "Healthy" if miss_pct < 5 else ("Warning" if miss_pct < 20 else "Critical")
            col_health.append({"Column": col, "Type": str(s.dtype), "Missing %": round(miss_pct, 1),
                               "Unique %": round(uniq_pct, 1), "Health": health})

        hc1, hc2, hc3 = st.columns(3)
        healthy = sum(1 for c in col_health if c["Health"] == "Healthy")
        warning = sum(1 for c in col_health if c["Health"] == "Warning")
        critical = sum(1 for c in col_health if c["Health"] == "Critical")
        with hc1:
            st.metric("Healthy Cols", healthy)
        with hc2:
            st.metric("Warning Cols", warning)
        with hc3:
            st.metric("Critical Cols", critical)

        _show_html_table(pd.DataFrame(col_health))

        measures = get_measure_columns(df)
        if measures:
            st.markdown("---")
            st.markdown("**Distribution Health**")
            skew_data = []
            for m in measures[:12]:
                sk = df[m].skew()
                skew_data.append({"Measure": m, "Skewness": round(sk, 2),
                                  "Status": "Normal" if abs(sk) < 1 else ("Skewed" if abs(sk) < 2 else "Highly Skewed")})
            _show_html_table(pd.DataFrame(skew_data))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD — render based on selected style
# ═════════════════════════════════════════════════════════════════════════════

df: pd.DataFrame = st.session_state.df
profile = st.session_state.profile
cache_key = st.session_state.get("cache_key")

t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
    ["🤖 AI Dashboard", "📊 Auto Dashboard", "💬 Ask AI", "💡 Intelligence",
     "📋 Data Profile", "📈 Charts", "🔍 SQL", "🛡 Governance", "🤖 Copilot"]
)

with t1:
    render_ai_dashboard(df)
with t2:
    render_auto_dashboard(df)
with t3:
    _render_full_qa(df)
with t4:
    _render_intelligence_tab(df, profile)
with t5:
    _render_profile_tab(df, profile)
with t6:
    with st.spinner("Generating charts..."):
        render_auto_charts(df, profile)
    st.markdown("---")
    render_custom_chart_builder(df)
with t7:
    _render_sql_tab(df)
with t8:
    _render_governance_tab(df)
with t9:
    _render_copilot_tab(df)

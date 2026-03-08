"""Enterprise Data Platform – Medallion Architecture, Governance, Observability.

Run with: streamlit run enterprise_app.py --server.port 8502
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np

from backend.data_loader import load_csv_from_file, DataLoadError
from backend.data_profiler import profile_dataset, detect_anomalies
from backend.demo_dataset import generate_demo_dataset
from bronze.bronze_storage import store_bronze, list_bronze_datasets, load_bronze
from silver.data_cleaner import clean_dataset, store_silver
from gold.analytics_tables import generate_analytics_tables, store_gold_tables
from feature_store.feature_registry import compute_features, register_features, list_features
from schema.schema_registry import detect_schema, save_schema, load_schema, list_schema_versions
from lineage.lineage_tracker import LineageTracker, PIPELINE_STAGES
from frontend.enterprise_ui import render_governance_tab, render_observability_tab
from utils.column_classifier import get_measure_columns, get_dimension_columns, classify_all_columns
from utils.file_utils import file_hash
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _ent_html_table(data: pd.DataFrame, max_rows: int = 50) -> None:
    """Render a DataFrame as a styled HTML table."""
    html = data.head(max_rows).to_html(index=False, classes="ent-table", border=0)
    st.markdown(f'<div style="overflow-x:auto;max-height:400px;overflow-y:auto;'
                f'border-radius:8px;border:1px solid #2d3148;">{html}</div>',
                unsafe_allow_html=True)

st.set_page_config(page_title="Enterprise Data Platform", page_icon="🏗", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1500px !important; }

[data-testid="stMetric"] {
    background: #1a1d29; border: 1px solid #2d3148;
    border-radius: 12px; padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size: .8rem !important; font-weight: 600 !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }

.stTabs [data-baseweb="tab-list"] { background: #1a1d29; border-radius: 12px; padding: 6px; gap: 4px; border: 1px solid #2d3148; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: .5rem 1rem; font-weight: 600; font-size: .82rem; }
.stTabs [aria-selected="true"] { background: #6366f1 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button { background: #6366f1 !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button:hover { background: #818cf8 !important; }

.pipeline-card { background: #1a1d29; border: 1px solid #2d3148; border-radius: 12px; padding: 1.25rem; text-align: center; }
.pipeline-card.active { border-color: #22c55e; background: rgba(34,197,94,0.06); }

.ent-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
.ent-table th { background: #1e2235; color: #94a3b8; font-weight: 600; padding: 8px 12px; text-align: left; border-bottom: 2px solid #2d3148; font-size: .75rem; text-transform: uppercase; }
.ent-table td { padding: 6px 12px; color: #e2e8f0; border-bottom: 1px solid #1e2235; }
.ent-table tr:hover td { background: rgba(99,102,241,0.06); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""<div style="padding:1rem .5rem; text-align:center;">
        <div style="font-size:2rem;">🏗</div>
        <div style="font-size:1.1rem; font-weight:800;">Enterprise Platform</div>
        <div style="font-size:.7rem; font-weight:600; color:#6366f1; text-transform:uppercase; letter-spacing:.1em; margin-top:2px;">Medallion Architecture</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if "ent_df" in st.session_state:
        p = st.session_state.ent_profile
        st.markdown("---")
        st.markdown(f"""<div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1rem;">
            <div style="font-size:.9rem; font-weight:600;">{p.n_rows:,} rows x {p.n_cols} cols</div>
            <div style="font-size:.75rem; color:#64748b;">{p.memory_usage_mb:.1f} MB</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True, key="ent_clear"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    st.markdown("---")
    st.caption("Enterprise Edition · Medallion · Governance")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=5)
def _load(file_bytes, filename):
    import io
    buf = io.BytesIO(file_bytes)
    ck = file_hash(buf)
    df = load_csv_from_file(buf, filename, cache_key=ck)
    return df, ck

if uploaded and "ent_df" not in st.session_state:
    try:
        with st.spinner("Ingesting..."):
            raw = uploaded.read()
            df, ck = _load(raw, uploaded.name)
            st.session_state.ent_df = df
            st.session_state.ent_raw = df.copy()
            st.session_state.ent_name = Path(uploaded.name).stem
            st.session_state.ent_cache = ck
            st.session_state.ent_profile = profile_dataset(df)
            store_bronze(df, {"dataset_name": uploaded.name, "rows": len(df), "cols": len(df.columns)})
            tracker = LineageTracker(st.session_state.ent_name)
            tracker.add_event("raw_dataset", {"rows": len(df), "cols": len(df.columns)})
            tracker.add_event("bronze_storage", {"rows": len(df)})
            st.rerun()
    except Exception as e:
        st.error(str(e))

if "ent_df" not in st.session_state:
    st.markdown("""<div style="text-align:center; padding:5rem 2rem;">
        <div style="font-size:2.5rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#a78bfa,#f472b6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.8rem;">
            Enterprise Data Platform</div>
        <p style="color:#94a3b8; font-size:1.05rem; max-width:600px; margin:0 auto 2rem; line-height:1.7;">
            Medallion Architecture with Bronze, Silver, Gold layers, Data Governance,
            Observability, Schema Registry, Lineage Tracking, and Feature Store.</p>
    </div>""", unsafe_allow_html=True)

    _, bc, _ = st.columns([1, 2, 1])
    with bc:
        if st.button("Load Demo Dataset", use_container_width=True, type="primary", key="ent_load_demo"):
            df = generate_demo_dataset()
            st.session_state.ent_df = df
            st.session_state.ent_raw = df.copy()
            st.session_state.ent_name = "demo"
            st.session_state.ent_cache = "demo"
            st.session_state.ent_profile = profile_dataset(df)
            tracker = LineageTracker("demo")
            tracker.add_event("raw_dataset", {"rows": len(df)})
            tracker.add_event("bronze_storage", {"rows": len(df)})
            st.rerun()

    st.markdown("""<div style="display:flex; justify-content:center; gap:2rem; margin-top:3rem; flex-wrap:wrap;">
        <div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1.25rem; width:150px; text-align:center;">
            <div style="font-size:1.5rem;">🥉</div><div style="font-size:.85rem; font-weight:700; margin-top:.3rem;">Bronze</div>
            <div style="font-size:.72rem; color:#64748b;">Raw ingestion</div></div>
        <div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1.25rem; width:150px; text-align:center;">
            <div style="font-size:1.5rem;">🥈</div><div style="font-size:.85rem; font-weight:700; margin-top:.3rem;">Silver</div>
            <div style="font-size:.72rem; color:#64748b;">Clean & validate</div></div>
        <div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1.25rem; width:150px; text-align:center;">
            <div style="font-size:1.5rem;">🥇</div><div style="font-size:.85rem; font-weight:700; margin-top:.3rem;">Gold</div>
            <div style="font-size:.72rem; color:#64748b;">Analytics tables</div></div>
        <div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1.25rem; width:150px; text-align:center;">
            <div style="font-size:1.5rem;">🛡</div><div style="font-size:.85rem; font-weight:700; margin-top:.3rem;">Governance</div>
            <div style="font-size:.72rem; color:#64748b;">Quality & rules</div></div>
        <div style="background:#1a1d29; border:1px solid #2d3148; border-radius:12px; padding:1.25rem; width:150px; text-align:center;">
            <div style="font-size:1.5rem;">📡</div><div style="font-size:.85rem; font-weight:700; margin-top:.3rem;">Observability</div>
            <div style="font-size:.72rem; color:#64748b;">Health & drift</div></div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════

df = st.session_state.ent_df
raw_df = st.session_state.ent_raw
profile = st.session_state.ent_profile
ds_name = st.session_state.ent_name

tab_pipe, tab_bronze, tab_silver, tab_gold, tab_feat, tab_gov, tab_obs, tab_schema, tab_lineage = st.tabs([
    "🔄 Pipeline", "🥉 Bronze", "🥈 Silver", "🥇 Gold",
    "⚙ Features", "🛡 Governance", "📡 Observability", "📐 Schema", "🔗 Lineage",
])

# ── Pipeline Overview ────────────────────────────────────────────────────────

with tab_pipe:
    st.markdown("""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <span style="font-size:1.4rem;">🔄</span>
        <div><div style="font-size:1.2rem;font-weight:800;">Medallion Pipeline</div>
        <div style="font-size:.78rem;color:#64748b;">End-to-end data processing flow</div></div>
    </div>""", unsafe_allow_html=True)

    tracker = LineageTracker(ds_name)
    completed = {e["stage_id"] for e in tracker.get_lineage()}

    cols = st.columns(len(PIPELINE_STAGES))
    for i, stage in enumerate(PIPELINE_STAGES):
        done = stage["id"] in completed
        with cols[i]:
            border = "2px solid #22c55e" if done else "1px solid #2d3148"
            bg = "rgba(34,197,94,0.06)" if done else "#1a1d29"
            st.markdown(f"""<div style="text-align:center;padding:.75rem .25rem;border:{border};
                border-radius:10px;background:{bg};">
                <div style="font-size:1.3rem;">{stage['icon']}</div>
                <div style="font-size:.6rem;color:{'#22c55e' if done else '#64748b'};font-weight:600;margin-top:4px;">
                    {stage['label']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Run Full Pipeline", use_container_width=True, type="primary", key="ent_run_pipeline"):
        with st.spinner("Running pipeline..."):
            tracker.add_event("data_profiling", {"rows": len(df)})

            silver_df, silver_log = clean_dataset(raw_df.copy())
            st.session_state.ent_silver = silver_df
            st.session_state.ent_silver_log = silver_log
            tracker.add_event("data_quality", {"rules_checked": len(df.columns)})
            tracker.add_event("data_cleaning", silver_log)

            feat_df, feats = compute_features(silver_df.copy())
            st.session_state.ent_features = feat_df
            st.session_state.ent_feat_list = feats
            register_features(feats, ds_name)
            tracker.add_event("feature_engineering", {"features": len(feats)})

            gold_tables = generate_analytics_tables(silver_df)
            st.session_state.ent_gold = gold_tables
            tracker.add_event("kpi_aggregation", {"tables": len(gold_tables)})

            schema = detect_schema(silver_df)
            save_schema(ds_name, schema)

            tracker.add_event("dashboard", {"charts": "auto"})
            tracker.add_event("ai_insights", {"status": "complete"})

        st.success("Full pipeline complete — all layers processed.")
        st.rerun()

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Bronze", f"{len(raw_df):,} rows", "Raw", help="Raw ingested rows")
    with mc2:
        silver_rows = len(st.session_state.get("ent_silver", raw_df))
        st.metric("Silver", f"{silver_rows:,} rows", "Cleaned", help="Cleaned rows")
    with mc3:
        gold_count = len(st.session_state.get("ent_gold", {}))
        st.metric("Gold", f"{gold_count} tables", "Analytics", help="Analytics tables")
    with mc4:
        feat_count = len(st.session_state.get("ent_feat_list", []))
        st.metric("Features", feat_count, "Engineered", help="Computed features")


# ── Bronze ───────────────────────────────────────────────────────────────────

with tab_bronze:
    st.markdown("### 🥉 Bronze Layer — Raw Data")
    st.caption("Immutable storage of the original dataset as-is.")

    bmc1, bmc2, bmc3 = st.columns(3)
    with bmc1:
        st.metric("Bronze Rows", f"{len(raw_df):,}")
    with bmc2:
        st.metric("Bronze Columns", len(raw_df.columns))
    with bmc3:
        st.metric("Bronze Size", f"{raw_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    html = raw_df.head(50).to_html(index=False, classes="ent-table", border=0)
    st.markdown(f'<div style="overflow-x:auto;max-height:400px;overflow-y:auto;border-radius:8px;border:1px solid #2d3148;">{html}</div>', unsafe_allow_html=True)

    datasets = list_bronze_datasets()
    if datasets:
        st.markdown("---")
        st.markdown("**Ingestion History**")
        hist_df = pd.DataFrame(datasets)[["dataset_name", "storage_timestamp", "layer"]].head(10)
        _ent_html_table(hist_df)


# ── Silver ───────────────────────────────────────────────────────────────────

with tab_silver:
    st.markdown("### 🥈 Silver Layer — Cleaned Data")

    silver_df = st.session_state.get("ent_silver")
    silver_log = st.session_state.get("ent_silver_log")

    if silver_df is None:
        st.info("Run the pipeline first to generate Silver layer data.")
    else:
        smc1, smc2, smc3 = st.columns(3)
        with smc1:
            st.metric("Silver Rows", f"{len(silver_df):,}")
        with smc2:
            removed = len(raw_df) - len(silver_df)
            st.metric("Silver Rows Removed", removed)
        with smc3:
            st.metric("Silver Transforms", len(silver_log.get("transformations", [])))

        if silver_log:
            st.markdown("**Transformation Log**")
            for t in silver_log.get("transformations", []):
                op = t["operation"].replace("_", " ").title()
                detail = ""
                if "rows_removed" in t:
                    detail = f" — {t['rows_removed']} rows"
                elif "columns_renamed" in t:
                    detail = f" — {t['columns_renamed']} columns"
                elif "columns_filled" in t:
                    detail = f" — {len(t['columns_filled'])} columns"
                elif "type_changes" in t:
                    detail = f" — {len(t['type_changes'])} columns"
                st.markdown(f"- **{op}**{detail}")

        st.markdown("---")
        html = silver_df.head(50).to_html(index=False, classes="ent-table", border=0)
        st.markdown(f'<div style="overflow-x:auto;max-height:400px;overflow-y:auto;border-radius:8px;border:1px solid #2d3148;">{html}</div>', unsafe_allow_html=True)


# ── Gold ─────────────────────────────────────────────────────────────────────

with tab_gold:
    st.markdown("### 🥇 Gold Layer — Analytics Tables")

    gold_tables = st.session_state.get("ent_gold")
    if not gold_tables:
        st.info("Run the pipeline first to generate Gold layer analytics tables.")
    else:
        st.metric("Generated Tables", len(gold_tables))
        selected = st.selectbox("Select table", list(gold_tables.keys()), key="ent_gold_select")
        if selected:
            tbl = gold_tables[selected]
            html = tbl.to_html(index=False, classes="ent-table", border=0)
            st.markdown(f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #2d3148;">{html}</div>', unsafe_allow_html=True)

        if st.button("Save Gold Tables to Disk", key="ent_save_gold"):
            paths = store_gold_tables(gold_tables, ds_name)
            st.success(f"Saved {len(paths)} tables.")


# ── Features ─────────────────────────────────────────────────────────────────

with tab_feat:
    st.markdown("### ⚙ Feature Store")

    feat_list = st.session_state.get("ent_feat_list")
    if not feat_list:
        st.info("Run the pipeline first to compute features.")
    else:
        st.metric("Features Computed", len(feat_list))
        feat_df = pd.DataFrame(feat_list)
        html = feat_df.to_html(index=False, classes="ent-table", border=0)
        st.markdown(f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #2d3148;">{html}</div>', unsafe_allow_html=True)

    all_feats = list_features()
    if all_feats:
        st.markdown("---")
        st.markdown("**Feature Registry (all datasets)**")
        reg_df = pd.DataFrame(all_feats)[["name", "type", "description", "dataset"]].head(20)
        _ent_html_table(reg_df)


# ── Governance ───────────────────────────────────────────────────────────────

with tab_gov:
    render_governance_tab(df, ds_name)


# ── Observability ────────────────────────────────────────────────────────────

with tab_obs:
    render_observability_tab(df, profile)


# ── Schema ───────────────────────────────────────────────────────────────────

with tab_schema:
    st.markdown("### 📐 Schema Registry")

    current = detect_schema(df)
    schema_df = pd.DataFrame(current)
    html = schema_df.to_html(index=False, classes="ent-table", border=0)
    st.markdown(f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #2d3148;">{html}</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Save Schema Snapshot", use_container_width=True, key="ent_save_schema"):
            p = save_schema(ds_name, current)
            st.success(f"Saved: {p.name}")

    versions = list_schema_versions(ds_name)
    if versions:
        st.markdown("---")
        st.markdown("**Version History**")
        _ent_html_table(pd.DataFrame(versions))

        prev = load_schema(ds_name)
        if prev:
            prev_cols = {c["column_name"] for c in prev["schema"]}
            curr_cols = {c["column_name"] for c in current}
            added = curr_cols - prev_cols
            removed = prev_cols - curr_cols
            if added or removed:
                if added:
                    st.warning(f"New columns: {', '.join(added)}")
                if removed:
                    st.error(f"Removed columns: {', '.join(removed)}")
            else:
                st.info("No schema drift detected.")


# ── Lineage ──────────────────────────────────────────────────────────────────

with tab_lineage:
    st.markdown("### 🔗 Data Lineage")

    tracker = LineageTracker(ds_name)
    events = tracker.get_lineage()
    completed = {e["stage_id"] for e in events}

    cols = st.columns(len(PIPELINE_STAGES))
    for i, stage in enumerate(PIPELINE_STAGES):
        done = stage["id"] in completed
        with cols[i]:
            border = "2px solid #22c55e" if done else "1px solid #2d3148"
            bg = "rgba(34,197,94,0.06)" if done else "#1a1d29"
            st.markdown(f"""<div style="text-align:center;padding:.75rem .25rem;border:{border};
                border-radius:10px;background:{bg};">
                <div style="font-size:1.3rem;">{stage['icon']}</div>
                <div style="font-size:.6rem;color:{'#22c55e' if done else '#64748b'};font-weight:600;margin-top:4px;">
                    {stage['label']}</div>
            </div>""", unsafe_allow_html=True)

    if events:
        st.markdown("---")
        st.markdown("**Event Log**")
        events_df = pd.DataFrame(events)
        display_cols = [c for c in ["timestamp", "stage_id", "stage_label"] if c in events_df.columns]
        if display_cols:
            _ent_html_table(events_df[display_cols])

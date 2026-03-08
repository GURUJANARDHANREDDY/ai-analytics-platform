"""Dashboard UI components – dataset overview, KPIs, profiling (native Streamlit)."""

from __future__ import annotations

from typing import Any
import html as html_lib

import pandas as pd
import streamlit as st

from backend.data_profiler import DatasetProfile


_TYPE_ICONS = {
    "numeric":     "🔢",
    "categorical": "◆",
    "datetime":    "📅",
    "boolean":     "⊘",
    "text":        "¶",
    "identifier":  "🔑",
}


# ── Dataset overview ──────────────────────────────────────────────────────────

def render_dataset_overview(df: pd.DataFrame, profile: DatasetProfile) -> None:
    """Render hero metric cards and a data quality gauge."""
    st.subheader("📋 Dataset Overview")

    total_cells = profile.n_rows * profile.n_cols
    total_missing = sum(c.missing_count for c in profile.columns)
    completeness = (1 - total_missing / total_cells) * 100 if total_cells else 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{profile.n_rows:,}")
    c2.metric("Columns", f"{profile.n_cols}")
    c3.metric("Memory", f"{profile.memory_usage_mb:.1f} MB")
    c4.metric("Completeness", f"{completeness:.1f}%")

    # Data quality bar
    st.caption(f"Data Quality — {total_missing:,} missing out of {total_cells:,} cells")
    st.progress(min(completeness / 100, 1.0))

    with st.expander("🔍 Data Preview (first 100 rows)", expanded=True):
        preview = df.head(100)
        html_table = preview.to_html(index=False, classes="data-preview-table", border=0)
        st.markdown(f"""<div style="overflow-x:auto; max-height:400px; overflow-y:auto;
            border-radius:8px; border:1px solid #2d3148;">
            <style>
            .data-preview-table {{
                width: 100%; border-collapse: collapse; font-size: .82rem;
                font-family: 'Inter', monospace;
            }}
            .data-preview-table th {{
                background: #1e2235; color: #94a3b8; font-weight: 600;
                padding: 8px 12px; text-align: left; position: sticky; top: 0;
                border-bottom: 2px solid #2d3148; font-size: .75rem;
                text-transform: uppercase; letter-spacing: .03em;
            }}
            .data-preview-table td {{
                padding: 6px 12px; color: #e2e8f0; border-bottom: 1px solid #1e2235;
            }}
            .data-preview-table tr:hover td {{ background: rgba(99,102,241,0.06); }}
            </style>
            {html_table}
        </div>""", unsafe_allow_html=True)


# ── Column profiles ──────────────────────────────────────────────────────────

def render_column_profiles(profile: DatasetProfile) -> None:
    """Show a clean table of per-column metadata."""
    st.subheader("🔬 Column Profiles")

    rows = []
    for c in profile.columns:
        icon = _TYPE_ICONS.get(c.detected_type, "?")
        rows.append({
            "Column": c.name,
            "Type": f"{icon} {c.detected_type}",
            "Dtype": c.dtype,
            "Missing": c.missing_count,
            "Missing %": f"{c.missing_pct}%",
            "Unique": c.unique_count,
            "Completeness %": round(100 - c.missing_pct, 1),
        })

    profile_df = pd.DataFrame(rows)
    html_prof = profile_df.to_html(index=False, classes="data-preview-table", border=0)
    st.markdown(f"""<div style="overflow-x:auto; border-radius:8px; border:1px solid #2d3148;">
        {html_prof}
    </div>""", unsafe_allow_html=True)


# ── Numeric KPIs ─────────────────────────────────────────────────────────────

def render_numeric_kpis(profile: DatasetProfile) -> None:
    """Render KPI metrics for numeric columns."""
    if not profile.numeric_kpis:
        return

    st.subheader("🔢 Numeric Statistics")

    items = list(profile.numeric_kpis.items())
    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        row_items = items[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (col_name, kpi) in enumerate(row_items):
            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"**{col_name}**")
                    m1, m2 = st.columns(2)
                    m1.metric("Mean", f"{kpi['mean']:,.2f}")
                    m2.metric("Median", f"{kpi['median']:,.2f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Min", f"{kpi['min']:,.2f}")
                    m4.metric("Max", f"{kpi['max']:,.2f}")
                    st.caption(f"Std Dev: {kpi['std']:,.2f}  ·  Range: {kpi['max'] - kpi['min']:,.2f}")


# ── Categorical KPIs ─────────────────────────────────────────────────────────

def render_categorical_kpis(profile: DatasetProfile) -> None:
    """Render top-category breakdown with bar charts."""
    if not profile.categorical_kpis:
        return

    import plotly.express as px

    st.subheader("🏷️ Categorical Breakdown")

    items = list(profile.categorical_kpis.items())
    cols_per_row = 2
    for i in range(0, len(items), cols_per_row):
        row_items = items[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (col_name, kpi) in enumerate(row_items):
            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"**{col_name}** — {kpi['unique_count']} unique")
                    top = list(kpi.get("top_categories", {}).items())[:10]
                    if not top:
                        st.caption("No data")
                        continue
                    chart_df = pd.DataFrame(top, columns=["Category", "Count"])
                    fig = px.bar(chart_df, x="Category", y="Count",
                                 color_discrete_sequence=["#6366f1"],
                                 text="Count")
                    fig.update_traces(textposition="outside", textfont_size=10)
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(26,29,41,0.8)",
                        font=dict(color="#94a3b8", size=10),
                        margin=dict(l=30, r=10, t=10, b=40),
                        height=250,
                        xaxis_tickangle=-30,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"cat_bar_{col_name}_{i}_{j}")


# ── Anomaly summary ──────────────────────────────────────────────────────────

def render_anomaly_summary(anomalies: dict[str, list[int]]) -> None:
    """Show anomaly detection results."""
    st.subheader("⚠️ Anomaly Detection")

    if not anomalies:
        st.success("No statistical outliers detected (z-score > 3σ). All columns are within normal range.")
        return

    for col, idxs in anomalies.items():
        st.warning(f"**{col}** — {len(idxs)} outlier{'s' if len(idxs) != 1 else ''} detected (z-score > 3σ)")

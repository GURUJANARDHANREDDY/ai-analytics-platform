"""AI-generated Tableau-style dashboard — LLM designs the layout, charts, and KPIs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.column_classifier import get_measure_columns, get_dimension_columns, classify_all_columns


def _html_table(data: pd.DataFrame, max_rows: int = 50) -> None:
    """Render a DataFrame as a styled HTML table (visible on dark theme)."""
    html = data.head(max_rows).to_html(index=False, classes="ht", border=0)
    st.markdown(f"""<div style="overflow-x:auto; max-height:400px; overflow-y:auto;
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

_COLORS = ["#6366f1", "#818cf8", "#a78bfa", "#22c55e", "#f59e0b",
           "#ef4444", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6"]

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,41,0.8)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor="#1e293b", font_size=11, font_color="#f1f5f9", bordercolor="#334155"),
)


def _fmt(n: float, prefix: str = "") -> str:
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1e9:
        return f"{sign}{prefix}{abs_n/1e9:.1f}B"
    if abs_n >= 1e6:
        return f"{sign}{prefix}{abs_n/1e6:.1f}M"
    if abs_n >= 1e3:
        return f"{sign}{prefix}{abs_n/1e3:.1f}K"
    if abs_n == int(abs_n):
        return f"{sign}{prefix}{int(abs_n):,}"
    return f"{sign}{prefix}{abs_n:,.2f}"


def _is_monetary(col: str) -> bool:
    return any(k in col.lower() for k in ["revenue", "sales", "amount", "price", "total", "cost", "profit", "income", "value", "pay", "wage", "salary", "gross", "net"])


def _prefix(col: str) -> str:
    return "$" if _is_monetary(col) else ""


# ═════════════════════════════════════════════════════════════════════════════
# AI CHART RENDERER — takes LLM chart spec and renders it
# ═════════════════════════════════════════════════════════════════════════════

def _render_ai_chart(chart: dict, filtered: pd.DataFrame, idx: int) -> bool:
    """Render a single chart from an AI design spec. Returns True if rendered."""
    ctype = chart.get("type", "bar")
    title = chart.get("title", "Chart")
    x = chart.get("x")
    y = chart.get("y")
    color = chart.get("color")
    agg = chart.get("agg", "sum")
    top_n = chart.get("top_n", 10)
    reason = chart.get("reason", "")

    if x and x not in filtered.columns:
        return False
    if y and y not in filtered.columns:
        return False
    if color and color not in filtered.columns:
        color = None

    try:
        if ctype in ("bar", "horizontal_bar"):
            if x and y:
                data = filtered.groupby(x)[y].agg(agg or "sum").sort_values(ascending=False).head(top_n).reset_index()
                orient = "h" if ctype == "horizontal_bar" else "v"
                if orient == "h":
                    fig = px.bar(data, y=x, x=y, orientation="h", color_discrete_sequence=[_COLORS[idx % len(_COLORS)]])
                else:
                    pp = _prefix(y)
                    fig = px.bar(data, x=x, y=y, color_discrete_sequence=[_COLORS[idx % len(_COLORS)]],
                                 text=data[y].apply(lambda v: _fmt(v, pp)))
                    fig.update_traces(textposition="outside", textfont_size=10)
                    fig.update_layout(xaxis_tickangle=-40)
            else:
                return False
            fig.update_layout(**_LAYOUT, title=title, height=380)
            st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")

        elif ctype == "grouped_bar":
            if x and y and color:
                data = filtered.groupby([x, color])[y].agg(agg or "sum").reset_index()
                top_x = filtered.groupby(x)[y].agg(agg or "sum").nlargest(top_n).index.tolist()
                data = data[data[x].isin(top_x)]
                fig = px.bar(data, x=x, y=y, color=color, barmode="group", color_discrete_sequence=_COLORS)
                fig.update_layout(**_LAYOUT, title=title, height=380, xaxis_tickangle=-40)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype == "line":
            if x and y:
                data = filtered.copy()
                if pd.api.types.is_datetime64_any_dtype(data[x]):
                    data["_period"] = data[x].dt.to_period("M").astype(str)
                    trend = data.groupby("_period")[y].agg(agg or "sum").reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=trend["_period"], y=trend[y],
                                             mode="lines+markers", line=dict(color=_COLORS[idx % len(_COLORS)], width=2.5),
                                             fill="tozeroy", fillcolor=f"rgba(99,102,241,0.08)",
                                             marker=dict(size=6)))
                    fig.update_layout(**_LAYOUT, title=title, height=380)
                else:
                    agg_data = data.groupby(x)[y].agg(agg or "sum").reset_index()
                    fig = px.line(agg_data, x=x, y=y, color_discrete_sequence=[_COLORS[idx % len(_COLORS)]], markers=True)
                    fig.update_layout(**_LAYOUT, title=title, height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype == "scatter":
            if x and y:
                sample = filtered.sample(min(500, len(filtered)))
                fig = px.scatter(sample, x=x, y=y, color=color, color_discrete_sequence=_COLORS, opacity=0.7)
                fig.update_layout(**_LAYOUT, title=title, height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype in ("pie", "donut"):
            col = x or y
            if col:
                counts = filtered[col].value_counts().head(top_n)
                hole = 0.4 if ctype == "donut" else 0
                fig = px.pie(values=counts.values, names=counts.index,
                             color_discrete_sequence=_COLORS, hole=hole)
                fig.update_layout(**_LAYOUT, title=title, height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype == "treemap":
            if x and y:
                data = filtered.groupby(x)[y].agg(agg or "sum").reset_index().nlargest(top_n, y)
                if len(data) >= 2:
                    fig = px.treemap(data, path=[x], values=y, color=y,
                                     color_continuous_scale=["#1e293b", "#6366f1", "#818cf8"])
                    fig.update_layout(**_LAYOUT, title=title, height=380)
                    fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,.0f}", textfont_size=11)
                    st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
                else:
                    return False
            else:
                return False

        elif ctype == "box":
            if x and y:
                fig = px.box(filtered, x=x, y=y, color_discrete_sequence=[_COLORS[idx % len(_COLORS)]])
                fig.update_layout(**_LAYOUT, title=title, height=380, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype == "histogram":
            col = x or y
            if col:
                fig = px.histogram(filtered, x=col, nbins=40, color_discrete_sequence=[_COLORS[idx % len(_COLORS)]])
                fig.update_layout(**_LAYOUT, title=title, height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False

        elif ctype == "heatmap":
            measures = get_measure_columns(filtered)
            if len(measures) >= 3:
                corr = filtered[measures[:10]].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#6366f1"]],
                    zmin=-1, zmax=1, text=np.round(corr.values, 2),
                    texttemplate="%{text}", textfont=dict(size=10, color="#f1f5f9"),
                ))
                fig.update_layout(**_LAYOUT, title=title, height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{idx}")
            else:
                return False
        else:
            return False

        if reason:
            st.caption(f"💡 {reason}")
        return True

    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def _render_filters(df: pd.DataFrame, measures: list[str], dimensions: list[str],
                    key_prefix: str) -> tuple[pd.DataFrame, int, int]:
    """Shared filter bar for both dashboard types. Returns (filtered_df, primary_idx, active_count)."""
    n_filter_cols = min(len(dimensions), 4)
    header_cols = st.columns([1.2] + [1] * n_filter_cols)
    filters: dict[str, list] = {}

    with header_cols[0]:
        primary_idx = st.selectbox(
            "Measure", range(len(measures)),
            format_func=lambda i: measures[i], key=f"{key_prefix}_measure",
        )

    for i, dim in enumerate(dimensions[:n_filter_cols]):
        with header_cols[i + 1]:
            unique_vals = sorted([str(v) for v in df[dim].dropna().unique().tolist()])
            selected = st.multiselect(dim, unique_vals, default=[], key=f"{key_prefix}_f_{dim}")
            if selected:
                filters[dim] = selected

    filtered = df.copy()
    for col, vals in filters.items():
        filtered = filtered[filtered[col].astype(str).isin(vals)]

    active = len(filters)
    st.caption(f"{'🔵 ' + str(active) + ' filter(s) active · ' if active else ''}{len(filtered):,} of {len(df):,} rows")
    return filtered, primary_idx, active


def _render_kpi_cards(filtered: pd.DataFrame, df: pd.DataFrame, measures: list[str],
                      active: int, key_prefix: str) -> None:
    """Render heuristic KPI cards."""
    kpi_measures = measures[:6]
    kpi_cols_ui = st.columns(len(kpi_measures))
    for i, col in enumerate(kpi_measures):
        with kpi_cols_ui[i]:
            p = _prefix(col)
            total = filtered[col].sum()
            avg = filtered[col].mean()
            if active and len(df) > len(filtered):
                base_total = df[col].sum()
                delta_pct = ((total - base_total) / abs(base_total) * 100) if base_total != 0 else 0
                st.metric(f"Total {col}", _fmt(total, p), f"{delta_pct:+.1f}% vs all")
            else:
                st.metric(f"Total {col}", _fmt(total, p), f"Avg: {_fmt(avg, p)}")


def _render_ranking_tables(filtered: pd.DataFrame, measures: list[str],
                           dimensions: list[str], primary_measure: str) -> None:
    """Render ranking tables section."""
    if not (dimensions and measures):
        return
    st.markdown("---")
    st.markdown("""<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;">
        <span style="font-size:1.1rem;">📋</span>
        <span style="font-size:1rem;font-weight:700;color:#f1f5f9;">Ranking Tables</span>
    </div>""", unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        dim = dimensions[0]
        agg_cols = [primary_measure] + [m for m in measures if m != primary_measure][:2]
        top = filtered.groupby(dim)[agg_cols].agg(["sum", "mean"]).round(2)
        top.columns = [f"{stat.title()} {col}" for col, stat in top.columns]
        top["Count"] = filtered.groupby(dim).size()
        top = top.sort_values(top.columns[0], ascending=False).head(10).reset_index()
        st.markdown(f"**Top {dim} by {primary_measure}**")
        _html_table(top)

    with tc2:
        if len(dimensions) >= 2:
            dim2 = dimensions[1]
            top2 = filtered.groupby(dim2)[primary_measure].agg(["sum", "mean", "count"]).round(2)
            top2 = top2.sort_values("sum", ascending=False).head(10).reset_index()
            top2.columns = [dim2, f"Total {primary_measure}", f"Avg {primary_measure}", "Count"]
            st.markdown(f"**Top {dim2} by {primary_measure}**")
            _html_table(top2)
        elif len(measures) >= 2:
            m2 = measures[1]
            top2 = filtered.groupby(dim)[m2].agg(["sum", "mean", "count"]).round(2)
            top2 = top2.sort_values("sum", ascending=False).head(10).reset_index()
            top2.columns = [dim, f"Total {m2}", f"Avg {m2}", "Count"]
            st.markdown(f"**Top {dim} by {m2}**")
            _html_table(top2)


# ═════════════════════════════════════════════════════════════════════════════
# AI-GENERATED DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def render_ai_dashboard(df: pd.DataFrame) -> None:
    """Render the AI-designed dashboard — LLM picks KPIs, charts, and layout."""
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    if not measures:
        st.warning("No numeric measure columns found.")
        return

    design = st.session_state.get("ai_dashboard_design")

    if design is None:
        st.markdown("""<div style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
            <div style="font-size:1.4rem; font-weight:800; color:#f1f5f9; margin-bottom:.5rem;">AI Dashboard Designer</div>
            <div style="font-size:.9rem; color:#94a3b8; max-width:500px; margin:0 auto 1.5rem; line-height:1.6;">
                Click the button below and the AI will analyze your data, pick the best KPIs,
                choose the right chart types, and design a custom dashboard layout.
            </div>
            <div style="font-size:.78rem; color:#64748b; margin-bottom:.5rem;">Powered by Llama 3.1 via Hugging Face</div>
        </div>""", unsafe_allow_html=True)

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("🤖 Generate AI Dashboard", use_container_width=True, type="primary", key="gen_ai_dash"):
                from ai.dashboard_designer import design_dashboard
                with st.spinner("🤖 AI is analyzing your data and designing the dashboard..."):
                    result = design_dashboard(df)
                if result:
                    st.session_state.ai_dashboard_design = result
                    st.rerun()
                else:
                    st.error("AI design failed. Check your HF_API_TOKEN in .env file.")

            if st.button("🔄 Regenerate", use_container_width=True, key="regen_ai_dash", disabled=True):
                pass
        return

    # ── Header ───────────────────────────────────────────────────────────────
    hdr1, hdr2 = st.columns([4, 1])
    with hdr1:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem;">
            <span style="font-size:1.4rem;">🤖</span>
            <div>
                <div style="display:flex;align-items:center;gap:.5rem;">
                    <span style="font-size:1.3rem;font-weight:800;color:#f1f5f9;">{design.get('title', 'AI Dashboard')}</span>
                    <span style="font-size:.7rem;font-weight:600;color:#22c55e;background:rgba(34,197,94,.12);
                           padding:3px 10px;border-radius:999px;">AI-DESIGNED</span>
                </div>
                <div style="font-size:.82rem;color:#64748b;margin-top:2px;">{design.get('subtitle', 'Dashboard designed by Llama 3.1')}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with hdr2:
        if st.button("🔄 Redesign", key="redesign_ai_dash", use_container_width=True):
            from ai.dashboard_designer import design_dashboard
            with st.spinner("Redesigning..."):
                result = design_dashboard(df)
            if result:
                st.session_state.ai_dashboard_design = result
                st.rerun()

    # ── AI Insights ──────────────────────────────────────────────────────────
    if design.get("insights"):
        insights = design["insights"][:3]
        icols = st.columns(len(insights))
        for i, insight in enumerate(insights):
            with icols[i]:
                st.markdown(f"""<div style="background:#1a1d29;border:1px solid #2d3148;border-radius:10px;
                    padding:.75rem 1rem;font-size:.8rem;color:#e2e8f0;">
                    <span style="color:#f59e0b;">💡</span> {insight}
                </div>""", unsafe_allow_html=True)
        st.markdown("")

    # ── Filters ──────────────────────────────────────────────────────────────
    filtered, primary_idx, active = _render_filters(df, measures, dimensions, "ai")

    # ── AI KPI Cards ─────────────────────────────────────────────────────────
    if design.get("kpis"):
        kpis = design["kpis"]
        kpi_cols_ui = st.columns(len(kpis))
        for i, kpi in enumerate(kpis):
            with kpi_cols_ui[i]:
                col_name = kpi.get("column", "")
                agg_fn = kpi.get("agg", "sum")
                label = kpi.get("label", col_name)
                if col_name in filtered.columns:
                    p = _prefix(col_name)
                    val = getattr(filtered[col_name], agg_fn, filtered[col_name].sum)()
                    if active and len(df) > len(filtered):
                        base_val = getattr(df[col_name], agg_fn, df[col_name].sum)()
                        delta = ((val - base_val) / abs(base_val) * 100) if base_val != 0 else 0
                        st.metric(label, _fmt(val, p), f"{delta:+.1f}% vs all")
                    else:
                        avg = filtered[col_name].mean()
                        st.metric(label, _fmt(val, p), f"Avg: {_fmt(avg, p)}")

    st.markdown("---")

    # ── AI Charts ────────────────────────────────────────────────────────────
    charts = design.get("charts", [])
    for row_start in range(0, len(charts), 2):
        row_charts = charts[row_start:row_start + 2]
        cols = st.columns(len(row_charts))
        for ci, chart in enumerate(row_charts):
            with cols[ci]:
                _render_ai_chart(chart, filtered, row_start + ci)

    # ── Ranking Tables ───────────────────────────────────────────────────────
    _render_ranking_tables(filtered, measures, dimensions, measures[primary_idx])


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATED DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def render_auto_dashboard(df: pd.DataFrame) -> None:
    """Render the heuristic auto-generated dashboard — rule-based chart selection."""
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    if not measures:
        st.warning("No numeric measure columns found.")
        return

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem;">
        <span style="font-size:1.4rem;">📊</span>
        <div>
            <div style="display:flex;align-items:center;gap:.5rem;">
                <span style="font-size:1.3rem;font-weight:800;color:#f1f5f9;">Analytics Dashboard</span>
                <span style="font-size:.7rem;font-weight:600;color:#6366f1;background:rgba(99,102,241,.12);
                       padding:3px 10px;border-radius:999px;">AUTO-GENERATED</span>
            </div>
            <div style="font-size:.82rem;color:#64748b;margin-top:2px;">Rule-based charts generated from data structure</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────────
    filtered, primary_idx, active = _render_filters(df, measures, dimensions, "auto")

    # ── KPI Cards ────────────────────────────────────────────────────────────
    _render_kpi_cards(filtered, df, measures, active, "auto")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    _render_fallback_charts(filtered, measures, dimensions, datetime_cols, primary_idx)

    # ── Ranking Tables ───────────────────────────────────────────────────────
    _render_ranking_tables(filtered, measures, dimensions, measures[primary_idx])

    # ── AI Q&A PANEL ─────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("""<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem;">
        <span style="font-size:1.2rem;">🤖</span>
        <span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">Ask anything about this data</span>
        <span style="font-size:.75rem;color:#64748b;margin-left:.5rem;">Powered by Llama 3.1</span>
    </div>""", unsafe_allow_html=True)

    from backend.narrative_engine import generate_smart_suggestions
    suggestions = generate_smart_suggestions(filtered)

    if "dash_chat" not in st.session_state:
        st.session_state.dash_chat = []

    if not st.session_state.dash_chat:
        scols = st.columns(min(len(suggestions), 4))
        for i, col in enumerate(scols):
            if i < len(suggestions):
                with col:
                    if st.button(f"✦ {suggestions[i]}", key=f"dsug_{i}", use_container_width=True):
                        _ask_dashboard_question(filtered, suggestions[i])
                        st.rerun()

    for msg in st.session_state.dash_chat:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📊"):
            st.markdown(msg["content"])
            if msg.get("data") is not None:
                _html_table(msg["data"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    dash_input = st.chat_input("Ask any question about the data...", key="dash_chat_input")
    if dash_input:
        _ask_dashboard_question(filtered, dash_input)
        st.rerun()


def _render_fallback_charts(filtered: pd.DataFrame, measures: list[str],
                            dimensions: list[str], datetime_cols: list[str],
                            primary_idx: int) -> None:
    """Heuristic chart generation when LLM is unavailable."""
    primary_measure = measures[primary_idx]
    pp = _prefix(primary_measure)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if dimensions:
            dim = dimensions[0]
            agg = filtered.groupby(dim)[primary_measure].sum().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(agg, x=dim, y=primary_measure, color_discrete_sequence=[_COLORS[0]],
                         text=agg[primary_measure].apply(lambda x: _fmt(x, pp)))
            fig.update_traces(textposition="outside", textfont_size=10)
            fig.update_layout(**_LAYOUT, title=f"{primary_measure} by {dim}", height=380, xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True, key="fb_bar1")

    with r1c2:
        if datetime_cols:
            dt_col = datetime_cols[0]
            time_df = filtered.copy()
            time_df["_period"] = time_df[dt_col].dt.to_period("M").astype(str)
            trend = time_df.groupby("_period")[primary_measure].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["_period"], y=trend[primary_measure],
                                     mode="lines+markers", line=dict(color=_COLORS[0], width=2.5),
                                     fill="tozeroy", fillcolor="rgba(99,102,241,0.08)", marker=dict(size=6)))
            fig.update_layout(**_LAYOUT, title=f"{primary_measure} Trend", height=380)
            st.plotly_chart(fig, use_container_width=True, key="fb_trend1")
        elif len(dimensions) >= 2:
            dim = dimensions[1]
            counts = filtered[dim].value_counts().head(8)
            fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=_COLORS, hole=0.4)
            fig.update_layout(**_LAYOUT, title=f"{dim} Distribution", height=380)
            st.plotly_chart(fig, use_container_width=True, key="fb_pie1")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if len(dimensions) >= 2:
            dim = dimensions[1]
            agg = filtered.groupby(dim)[primary_measure].sum().sort_values(ascending=True).tail(10).reset_index()
            fig = px.bar(agg, y=dim, x=primary_measure, orientation="h", color_discrete_sequence=[_COLORS[3]])
            fig.update_layout(**_LAYOUT, title=f"{primary_measure} by {dim}", height=380)
            st.plotly_chart(fig, use_container_width=True, key="fb_hbar")
    with r2c2:
        if len(measures) >= 2:
            m2 = [m for m in measures if m != primary_measure][0]
            sample = filtered.sample(min(500, len(filtered)))
            color_dim = dimensions[0] if dimensions and filtered[dimensions[0]].nunique() <= 10 else None
            fig = px.scatter(sample, x=primary_measure, y=m2, color=color_dim,
                             color_discrete_sequence=_COLORS, opacity=0.7)
            fig.update_layout(**_LAYOUT, title=f"{m2} vs {primary_measure}", height=380)
            st.plotly_chart(fig, use_container_width=True, key="fb_scatter")

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        if dimensions:
            dim = dimensions[0]
            tree_data = filtered.groupby(dim)[primary_measure].sum().reset_index().nlargest(12, primary_measure)
            if len(tree_data) >= 2:
                fig = px.treemap(tree_data, path=[dim], values=primary_measure, color=primary_measure,
                                 color_continuous_scale=["#1e293b", "#6366f1", "#818cf8"])
                fig.update_layout(**_LAYOUT, title=f"{primary_measure} Treemap", height=380)
                fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,.0f}", textfont_size=11)
                st.plotly_chart(fig, use_container_width=True, key="fb_tree")
    with r3c2:
        if len(measures) >= 3:
            corr = filtered[measures[:8]].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#6366f1"]],
                zmin=-1, zmax=1, text=np.round(corr.values, 2),
                texttemplate="%{text}", textfont=dict(size=10, color="#f1f5f9")))
            fig.update_layout(**_LAYOUT, title="Correlation Matrix", height=380)
            st.plotly_chart(fig, use_container_width=True, key="fb_corr")


def _ask_dashboard_question(df: pd.DataFrame, question: str) -> None:
    """Process a question and add to dashboard chat history."""
    st.session_state.dash_chat.append({"role": "user", "content": question})

    from backend.ai_chat_engine import ask_question
    resp = ask_question(df, question)

    entry: dict[str, Any] = {"role": "assistant", "content": resp.answer}

    if resp.data is not None:
        entry["data"] = resp.data.head(20)
        if len(resp.data.columns) >= 2:
            num_cols = resp.data.select_dtypes(include="number").columns.tolist()
            other_cols = [c for c in resp.data.columns if c not in num_cols]
            if other_cols and num_cols:
                try:
                    fig = px.bar(resp.data.head(15), x=other_cols[0], y=num_cols[0],
                                 color_discrete_sequence=["#6366f1"])
                    fig.update_layout(**_LAYOUT, height=300)
                    entry["chart"] = fig
                except Exception:
                    pass

    st.session_state.dash_chat.append(entry)

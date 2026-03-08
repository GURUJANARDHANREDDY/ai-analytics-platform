"""Visualization rendering components (modern SaaS style)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from backend.visualization_engine import auto_generate_charts
from backend.data_profiler import DatasetProfile


def render_auto_charts(df: pd.DataFrame, profile: DatasetProfile) -> None:
    """Generate and display automatic charts inside styled card wrappers."""
    charts = auto_generate_charts(df, profile)

    if not charts:
        st.info("No suitable columns found for automatic chart generation. Upload a dataset with numeric or categorical columns.")
        return

    type_counts: dict[str, int] = {}
    for c in charts:
        ctype = c["title"].split(" – ")[0].split("–")[0].strip()
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    st.caption("Auto-generated: " + " · ".join(f"{t} ({n})" for t, n in type_counts.items()))

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col_ui in enumerate(cols):
            idx = i + j
            if idx >= len(charts):
                break
            chart = charts[idx]
            with col_ui:
                with st.container(border=True):
                    st.caption(chart["title"])
                    st.plotly_chart(
                        chart["figure"],
                        use_container_width=True,
                        key=f"auto_chart_{idx}",
                    )


def render_custom_chart_builder(df: pd.DataFrame) -> None:
    """Interactive chart builder with modern controls."""

    st.subheader("🛠️ Custom Chart Builder")
    st.caption("Select a chart type and columns to create your own visualization.")

    from utils.column_classifier import get_measure_columns
    numeric_cols = get_measure_columns(df)
    all_cols = df.columns.tolist()

    builder_col1, builder_col2 = st.columns([1, 2])

    with builder_col1:
        chart_type = st.radio(
            "Chart type",
            ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart"],
            key="custom_chart_type",
            label_visibility="collapsed",
        )

    with builder_col2:
        _render_chart_controls(df, chart_type, numeric_cols, all_cols)


def _render_chart_controls(
    df: pd.DataFrame,
    chart_type: str,
    numeric_cols: list[str],
    all_cols: list[str],
) -> None:
    """Render column selectors and generate button for the chosen chart type."""

    if chart_type == "Histogram":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        col = st.selectbox("Column", numeric_cols, key="hist_col")
        if col and st.button("Generate Chart", key="btn_hist", use_container_width=True):
            with st.spinner("Rendering…"):
                from backend.visualization_engine import histogram
                fig = histogram(df, col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        col = st.selectbox("Column", all_cols, key="bar_col")
        if col and st.button("Generate Chart", key="btn_bar", use_container_width=True):
            with st.spinner("Rendering…"):
                from backend.visualization_engine import bar_chart
                fig = bar_chart(df, col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a scatter plot.")
        else:
            sc1, sc2 = st.columns(2)
            with sc1:
                x = st.selectbox("X axis", numeric_cols, key="scatter_x")
            with sc2:
                y = st.selectbox("Y axis", [c for c in numeric_cols if c != x], key="scatter_y")
            if x and y and st.button("Generate Chart", key="btn_scatter", use_container_width=True):
                with st.spinner("Rendering…"):
                    import plotly.express as px
                    fig = px.scatter(
                        df, x=x, y=y,
                        title=f"{y} vs {x}",
                        template="plotly_dark",
                        color_discrete_sequence=["#818cf8"],
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                    )
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        col = st.selectbox("Column", numeric_cols, key="box_col")
        if col and st.button("Generate Chart", key="btn_box", use_container_width=True):
            with st.spinner("Rendering…"):
                from backend.visualization_engine import box_plot
                fig = box_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        lc1, lc2 = st.columns(2)
        with lc1:
            x = st.selectbox("X axis", all_cols, key="line_x")
        with lc2:
            y = st.selectbox("Y axis", numeric_cols, key="line_y") if numeric_cols else None
        if not numeric_cols:
            st.info("No numeric columns available for Y axis.")
            return
        if x and y and st.button("Generate Chart", key="btn_line", use_container_width=True):
            with st.spinner("Rendering…"):
                import plotly.express as px
                fig = px.line(
                    df.sort_values(x), x=x, y=y,
                    title=f"{y} over {x}",
                    template="plotly_dark",
                    color_discrete_sequence=["#818cf8"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                )
            st.plotly_chart(fig, use_container_width=True)

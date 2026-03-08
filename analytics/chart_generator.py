"""Visualization Engine – generates interactive Plotly charts, skipping ID columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.column_classifier import get_measure_columns, get_dimension_columns

CHART_THEME = "plotly_dark"
COLOR_PALETTE = px.colors.qualitative.Set2


def _common_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,41,0.8)",
        font=dict(color="#94a3b8", size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def generate_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#6366f1"])
    return _common_layout(fig, f"Distribution of {col}")


def generate_bar_chart(df: pd.DataFrame, x: str, y: str, top_n: int = 15) -> go.Figure:
    plot_df = df.nlargest(top_n, y) if pd.api.types.is_numeric_dtype(df[y]) else df.head(top_n)
    fig = px.bar(plot_df, x=x, y=y, color_discrete_sequence=["#6366f1"])
    return _common_layout(fig, f"{y} by {x}")


def generate_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    sorted_df = df.sort_values(date_col)
    fig = px.line(sorted_df, x=date_col, y=value_col, color_discrete_sequence=["#818cf8"])
    fig.update_traces(line=dict(width=2))
    return _common_layout(fig, f"{value_col} over Time")


def generate_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    measure_cols = get_measure_columns(df)
    if len(measure_cols) < 2:
        return None
    cols_to_use = measure_cols[:15]
    corr = df[cols_to_use].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
    ))
    return _common_layout(fig, "Correlation Heatmap")


def generate_top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", color_discrete_sequence=["#22c55e"])
    return _common_layout(fig, f"Top {top_n} – {col}")


def generate_pie_chart(df: pd.DataFrame, col: str, top_n: int = 8) -> go.Figure:
    counts = df[col].value_counts().head(top_n)
    fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=COLOR_PALETTE)
    return _common_layout(fig, f"Distribution – {col}")


def generate_scatter(df: pd.DataFrame, x: str, y: str,
                     color: str | None = None) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, color_discrete_sequence=COLOR_PALETTE,
                     opacity=0.7)
    return _common_layout(fig, f"{y} vs {x}")


def generate_box_plot(df: pd.DataFrame, col: str,
                      group_col: str | None = None) -> go.Figure:
    fig = px.box(df, x=group_col, y=col, color_discrete_sequence=["#6366f1"])
    return _common_layout(fig, f"Box Plot – {col}")


def auto_generate_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Automatically generate charts using only meaningful measure columns."""
    charts: list[dict[str, Any]] = []
    measure_cols = get_measure_columns(df)
    dimension_cols = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    for col in measure_cols[:4]:
        charts.append({"title": f"Distribution of {col}", "type": "histogram",
                       "figure": generate_histogram(df, col)})

    heatmap = generate_correlation_heatmap(df)
    if heatmap:
        charts.append({"title": "Correlation Heatmap", "type": "heatmap", "figure": heatmap})

    for cat_col in dimension_cols[:3]:
        charts.append({"title": f"Top Categories – {cat_col}", "type": "bar",
                       "figure": generate_top_categories(df, cat_col)})

    for dt_col in datetime_cols[:2]:
        for num_col in measure_cols[:2]:
            charts.append({"title": f"{num_col} over Time", "type": "time_series",
                           "figure": generate_time_series(df, dt_col, num_col)})

    for cat_col in dimension_cols[:2]:
        for num_col in measure_cols[:2]:
            agg = df.groupby(cat_col, dropna=False)[num_col].sum().reset_index()
            agg = agg.sort_values(num_col, ascending=False).head(15)
            charts.append({"title": f"{num_col} by {cat_col}", "type": "bar",
                           "figure": generate_bar_chart(agg, cat_col, num_col)})

    if len(measure_cols) >= 2:
        charts.append({"title": f"{measure_cols[1]} vs {measure_cols[0]}", "type": "scatter",
                       "figure": generate_scatter(df, measure_cols[0], measure_cols[1])})

    return charts

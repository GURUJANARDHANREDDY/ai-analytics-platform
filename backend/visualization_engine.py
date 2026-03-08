"""Automatic chart generation with modern dark-theme styling."""

from __future__ import annotations

import io
import base64
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from backend.data_profiler import DatasetProfile, classify_column, compute_correlations
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Design tokens (shared palette) ───────────────────────────────────────────

_PALETTE = [
    "#818cf8", "#6366f1", "#a78bfa", "#c084fc",
    "#f472b6", "#fb7185", "#f59e0b", "#34d399",
    "#22d3ee", "#60a5fa",
]

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    title_font=dict(size=15, color="#f1f5f9"),
    margin=dict(l=48, r=24, t=52, b=48),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=12,
        font_family="Inter, sans-serif",
        font_color="#f1f5f9",
        bordercolor="#334155",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
    ),
)


def _apply_defaults(fig: go.Figure) -> go.Figure:
    """Apply the shared dark layout to a Plotly figure."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="#1e2235", zerolinecolor="#1e2235")
    fig.update_yaxes(gridcolor="#1e2235", zerolinecolor="#1e2235")
    return fig


# ── Plotly interactive charts ─────────────────────────────────────────────────

def histogram(df: pd.DataFrame, column: str, nbins: int = 30) -> go.Figure:
    """Interactive histogram for a numeric column."""
    fig = px.histogram(
        df, x=column, nbins=nbins,
        title=f"Distribution of {column}",
        color_discrete_sequence=[_PALETTE[0]],
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=0.85,
        hovertemplate=f"{column}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    )
    fig.update_layout(bargap=0.06)
    return _apply_defaults(fig)


def bar_chart(df: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    """Interactive bar chart for a categorical column (top N values)."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]

    n = len(counts)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]

    fig = go.Figure(
        go.Bar(
            x=counts[column], y=counts["count"],
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Categories – {column}",
        xaxis_title=column,
        yaxis_title="Count",
        xaxis_tickangle=-40,
    )
    return _apply_defaults(fig)


def correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Interactive Plotly heatmap of the Pearson correlation matrix."""
    corr = compute_correlations(df)
    if corr.empty:
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[
                [0.0, "#ef4444"], [0.25, "#fb7185"],
                [0.5, "#1e293b"],
                [0.75, "#818cf8"], [1.0, "#6366f1"],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#f1f5f9"),
            hovertemplate="(%{x}, %{y}): %{z:.2f}<extra></extra>",
            colorbar=dict(
                thickness=12,
                outlinewidth=0,
                tickfont=dict(color="#94a3b8"),
            ),
        )
    )
    fig.update_layout(title="Correlation Heatmap", height=520)
    return _apply_defaults(fig)


def time_series_chart(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    """Interactive time-series line chart with gradient fill."""
    sorted_df = df[[date_col, value_col]].dropna().sort_values(date_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_df[date_col],
        y=sorted_df[value_col],
        mode="lines",
        line=dict(color=_PALETTE[0], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.10)",
        hovertemplate=f"{date_col}: %{{x}}<br>{value_col}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{value_col} over Time",
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return _apply_defaults(fig)


def box_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Interactive box plot for outlier visualization."""
    fig = px.box(
        df, y=column,
        title=f"Box Plot – {column}",
        color_discrete_sequence=[_PALETTE[7]],
    )
    fig.update_traces(
        marker_color=_PALETTE[7],
        line_color=_PALETTE[7],
        fillcolor="rgba(52,211,153,0.15)",
    )
    return _apply_defaults(fig)


# ── Seaborn / Matplotlib static charts ───────────────────────────────────────

def seaborn_pairplot_base64(df: pd.DataFrame, max_cols: int = 5) -> Optional[str]:
    """Generate a Seaborn pair plot for measure columns (no IDs)."""
    from utils.column_classifier import get_measure_columns
    cols = get_measure_columns(df)
    if len(cols) < 2:
        return None
    subset = df[cols[:max_cols]]
    try:
        sns.set_theme(style="darkgrid", palette="muted")
        g = sns.pairplot(subset, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15})
        buf = io.BytesIO()
        g.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f1117")
        plt.close("all")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning("Pair plot generation failed: %s", exc)
        return None


# ── Auto-chart dispatcher ─────────────────────────────────────────────────────

def auto_generate_charts(
    df: pd.DataFrame,
    profile: DatasetProfile,
    max_charts: int = 12,
) -> list[dict[str, Any]]:
    """Automatically pick appropriate charts based on column types.

    Returns a list of dicts ``{"title": str, "figure": go.Figure}``.
    """
    charts: list[dict[str, Any]] = []

    numeric_cols = [c.name for c in profile.columns if c.detected_type == "numeric"]
    categorical_cols = [c.name for c in profile.columns if c.detected_type == "categorical"]
    datetime_cols = [c.name for c in profile.columns if c.detected_type == "datetime"]

    for col in numeric_cols[:4]:
        charts.append({"title": f"Histogram – {col}", "figure": histogram(df, col)})
        if len(charts) >= max_charts:
            break

    for col in categorical_cols[:4]:
        charts.append({"title": f"Bar Chart – {col}", "figure": bar_chart(df, col)})
        if len(charts) >= max_charts:
            break

    heatmap = correlation_heatmap(df)
    if heatmap and len(charts) < max_charts:
        charts.append({"title": "Correlation Heatmap", "figure": heatmap})

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        for val_col in numeric_cols[:2]:
            if len(charts) >= max_charts:
                break
            charts.append({
                "title": f"Time Series – {val_col}",
                "figure": time_series_chart(df, date_col, val_col),
            })

    for col in numeric_cols[:2]:
        if len(charts) >= max_charts:
            break
        charts.append({"title": f"Box Plot – {col}", "figure": box_plot(df, col)})

    logger.info("Auto-generated %d charts.", len(charts))
    return charts

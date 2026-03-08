"""Narrative Insight Engine – generates InsightForge-quality data-driven narratives.

Produces specific, numbered insights with actual dollar amounts, percentages,
growth rates, performance drivers, and weak segment analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.column_classifier import get_measure_columns, get_dimension_columns, get_id_columns


def _fmt(n: float, prefix: str = "$") -> str:
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1_000_000_000:
        return f"{sign}{prefix}{abs_n / 1e9:.1f}B"
    if abs_n >= 1_000_000:
        return f"{sign}{prefix}{abs_n / 1e6:.1f}M"
    if abs_n >= 1_000:
        return f"{sign}{prefix}{abs_n / 1e3:.1f}K"
    if abs_n == int(abs_n):
        return f"{sign}{prefix}{int(abs_n):,}"
    return f"{sign}{prefix}{abs_n:,.2f}"


def _is_monetary(col_name: str) -> bool:
    return any(k in col_name.lower() for k in
               ["revenue", "sales", "amount", "price", "total", "cost", "profit", "income", "value"])


def _prefix(col_name: str) -> str:
    return "$" if _is_monetary(col_name) else ""


# ─── KEY INSIGHTS ────────────────────────────────────────────────────────────

def generate_key_insights(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate specific, data-driven key insights like InsightForge."""
    insights = []
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    for col in measures[:3]:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        p = _prefix(col)
        total = clean.sum()
        avg = clean.mean()
        insights.append({
            "icon": "📊",
            "text": (f"Total {col} across the dataset is {_fmt(total, p)}, "
                     f"with an average of {_fmt(avg, p)} per record across {len(clean):,} entries."),
        })

        mn, mx = clean.min(), clean.max()
        if avg != 0:
            spread_pct = ((mx - mn) / abs(avg)) * 100
            insights.append({
                "icon": "📏",
                "text": (f"Values range from {_fmt(mn, p)} to {_fmt(mx, p)}, "
                         f"indicating a {spread_pct:.0f}% spread relative to the mean."),
            })

    for col in measures[:2]:
        clean = df[col].dropna()
        if len(clean) < 2:
            continue
        if datetime_cols:
            dt_col = datetime_cols[0]
            temp = df[[dt_col, col]].dropna().sort_values(dt_col)
            if len(temp) >= 2:
                first_period = temp.head(max(1, len(temp) // 4))[col].mean()
                last_period = temp.tail(max(1, len(temp) // 4))[col].mean()
                if first_period != 0:
                    growth = ((last_period - first_period) / abs(first_period)) * 100
                    first_date = temp[dt_col].iloc[0]
                    last_date = temp[dt_col].iloc[-1]
                    fd = first_date.strftime("%Y-%m") if hasattr(first_date, "strftime") else str(first_date)[:7]
                    ld = last_date.strftime("%Y-%m") if hasattr(last_date, "strftime") else str(last_date)[:7]
                    direction = "grew" if growth > 0 else "declined"
                    insights.append({
                        "icon": "📈" if growth > 0 else "📉",
                        "text": f"{col} {direction} {abs(growth):.1f}% from {fd} to {ld}.",
                    })

    for col in measures[:2]:
        clean = df[col].dropna()
        if len(clean) >= 10:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    p = _prefix(col)
                    insights.append({
                        "icon": "⚠️",
                        "text": (f"{outliers:,} outlier{'s' if outliers > 1 else ''} detected in {col} "
                                 f"(values outside {_fmt(q1 - 1.5 * iqr, p)} – {_fmt(q3 + 1.5 * iqr, p)} range)."),
                    })

    return insights


# ─── PERFORMANCE DRIVERS ─────────────────────────────────────────────────────

def generate_performance_drivers(df: pd.DataFrame) -> list[dict[str, str]]:
    """Identify top-performing segments with specific numbers."""
    drivers = []
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)

    if not measures or not dimensions:
        return drivers

    primary_measure = measures[0]
    p = _prefix(primary_measure)
    total_val = df[primary_measure].sum()
    if total_val == 0:
        return drivers

    for dim in dimensions[:2]:
        grouped = df.groupby(dim)[primary_measure].sum().sort_values(ascending=False)
        top_3 = grouped.head(3)
        for name, val in top_3.items():
            pct = val / total_val * 100
            drivers.append({
                "icon": "🚀",
                "text": f"{name} accounts for {pct:.1f}% of total {primary_measure} ({_fmt(val, p)}).",
            })
        break

    if len(dimensions) >= 2 and len(measures) >= 2:
        dim = dimensions[1]
        measure = measures[1] if len(measures) > 1 else measures[0]
        p2 = _prefix(measure)
        total_m = df[measure].sum()
        if total_m != 0:
            grouped = df.groupby(dim)[measure].sum().sort_values(ascending=False)
            top = grouped.head(2)
            for name, val in top.items():
                pct = val / total_m * 100
                drivers.append({
                    "icon": "📊",
                    "text": f"{name} leads in {measure} with {_fmt(val, p2)} ({pct:.1f}% share).",
                })
                break

    return drivers[:5]


# ─── WEAK SEGMENTS ───────────────────────────────────────────────────────────

def generate_weak_segments(df: pd.DataFrame) -> list[dict[str, str]]:
    """Identify underperforming segments with actionable language."""
    weak = []
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)

    if not measures or not dimensions:
        return weak

    primary = measures[0]
    p = _prefix(primary)
    total = df[primary].sum()
    if total == 0:
        return weak

    for dim in dimensions[:2]:
        grouped = df.groupby(dim)[primary].sum().sort_values(ascending=True)
        bottom = grouped.head(3)
        for name, val in bottom.items():
            pct = val / total * 100
            if pct < 10:
                weak.append({
                    "icon": "⚠️",
                    "text": (f"{name} contributes only {pct:.1f}% ({_fmt(val, p)}), "
                             "suggesting potential for improvement."),
                })
        break

    for col in measures[:3]:
        clean = df[col].dropna()
        if len(clean) > 0:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 5:
                weak.append({
                    "icon": "🔴",
                    "text": f"{col} has {missing_pct:.1f}% missing data, which may affect analysis accuracy.",
                })

    return weak[:4]


# ─── DATA EXPLANATION REPORT ─────────────────────────────────────────────────

def generate_data_explanation(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive 'Explain My Data' narrative report."""
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    id_cols = get_id_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    report: dict[str, Any] = {}

    overview_lines = [f"This dataset contains {len(df):,} records across {len(df.columns)} columns."]
    if datetime_cols:
        overview_lines.append(f"Date columns: {', '.join(datetime_cols)}.")
    if measures:
        overview_lines.append(f"Numeric columns: {', '.join(measures)}.")
    if dimensions:
        overview_lines.append(f"Categorical columns: {', '.join(dimensions)}.")
    if id_cols:
        overview_lines.append(f"ID/code columns (excluded from analysis): {', '.join(id_cols)}.")
    report["dataset_overview"] = overview_lines

    report["key_insights"] = generate_key_insights(df)
    report["performance_drivers"] = generate_performance_drivers(df)
    report["weak_segments"] = generate_weak_segments(df)

    correlations = []
    if len(measures) >= 2:
        corr_matrix = df[measures].corr()
        for i in range(len(measures)):
            for j in range(i + 1, len(measures)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.5:
                    direction = "positively" if val > 0 else "negatively"
                    strength = "strongly" if abs(val) > 0.7 else "moderately"
                    correlations.append({
                        "icon": "🔗",
                        "text": (f"{measures[i]} and {measures[j]} are {strength} {direction} "
                                 f"correlated ({val:.2f})."),
                    })
    report["correlations"] = correlations[:4]

    recommendations = []
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        worst = df.isna().sum().idxmax()
        worst_pct = df[worst].isna().mean() * 100
        recommendations.append({
            "icon": "🔧",
            "text": f"Address missing data in '{worst}' ({worst_pct:.1f}% missing) to improve analysis quality.",
        })
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        recommendations.append({
            "icon": "🔧",
            "text": f"Remove {dup_count:,} duplicate rows ({dup_count / len(df) * 100:.1f}% of data).",
        })
    if measures and dimensions:
        recommendations.append({
            "icon": "💡",
            "text": f"Consider segmenting {measures[0]} by {dimensions[0]} for targeted business strategies.",
        })
    report["recommendations"] = recommendations

    return report


# ─── SMART CHAT SUGGESTIONS ─────────────────────────────────────────────────

def generate_smart_suggestions(df: pd.DataFrame) -> list[str]:
    """Generate context-aware chat suggestions based on actual column names."""
    suggestions = []
    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    if dimensions and measures:
        suggestions.append(f"Which {dimensions[0]} generates the most {measures[0]}?")
        suggestions.append(f"Show {measures[0]} by {dimensions[0]}")

    if len(measures) >= 2:
        suggestions.append(f"What product sells the most {measures[1] if len(measures) > 1 else measures[0]}?")

    if datetime_cols and measures:
        suggestions.append("What is the overall growth trend?")

    if measures:
        suggestions.append(f"What is the average {measures[0]}?")

    if dimensions:
        suggestions.append(f"Show top 5 {dimensions[0]} values")

    if not suggestions:
        suggestions = [
            "How many rows are in the dataset?",
            "Show a summary of the dataset",
            "What columns are available?",
        ]

    return suggestions[:5]

"""KPI Engine – computes business metrics only on meaningful measure columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.column_classifier import get_measure_columns, get_dimension_columns


def compute_kpis(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Auto-detect and compute relevant KPIs, skipping IDs and codes."""
    kpis: list[dict[str, Any]] = []
    measure_cols = get_measure_columns(df)
    dimension_cols = get_dimension_columns(df)

    kpis.append({"name": "Total Records", "value": f"{len(df):,}",
                 "raw_value": len(df), "icon": "📋", "category": "general"})
    kpis.append({"name": "Total Columns", "value": str(len(df.columns)),
                 "raw_value": len(df.columns), "icon": "📊", "category": "general"})

    for col in measure_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        total = float(clean.sum())
        kpis.append({"name": f"Total {col}", "value": _format_number(total),
                     "raw_value": total, "icon": "💰", "category": "numeric", "column": col})
        avg = float(clean.mean())
        kpis.append({"name": f"Avg {col}", "value": _format_number(avg),
                     "raw_value": avg, "icon": "📈", "category": "numeric", "column": col})

    for col in dimension_cols[:3]:
        top = df[col].value_counts().head(1)
        if len(top) > 0:
            kpis.append({"name": f"Top {col}", "value": str(top.index[0]),
                         "raw_value": str(top.index[0]), "icon": "🏆",
                         "category": "categorical", "column": col, "count": int(top.values[0])})

    if measure_cols:
        primary = measure_cols[0]
        clean = df[primary].dropna()
        if len(clean) > 1 and clean.iloc[0] != 0:
            growth = (clean.iloc[-1] - clean.iloc[0]) / abs(clean.iloc[0]) * 100
            kpis.append({"name": f"Growth Rate ({primary})", "value": f"{growth:+.1f}%",
                         "raw_value": round(growth, 2),
                         "icon": "📉" if growth < 0 else "📈", "category": "trend"})

    return kpis


def _format_number(n: float) -> str:
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs_n >= 1_000:
        return f"{n / 1_000:.1f}K"
    if abs_n == int(abs_n):
        return f"{int(n):,}"
    return f"{n:,.2f}"


def get_top_items(df: pd.DataFrame, group_col: str, value_col: str,
                  n: int = 10, ascending: bool = False) -> pd.DataFrame:
    result = df.groupby(group_col, dropna=False)[value_col].sum().reset_index()
    result = result.sort_values(value_col, ascending=ascending).head(n)
    return result

"""Gold Layer – generates business-level analytics tables and KPIs from clean data."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_agg(df: pd.DataFrame, group_col: str, value_col: str,
              agg: str = "sum", top_n: int = 20) -> pd.DataFrame | None:
    if group_col not in df.columns or value_col not in df.columns:
        return None
    try:
        result = df.groupby(group_col, dropna=False)[value_col].agg(agg).reset_index()
        result.columns = [group_col, f"{value_col}_{agg}"]
        result = result.sort_values(f"{value_col}_{agg}", ascending=False).head(top_n)
        return result
    except Exception:
        return None


def generate_analytics_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Auto-generate business analytics tables, skipping ID/code columns."""
    from utils.column_classifier import get_measure_columns, get_dimension_columns

    tables: dict[str, pd.DataFrame] = {}
    measure_cols = get_measure_columns(df)
    dimension_cols = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    for cat_col in dimension_cols[:5]:
        for num_col in measure_cols[:5]:
            table = _safe_agg(df, cat_col, num_col, "sum")
            if table is not None and len(table) > 1:
                tables[f"{num_col}_by_{cat_col}"] = table

    for cat_col in dimension_cols[:5]:
        for num_col in measure_cols[:3]:
            table = _safe_agg(df, cat_col, num_col, "mean")
            if table is not None and len(table) > 1:
                tables[f"avg_{num_col}_by_{cat_col}"] = table

    for dt_col in datetime_cols[:2]:
        temp = df.copy()
        temp["_month"] = temp[dt_col].dt.to_period("M").astype(str)
        for num_col in measure_cols[:3]:
            table = _safe_agg(temp, "_month", num_col, "sum")
            if table is not None and len(table) > 1:
                table = table.rename(columns={"_month": "month"})
                tables[f"monthly_{num_col}"] = table

    if measure_cols:
        summary_data = {}
        for col in measure_cols:
            summary_data[col] = {
                "total": float(df[col].sum()),
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "std": round(float(df[col].std()), 2),
            }
        tables["numeric_summary"] = pd.DataFrame(summary_data).T.reset_index()
        tables["numeric_summary"].columns = ["metric"] + list(tables["numeric_summary"].columns[1:])

    return tables


def store_gold_tables(tables: dict[str, pd.DataFrame], dataset_name: str) -> list[Path]:
    """Persist gold-layer analytics tables to disk."""
    paths = []
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_dir = GOLD_DIR / dataset_name
    ds_dir.mkdir(exist_ok=True)
    for name, table_df in tables.items():
        filepath = ds_dir / f"{name}__{timestamp}.csv"
        table_df.to_csv(filepath, index=False)
        paths.append(filepath)
    return paths

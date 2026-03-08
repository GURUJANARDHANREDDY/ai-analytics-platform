"""Transformation Pipeline – modular data transformations with lineage tracking."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import numpy as np
import pandas as pd

from lineage.lineage_tracker import LineageTracker


def add_derived_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add common derived columns based on dataset content."""
    added = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) >= 2:
        df["_row_total"] = df[numeric_cols].sum(axis=1)
        added.append("_row_total")
        df["_row_mean"] = df[numeric_cols].mean(axis=1)
        added.append("_row_mean")

    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        added.extend([f"{col}_year", f"{col}_month", f"{col}_dayofweek"])

    return df, added


def compute_business_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute common business metrics from the dataset."""
    metrics = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        metrics[col] = {
            "total": float(clean.sum()),
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4),
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

    return metrics


def aggregate_by_column(df: pd.DataFrame, group_col: str, agg_col: str,
                        agg_func: str = "sum") -> pd.DataFrame:
    """Aggregate data by a grouping column."""
    valid_funcs = {"sum", "mean", "count", "min", "max", "median"}
    if agg_func not in valid_funcs:
        raise ValueError(f"Invalid aggregation: {agg_func}. Use: {valid_funcs}")
    return df.groupby(group_col, dropna=False)[agg_col].agg(agg_func).reset_index()


def run_transform_pipeline(
    df: pd.DataFrame, dataset_name: str, tracker: LineageTracker | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full transformation pipeline."""
    log: dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(),
        "input_shape": list(df.shape),
        "steps": [],
    }

    df, derived = add_derived_columns(df)
    if derived:
        log["steps"].append({"operation": "add_derived_columns", "columns_added": derived})

    metrics = compute_business_metrics(df)
    log["steps"].append({"operation": "compute_business_metrics", "metrics_count": len(metrics)})
    log["business_metrics"] = metrics

    log["output_shape"] = list(df.shape)

    if tracker:
        tracker.add_event("feature_engineering", {"derived_columns": derived})

    return df, log

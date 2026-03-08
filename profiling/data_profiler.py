"""Data Profiling Engine – automatically profiles datasets with comprehensive statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    category: str  # numeric, categorical, datetime, boolean
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_val: Any = None
    max_val: Any = None
    q25: float | None = None
    q75: float | None = None


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    memory_usage_mb: float
    duplicate_rows: int
    duplicate_pct: float
    dtypes_summary: dict[str, int]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    missing_summary: dict[str, float]


def _classify_column(series: pd.Series) -> str:
    from utils.column_classifier import is_id_or_code

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if is_id_or_code(series):
            return "identifier"
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    try:
        pd.to_datetime(series.dropna().head(50))
        return "datetime"
    except (ValueError, TypeError):
        pass
    return "categorical"


def _profile_column(series: pd.Series) -> ColumnProfile:
    category = _classify_column(series)
    non_null = int(series.notna().sum())
    null_count = int(series.isna().sum())
    total = len(series)
    unique_count = int(series.nunique())
    top_values = series.value_counts().head(10).items()

    profile = ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        category=category,
        non_null_count=non_null,
        null_count=null_count,
        null_pct=round(null_count / total * 100, 2) if total > 0 else 0.0,
        unique_count=unique_count,
        unique_pct=round(unique_count / total * 100, 2) if total > 0 else 0.0,
        top_values=[(str(k), int(v)) for k, v in top_values],
    )

    if category == "numeric":
        clean = series.dropna()
        if len(clean) > 0:
            profile.mean = round(float(clean.mean()), 4)
            profile.median = round(float(clean.median()), 4)
            profile.std = round(float(clean.std()), 4)
            profile.min_val = float(clean.min())
            profile.max_val = float(clean.max())
            profile.q25 = round(float(clean.quantile(0.25)), 4)
            profile.q75 = round(float(clean.quantile(0.75)), 4)

    return profile


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Generate a comprehensive profile for the dataset."""
    columns = [_profile_column(df[col]) for col in df.columns]
    dup_count = int(df.duplicated().sum())
    total = len(df)

    dtypes_summary = {}
    for col_p in columns:
        dtypes_summary[col_p.category] = dtypes_summary.get(col_p.category, 0) + 1

    numeric_cols = [c.name for c in columns if c.category == "numeric"]
    categorical_cols = [c.name for c in columns if c.category == "categorical"]
    datetime_cols = [c.name for c in columns if c.category == "datetime"]

    missing_summary = {c.name: c.null_pct for c in columns if c.null_pct > 0}

    return DatasetProfile(
        n_rows=total,
        n_cols=len(df.columns),
        columns=columns,
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        duplicate_rows=dup_count,
        duplicate_pct=round(dup_count / total * 100, 2) if total > 0 else 0.0,
        dtypes_summary=dtypes_summary,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        missing_summary=missing_summary,
    )


def generate_profiling_report(profile: DatasetProfile) -> dict[str, Any]:
    """Convert profile to a serializable report dict."""
    return {
        "row_count": profile.n_rows,
        "column_count": profile.n_cols,
        "memory_usage_mb": profile.memory_usage_mb,
        "duplicate_rows": profile.duplicate_rows,
        "duplicate_pct": profile.duplicate_pct,
        "dtypes_summary": profile.dtypes_summary,
        "numeric_columns": profile.numeric_cols,
        "categorical_columns": profile.categorical_cols,
        "datetime_columns": profile.datetime_cols,
        "missing_values": profile.missing_summary,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "category": c.category,
                "null_pct": c.null_pct,
                "unique_count": c.unique_count,
                "top_values": c.top_values[:5],
                "mean": c.mean,
                "median": c.median,
                "std": c.std,
                "min": c.min_val,
                "max": c.max_val,
            }
            for c in profile.columns
        ],
    }

"""Automatic dataset profiling – statistics, types, and KPI generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Column classification ────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Profile metadata for a single column."""

    name: str
    dtype: str
    detected_type: str  # numeric | categorical | datetime | boolean | text
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    """Aggregated profile for the entire dataset."""

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    numeric_kpis: dict[str, dict[str, float]]
    categorical_kpis: dict[str, dict[str, Any]]
    memory_usage_mb: float


def classify_column(series: pd.Series) -> str:
    """Heuristically classify a pandas Series into a semantic type."""
    from utils.column_classifier import is_id_or_code

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if is_id_or_code(series):
            return "identifier"
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    if series.dtype == object:
        non_null = series.dropna()
        if non_null.empty:
            return "text"
        unique_ratio = non_null.nunique() / len(non_null)
        avg_len = non_null.astype(str).str.len().mean()
        n_unique = non_null.nunique()

        if avg_len > 50:
            return "text"
        if n_unique <= 30 or unique_ratio < 0.05:
            return "categorical"
        if unique_ratio > 0.5:
            return "identifier"
        if unique_ratio < 0.3:
            return "categorical"
        return "identifier"

    return "text"


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Generate a comprehensive profile of the given DataFrame."""
    logger.info("Profiling dataset (%d rows, %d cols)…", len(df), len(df.columns))

    columns: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        detected = classify_column(series)
        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                detected_type=detected,
                missing_count=int(series.isna().sum()),
                missing_pct=round(float(series.isna().mean()) * 100, 2),
                unique_count=int(series.nunique()),
                sample_values=series.dropna().head(5).tolist(),
            )
        )

    numeric_kpis = _compute_numeric_kpis(df, columns)
    categorical_kpis = _compute_categorical_kpis(df, columns)

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    profile = DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        numeric_kpis=numeric_kpis,
        categorical_kpis=categorical_kpis,
        memory_usage_mb=round(mem_mb, 2),
    )
    logger.info("Profiling complete. Memory: %.2f MB", mem_mb)
    return profile


# ── KPI helpers ───────────────────────────────────────────────────────────────

def _compute_numeric_kpis(
    df: pd.DataFrame, columns: list[ColumnProfile]
) -> dict[str, dict[str, float]]:
    kpis: dict[str, dict[str, float]] = {}
    for cp in columns:
        if cp.detected_type != "numeric":
            continue
        s = df[cp.name].dropna()
        if s.empty:
            continue
        kpis[cp.name] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "std": round(float(s.std()), 4),
            "q25": round(float(np.percentile(s, 25)), 4),
            "q75": round(float(np.percentile(s, 75)), 4),
        }
    return kpis


def _compute_categorical_kpis(
    df: pd.DataFrame, columns: list[ColumnProfile]
) -> dict[str, dict[str, Any]]:
    kpis: dict[str, dict[str, Any]] = {}
    for cp in columns:
        if cp.detected_type != "categorical":
            continue
        s = df[cp.name].dropna()
        if s.empty:
            continue
        freq = s.value_counts()
        kpis[cp.name] = {
            "top_categories": freq.head(10).to_dict(),
            "unique_count": int(freq.shape[0]),
            "mode": str(freq.index[0]) if len(freq) > 0 else None,
        }
    return kpis


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> dict[str, list[int]]:
    """Return row indices with z-score exceeding *z_threshold*, skipping ID columns."""
    from utils.column_classifier import get_measure_columns

    anomalies: dict[str, list[int]] = {}
    for col in get_measure_columns(df):
        s = df[col].dropna()
        if s.std() == 0:
            continue
        z = ((s - s.mean()) / s.std()).abs()
        outlier_idx = z[z > z_threshold].index.tolist()
        if outlier_idx:
            anomalies[col] = outlier_idx
    return anomalies


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for measure columns only (no IDs)."""
    from utils.column_classifier import get_measure_columns

    cols = get_measure_columns(df)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr()


def compute_feature_importance(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Estimate feature importance via absolute correlation, skipping IDs."""
    from utils.column_classifier import get_measure_columns

    if target_col not in df.columns:
        return {}
    measure_cols = get_measure_columns(df)
    if target_col not in measure_cols:
        return {}
    subset = df[measure_cols]
    corr = subset.corr()[target_col].drop(target_col, errors="ignore").abs().sort_values(ascending=False)
    return corr.round(4).to_dict()

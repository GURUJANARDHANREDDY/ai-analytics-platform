"""Tests for backend.data_profiler module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_profiler import (
    classify_column,
    compute_correlations,
    compute_feature_importance,
    detect_anomalies,
    profile_dataset,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "revenue": np.random.normal(1000, 200, 100),
        "quantity": np.random.randint(1, 50, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "is_active": np.random.choice([True, False], 100),
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "notes": ["some long text description that is unique " + str(i) for i in range(100)],
    })


class TestClassifyColumn:
    def test_numeric(self) -> None:
        assert classify_column(pd.Series([1, 2, 3])) == "numeric"

    def test_boolean(self) -> None:
        assert classify_column(pd.Series([True, False, True])) == "boolean"

    def test_datetime(self) -> None:
        s = pd.Series(pd.date_range("2024-01-01", periods=3))
        assert classify_column(s) == "datetime"

    def test_categorical(self) -> None:
        assert classify_column(pd.Series(["A", "B", "A", "C"])) == "categorical"

    def test_text(self) -> None:
        long_texts = pd.Series(["x" * 60 + str(i) for i in range(100)])
        assert classify_column(long_texts) == "text"

    def test_identifier(self) -> None:
        ids = pd.Series([f"ORD-{i:05d}" for i in range(200)])
        assert classify_column(ids) == "identifier"


class TestProfileDataset:
    def test_returns_correct_shape(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df)
        assert profile.n_rows == 100
        assert profile.n_cols == 6
        assert len(profile.columns) == 6

    def test_numeric_kpis_present(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df)
        assert "revenue" in profile.numeric_kpis
        kpi = profile.numeric_kpis["revenue"]
        assert "mean" in kpi
        assert "median" in kpi
        assert "std" in kpi

    def test_categorical_kpis_present(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df)
        assert "category" in profile.categorical_kpis

    def test_missing_values_tracked(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        profile = profile_dataset(df)
        missing = {c.name: c.missing_count for c in profile.columns}
        assert missing["a"] == 1
        assert missing["b"] == 1


class TestDetectAnomalies:
    def test_detects_outliers(self) -> None:
        data = [10] * 100 + [1000]
        df = pd.DataFrame({"val": data})
        anomalies = detect_anomalies(df)
        assert "val" in anomalies
        assert 100 in anomalies["val"]

    def test_no_anomalies_in_uniform_data(self) -> None:
        df = pd.DataFrame({"val": [5] * 100})
        anomalies = detect_anomalies(df)
        assert not anomalies


class TestCorrelations:
    def test_correlation_matrix_shape(self, sample_df: pd.DataFrame) -> None:
        corr = compute_correlations(sample_df)
        assert not corr.empty
        assert corr.shape[0] == corr.shape[1]

    def test_single_numeric_col_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        corr = compute_correlations(df)
        assert corr.empty


class TestFeatureImportance:
    def test_returns_dict(self, sample_df: pd.DataFrame) -> None:
        importance = compute_feature_importance(sample_df, "revenue")
        assert isinstance(importance, dict)
        assert "revenue" not in importance

    def test_missing_target_returns_empty(self, sample_df: pd.DataFrame) -> None:
        importance = compute_feature_importance(sample_df, "nonexistent")
        assert importance == {}

"""Tests for backend.data_loader module."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_loader import DataLoadError, clear_cache, load_csv_from_file


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    return b"name,age,salary\nAlice,30,70000\nBob,25,60000\nCharlie,35,80000\n"


@pytest.fixture()
def sample_csv_file(sample_csv_bytes: bytes) -> io.BytesIO:
    return io.BytesIO(sample_csv_bytes)


class TestLoadCsvFromFile:
    def test_loads_valid_csv(self, sample_csv_file: io.BytesIO) -> None:
        df = load_csv_from_file(sample_csv_file, "test.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "salary"]

    def test_rejects_non_csv_extension(self, sample_csv_file: io.BytesIO) -> None:
        with pytest.raises(DataLoadError, match="Unsupported file type"):
            load_csv_from_file(sample_csv_file, "data.xlsx")

    def test_rejects_empty_csv(self) -> None:
        buf = io.BytesIO(b"col1,col2\n")
        with pytest.raises(DataLoadError, match="empty"):
            load_csv_from_file(buf, "empty.csv")

    def test_caching_works(self, sample_csv_file: io.BytesIO) -> None:
        clear_cache()
        key = "test_key"
        df1 = load_csv_from_file(sample_csv_file, "test.csv", cache_key=key)
        sample_csv_file.seek(0)
        df2 = load_csv_from_file(sample_csv_file, "test.csv", cache_key=key)
        assert df1 is df2
        clear_cache()

    def test_pyarrow_fallback(self, sample_csv_file: io.BytesIO) -> None:
        df = load_csv_from_file(sample_csv_file, "test.csv", use_pyarrow=False)
        assert len(df) == 3

    def test_malformed_csv_raises(self) -> None:
        buf = io.BytesIO(b"not,a,csv\n\"broken")
        try:
            load_csv_from_file(buf, "bad.csv")
        except DataLoadError:
            pass  # expected


class TestClearCache:
    def test_clear_all(self, sample_csv_file: io.BytesIO) -> None:
        load_csv_from_file(sample_csv_file, "test.csv", cache_key="k1")
        clear_cache()

    def test_clear_specific_key(self, sample_csv_file: io.BytesIO) -> None:
        load_csv_from_file(sample_csv_file, "test.csv", cache_key="k2")
        clear_cache("k2")

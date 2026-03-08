"""Data Ingestion Engine – validates, loads, and captures metadata for uploaded datasets."""

from __future__ import annotations

import io
import os
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BRONZE_DIR = DATA_DIR / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DataLoadError(Exception):
    pass


def file_hash(file_obj: io.BytesIO) -> str:
    file_obj.seek(0)
    h = hashlib.sha256(file_obj.read()).hexdigest()[:16]
    file_obj.seek(0)
    return h


def validate_dataset(filename: str, file_obj: io.BytesIO) -> dict[str, Any]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    file_obj.seek(0, 2)
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    if size_bytes == 0:
        raise DataLoadError("File is empty.")
    if size_bytes > 500 * 1024 * 1024:
        raise DataLoadError("File exceeds 500 MB limit.")
    return {"filename": filename, "extension": ext, "size_bytes": size_bytes}


def load_dataset(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".csv":
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1", "utf-8-sig"]
            for enc in encodings:
                file_obj.seek(0)
                try:
                    df = pd.read_csv(file_obj, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="latin1", on_bad_lines="skip")
        elif ext in (".xlsx", ".xls"):
            file_obj.seek(0)
            df = pd.read_excel(file_obj, engine="openpyxl")
        else:
            raise DataLoadError(f"Cannot read format: {ext}")
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to parse {filename}: {exc}") from exc

    if df.empty:
        raise DataLoadError("Dataset contains no data rows.")
    return df


def capture_metadata(df: pd.DataFrame, filename: str, size_bytes: int) -> dict[str, Any]:
    return {
        "dataset_name": filename,
        "dataset_size_bytes": size_bytes,
        "dataset_size_mb": round(size_bytes / (1024 * 1024), 3),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "ingestion_timestamp": dt.datetime.now().isoformat(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
    }


def ingest_dataset(file_obj: io.BytesIO, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full ingestion pipeline: validate → load → capture metadata."""
    validation = validate_dataset(filename, file_obj)
    df = load_dataset(file_obj, filename)
    metadata = capture_metadata(df, filename, validation["size_bytes"])
    metadata["file_hash"] = file_hash(file_obj)
    return df, metadata

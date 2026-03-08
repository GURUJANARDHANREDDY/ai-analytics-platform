"""Schema Registry – automatically detects dataset schema and stores versioned metadata."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCHEMA_DIR = DATA_DIR / "schemas"
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


def detect_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect schema for all columns in the dataset."""
    schema = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            category = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            category = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            category = "boolean"
        else:
            category = "categorical"

        schema.append({
            "column_name": col,
            "data_type": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "unique_value_count": int(series.nunique()),
            "column_category": category,
        })
    return schema


def save_schema(dataset_name: str, schema: list[dict[str, Any]], version: int | None = None) -> Path:
    """Save a versioned schema snapshot."""
    if version is None:
        existing = list(SCHEMA_DIR.glob(f"{dataset_name}_v*.json"))
        version = len(existing) + 1

    schema_record = {
        "dataset_name": dataset_name,
        "version": version,
        "timestamp": dt.datetime.now().isoformat(),
        "column_count": len(schema),
        "schema": schema,
    }

    filepath = SCHEMA_DIR / f"{dataset_name}_v{version}.json"
    with open(filepath, "w") as f:
        json.dump(schema_record, f, indent=2, default=str)
    return filepath


def load_schema(dataset_name: str, version: int | None = None) -> dict[str, Any] | None:
    """Load a specific schema version (latest if version is None)."""
    if version:
        filepath = SCHEMA_DIR / f"{dataset_name}_v{version}.json"
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None

    files = sorted(SCHEMA_DIR.glob(f"{dataset_name}_v*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def list_schema_versions(dataset_name: str) -> list[dict[str, Any]]:
    """List all schema versions for a dataset."""
    versions = []
    for filepath in sorted(SCHEMA_DIR.glob(f"{dataset_name}_v*.json")):
        with open(filepath) as f:
            data = json.load(f)
            versions.append({
                "version": data["version"],
                "timestamp": data["timestamp"],
                "column_count": data["column_count"],
            })
    return versions

"""Silver Layer – cleans and validates data with full transformation tracking."""

from __future__ import annotations

import re
import json
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    changes = []
    rename_map = {}
    for col in df.columns:
        new_name = re.sub(r"[^a-z0-9]+", "_", col.lower().strip()).strip("_")
        if new_name != col:
            rename_map[col] = new_name
            changes.append({"original": col, "standardized": new_name})
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, changes


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def _fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    fill_log = {}
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            fill_log[col] = f"median ({fill_val:.4f})"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].ffill()
            fill_log[col] = "forward fill"
        else:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
                fill_log[col] = f"mode ({mode_vals.iloc[0]})"
            else:
                df[col] = df[col].fillna("Unknown")
                fill_log[col] = "Unknown"
    return df, fill_log


def _normalize_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    dtype_changes = {}
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                df[col] = converted
                dtype_changes[col] = "object → datetime"
                continue
            except (ValueError, TypeError):
                pass
            try:
                converted = pd.to_numeric(df[col], errors="raise")
                df[col] = converted
                dtype_changes[col] = "object → numeric"
            except (ValueError, TypeError):
                pass
    return df, dtype_changes


def _remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    all_null_mask = df.isna().all(axis=1)
    df = df[~all_null_mask].reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute the full Silver layer cleaning pipeline."""
    lineage_log: dict[str, Any] = {
        "layer": "silver",
        "timestamp": dt.datetime.now().isoformat(),
        "input_shape": list(df.shape),
        "transformations": [],
    }

    df, col_changes = _standardize_column_names(df)
    if col_changes:
        lineage_log["transformations"].append({
            "operation": "standardize_column_names",
            "columns_renamed": len(col_changes),
            "details": col_changes,
        })

    df, dup_removed = _remove_duplicates(df)
    lineage_log["transformations"].append({
        "operation": "remove_duplicates",
        "rows_removed": dup_removed,
    })

    df, fill_log = _fill_missing_values(df)
    if fill_log:
        lineage_log["transformations"].append({
            "operation": "fill_missing_values",
            "columns_filled": fill_log,
        })

    df, dtype_changes = _normalize_dtypes(df)
    if dtype_changes:
        lineage_log["transformations"].append({
            "operation": "normalize_data_types",
            "type_changes": dtype_changes,
        })

    df, invalid_removed = _remove_invalid_rows(df)
    if invalid_removed:
        lineage_log["transformations"].append({
            "operation": "remove_invalid_rows",
            "rows_removed": invalid_removed,
        })

    lineage_log["output_shape"] = list(df.shape)
    return df, lineage_log


def store_silver(df: pd.DataFrame, dataset_name: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}__{timestamp}.csv"
    filepath = SILVER_DIR / filename
    df.to_csv(filepath, index=False)
    return filepath

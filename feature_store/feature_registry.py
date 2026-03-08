"""Feature Store – registry for derived metrics that are reusable across analytics."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURE_DIR = DATA_DIR / "feature_store"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = FEATURE_DIR / "feature_registry.json"


def _load_registry() -> dict[str, Any]:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"features": {}}


def _save_registry(registry: dict[str, Any]):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute derived features, skipping ID/code columns."""
    from utils.column_classifier import get_measure_columns

    features_added: list[dict[str, Any]] = []
    measure_cols = get_measure_columns(df)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    revenue_cols = [c for c in measure_cols if any(k in c.lower() for k in ["revenue", "sales", "amount", "total", "price"])]
    quantity_cols = [c for c in measure_cols if any(k in c.lower() for k in ["quantity", "qty", "count", "units"])]
    customer_cols = [c for c in categorical_cols if any(k in c.lower() for k in ["customer", "client", "user"])]

    if revenue_cols and customer_cols:
        rev_col = revenue_cols[0]
        cust_col = customer_cols[0]
        clv = df.groupby(cust_col)[rev_col].sum().reset_index()
        clv.columns = [cust_col, "customer_lifetime_value"]
        df = df.merge(clv, on=cust_col, how="left")
        features_added.append({
            "name": "customer_lifetime_value",
            "description": f"Total {rev_col} per {cust_col}",
            "source_columns": [rev_col, cust_col],
            "type": "aggregation",
        })

    if revenue_cols and categorical_cols:
        rev_col = revenue_cols[0]
        total_rev = df[rev_col].sum()
        if total_rev > 0:
            for cat in categorical_cols[:2]:
                col_name = f"{cat}_share_pct"
                cat_totals = df.groupby(cat)[rev_col].sum()
                share_map = (cat_totals / total_rev * 100).to_dict()
                df[col_name] = df[cat].map(share_map)
                features_added.append({
                    "name": col_name,
                    "description": f"Percentage share of {rev_col} by {cat}",
                    "source_columns": [rev_col, cat],
                    "type": "share",
                })

    if quantity_cols and customer_cols:
        qty_col = quantity_cols[0]
        cust_col = customer_cols[0]
        freq = df.groupby(cust_col)[qty_col].count().reset_index()
        freq.columns = [cust_col, "purchase_frequency"]
        df = df.merge(freq, on=cust_col, how="left", suffixes=("", "_freq"))
        features_added.append({
            "name": "purchase_frequency",
            "description": f"Number of transactions per {cust_col}",
            "source_columns": [qty_col, cust_col],
            "type": "frequency",
        })

    for col in measure_cols[:5]:
        clean = df[col].dropna()
        if len(clean) > 0 and clean.std() > 0:
            z_col = f"{col}_zscore"
            df[z_col] = (df[col] - clean.mean()) / clean.std()
            features_added.append({
                "name": z_col,
                "description": f"Z-score normalized {col}",
                "source_columns": [col],
                "type": "normalization",
            })

    return df, features_added


def register_features(features: list[dict[str, Any]], dataset_name: str):
    """Register computed features in the feature store."""
    registry = _load_registry()
    for feat in features:
        feat["dataset"] = dataset_name
        feat["registered_at"] = dt.datetime.now().isoformat()
        registry["features"][feat["name"]] = feat
    _save_registry(registry)


def list_features() -> list[dict[str, Any]]:
    registry = _load_registry()
    return list(registry["features"].values())


def get_feature(name: str) -> dict[str, Any] | None:
    registry = _load_registry()
    return registry["features"].get(name)

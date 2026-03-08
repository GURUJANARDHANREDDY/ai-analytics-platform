"""Bronze Layer – stores raw datasets exactly as uploaded with ingestion metadata."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BRONZE_DIR = DATA_DIR / "bronze"
METADATA_DIR = DATA_DIR / "metadata"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def store_bronze(df: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    """Store raw dataset in bronze layer without any modifications."""
    dataset_name = Path(metadata["dataset_name"]).stem
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}__{timestamp}.csv"
    filepath = BRONZE_DIR / filename
    df.to_csv(filepath, index=False)

    bronze_meta = {
        **metadata,
        "layer": "bronze",
        "bronze_path": str(filepath),
        "storage_timestamp": dt.datetime.now().isoformat(),
        "modifications": "none – raw data preserved as-is",
    }
    meta_path = METADATA_DIR / f"bronze_{dataset_name}__{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(bronze_meta, f, indent=2, default=str)

    return filepath


def load_bronze(filepath: str | Path) -> pd.DataFrame:
    """Load a dataset from bronze storage."""
    return pd.read_csv(filepath)


def list_bronze_datasets() -> list[dict[str, Any]]:
    """List all datasets in the bronze layer."""
    datasets = []
    for meta_file in sorted(METADATA_DIR.glob("bronze_*.json"), reverse=True):
        with open(meta_file) as f:
            datasets.append(json.load(f))
    return datasets

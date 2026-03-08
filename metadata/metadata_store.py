"""Metadata Catalog – central registry for all dataset metadata, profiles, and lineage."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_FILE = METADATA_DIR / "catalog.json"


def _load_catalog() -> dict[str, Any]:
    if CATALOG_FILE.exists():
        with open(CATALOG_FILE) as f:
            return json.load(f)
    return {"datasets": {}}


def _save_catalog(catalog: dict[str, Any]):
    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2, default=str)


def register_dataset(
    dataset_name: str,
    schema: list[dict] | None = None,
    profiling_report: dict | None = None,
    quality_report: dict | None = None,
    lineage_info: list[dict] | None = None,
    owner: str = "system",
    description: str = "",
) -> dict[str, Any]:
    """Register or update a dataset in the metadata catalog."""
    catalog = _load_catalog()
    now = dt.datetime.now().isoformat()

    entry = catalog["datasets"].get(dataset_name, {
        "dataset_name": dataset_name,
        "created_at": now,
        "owner": owner,
    })

    entry["updated_at"] = now
    entry["description"] = description or entry.get("description", "")
    entry["owner"] = owner or entry.get("owner", "system")

    if schema is not None:
        entry["schema"] = schema
    if profiling_report is not None:
        entry["profiling_report"] = profiling_report
    if quality_report is not None:
        entry["quality_report"] = quality_report
    if lineage_info is not None:
        entry["lineage"] = lineage_info

    catalog["datasets"][dataset_name] = entry
    _save_catalog(catalog)
    return entry


def get_dataset_metadata(dataset_name: str) -> dict[str, Any] | None:
    catalog = _load_catalog()
    return catalog["datasets"].get(dataset_name)


def list_datasets() -> list[dict[str, Any]]:
    catalog = _load_catalog()
    return [
        {"name": name, "owner": meta.get("owner", ""), "updated_at": meta.get("updated_at", "")}
        for name, meta in catalog["datasets"].items()
    ]


def search_catalog(query: str) -> list[dict[str, Any]]:
    catalog = _load_catalog()
    results = []
    query_lower = query.lower()
    for name, meta in catalog["datasets"].items():
        if (query_lower in name.lower() or
                query_lower in meta.get("description", "").lower() or
                query_lower in meta.get("owner", "").lower()):
            results.append(meta)
    return results


def delete_dataset(dataset_name: str) -> bool:
    catalog = _load_catalog()
    if dataset_name in catalog["datasets"]:
        del catalog["datasets"][dataset_name]
        _save_catalog(catalog)
        return True
    return False

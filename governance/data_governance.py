"""Data Governance Layer – tracks ownership, classification, and access logs."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOV_DIR = DATA_DIR / "governance"
GOV_DIR.mkdir(parents=True, exist_ok=True)

GOVERNANCE_FILE = GOV_DIR / "governance_registry.json"
ACCESS_LOG_FILE = GOV_DIR / "access_log.json"


def _load_governance() -> dict[str, Any]:
    if GOVERNANCE_FILE.exists():
        with open(GOVERNANCE_FILE) as f:
            return json.load(f)
    return {"datasets": {}}


def _save_governance(data: dict[str, Any]):
    with open(GOVERNANCE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_access_log() -> list[dict[str, Any]]:
    if ACCESS_LOG_FILE.exists():
        with open(ACCESS_LOG_FILE) as f:
            return json.load(f)
    return []


def _save_access_log(log: list[dict[str, Any]]):
    with open(ACCESS_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)


def register_governance(
    dataset_name: str,
    owner: str = "system",
    description: str = "",
    classification: str = "internal",
    schema_version: int = 1,
) -> dict[str, Any]:
    """Register governance metadata for a dataset."""
    gov = _load_governance()
    now = dt.datetime.now().isoformat()

    entry = gov["datasets"].get(dataset_name, {
        "dataset_name": dataset_name,
        "created_at": now,
        "schema_history": [],
    })

    entry.update({
        "owner": owner,
        "description": description,
        "classification": classification,
        "updated_at": now,
        "current_schema_version": schema_version,
    })

    if schema_version not in [s.get("version") for s in entry.get("schema_history", [])]:
        entry.setdefault("schema_history", []).append({
            "version": schema_version,
            "timestamp": now,
        })

    gov["datasets"][dataset_name] = entry
    _save_governance(gov)
    return entry


def log_access(dataset_name: str, action: str, user: str = "system"):
    """Log a data access event."""
    log = _load_access_log()
    log.append({
        "dataset_name": dataset_name,
        "action": action,
        "user": user,
        "timestamp": dt.datetime.now().isoformat(),
    })
    if len(log) > 10000:
        log = log[-5000:]
    _save_access_log(log)


def get_governance_info(dataset_name: str) -> dict[str, Any] | None:
    gov = _load_governance()
    return gov["datasets"].get(dataset_name)


def get_access_log(dataset_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    log = _load_access_log()
    if dataset_name:
        log = [e for e in log if e["dataset_name"] == dataset_name]
    return log[-limit:]


def list_governed_datasets() -> list[dict[str, Any]]:
    gov = _load_governance()
    return [
        {
            "name": name,
            "owner": meta.get("owner", ""),
            "classification": meta.get("classification", ""),
            "updated_at": meta.get("updated_at", ""),
        }
        for name, meta in gov["datasets"].items()
    ]

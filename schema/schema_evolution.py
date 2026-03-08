"""Schema Evolution Engine – detects and reports changes across dataset schema versions."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from schema.schema_registry import load_schema, SCHEMA_DIR


def detect_schema_changes(dataset_name: str, old_version: int, new_version: int) -> dict[str, Any]:
    """Compare two schema versions and detect changes."""
    old_schema = load_schema(dataset_name, old_version)
    new_schema = load_schema(dataset_name, new_version)

    if not old_schema or not new_schema:
        return {"error": "One or both schema versions not found"}

    old_cols = {col["column_name"]: col for col in old_schema["schema"]}
    new_cols = {col["column_name"]: col for col in new_schema["schema"]}

    old_names = set(old_cols.keys())
    new_names = set(new_cols.keys())

    added = new_names - old_names
    removed = old_names - new_names
    common = old_names & new_names

    dtype_changes = []
    for col in common:
        if old_cols[col]["data_type"] != new_cols[col]["data_type"]:
            dtype_changes.append({
                "column": col,
                "old_type": old_cols[col]["data_type"],
                "new_type": new_cols[col]["data_type"],
            })

    possible_renames = []
    if added and removed:
        for old_col in removed:
            for new_col in added:
                if old_cols[old_col]["data_type"] == new_cols[new_col]["data_type"]:
                    possible_renames.append({
                        "old_name": old_col,
                        "new_name": new_col,
                        "shared_type": old_cols[old_col]["data_type"],
                    })

    report = {
        "dataset_name": dataset_name,
        "old_version": old_version,
        "new_version": new_version,
        "timestamp": dt.datetime.now().isoformat(),
        "has_changes": bool(added or removed or dtype_changes),
        "new_columns": sorted(added),
        "removed_columns": sorted(removed),
        "datatype_changes": dtype_changes,
        "possible_renames": possible_renames,
        "summary": _build_summary(added, removed, dtype_changes, possible_renames),
    }
    return report


def _build_summary(added: set, removed: set, dtype_changes: list, possible_renames: list) -> str:
    parts = []
    if added:
        parts.append(f"{len(added)} new column(s) added")
    if removed:
        parts.append(f"{len(removed)} column(s) removed")
    if dtype_changes:
        parts.append(f"{len(dtype_changes)} datatype change(s)")
    if possible_renames:
        parts.append(f"{len(possible_renames)} possible rename(s) detected")
    return "; ".join(parts) if parts else "No schema changes detected"


def get_evolution_history(dataset_name: str) -> list[dict[str, Any]]:
    """Generate evolution history across all consecutive versions."""
    from schema.schema_registry import list_schema_versions
    versions = list_schema_versions(dataset_name)
    if len(versions) < 2:
        return []

    history = []
    for i in range(len(versions) - 1):
        change = detect_schema_changes(
            dataset_name, versions[i]["version"], versions[i + 1]["version"]
        )
        if change.get("has_changes"):
            history.append(change)
    return history

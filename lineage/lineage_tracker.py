"""Data Lineage System – tracks data flow across the entire pipeline and visualizes it."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LINEAGE_DIR = DATA_DIR / "lineage"
LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_STAGES = [
    {"id": "raw_dataset", "label": "Raw Dataset", "icon": "📁"},
    {"id": "bronze_storage", "label": "Bronze Layer", "icon": "🥉"},
    {"id": "data_profiling", "label": "Data Profiling", "icon": "🔍"},
    {"id": "data_quality", "label": "Quality Validation", "icon": "✅"},
    {"id": "data_cleaning", "label": "Silver Layer (Cleaning)", "icon": "🥈"},
    {"id": "feature_engineering", "label": "Feature Engineering", "icon": "⚙️"},
    {"id": "kpi_aggregation", "label": "Gold Layer (KPIs)", "icon": "🥇"},
    {"id": "dashboard", "label": "Dashboard Visualization", "icon": "📊"},
    {"id": "ai_insights", "label": "AI Insights", "icon": "🤖"},
]


class LineageTracker:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.events: list[dict[str, Any]] = []
        self._load_existing()

    def _lineage_path(self) -> Path:
        return LINEAGE_DIR / f"{self.dataset_name}_lineage.json"

    def _load_existing(self):
        path = self._lineage_path()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.events = data.get("events", [])

    def _save(self):
        record = {
            "dataset_name": self.dataset_name,
            "last_updated": dt.datetime.now().isoformat(),
            "total_events": len(self.events),
            "events": self.events,
        }
        with open(self._lineage_path(), "w") as f:
            json.dump(record, f, indent=2, default=str)

    def add_event(self, stage_id: str, details: dict[str, Any] | None = None):
        event = {
            "stage_id": stage_id,
            "timestamp": dt.datetime.now().isoformat(),
            "details": details or {},
        }
        stage_info = next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)
        if stage_info:
            event["stage_label"] = stage_info["label"]
            event["stage_icon"] = stage_info["icon"]
        self.events.append(event)
        self._save()

    def get_lineage(self) -> list[dict[str, Any]]:
        return self.events

    def get_lineage_graph_data(self) -> dict[str, Any]:
        """Return structured data for lineage graph visualization."""
        completed_stages = [e["stage_id"] for e in self.events]
        nodes = []
        edges = []
        for i, stage in enumerate(PIPELINE_STAGES):
            nodes.append({
                "id": stage["id"],
                "label": f"{stage['icon']} {stage['label']}",
                "completed": stage["id"] in completed_stages,
                "position": i,
            })
            if i > 0:
                edges.append({
                    "source": PIPELINE_STAGES[i - 1]["id"],
                    "target": stage["id"],
                })
        return {"nodes": nodes, "edges": edges}

    def clear(self):
        self.events = []
        path = self._lineage_path()
        if path.exists():
            path.unlink()

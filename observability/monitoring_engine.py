"""Data Observability – tracks system performance, pipeline executions, and quality trends."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OBS_DIR = DATA_DIR / "observability"
OBS_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_FILE = OBS_DIR / "events.json"
METRICS_FILE = OBS_DIR / "metrics.json"


def _load_events() -> list[dict[str, Any]]:
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE) as f:
            return json.load(f)
    return []


def _save_events(events: list[dict[str, Any]]):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2, default=str)


def _load_metrics() -> dict[str, Any]:
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {"pipeline_runs": [], "quality_scores": [], "ingestion_events": []}


def _save_metrics(metrics: dict[str, Any]):
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def log_event(event_type: str, dataset_name: str, details: dict[str, Any] | None = None):
    """Log a system event."""
    events = _load_events()
    events.append({
        "event_type": event_type,
        "dataset_name": dataset_name,
        "timestamp": dt.datetime.now().isoformat(),
        "details": details or {},
    })
    if len(events) > 10000:
        events = events[-5000:]
    _save_events(events)


def log_pipeline_run(dataset_name: str, duration_ms: float, status: str,
                     steps_completed: int = 0):
    """Log a pipeline execution."""
    metrics = _load_metrics()
    metrics["pipeline_runs"].append({
        "dataset_name": dataset_name,
        "timestamp": dt.datetime.now().isoformat(),
        "duration_ms": round(duration_ms, 2),
        "status": status,
        "steps_completed": steps_completed,
    })
    if len(metrics["pipeline_runs"]) > 1000:
        metrics["pipeline_runs"] = metrics["pipeline_runs"][-500:]
    _save_metrics(metrics)


def log_quality_score(dataset_name: str, score: float):
    """Log a data quality score."""
    metrics = _load_metrics()
    metrics["quality_scores"].append({
        "dataset_name": dataset_name,
        "timestamp": dt.datetime.now().isoformat(),
        "score": round(score, 2),
    })
    if len(metrics["quality_scores"]) > 1000:
        metrics["quality_scores"] = metrics["quality_scores"][-500:]
    _save_metrics(metrics)


def log_ingestion(dataset_name: str, row_count: int, col_count: int, size_mb: float):
    """Log a dataset ingestion event."""
    metrics = _load_metrics()
    metrics["ingestion_events"].append({
        "dataset_name": dataset_name,
        "timestamp": dt.datetime.now().isoformat(),
        "row_count": row_count,
        "col_count": col_count,
        "size_mb": round(size_mb, 3),
    })
    _save_metrics(metrics)


def get_events(event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    events = _load_events()
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    return events[-limit:]


def get_pipeline_stats() -> dict[str, Any]:
    metrics = _load_metrics()
    runs = metrics.get("pipeline_runs", [])
    if not runs:
        return {"total_runs": 0}
    durations = [r["duration_ms"] for r in runs]
    return {
        "total_runs": len(runs),
        "avg_duration_ms": round(sum(durations) / len(durations), 2),
        "last_run": runs[-1] if runs else None,
        "success_rate": round(
            sum(1 for r in runs if r["status"] == "success") / len(runs) * 100, 1
        ),
    }


def get_quality_trend(dataset_name: str | None = None) -> list[dict[str, Any]]:
    metrics = _load_metrics()
    scores = metrics.get("quality_scores", [])
    if dataset_name:
        scores = [s for s in scores if s["dataset_name"] == dataset_name]
    return scores


def get_observability_summary() -> dict[str, Any]:
    events = _load_events()
    metrics = _load_metrics()
    return {
        "total_events": len(events),
        "total_pipeline_runs": len(metrics.get("pipeline_runs", [])),
        "total_ingestions": len(metrics.get("ingestion_events", [])),
        "pipeline_stats": get_pipeline_stats(),
        "recent_events": events[-10:] if events else [],
    }

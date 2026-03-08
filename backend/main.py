"""FastAPI REST API for the AI Analytics Platform backend."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path so sibling packages resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.config import settings
from backend.data_loader import DataLoadError, clear_cache, load_csv_from_file
from backend.data_profiler import (
    compute_correlations,
    compute_feature_importance,
    detect_anomalies,
    profile_dataset,
)
from backend.insights_engine import generate_insights
from backend.ai_chat_engine import ask_question
from utils.file_utils import file_hash
from utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Analytics Platform API",
    version="1.0.0",
    description="Upload datasets and receive automated profiling, visualizations, and AI insights.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_current_df = None
_current_profile = None
_current_cache_key: str | None = None


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV file and return a basic preview."""
    global _current_df, _current_profile, _current_cache_key

    try:
        contents = await file.read()
        buf = io.BytesIO(contents)
        cache_key = file_hash(buf)
        df = load_csv_from_file(buf, file.filename or "upload.csv", cache_key=cache_key)
        _current_df = df
        _current_cache_key = cache_key
        _current_profile = profile_dataset(df)

        return {
            "status": "ok",
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "preview": df.head(10).to_dict(orient="records"),
        }
    except DataLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/profile")
async def get_profile() -> dict[str, Any]:
    """Return the dataset profile (stats, types, KPIs)."""
    if _current_df is None or _current_profile is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a file first.")

    p = _current_profile
    return {
        "n_rows": p.n_rows,
        "n_cols": p.n_cols,
        "memory_mb": p.memory_usage_mb,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "detected_type": c.detected_type,
                "missing_count": c.missing_count,
                "missing_pct": c.missing_pct,
                "unique_count": c.unique_count,
            }
            for c in p.columns
        ],
        "numeric_kpis": p.numeric_kpis,
        "categorical_kpis": p.categorical_kpis,
    }


@app.get("/api/insights")
async def get_insights() -> dict[str, Any]:
    """Generate AI insights for the loaded dataset."""
    if _current_df is None or _current_profile is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    insights = generate_insights(_current_df, _current_profile, cache_key=_current_cache_key)
    return {"insights": insights}


@app.post("/api/chat")
async def chat(question: str = Query(...)) -> dict[str, Any]:
    """Answer a natural-language question about the loaded dataset."""
    if _current_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    resp = ask_question(_current_df, question)
    result: dict[str, Any] = {"answer": resp.answer}
    if resp.data is not None:
        result["data"] = resp.data.head(50).to_dict(orient="records")
    return result


@app.get("/api/anomalies")
async def get_anomalies() -> dict[str, Any]:
    """Detect statistical anomalies in the dataset."""
    if _current_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    anomalies = detect_anomalies(_current_df)
    return {"anomalies": {col: len(idxs) for col, idxs in anomalies.items()}}


@app.get("/api/correlations")
async def get_correlations() -> dict[str, Any]:
    """Return the correlation matrix as nested dict."""
    if _current_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    corr = compute_correlations(_current_df)
    if corr.empty:
        return {"correlations": {}}
    return {"correlations": corr.round(4).to_dict()}


@app.get("/api/feature-importance")
async def get_feature_importance(target: str = Query(...)) -> dict[str, Any]:
    """Return approximate feature importance relative to *target*."""
    if _current_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    importance = compute_feature_importance(_current_df, target)
    return {"target": target, "importance": importance}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def start_server() -> None:
    """Run the API server via Uvicorn."""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    start_server()

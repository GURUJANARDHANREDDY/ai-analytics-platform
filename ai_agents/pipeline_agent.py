"""Pipeline Recommendation Agent – suggests optimal data pipeline configurations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def analyze_pipeline_needs(df: pd.DataFrame) -> dict[str, Any]:
    analysis: dict[str, Any] = {"steps": [], "optimizations": [], "warnings": []}
    analysis["steps"].append({"step": "Data Ingestion", "status": "completed", "detail": f"Loaded {len(df):,} rows x {len(df.columns)} cols"})
    analysis["steps"].append({"step": "Bronze Storage", "status": "completed", "detail": "Raw data preserved"})
    missing_pct = df.isna().mean().mean() * 100
    dup_count = df.duplicated().sum()
    parts = []
    if missing_pct > 0:
        parts.append(f"{missing_pct:.1f}% missing")
    if dup_count > 0:
        parts.append(f"{dup_count} duplicates")
    analysis["steps"].append({"step": "Data Cleaning (Silver)", "status": "recommended", "detail": "; ".join(parts) if parts else "Clean"})
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if numeric_cols and categorical_cols:
        analysis["steps"].append({"step": "Feature Engineering", "status": "recommended", "detail": f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical"})
    analysis["steps"].append({"step": "Gold Layer Analytics", "status": "recommended", "detail": "Business analytics tables"})
    if len(df) > 100_000:
        analysis["optimizations"].append("Consider chunked processing for large dataset")
    high_missing = df.columns[df.isna().mean() > 0.5].tolist()
    if high_missing:
        analysis["warnings"].append(f"Columns with >50% missing: {high_missing}")
    return analysis


def get_ai_pipeline_recommendation(df: pd.DataFrame) -> str:
    needs = analyze_pipeline_needs(df)
    context = f"Pipeline analysis ({len(df)} rows, {len(df.columns)} cols):\n"
    for step in needs["steps"]:
        context += f"  {step['step']}: {step['status']} – {step['detail']}\n"
    if not HF_AVAILABLE:
        lines = ["Pipeline Recommendations:"]
        for step in needs["steps"]:
            lines.append(f"  [{step['status'].upper()}] {step['step']}: {step['detail']}")
        return "\n".join(lines)
    resp = call_chat([{"role": "user", "content": f"Recommend the optimal data pipeline:\n\n{context}"}])
    return resp if resp and not resp.startswith("[") else "Set HF_API_TOKEN for AI recommendations."

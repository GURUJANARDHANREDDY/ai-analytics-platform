"""Dashboard Builder Agent – recommends charts and dashboard layouts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def recommend_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    recs = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    for col in numeric_cols[:4]:
        recs.append({"chart_type": "histogram", "columns": [col], "reason": f"Distribution of '{col}'", "priority": "high"})
    for cat in categorical_cols[:3]:
        if df[cat].nunique() <= 20:
            recs.append({"chart_type": "bar_chart", "columns": [cat], "reason": f"Frequency of '{cat}'", "priority": "high"})
    for cat in categorical_cols[:2]:
        for num in numeric_cols[:2]:
            if df[cat].nunique() <= 20:
                recs.append({"chart_type": "grouped_bar", "columns": [cat, num], "reason": f"Compare {num} across {cat}", "priority": "high"})
    for dt_col in datetime_cols:
        for num in numeric_cols[:3]:
            recs.append({"chart_type": "time_series", "columns": [dt_col, num], "reason": f"Trend of {num} over time", "priority": "high"})
    if len(numeric_cols) >= 2:
        recs.append({"chart_type": "correlation_heatmap", "columns": numeric_cols[:10], "reason": "Correlations", "priority": "medium"})
        recs.append({"chart_type": "scatter", "columns": [numeric_cols[0], numeric_cols[1]], "reason": f"Relationship", "priority": "medium"})
    for cat in categorical_cols[:2]:
        if 2 <= df[cat].nunique() <= 8:
            recs.append({"chart_type": "pie_chart", "columns": [cat], "reason": f"Proportions of '{cat}'", "priority": "low"})
    for num in numeric_cols[:3]:
        recs.append({"chart_type": "box_plot", "columns": [num], "reason": f"Spread of '{num}'", "priority": "medium"})
    return recs


def recommend_dashboard_layout(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    sections = [{"section": "KPI Cards", "position": "top", "items": [f"Total {col}" for col in numeric_cols[:4]]}]
    if datetime_cols and numeric_cols:
        sections.append({"section": "Trend Analysis", "position": "upper", "charts": [{"type": "time_series", "x": datetime_cols[0], "y": col} for col in numeric_cols[:2]]})
    if categorical_cols and numeric_cols:
        sections.append({"section": "Category Breakdown", "position": "middle", "charts": [{"type": "bar", "x": categorical_cols[0], "y": numeric_cols[0]}]})
    if len(numeric_cols) >= 2:
        sections.append({"section": "Correlations", "position": "lower", "charts": [{"type": "heatmap"}, {"type": "histogram"}]})
    sections.append({"section": "Data Quality", "position": "bottom", "items": ["Quality Score", "Missing Values"]})
    return {"total_sections": len(sections), "layout": sections, "recommended_charts_count": sum(len(s.get("charts", s.get("items", []))) for s in sections)}


def get_ai_recommendations(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()[:20]
    dtypes = {col: str(df[col].dtype) for col in cols}
    if not HF_AVAILABLE:
        recs = recommend_charts(df)
        return "Dashboard Recommendations:\n" + "\n".join(f"  - {r['chart_type']}: {r['reason']}" for r in recs[:8])
    resp = call_chat([{"role": "user", "content": f"Dataset columns {cols}, types {dtypes}. Suggest 5 specific charts for a dashboard. State chart type, columns, and why. Be concise."}])
    return resp if resp and not resp.startswith("[") else "Set HF_API_TOKEN for AI recommendations."

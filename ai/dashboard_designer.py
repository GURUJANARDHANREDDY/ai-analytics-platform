"""AI Dashboard Designer — uses LLM to analyze a dataset and recommend dashboard layout."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE
from utils.column_classifier import get_measure_columns, get_dimension_columns


def design_dashboard(df: pd.DataFrame) -> dict[str, Any] | None:
    """Ask the LLM to design a dashboard for the given dataset.

    Returns a dict with keys: title, subtitle, kpis, charts, insights
    or None if LLM is unavailable / fails.
    """
    if not HF_AVAILABLE:
        return None

    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique())
        sample = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(s) for s in sample)
        col_info.append(f"  - {col} (type={dtype}, unique={nunique}, sample=[{sample_str}])")

    col_block = "\n".join(col_info)

    prompt = f"""You are a data analytics expert. Analyze this dataset and design the best dashboard.

Dataset: {len(df):,} rows x {len(df.columns)} columns
Measures (numeric): {measures}
Dimensions (categorical): {dimensions}
Datetime columns: {datetime_cols}

Columns:
{col_block}

Design a dashboard. Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "title": "Dashboard title based on the data",
  "subtitle": "One line describing what this data is about",
  "kpis": [
    {{"column": "column_name", "agg": "sum|mean|count|max|min", "label": "KPI Label"}}
  ],
  "charts": [
    {{
      "type": "bar|line|scatter|pie|treemap|box|heatmap|histogram|horizontal_bar|grouped_bar|donut",
      "title": "Chart title",
      "x": "column_name",
      "y": "column_name or null",
      "color": "column_name or null",
      "agg": "sum|mean|count|null",
      "top_n": 10,
      "reason": "Why this chart is useful"
    }}
  ],
  "insights": ["Key insight 1 about this data", "Key insight 2", "Key insight 3"]
}}

Rules:
- Pick 4-6 KPIs that matter most for this data
- Pick 6-8 charts that tell the most interesting story
- Use the right chart type for each analysis (bar for comparison, line for trends, scatter for correlation, pie for composition, etc.)
- For bar/treemap, always specify top_n (5-15)
- Charts should be diverse — don't repeat the same type
- Insights should be specific observations about the data
- Return ONLY the JSON object, nothing else"""

    response = call_chat([
        {"role": "system", "content": "You are a dashboard designer. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ], max_tokens=2000, temperature=0.3)

    if not response or response.startswith("[HuggingFace Error"):
        return None

    return _parse_design(response, measures, dimensions, datetime_cols)


def _parse_design(response: str, measures: list[str], dimensions: list[str],
                  datetime_cols: list[str]) -> dict[str, Any] | None:
    """Parse LLM JSON response into a dashboard design."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return None

    try:
        design = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    if "charts" not in design or "kpis" not in design:
        return None

    valid_charts = []
    all_cols = set(measures + dimensions + datetime_cols)
    for chart in design.get("charts", []):
        if not isinstance(chart, dict) or "type" not in chart:
            continue
        x = chart.get("x", "")
        y = chart.get("y", "")
        if x and x not in all_cols:
            continue
        if y and y not in all_cols:
            continue
        valid_charts.append(chart)

    design["charts"] = valid_charts[:8]

    valid_kpis = []
    for kpi in design.get("kpis", []):
        if not isinstance(kpi, dict):
            continue
        col = kpi.get("column", "")
        if col in measures:
            valid_kpis.append(kpi)

    design["kpis"] = valid_kpis[:6]

    return design if valid_charts else None

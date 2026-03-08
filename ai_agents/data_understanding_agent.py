"""Data Understanding Agent – detects dataset structure, suggests KPIs and transformations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def analyze_structure(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    id_candidates = [c for c in df.columns if df[c].nunique() == len(df) or "id" in c.lower()]
    return {
        "total_columns": len(df.columns), "total_rows": len(df),
        "numeric_columns": numeric_cols, "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols, "boolean_columns": bool_cols,
        "id_candidates": id_candidates,
        "high_cardinality_columns": [c for c in categorical_cols if df[c].nunique() > 50],
        "low_cardinality_columns": [c for c in categorical_cols if df[c].nunique() <= 10],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def suggest_kpis(df: pd.DataFrame) -> list[dict[str, str]]:
    suggestions = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    revenue_cols = [c for c in numeric_cols if any(k in c.lower() for k in ["revenue", "sales", "amount", "price", "total", "value"])]
    customer_cols = [c for c in categorical_cols if any(k in c.lower() for k in ["customer", "client", "user", "buyer"])]
    region_cols = [c for c in categorical_cols if any(k in c.lower() for k in ["region", "country", "city", "state", "location"])]
    product_cols = [c for c in categorical_cols if any(k in c.lower() for k in ["product", "item", "sku", "category"])]
    for col in revenue_cols:
        suggestions.append({"kpi": f"Total {col}", "formula": f"df['{col}'].sum()", "type": "aggregate"})
        suggestions.append({"kpi": f"Average {col}", "formula": f"df['{col}'].mean()", "type": "aggregate"})
    if revenue_cols and customer_cols:
        suggestions.append({"kpi": f"{revenue_cols[0]} by {customer_cols[0]}", "formula": f"df.groupby('{customer_cols[0]}')['{revenue_cols[0]}'].sum()", "type": "grouped"})
    if revenue_cols and region_cols:
        suggestions.append({"kpi": f"{revenue_cols[0]} by {region_cols[0]}", "formula": f"df.groupby('{region_cols[0]}')['{revenue_cols[0]}'].sum()", "type": "grouped"})
    if product_cols:
        suggestions.append({"kpi": f"Top {product_cols[0]}", "formula": f"df['{product_cols[0]}'].value_counts().head(10)", "type": "ranking"})
    for col in numeric_cols[:3]:
        if col not in revenue_cols:
            suggestions.append({"kpi": f"Total {col}", "formula": f"df['{col}'].sum()", "type": "aggregate"})
    return suggestions


def suggest_transformations(df: pd.DataFrame) -> list[dict[str, str]]:
    suggestions = []
    cols_with_missing = df.columns[df.isna().sum() > 0].tolist()
    if cols_with_missing:
        suggestions.append({"transformation": "Handle missing values", "details": f"Columns: {', '.join(cols_with_missing[:5])}", "priority": "high"})
    if df.duplicated().sum() > 0:
        suggestions.append({"transformation": "Remove duplicates", "details": f"{df.duplicated().sum()} duplicate rows", "priority": "high"})
    for col in df.select_dtypes(include="datetime").columns:
        suggestions.append({"transformation": f"Extract date parts from '{col}'", "details": "year, month, day_of_week", "priority": "medium"})
    for col in df.select_dtypes(include="number").columns:
        if abs(df[col].skew()) > 2 and df[col].std() > 0:
            suggestions.append({"transformation": f"Log-transform '{col}'", "details": f"Skewness: {df[col].skew():.1f}", "priority": "medium"})
    return suggestions


def get_ai_understanding(df: pd.DataFrame) -> str:
    structure = analyze_structure(df)
    summary = (
        f"Dataset with {structure['total_rows']} rows and {structure['total_columns']} columns. "
        f"Numeric: {structure['numeric_columns'][:10]}. Categorical: {structure['categorical_columns'][:10]}. "
        f"Sample:\n{df.head(5).to_string()}"
    )
    if not HF_AVAILABLE:
        return (f"Dataset: {structure['total_rows']:,} rows, {structure['total_columns']} cols\n"
                f"- {len(structure['numeric_columns'])} numeric, {len(structure['categorical_columns'])} categorical\n"
                f"- Memory: {structure['memory_mb']} MB")
    resp = call_chat([{"role": "user", "content": f"Describe this dataset in 3-4 sentences. What domain? Key observations?\n\n{summary}"}])
    return resp if resp and not resp.startswith("[") else f"Dataset: {structure['total_rows']:,} rows, {structure['total_columns']} cols."

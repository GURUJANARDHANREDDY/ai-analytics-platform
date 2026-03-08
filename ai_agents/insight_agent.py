"""Insight Generation Agent – identifies patterns, anomalies, and recommendations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def detect_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    anomalies = []
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
        count = outlier_mask.sum()
        if count > 0:
            anomalies.append({"column": col, "outlier_count": int(count), "outlier_pct": round(count / len(clean) * 100, 2),
                              "lower_bound": round(float(q1 - 1.5 * iqr), 2), "upper_bound": round(float(q3 + 1.5 * iqr), 2)})
    return anomalies


def detect_patterns(df: pd.DataFrame) -> list[dict[str, str]]:
    patterns = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    patterns.append({"type": "strong_correlation", "detail": f"Strong {'positive' if val > 0 else 'negative'} correlation ({val:.2f}) between '{corr.columns[i]}' and '{corr.columns[j]}'"})
    for col in df.select_dtypes(include=["object", "category"]).columns:
        counts = df[col].value_counts()
        if len(counts) > 0 and counts.iloc[0] / len(df) * 100 > 50:
            patterns.append({"type": "dominant_category", "detail": f"'{counts.index[0]}' dominates '{col}' at {counts.iloc[0] / len(df) * 100:.1f}%"})
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 2:
            patterns.append({"type": "skewed_distribution", "detail": f"'{col}' is {'right' if skew > 0 else 'left'}-skewed ({skew:.2f})"})
    return patterns


def generate_recommendations(df: pd.DataFrame) -> list[dict[str, str]]:
    recs = []
    missing_pct = df.isna().mean().mean() * 100
    if missing_pct > 10:
        recs.append({"type": "data_quality", "priority": "high", "recommendation": f"Address data completeness – {missing_pct:.1f}% missing."})
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for cat in categorical_cols[:3]:
        for num in numeric_cols[:3]:
            try:
                grouped = df.groupby(cat)[num].mean()
                if len(grouped) >= 2 and grouped.min() > 0:
                    ratio = grouped.max() / grouped.min()
                    if ratio > 3:
                        recs.append({"type": "performance_gap", "priority": "medium",
                                     "recommendation": f"Gap in '{num}' across '{cat}': top='{grouped.idxmax()}' ({grouped.max():,.0f}) vs bottom='{grouped.idxmin()}' ({grouped.min():,.0f})"})
            except Exception:
                continue
    return recs


def get_ai_insights(df: pd.DataFrame) -> str:
    anomalies = detect_anomalies(df)
    patterns = detect_patterns(df)
    context = f"Dataset: {len(df)} rows, {len(df.columns)} columns\nAnomalies: {len(anomalies)}\nPatterns: {len(patterns)}\n"
    for a in anomalies[:3]:
        context += f"  - {a['column']}: {a['outlier_count']} outliers\n"
    for p in patterns[:3]:
        context += f"  - {p['detail']}\n"
    if not HF_AVAILABLE:
        recs = generate_recommendations(df)
        return "Insights:\n" + "\n".join(f"  [{r['priority'].upper()}] {r['recommendation']}" for r in recs[:5])
    resp = call_chat([{"role": "user", "content": f"Based on this analysis, provide 5 business insights:\n\n{context}"}])
    return resp if resp and not resp.startswith("[") else "Set HF_API_TOKEN for AI insights."

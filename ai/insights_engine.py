"""AI Insights Engine – uses Hugging Face LLM to generate dataset insights."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def _build_dataset_summary(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    lines = [
        f"Dataset: {len(df):,} rows x {len(df.columns)} columns",
        f"Columns: {', '.join(df.columns.tolist()[:30])}",
        f"Numeric columns: {', '.join(numeric_cols[:15])}",
        f"Categorical columns: {', '.join(categorical_cols[:15])}",
        f"Missing values: {df.isna().sum().sum()} total",
    ]
    for col in numeric_cols[:8]:
        clean = df[col].dropna()
        if len(clean) > 0:
            lines.append(
                f"  {col}: mean={clean.mean():.2f}, median={clean.median():.2f}, "
                f"min={clean.min():.2f}, max={clean.max():.2f}, std={clean.std():.2f}"
            )
    for col in categorical_cols[:5]:
        top = df[col].value_counts().head(5)
        lines.append(f"  {col}: top values = {dict(top)}")
    return "\n".join(lines)


def generate_insights(df: pd.DataFrame, cache_key: str | None = None) -> list[str]:
    summary = _build_dataset_summary(df)
    prompt = (
        "You are a senior data analyst. Analyze this dataset and provide exactly 8 actionable insights.\n\n"
        f"{summary}\n\n"
        "For each insight, identify:\n"
        "1. Top performing segments\n2. Underperforming areas\n3. Growth opportunities\n"
        "4. Anomalies\n5. Key correlations\n6. Business recommendations\n"
        "7. Risk factors\n8. Strategic next steps\n\n"
        "Number them 1-8. Be specific with numbers and column names."
    )

    if not HF_AVAILABLE:
        return _generate_fallback_insights(df)

    response = call_chat([{"role": "user", "content": prompt}])
    if not response or response.startswith("[HuggingFace Error"):
        fallback = _generate_fallback_insights(df)
        if response.startswith("["):
            fallback.append(response)
        return fallback

    insights = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 20:
            if line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
            if line:
                insights.append(line)
    return insights[:8] if insights else _generate_fallback_insights(df)


def _generate_fallback_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    insights.append(
        f"The dataset contains {len(df):,} records across {len(df.columns)} columns, "
        f"with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features."
    )
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        worst_col = df.isna().sum().idxmax()
        worst_pct = df[worst_col].isna().mean() * 100
        insights.append(
            f"Data completeness concern: {missing_total:,} missing values detected. "
            f"Column '{worst_col}' has the highest missing rate at {worst_pct:.1f}%."
        )
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        insights.append(f"Found {dup_count:,} duplicate rows ({dup_count / len(df) * 100:.1f}% of total).")
    for col in numeric_cols[:3]:
        clean = df[col].dropna()
        if len(clean) > 0:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    insights.append(
                        f"Column '{col}' has {outliers} outliers "
                        f"(range: {clean.min():.2f} to {clean.max():.2f}, mean: {clean.mean():.2f})."
                    )
    for col in categorical_cols[:2]:
        top = df[col].value_counts().head(3)
        total = len(df)
        top_items = ", ".join(f"'{k}' ({v / total * 100:.1f}%)" for k, v in top.items())
        insights.append(f"Top '{col}' values: {top_items}.")
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    insights.append(
                        f"Strong correlation ({val:.2f}) between "
                        f"'{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}'."
                    )
                    break
            else:
                continue
            break
    while len(insights) < 4:
        insights.append(
            f"Dataset memory footprint: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB."
        )
        break
    return insights

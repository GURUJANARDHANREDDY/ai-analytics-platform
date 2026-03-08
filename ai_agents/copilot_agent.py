"""AI Data Copilot – conversational assistant for the data platform."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


SYSTEM_PROMPT = (
    "You are an AI Data Copilot for an enterprise analytics platform. "
    "You help users understand datasets, suggest KPIs, recommend dashboards, "
    "suggest transformations, explain anomalies, and provide data strategy advice. "
    "Be concise, specific, and actionable. Reference column names and statistics."
)


def _build_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    lines = [
        f"Dataset: {len(df):,} rows x {len(df.columns)} columns",
        f"Numeric: {', '.join(numeric_cols[:15])}",
        f"Categorical: {', '.join(categorical_cols[:15])}",
        f"Missing: {df.isna().sum().sum()} values",
        f"Duplicates: {df.duplicated().sum()} rows",
    ]
    for col in numeric_cols[:5]:
        c = df[col].dropna()
        if len(c) > 0:
            lines.append(f"  {col}: mean={c.mean():.2f}, min={c.min():.2f}, max={c.max():.2f}")
    for col in categorical_cols[:3]:
        top = df[col].value_counts().head(3)
        lines.append(f"  {col}: {dict(top)}")
    return "\n".join(lines)


def copilot_chat(query: str, df: pd.DataFrame, history: list[dict[str, str]] | None = None) -> str:
    context = _build_context(df)

    if not HF_AVAILABLE:
        return _fallback_copilot(query, df)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"Dataset context:\n{context}"})
    messages.append({"role": "assistant", "content": "I have analyzed the dataset. How can I help?"})

    if history:
        for msg in history[-6:]:
            messages.append(msg)

    messages.append({"role": "user", "content": query})

    resp = call_chat(messages)
    if resp and not resp.startswith("[HuggingFace Error"):
        return resp
    return _fallback_copilot(query, df)


def _fallback_copilot(query: str, df: pd.DataFrame) -> str:
    q = query.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if "explain" in q or "describe" in q or "what" in q:
        return (f"This dataset has {len(df):,} rows and {len(df.columns)} columns.\n"
                f"- {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:8])}\n"
                f"- {len(categorical_cols)} categorical columns: {', '.join(categorical_cols[:8])}\n"
                f"- Missing values: {df.isna().sum().sum():,}\n"
                f"- Duplicates: {df.duplicated().sum():,}")

    if "kpi" in q or "metric" in q:
        kpis = []
        for col in numeric_cols[:5]:
            kpis.append(f"- Total {col}: {df[col].sum():,.2f}")
            kpis.append(f"- Avg {col}: {df[col].mean():,.2f}")
        return "Suggested KPIs:\n" + "\n".join(kpis)

    if "dashboard" in q or "chart" in q:
        suggestions = []
        for col in numeric_cols[:3]:
            suggestions.append(f"- Histogram of {col}")
        for col in categorical_cols[:2]:
            if df[col].nunique() <= 20:
                suggestions.append(f"- Bar chart of {col}")
        return "Dashboard suggestions:\n" + "\n".join(suggestions)

    if "transform" in q or "clean" in q:
        steps = []
        if df.isna().sum().sum() > 0:
            steps.append("- Fill missing values (median for numeric, mode for categorical)")
        if df.duplicated().sum() > 0:
            steps.append(f"- Remove {df.duplicated().sum()} duplicate rows")
        steps.append("- Standardize column names")
        return "Recommended transformations:\n" + "\n".join(steps)

    if "anomal" in q or "outlier" in q:
        findings = []
        for col in numeric_cols[:5]:
            c = df[col].dropna()
            if len(c) >= 10:
                q1, q3 = c.quantile(0.25), c.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    n = ((c < q1 - 1.5 * iqr) | (c > q3 + 1.5 * iqr)).sum()
                    if n > 0:
                        findings.append(f"- {col}: {n} outliers")
        return "Anomalies detected:\n" + ("\n".join(findings) if findings else "  No significant outliers found.")

    return (f"I can help with this dataset ({len(df):,} rows, {len(df.columns)} cols). "
            "Ask me to:\n- Explain the dataset\n- Suggest KPIs\n- Recommend dashboards\n"
            "- Suggest transformations\n- Explain anomalies\n\n"
            "Set HF_API_TOKEN in .env for full AI-powered responses.")

"""AI-powered insights generation using OpenAI LLMs."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from backend.config import settings
from backend.data_profiler import (
    DatasetProfile,
    compute_correlations,
    detect_anomalies,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_insights_cache: dict[str, list[str]] = {}


def _build_summary_prompt(df: pd.DataFrame, profile: DatasetProfile) -> str:
    """Construct a system + user prompt for the LLM from profile data."""
    numeric_summary = ""
    for col, kpi in profile.numeric_kpis.items():
        numeric_summary += (
            f"  - {col}: mean={kpi['mean']}, median={kpi['median']}, "
            f"min={kpi['min']}, max={kpi['max']}, std={kpi['std']}\n"
        )

    categorical_summary = ""
    for col, kpi in profile.categorical_kpis.items():
        top = list(kpi["top_categories"].items())[:5]
        categorical_summary += f"  - {col}: top values = {top}\n"

    corr_matrix = compute_correlations(df)
    corr_text = ""
    if not corr_matrix.empty:
        strong = []
        cols = corr_matrix.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                val = corr_matrix.loc[c1, c2]
                if abs(val) > 0.5:
                    strong.append(f"{c1} & {c2}: {val:.2f}")
        if strong:
            corr_text = "Strong correlations:\n" + "\n".join(f"  - {s}" for s in strong)

    anomalies = detect_anomalies(df)
    anomaly_text = ""
    if anomalies:
        anomaly_text = "Columns with outliers (z > 3):\n"
        for col, idxs in anomalies.items():
            anomaly_text += f"  - {col}: {len(idxs)} outlier(s)\n"

    prompt = f"""You are a senior data analyst. Analyze the following dataset summary and provide 5-8 concise, 
actionable insights in bullet-point form. Focus on trends, anomalies, key distributions, and business-relevant observations.

Dataset: {profile.n_rows} rows × {profile.n_cols} columns | Memory: {profile.memory_usage_mb} MB

Numeric columns:
{numeric_summary or '  (none)'}

Categorical columns:
{categorical_summary or '  (none)'}

{corr_text}

{anomaly_text}

Respond ONLY with the bullet-point insights, no preamble."""
    return prompt


def generate_insights(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cache_key: Optional[str] = None,
) -> list[str]:
    """Call the OpenAI API to generate natural-language insights.

    Falls back to rule-based insights when the API key is unavailable.
    """
    if cache_key and cache_key in _insights_cache:
        logger.info("Returning cached insights for key=%s", cache_key)
        return _insights_cache[cache_key]

    if settings.hf_api_token:
        insights = _hf_insights(df, profile)
    elif settings.gemini_api_key:
        insights = _gemini_insights(df, profile)
    elif settings.openai_api_key:
        insights = _llm_insights(df, profile)
    else:
        logger.warning("No AI key – falling back to rule-based insights.")
        insights = _rule_based_insights(df, profile)

    if cache_key:
        _insights_cache[cache_key] = insights

    return insights


def _hf_insights(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Generate insights via Hugging Face Inference API (direct requests, no SSL issues)."""
    try:
        import requests as _req
        import warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

        prompt = _build_summary_prompt(df, profile)
        messages = [
            {"role": "system", "content": "You are a world-class data analyst."},
            {"role": "user", "content": prompt},
        ]

        url = "https://router.huggingface.co/novita/v3/openai/chat/completions"
        headers = {"Authorization": f"Bearer {settings.hf_api_token}",
                   "Content-Type": "application/json"}
        payload = {"model": settings.hf_model, "messages": messages,
                   "max_tokens": 1024, "temperature": 0.4, "stream": False}
        resp = _req.post(url, headers=headers, json=payload, verify=False, timeout=90)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]

        insights = [line.strip().lstrip("•-–*# 0123456789.") for line in text.strip().split("\n") if line.strip()]
        insights = [i for i in insights if len(i) > 10]
        logger.info("HuggingFace generated %d insights.", len(insights))
        return insights if insights else _rule_based_insights(df, profile)

    except Exception as exc:
        logger.warning("HuggingFace insights failed (falling back to rule-based): %s", str(exc)[:120])
        return _rule_based_insights(df, profile)


def _llm_insights(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Generate insights via OpenAI chat completions."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        prompt = _build_summary_prompt(df, profile)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a world-class data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
        )

        text = response.choices[0].message.content or ""
        insights = [line.strip().lstrip("•-– ") for line in text.strip().split("\n") if line.strip()]
        logger.info("LLM generated %d insights.", len(insights))
        return insights

    except Exception as exc:
        logger.error("OpenAI insight generation failed: %s", exc)
        return _rule_based_insights(df, profile)


def _gemini_insights(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Generate insights via Google Gemini API."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        prompt = _build_summary_prompt(df, profile)

        response = model.generate_content(
            prompt,
            request_options={"timeout": 15},
        )
        text = response.text or ""
        insights = [line.strip().lstrip("•-–*# ") for line in text.strip().split("\n") if line.strip()]
        insights = [i for i in insights if len(i) > 10]
        logger.info("Gemini generated %d insights.", len(insights))
        return insights

    except Exception as exc:
        logger.warning("Gemini insights failed (falling back to rule-based): %s", str(exc)[:120])
        return _rule_based_insights(df, profile)


def _rule_based_insights(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Deterministic insights when the LLM is unavailable."""
    insights: list[str] = []

    insights.append(f"The dataset contains {profile.n_rows:,} rows and {profile.n_cols} columns.")

    total_missing = sum(cp.missing_count for cp in profile.columns)
    if total_missing:
        pct = total_missing / (profile.n_rows * profile.n_cols) * 100
        insights.append(f"Overall missing-value rate is {pct:.1f}% ({total_missing:,} cells).")

    for col, kpi in profile.numeric_kpis.items():
        spread = kpi["max"] - kpi["min"]
        if kpi["std"] > 0 and spread > 0:
            cv = kpi["std"] / abs(kpi["mean"]) if kpi["mean"] != 0 else 0
            if cv > 1:
                insights.append(f"Column '{col}' has high variability (CV = {cv:.2f}).")

    for col, kpi in profile.categorical_kpis.items():
        top = list(kpi["top_categories"].items())
        if top:
            dominant = top[0]
            total = sum(v for _, v in top)
            share = dominant[1] / total * 100 if total else 0
            if share > 40:
                insights.append(
                    f"'{dominant[0]}' dominates column '{col}' with {share:.0f}% of values."
                )

    corr = compute_correlations(df)
    if not corr.empty:
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                val = corr.loc[c1, c2]
                if abs(val) > 0.7:
                    direction = "positively" if val > 0 else "negatively"
                    insights.append(f"'{c1}' and '{c2}' are strongly {direction} correlated ({val:.2f}).")

    anomalies = detect_anomalies(df)
    for col, idxs in anomalies.items():
        insights.append(f"Column '{col}' has {len(idxs)} statistical outlier(s) (z-score > 3).")

    if not insights:
        insights.append("No notable patterns detected – the dataset appears uniform.")

    return insights


def clear_insights_cache(key: Optional[str] = None) -> None:
    """Remove cached insights."""
    if key:
        _insights_cache.pop(key, None)
    else:
        _insights_cache.clear()

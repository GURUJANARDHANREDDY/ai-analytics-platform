"""Data Quality Debug Agent – diagnoses data issues and suggests fixes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def diagnose_data_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues = []
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct > 0:
            severity = "critical" if null_pct > 50 else "warning" if null_pct > 20 else "info"
            issues.append({"issue": "missing_values", "column": col, "severity": severity,
                           "detail": f"{null_pct:.1f}% missing",
                           "fix": f"Fill with {'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'}"})
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({"issue": "duplicate_rows", "column": "__dataset__", "severity": "warning",
                       "detail": f"{dup_count} duplicates ({dup_count / len(df) * 100:.1f}%)", "fix": "df.drop_duplicates()"})
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            numeric_pct = pd.to_numeric(non_null, errors="coerce").notna().mean() * 100
            if 30 < numeric_pct < 100:
                issues.append({"issue": "mixed_types", "column": col, "severity": "warning",
                               "detail": f"{numeric_pct:.0f}% numeric in object column", "fix": f"pd.to_numeric(df['{col}'], errors='coerce')"})
    for col in df.select_dtypes(include="number").columns:
        if df[col].std() == 0 and df[col].notna().sum() > 0:
            issues.append({"issue": "zero_variance", "column": col, "severity": "info",
                           "detail": f"Constant value: {df[col].iloc[0]}", "fix": "Consider dropping"})
    for col in df.columns:
        if df[col].isna().all():
            issues.append({"issue": "empty_column", "column": col, "severity": "critical",
                           "detail": "Entirely empty", "fix": f"df.drop('{col}', axis=1)"})
    return issues


def generate_debug_report(df: pd.DataFrame) -> dict[str, Any]:
    issues = diagnose_data_issues(df)
    critical = [i for i in issues if i["severity"] == "critical"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]
    score = max(0, min(100, 100 - len(critical) * 20 - len(warnings) * 5 - len(infos)))
    return {"health_score": score, "total_issues": len(issues), "critical_issues": len(critical),
            "warning_issues": len(warnings), "info_issues": len(infos),
            "issues": issues, "critical": critical, "warnings": warnings, "infos": infos}


def get_ai_debug_analysis(df: pd.DataFrame) -> str:
    report = generate_debug_report(df)
    context = f"Health Score: {report['health_score']}/100\nIssues: {report['critical_issues']} critical, {report['warning_issues']} warnings\n"
    for issue in report["issues"][:8]:
        context += f"  [{issue['severity'].upper()}] {issue['column']}: {issue['detail']} → Fix: {issue['fix']}\n"
    if not HF_AVAILABLE:
        return context
    resp = call_chat([{"role": "user", "content": f"Analyze and prioritize these data quality fixes:\n\n{context}"}])
    return resp if resp and not resp.startswith("[") else context

"""Data Quality Framework – automated validation rules, scoring, and alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualityAlert:
    severity: str  # critical, warning, info
    column: str
    rule: str
    message: str
    value: float | int | str


@dataclass
class QualityReport:
    overall_score: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    alerts: list[QualityAlert]
    column_scores: dict[str, float]
    summary: dict[str, Any]


def _check_nulls(df: pd.DataFrame) -> list[QualityAlert]:
    alerts = []
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct > 50:
            alerts.append(QualityAlert("critical", col, "null_check",
                                       f"Column '{col}' has {null_pct:.1f}% null values", round(null_pct, 2)))
        elif null_pct > 20:
            alerts.append(QualityAlert("warning", col, "null_check",
                                       f"Column '{col}' has {null_pct:.1f}% null values", round(null_pct, 2)))
        elif null_pct > 0:
            alerts.append(QualityAlert("info", col, "null_check",
                                       f"Column '{col}' has {null_pct:.1f}% null values", round(null_pct, 2)))
    return alerts


def _check_duplicates(df: pd.DataFrame) -> list[QualityAlert]:
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / len(df) * 100 if len(df) > 0 else 0
    alerts = []
    if dup_pct > 10:
        alerts.append(QualityAlert("critical", "__dataset__", "duplicate_check",
                                   f"{dup_count} duplicate rows ({dup_pct:.1f}%)", int(dup_count)))
    elif dup_pct > 0:
        alerts.append(QualityAlert("warning", "__dataset__", "duplicate_check",
                                   f"{dup_count} duplicate rows ({dup_pct:.1f}%)", int(dup_count)))
    return alerts


def _check_negative_values(df: pd.DataFrame) -> list[QualityAlert]:
    alerts = []
    for col in df.select_dtypes(include="number").columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            neg_pct = neg_count / len(df) * 100
            alerts.append(QualityAlert("warning", col, "negative_check",
                                       f"Column '{col}' has {neg_count} negative values ({neg_pct:.1f}%)",
                                       int(neg_count)))
    return alerts


def _check_outliers(df: pd.DataFrame) -> list[QualityAlert]:
    alerts = []
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_count = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
        if outlier_count > 0:
            outlier_pct = outlier_count / len(clean) * 100
            severity = "warning" if outlier_pct > 5 else "info"
            alerts.append(QualityAlert(severity, col, "outlier_check",
                                       f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%)",
                                       int(outlier_count)))
    return alerts


def _check_datatype_mismatches(df: pd.DataFrame) -> list[QualityAlert]:
    alerts = []
    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
            numeric_pct = numeric_count / len(non_null) * 100
            if 30 < numeric_pct < 100:
                alerts.append(QualityAlert("warning", col, "dtype_mismatch",
                                           f"Column '{col}' has mixed types: {numeric_pct:.0f}% could be numeric",
                                           round(numeric_pct, 1)))
    return alerts


def run_quality_checks(df: pd.DataFrame) -> QualityReport:
    """Run all data quality checks and compute scores."""
    all_alerts: list[QualityAlert] = []
    all_alerts.extend(_check_nulls(df))
    all_alerts.extend(_check_duplicates(df))
    all_alerts.extend(_check_negative_values(df))
    all_alerts.extend(_check_outliers(df))
    all_alerts.extend(_check_datatype_mismatches(df))

    total_checks = len(df.columns) * 5
    critical_count = sum(1 for a in all_alerts if a.severity == "critical")
    warning_count = sum(1 for a in all_alerts if a.severity == "warning")
    info_count = sum(1 for a in all_alerts if a.severity == "info")
    failed = critical_count + warning_count
    passed = total_checks - failed

    penalty = critical_count * 15 + warning_count * 5 + info_count * 1
    score = max(0.0, min(100.0, 100.0 - penalty))

    column_scores = {}
    for col in df.columns:
        col_alerts = [a for a in all_alerts if a.column == col]
        col_penalty = sum(15 if a.severity == "critical" else 5 if a.severity == "warning" else 1
                          for a in col_alerts)
        column_scores[col] = max(0.0, min(100.0, 100.0 - col_penalty))

    summary = {
        "total_alerts": len(all_alerts),
        "critical_alerts": critical_count,
        "warning_alerts": warning_count,
        "info_alerts": info_count,
        "completeness": round((1 - df.isna().mean().mean()) * 100, 2),
        "uniqueness": round(
            (1 - df.duplicated().mean()) * 100, 2) if len(df) > 0 else 100.0,
    }

    return QualityReport(
        overall_score=round(score, 1),
        total_checks=total_checks,
        passed_checks=passed,
        failed_checks=failed,
        alerts=all_alerts,
        column_scores=column_scores,
        summary=summary,
    )


def quality_report_to_dict(report: QualityReport) -> dict[str, Any]:
    return {
        "overall_score": report.overall_score,
        "total_checks": report.total_checks,
        "passed_checks": report.passed_checks,
        "failed_checks": report.failed_checks,
        "summary": report.summary,
        "column_scores": report.column_scores,
        "alerts": [
            {"severity": a.severity, "column": a.column, "rule": a.rule,
             "message": a.message, "value": a.value}
            for a in report.alerts
        ],
    }

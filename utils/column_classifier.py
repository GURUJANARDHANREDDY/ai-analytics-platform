"""Smart column classifier – distinguishes IDs, codes, and dimensions from real measures.

Every module that needs 'meaningful numeric columns' should call
get_measure_columns(df) instead of df.select_dtypes(include='number').
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|row.?id|order.?id|trans(action)?.?id|invoice.?id|"
    r"product.?id|customer.?id|user.?id|employee.?id|item.?id|ticket.?id|"
    r"account.?id|session.?id|record.?id|"
    r"index$|^idx$|^pk$|^fk$|^key$|^sk$|serial|"
    r"postal.?code|zip.?code|^zip$|pin.?code|area.?code|"
    r"phone|fax|^tel$|mobile|"
    r"ssn|^ein$|^tin$|license|passport|"
    r"^code$|_code$|^no$|_no$|^num$|_num$|number$)",
    re.IGNORECASE,
)


def is_id_or_code(series: pd.Series) -> bool:
    """Return True if a numeric column is actually an identifier or code, not a measure."""
    col_name = str(series.name).lower().strip()

    if _ID_PATTERNS.search(col_name):
        return True

    if not pd.api.types.is_numeric_dtype(series):
        return False

    clean = series.dropna()
    if len(clean) == 0:
        return False

    total = len(clean)
    unique_ratio = clean.nunique() / total if total > 0 else 0

    if unique_ratio > 0.95 and total > 20:
        if pd.api.types.is_integer_dtype(clean):
            diffs = clean.sort_values().diff().dropna()
            if len(diffs) > 0 and (diffs == 1).mean() > 0.9:
                return True
            if clean.min() >= 0 and unique_ratio > 0.99:
                return True

    if pd.api.types.is_integer_dtype(clean):
        all_positive = (clean >= 0).all()
        high_unique = unique_ratio > 0.8
        no_meaningful_stats = clean.std() / clean.mean() < 0.3 if clean.mean() != 0 else False
        if all_positive and high_unique and total > 50:
            sorted_vals = clean.sort_values().values
            diffs = np.diff(sorted_vals)
            if len(diffs) > 0 and np.std(diffs) < 2:
                return True

    return False


def get_measure_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are actual business measures (not IDs or codes)."""
    measures = []
    for col in df.select_dtypes(include="number").columns:
        if not is_id_or_code(df[col]):
            measures.append(col)
    return measures


def get_id_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are identifiers or codes."""
    ids = []
    for col in df.select_dtypes(include="number").columns:
        if is_id_or_code(df[col]):
            ids.append(col)
    return ids


def get_dimension_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for grouping."""
    dims = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique()
        if 2 <= nunique <= 50:
            dims.append(col)
    return dims


def classify_all_columns(df: pd.DataFrame) -> dict[str, str]:
    """Classify every column as: measure, id, dimension, datetime, boolean, text, high_cardinality."""
    result = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            result[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            result[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            result[col] = "id" if is_id_or_code(s) else "measure"
        elif s.dtype == "object" or isinstance(s.dtype, pd.CategoricalDtype):
            nunique = s.nunique()
            total = len(s)
            if total > 0 and nunique / total > 0.5 and nunique > 50:
                result[col] = "high_cardinality"
            else:
                result[col] = "dimension"
        else:
            result[col] = "other"
    return result

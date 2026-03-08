"""Auto SQL Generator – converts natural language queries to SQL and equivalent pandas code."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def generate_sql(query: str, df: pd.DataFrame, table_name: str = "dataset") -> dict[str, Any]:
    """Convert a natural language query to SQL and execute equivalent pandas."""
    columns_info = ", ".join(f"{col} ({df[col].dtype})" for col in df.columns)

    if HF_AVAILABLE:
        prompt = (
            f"Given a table '{table_name}' with columns: {columns_info}\n\n"
            f"User question: {query}\n\n"
            "1. Write the SQL query to answer this question.\n"
            "2. Write equivalent pandas code using a DataFrame called `df`. Store result in `result`.\n\n"
            "Format:\nSQL:\n```sql\n<query>\n```\n\nPandas:\n```python\n<code>\n```"
        )
        response = call_chat([
            {"role": "system", "content": "You convert natural language to SQL and pandas code."},
            {"role": "user", "content": prompt},
        ])
        if response and not response.startswith("[HuggingFace Error"):
            return _parse_response(response, df)

    return _fallback_sql(query, df, table_name)


def _parse_response(response: str, df: pd.DataFrame) -> dict[str, Any]:
    import re
    sql_blocks = re.findall(r"```sql\s*\n(.*?)```", response, re.DOTALL)
    py_blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)

    sql_query = sql_blocks[0].strip() if sql_blocks else ""
    raw_code = py_blocks[0].strip() if py_blocks else ""
    pandas_code = "\n".join(
        line for line in raw_code.split("\n")
        if not line.strip().startswith(("import ", "from ", "print("))
    ).strip() or raw_code

    result_data = None
    error = None
    if pandas_code:
        cleaned = "\n".join(
            line for line in pandas_code.split("\n")
            if not line.strip().startswith(("import ", "from ", "print(", "#"))
        )
        if not cleaned.strip():
            cleaned = pandas_code
        try:
            import numpy as np
            local_ns: dict[str, Any] = {}
            exec(cleaned, {"pd": pd, "np": np, "df": df.copy(), "__builtins__": {
                "len": len, "str": str, "int": int, "float": float, "list": list,
                "dict": dict, "sorted": sorted, "round": round, "abs": abs,
                "sum": sum, "min": min, "max": max, "True": True, "False": False, "None": None,
                "print": lambda *a, **k: None,
            }}, local_ns)
            result_data = local_ns.get("result")
        except Exception as e:
            error = str(e)

    return {
        "sql": sql_query,
        "pandas_code": pandas_code,
        "result": result_data,
        "error": error,
        "explanation": response,
    }


def _fallback_sql(query: str, df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    q = query.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    result_data = None

    if ("top" in q or "highest" in q) and categorical_cols and numeric_cols:
        cat, num = categorical_cols[0], numeric_cols[0]
        for c in categorical_cols:
            if c.lower() in q:
                cat = c
                break
        for c in numeric_cols:
            if c.lower() in q:
                num = c
                break
        sql = f"SELECT {cat}, SUM({num}) as total_{num}\nFROM {table_name}\nGROUP BY {cat}\nORDER BY total_{num} DESC\nLIMIT 10"
        code = f"result = df.groupby('{cat}')['{num}'].sum().sort_values(ascending=False).head(10).reset_index()"
        try:
            result_data = df.groupby(cat)[num].sum().sort_values(ascending=False).head(10).reset_index()
        except Exception:
            pass
        return {"sql": sql, "pandas_code": code, "result": result_data, "error": None, "explanation": ""}

    if ("average" in q or "mean" in q) and numeric_cols:
        col = numeric_cols[0]
        for c in numeric_cols:
            if c.lower() in q:
                col = c
                break
        sql = f"SELECT AVG({col}) as avg_{col}\nFROM {table_name}"
        code = f"result = df['{col}'].mean()"
        try:
            result_data = df[col].mean()
        except Exception:
            pass
        return {"sql": sql, "pandas_code": code, "result": result_data, "error": None, "explanation": ""}

    if ("total" in q or "sum" in q) and numeric_cols:
        col = numeric_cols[0]
        for c in numeric_cols:
            if c.lower() in q:
                col = c
                break
        sql = f"SELECT SUM({col}) as total_{col}\nFROM {table_name}"
        code = f"result = df['{col}'].sum()"
        try:
            result_data = df[col].sum()
        except Exception:
            pass
        return {"sql": sql, "pandas_code": code, "result": result_data, "error": None, "explanation": ""}

    sql = f"SELECT *\nFROM {table_name}\nLIMIT 10"
    code = "result = df.head(10)"
    result_data = df.head(10)
    return {"sql": sql, "pandas_code": code, "result": result_data, "error": None,
            "explanation": "Set HF_API_TOKEN for full AI SQL generation."}

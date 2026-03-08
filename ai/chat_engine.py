"""Chat with Data Engine – natural language interface using Hugging Face LLM."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ai.llm_client import call_chat, HF_AVAILABLE


def _get_schema_description(df: pd.DataFrame) -> str:
    lines = [f"DataFrame with {len(df)} rows and {len(df.columns)} columns:\n"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().head(3).tolist())
        lines.append(f"  - {col} ({dtype}): sample values = {sample}")
    return "\n".join(lines)


def _extract_code(response: str) -> str:
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    lines = []
    for line in response.split("\n"):
        stripped = line.strip()
        if (stripped.startswith("df") or stripped.startswith("result") or
                stripped.startswith("import") or "=" in stripped or stripped.startswith("print")):
            lines.append(line)
    return "\n".join(lines) if lines else ""


def _safe_execute(code: str, df: pd.DataFrame) -> Any:
    import numpy as np
    safe_globals = {
        "__builtins__": {
            "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "tuple": tuple, "range": range,
            "sorted": sorted, "round": round, "abs": abs, "sum": sum,
            "min": min, "max": max, "enumerate": enumerate, "zip": zip,
            "print": print, "True": True, "False": False, "None": None,
            "isinstance": isinstance, "type": type,
        },
        "pd": pd, "np": np, "df": df.copy(),
    }
    local_vars: dict[str, Any] = {}
    exec(code, safe_globals, local_vars)
    return local_vars.get("result", local_vars.get("output", local_vars.get("answer", None)))


def chat_with_data(query: str, df: pd.DataFrame, history: list[dict] | None = None) -> dict[str, Any]:
    schema = _get_schema_description(df)
    prompt = (
        f"You are a data analyst assistant. The user asks questions about a pandas DataFrame called `df`.\n\n"
        f"{schema}\n\nUser question: {query}\n\n"
        "Write Python code using pandas to answer the question. Store the final answer in a variable called `result`.\n"
        "Only use the `df` variable. No file I/O. Wrap code in ```python ``` tags."
    )

    if not HF_AVAILABLE:
        return _fallback_query(query, df)

    try:
        llm_output = call_chat([
            {"role": "system", "content": "You are a data analyst. Write pandas code to answer questions about dataframes."},
            {"role": "user", "content": prompt},
        ])
    except Exception:
        return _fallback_query(query, df)

    if not llm_output or llm_output.startswith("[HuggingFace Error"):
        return _fallback_query(query, df)

    code = _extract_code(llm_output)
    if not code:
        return {"answer": llm_output, "code": None, "chart": None, "success": True}

    try:
        result = _safe_execute(code, df)
        answer = _format_result(result)
        chart = _try_generate_chart(result, query)
        return {"answer": answer, "code": code, "result_data": result if isinstance(result, pd.DataFrame) else None,
                "chart": chart, "success": True}
    except Exception as exc:
        return {"answer": f"I understood your question but encountered an error: {exc}",
                "code": code, "chart": None, "success": False}


def _format_result(result: Any) -> str:
    if result is None:
        return "Query executed successfully but returned no result."
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False, max_rows=20) if len(result) > 0 else "No matching data found."
    if isinstance(result, pd.Series):
        return result.to_string(max_rows=20)
    return str(result)


def _try_generate_chart(result: Any, query: str) -> Any:
    try:
        import plotly.express as px
        if isinstance(result, pd.DataFrame) and len(result) > 0 and len(result.columns) >= 2:
            cols = result.columns.tolist()
            num_cols = result.select_dtypes(include="number").columns.tolist()
            cat_cols = [c for c in cols if c not in num_cols]
            if cat_cols and num_cols:
                fig = px.bar(result.head(20), x=cat_cols[0], y=num_cols[0],
                             color_discrete_sequence=["#6366f1"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(26,29,41,0.8)")
                return fig
    except Exception:
        pass
    return None


def _fallback_query(query: str, df: pd.DataFrame) -> dict[str, Any]:
    query_lower = query.lower()
    try:
        if "how many" in query_lower or "count" in query_lower:
            return {"answer": f"The dataset has {len(df):,} rows and {len(df.columns)} columns.",
                    "code": None, "chart": None, "success": True}
        if "columns" in query_lower or "column" in query_lower:
            return {"answer": f"Columns: {', '.join(df.columns.tolist())}",
                    "code": None, "chart": None, "success": True}
        if "describe" in query_lower or "summary" in query_lower or "statistics" in query_lower:
            return {"answer": df.describe().to_string(), "code": None, "chart": None, "success": True}
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if "average" in query_lower or "mean" in query_lower:
            for col in numeric_cols:
                if col.lower() in query_lower:
                    return {"answer": f"The average {col} is {df[col].mean():,.2f}",
                            "code": None, "chart": None, "success": True}
            if numeric_cols:
                avgs = {col: f"{df[col].mean():,.2f}" for col in numeric_cols[:5]}
                return {"answer": f"Averages: {avgs}", "code": None, "chart": None, "success": True}
        if "top" in query_lower or "highest" in query_lower or "most" in query_lower:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            for col in cat_cols:
                if col.lower() in query_lower:
                    top = df[col].value_counts().head(5)
                    return {"answer": f"Top {col}:\n{top.to_string()}",
                            "code": None, "chart": None, "success": True}
        if "total" in query_lower or "sum" in query_lower:
            for col in numeric_cols:
                if col.lower() in query_lower:
                    return {"answer": f"Total {col}: {df[col].sum():,.2f}",
                            "code": None, "chart": None, "success": True}
        return {
            "answer": (f"Dataset: {len(df):,} rows, {len(df.columns)} columns.\n"
                       f"Columns: {', '.join(df.columns.tolist()[:15])}\n\n"
                       "Try asking about averages, totals, top items, or column details. "
                       "Set HF_API_TOKEN in .env for full AI-powered queries."),
            "code": None, "chart": None, "success": True}
    except Exception as e:
        return {"answer": f"Error processing query: {e}", "code": None, "chart": None, "success": False}

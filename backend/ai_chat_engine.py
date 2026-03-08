"""Chat-with-data engine – converts natural language questions into Pandas operations."""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from backend.config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ChatResponse:
    """Wrapper for a chat answer with optional DataFrame result."""

    def __init__(self, answer: str, data: Optional[pd.DataFrame] = None) -> None:
        self.answer = answer
        self.data = data

    def __repr__(self) -> str:
        return f"ChatResponse(answer={self.answer!r}, has_data={self.data is not None})"


def ask_question(df: pd.DataFrame, question: str) -> ChatResponse:
    """Answer a natural-language question about *df*.

    Tries HuggingFace first, then Gemini, then OpenAI, then keyword fallback.
    """
    if not question or not question.strip():
        return ChatResponse("Please enter a question about the dataset.")

    # HuggingFace — try first when configured
    if settings.hf_api_token:
        response = _try_huggingface(df, question)
        if response:
            return response

    # Gemini — try second
    if settings.gemini_api_key:
        response = _try_gemini(df, question)
        if response:
            return response

    # OpenAI strategies — only if others are NOT configured
    if not settings.gemini_api_key and not settings.hf_api_token and settings.openai_api_key:
        response = _try_pandasai(df, question)
        if response:
            return response

        response = _try_openai(df, question)
        if response:
            return response

    return _keyword_fallback(df, question)


# ── HuggingFace strategy ─────────────────────────────────────────────────────

_HF_PROMPT = """You are a data analyst assistant. A pandas DataFrame called `df` is already loaded in memory.

EXACT column names (case-sensitive): {columns}

Column types:
{col_info}

Sample data (first 3 rows):
{sample}

Numeric stats:
{stats}

User question: {question}

STRICT RULES:
1. ALWAYS start your reply with a plain-English summary answering the question (2-3 sentences with actual numbers).
2. If you need to compute something, write a SIMPLE pandas expression after your explanation, wrapped in ```python ... ```.
3. Store the result in a variable called `result`. Example:
   ```python
   result = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
   ```
4. Use ONLY exact column names from above (case-sensitive, with spaces/underscores exactly as shown).
5. NEVER use: import, print, read_csv, lambda, apply, rolling, or multi-step code.
6. Keep code to 1-2 lines MAX. Use simple groupby, sum, mean, nlargest, nsmallest, value_counts, describe.
7. If the question is simple (e.g. "what is the total sales?"), just answer with text and numbers — no code needed.
8. NEVER return only code with no explanation."""


def _hf_chat_call(messages: list, model: str, token: str, max_tokens: int = 1024) -> str:
    """Call HF Inference API directly via requests (bypasses SSL issues)."""
    import requests as _req
    import warnings
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    url = "https://router.huggingface.co/novita/v3/openai/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.3}
    resp = _req.post(url, headers=headers, json=payload, verify=False, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _clean_ai_code(code: str) -> str:
    """Strip imports, prints, comments, and markdown from AI-generated code."""
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ", "print(", "#", "```")):
            continue
        if stripped == "```python" or stripped == "```":
            continue
        if stripped.startswith("plt.") or stripped.startswith("fig"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    if "lambda" in cleaned and "apply" in cleaned:
        cleaned = ""
    return cleaned


def _clean_response_text(text: str) -> str:
    """Remove code blocks from AI response and return only the text explanation."""
    import re
    cleaned = re.sub(r"```(?:python)?\s*\n.*?```", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^(result|output|df\[|df\.).*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    if not cleaned or len(cleaned) < 5:
        text_no_code = text.replace("```python", "").replace("```", "")
        lines = [l for l in text_no_code.split("\n") if not l.strip().startswith(("result", "df.", "df[", "import"))]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _safe_exec(code: str, df: pd.DataFrame) -> any:
    """Execute multi-line code safely and return the result variable."""
    import numpy as np
    safe_globals = {
        "__builtins__": {
            "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "tuple": tuple, "range": range,
            "sorted": sorted, "round": round, "abs": abs, "sum": sum,
            "min": min, "max": max, "enumerate": enumerate, "zip": zip,
            "True": True, "False": False, "None": None,
            "isinstance": isinstance, "type": type, "map": map, "filter": filter,
        },
        "pd": pd, "np": np, "df": df.copy(),
    }
    local_vars = {}
    exec(code, safe_globals, local_vars)
    return local_vars.get("result", local_vars.get("output", local_vars.get("answer")))


def _extract_code_from_response(text: str) -> tuple[str, str]:
    """Extract code and explanation from AI response."""
    import re

    # Check for ```python code blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        explanation = re.sub(r"```(?:python)?\s*\n.*?```", "", text, flags=re.DOTALL).strip()
        return code, explanation

    # Check for CODE: prefix
    if "CODE:" in text:
        parts = text.split("CODE:", 1)
        return parts[1].strip().strip("`").strip(), parts[0].strip()

    # Check for lines that look like pandas code
    code_lines = []
    text_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s and (s.startswith("result") or s.startswith("df[") or s.startswith("df.") or
                  "=" in s and ("df" in s or "pd." in s)):
            code_lines.append(line)
        else:
            text_lines.append(line)

    if code_lines:
        return "\n".join(code_lines), "\n".join(text_lines).strip()

    return "", text


def _try_huggingface(df: pd.DataFrame, question: str) -> Optional[ChatResponse]:
    if not settings.hf_api_token:
        return None
    try:
        columns = ", ".join(df.columns.tolist())
        col_info = "\n".join(f"  - {c}: {df[c].dtype}" for c in df.columns)
        sample = df.head(3).to_string(index=False)
        from utils.column_classifier import get_measure_columns
        measure_cols = get_measure_columns(df)
        stats = df[measure_cols].describe().to_string() if measure_cols else "No numeric measure columns"

        prompt = _HF_PROMPT.format(
            columns=columns, col_info=col_info, sample=sample,
            stats=stats, question=question,
        )

        text = _hf_chat_call(
            messages=[
                {"role": "system", "content": (
                    "You are a helpful data analyst. ALWAYS start with a clear plain-English answer with numbers. "
                    "Only include simple pandas code if needed. NEVER return just code without explanation. "
                    "Use exact column names. Keep code to 1 line. Store results in `result`."
                )},
                {"role": "user", "content": prompt},
            ],
            model=settings.hf_model,
            token=settings.hf_api_token,
        )

        code, explanation = _extract_code_from_response(text)

        if code:
            code = _clean_ai_code(code)
            if code:
                try:
                    result = _safe_exec(code, df)
                    answer = explanation if explanation else f"Here's what I found for: *{question}*"
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        if isinstance(result, pd.Series):
                            result = result.reset_index()
                            result.columns = [result.columns[0], "Value"] if len(result.columns) == 2 else result.columns
                        return ChatResponse(answer, data=result.head(50))
                    if result is not None:
                        return ChatResponse(f"{answer}\n\n**Answer:** {result}")
                    elif explanation:
                        return ChatResponse(explanation)
                except Exception as exc:
                    logger.warning("AI code execution failed: %s | Code: %s", exc, code[:200])
                    if explanation:
                        return ChatResponse(explanation)

        clean_text = _clean_response_text(text)
        if clean_text:
            return ChatResponse(clean_text)
        return None

    except Exception as exc:
        logger.warning("HuggingFace chat failed: %s", exc)
        return None


# ── Gemini strategy ──────────────────────────────────────────────────────────

_GEMINI_PROMPT = """You are a data analyst. Given this DataFrame info, answer the user's question directly and concisely.

Columns and dtypes:
{col_info}

First 5 rows:
{sample}

Basic stats:
{stats}

Question: {question}

If the question asks for data, write a short Python pandas expression (variable is `df`) prefixed with CODE: on its own line.
Otherwise just answer directly."""


def _try_gemini(df: pd.DataFrame, question: str) -> Optional[ChatResponse]:
    if not settings.gemini_api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        col_info = "\n".join(f"  - {c}: {df[c].dtype}" for c in df.columns)
        sample = df.head(5).to_string(index=False)
        stats = df.describe().to_string() if len(df.select_dtypes(include="number").columns) > 0 else "No numeric columns"

        prompt = _GEMINI_PROMPT.format(
            col_info=col_info, sample=sample, stats=stats, question=question
        )

        response = model.generate_content(
            prompt,
            request_options={"timeout": 15},
        )
        text = (response.text or "").strip()

        if "CODE:" in text:
            lines = text.split("\n")
            answer_lines = []
            code_line = None
            for line in lines:
                if line.strip().startswith("CODE:"):
                    code_line = line.strip().replace("CODE:", "").strip().strip("`").strip()
                else:
                    answer_lines.append(line)

            if code_line and _is_safe_expression(code_line):
                try:
                    result = eval(code_line, {"__builtins__": {}}, {"df": df, "pd": pd})
                    answer = "\n".join(answer_lines).strip() or f"Result for: *{question}*"
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        if isinstance(result, pd.Series):
                            result = result.to_frame()
                        return ChatResponse(answer, data=result.head(50))
                    return ChatResponse(f"{answer}\n\n**Answer:** {result}")
                except Exception:
                    pass

            return ChatResponse("\n".join(answer_lines).strip() or text)

        return ChatResponse(text)

    except Exception as exc:
        logger.warning("Gemini chat failed: %s", exc)
        return None


# ── PandasAI strategy ────────────────────────────────────────────────────────

def _try_pandasai(df: pd.DataFrame, question: str) -> Optional[ChatResponse]:
    if not settings.openai_api_key:
        return None
    try:
        from pandasai import SmartDataframe
        from pandasai.llm import OpenAI as PandasAIOpenAI

        llm = PandasAIOpenAI(api_token=settings.openai_api_key, model=settings.openai_model)
        sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": True})
        result = sdf.chat(question)

        if isinstance(result, pd.DataFrame):
            return ChatResponse(f"Here are the results for: *{question}*", data=result)
        return ChatResponse(str(result))

    except ImportError:
        logger.info("pandasai not installed – skipping PandasAI strategy.")
        return None
    except Exception as exc:
        logger.warning("PandasAI failed: %s", exc)
        return _check_quota_error(exc)


# ── OpenAI code-generation strategy ──────────────────────────────────────────

_CODE_PROMPT_TEMPLATE = """You are a Python data analyst. Given the following DataFrame column info, write a SINGLE Python expression 
(using pandas) that answers the user's question. The DataFrame variable is named `df`.

Columns and dtypes:
{col_info}

First 3 rows:
{sample}

Question: {question}

Reply ONLY with the Python expression, no explanation, no markdown fences."""


def _try_openai(df: pd.DataFrame, question: str) -> Optional[ChatResponse]:
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        col_info = "\n".join(f"  - {c}: {df[c].dtype}" for c in df.columns)
        sample = df.head(3).to_string(index=False)

        prompt = _CODE_PROMPT_TEMPLATE.format(col_info=col_info, sample=sample, question=question)

        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        code = (resp.choices[0].message.content or "").strip().strip("`").strip()

        if not _is_safe_expression(code):
            logger.warning("LLM generated unsafe code – refusing to execute: %s", code)
            return ChatResponse("Sorry, I couldn't generate a safe query for that question.")

        result = eval(code, {"__builtins__": {}}, {"df": df, "pd": pd})  # noqa: S307

        if isinstance(result, pd.DataFrame):
            return ChatResponse(f"Result for: *{question}*\n\n`{code}`", data=result)
        if isinstance(result, pd.Series):
            return ChatResponse(
                f"Result for: *{question}*\n\n`{code}`",
                data=result.to_frame(),
            )
        return ChatResponse(f"**Answer:** {result}\n\n_Generated code:_ `{code}`")

    except Exception as exc:
        logger.warning("OpenAI chat strategy failed: %s", exc)
        return _check_quota_error(exc)


_BLOCKED_TOKENS = {"exec(", "eval(", "open(", "os.", "sys.", "subprocess", "__", "shutil", "rm ", "del ", "rmdir"}


def _is_safe_expression(code: str) -> bool:
    """Basic safety check – reject code that uses dangerous tokens."""
    lower = code.lower()
    return not any(tok in lower for tok in _BLOCKED_TOKENS)


def _check_quota_error(exc: Exception) -> Optional[ChatResponse]:
    """Return a helpful message if the error is a quota/billing issue."""
    msg = str(exc).lower()
    if "quota" in msg or "billing" in msg or "429" in msg or "insufficient" in msg:
        if "generativelanguage" in msg or "gemini" in msg or "google" in msg:
            logger.warning("Gemini quota exceeded – falling through to keyword engine.")
            return None
        return ChatResponse(
            "**API quota exceeded.** Your account has no credits remaining.\n\n"
            "**Free alternative:** Get a free Gemini key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
            "and add it to your `.env` file as `GEMINI_API_KEY=...`"
        )
    return None


# ── Enhanced keyword fallback ─────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, question: str) -> Optional[str]:
    """Try to find a column name mentioned in the question."""
    q_lower = question.lower()
    for col in df.columns:
        if col.lower() in q_lower:
            return col
    for col in df.columns:
        col_words = re.split(r'[_\s-]+', col.lower())
        if any(w in q_lower for w in col_words if len(w) > 2):
            return col
    return None


def _find_number(question: str) -> Optional[int]:
    """Extract a number from the question (e.g., 'top 5' → 5)."""
    match = re.search(r'\b(\d+)\b', question)
    return int(match.group(1)) if match else None


def _keyword_fallback(df: pd.DataFrame, question: str) -> ChatResponse:
    """Enhanced pattern-matching answerer when no LLM is available."""
    q = question.lower().strip()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    mentioned_col = _find_column(df, question)
    n = _find_number(question) or 5

    # Row / column counts
    if any(w in q for w in ("how many rows", "row count", "number of rows", "total rows", "count of rows")):
        return ChatResponse(f"The dataset has **{len(df):,}** rows.")

    if any(w in q for w in ("how many columns", "column count", "number of columns", "total columns")):
        return ChatResponse(f"The dataset has **{len(df.columns)}** columns: {', '.join(all_cols)}")

    if any(w in q for w in ("column names", "list columns", "show columns", "what columns", "which columns")):
        return ChatResponse(f"The columns are: {', '.join(all_cols)}")

    if any(w in q for w in ("data types", "dtypes", "column types")):
        dtypes = "\n".join(f"- **{c}**: {df[c].dtype}" for c in all_cols)
        return ChatResponse(f"Column data types:\n\n{dtypes}")

    # Missing values
    if any(w in q for w in ("missing", "null", "nan", "empty")):
        missing = df.isnull().sum()
        if mentioned_col:
            return ChatResponse(f"**{mentioned_col}** has **{missing[mentioned_col]:,}** missing values.")
        missing_df = missing[missing > 0].sort_values(ascending=False)
        if missing_df.empty:
            return ChatResponse("There are **no missing values** in the dataset.")
        result = missing_df.reset_index()
        result.columns = ["Column", "Missing Count"]
        return ChatResponse("Columns with missing values:", data=result)

    # Describe / summary
    if any(w in q for w in ("describe", "summary", "statistics", "stats")):
        if mentioned_col and mentioned_col in numeric_cols:
            desc = df[mentioned_col].describe().to_frame().T
            return ChatResponse(f"Summary of **{mentioned_col}**:", data=desc)
        return ChatResponse("Dataset summary:", data=df.describe().T.reset_index().rename(columns={"index": "Column"}))

    # Average / mean
    if any(w in q for w in ("average", "mean", "avg")):
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            return ChatResponse(f"The average of **{col}** is **{df[col].mean():,.2f}**.")

    # Sum / total
    if any(w in q for w in ("sum", "total")):
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            return ChatResponse(f"The sum of **{col}** is **{df[col].sum():,.2f}**.")

    # Max / highest / largest
    if any(w in q for w in ("max", "maximum", "highest", "largest", "greatest", "best")):
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            max_val = df[col].max()
            max_row = df[df[col] == max_val].head(1)
            return ChatResponse(f"The maximum of **{col}** is **{max_val:,.2f}**.", data=max_row)

    # Min / lowest / smallest
    if any(w in q for w in ("min", "minimum", "lowest", "smallest", "least", "worst")):
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            min_val = df[col].min()
            min_row = df[df[col] == min_val].head(1)
            return ChatResponse(f"The minimum of **{col}** is **{min_val:,.2f}**.", data=min_row)

    # Top N / bottom N
    if "top" in q or "first" in q or "head" in q:
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            result = df.nlargest(n, col)
            return ChatResponse(f"Top {n} rows by **{col}**:", data=result)
        return ChatResponse(f"First {n} rows:", data=df.head(n))

    if "bottom" in q or "last" in q or "tail" in q:
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            result = df.nsmallest(n, col)
            return ChatResponse(f"Bottom {n} rows by **{col}**:", data=result)
        return ChatResponse(f"Last {n} rows:", data=df.tail(n))

    # Unique values / categories
    if any(w in q for w in ("unique", "distinct", "categories", "values of")):
        col = mentioned_col or all_cols[0]
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 30:
            vals = ", ".join(str(v) for v in unique_vals[:30])
            return ChatResponse(f"**{col}** has {len(unique_vals)} unique values: {vals}")
        return ChatResponse(f"**{col}** has **{len(unique_vals):,}** unique values (too many to list).")

    # Value counts / frequency / distribution
    if any(w in q for w in ("value counts", "frequency", "distribution", "breakdown", "count of", "how many of each")):
        col = mentioned_col or all_cols[0]
        counts = df[col].value_counts().head(n).reset_index()
        counts.columns = [col, "Count"]
        return ChatResponse(f"Top {n} value counts for **{col}**:", data=counts)

    # Group by
    if "group" in q or "by each" in q or "per " in q:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        group_col = None
        agg_col = None
        for c in cat_cols:
            if c.lower() in q:
                group_col = c
                break
        for c in numeric_cols:
            if c.lower() in q:
                agg_col = c
                break
        if group_col and agg_col:
            result = df.groupby(group_col)[agg_col].agg(["sum", "mean", "count"]).reset_index()
            return ChatResponse(f"**{agg_col}** grouped by **{group_col}**:", data=result)
        if group_col and numeric_cols:
            agg_col = numeric_cols[0]
            result = df.groupby(group_col)[agg_col].agg(["sum", "mean", "count"]).reset_index()
            return ChatResponse(f"**{agg_col}** grouped by **{group_col}**:", data=result)

    # Trend / growth
    if any(w in q for w in ("trend", "growth", "over time", "monthly", "yearly", "year over year", "yoy", "month over month")):
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        agg_col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if dt_cols and agg_col:
            dt = dt_cols[0]
            monthly = df.set_index(dt).resample("ME")[agg_col].sum().reset_index()
            monthly.columns = ["Period", f"Total {agg_col}"]
            monthly["Period"] = monthly["Period"].dt.strftime("%Y-%m")
            if len(monthly) >= 2:
                first = monthly[f"Total {agg_col}"].iloc[0]
                last = monthly[f"Total {agg_col}"].iloc[-1]
                change_pct = ((last - first) / abs(first) * 100) if first != 0 else 0
                direction = "grew" if change_pct > 0 else "declined"
                return ChatResponse(
                    f"**{agg_col}** {direction} by **{abs(change_pct):.1f}%** from {monthly['Period'].iloc[0]} to {monthly['Period'].iloc[-1]}.\n\n"
                    f"First period: **{first:,.2f}** | Last period: **{last:,.2f}**",
                    data=monthly,
                )
            return ChatResponse(f"Monthly trend for **{agg_col}**:", data=monthly)
        elif agg_col:
            return ChatResponse(f"No date column found to calculate trends. Total **{agg_col}**: **{df[agg_col].sum():,.2f}**")

    # Compare
    if any(w in q for w in ("compare", "comparison", "vs", "versus", "difference between")):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        agg_col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        group_col = None
        for c in cat_cols:
            if c.lower() in q:
                group_col = c
                break
        if not group_col and cat_cols:
            group_col = cat_cols[0]
        if group_col and agg_col:
            result = df.groupby(group_col)[agg_col].agg(["sum", "mean", "count"]).round(2).sort_values("sum", ascending=False).reset_index()
            result.columns = [group_col, f"Total {agg_col}", f"Avg {agg_col}", "Count"]
            return ChatResponse(f"Comparison of **{agg_col}** by **{group_col}**:", data=result)

    # Correlation
    if any(w in q for w in ("correlation", "correlate", "related")):
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            return ChatResponse("Correlation matrix:", data=corr.reset_index().rename(columns={"index": "Column"}))

    # Shape
    if "shape" in q or "size" in q or "dimensions" in q:
        return ChatResponse(f"The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**.")

    # Show / display rows
    if any(w in q for w in ("show", "display", "print", "see")):
        return ChatResponse(f"Here are {n} rows:", data=df.head(n))

    # Median
    if "median" in q:
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            return ChatResponse(f"The median of **{col}** is **{df[col].median():,.2f}**.")

    # Standard deviation
    if any(w in q for w in ("std", "standard deviation", "deviation")):
        col = mentioned_col if mentioned_col and mentioned_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            return ChatResponse(f"The standard deviation of **{col}** is **{df[col].std():,.2f}**.")

    # Fallback
    return ChatResponse(
        "I couldn't interpret that question. Try asking things like:\n\n"
        "- *How many rows are in the dataset?*\n"
        "- *What is the average of Sales?*\n"
        "- *Show top 5 rows by Revenue*\n"
        "- *What are the unique values of Category?*\n"
        "- *Show missing values*\n"
        "- *Describe the dataset*\n"
        "- *Group Sales by Region*"
    )

"""Dataset loading, validation, and caching layer."""

from __future__ import annotations

import functools
import io
from pathlib import Path
from typing import BinaryIO, Optional

import pandas as pd
import pyarrow.csv as pv

from utils.file_utils import validate_file_extension, validate_file_size
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_dataframe_cache: dict[str, pd.DataFrame] = {}


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""


def load_csv_from_file(
    file: BinaryIO,
    filename: str,
    cache_key: Optional[str] = None,
    use_pyarrow: bool = True,
) -> pd.DataFrame:
    """Read a CSV upload into a DataFrame with validation and caching.

    Args:
        file: File-like binary stream.
        filename: Original filename (used for extension check).
        cache_key: Optional key for in-memory caching.
        use_pyarrow: Use PyArrow for faster CSV parsing when possible.

    Returns:
        A validated ``pd.DataFrame``.

    Raises:
        DataLoadError: On validation failure or parse errors.
    """
    if cache_key and cache_key in _dataframe_cache:
        logger.info("Returning cached DataFrame for key=%s", cache_key)
        return _dataframe_cache[cache_key]

    if not validate_file_extension(filename):
        raise DataLoadError(f"Unsupported file type: {filename}. Accepted: CSV, XLSX, XLS.")

    if not validate_file_size(file):
        raise DataLoadError("File exceeds the 100 MB size limit.")

    try:
        raw = file.read()
        file.seek(0)
    except Exception as exc:
        raise DataLoadError(f"Could not read uploaded file: {exc}") from exc

    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = _parse_excel_bytes(raw)
    else:
        df = _parse_csv_bytes(raw, use_pyarrow=use_pyarrow)

    if df.empty:
        raise DataLoadError("The uploaded CSV is empty.")

    logger.info("Loaded DataFrame: %d rows × %d columns", len(df), len(df.columns))

    _auto_parse_dates(df)

    if cache_key:
        _dataframe_cache[cache_key] = df

    return df


def load_csv_from_path(path: Path) -> pd.DataFrame:
    """Read a CSV from a local path using PyArrow."""
    try:
        table = pv.read_csv(str(path))
        df = table.to_pandas()
    except Exception:
        raw = path.read_bytes()
        encoding = _detect_encoding(raw)
        df = pd.read_csv(path, encoding=encoding)
    _auto_parse_dates(df)
    return df


def _parse_excel_bytes(raw: bytes) -> pd.DataFrame:
    """Parse Excel (.xlsx/.xls) bytes into a DataFrame."""
    try:
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
        except Exception as exc:
            raise DataLoadError(f"Failed to parse Excel file: {exc}") from exc
    return df


def _parse_csv_bytes(raw: bytes, use_pyarrow: bool = True) -> pd.DataFrame:
    """Try every reasonable strategy to turn raw CSV bytes into a DataFrame."""
    encoding = _detect_encoding(raw)
    logger.info("Detected file encoding: %s", encoding)

    # Strategy 1: PyArrow (fastest, but only supports UTF-8)
    if use_pyarrow and encoding in ("utf-8", "utf-8-sig"):
        try:
            table = pv.read_csv(io.BytesIO(raw))
            return table.to_pandas()
        except Exception:
            logger.info("PyArrow failed; trying Pandas.")

    # Strategy 2: Pandas with detected encoding
    for enc in dict.fromkeys([encoding, "utf-8", "cp1252", "latin-1"]):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            logger.info("Pandas read failed with encoding=%s, trying next.", enc)
            continue
        except Exception as exc:
            raise DataLoadError(f"Failed to parse CSV: {exc}") from exc

    raise DataLoadError(
        "Could not decode the CSV file with any supported encoding "
        "(tried UTF-8, CP1252, Latin-1). Please re-save the file as UTF-8."
    )


def _detect_encoding(raw: bytes) -> str:
    """Try UTF-8 first, then probe for BOM and common fallbacks."""
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"

    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    for enc in ("cp1252", "latin-1", "iso-8859-1"):
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue

    return "latin-1"


def clear_cache(key: Optional[str] = None) -> None:
    """Remove one or all cached DataFrames."""
    if key:
        _dataframe_cache.pop(key, None)
    else:
        _dataframe_cache.clear()
    logger.info("Cache cleared (key=%s)", key or "ALL")


def _auto_parse_dates(df: pd.DataFrame) -> None:
    """Attempt in-place conversion of object columns that look like dates."""
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            converted = pd.to_datetime(df[col], format="mixed", errors="coerce")
            if converted.notna().mean() > 0.5:
                df[col] = converted
                logger.info("Auto-parsed column '%s' as datetime.", col)
        except Exception:
            pass

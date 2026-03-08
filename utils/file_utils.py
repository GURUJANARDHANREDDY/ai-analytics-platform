"""File validation and I/O utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

from utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file_extension(filename: str) -> bool:
    """Return *True* if the file has an allowed extension."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected file with extension %s", ext)
        return False
    return True


def validate_file_size(file: BinaryIO) -> bool:
    """Return *True* if the file is within the size limit.

    The stream position is reset after measuring.
    """
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE_BYTES:
        logger.warning("File too large: %.2f MB (limit %d MB)", size / (1024 * 1024), MAX_FILE_SIZE_MB)
        return False
    return True


def file_hash(file: BinaryIO, algorithm: str = "sha256") -> str:
    """Compute a hex-digest hash for cache-keying uploaded files.

    The stream position is reset after hashing.
    """
    h = hashlib.new(algorithm)
    file.seek(0)
    for chunk in iter(lambda: file.read(8192), b""):
        h.update(chunk)
    file.seek(0)
    return h.hexdigest()


def save_uploaded_file(file: BinaryIO, dest: Path) -> Path:
    """Persist an uploaded file to *dest* and return the written path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(file.read())
    file.seek(0)
    logger.info("Saved uploaded file to %s", dest)
    return dest

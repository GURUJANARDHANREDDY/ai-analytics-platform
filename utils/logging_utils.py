"""Centralized logging configuration for the AI Analytics Platform."""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console and optional file handlers.

    Args:
        name: Logger namespace (typically ``__name__``).
        level: Minimum severity level to capture.

    Returns:
        A ``logging.Logger`` instance ready for use.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create file log handler – logging to console only.")

    return logger

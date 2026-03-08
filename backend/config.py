"""Application-wide configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_secret(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets."""
    val = os.getenv(key, "")
    if val:
        return val.strip()
    try:
        import streamlit as st
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings populated from env vars at startup."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip())
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip())
    hf_api_token: str = field(default_factory=lambda: _get_secret("HF_API_TOKEN"))
    hf_model: str = field(default_factory=lambda: _get_secret("HF_MODEL", "meta-llama/llama-3.1-8b-instruct"))
    upload_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "uploads")
    max_file_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def has_ai_key(self) -> bool:
        """Return True if any AI provider key is configured."""
        return bool(self.openai_api_key or self.gemini_api_key or self.hf_api_token)

    def __post_init__(self) -> None:
        object.__setattr__(self, "upload_dir", Path(self.upload_dir))
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if self.hf_api_token:
            logger.info("Using Hugging Face as AI provider.")
        elif self.gemini_api_key:
            logger.info("Using Google Gemini as AI provider.")
        elif self.openai_api_key:
            logger.info("Using OpenAI as AI provider.")
        else:
            logger.warning(
                "No AI API key set (HF_API_TOKEN, OPENAI_API_KEY, or GEMINI_API_KEY). "
                "AI insights and chat will use the basic engine."
            )


settings = Settings()

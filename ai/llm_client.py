"""Shared Hugging Face LLM client – direct HTTP calls."""

from __future__ import annotations

import os
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _get_secret(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets (for cloud deployment)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


HF_TOKEN = _get_secret("HF_API_TOKEN")
HF_MODEL = _get_secret("HF_MODEL", "meta-llama/llama-3.1-8b-instruct")
HF_AVAILABLE = bool(HF_TOKEN)

_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"


def call_llm(prompt: str, max_tokens: int = 1500, temperature: float = 0.3) -> str:
    if not HF_AVAILABLE:
        return ""
    return call_chat([{"role": "user", "content": prompt}], max_tokens, temperature)


def call_chat(messages: list[dict[str, str]], max_tokens: int = 1500,
              temperature: float = 0.3) -> str:
    if not HF_AVAILABLE:
        return ""
    try:
        import requests
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {"model": HF_MODEL, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        resp = requests.post(_API_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[HuggingFace Error: {e}]"

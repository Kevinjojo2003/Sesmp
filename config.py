"""Centralized runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets (used on Streamlit Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default

# ── API Keys ──
GEMINI_API_KEY: str = _secret("GEMINI_API_KEY")
NEWS_API_KEY: str = _secret("NEWS_API_KEY")
REDDIT_CLIENT_ID: str = _secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET: str = _secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "stock_sentiment_bot/1.0")

# ── Paths ──
MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
DATA_DIR: str = os.getenv("DATA_DIR", "data")

# ── Model / Runtime ──
LSTM_SEQUENCE_LENGTH: int = int(os.getenv("LSTM_SEQUENCE_LENGTH", "60"))
NEWS_PAGE_SIZE: int = int(os.getenv("NEWS_PAGE_SIZE", "10"))
DEFAULT_TICKER: str = os.getenv("DEFAULT_TICKER", "AAPL")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro")

# ── Ensure directories exist ──
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

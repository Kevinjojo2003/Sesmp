"""Centralized runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
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

"""Batch trainer for a multi-ticker, N-year daily dataset run."""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import DATA_DIR
from train_lstm import train_lstm

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
]


def build_processed_dataset(ticker: str, years: int = 5) -> str:
    """Download daily data, compute indicators, save processed CSV."""
    period = f"{years}y"
    df = yf.Ticker(ticker).history(period=period, interval="1d")

    if df.empty or len(df) < 200:
        raise ValueError(f"Insufficient data for {ticker} ({len(df)} rows)")

    df.dropna(subset=["Close"], inplace=True)

    # ── Technical indicators ──
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd.get("MACD_12_26_9")
        df["MACD_Signal"] = macd.get("MACDs_12_26_9")
        df["MACD_Hist"] = macd.get("MACDh_12_26_9")

    bbands = ta.bbands(df["Close"], length=20)
    if bbands is not None:
        df["BB_Upper"] = bbands.get("BBU_20_2.0")
        df["BB_Middle"] = bbands.get("BBM_20_2.0")
        df["BB_Lower"] = bbands.get("BBL_20_2.0")

    df["Sentiment_Count"] = 0.0
    df = df.ffill().bfill().dropna()

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(path)
    logger.info("Dataset saved: %s (%d rows)", path, len(df))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM for multiple tickers.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    succeeded, failed = [], []

    for i, ticker in enumerate(args.tickers, 1):
        logger.info("=== [%d/%d] %s: building dataset ===", i, len(args.tickers), ticker)
        try:
            build_processed_dataset(ticker, years=args.years)
            logger.info("=== %s: training LSTM (%d epochs) ===", ticker, args.epochs)
            train_lstm(ticker, epochs=args.epochs, batch_size=args.batch_size)
            succeeded.append(ticker)
        except (ValueError, OSError) as exc:
            logger.error("Skipping %s: %s", ticker, exc)
            failed.append(ticker)

    logger.info("Done. Succeeded: %s | Failed: %s", succeeded, failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

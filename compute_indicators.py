"""Compute technical indicators and write to CSV."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import pandas_ta as ta

from config import DATA_DIR

logger = logging.getLogger(__name__)


def compute_indicators(ticker: str, poll_interval: int = 60) -> None:
    """Merge historical + realtime data, compute indicators, save CSV.

    Runs in a continuous loop for live operation.
    """
    hist_csv = os.path.join(DATA_DIR, f"{ticker}_historical.csv")
    real_time_csv = os.path.join(DATA_DIR, f"{ticker}_realtime.csv")
    indicators_csv = os.path.join(DATA_DIR, f"{ticker}_indicators.csv")

    while True:
        try:
            if not os.path.exists(hist_csv):
                logger.info("Waiting for %s ...", hist_csv)
                time.sleep(30)
                continue

            hist_data = pd.read_csv(hist_csv, index_col=0, parse_dates=True)

            # Realtime data is optional
            if os.path.exists(real_time_csv):
                real_time_data = pd.read_csv(real_time_csv, index_col=0, parse_dates=True)
                df = pd.concat([hist_data, real_time_data]).drop_duplicates().sort_index()
            else:
                df = hist_data.copy()

            # ── Technical Indicators ──
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

            df.to_csv(indicators_csv)
            logger.info("Indicators updated: %s  (%d rows)", indicators_csv, len(df))

        except Exception as exc:
            logger.error("Error computing indicators for %s: %s", ticker, exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    compute_indicators(ticker)

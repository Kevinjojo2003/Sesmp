"""Fetch historical and real-time stock data via yfinance."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import yfinance as yf

from config import DATA_DIR

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame | None:
    """Fetch historical OHLCV data and save to CSV. Returns the DataFrame."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=interval)

        if hist_data.empty:
            logger.warning("No historical data found for %s.", ticker)
            return None

        hist_csv = os.path.join(DATA_DIR, f"{ticker}_historical.csv")
        hist_data.to_csv(hist_csv)
        logger.info("Historical data saved: %s  (%d rows)", hist_csv, len(hist_data))
        return hist_data

    except Exception as exc:
        logger.error("Error fetching historical data for %s: %s", ticker, exc)
        return None


def fetch_realtime_data(
    ticker: str,
    poll_interval: int = 60,
    max_retries: int = 5,
) -> None:
    """Continuously fetch 1-minute bars and append to CSV.

    Runs in an infinite loop (designed for background thread / subprocess).
    """
    real_time_csv = os.path.join(DATA_DIR, f"{ticker}_realtime.csv")
    retries = 0

    while True:
        try:
            stock = yf.Ticker(ticker)
            real_time_data = stock.history(period="1d", interval="1m")

            if real_time_data.empty:
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached for %s realtime data.", ticker)
                    break
                logger.warning("No realtime data for %s (retry %d/%d).", ticker, retries, max_retries)
                time.sleep(poll_interval)
                continue

            retries = 0  # reset on success

            if os.path.exists(real_time_csv):
                existing = pd.read_csv(real_time_csv, index_col=0, parse_dates=True)
                new_data = real_time_data.loc[~real_time_data.index.isin(existing.index)]
                if not new_data.empty:
                    new_data.to_csv(real_time_csv, mode="a", header=False)
                    logger.info("Appended %d rows to %s", len(new_data), real_time_csv)
            else:
                real_time_data.to_csv(real_time_csv)
                logger.info("Created realtime file: %s", real_time_csv)

        except Exception as exc:
            logger.error("Error fetching realtime data for %s: %s", ticker, exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    fetch_stock_data(ticker)
    fetch_realtime_data(ticker)

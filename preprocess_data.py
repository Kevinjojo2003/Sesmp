"""Merge sentiment + indicator data into a single processed CSV for training."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


def preprocess_data(ticker: str, poll_interval: int = 60) -> None:
    """Merge sentiment and indicator CSVs into ``data/{ticker}_processed.csv``.

    No scaling is applied here; scaling is handled during training.
    Runs in a continuous loop for live operation.
    """
    sentiment_csv = os.path.join(DATA_DIR, f"{ticker}_sentiment.csv")
    indicators_csv = os.path.join(DATA_DIR, f"{ticker}_indicators.csv")
    output_csv = os.path.join(DATA_DIR, f"{ticker}_processed.csv")

    while True:
        try:
            if not os.path.exists(indicators_csv):
                logger.info("Waiting for %s …", indicators_csv)
                time.sleep(30)
                continue

            indicators = pd.read_csv(indicators_csv, index_col=0, parse_dates=True)

            # Sentiment is optional – fill with zeros if unavailable
            if os.path.exists(sentiment_csv):
                sentiment = pd.read_csv(sentiment_csv)
                if "PublishedAt" in sentiment.columns:
                    sentiment["PublishedAt"] = pd.to_datetime(sentiment["PublishedAt"], errors="coerce")
                    sentiment.dropna(subset=["PublishedAt"], inplace=True)
                    sentiment.set_index("PublishedAt", inplace=True)
                    sentiment_counts = sentiment.resample("min").size().rename("Sentiment_Count")
                    indicators = indicators.merge(
                        sentiment_counts, how="left", left_index=True, right_index=True,
                    )

            if "Sentiment_Count" not in indicators.columns:
                indicators["Sentiment_Count"] = 0.0

            indicators["Sentiment_Count"] = indicators["Sentiment_Count"].fillna(0)
            indicators = indicators.ffill().bfill()

            # Drop rows that are still NaN (e.g. early SMA/EMA rows)
            indicators.dropna(inplace=True)

            indicators.to_csv(output_csv)
            logger.info("Processed data saved: %s  (%d rows)", output_csv, len(indicators))

        except (OSError, ValueError, KeyError) as exc:
            logger.error("Preprocessing error for %s: %s", ticker, exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    preprocess_data(ticker)

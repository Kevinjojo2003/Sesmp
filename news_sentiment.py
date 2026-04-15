"""Fetch news headlines for a ticker and score sentiment with FinBERT."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests
from transformers import pipeline

from config import DATA_DIR, NEWS_API_KEY, NEWS_PAGE_SIZE

logger = logging.getLogger(__name__)

# Lazy-load the model so the module can be imported without GPU.
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        logger.info("Loading FinBERT sentiment model …")
        _analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _analyzer


def fetch_news_sentiment(ticker: str, poll_interval: int = 300) -> None:
    """Fetch latest news, run sentiment analysis, append to CSV.

    Runs in a continuous loop (designed for background operation).
    """
    sentiment_csv = os.path.join(DATA_DIR, f"{ticker}_sentiment.csv")

    while True:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)

            if not NEWS_API_KEY:
                logger.warning("NEWS_API_KEY not set. Skipping news fetch.")
                time.sleep(poll_interval)
                continue

            url = (
                f"https://newsapi.org/v2/everything?q={ticker}&language=en"
                f"&sortBy=publishedAt&pageSize={NEWS_PAGE_SIZE}&apiKey={NEWS_API_KEY}"
            )
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            if not articles:
                logger.info("No news found for %s. Retrying in %ds …", ticker, poll_interval)
                time.sleep(poll_interval)
                continue

            analyzer = _get_analyzer()
            rows: list[list] = []

            for article in articles:
                title = article.get("title") or ""
                description = article.get("description") or ""
                article_url = article.get("url") or ""
                published_at = pd.to_datetime(article.get("publishedAt"), errors="coerce")

                if not title:
                    continue

                result = analyzer(title[:512])[0]  # truncate to model max
                rows.append([
                    published_at,
                    title,
                    description,
                    result["label"],
                    result["score"],
                    article_url,
                ])

            df = pd.DataFrame(
                rows,
                columns=["PublishedAt", "Title", "Description", "Sentiment", "Confidence", "URL"],
            )

            if os.path.exists(sentiment_csv):
                existing = pd.read_csv(sentiment_csv, parse_dates=["PublishedAt"])
                df = pd.concat([existing, df]).drop_duplicates(subset=["Title"]).reset_index(drop=True)

            df.to_csv(sentiment_csv, index=False)
            logger.info("Sentiment updated: %s  (%d articles)", sentiment_csv, len(df))

        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.error("Error in sentiment analysis for %s: %s", ticker, exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    fetch_news_sentiment(ticker)

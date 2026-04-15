"""Real-time CNN prediction loop."""

from __future__ import annotations

import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

# Rolling window for error metrics
_actuals: list[float] = []
_preds: list[float] = []


def _compute_errors(actual: float, predicted: float):
    _actuals.append(actual)
    _preds.append(predicted)
    if len(_actuals) > 10:
        _actuals.pop(0)
        _preds.pop(0)

    mae = mean_absolute_error([actual], [predicted])
    rmse = float(np.sqrt(mean_squared_error([actual], [predicted])))
    r2 = r2_score(_actuals, _preds) if len(_actuals) > 1 else float("nan")
    return mae, rmse, r2


def load_data(ticker: str) -> pd.DataFrame | None:
    indicators_path = os.path.join(DATA_DIR, f"{ticker}_indicators.csv")
    sentiment_path = os.path.join(DATA_DIR, f"{ticker}_sentiment.csv")

    if not os.path.exists(indicators_path):
        logger.warning("Indicators file missing: %s", indicators_path)
        return None

    stock = pd.read_csv(indicators_path, index_col=0, parse_dates=True)

    if os.path.exists(sentiment_path):
        sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        sentiment["Sentiment_Count"] = 1
        counts = sentiment.resample("min").sum()["Sentiment_Count"]
        stock = stock.join(counts, how="left").fillna(0)
    else:
        stock["Sentiment_Count"] = 0

    return stock


def real_time_cnn_prediction(ticker: str, poll_interval: int = 60) -> None:
    model_path = os.path.join(MODEL_DIR, f"{ticker}_cnn.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_cnn_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"CNN model/scaler not found for {ticker}")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    logger.info("Starting real-time CNN prediction for %s …", ticker)

    while True:
        stock_data = load_data(ticker)
        if stock_data is None or stock_data.empty:
            time.sleep(poll_interval)
            continue

        try:
            features = stock_data.drop(columns=["Close"], errors="ignore")
            scaled = scaler.transform(features)
            latest = scaled[-1].reshape(1, scaled.shape[1], 1)

            pred_scaled = model.predict(latest, verbose=0)[0][0]

            # Inverse-transform using a dummy row
            dummy = np.zeros((1, scaled.shape[1]))
            dummy[0, -1] = pred_scaled
            predicted_price = scaler.inverse_transform(dummy)[0, -1]

            actual_price = float(stock_data["Close"].iloc[-1])
            mae, rmse, r2 = _compute_errors(actual_price, predicted_price)

            logger.info(
                "%s  Predicted=%.2f  Actual=%.2f  MAE=%.4f  RMSE=%.4f  R²=%.4f",
                ticker, predicted_price, actual_price, mae, rmse, r2,
            )
        except Exception as exc:
            logger.error("Prediction error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ticker = input("Enter stock ticker: ").upper()
    real_time_cnn_prediction(ticker)

"""Real-time LSTM prediction loop with online retraining."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from config import DATA_DIR, LSTM_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

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
        logger.warning("Missing: %s", indicators_path)
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


def _create_sequences(data: np.ndarray, close_index: int, time_steps: int = LSTM_SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, close_index])
    return np.array(X), np.array(y)


def _build_lstm(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _inverse_close(scaler: MinMaxScaler, scaled_close: float, n_features: int, close_index: int) -> float:
    dummy = np.zeros((1, n_features))
    dummy[0, close_index] = scaled_close
    return float(scaler.inverse_transform(dummy)[0, close_index])


def real_time_prediction(ticker: str, poll_interval: int = 60) -> None:
    logger.info("Starting real-time LSTM prediction for %s …", ticker)

    stock_data = load_data(ticker)
    if stock_data is None:
        return

    close_index = stock_data.columns.get_loc("Close")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(stock_data)
    X, y = _create_sequences(scaled, close_index=close_index)

    model = _build_lstm((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    while True:
        stock_data = load_data(ticker)
        if stock_data is None or stock_data.empty:
            time.sleep(poll_interval)
            continue

        try:
            scaled, _ = scaler.fit_transform(stock_data), scaler
            latest = scaled[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, scaled.shape[1])

            pred_scaled = model.predict(latest, verbose=0)[0][0]
            predicted = _inverse_close(scaler, pred_scaled, scaled.shape[1], close_index)
            actual = float(stock_data["Close"].iloc[-1])
            mae, rmse, r2 = _compute_errors(actual, predicted)

            logger.info(
                "%s  Predicted=%.2f  Actual=%.2f  MAE=%.4f  RMSE=%.4f  R²=%.4f",
                ticker, predicted, actual, mae, rmse, r2,
            )
        except Exception as exc:
            logger.error("Prediction error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ticker = input("Enter stock ticker: ").upper()
    real_time_prediction(ticker)

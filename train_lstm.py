"""Train an LSTM model for a single ticker with walk-forward validation."""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

from config import DATA_DIR, LSTM_SEQUENCE_LENGTH, MODEL_DIR

logger = logging.getLogger(__name__)


def load_data(
    ticker: str,
    time_steps: int = LSTM_SEQUENCE_LENGTH,
    forecast_horizon: int = 1,
):
    """Load processed CSV, scale, and build sequences with an 80/20 time split."""
    data_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(data_path):
        logger.error("File not found: %s", data_path)
        return None, None, None, None, None, None, None

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)

    if len(df) < time_steps + forecast_horizon + 10:
        logger.error("Not enough data for %s (%d rows).", ticker, len(df))
        return None, None, None, None, None, None, None

    features = df.columns.tolist()
    target_col = "Close"

    if target_col not in features:
        logger.error("'Close' column missing from %s", data_path)
        return None, None, None, None, None, None, None

    close_index = features.index(target_col)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled) - time_steps - forecast_horizon):
        X.append(scaled[i : i + time_steps])
        y.append(scaled[i + time_steps + forecast_horizon - 1, close_index])

    X, y = np.array(X), np.array(y)

    # ── Walk-forward split (time-ordered, no shuffle) ──
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info("Data: %d train / %d test sequences, %d features", len(X_train), len(X_test), len(features))
    return X_train, X_test, y_train, y_test, scaler, features, close_index


def build_lstm_model(input_shape: tuple[int, int]) -> Sequential:
    """Build a 2-layer LSTM with batch normalization."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def train_lstm(
    ticker: str,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    """Full training pipeline with early stopping."""
    result = load_data(ticker)
    X_train, X_test, y_train, y_test, scaler, features, close_index = result

    if X_train is None:
        logger.error("No data to train for %s.", ticker)
        return

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save artefacts ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"{ticker}_lstm.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"))
    np.save(
        os.path.join(MODEL_DIR, f"{ticker}_feature_columns.npy"),
        np.array(features, dtype=object),
        allow_pickle=True,
    )
    np.save(os.path.join(MODEL_DIR, f"{ticker}_close_index.npy"), np.array([close_index]))

    logger.info("Model saved: %s/%s_lstm.h5", MODEL_DIR, ticker)
    logger.info("Scaler saved: %s/%s_scaler.pkl", MODEL_DIR, ticker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    train_lstm(ticker)

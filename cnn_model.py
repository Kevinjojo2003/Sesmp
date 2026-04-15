"""Train a 1-D CNN regression model for stock price prediction."""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from config import DATA_DIR, MODEL_DIR

logger = logging.getLogger(__name__)


def load_data(ticker: str):
    """Load processed CSV, scale features, return arrays + scaler."""
    data_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing in {data_path}")

    features = df.drop(columns=["Close"])
    target = df["Close"]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    y = target.values

    return X, y, scaler


def build_cnn(input_shape: tuple[int, int]) -> Sequential:
    """Build a 2-layer 1-D CNN for regression."""
    model = Sequential([
        Conv1D(128, kernel_size=3, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_cnn(ticker: str, epochs: int = 100, batch_size: int = 32) -> None:
    """End-to-end CNN training pipeline."""
    X, y, scaler = load_data(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_cnn(input_shape=(X.shape[1], 1))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"{ticker}_cnn.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_cnn_scaler.pkl"))
    logger.info("CNN model + scaler saved for %s.", ticker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter stock ticker: ").upper()
    train_cnn(ticker)

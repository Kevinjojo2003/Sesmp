"""Canny edge detection on stock chart images for pattern analysis."""

from __future__ import annotations

import io
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)


def process_and_save_canny(
    data,
    column: str,
    title: str,
    filename: str,
    save_dir: str = "cannyedge",
    display: bool = False,
) -> str:
    """Render a chart, apply Canny edge detection, save the result."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data[column], label=column, color="blue")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)

    img = Image.open(img_buf).convert("RGB")
    img_cv = np.array(img)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, edges)

    if display:
        st.subheader(f"{title} — Canny Edge Detection")
        st.image(edges, caption=filename, use_container_width=True)

    return path


def apply_and_save_canny_on_all(stock_data, ticker: str, display: bool = False) -> list[str]:
    """Apply Canny edge detection on close price + each available indicator."""
    if stock_data is None or stock_data.empty:
        st.warning("No stock data available for Canny edge processing.")
        return []

    paths: list[str] = []

    paths.append(
        process_and_save_canny(stock_data, "Close", f"{ticker} Price Chart", f"{ticker}_price_edges.png", display=display)
    )

    for col, label in [("SMA", "SMA"), ("EMA", "EMA"), ("RSI", "RSI")]:
        if col in stock_data.columns:
            paths.append(
                process_and_save_canny(stock_data, col, f"{ticker} {label}", f"{ticker}_{col.lower()}_edges.png", display=display)
            )

    return paths

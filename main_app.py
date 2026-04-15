"""Streamlit main application — AI Stock Market Prediction & Analysis."""

from __future__ import annotations

import io
import os

import cv2
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai
import joblib

from config import (
    DEFAULT_TICKER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LSTM_SEQUENCE_LENGTH,
    MODEL_DIR,
    NEWS_API_KEY,
)

# ── Gemini setup ──
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ────────────────────────── helpers ──────────────────────────


def load_model_and_scaler(ticker: str):
    """Load a trained LSTM model + scaler + close_index for *ticker*."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    close_index_path = os.path.join(MODEL_DIR, f"{ticker}_close_index.npy")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        close_index = int(np.load(close_index_path)[0]) if os.path.exists(close_index_path) else 0
        return model, scaler, close_index
    except (OSError, ValueError, IndexError) as exc:
        st.warning(f"Model/scaler load failed for {ticker}: {exc}")
        return None, None, None


@st.cache_data(ttl=60)
def get_stock_data(symbol: str) -> pd.DataFrame:
    stock = yf.Ticker(symbol)
    return stock.history(period="1mo", interval="1h")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA"] = ta.sma(out["Close"], length=10)
    out["EMA"] = ta.ema(out["Close"], length=10)
    out["RSI"] = ta.rsi(out["Close"], length=14)
    return out


# ────────────────────────── prediction ──────────────────────────


def predict_next_price(model, scaler, close_index: int, data: pd.DataFrame) -> float | None:
    if data.empty or len(data) < LSTM_SEQUENCE_LENGTH:
        return None

    window = data.tail(LSTM_SEQUENCE_LENGTH)
    scaled = scaler.transform(window.values)
    seq = scaled.reshape(1, LSTM_SEQUENCE_LENGTH, scaled.shape[1])

    pred_scaled = model.predict(seq, verbose=0)[0][0]
    dummy = np.zeros((1, scaled.shape[1]))
    dummy[0, close_index] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, close_index])


def predict_next_7_days(model, scaler, close_index: int, data: pd.DataFrame, n_days: int = 7):
    if data.empty or len(data) < LSTM_SEQUENCE_LENGTH:
        return None, None

    inputs = scaler.transform(data.tail(LSTM_SEQUENCE_LENGTH).values)
    predicted_scaled: list[float] = []

    for _ in range(n_days):
        seq = inputs[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, inputs.shape[1])
        next_val = model.predict(seq, verbose=0)[0][0]
        predicted_scaled.append(next_val)
        new_row = np.zeros(inputs.shape[1])
        new_row[close_index] = next_val
        inputs = np.vstack([inputs, new_row])

    pred = []
    for val in predicted_scaled:
        dummy = np.zeros((1, inputs.shape[1]))
        dummy[0, close_index] = val
        pred.append(scaler.inverse_transform(dummy)[0, close_index])

    pred = np.array(pred)
    band = np.std(pred) * 0.75
    return pred, (pred - band, pred + band)


# ────────────────────────── canny edge ──────────────────────────


def apply_canny_on_chart(data: pd.DataFrame, title: str = "Chart"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], color="cyan")
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)
    st.image(edges, caption=f"Canny Edge — {title}", use_container_width=True)


# ────────────────────────── news ──────────────────────────


@st.cache_data(ttl=300)
def fetch_news() -> list[dict]:
    if not NEWS_API_KEY:
        return []
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except (requests.RequestException, ValueError):
        return []


# ────────────────────────── ticker tape ──────────────────────────


@st.cache_data(ttl=60)
def get_ticker_tape_data() -> list[dict]:
    symbols = {
        "BANK": "^NSEBANK",
        "BAJFINANCE": "BAJFINANCE.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "HINDUNILVR": "HINDUNILVR.NS",
        "INDIGO": "INDIGO.NS",
    }
    tape = []
    for label, symbol in symbols.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d").dropna()
            if len(hist) < 2:
                continue
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            pct = ((last - prev) / prev) * 100 if prev else 0.0
            tape.append({"label": label, "price": last, "pct": pct})
        except (ValueError, KeyError, IndexError):
            continue
    return tape


def render_ticker_tape():
    tape = get_ticker_tape_data()
    if not tape:
        return

    items = []
    for row in tape:
        arrow = "▲" if row["pct"] >= 0 else "▼"
        cls = "up" if row["pct"] >= 0 else "down"
        items.append(
            f'<span class="item"><span class="sym">{row["label"]}</span> '
            f'{row["price"]:,.2f} <span class="{cls}">{arrow} {abs(row["pct"]):.2f}%</span></span>'
        )

    css = """
    <style>
        .ticker-wrap{width:100%;background:#0c1420;border:1px solid #1f2a38;
            padding:8px 12px;overflow:hidden;white-space:nowrap;margin-bottom:12px}
        .ticker-wrap .item{color:#e5e7eb;display:inline-block;margin-right:20px;font-size:1rem}
        .ticker-wrap .sym{font-weight:700;margin-right:6px}
        .ticker-wrap .up{color:#10b981;font-weight:700}
        .ticker-wrap .down{color:#ef4444;font-weight:700}
    </style>
    """
    st.markdown(css + f'<div class="ticker-wrap">{" ".join(items)}</div>', unsafe_allow_html=True)


# ────────────────────────── Gemini chatbot ──────────────────────────


def get_chatbot_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini key not configured. Add GEMINI_API_KEY to .env."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except (RuntimeError, ValueError) as exc:
        return f"Gemini API error: {exc}"


# ════════════════════════ STREAMLIT APP ════════════════════════

st.set_page_config(page_title="AI Stock Market App", layout="wide")
render_ticker_tape()
st.title("📈 AI Stock Market App")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock ticker", DEFAULT_TICKER).upper().strip()
show_edges = st.sidebar.checkbox("Enable Canny edge panel", value=False)

if not ticker:
    st.stop()

try:
    stock_data = get_stock_data(ticker)
except (ValueError, RuntimeError, requests.RequestException) as exc:
    st.error(f"Unable to load stock data for {ticker}: {exc}")
    st.stop()

if stock_data.empty:
    st.warning("No market data returned for selected ticker.")
    st.stop()

stock_data = add_indicators(stock_data)

live_tab, forecast_tab, sentiment_tab, metrics_tab = st.tabs(
    ["Live Dashboard", "Forecast View", "Sentiment Feed", "Model Metrics"]
)

# ── Live Dashboard ──
with live_tab:
    st.subheader(f"Current price: ${stock_data['Close'].iloc[-1]:.2f}")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Bar(x=stock_data.index, y=stock_data["Volume"], name="Volume", yaxis="y2", opacity=0.35)
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("SMA(10)", f"{stock_data['SMA'].iloc[-1]:.2f}")
    col2.metric("EMA(10)", f"{stock_data['EMA'].iloc[-1]:.2f}")
    col3.metric("RSI(14)", f"{stock_data['RSI'].iloc[-1]:.2f}")

    if show_edges:
        apply_canny_on_chart(stock_data, f"{ticker} Close")

# ── Forecast View ──
with forecast_tab:
    model, scaler, close_index = load_model_and_scaler(ticker)
    if model is None or scaler is None:
        st.info(
            f"No trained model found for **{ticker}**. "
            "Run `python train_lstm.py` or `python train_multi_ticker.py` first."
        )
    else:
        tomorrow = predict_next_price(model, scaler, close_index, stock_data)
        forecast, bands = predict_next_7_days(model, scaler, close_index, stock_data)

        if tomorrow is not None:
            st.success(f"Predicted next close: **${tomorrow:.2f}**")

        if forecast is not None:
            lower, upper = bands
            dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=7)
            table = pd.DataFrame({
                "Date": dates.date,
                "Predicted Close": np.round(forecast, 2),
                "Lower CI": np.round(lower, 2),
                "Upper CI": np.round(upper, 2),
            })
            st.dataframe(table, use_container_width=True)

            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=dates, y=forecast, mode="lines+markers", name="Forecast"))
            pred_fig.add_trace(go.Scatter(x=dates, y=lower, mode="lines", name="Lower CI", line=dict(dash="dash")))
            pred_fig.add_trace(go.Scatter(x=dates, y=upper, mode="lines", name="Upper CI", line=dict(dash="dash")))
            pred_fig.update_layout(template="plotly_dark", title="7-Day Forecast + Confidence Band")
            st.plotly_chart(pred_fig, use_container_width=True)

# ── Sentiment Feed ──
with sentiment_tab:
    st.subheader("Latest business headlines")
    articles = fetch_news()
    if not articles:
        st.info("No news available. Check your NEWS_API_KEY in .env.")
    for article in articles[:10]:
        st.markdown(f"**{article.get('title', 'Untitled')}**")
        st.caption(article.get("publishedAt", ""))
        st.write(article.get("description", ""))
        if article.get("url"):
            st.markdown(f"[Read more]({article['url']})")
        st.markdown("---")

# ── Model Metrics ──
with metrics_tab:
    returns = stock_data["Close"].pct_change().dropna()
    mae_proxy = float(np.mean(np.abs(returns)))
    rmse_proxy = float(np.sqrt(np.mean(np.square(returns))))
    mape_proxy = float(np.mean(np.abs(returns / stock_data["Close"].shift(1).dropna()))) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (proxy)", f"{mae_proxy:.5f}")
    col2.metric("RMSE (proxy)", f"{rmse_proxy:.5f}")
    col3.metric("MAPE (proxy)", f"{mape_proxy:.2f}%")

    rolling_vol = returns.rolling(24).std().dropna()
    vol_fig = go.Figure(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", name="Rolling Volatility")
    )
    vol_fig.update_layout(template="plotly_dark", title="Rolling 24-step Volatility")
    st.plotly_chart(vol_fig, use_container_width=True)

# ── Gemini Chat (sidebar) ──
st.sidebar.header("Gemini Chat")
query = st.sidebar.text_area("Ask about this stock/company")
if st.sidebar.button("Ask Gemini") and query:
    with st.spinner("Thinking …"):
        answer = get_chatbot_response(query)
    st.markdown(f"**Gemini:** {answer}")

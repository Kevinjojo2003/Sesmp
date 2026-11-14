# This is only for Education Purposes

# 📈 AI Stock Market Prediction & Analysis App

Streamlit-based AI application for stock price prediction, technical analysis, chart pattern detection, and real-time financial news — powered by LSTM deep learning, Gemini AI chatbot, and Canny edge detection.

## 🚀 Features

- **Live Dashboard** — Candlestick charts, volume, ticker tape (NSE indices)
- **Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger Bands
- **LSTM Predictions** — Tomorrow + 7-day forecast with confidence bands
- **Gemini AI Chat** — Ask questions about any stock/company
- **News Feed** — Live business headlines via NewsAPI
- **Canny Edge Detection** — Chart pattern visualization

## ⚙️ Quick Start

```bash
pip install -r requirements.txt
# Copy .env.example to .env and add your API keys
streamlit run main_app.py
```

## 🧠 Training

```bash
# Single ticker
python train_lstm.py

# Batch (downloads 5yr data + trains)
python train_multi_ticker.py --years 5 --epochs 150
```

## 🛡️ Disclaimer

Educational purposes only. Not financial advice.

## Deploy on Streamlit Community Cloud
1. Create an app at https://share.streamlit.io pointing at `main_app.py` on `main`.
2. Under **Settings > Secrets**, add `GEMINI_API_KEY` and `NEWS_API_KEY` (and the optional Reddit values).
3. Trained models are not stored in this repo. Run `python train_multi_ticker.py` locally and commit or host the files in `models/`, or the prediction tabs will have nothing to load.
4. The pinned TensorFlow and PyTorch packages are large and may exceed the free tier's memory. If the build fails, trim `requirements.txt` to what you use.

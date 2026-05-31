# 🔮 CryptoSight — ML-Powered Crypto Forecasting Dashboard

> **End-to-end ML project** combining LSTM time-series forecasting, Facebook Prophet, and FinBERT sentiment analysis in a production-quality Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

CryptoSight is a full-stack ML application that forecasts cryptocurrency prices using two models (LSTM neural network + Facebook Prophet) and enriches predictions with real-time news sentiment scored by FinBERT — a BERT model fine-tuned on financial text.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## Features

| Feature | Technology |
|---|---|
| Real-time OHLCV data | Yahoo Finance API via `yfinance` |
| Candlestick + technical indicators | Plotly, Pandas |
| Price forecasting (LSTM) | TensorFlow/Keras — stacked LSTM |
| Price forecasting (Prophet) | Facebook Prophet |
| Forecast confidence intervals | Walk-forward uncertainty estimation |
| News sentiment scoring | FinBERT (ProsusAI/finbert) |
| Sentiment dashboard | Gauge + per-headline scoring |
| Model performance metrics | MAE, RMSE, MAPE on validation set |
| Training loss curves | Epoch-level train vs val loss |
| Feature correlation matrix | Seaborn-style Plotly heatmap |
| Dark professional UI | Custom CSS, DM Mono font |

---

## Architecture

```
crypto-forecast/
│
├── app/
│   ├── main.py               # Streamlit entry point
│   ├── style.css             # Custom dark theme CSS
│   ├── components/
│   │   └── sidebar.py        # Config sidebar
│   └── pages/
│       ├── overview.py       # Price chart + metrics
│       ├── forecast.py       # LSTM + Prophet predictions
│       ├── sentiment.py      # FinBERT news sentiment
│       └── model_performance.py  # Metrics + architecture
│
├── models/
│   ├── lstm_model.py         # Stacked LSTM (TF/Keras)
│   ├── prophet_model.py      # Facebook Prophet
│   └── sentiment_model.py   # FinBERT / VADER pipeline
│
├── utils/
│   ├── data_fetcher.py       # Yahoo Finance + news API
│   └── charts.py             # Plotly figure builders
│
├── tests/
│   └── test_models.py        # Unit tests
│
├── requirements.txt
└── README.md
```

---

## ML Models

### LSTM Neural Network
- **Architecture:** 3 stacked LSTM layers (128→64→32) with Dropout + BatchNorm
- **Features:** Close price, Volume, RSI(14), MA7, MA25, Annualised Volatility
- **Lookback:** 60 trading days
- **Loss function:** Huber loss (robust to outliers)
- **Training:** EarlyStopping with patience=8

### Facebook Prophet
- **Seasonality:** Weekly + Yearly + custom Monthly (Fourier order 5)
- **Changepoint prior:** 0.05 (conservative)
- **Interval width:** 80%

### FinBERT Sentiment
- **Model:** `ProsusAI/finbert` — BERT fine-tuned on 10k financial news items
- **Labels:** Positive / Neutral / Negative → mapped to [-1, +1]
- **Fallback:** VADER lexicon-based sentiment if GPU/memory unavailable

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/crypto-forecast.git
cd crypto-forecast
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment (Hugging Face Spaces)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Streamlit** SDK
2. Push this repo to the Space's git remote
3. Add a `README.md` with YAML front matter specifying `sdk: streamlit` and `app_file: app/main.py`
4. The Space builds automatically — free hosting with GPU support

---

## Results

| Model | MAE ($) | RMSE ($) | MAPE (%) |
|-------|---------|----------|----------|
| LSTM  | ~1,200  | ~1,900   | ~4.2%    |
| Prophet | ~1,800 | ~2,500  | ~6.1%    |

*Results vary by coin and date range.*

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow 2.16** — LSTM model
- **Prophet 1.1** — Bayesian time-series
- **HuggingFace Transformers** — FinBERT
- **Streamlit** — web dashboard
- **Plotly** — interactive charts
- **yfinance** — market data
- **scikit-learn** — preprocessing & metrics

---

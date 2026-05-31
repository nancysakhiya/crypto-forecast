import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest


# Data fetcher 
def test_rsi_range():
    from utils.data_fetcher import _compute_rsi
    prices = pd.Series(np.random.uniform(100, 200, 100))
    rsi = _compute_rsi(prices, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"


def test_fetch_news_returns_list():
    from utils.data_fetcher import fetch_news
    news = fetch_news("BTC-USD")
    assert isinstance(news, list)
    assert len(news) > 0
    assert "title" in news[0]


# Sentiment 
def test_score_headline_range():
    from models.sentiment_model import load_sentiment_pipeline, score_headline
    pipe = load_sentiment_pipeline()
    score = score_headline("Bitcoin surges to new all-time high", pipe)
    assert -1.0 <= score <= 1.0, "Score must be in [-1, +1]"


def test_aggregate_sentiment_labels():
    from models.sentiment_model import aggregate_sentiment
    bullish_items = [{"sentiment_score": 0.8}, {"sentiment_score": 0.6}]
    result = aggregate_sentiment(bullish_items)
    assert result["label"] == "Bullish"

    bearish_items = [{"sentiment_score": -0.7}, {"sentiment_score": -0.5}]
    result = aggregate_sentiment(bearish_items)
    assert result["label"] == "Bearish"


# Charts
def make_dummy_df(n=200):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.Series(np.cumsum(np.random.randn(n)) + 30000, index=idx)
    df = pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.random.randint(1e8, 1e9, n),
        "rsi": np.random.uniform(30, 70, n),
        "ma7": close.rolling(7).mean(),
        "ma25": close.rolling(25).mean(),
        "ma99": close.rolling(99).mean(),
        "bb_mid": close.rolling(20).mean(),
        "bb_up": close.rolling(20).mean() + 500,
        "bb_lo": close.rolling(20).mean() - 500,
        "return": close.pct_change(),
        "volatility": close.pct_change().rolling(7).std() * (365**0.5),
    })
    return df.dropna()


def test_candlestick_chart_creates_figure():
    from utils.charts import candlestick_chart
    import plotly.graph_objects as go
    fig = candlestick_chart(make_dummy_df(), "BTC")
    assert isinstance(fig, go.Figure)


def test_forecast_chart_with_no_models():
    from utils.charts import forecast_chart
    import plotly.graph_objects as go
    fig = forecast_chart(make_dummy_df(), None, None, "BTC")
    assert isinstance(fig, go.Figure)

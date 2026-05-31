
import yfinance as yf
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, date

# Price data 

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download OHLCV + compute technical indicators."""
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the date range.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    # Technical indicators 
    df["ma7"]    = df["close"].rolling(7).mean()
    df["ma25"]   = df["close"].rolling(25).mean()
    df["ma99"]   = df["close"].rolling(99).mean()
    df["rsi"]    = _compute_rsi(df["close"], 14)
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_up"]  = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lo"]  = df["bb_mid"] - 2 * df["bb_std"]

    # Daily return & volatility
    df["return"]     = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(7).std() * (365 ** 0.5)

    return df.dropna()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


# News / sentiment data 

DUMMY_NEWS = {
    "BTC": [
        ("Bitcoin hits new support level as institutional buying increases", "2h ago", 0.62),
        ("BlackRock ETF sees record inflows amid market uncertainty",         "4h ago", 0.71),
        ("Fed rate decision looms: crypto markets brace for volatility",      "6h ago",-0.15),
        ("BTC miners report record profitability this quarter",               "8h ago", 0.55),
        ("Analyst predicts BTC consolidation before next leg up",            "10h ago", 0.30),
    ],
    "ETH": [
        ("Ethereum staking yields remain attractive for long-term holders",   "1h ago", 0.58),
        ("Layer-2 ecosystem reaches new TVL milestone",                       "3h ago", 0.72),
        ("ETH/BTC ratio under pressure as Bitcoin dominance rises",           "5h ago",-0.22),
        ("Vitalik comments on upcoming protocol improvements",                "7h ago", 0.45),
        ("DeFi volumes surge on Ethereum mainnet",                            "9h ago", 0.63),
    ],
    "SOL": [
        ("Solana network uptime hits 99.9% over last 90 days",               "2h ago", 0.68),
        ("Major NFT marketplace migrates to Solana citing low fees",          "4h ago", 0.74),
        ("SOL faces resistance at key technical level",                       "6h ago",-0.10),
        ("Solana Foundation announces new developer grants",                  "8h ago", 0.55),
        ("Institutional interest in SOL grows ahead of token unlock",        "10h ago", 0.20),
    ],
}
DUMMY_NEWS["BNB"] = DUMMY_NEWS["BTC"]
DUMMY_NEWS["XRP"] = DUMMY_NEWS["ETH"]
DUMMY_NEWS["ADA"] = DUMMY_NEWS["SOL"]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(ticker: str) -> list[dict]:
    """
    Try CryptoPanic free API first; fall back to dummy data.
    Returns list of dicts: {title, age, sentiment_score}.
    """
    coin = ticker.replace("-USD", "")

    try:
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token=free&currencies={coin}&public=true&kind=news"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            posts = r.json().get("results", [])[:5]
            if posts:
                return [
                    {
                        "title": p.get("title", ""),
                        "age":   _relative_time(p.get("published_at", "")),
                        "url":   p.get("url", "#"),
                        "sentiment_score": None,   # scored later by FinBERT
                    }
                    for p in posts
                ]
    except Exception:
        pass

    # fallback
    dummy = DUMMY_NEWS.get(coin, DUMMY_NEWS["BTC"])
    return [
        {"title": t, "age": a, "url": "#", "sentiment_score": s}
        for t, a, s in dummy
    ]


def _relative_time(iso: str) -> str:
    try:
        dt   = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        diff = datetime.now(dt.tzinfo) - dt
        h    = int(diff.total_seconds() // 3600)
        return f"{h}h ago" if h < 24 else f"{diff.days}d ago"
    except Exception:
        return "recently"

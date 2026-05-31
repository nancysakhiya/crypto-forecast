"""app/pages/overview.py — Price overview with metrics and candlestick chart."""

import streamlit as st
import pandas as pd
from utils.data_fetcher import fetch_price_data
from utils import charts


def render(config: dict):
    ticker = config["ticker"]
    coin   = config["coin_label"]

    with st.spinner("Fetching price data…"):
        try:
            df = fetch_price_data(ticker, config["start_date"], config["end_date"])
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            return

    # Top metrics 
    latest   = df["close"].iloc[-1]
    prev     = df["close"].iloc[-2]
    chg_pct  = (latest - prev) / prev * 100
    vol_7d   = df["volatility"].iloc[-1] * 100
    rsi_now  = df["rsi"].iloc[-1]
    hi_52    = df["high"].max()
    lo_52    = df["low"].min()

    arrow = "▲" if chg_pct >= 0 else "▼"
    delta_class = "delta-up" if chg_pct >= 0 else "delta-down"
    price_fmt = f"${latest:,.2f}"

    cols = st.columns(5)
    metrics = [
        ("Current Price",   price_fmt,          f'{arrow} {chg_pct:+.2f}% today', delta_class),
        ("24h Change",      f"{chg_pct:+.2f}%", "vs yesterday close",              delta_class),
        ("7d Volatility",   f"{vol_7d:.1f}%",   "annualised",                      "delta-up" if vol_7d < 60 else "delta-down"),
        ("RSI (14)",        f"{rsi_now:.1f}",    "overbought >70  oversold <30",    "delta-up" if 30 < rsi_now < 70 else "delta-down"),
        ("52-week Range",   f"${lo_52:,.0f}",    f"↔ ${hi_52:,.0f}",               "delta-up"),
    ]

    for col, (label, value, sub, cls) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {cls}">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Candlestick chart 
    st.markdown('<div class="section-title">📈 Price Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(charts.candlestick_chart(df, coin), use_container_width=True)

    # Volume + RSI side by side 
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.volume_chart(df),     use_container_width=True)
    with c2:
        st.plotly_chart(charts.rsi_chart(df),        use_container_width=True)

    # Bollinger bands 
    st.markdown('<div class="section-title">📐 Bollinger Bands</div>', unsafe_allow_html=True)
    st.plotly_chart(charts.bollinger_chart(df, coin), use_container_width=True)

    # Raw data expander 
    with st.expander("🗂️ Raw OHLCV Data"):
        st.dataframe(
            df[["open","high","low","close","volume","rsi","ma7","ma25"]].tail(30).style
              .format({"open":"${:.2f}","high":"${:.2f}","low":"${:.2f}",
                       "close":"${:.2f}","volume":"{:,.0f}","rsi":"{:.1f}",
                       "ma7":"${:.2f}","ma25":"${:.2f}"}),
            use_container_width=True,
        )

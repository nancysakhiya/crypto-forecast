"""app/pages/model_performance.py — Training metrics + feature correlation."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import fetch_price_data
from utils import charts

THEME = {
    "bg": "#080c14", "surface": "#0e1420", "border": "#1c2438",
    "accent": "#00f5c4", "accent2": "#6c63ff", "warn": "#ff6b6b",
    "text": "#e2e8f0", "muted": "#64748b",
}


def _loss_chart(train_loss, val_loss) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=train_loss, name="Train loss",
        line=dict(color=THEME["accent"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        y=val_loss, name="Val loss",
        line=dict(color=THEME["accent2"], width=1.5, dash="dash"),
    ))
    fig.update_layout(
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(family="DM Mono", color=THEME["text"], size=11),
        xaxis=dict(gridcolor=THEME["border"], title="Epoch"),
        yaxis=dict(gridcolor=THEME["border"], title="Huber Loss"),
        title="LSTM Training Curve",
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def render(config: dict):
    ticker = config["ticker"]
    coin   = config["coin_label"]

    with st.spinner("Fetching data…"):
        try:
            df = fetch_price_data(ticker, config["start_date"], config["end_date"])
        except Exception as e:
            st.error(str(e)); return

    if len(df) < 100:
        st.warning("Need at least 6 months of data for model metrics.")
        return

    df_hash = hash(str(df.index[-1]) + ticker + str(len(df)))
    active  = config["models"]

    # LSTM metrics 
    if "LSTM" in active:
        st.markdown('<div class="section-title">🧠 LSTM Performance</div>',
                    unsafe_allow_html=True)
        try:
            from models.lstm_model import train_lstm, forecast_lstm
            _, _, _, _, lstm_metrics = train_lstm(ticker, df_hash, df, config["forecast_days"])

            c1, c2, c3 = st.columns(3)
            for col, key in zip([c1, c2, c3], ["MAE", "RMSE", "MAPE"]):
                suffix = "%" if key == "MAPE" else "$"
                col.markdown(f"""
                <div class="metric-card" style="text-align:center;">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{lstm_metrics[key]}{suffix}</div>
                    <div class="metric-delta delta-up">Validation set</div>
                </div>""", unsafe_allow_html=True)

            if "train_loss" in lstm_metrics:
                st.plotly_chart(
                    _loss_chart(lstm_metrics["train_loss"], lstm_metrics["val_loss"]),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Could not load LSTM metrics: {e}")

    # Prophet metrics 
    if "Prophet" in active:
        st.markdown('<div class="section-title">📈 Prophet Performance</div>',
                    unsafe_allow_html=True)
        try:
            from models.prophet_model import train_prophet
            _, p_metrics = train_prophet(ticker, df_hash, df)

            c1, c2, c3 = st.columns(3)
            for col, key in zip([c1, c2, c3], ["MAE", "RMSE", "MAPE"]):
                suffix = "%" if key == "MAPE" else "$"
                col.markdown(f"""
                <div class="metric-card" style="text-align:center;">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{p_metrics[key]}{suffix}</div>
                    <div class="metric-delta delta-up">Validation set</div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load Prophet metrics: {e}")

    # Feature correlation 
    st.markdown('<div class="section-title">🔗 Feature Correlation Matrix</div>',
                unsafe_allow_html=True)
    st.plotly_chart(charts.correlation_heatmap(df), use_container_width=True)

    # Data stats 
    st.markdown('<div class="section-title">📋 Dataset Summary</div>',
                unsafe_allow_html=True)
    desc = df[["close","volume","rsi","volatility"]].describe().round(4)
    st.dataframe(desc, use_container_width=True)

    # Architecture notes 
    with st.expander("📖 Model Architecture Details"):
        st.markdown("""
**LSTM Architecture**
```
Input  → (batch, 60, 6)          # 60-day lookback, 6 features
LSTM   → 128 units, return_sequences=True
Dropout(0.2) + BatchNorm
LSTM   → 64 units, return_sequences=True
Dropout(0.2) + BatchNorm
LSTM   → 32 units
Dropout(0.2)
Dense  → 16 units (ReLU)
Dense  → 1  (price prediction)
Loss   : Huber  |  Optimizer : Adam(lr=1e-3)  |  EarlyStopping(patience=8)
```

**Prophet Configuration**
```
Weekly seasonality  : True
Yearly seasonality  : True
Monthly seasonality : custom, fourier_order=5
Changepoint prior   : 0.05
Interval width      : 80%
```

**Features used by LSTM**
- `close` (target), `volume`, `RSI(14)`, `MA7`, `MA25`, `annualised volatility`
        """)

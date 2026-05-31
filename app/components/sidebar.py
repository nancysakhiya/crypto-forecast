import streamlit as st
from datetime import date, timedelta

COINS = {
    "Bitcoin (BTC)":  "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)":   "SOL-USD",
    "BNB (BNB)":      "BNB-USD",
    "XRP (XRP)":      "XRP-USD",
    "Cardano (ADA)":  "ADA-USD",
}

def render_sidebar() -> dict:
    st.sidebar.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.7rem;
                color:#64748b;text-transform:uppercase;letter-spacing:.08em;
                margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid #1c2438;'>
        ⚙️ Configuration
    </div>
    """, unsafe_allow_html=True)

    coin_label = st.sidebar.selectbox("Cryptocurrency", list(COINS.keys()))
    ticker     = COINS[coin_label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("<small style='color:#64748b;font-family:DM Mono'>📅 Date Range</small>",
                        unsafe_allow_html=True)

    end_date   = date.today()
    start_date = st.sidebar.date_input("Start date", end_date - timedelta(days=365))
    end_date   = st.sidebar.date_input("End date",   end_date)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<small style='color:#64748b;font-family:DM Mono'>🤖 Model Settings</small>",
                        unsafe_allow_html=True)

    forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
    models        = st.sidebar.multiselect(
        "Active models",
        ["LSTM", "Prophet"],
        default=["LSTM", "Prophet"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.68rem;color:#64748b;line-height:1.7;'>
        <b style='color:#00f5c4'>CryptoSight v1.0</b><br>
        LSTM + Prophet forecasting<br>
        FinBERT sentiment analysis<br>
        Built for ML portfolio
    </div>
    """, unsafe_allow_html=True)

    return {
        "ticker":        ticker,
        "coin_label":    coin_label.split(" ")[0],
        "start_date":    start_date,
        "end_date":      end_date,
        "forecast_days": forecast_days,
        "models":        models,
    }

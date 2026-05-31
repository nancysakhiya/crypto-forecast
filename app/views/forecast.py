"""app/pages/forecast.py — LSTM & Prophet forecast comparison."""

import streamlit as st
from utils.data_fetcher import fetch_price_data
from utils import charts


def render(config: dict):
    ticker        = config["ticker"]
    coin          = config["coin_label"]
    forecast_days = config["forecast_days"]
    active_models = config["models"]

    with st.spinner("Fetching price data…"):
        try:
            df = fetch_price_data(ticker, config["start_date"], config["end_date"])
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            return

    if len(df) < 100:
        st.warning("Not enough data to train models. Try a longer date range (at least 6 months).")
        return

    # Info banner 
    st.markdown(f"""
    <div class="info-box">
        Forecasting <b>{coin}</b> for the next <b>{forecast_days} days</b>
        using {" and ".join(active_models) if active_models else "no models selected"}.<br>
        LSTM: sequence-to-one stacked LSTM (60-day lookback, 6 features) &nbsp;|&nbsp;
        Prophet: additive decomposition with weekly + yearly seasonality.
    </div>
    """, unsafe_allow_html=True)

    lstm_forecast    = None
    prophet_forecast = None

    df_hash = hash(str(df.index[-1]) + ticker + str(len(df)))

    # LSTM 
    if "LSTM" in active_models:
        with st.spinner("🧠 Training LSTM… (cached after first run)"):
            try:
                from models.lstm_model import train_lstm, forecast_lstm
                model, scaler, features, seq_len, _ = train_lstm(
                    ticker, df_hash, df, forecast_days
                )
                lstm_forecast = forecast_lstm(model, scaler, features, seq_len, df, forecast_days)
                st.success("✅ LSTM trained successfully")
            except Exception as e:
                st.error(f"LSTM failed: {e}")

    # Prophet 
    if "Prophet" in active_models:
        with st.spinner("📈 Fitting Prophet… (cached after first run)"):
            try:
                from models.prophet_model import train_prophet, forecast_prophet
                p_model, _ = train_prophet(ticker, df_hash, df)
                prophet_forecast = forecast_prophet(p_model, df, forecast_days)
                st.success("✅ Prophet fitted successfully")
            except Exception as e:
                st.error(f"Prophet failed: {e}")

    if lstm_forecast is None and prophet_forecast is None:
        st.info("Select at least one model in the sidebar.")
        return

    # Forecast chart 
    st.markdown('<div class="section-title">🔮 Forecast Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(
        charts.forecast_chart(df, lstm_forecast, prophet_forecast, coin),
        use_container_width=True,
    )

    # Legend 
    st.markdown("""
    <div style="display:flex;gap:24px;margin-top:-8px;margin-bottom:16px;">
        <div class="legend-row"><div class="legend-dot" style="background:#e2e8f0"></div> Historical close</div>
        <div class="legend-row"><div class="legend-dot" style="background:#00f5c4"></div> LSTM forecast</div>
        <div class="legend-row"><div class="legend-dot" style="background:#6c63ff"></div> Prophet forecast</div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast tables 
    c1, c2 = st.columns(2)

    if lstm_forecast is not None:
        with c1:
            st.markdown('<div class="section-title">LSTM Forecast</div>', unsafe_allow_html=True)
            st.dataframe(
                lstm_forecast.set_index("ds")[["yhat","yhat_lower","yhat_upper"]]
                  .rename(columns={"yhat":"Price","yhat_lower":"Low","yhat_upper":"High"})
                  .style.format("${:.2f}"),
                use_container_width=True,
            )

    if prophet_forecast is not None:
        with c2:
            st.markdown('<div class="section-title">Prophet Forecast</div>', unsafe_allow_html=True)
            st.dataframe(
                prophet_forecast.set_index("ds")[["yhat","yhat_lower","yhat_upper"]]
                  .rename(columns={"yhat":"Price","yhat_lower":"Low","yhat_upper":"High"})
                  .style.format("${:.2f}"),
                use_container_width=True,
            )

    # Model comparison numbers 
    if lstm_forecast is not None and prophet_forecast is not None:
        st.markdown('<div class="section-title">📊 7-day Forecast Comparison</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        comp = pd.DataFrame({
            "Date":            lstm_forecast["ds"].dt.strftime("%b %d").head(7),
            "LSTM ($)":        lstm_forecast["yhat"].round(2).head(7).values,
            "Prophet ($)":     prophet_forecast["yhat"].round(2).head(7).values,
        })
        comp["Difference"] = (comp["LSTM ($)"] - comp["Prophet ($)"]).round(2)
        st.dataframe(comp.set_index("Date"), use_container_width=True)

"""
models/prophet_model.py

Facebook Prophet for time-series forecasting.
Prophet handles trend, seasonality, and holiday effects automatically.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st


@st.cache_resource(show_spinner=False)
def train_prophet(ticker: str, df_hash: int, df: pd.DataFrame):
    """Fit Prophet and return (model, metrics)."""
    from prophet import Prophet

    prophet_df = df[["close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    # Train / val split
    split     = int(len(prophet_df) * 0.8)
    train_df  = prophet_df.iloc[:split]
    val_df    = prophet_df.iloc[split:]

    model = Prophet(
        daily_seasonality  = False,
        weekly_seasonality = True,
        yearly_seasonality = True,
        changepoint_prior_scale = 0.05,
        interval_width     = 0.80,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(train_df)

    # Validate
    val_forecast = model.predict(val_df[["ds"]])
    mae   = mean_absolute_error(val_df["y"].values, val_forecast["yhat"].values)
    rmse  = mean_squared_error(val_df["y"].values, val_forecast["yhat"].values) ** 0.5
    mape  = np.mean(
        np.abs((val_df["y"].values - val_forecast["yhat"].values)
               / (val_df["y"].values + 1e-9))
    ) * 100

    metrics = {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }

    return model, metrics


def forecast_prophet(model, df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """Generate future forecast from fitted Prophet model."""
    future   = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)
    result   = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
    return result.reset_index(drop=True)

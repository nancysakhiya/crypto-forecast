"""
models/lstm_model.py
────────────────────
Builds, trains, and forecasts with a stacked LSTM.
Uses MinMaxScaler; returns forecast as a tidy DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st


# ── Build model lazily to avoid TF import at module level ─────────────────────
def _build_model(seq_len: int, n_features: int):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber")
    return model


# Public API 

@st.cache_resource(show_spinner=False)
def train_lstm(ticker: str, df_hash: int, df: pd.DataFrame, forecast_days: int):
    """
    Train LSTM and return (model, scaler, metrics_dict).
    Cached by ticker + data hash to avoid re-training on re-runs.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    SEQ_LEN    = 60
    FEATURES   = ["close", "volume", "rsi", "ma7", "ma25", "volatility"]
    FEATURES   = [f for f in FEATURES if f in df.columns]

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    # Build sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN: i])
        y.append(scaled[i, 0])        # target = close (index 0)
    X, y = np.array(X), np.array(y)

    # Train / val split (80/20)
    split     = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = _build_model(SEQ_LEN, len(FEATURES))
    cb    = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=80, batch_size=32,
        callbacks=[cb], verbose=0,
    )

    # Metrics on validation
    val_pred = model.predict(X_val, verbose=0).flatten()
    # Inverse-transform: rebuild full feature matrix
    dummy         = np.zeros((len(val_pred), len(FEATURES)))
    dummy[:, 0]   = val_pred
    inv_pred      = scaler.inverse_transform(dummy)[:, 0]

    dummy2        = np.zeros((len(y_val), len(FEATURES)))
    dummy2[:, 0]  = y_val
    inv_true      = scaler.inverse_transform(dummy2)[:, 0]

    mae   = mean_absolute_error(inv_true, inv_pred)
    rmse  = mean_squared_error(inv_true, inv_pred) ** 0.5
    mape  = np.mean(np.abs((inv_true - inv_pred) / (inv_true + 1e-9))) * 100

    metrics = {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "val_loss":   history.history["val_loss"],
        "train_loss": history.history["loss"],
    }

    return model, scaler, FEATURES, SEQ_LEN, metrics


def forecast_lstm(
    model, scaler, features: list, seq_len: int,
    df: pd.DataFrame, forecast_days: int,
) -> pd.DataFrame:
    """Walk-forward forecast for `forecast_days` days."""
    scaled = scaler.transform(df[features])
    window = list(scaled[-seq_len:])

    preds = []
    for _ in range(forecast_days):
        x       = np.array(window[-seq_len:]).reshape(1, seq_len, len(features))
        p       = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        new_row  = window[-1].copy()
        new_row[0] = p
        window.append(new_row)

    # Inverse-transform
    dummy       = np.zeros((len(preds), len(features)))
    dummy[:, 0] = preds
    prices      = scaler.inverse_transform(dummy)[:, 0]

    # Build date index
    last_date = df.index[-1]
    dates     = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

    # Simple uncertainty: ±2% * sqrt(t)
    std = prices[0] * 0.02
    return pd.DataFrame({
        "ds":         dates,
        "yhat":       prices,
        "yhat_upper": prices + std * np.sqrt(np.arange(1, forecast_days + 1)),
        "yhat_lower": prices - std * np.sqrt(np.arange(1, forecast_days + 1)),
    })

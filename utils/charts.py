
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Shared theme 
THEME = dict(
    bg       = "#080c14",
    surface  = "#0e1420",
    border   = "#1c2438",
    accent   = "#00f5c4",
    accent2  = "#6c63ff",
    warn     = "#ff6b6b",
    text     = "#e2e8f0",
    muted    = "#64748b",
    grid     = "rgba(28,36,56,0.8)",
    font     = "DM Mono",
)

def _base_layout(**kwargs):
    return dict(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["bg"],
        font          = dict(family=THEME["font"], color=THEME["text"], size=11),
        xaxis         = dict(gridcolor=THEME["grid"], linecolor=THEME["border"],
                             showgrid=True, zeroline=False),
        yaxis         = dict(gridcolor=THEME["grid"], linecolor=THEME["border"],
                             showgrid=True, zeroline=False),
        margin        = dict(l=0, r=0, t=40, b=0),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel    = dict(bgcolor=THEME["surface"], font_family=THEME["font"],
                             font_size=11),
        **kwargs,
    )


# Candlestick + moving averages 
def candlestick_chart(df: pd.DataFrame, coin: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=THEME["accent"],
        decreasing_line_color=THEME["warn"],
        name="OHLC",
        increasing_fillcolor="rgba(0,245,196,0.25)",
        decreasing_fillcolor="rgba(255,107,107,0.25)",
    ))

    for col, color, label in [
        ("ma7",  "#facc15", "MA 7"),
        ("ma25", "#a78bfa", "MA 25"),
        ("ma99", "#f97316", "MA 99"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=1.2),
                name=label, opacity=0.85,
            ))

    fig.update_layout(
        **_base_layout(title=f"{coin} — Candlestick + Moving Averages"),
        xaxis_rangeslider_visible=False,
        height=420,
    )
    return fig


#  Volume bar chart 
def volume_chart(df: pd.DataFrame) -> go.Figure:
    colors = [THEME["accent"] if r >= 0 else THEME["warn"] for r in df["return"]]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["volume"],
        marker_color=colors, opacity=0.7, name="Volume",
    ))
    fig.update_layout(**_base_layout(title="Trading Volume", height=160))
    return fig


# RSI chart 
def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=70, line_dash="dash", line_color=THEME["warn"],  opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=THEME["accent"], opacity=0.5)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        line=dict(color=THEME["accent2"], width=1.5),
        name="RSI (14)",
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.07)",
    ))
    layout = _base_layout(title="RSI (14)", height=160)
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig


# Bollinger bands 
def bollinger_chart(df: pd.DataFrame, coin: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["bb_up"], df["bb_lo"][::-1]]),
        fill="toself",
        fillcolor="rgba(108,99,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="BB Band",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        line=dict(color=THEME["accent"], width=1.5), name="Close",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_mid"],
        line=dict(color=THEME["muted"], width=1, dash="dot"), name="BB Mid",
    ))

    fig.update_layout(**_base_layout(title=f"{coin} — Bollinger Bands (20,2)"), height=320)
    return fig


# Forecast chart 
def forecast_chart(
    historical: pd.DataFrame,
    lstm_forecast,
    prophet_forecast,
    coin: str,
) -> go.Figure:
    fig = go.Figure()

    # historical price (last 60 days for clarity)
    hist = historical.tail(60)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["close"],
        line=dict(color=THEME["text"], width=1.5),
        name="Historical", opacity=0.9,
    ))

    if lstm_forecast is not None:
        fig.add_trace(go.Scatter(
            x=lstm_forecast["ds"], y=lstm_forecast["yhat"],
            line=dict(color=THEME["accent"], width=2.5, dash="dash"),
            name="LSTM Forecast",
        ))
        if "yhat_upper" in lstm_forecast.columns:
            fig.add_traces([
                go.Scatter(
                    x=lstm_forecast["ds"], y=lstm_forecast["yhat_upper"],
                    line=dict(width=0), showlegend=False,
                ),
                go.Scatter(
                    x=lstm_forecast["ds"], y=lstm_forecast["yhat_lower"],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,245,196,0.10)",
                    name="LSTM Confidence",
                ),
            ])

    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(
            x=prophet_forecast["ds"], y=prophet_forecast["yhat"],
            line=dict(color=THEME["accent2"], width=2.5, dash="dot"),
            name="Prophet Forecast",
        ))
        if "yhat_upper" in prophet_forecast.columns:
            fig.add_traces([
                go.Scatter(
                    x=prophet_forecast["ds"], y=prophet_forecast["yhat_upper"],
                    line=dict(width=0), showlegend=False,
                ),
                go.Scatter(
                    x=prophet_forecast["ds"], y=prophet_forecast["yhat_lower"],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(108,99,255,0.08)",
                    name="Prophet Confidence",
                ),
            ])

    # "Today" marker using add_shape + add_annotation (avoids add_vline datetime bug)
    last_date = historical.index[-1]
    fig.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1, yref="paper",
        line=dict(color=THEME["muted"], dash="dash", width=1),
        opacity=0.6,
    )
    fig.add_annotation(
        x=last_date, y=1, yref="paper",
        text="Today", showarrow=False,
        font=dict(color=THEME["muted"], size=10, family=THEME["font"]),
        xanchor="left", yanchor="bottom",
    )

    fig.update_layout(
        **_base_layout(title=f"{coin} — Price Forecast"),
        height=440,
    )
    return fig


# Sentiment gauge 
def sentiment_gauge(score: float) -> go.Figure:
    """Score: -1 (bearish) to +1 (bullish)."""
    color = THEME["accent"] if score > 0.1 else (THEME["warn"] if score < -0.1 else THEME["muted"])

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(score * 100, 1),
        number= dict(suffix=" / 100", font=dict(family=THEME["font"], color=THEME["text"], size=28)),
        gauge = dict(
            axis        = dict(range=[-100, 100], tickfont=dict(size=9)),
            bar         = dict(color=color, thickness=0.25),
            bgcolor     = THEME["surface"],
            borderwidth = 0,
            steps       = [
                dict(range=[-100, -20], color="rgba(255,107,107,0.12)"),
                dict(range=[-20,   20], color="rgba(100,116,139,0.08)"),
                dict(range=[ 20,  100], color="rgba(0,245,196,0.12)"),
            ],
            threshold   = dict(line=dict(color=color, width=3), thickness=0.8, value=score*100),
        ),
        title = dict(text="Overall Sentiment", font=dict(size=13, color=THEME["muted"])),
        domain= dict(x=[0,1], y=[0,1]),
    ))
    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(family=THEME["font"]),
    )
    return fig


# Correlation heatmap
def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols  = ["close", "volume", "rsi", "volatility", "ma7", "ma25"]
    cols  = [c for c in cols if c in df.columns]
    corr  = df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z           = corr.values,
        x           = corr.columns,
        y           = corr.columns,
        colorscale  = [[0,"#ff6b6b"],[0.5,"#1c2438"],[1,"#00f5c4"]],
        zmid        = 0,
        text        = corr.round(2).values,
        texttemplate="%{text}",
        textfont    = dict(size=10),
        showscale   = True,
    ))
    fig.update_layout(**_base_layout(title="Feature Correlation Matrix"), height=350)
    return fig

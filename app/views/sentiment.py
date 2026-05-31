"""app/pages/sentiment.py — News sentiment analysis via FinBERT."""

import streamlit as st
from utils.data_fetcher import fetch_news
from models.sentiment_model import load_sentiment_pipeline, batch_score, aggregate_sentiment
from utils import charts


def render(config: dict):
    ticker = config["ticker"]
    coin   = config["coin_label"]

    st.markdown(f"""
    <div class="info-box">
        Sentiment scored by <b>FinBERT</b> (ProsusAI/finbert) — a BERT model fine-tuned
        on financial text. Falls back to VADER if transformers unavailable.
        News sourced from CryptoPanic API.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading sentiment model…"):
        pipeline_tuple = load_sentiment_pipeline()
        model_name = "FinBERT" if pipeline_tuple[0] == "finbert" else \
                     ("VADER"  if pipeline_tuple[0] == "vader"   else "Keyword fallback")

    st.caption(f"🤖 Active sentiment engine: **{model_name}**")

    with st.spinner(f"Fetching {coin} news…"):
        raw_news = fetch_news(ticker)

    with st.spinner("Scoring headlines…"):
        scored = batch_score(raw_news, pipeline_tuple)
        summary = aggregate_sentiment(scored)

    # Overall gauge + stats 
    c1, c2 = st.columns([1, 1.6])

    with c1:
        st.plotly_chart(charts.sentiment_gauge(summary["score"]), use_container_width=True)

    with c2:
        badge_map = {
            "Bullish": "badge-bullish",
            "Bearish": "badge-bearish",
            "Neutral": "badge-neutral",
        }
        badge_cls = badge_map.get(summary["label"], "badge-neutral")

        st.markdown(f"""
        <div style="margin-top:24px;">
            <div class="section-title">Market Mood</div>
            <span class="badge {badge_cls}">{summary["label"]}</span>
            <div style="margin-top:20px;display:flex;gap:24px;">
                <div class="metric-card" style="flex:1;text-align:center;">
                    <div class="metric-label">Bullish</div>
                    <div class="metric-value" style="color:var(--up)">{summary["bullish"]}</div>
                </div>
                <div class="metric-card" style="flex:1;text-align:center;">
                    <div class="metric-label">Neutral</div>
                    <div class="metric-value">{summary["neutral"]}</div>
                </div>
                <div class="metric-card" style="flex:1;text-align:center;">
                    <div class="metric-label">Bearish</div>
                    <div class="metric-value" style="color:var(--down)">{summary["bearish"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Individual headlines 
    st.markdown('<div class="section-title">📰 Headlines</div>', unsafe_allow_html=True)

    for item in scored:
        sc    = item["sentiment_score"]
        label = "Bullish" if sc > 0.15 else ("Bearish" if sc < -0.15 else "Neutral")
        cls   = "badge-bullish" if label=="Bullish" else \
                ("badge-bearish" if label=="Bearish" else "badge-neutral")
        bar_w  = int(abs(sc) * 100)
        bar_c  = "#00f5c4" if sc >= 0 else "#ff6b6b"
        url    = item.get("url", "#")
        link   = f'<a href="{url}" target="_blank" style="color:inherit;text-decoration:none;">↗</a>' \
                 if url != "#" else ""

        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{item['title']} {link}</div>
            <div style="display:flex;align-items:center;gap:12px;margin-top:8px;">
                <span class="badge {cls}">{label}</span>
                <div style="flex:1;height:4px;background:rgba(255,255,255,0.06);
                            border-radius:2px;overflow:hidden;">
                    <div style="width:{bar_w}%;height:100%;background:{bar_c};
                                border-radius:2px;"></div>
                </div>
                <span class="news-meta">{sc:+.3f}</span>
                <span class="news-meta">{item.get('age','')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

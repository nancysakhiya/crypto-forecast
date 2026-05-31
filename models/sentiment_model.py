"""
models/sentiment_model.py

Sentiment scoring pipeline:
  1. Tries FinBERT (ProsusAI/finbert) — best for financial text
  2. Falls back to VADER if transformers/GPU unavailable
  3. Returns a score in [-1, +1] per headline
"""

import streamlit as st
import numpy as np


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Load FinBERT pipeline once and cache across sessions."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
        )
        return ("finbert", pipe)
    except Exception:
        pass

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return ("vader", SentimentIntensityAnalyzer())
    except Exception:
        return ("none", None)


def score_headline(text: str, pipeline_tuple) -> float:
    """
    Returns a float in [-1, +1]:
      +1 = very bullish, -1 = very bearish, 0 = neutral.
    """
    kind, model = pipeline_tuple

    if kind == "finbert":
        results = model(text[:512])[0]
        label_map = {"positive": 1, "neutral": 0, "negative": -1}
        score = sum(label_map.get(r["label"].lower(), 0) * r["score"] for r in results)
        return round(float(score), 4)

    elif kind == "vader":
        vs = model.polarity_scores(text)
        return round(vs["compound"], 4)

    else:
        # Hard-coded fallback based on simple keyword matching
        text_lower = text.lower()
        positive_words = ["surge", "bull", "gains", "record", "growth", "rally",
                          "inflow", "outperform", "buy", "up", "positive", "strong"]
        negative_words = ["crash", "bear", "loss", "drop", "fear", "down", "sell",
                          "weak", "decline", "pressure", "uncertainty", "volatile"]
        pos = sum(w in text_lower for w in positive_words)
        neg = sum(w in text_lower for w in negative_words)
        if pos + neg == 0:
            return 0.0
        return round((pos - neg) / (pos + neg), 4)


def batch_score(headlines: list[dict], pipeline_tuple) -> list[dict]:
    """Score a list of news dicts, filling in sentiment_score."""
    scored = []
    for item in headlines:
        sc = item.get("sentiment_score")
        if sc is None:
            sc = score_headline(item["title"], pipeline_tuple)
        scored.append({**item, "sentiment_score": sc})
    return scored


def aggregate_sentiment(scored_items: list[dict]) -> dict:
    """Return summary stats for the overall sentiment."""
    scores = [i["sentiment_score"] for i in scored_items]
    avg    = float(np.mean(scores)) if scores else 0.0

    if avg > 0.15:
        label = "Bullish"
    elif avg < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "score":   round(avg, 4),
        "label":   label,
        "bullish": sum(1 for s in scores if s > 0.15),
        "neutral": sum(1 for s in scores if -0.15 <= s <= 0.15),
        "bearish": sum(1 for s in scores if s < -0.15),
    }

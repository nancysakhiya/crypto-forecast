import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.views import overview, forecast, sentiment, model_performance
from app.components.sidebar import render_sidebar

# Page config 
st.set_page_config(
    page_title="CryptoSight | ML Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS 
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header 
st.markdown("""
<div class="main-header">
    <div class="header-logo">🔮</div>
    <div>
        <h1 class="header-title">CryptoSight</h1>
        <p class="header-sub">ML-Powered Price Forecasting & Sentiment Analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar (returns config) 
config = render_sidebar()

# Tab navigation 
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Overview",
    "🤖  Forecast",
    "💬  Sentiment",
    "📊  Model Performance",
])

with tab1:
    overview.render(config)

with tab2:
    forecast.render(config)

with tab3:
    sentiment.render(config)

with tab4:
    model_performance.render(config)

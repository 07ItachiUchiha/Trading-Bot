import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import sentiment utilities
from utils.sentiment_analyzer import SentimentAnalyzer
from config import NEWS_API_KEY, ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY

sentiment_analyzer = None

def get_sentiment_analyzer():
    """Get or create a sentiment analyzer instance."""
    global sentiment_analyzer
    
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentAnalyzer(
            api_keys={
                "newsapi": NEWS_API_KEY,
                "alphavantage": ALPHAVANTAGE_API_KEY,
                "finnhub": FINNHUB_API_KEY,
            }
        )
    
    return sentiment_analyzer

def display_news_analysis(symbol=None):
    """Display news-based sentiment analysis."""
    st.header(" News Sentiment Analysis")
    
    analyzer = get_sentiment_analyzer()
    
    # Symbol selection
    if not symbol:
        symbol = st.text_input("Enter Symbol", value="BTC/USD")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        refresh = st.button("🔄 Refresh")
    
    if not symbol:
        st.warning("Please enter a symbol")
        return
    
    try:
        with st.spinner(f"Analyzing sentiment for {symbol}..."):
            normalized_symbol = symbol.replace("/", "")
            sentiment_result = analyzer.get_sentiment(
                normalized_symbol,
                force_refresh=refresh,
            )
        
        if not sentiment_result:
            st.error("Could not retrieve sentiment data")
            return
        
        sentiment_score = sentiment_result.get('score', 0)
        sentiment_label = str(sentiment_result.get('signal', 'neutral')).upper()
        confidence = sentiment_result.get('confidence', 0)
        
        # Sentiment metrics
        st.subheader("Sentiment Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        with col2:
            st.metric("Label", sentiment_label)
        with col3:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Sentiment Gauge"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, -0.2], 'color': "lightsalmon"},
                    {'range': [-0.2, 0.2], 'color': "lightgray"},
                    {'range': [0.2, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        if sentiment_score > 0.5:
            st.success(" Strong positive sentiment detected")
        elif sentiment_score > 0.2:
            st.success(" Positive sentiment detected")
        elif sentiment_score > -0.2:
            st.info(" Neutral sentiment")
        elif sentiment_score > -0.5:
            st.warning(" Negative sentiment detected")
        else:
            st.error(" Strong negative sentiment detected")
        
        # Raw data
        st.subheader("Raw Results")
        st.json(sentiment_result)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please verify your API keys are configured in the .env file")

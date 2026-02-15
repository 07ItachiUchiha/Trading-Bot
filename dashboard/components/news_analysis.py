import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the enhanced sentiment analyzer
from utils.enhanced_sentiment import EnhancedSentimentAnalyzer
from utils.news_trading_advisor import NewsTradingAdvisor
from config import NEWS_API_KEY, ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY

# Initialize a global news prediction advisor instance
news_advisor = None

def get_news_advisor():
    """Get or create a news prediction advisor instance."""
    global news_advisor
    
    if news_advisor is None:
        api_keys = {
            'newsapi': NEWS_API_KEY,
            'alphavantage': ALPHAVANTAGE_API_KEY,
            'finnhub': FINNHUB_API_KEY
        }
        news_advisor = NewsTradingAdvisor(api_keys)
    
    return news_advisor

def display_news_analysis(symbol=None):
    """Display news-based prediction recommendations."""
    st.header("📰 News Sentiment Analysis")
    
    advisor = get_news_advisor()
    
    # Symbol selection if not provided
    if not symbol:
        symbol = st.text_input("Symbol", value="BTC/USD")
    
    # Add the symbol to tracked symbols
    advisor.track_symbol(symbol)
    
    # Force refresh option
    col1, col2 = st.columns([3, 1])
    with col2:
        refresh = st.button("🔄 Refresh Analysis")
    
    # Get recommendation
    recommendation = advisor.get_recommendation(symbol, force_refresh=refresh)
    sentiment_data = recommendation['sentiment_data']
    
    # Display recommendation header with action color
    action_colors = {
        "BUY": "green",
        "SELL": "red",
        "HOLD": "orange"
    }
    
    action_color = action_colors.get(recommendation['action'], "gray")
    
    st.markdown(f"""
    ## <span style='color:{action_color}'>{recommendation['action']}</span> Prediction
    
    **Confidence:** {recommendation['confidence']*100:.1f}% | 
    **Urgency:** {recommendation['urgency']} | 
    **Risk Level:** {recommendation['risk_level']}
    
    **Suggested Exposure:** {recommendation['position_size']}
    """, unsafe_allow_html=True)
    
    st.info(recommendation['recommendation'])
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Sentiment Overview", "News Articles", "Sentiment History"])
    
    with tab1:
        # Display sentiment score gauge
        sentiment_score = sentiment_data['sentiment_score']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Sentiment Score"},
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
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("News Articles Analyzed", sentiment_data['news_count'])
        with col2:
            st.metric("Signal", sentiment_data['signal'].upper())
        with col3:
            trend_value = sentiment_data.get('sentiment_trend', 0)
            trend_arrow = "↗️" if trend_value > 0.1 else "↘️" if trend_value < -0.1 else "↔️"
            st.metric("Sentiment Trend", f"{trend_arrow} {trend_value:.2f}")
        
        # Analysis summary
        st.subheader("Analysis Summary")
        st.write(sentiment_data['analysis'])
        
        # Display sources
        st.subheader("News Sources")
        st.write(", ".join(sentiment_data['sources']) if sentiment_data['sources'] else "No sources available")
    
    with tab2:
        # Display news articles
        if 'article_sentiments' in sentiment_data and sentiment_data['article_sentiments']:
            st.subheader("Recent News Articles")
            
            for idx, article in enumerate(sentiment_data['article_sentiments']):
                sentiment_value = article['sentiment_score']
                sentiment_color = "green" if sentiment_value > 0.2 else "red" if sentiment_value < -0.2 else "gray"
                
                st.markdown(f"""
                #### Article {idx + 1} (Score: <span style='color:{sentiment_color}'>{sentiment_value:.2f}</span>)
                
                {article['text']}
                
                *Recency weight: {article['recency_weight']:.2f}*
                ---
                """, unsafe_allow_html=True)
        else:
            st.info("No news articles available for analysis")
    
    with tab3:
        st.subheader("Sentiment History")
        
        # Create dummy sentiment history if not available
        if symbol not in advisor.sentiment_analyzer.sentiment_history or not advisor.sentiment_analyzer.sentiment_history[symbol]:
            sentiment_history = []
            for i in range(7):
                sentiment_history.append({
                    'timestamp': datetime.now() - timedelta(days=i),
                    'sentiment': 0.2 - (i % 3) * 0.2  # Create some dummy values
                })
            st.info("Insufficient sentiment history. Showing sample data.")
        else:
            sentiment_history = advisor.sentiment_analyzer.sentiment_history[symbol]
        
        # Convert history to DataFrame and plot
        if sentiment_history:
            df = pd.DataFrame(sentiment_history)
            df = df.sort_values('timestamp')
            
            fig = px.line(
                df,
                x='timestamp',
                y='sentiment',
                title=f"Sentiment History for {symbol}",
                labels={'sentiment': 'Sentiment Score', 'timestamp': 'Date'}
            )
            
            # Add reference lines for sentiment zones
            fig.add_shape(
                type="line",
                x0=df['timestamp'].min(),
                x1=df['timestamp'].max(),
                y0=0.5,
                y1=0.5,
                line=dict(color="green", width=1, dash="dash"),
                name="Very Positive"
            )
            
            fig.add_shape(
                type="line",
                x0=df['timestamp'].min(),
                x1=df['timestamp'].max(),
                y0=0.2,
                y1=0.2,
                line=dict(color="lightgreen", width=1, dash="dash"),
                name="Positive"
            )
            
            fig.add_shape(
                type="line",
                x0=df['timestamp'].min(),
                x1=df['timestamp'].max(),
                y0=-0.2,
                y1=-0.2,
                line=dict(color="lightsalmon", width=1, dash="dash"),
                name="Negative"
            )
            
            fig.add_shape(
                type="line",
                x0=df['timestamp'].min(),
                x1=df['timestamp'].max(),
                y0=-0.5,
                y1=-0.5,
                line=dict(color="red", width=1, dash="dash"),
                name="Very Negative"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment history available")

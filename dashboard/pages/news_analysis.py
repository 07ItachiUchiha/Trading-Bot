import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.news_analysis import display_news_analysis

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - News Analysis",
        page_icon="📰",
        layout="wide"
    )

    # Display page content
    st.title("📰 News Trading Analysis")
    st.write("Analyze news sentiment and get trading recommendations based on market sentiment.")

    # Display symbol selector
    symbol = st.selectbox(
        "Select Symbol",
        options=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOGE/USD", "AAPL", "MSFT", "GOOGL", "AMZN"],
        index=0
    )

    # Display news analysis for the selected symbol
    display_news_analysis(symbol)

    # Additional information
    with st.expander("About News Analysis"):
        st.write("""
        This page provides trading recommendations based on news sentiment analysis:
        
        * **Sentiment Score**: Overall sentiment score from -1 (very negative) to 1 (very positive)
        * **Signal**: Recommended trading action based on sentiment analysis
        * **Confidence**: Confidence level in the recommendation
        * **Urgency**: How quickly you should consider acting on the recommendation
        * **Position Size**: Suggested position size based on sentiment strength and confidence
        
        The system analyzes news articles from multiple sources, weighing more recent news more heavily.
        Sentiment is tracked over time to identify trends and changes in market perception.
        """)

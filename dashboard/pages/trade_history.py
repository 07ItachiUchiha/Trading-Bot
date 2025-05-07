import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trade_filter import display_trade_filter

# Set page config
st.set_page_config(
    page_title="Trading Bot - Trade History",
    page_icon="📋",
    layout="wide"
)

# Display page content
st.title("🔍 Trade History")
st.write("Review, filter and analyze your trading history.")

# Display trade filtering system
display_trade_filter()

# Additional information
with st.expander("About Trade Filtering"):
    st.write("""
    This page allows you to filter and analyze your trading history. You can:
    
    * Filter trades by date range
    * Filter by symbol or action (Buy/Sell)
    * Search for specific trades
    * Download filtered data as CSV
    * View summary statistics for filtered trades
    
    Use this information to gain insights into your trading patterns and performance.
    """)

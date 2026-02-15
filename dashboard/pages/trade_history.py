import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trade_filter import display_trade_filter

def display_page():
    """Render trade history page."""
    st.title("Trade History")
    st.write("Review, filter and analyze your trading history.")

    display_trade_filter()

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


if __name__ == "__main__":
    st.set_page_config(page_title="Trade History", layout="wide")
    display_page()
else:
    display_page()

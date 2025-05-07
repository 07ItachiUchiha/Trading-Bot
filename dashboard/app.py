import streamlit as st

# This MUST be the first Streamlit command in the app
st.set_page_config(
    page_title="Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# All other imports and code should come after this line
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import page modules - note we're importing the display functions, not setting page config
try:
    from dashboard.pages.manual_trading import display_manual_trading
    from dashboard.pages.news_analysis import display_news_analysis
    from dashboard.pages.market_analysis import display_market_analysis_page
    from dashboard.pages.backtest_results import display_backtest_page
    from dashboard.pages.strategy_tester import display_strategy_tester_page
    from dashboard.pages.pnl_analysis import display_pnl_analysis_page
    from dashboard.pages.settings import display_settings_page
except ImportError as e:
    st.error(f"Error importing page modules: {e}")

# Import components
from dashboard.components.wallet import display_quick_wallet

def main():
    """Main app entry point"""
    st.sidebar.title("🤖 Trading Bot")
    
    # Display quick wallet summary in sidebar
    display_quick_wallet()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Manual Trading", "Market Analysis", "News Analysis", 
         "Strategy Tester", "Backtest Results", "PnL Analysis", "Settings"]
    )
    
    # Display the selected page with error handling
    try:
        if page == "Dashboard":
            display_dashboard()
        elif page == "Manual Trading":
            display_manual_trading()
        elif page == "Market Analysis":
            display_market_analysis_page()
        elif page == "News Analysis":
            display_news_analysis()
        elif page == "Strategy Tester":
            display_strategy_tester_page()
        elif page == "Backtest Results":
            display_backtest_page()
        elif page == "PnL Analysis":
            display_pnl_analysis_page()
        elif page == "Settings":
            display_settings_page()
    except Exception as e:
        st.error(f"Error displaying page: {e}")
        st.info("Some components might be missing. Please check your installation.")

def display_dashboard():
    """Display the main dashboard"""
    st.title("🤖 Trading Bot Dashboard")
    st.write("Welcome to your automated trading assistant.")
    
    # Dashboard components
    # ...existing code...

if __name__ == "__main__":
    main()
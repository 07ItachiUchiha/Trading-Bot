import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.pnl_visualization import display_pnl_visualization
from dashboard.components.trade_stats import display_trade_stats

def display_pnl_analysis_page():
    """Display the PnL analysis page"""
    st.title("💰 Profit and Loss Analysis")
    st.write("Analyze your trading performance and profitability metrics.")
    
    # Display PnL visualization
    display_pnl_visualization()
    
    # Display trade stats
    display_trade_stats()
    
    # Additional content goes here...

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - PnL Analysis",
        page_icon="💰",
        layout="wide"
    )
    display_pnl_analysis_page()
else:
    # When imported, just call the display function without setting page config
    main = display_pnl_analysis_page

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.technical_analysis import display_technical_analysis
from dashboard.components.market_overview import display_market_overview

def display_market_analysis_page():
    """Display the market analysis page"""
    st.title("📊 Market Analysis")
    st.write("Analyze markets with advanced technical and fundamental indicators.")
    
    # Create tabs for different analysis types
    tab1, tab2 = st.tabs(["Market Overview", "Technical Analysis"])
    
    with tab1:
        display_market_overview()
    
    with tab2:
        display_technical_analysis()
    
    # Additional content goes here...

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - Market Analysis",
        page_icon="📊",
        layout="wide"
    )
    display_market_analysis_page()
else:
    # When imported, just call the display function without setting page config
    main = display_market_analysis_page

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components with error handling
try:
    from dashboard.components.strategy_selector import display_strategy_selector
except ImportError:
    def display_strategy_selector():
        st.warning("Strategy selector component is not available.")
        st.info("Make sure you have implemented the strategy_selector.py component.")

try:
    from dashboard.components.strategy_parameters import display_strategy_parameters
except ImportError:
    def display_strategy_parameters():
        st.warning("Strategy parameters component is not available.")

def display_strategy_tester_page():
    """Display the strategy tester page"""
    st.title("🧪 Strategy Tester")
    st.write("Configure and test trading strategies against historical data.")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Strategy Selection", "Parameter Tuning"])
    
    with tab1:
        # Display strategy selector
        display_strategy_selector()
    
    with tab2:
        # Display strategy parameters
        try:
            display_strategy_parameters()
        except Exception as e:
            st.error(f"Error displaying strategy parameters: {e}")
    
    # Display warning about API errors
    st.warning("""
    **Note:** If you see API errors in the logs (403 Forbidden), please check your API credentials.
    Many errors show "Trading halted" due to authentication issues with the market data provider.
    """)

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - Strategy Tester",
        page_icon="🧪",
        layout="wide"
    )
    display_strategy_tester_page()
else:
    # When imported, just call the display function without setting page config
    main = display_strategy_tester_page

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components with error handling
try:
    from dashboard.components.backtest_visualizer import display_backtest_results
except ImportError:
    def display_backtest_results():
        st.warning("Backtest visualizer component is not available.")
        st.info("Make sure you have implemented the backtest_visualizer.py component.")

def display_backtest_page():
    """Display the backtest results page"""
    st.title("🔍 Backtest Results")
    st.write("Analyze strategy performance through historical simulations.")
    
    try:
        # Display backtest results component
        display_backtest_results()
    except Exception as e:
        st.error(f"Error displaying backtest results: {e}")
        st.info("This could be due to missing dependencies or data files.")
    
    # Display API errors warning
    st.warning("""
    **Note:** If you encounter "403 Forbidden" API errors, please check your API credentials.
    The current logs show authentication issues with the market data provider.
    """)

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - Backtest Results",
        page_icon="🔍",
        layout="wide"
    )
    display_backtest_page()
else:
    # When imported, just call the display function without setting page config
    main = display_backtest_page

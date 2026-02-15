import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.pnl_visualization import display_pnl_chart

def display_page():
    """Render PnL analysis page."""
    st.title("PnL Analysis")
    st.write("View and analyze your trading performance over time.")

    display_pnl_chart()

    with st.expander("About PnL Analysis"):
        st.write("""
        This page shows your Profit and Loss (PnL) performance over time. You can:

        * View cumulative PnL as a line chart
        * See separate buy/sell markers
        * Aggregate PnL by day
        * Analyze summary statistics

        Use this information to evaluate your trading strategy performance and identify areas for improvement.
        """)


if __name__ == "__main__":
    st.set_page_config(page_title="PnL Analysis", layout="wide")
    display_page()
else:
    display_page()

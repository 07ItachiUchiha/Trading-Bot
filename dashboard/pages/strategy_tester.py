import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart
from dashboard.components.strategy_selector import display_strategy_selector, display_strategy_performance

def main():
    """Display the strategy tester page"""
    st.title("Strategy Tester")
    st.write("Test and compare different trading strategies on historical data.")

    # Sidebar options
    with st.sidebar:
        st.header("Data Settings")
        
        symbol = st.text_input("Symbol", value=st.session_state.get('symbol', "BTC/USD"))
        
        interval = st.selectbox(
            "Time Interval",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=4
        )
        
        limit = st.slider(
            "Number of Candles",
            min_value=50,
            max_value=500,
            value=200,
            step=10
        )
        
        provider = st.selectbox(
            "Data Provider",
            options=["alpaca"],  # Removed binance option
            index=0
        )
        
        # Store symbol in session state for other components to use
        st.session_state.symbol = symbol

    # Fetch historical data
    try:
        with st.spinner("Fetching market data..."):
            data = fetch_historical_data(symbol, interval, limit, provider)
        
        # Display strategy selector
        strategy_name, signals, _ = display_strategy_selector(data)
        
        # Show candlestick chart with strategy signals
        st.subheader(f"{symbol} {interval} Chart with {strategy_name.replace('_', ' ').title()} Signals")
        fig = plot_candlestick_chart(data, signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display strategy performance
        display_strategy_performance(strategy_name, data, signals)
        
        st.info("Monitoring mode only: this tester reports signals and simulated performance.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Please check your input parameters and try again.")

if __name__ == "__main__":
    main()

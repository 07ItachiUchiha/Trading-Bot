import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart
from dashboard.components.strategy_selector import display_strategy_selector, display_strategy_performance

# Page configuration
st.set_page_config(
    page_title="Trading Bot - Strategy Tester",
    page_icon="📊",
    layout="wide"
)

st.title("Strategy Tester")
st.write("Test and compare different trading strategies on historical data.")

# Sidebar options
with st.sidebar:
    st.header("Data Settings")
    
    symbol = st.text_input("Symbol", value="BTC/USD")
    
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
        options=["alpaca", "binance"],
        index=0
    )
    
    # Store symbol in session state for other components to use
    st.session_state.symbol = symbol

# Fetch historical data
try:
    with st.spinner("Fetching market data..."):
        data = fetch_historical_data(symbol, interval, limit, provider)
    
    # Display strategy selector
    strategy_name, signals, live_execution = display_strategy_selector(data)
    
    # Show candlestick chart with strategy signals
    st.subheader(f"{symbol} {interval} Chart with {strategy_name.replace('_', ' ').title()} Signals")
    fig = plot_candlestick_chart(data, signals)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display strategy performance
    display_strategy_performance(strategy_name, data, signals)
    
    # Display execution status
    if live_execution:
        st.success("✅ Live execution is enabled for this strategy")
        
        with st.expander("Live Execution Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                risk_percent = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1)
                position_size = st.slider("Max Position Size (%)", 1.0, 50.0, 10.0, 1.0)
            
            with col2:
                take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 3.0, 0.5)
                stop_loss = st.slider("Stop Loss (%)", 1.0, 10.0, 2.0, 0.5)
            
            confirm_execution = st.button("Confirm Live Execution Settings")
            
            if confirm_execution:
                st.session_state["execution_confirmed"] = True
                st.session_state["execution_settings"] = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "interval": interval,
                    "risk_percent": risk_percent,
                    "position_size": position_size,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss
                }
                st.success("Execution settings confirmed and saved!")
    else:
        st.info("ℹ️ Live execution is disabled for this strategy")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your input parameters and try again.")

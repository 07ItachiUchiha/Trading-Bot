import streamlit as st
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trade_controls import display_trade_controls
from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart
from dashboard.components.strategy_selector import display_strategy_selector
from dashboard.components.wallet import display_quick_wallet

# Page configuration
st.set_page_config(
    page_title="Trading Bot - Manual Trading",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for trade notifications
if 'trade_notifications' not in st.session_state:
    st.session_state.trade_notifications = []

# Page title
st.title("Manual Trading")

# Symbol selection and chart options in sidebar
with st.sidebar:
    st.header("Chart Settings")
    
    symbol = st.text_input("Symbol", value="BTC/USD")
    
    interval = st.selectbox(
        "Time Interval",
        options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=4
    )
    
    limit = st.slider(
        "Number of Candles",
        min_value=20,
        max_value=500,
        value=100,
        step=10
    )
    
    provider = st.selectbox(
        "Data Provider",
        options=["alpaca", "binance"],
        index=0
    )
    
    # Store symbol in session state for other components to use
    st.session_state.symbol = symbol
    
    # Add wallet summary at bottom of sidebar
    st.markdown("---")
    display_quick_wallet()

# Create tabs for different views
chart_tab, trade_tab, strategy_tab, history_tab = st.tabs(["Chart", "Trade", "Strategy", "History"])

# Common data fetching for all tabs
@st.cache_data(ttl=5*60)  # Cache for 5 minutes
def get_chart_data():
    data = fetch_historical_data(symbol, interval, limit, provider)
    return data

try:
    data = get_chart_data()
    
    # Chart View Tab
    with chart_tab:
        st.subheader(f"{symbol} {interval} Chart")
        
        # Get signals from the selected strategy (if available)
        if 'selected_strategy_signals' in st.session_state:
            signals = st.session_state.selected_strategy_signals
        else:
            # Default to using the calculate_signals function as before
            from dashboard.components.trading import calculate_signals
            signals, data_with_indicators = calculate_signals(data)
        
        # Show candlestick chart
        fig = plot_candlestick_chart(data, signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal summary
        if signals and 'combined_signal' in signals:
            signal_info = signals['combined_signal']
            
            # Show signal with color coding
            col1, col2, col3 = st.columns(3)
            with col1:
                signal_color = (
                    "green" if signal_info['signal'].lower() == 'buy' 
                    else "red" if signal_info['signal'].lower() == 'sell' 
                    else "gray"
                )
                signal_text = signal_info['signal'].upper()
                confidence = signal_info.get('confidence', 0) * 100
                st.markdown(f"<h3 style='color: {signal_color};'>Signal: {signal_text}</h3>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col3:
                st.text(f"Reasoning: {signal_info.get('reasoning', 'N/A')}")
                
            # Quick trade buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Quick Buy", key="quick_buy", use_container_width=True, type="primary"):
                    st.session_state.quick_action = "buy"
                    st.experimental_rerun()
                    
            with col2:
                if st.button("Quick Sell", key="quick_sell", use_container_width=True):
                    st.session_state.quick_action = "sell"
                    st.experimental_rerun()
    
    # Strategy Tab - New Tab for Strategy Selection
    with strategy_tab:
        # Display strategy selector
        strategy_name, strategy_signals, live_execution = display_strategy_selector(data, key_prefix="manual_")
        
        # Store selected strategy signals in session state for use in chart tab
        st.session_state.selected_strategy_signals = strategy_signals
        
        # Show live execution settings if enabled
        if live_execution:
            st.success("✅ Live execution is enabled for this strategy")
            
            with st.expander("Live Execution Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_percent = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="strategy_risk")
                    position_size = st.slider("Max Position Size (%)", 1.0, 50.0, 10.0, 1.0, key="strategy_size")
                
                with col2:
                    take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 3.0, 0.5, key="strategy_tp")
                    stop_loss = st.slider("Stop Loss (%)", 1.0, 10.0, 2.0, 0.5, key="strategy_sl")
                
                confirm_execution = st.button("Confirm Live Execution Settings", key="confirm_strat_exec")
                
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
    
    # Trade View Tab        
    with trade_tab:
        # Check if quick action was requested from chart tab
        quick_action = None
        default_quantity = 0.01
        
        if hasattr(st.session_state, 'quick_action'):
            quick_action = st.session_state.quick_action
            # Reset after use
            del st.session_state.quick_action
        
        # Show the trade controls
        display_trade_controls(symbol, default_quantity)
        
    # History Tab
    with history_tab:
        st.subheader("Trade History")
        
        # Import the trade filtering component
        from dashboard.components.trade_filter import display_trade_filter
        
        # Show the trade filter
        display_trade_filter()
        
except Exception as e:
    st.error(f"Error loading chart data: {e}")
    st.info("Please check your symbol and try again.")

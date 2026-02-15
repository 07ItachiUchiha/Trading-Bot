import streamlit as st

# This MUST be the first Streamlit command in the app
st.set_page_config(
    page_title="Market Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# All other imports and code should come after this line
import os
import sys
import logging
from pathlib import Path
import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from components
from dashboard.components.auth import authenticate
from dashboard.components.database import ensure_db_exists
from dashboard.components.dashboard_ui import (
    render_prediction_log_tab,
    render_performance_tab,
    render_settings_tab,
)

# Import utils
from utils.websocket_manager import WebSocketManager

# Import from config
from config import API_KEY, API_SECRET

# Import our new components
from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart, calculate_signals
from dashboard.components.strategy_selector import display_strategy_selector

# Create database directory if it doesn't exist
os.makedirs(os.path.join(Path(__file__).parent, "data"), exist_ok=True)

# Initialize websocket managers globally
ws_manager = None
binance_ws_manager = None

def initialize_websocket(provider='alpaca'):
    """Initialize WebSocket manager for real-time data"""
    global ws_manager
    
    if provider == 'alpaca':
        if ws_manager is None:
            ws_manager = WebSocketManager(API_KEY, API_SECRET)
            ws_manager.start()
        return ws_manager
    else:
        logger = logging.getLogger('dashboard')
        logger.warning("Only Alpaca WebSocket provider is supported; falling back to Alpaca")
        return initialize_websocket('alpaca')

def handle_real_time_update(data):
    """Handle real-time data updates from WebSocket"""
    if 'data' in st.session_state and 'symbol' in st.session_state:
        symbol = st.session_state.symbol
        
        # Make sure the data is for our current symbol
        if data.get('symbol') == symbol:
            # Update session state with latest price
            st.session_state.last_price = data.get('price') or data.get('close')
            st.session_state.last_update_time = datetime.datetime.now()
            
            # If this is bar data, update the chart
            if data.get('type') == 'bar' or ('time' in data and 'open' in data and 'close' in data):
                if 'data' in st.session_state:
                    df = st.session_state.data
                    
                    # Use the Alpaca WebSocket manager
                    if ws_manager:
                        st.session_state.data = ws_manager.add_to_dataframe(df, data)
                    
                    # Update signals after new data
                    if st.session_state.get('update_signals', True):
                        from dashboard.components.trading import calculate_signals
                        signals, updated_data = calculate_signals(st.session_state.data)
                        st.session_state.signals = signals
                        st.session_state.data = updated_data
                
                # Force a rerun periodically to refresh the UI
                if not hasattr(st.session_state, 'last_rerun') or \
                   (datetime.datetime.now() - st.session_state.last_rerun).total_seconds() > 5:
                    st.session_state.last_rerun = datetime.datetime.now()
                    st.rerun()

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def fetch_cached_historical_data(symbol, interval, limit, provider):
    return fetch_historical_data(symbol, interval, limit, provider)

def main():
    """Main function to run the dashboard"""
    # Set up the database
    ensure_db_exists()
    
    # Check authentication
    authenticate()
    
    # Main page layout
    st.title("Market Prediction Dashboard")
    
    # Initialize session state variables
    if "update_signals" not in st.session_state:
        st.session_state.update_signals = True
    
    # Sidebar
    with st.sidebar:
        st.sidebar.subheader(f"Welcome, {st.session_state.user['username']}")
        
        # Data provider selection - only show Alpaca option
        data_provider = "alpaca"
        st.sidebar.text("Data Provider: Alpaca")
        
        # Store provider in session state
        st.session_state.data_provider = data_provider
            
        # Symbol selection
        st.sidebar.title("Market Selection")
        symbol = st.sidebar.text_input("Symbol", value="BTCUSD", key="sidebar_symbol")
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
            index=5,  # Default to 1h
            key="sidebar_timeframe"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True, key="auto_refresh")
        
        # Auto-update signals toggle - fixed to avoid duplicate widget ID
        auto_update_signals = st.sidebar.checkbox(
            "Auto-update signals", 
            value=st.session_state.update_signals,
            key="update_signals_toggle"        )
        
        # Update session state
        if st.session_state.update_signals != auto_update_signals:
            st.session_state.update_signals = auto_update_signals
        
        if st.sidebar.button("Refresh Data", key="sidebar_refresh_button") or (auto_refresh and "last_refresh" not in st.session_state):
            # Set up the appropriate WebSocket manager
            ws = initialize_websocket(data_provider)
            
            st.session_state.data = fetch_historical_data(symbol, timeframe, limit=100, provider=data_provider)
            st.session_state.last_updated = datetime.datetime.now()
            st.session_state.last_refresh = datetime.datetime.now()
            
            # Subscribe to real-time updates for this symbol
            if ws and auto_refresh:
                ws.subscribe(symbol, handle_real_time_update)
    
        st.sidebar.title("Notifications")
        notification_options = st.sidebar.multiselect(
            "Select notification channels",
            options=["Telegram", "Discord", "Slack", "Console"],
            default=st.session_state.get("notification_options", ["Console"]),
            key="sidebar_notification_options"
        )
        st.session_state.notification_options = notification_options
        
        # Display last update time if available
        if 'last_update_time' in st.session_state:
            st.sidebar.caption(f"Last update: {st.session_state.last_update_time.strftime('%H:%M:%S')}")
    
    # Always make sure these keys are initialized in session state
    if "symbol" not in st.session_state:
        st.session_state.symbol = symbol
    
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = timeframe
    
    # Initialize WebSocket for real-time updates
    ws = initialize_websocket(data_provider)
      # Initialize data if it doesn't exist
    if "data" not in st.session_state:
        st.session_state.data = fetch_historical_data(symbol, timeframe, limit=100, provider=data_provider)
        
        # Subscribe to real-time updates for this symbol
        if ws and auto_refresh:
            ws.subscribe(symbol, handle_real_time_update)
    # Update when symbol or timeframe changes
    elif st.session_state.symbol != symbol or st.session_state.timeframe != timeframe:
        # Symbol or timeframe has changed, update data
        if ws and auto_refresh:
            # Unsubscribe from old symbol
            ws.unsubscribe(st.session_state.symbol, handle_real_time_update)
        st.session_state.symbol = symbol
        st.session_state.timeframe = timeframe
        st.session_state.data = fetch_historical_data(symbol, timeframe, limit=100, provider=data_provider)
        
        # Subscribe to real-time updates for new symbol
        if ws and auto_refresh:
            ws.subscribe(symbol, handle_real_time_update)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Prediction Dashboard", "News Analysis", "Strategy Explorer", "Prediction Analytics", "Prediction Log"]
    )
    
    # Display selected page
    if page == "Prediction Dashboard":
        display_trading_dashboard()
    elif page == "News Analysis":
        from dashboard.pages.news_analysis import display_news_analysis
        display_news_analysis(symbol)
    elif page == "Strategy Explorer":
        display_strategy_tester()
    elif page == "Prediction Analytics":
        display_pnl_analysis()
    elif page == "Prediction Log":
        display_trade_history()

def display_strategy_tester():
    """Display the strategy explorer page."""
    # Import necessary components directly
    from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart
    from dashboard.components.strategy_selector import display_strategy_selector, display_strategy_performance
    
    st.title("Strategy Explorer")
    st.write("Test and compare prediction strategies on historical data.")

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
            options=["alpaca"],
            index=0
        )
        
        # Store symbol in session state for other components to use
        st.session_state.symbol = symbol

    # Fetch historical data
    try:
        with st.spinner("Fetching market data..."):
            # Use cached function to fetch data
            data = fetch_cached_historical_data(symbol, interval, limit, provider)
        
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

def display_trading_dashboard():
    """Display the main prediction dashboard."""
    st.title("Prediction Dashboard")
    
    # Add strategy selection in the prediction dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get market data for the default symbol
        symbol = st.session_state.get('symbol', 'BTC/USD')
        interval = st.session_state.get('interval', '1h')
        
        data = fetch_historical_data(symbol, interval, 100)
        
        # Display strategy selector in a smaller format
        strategy_name, signals, _ = display_strategy_selector(data, key_prefix="dashboard_")
        
        st.info(f"{strategy_name.replace('_', ' ').title()} is running in monitoring mode.")
    
    with col2:
        st.subheader("Model Snapshot")
        combined_signal = signals.get('combined_signal', {})
        st.metric("Signal", str(combined_signal.get('signal', 'neutral')).upper())
        st.metric("Confidence", f"{float(combined_signal.get('confidence', 0.0)):.2f}")
        st.caption(combined_signal.get('reasoning', 'No model rationale available.'))
    
    # Continue with other tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "News", "Prediction Log", "Settings"])
    
    with tab1:
        # Display market data and strategy information
        st.subheader(f"{symbol} {interval} Chart")
        fig = plot_candlestick_chart(data, signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key market stats
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'close' in data:
                current_price = data['close'].iloc[-1]
                open_price = data['open'].iloc[-1]
                change = (current_price - open_price) / open_price * 100
                st.metric("Current Price", f"${current_price:.2f}", f"{change:.2f}%")
            else:
                st.metric("Current Price", "N/A")
        
        with col2:
            if 'volume' in data:
                volume = data['volume'].iloc[-1]
                st.metric("Volume", f"{volume:,.0f}")
            else:
                st.metric("Volume", "N/A")
                
        with col3:
            # Prediction-only mode: show current signal rather than order activity
            combined_signal = signals.get('combined_signal', {})
            signal_name = str(combined_signal.get('signal', 'neutral')).upper()
            signal_conf = float(combined_signal.get('confidence', 0.0))
            if signal_name and signal_name != "NEUTRAL":
                st.metric("Current Signal", signal_name, f"{signal_conf:.2f} confidence")
            else:
                st.metric("Current Signal", "NEUTRAL")
    
    with tab2:
        # Display news sentiment analysis
        from dashboard.components.news_analysis import display_news_analysis
        display_news_analysis(symbol)
    
    with tab3:
        render_prediction_log_tab()
    
    with tab4:
        render_settings_tab()

def display_pnl_analysis():
    """Display prediction analytics page."""
    st.title("Prediction Analytics")
    render_performance_tab()

def display_trade_history():
    """Display prediction outcome log page."""
    st.title("Prediction Log")
    render_prediction_log_tab()

def display_settings():
    """Display settings page"""
    st.title("Settings")
    
    # ... existing code for settings ...

if __name__ == "__main__":
    main()

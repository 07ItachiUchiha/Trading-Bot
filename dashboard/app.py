import streamlit as st
import sys
from pathlib import Path
import os
import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set page config at the very beginning - must be the first Streamlit command
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Import from components
from components.auth import authenticate
from components.database import ensure_db_exists
from components.dashboard_ui import render_live_trading_tab, render_trade_history_tab, render_performance_tab, render_settings_tab
from components.trading import fetch_historical_data

# Import utils
from utils.websocket_manager import WebSocketManager
from utils.binance_websocket import BinanceWebSocketManager

# Import from config
from config import API_KEY, API_SECRET, DEFAULT_SYMBOLS

# Import our new components
from dashboard.components.pnl_visualization import display_pnl_chart
from dashboard.components.trade_filter import display_trade_filter
from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart, calculate_signals
from dashboard.components.strategy_selector import display_strategy_selector

# Create database directory if it doesn't exist
os.makedirs(os.path.join(Path(__file__).parent, "data"), exist_ok=True)

# Initialize websocket managers globally
ws_manager = None
binance_ws_manager = None

def initialize_websocket(provider='alpaca'):
    """Initialize WebSocket manager for real-time data"""
    global ws_manager, binance_ws_manager
    
    if provider == 'alpaca':
        if ws_manager is None:
            # Create a new instance if we don't have one yet
            # Note: The WebSocketManager class has its own singleton management
            ws_manager = WebSocketManager(API_KEY, API_SECRET)
            ws_manager.start()
        return ws_manager
    elif provider == 'binance':
        if binance_ws_manager is None:
            binance_ws_manager = BinanceWebSocketManager()
            binance_ws_manager.start()
        return binance_ws_manager

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
                    
                    # Use the appropriate WebSocket manager based on provider
                    if st.session_state.get('data_provider', 'alpaca') == 'alpaca' and ws_manager:
                        st.session_state.data = ws_manager.add_to_dataframe(df, data)
                    elif st.session_state.get('data_provider', 'alpaca') == 'binance' and binance_ws_manager:
                        st.session_state.data = binance_ws_manager.add_to_dataframe(df, data)
                    
                    # Update signals after new data
                    if st.session_state.get('update_signals', True):
                        from components.trading import calculate_signals
                        signals, updated_data = calculate_signals(st.session_state.data)
                        st.session_state.signals = signals
                        st.session_state.data = updated_data
                
                # Force a rerun periodically to refresh the UI
                if not hasattr(st.session_state, 'last_rerun') or \
                   (datetime.datetime.now() - st.session_state.last_rerun).total_seconds() > 5:
                    st.session_state.last_rerun = datetime.datetime.now()
                    st.rerun()

def main():
    """Main function to run the dashboard"""
    # Initialize the database
    ensure_db_exists()
    
    # Check authentication
    authenticate()
    
    # Main page layout
    st.title("🚀 Trading Bot Dashboard")
    
    # Initialize session state variables
    if "update_signals" not in st.session_state:
        st.session_state.update_signals = True
    
    # Sidebar
    with st.sidebar:
        st.sidebar.subheader(f"Welcome, {st.session_state.user['username']}")
        
        # Data provider selection
        data_provider = st.sidebar.selectbox(
            "Data Provider",
            options=["Alpaca", "Binance"],
            index=0,
            key="data_provider_select"
        )
        
        # Convert to lowercase for consistency in code
        data_provider = data_provider.lower()
        
        # Store provider in session state
        if "data_provider" not in st.session_state or st.session_state.data_provider != data_provider:
            st.session_state.data_provider = data_provider
            # Reset data if provider changes
            if "data" in st.session_state:
                del st.session_state.data
        
        st.sidebar.title("📊 Market Selection")
        symbol = st.sidebar.text_input("Symbol", value="BTCUSD", key="sidebar_symbol")
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
            index=5,  # Default to 1h
            key="sidebar_timeframe"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True, key="auto_refresh")
        
        # Auto-update signals toggle without directly modifying session state
        def update_signals_changed():
            pass  # The value is updated automatically via the key
            
        st.sidebar.checkbox(
            "Auto-update signals", 
            value=st.session_state.update_signals,
            key="update_signals",
            on_change=update_signals_changed
        )
        
        if st.sidebar.button("Refresh Data", key="sidebar_refresh_button") or (auto_refresh and "last_refresh" not in st.session_state):
            # Initialize the appropriate WebSocket manager
            ws = initialize_websocket(data_provider)
            
            st.session_state.data = fetch_historical_data(symbol, timeframe, provider=data_provider)
            st.session_state.last_updated = datetime.datetime.now()
            st.session_state.last_refresh = datetime.datetime.now()
            
            # Subscribe to real-time updates for this symbol
            if ws and auto_refresh:
                ws.subscribe(symbol, handle_real_time_update)
    
        st.sidebar.title("🔔 Notifications")
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
        st.session_state.data = fetch_historical_data(symbol, timeframe, provider=data_provider)
        
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
        st.session_state.data = fetch_historical_data(symbol, timeframe, provider=data_provider)
        
        # Subscribe to real-time updates for new symbol
        if ws and auto_refresh:
            ws.subscribe(symbol, handle_real_time_update)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Trading Dashboard", "Strategy Tester", "PnL Analysis", "Trade History", "Wallet", "Settings"]
    )
    
    # Display selected page
    if page == "Trading Dashboard":
        display_trading_dashboard()
    elif page == "Strategy Tester":
        display_strategy_tester()
    elif page == "PnL Analysis":
        display_pnl_analysis()
    elif page == "Trade History":
        display_trade_history()
    elif page == "Wallet":
        display_wallet_page()
    elif page == "Settings":
        display_settings()

    # Initialize default values in session state if not exist
    if "symbols" not in st.session_state:
        st.session_state.symbols = DEFAULT_SYMBOLS

def display_strategy_tester():
    """Display the strategy tester page"""
    # Import necessary components directly
    from dashboard.components.trading import fetch_historical_data, plot_candlestick_chart
    from dashboard.components.strategy_selector import display_strategy_selector, display_strategy_performance
    
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

def display_trading_dashboard():
    """Display the main trading dashboard"""
    st.title("Trading Dashboard")
    
    # Add strategy selection in the trading dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get market data for the default symbol
        symbol = st.session_state.get('symbol', 'BTC/USD')
        interval = st.session_state.get('interval', '1h')
        
        data = fetch_historical_data(symbol, interval, 100)
        
        # Display strategy selector in a smaller format
        strategy_name, signals, live_execution = display_strategy_selector(data, key_prefix="dashboard_")
        
        # Show status of selected strategy
        if live_execution:
            st.success(f"✅ {strategy_name.replace('_', ' ').title()} strategy is active for live trading")
        else:
            st.warning(f"⚠️ {strategy_name.replace('_', ' ').title()} strategy is in monitoring mode only")
    
    with col2:
        # Display quick account summary or other relevant info
        from dashboard.components.wallet import display_quick_wallet
        display_quick_wallet()
    
    # Continue with other tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Live Trading", "Trade History", "Performance", "Settings"])
    
    with tab1:
        # Pass both data and signals to render_live_trading_tab
        render_live_trading_tab(data, signals)
    
    with tab2:
        render_trade_history_tab()
    
    with tab3:
        render_performance_tab()
    
    with tab4:
        render_settings_tab()

def display_pnl_analysis():
    """Display PnL analysis page"""
    st.title("PnL Analysis")
    
    # Display PnL chart
    display_pnl_chart()

def display_trade_history():
    """Display trade history page"""
    st.title("Trade History")
    
    # Display trade filtering system
    display_trade_filter()

def display_settings():
    """Display settings page"""
    st.title("Settings")
    
    # ... existing code for settings ...

def display_wallet_page():
    """Display wallet page"""
    st.title("Wallet & Account")
    
    # Import and display wallet component
    from dashboard.components.wallet import display_wallet
    display_wallet()

if __name__ == "__main__":
    main()
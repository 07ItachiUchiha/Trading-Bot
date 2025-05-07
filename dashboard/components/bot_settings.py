import streamlit as st
import os
import json
from pathlib import Path

def display_bot_settings():
    """Display bot settings configuration UI"""
    st.header("🤖 Bot Settings")
    
    # Load current settings
    settings = load_bot_settings()
    
    # Create tabs for different setting groups
    tab1, tab2, tab3 = st.tabs(["General Settings", "Risk Management", "Advanced Settings"])
    
    with tab1:
        settings = display_general_settings(settings)
    
    with tab2:
        settings = display_risk_management_settings(settings)
    
    with tab3:
        settings = display_advanced_settings(settings)
    
    # Save settings button
    if st.button("Save Settings", type="primary"):
        save_bot_settings(settings)
        st.success("Settings saved successfully!")
        
        # Re-initialize the bot with new settings (this would be replaced with your actual code)
        st.info("Bot re-initialized with new settings")

def display_general_settings(settings):
    """Display general bot settings"""
    st.subheader("General Settings")
    
    general = settings.get('general', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        general['bot_name'] = st.text_input(
            "Bot Name",
            value=general.get('bot_name', 'Trading Bot'),
            help="Name of your trading bot"
        )
        
        trading_enabled = general.get('trading_enabled', False)
        general['trading_enabled'] = st.checkbox(
            "Enable Trading",
            value=trading_enabled,
            help="Enable/disable automated trading"
        )
        
        if trading_enabled:
            st.success("Automated trading is enabled")
        else:
            st.warning("Automated trading is disabled")
    
    with col2:
        general['trading_environment'] = st.selectbox(
            "Trading Environment",
            options=["Paper Trading", "Live Trading"],
            index=0 if general.get('trading_environment', 'Paper Trading') == "Paper Trading" else 1,
            help="Select the trading environment"
        )
        
        # Add warning message for live trading
        if general['trading_environment'] == "Live Trading":
            st.warning("⚠️ Live trading mode will use real funds!")
    
    # Trading pairs settings
    st.subheader("Trading Pairs")
    
    default_pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    trading_pairs_text = "\n".join(general.get('trading_pairs', default_pairs))
    
    trading_pairs_input = st.text_area(
        "Trading Pairs (one per line)",
        value=trading_pairs_text,
        height=100,
        help="Enter trading pairs, one per line (e.g., BTC/USD)"
    )
    
    # Parse trading pairs
    general['trading_pairs'] = [pair.strip() for pair in trading_pairs_input.split("\n") if pair.strip()]
    
    # Display paired tokens
    if general['trading_pairs']:
        st.write(f"**{len(general['trading_pairs'])} trading pairs configured**")
    else:
        st.error("No trading pairs configured")
    
    # Timeframe settings
    st.subheader("Trading Timeframes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        general['primary_timeframe'] = st.selectbox(
            "Primary Timeframe",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(general.get('primary_timeframe', '1h')),
            help="Primary timeframe for trading decisions"
        )
    
    with col2:
        default_secondary = ["5m", "15m", "1h"]
        if 'secondary_timeframes' not in general:
            general['secondary_timeframes'] = default_secondary
            
        general['secondary_timeframes'] = st.multiselect(
            "Secondary Timeframes",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            default=general['secondary_timeframes'],
            help="Secondary timeframes for confirmation"
        )
    
    # Update settings
    settings['general'] = general
    return settings

def display_risk_management_settings(settings):
    """Display risk management settings"""
    st.subheader("Risk Management Settings")
    
    risk = settings.get('risk_management', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk['max_risk_per_trade'] = st.number_input(
            "Max Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(risk.get('max_risk_per_trade', 1.0)),
            step=0.1,
            help="Maximum percentage of portfolio to risk on any single trade"
        )
        
        risk['max_open_positions'] = st.number_input(
            "Max Open Positions",
            min_value=1,
            max_value=50,
            value=int(risk.get('max_open_positions', 5)),
            step=1,
            help="Maximum number of open positions at any time"
        )
        
        risk['max_open_positions_per_symbol'] = st.number_input(
            "Max Positions Per Symbol",
            min_value=1,
            max_value=10,
            value=int(risk.get('max_open_positions_per_symbol', 1)),
            step=1,
            help="Maximum positions for a single trading symbol"
        )
    
    with col2:
        risk['max_daily_drawdown'] = st.number_input(
            "Max Daily Drawdown (%)",
            min_value=0.1,
            max_value=20.0,
            value=float(risk.get('max_daily_drawdown', 5.0)),
            step=0.1,
            help="Bot will stop trading if daily drawdown exceeds this percentage"
        )
        
        risk['trailing_stop_enabled'] = st.checkbox(
            "Enable Trailing Stop",
            value=risk.get('trailing_stop_enabled', True),
            help="Automatically adjust stop loss as price moves in your favor"
        )
        
        if risk['trailing_stop_enabled']:
            risk['trailing_stop_activation'] = st.number_input(
                "Trailing Stop Activation (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(risk.get('trailing_stop_activation', 1.0)),
                step=0.1,
                help="Profit percentage required to activate trailing stop"
            )
    
    # Stop loss settings
    st.subheader("Stop Loss Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk['default_stop_loss'] = st.number_input(
            "Default Stop Loss (%)",
            min_value=0.1,
            max_value=15.0,
            value=float(risk.get('default_stop_loss', 2.0)),
            step=0.1,
            help="Default stop loss percentage"
        )
    
    with col2:
        risk['stop_loss_type'] = st.selectbox(
            "Stop Loss Type",
            options=["Fixed", "ATR", "Volatility"],
            index=["Fixed", "ATR", "Volatility"].index(risk.get('stop_loss_type', 'Fixed')),
            help="Method for calculating stop loss"
        )
    
    with col3:
        risk['stop_loss_timeframe'] = st.selectbox(
            "Stop Loss Timeframe",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(risk.get('stop_loss_timeframe', '1h')),
            help="Timeframe to use for stop loss calculation"
        )
    
    # Take profit settings
    st.subheader("Take Profit Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk['take_profit_type'] = st.selectbox(
            "Take Profit Type",
            options=["Fixed", "Risk-Reward", "Fibonacci"],
            index=["Fixed", "Risk-Reward", "Fibonacci"].index(risk.get('take_profit_type', 'Risk-Reward')),
            help="Method for calculating take profit"
        )
    
    with col2:
        if risk['take_profit_type'] == "Fixed":
            risk['take_profit_value'] = st.number_input(
                "Take Profit (%)",
                min_value=0.1,
                max_value=50.0,
                value=float(risk.get('take_profit_value', 5.0)),
                step=0.1,
                help="Fixed take profit percentage"
            )
        elif risk['take_profit_type'] == "Risk-Reward":
            risk['risk_reward_ratio'] = st.number_input(
                "Risk-Reward Ratio",
                min_value=0.5,
                max_value=10.0,
                value=float(risk.get('risk_reward_ratio', 2.0)),
                step=0.1,
                help="Take profit as multiple of risk (e.g., 2.0 = 2:1 reward:risk)"
            )
        else:  # Fibonacci
            risk['fibonacci_level'] = st.selectbox(
                "Fibonacci Level",
                options=["0.382", "0.618", "1.0", "1.618", "2.618"],
                index=["0.382", "0.618", "1.0", "1.618", "2.618"].index(risk.get('fibonacci_level', '1.618')),
                help="Fibonacci extension level for take profit"
            )
    
    # Position sizing settings
    st.subheader("Position Sizing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk['position_sizing_type'] = st.selectbox(
            "Position Sizing Method",
            options=["Fixed", "Risk-Based", "Kelly"],
            index=["Fixed", "Risk-Based", "Kelly"].index(risk.get('position_sizing_type', 'Risk-Based')),
            help="Method for calculating position size"
        )
    
    with col2:
        if risk['position_sizing_type'] == "Fixed":
            risk['fixed_position_size'] = st.number_input(
                "Fixed Position Size (%)",
                min_value=0.1,
                max_value=100.0,
                value=float(risk.get('fixed_position_size', 10.0)),
                step=0.1,
                help="Fixed position size as percentage of portfolio"
            )
        elif risk['position_sizing_type'] == "Kelly":
            risk['kelly_fraction'] = st.number_input(
                "Kelly Fraction",
                min_value=0.1,
                max_value=1.0,
                value=float(risk.get('kelly_fraction', 0.5)),
                step=0.1,
                help="Fraction of full Kelly criterion to use (0.5 = Half Kelly)"
            )
    
    # Advanced risk settings
    with st.expander("Advanced Risk Settings"):
        risk['max_correlated_trades'] = st.number_input(
            "Max Correlated Trades",
            min_value=1,
            max_value=10,
            value=int(risk.get('max_correlated_trades', 2)),
            step=1,
            help="Maximum number of highly correlated trades"
        )
        
        risk['correlation_threshold'] = st.number_input(
            "Correlation Threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(risk.get('correlation_threshold', 0.7)),
            step=0.05,
            help="Correlation threshold for considering assets as correlated"
        )
        
        risk['max_drawdown_exit'] = st.checkbox(
            "Exit All Positions on Max Drawdown",
            value=risk.get('max_drawdown_exit', True),
            help="Close all positions if max drawdown is reached"
        )
    
    # Update settings
    settings['risk_management'] = risk
    return settings

def display_advanced_settings(settings):
    """Display advanced bot settings"""
    st.subheader("Advanced Settings")
    
    advanced = settings.get('advanced', {})
    
    # Strategy settings
    st.subheader("Strategy Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        advanced['primary_strategy'] = st.selectbox(
            "Primary Strategy",
            options=["Combined", "EMA Crossover", "Bollinger Bands", "Breakout", "Sentiment-Based"],
            index=["Combined", "EMA Crossover", "Bollinger Bands", "Breakout", "Sentiment-Based"].index(advanced.get('primary_strategy', 'Combined')),
            help="Primary trading strategy"
        )
    
    with col2:
        advanced['strategy_conflict_resolution'] = st.selectbox(
            "Signal Conflict Resolution",
            options=["Majority Vote", "Weighted", "Primary Only", "Most Confident"],
            index=["Majority Vote", "Weighted", "Primary Only", "Most Confident"].index(advanced.get('strategy_conflict_resolution', 'Weighted')),
            help="How to handle conflicting signals from multiple strategies"
        )
    
    # Custom strategy weights
    if advanced['strategy_conflict_resolution'] == "Weighted":
        st.subheader("Strategy Weights")
        
        weights = advanced.get('strategy_weights', {
            "technical": 0.6,
            "sentiment": 0.3,
            "fundamental": 0.1
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weights['technical'] = st.slider(
                "Technical Analysis Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(weights.get('technical', 0.6)),
                step=0.05,
                help="Weight for technical analysis signals"
            )
        
        with col2:
            weights['sentiment'] = st.slider(
                "Sentiment Analysis Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(weights.get('sentiment', 0.3)),
                step=0.05,
                help="Weight for sentiment analysis signals"
            )
        
        with col3:
            weights['fundamental'] = st.slider(
                "Fundamental Analysis Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(weights.get('fundamental', 0.1)),
                step=0.05,
                help="Weight for fundamental analysis signals"
            )
        
        # Normalize weights
        total = weights['technical'] + weights['sentiment'] + weights['fundamental']
        if total > 0:
            normalized = {k: v/total for k, v in weights.items()}
            st.caption(f"Normalized weights: Technical {normalized['technical']:.2f}, Sentiment {normalized['sentiment']:.2f}, Fundamental {normalized['fundamental']:.2f}")
        
        advanced['strategy_weights'] = weights
    
    # Performance management
    st.subheader("Performance Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        advanced['profit_target_daily'] = st.number_input(
            "Daily Profit Target (%)",
            min_value=0.1,
            max_value=20.0,
            value=float(advanced.get('profit_target_daily', 1.0)),
            step=0.1,
            help="Stop trading when daily profit target is reached"
        )
    
    with col2:
        advanced['auto_shutdown_enabled'] = st.checkbox(
            "Auto-Shutdown on Target",
            value=advanced.get('auto_shutdown_enabled', False),
            help="Automatically shut down trading when profit target is reached"
        )
    
    # Execution settings
    st.subheader("Execution Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        advanced['order_type'] = st.selectbox(
            "Default Order Type",
            options=["Market", "Limit"],
            index=["Market", "Limit"].index(advanced.get('order_type', 'Market')),
            help="Default order type for trade execution"
        )
        
        if advanced['order_type'] == "Limit":
            advanced['limit_offset_pct'] = st.number_input(
                "Limit Price Offset (%)",
                min_value=-5.0,
                max_value=5.0,
                value=float(advanced.get('limit_offset_pct', 0.1)),
                step=0.05,
                help="Offset from current price for limit orders (negative = below market, positive = above market)"
            )
    
    with col2:
        advanced['slippage_tolerance'] = st.number_input(
            "Slippage Tolerance (%)",
            min_value=0.01,
            max_value=5.0,
            value=float(advanced.get('slippage_tolerance', 0.1)),
            step=0.01,
            help="Maximum allowed slippage percentage"
        )
        
        advanced['retry_attempts'] = st.number_input(
            "Order Retry Attempts",
            min_value=0,
            max_value=10,
            value=int(advanced.get('retry_attempts', 3)),
            step=1,
            help="Number of retry attempts for failed orders"
        )
    
    # System settings
    with st.expander("System Settings"):
        advanced['log_level'] = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(advanced.get('log_level', 'INFO')),
            help="Logging verbosity level"
        )
        
        advanced['data_storage_days'] = st.number_input(
            "Data Storage (days)",
            min_value=1,
            max_value=365,
            value=int(advanced.get('data_storage_days', 30)),
            step=1,
            help="Number of days to keep historical data"
        )
        
        advanced['backup_enabled'] = st.checkbox(
            "Enable Automatic Backups",
            value=advanced.get('backup_enabled', True),
            help="Automatically backup trading data and settings"
        )
        
        if advanced['backup_enabled']:
            advanced['backup_frequency'] = st.selectbox(
                "Backup Frequency",
                options=["Daily", "Weekly", "Monthly"],
                index=["Daily", "Weekly", "Monthly"].index(advanced.get('backup_frequency', 'Daily')),
                help="How often to create backups"
            )
    
    # Update settings
    settings['advanced'] = advanced
    return settings

def load_bot_settings():
    """Load bot settings from storage"""
    # In a real implementation, this would load from a file or database
    # For now, return default settings
    return {
        'general': {
            'bot_name': 'Trading Bot',
            'trading_enabled': False,
            'trading_environment': 'Paper Trading',
            'trading_pairs': ["BTC/USD", "ETH/USD", "SOL/USD"],
            'primary_timeframe': '1h',
            'secondary_timeframes': ["5m", "15m", "1h"]
        },
        'risk_management': {
            'max_risk_per_trade': 1.0,
            'max_open_positions': 5,
            'max_open_positions_per_symbol': 1,
            'max_daily_drawdown': 5.0,
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 1.0,
            'default_stop_loss': 2.0,
            'stop_loss_type': 'Fixed',
            'stop_loss_timeframe': '1h',
            'take_profit_type': 'Risk-Reward',
            'risk_reward_ratio': 2.0,
            'position_sizing_type': 'Risk-Based',
            'max_correlated_trades': 2,
            'correlation_threshold': 0.7,
            'max_drawdown_exit': True
        },
        'advanced': {
            'primary_strategy': 'Combined',
            'strategy_conflict_resolution': 'Weighted',
            'strategy_weights': {
                'technical': 0.6,
                'sentiment': 0.3,
                'fundamental': 0.1
            },
            'profit_target_daily': 1.0,
            'auto_shutdown_enabled': False,
            'order_type': 'Market',
            'slippage_tolerance': 0.1,
            'retry_attempts': 3,
            'log_level': 'INFO',
            'data_storage_days': 30,
            'backup_enabled': True,
            'backup_frequency': 'Daily'
        }
    }

def save_bot_settings(settings):
    """Save bot settings to storage"""
    # In a real implementation, this would save to a file or database
    # For now, just save to session state
    st.session_state.bot_settings = settings
    
    # Optionally, you could save to a file
    try:
        settings_dir = Path(__file__).parent.parent.parent / "config"
        settings_dir.mkdir(exist_ok=True)
        
        with open(settings_dir / "bot_settings.json", "w") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        st.warning(f"Could not save settings to file: {e}")

if __name__ == "__main__":
    # Test the component
    st.set_page_config(page_title="Bot Settings Test", layout="wide")
    display_bot_settings()

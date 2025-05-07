import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import necessary modules
from utils.telegram_alert import send_alert
from utils.discord_webhook import send_discord_alert
from utils.slack_webhook import send_slack_alert
from utils.export import export_to_excel, export_to_google_sheets
from config import CAPITAL, RISK_PERCENT, MAX_CAPITAL_PER_TRADE, DEFAULT_SYMBOLS, PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT

# Import local modules
from .trading import (
    fetch_historical_data, calculate_signals, plot_candlestick_chart, 
    calculate_pnl, start_auto_trader, stop_auto_trader, get_auto_trader_status
)
from .database import get_trades_from_db, add_trade_to_db, update_trade_in_db

def render_live_trading_tab(data, signals=None):
    """Render the Live Trading tab content with strategy signals"""
    st.subheader("Live Trading")
    
    # Safety check for data
    if data is None or len(data) < 2:
        st.error("Insufficient data to display trading interface")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Market Status")
        last_price = data.iloc[-1]['close']
        price_change = last_price - data.iloc[-2]['close']
        price_change_pct = price_change / data.iloc[-2]['close'] * 100
        
        st.metric(
            label=f"{st.session_state.symbol} Price",
            value=f"${last_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        st.subheader("Strategy Status")
        # If signals are not provided, calculate them
        if signals is None:
            signals, _ = calculate_signals(data)
        
        if 'buy_signals' in signals and signals['buy_signals'].iloc[-5:].any():
            status = "BUY"
            color = "green"
        elif 'sell_signals' in signals and signals['sell_signals'].iloc[-5:].any():
            status = "SELL"
            color = "red"
        else:
            status = "NEUTRAL"
            color = "gray"
        
        st.markdown(f"<h3 style='color: {color};'>{status}</h3>", unsafe_allow_html=True)
    
    with col3:
        st.subheader("Auto Trading")
        auto_status = get_auto_trader_status()
        
        if auto_status["running"]:
            st.success("Auto trader is running")
            if st.button("Stop Auto Trader"):
                success, msg = stop_auto_trader()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.warning("Auto trader is stopped")
            if st.button("Start Auto Trader"):
                # Get parameters from session state or settings
                symbols = st.session_state.get('symbols', DEFAULT_SYMBOLS)
                timeframe = st.session_state.get('timeframe', '1h')
                capital = st.session_state.get('capital', CAPITAL)
                risk_percent = st.session_state.get('risk_percent', RISK_PERCENT)
                profit_target = st.session_state.get('profit_target', PROFIT_TARGET_PERCENT)
                daily_profit_target = st.session_state.get('daily_profit_target', DAILY_PROFIT_TARGET_PERCENT)
                
                success, msg = start_auto_trader(
                    symbols=symbols,
                    timeframe=timeframe,
                    capital=capital,
                    risk_percent=risk_percent,
                    profit_target=profit_target,
                    daily_profit_target=daily_profit_target
                )
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    
    # Use signals if provided, otherwise calculate them
    if signals is None:
        signals, data_with_indicators = calculate_signals(data)
        # Create a copy to avoid modifying input data
        chart_data = data_with_indicators
    else:
        # Use input data for chart
        chart_data = data
    
    # Add sentiment analysis display
    if 'combined_signal' in signals:
        combined_signal = signals['combined_signal']
        st.subheader("Signal Analysis")
        
        # Create columns for displaying signal information
        sig_col1, sig_col2, sig_col3 = st.columns(3)
        
        with sig_col1:
            signal_color = {
                'buy': 'green',
                'sell': 'red',
                'neutral': 'gray'
            }.get(combined_signal['signal'], 'gray')
            
            st.markdown(f"""
            <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 5px;'>
                <h3 style='color: {signal_color}; margin:0;'>{combined_signal['signal'].upper()}</h3>
                <p>Confidence: {combined_signal['confidence']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with sig_col2:
            st.markdown(f"""
            <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 5px; height: 100%;'>
                <p style='margin:0;'><strong>Analysis:</strong></p>
                <p>{combined_signal['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with sig_col3:
            if 'sentiment_signal' in combined_signal:
                sentiment = combined_signal['sentiment_signal']
                st.markdown(f"""
                <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 5px;'>
                    <p style='margin:0;'><strong>News Sentiment:</strong></p>
                    <p>Score: {sentiment['score']:.2f}</p>
                    <p>Articles: {sentiment.get('news_count', 0)} articles from {sentiment.get('source_count', 0)} sources</p>
                </div>
                """, unsafe_allow_html=True)
    
    chart = plot_candlestick_chart(chart_data, signals)
    st.plotly_chart(chart, use_container_width=True)
    
    # Trading interface
    st.subheader("Manual Trading")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Open Position")
        
        trade_direction = st.radio(
            "Direction",
            options=["Long", "Short"],
            horizontal=True
        )
        
        entry_price = st.number_input("Entry Price", value=last_price, format="%.2f", key="manual_entry_price")
        
        risk_percent = st.slider("Risk (%)", 0.5, 5.0, RISK_PERCENT, 0.1, key="manual_risk_percent")
        
        stop_distance = st.number_input(
            "Stop Distance (%)",
            value=2.0,
            format="%.2f",
            key="manual_stop_distance"
        )
        
        # Calculate position size
        if trade_direction == "Long":
            direction = "long"
            stop_price = entry_price * (1 - stop_distance / 100)
        else:
            direction = "short"
            stop_price = entry_price * (1 + stop_distance / 100)
        
        risk_amount = CAPITAL * risk_percent / 100
        position_size = risk_amount / abs(entry_price - stop_price)
        
        st.info(f"Position Size: {position_size:.4f}")
        
        # Calculate profit targets
        if trade_direction == "Long":
            target1 = entry_price * (1 + stop_distance / 100)
            target2 = entry_price * (1 + stop_distance / 100 * 1.5)
            target3 = entry_price * (1 + stop_distance / 100 * 2)
        else:
            target1 = entry_price * (1 - stop_distance / 100)
            target2 = entry_price * (1 - stop_distance / 100 * 1.5)
            target3 = entry_price * (1 - stop_distance / 100 * 2)
        
        col_tp1, col_tp2, col_tp3 = st.columns(3)
        with col_tp1:
            st.metric("Target 1", f"{target1:.2f}")
        with col_tp2:
            st.metric("Target 2", f"{target2:.2f}")
        with col_tp3:
            st.metric("Target 3", f"{target3:.2f}")
        
        if st.button("Open Position"):
            trade_data = {
                'symbol': st.session_state.get('symbol', 'UNKNOWN'),
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_price,
                'targets': [target1, target2, target3],
                'size': position_size,
                'entry_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'open'
            }
            
            add_trade_to_db(trade_data)
            
            # Send alerts if enabled
            notification_options = st.session_state.get('notification_options', ['Console'])
            if "Telegram" in notification_options:
                send_alert(f"New trade opened: {direction} {trade_data['symbol']} at {entry_price}")
            if "Discord" in notification_options:
                send_discord_alert(f"New trade opened: {direction} {trade_data['symbol']} at {entry_price}")
            if "Slack" in notification_options:
                send_slack_alert(f"New trade opened: {direction} {trade_data['symbol']} at {entry_price}")
            
            st.success(f"{direction.capitalize()} position opened at {entry_price}")
    
    with col2:
        st.subheader("Close Position")
        
        # Get open trades from database
        trades_df = get_trades_from_db()
        if trades_df is None or trades_df.empty:
            st.info("No open positions to close")
            return
        
        open_trades = trades_df[trades_df['status'] == 'open']
        
        if open_trades.empty:
            st.info("No open positions to close")
        else:
            # Format trade info for selectbox
            trade_options = []
            for _, trade in open_trades.iterrows():
                trade_options.append(f"{trade['id']} - {trade['symbol']} {trade['direction']} at {trade['entry_price']}")
            
            selected_trade_option = st.selectbox("Select trade to close", trade_options)
            
            if selected_trade_option:
                selected_trade_id = int(selected_trade_option.split(" - ")[0])
                selected_trade = open_trades[open_trades['id'] == selected_trade_id].iloc[0]
                
                col_entry, col_dir = st.columns(2)
                with col_entry:
                    st.metric("Entry Price", f"{selected_trade['entry_price']:.2f}")
                with col_dir:
                    st.metric("Direction", selected_trade['direction'])
                
                exit_price = st.number_input("Exit Price", value=last_price, format="%.2f", key="close_exit_price")
                
                # Calculate PNL
                pnl = calculate_pnl(
                    selected_trade['entry_price'],
                    exit_price,
                    selected_trade['direction'],
                    selected_trade['size']
                )
                
                pnl_color = "green" if pnl >= 0 else "red"
                st.markdown(f"<h3 style='color: {pnl_color};'>PNL: ${pnl:.2f}</h3>", unsafe_allow_html=True)
                
                if st.button("Close Position"):
                    exit_data = {
                        'exit_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'status': 'closed'
                    }
                    
                    update_trade_in_db(selected_trade_id, exit_data)
                    
                    # Send alerts if enabled
                    notification_options = st.session_state.get('notification_options', ['Console'])
                    if "Telegram" in notification_options:
                        send_alert(f"Trade closed: {selected_trade['direction']} {selected_trade['symbol']} with PNL: ${pnl:.2f}")
                    if "Discord" in notification_options:
                        send_discord_alert(f"Trade closed: {selected_trade['direction']} {selected_trade['symbol']} with PNL: ${pnl:.2f}")
                    if "Slack" in notification_options:
                        send_slack_alert(f"Trade closed: {selected_trade['direction']} {selected_trade['symbol']} with PNL: ${pnl:.2f}")
                    
                    st.success(f"Position closed with PNL: ${pnl:.2f}")

def render_trade_history_tab():
    """Render the Trade History tab content"""
    st.subheader("Trade History")
    
    # Get trades from database
    trades_df = get_trades_from_db()
    
    if trades_df.empty:
        st.info("No trade history available")
    else:
        # Create filters
        col1, col2, col3 = st.columns(3)
        with col1:
            symbols = ['All'] + trades_df['symbol'].unique().tolist()
            symbol_filter = st.selectbox("Filter by Symbol", symbols)
            
        with col2:
            statuses = ['All'] + trades_df['status'].unique().tolist()
            status_filter = st.selectbox("Filter by Status", statuses)
            
        with col3:
            directions = ['All'] + trades_df['direction'].unique().tolist()
            direction_filter = st.selectbox("Filter by Direction", directions)
        
        # Apply filters
        filtered_df = trades_df.copy()
        
        if symbol_filter != 'All':
            filtered_df = filtered_df[filtered_df['symbol'] == symbol_filter]
            
        if status_filter != 'All':
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
        if direction_filter != 'All':
            filtered_df = filtered_df[filtered_df['direction'] == direction_filter]
        
        # Calculate summary statistics
        if not filtered_df.empty:
            closed_trades = filtered_df[filtered_df['status'] == 'closed']
            
            if not closed_trades.empty:
                total_pnl = closed_trades['pnl'].sum()
                win_trades = closed_trades[closed_trades['pnl'] > 0]
                loss_trades = closed_trades[closed_trades['pnl'] <= 0]
                
                win_rate = len(win_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total PNL", f"${total_pnl:.2f}")
                with col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("Win Trades", f"{len(win_trades)}")
                with col4:
                    st.metric("Loss Trades", f"{len(loss_trades)}")
        
        # Format data for display
        display_df = filtered_df.copy()
        
        # Format targets
        display_df['targets'] = display_df['targets'].apply(
            lambda x: ", ".join([f"{target:.2f}" for target in x]) if isinstance(x, list) else ""
        )
        
        # Format times
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Reorder columns
        columns_order = [
            'id', 'symbol', 'direction', 'status',
            'entry_price', 'exit_price', 'pnl',
            'entry_time', 'exit_time'
        ]
        display_df = display_df[columns_order]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to Excel"):
                file_path = export_to_excel(filtered_df, "trade_history")
                st.success(f"Exported to {file_path}")
                
        with col2:
            if st.button("Export to Google Sheets"):
                sheet_url = export_to_google_sheets(filtered_df, "Trading Bot - Trade History")
                if sheet_url:
                    st.success(f"Exported to Google Sheets")
                else:
                    st.error("Failed to export to Google Sheets")

def render_performance_tab():
    """Render the Performance tab content"""
    st.subheader("Performance Analytics")
    
    # Get trades data
    trades_df = get_trades_from_db()
    closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
    
    if closed_trades.empty:
        st.info("No closed trades available for analysis")
        return
    
    # Calculate daily PnL
    closed_trades['date'] = closed_trades['exit_time'].dt.date
    daily_pnl = closed_trades.groupby('date')['pnl'].sum().reset_index()
    daily_pnl['cumulative_pnl'] = daily_pnl['pnl'].cumsum()
    
    # Plot cumulative PnL
    st.subheader("Cumulative Profit/Loss")
    import plotly.express as px
    
    fig = px.line(
        daily_pnl,
        x='date',
        y='cumulative_pnl',
        title='Cumulative PnL Over Time'
    )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative PnL ($)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Statistics")
        
        total_pnl = closed_trades['pnl'].sum()
        win_trades = closed_trades[closed_trades['pnl'] > 0]
        loss_trades = closed_trades[closed_trades['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
        
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        
        profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else 0
        
        # Calculate max drawdown
        running_pnl = closed_trades.sort_values('exit_time')['pnl'].cumsum()
        max_drawdown = 0
        peak = running_pnl.iloc[0] if len(running_pnl) > 0 else 0
        
        for pnl in running_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        stats = pd.DataFrame({
            'Metric': [
                'Total PnL', 'Win Rate', 'Total Trades',
                'Win Trades', 'Loss Trades', 'Avg Win',
                'Avg Loss', 'Profit Factor', 'Max Drawdown'
            ],
            'Value': [
                f"${total_pnl:.2f}", f"{win_rate:.1f}%", str(len(closed_trades)),
                str(len(win_trades)), str(len(loss_trades)), f"${avg_win:.2f}",
                f"${avg_loss:.2f}", f"{profit_factor:.2f}", f"${max_drawdown:.2f}"
            ]
        })
        
        # Convert all values to strings to avoid type conversion issues with Arrow
        stats['Value'] = stats['Value'].astype(str)
        
        st.dataframe(stats, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Performance by Symbol")
        
        symbol_perf = closed_trades.groupby('symbol').agg({
            'pnl': 'sum',
            'id': 'count'
        }).reset_index()
        
        symbol_perf.columns = ['Symbol', 'PnL', 'Trades']
        symbol_perf['Win Rate'] = [
            f"{len(closed_trades[(closed_trades['symbol'] == symbol) & (closed_trades['pnl'] > 0)]) / count * 100:.1f}%"
            for symbol, count in zip(symbol_perf['Symbol'], symbol_perf['Trades'])
        ]
        
        symbol_perf['PnL'] = symbol_perf['PnL'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(symbol_perf, use_container_width=True, hide_index=True)
    
    # Display monthly performance
    st.subheader("Monthly Performance")
    closed_trades['month'] = closed_trades['exit_time'].dt.strftime('%Y-%m')
    monthly_perf = closed_trades.groupby('month').agg({
        'pnl': 'sum',
        'id': 'count'
    }).reset_index()
    
    monthly_perf.columns = ['Month', 'PnL', 'Trades']
    monthly_perf['Win Rate'] = [
        f"{len(closed_trades[(closed_trades['month'] == month) & (closed_trades['pnl'] > 0)]) / count * 100:.1f}%"
        for month, count in zip(monthly_perf['Month'], monthly_perf['Trades'])
    ]
    
    monthly_perf['PnL'] = monthly_perf['PnL'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(monthly_perf, use_container_width=True, hide_index=True)

def render_settings_tab():
    """Render the Settings tab content"""
    st.subheader("Application Settings")
    
    # Risk management settings
    st.subheader("Risk Management")
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        capital = st.number_input("Trading Capital ($)", value=CAPITAL, format="%.2f", key="settings_capital")
        risk_percent = st.number_input("Risk Per Trade (%)", value=RISK_PERCENT, format="%.2f", key="settings_risk_percent")
    
    with risk_col2:
        max_capital_per_trade = st.number_input("Max Capital Per Trade (%)", value=MAX_CAPITAL_PER_TRADE*100, format="%.2f", key="settings_max_capital")
        st.info(f"Max trade size: ${capital * max_capital_per_trade/100:.2f}")
    
    # Save risk settings to session state
    if st.button("Save Risk Settings"):
        st.session_state.capital = capital
        st.session_state.risk_percent = risk_percent
        st.session_state.max_capital_per_trade = max_capital_per_trade / 100
        st.success("Risk settings saved!")
    
    # Auto trading settings
    st.subheader("Auto Trading Settings")
    
    auto_col1, auto_col2 = st.columns(2)
    
    with auto_col1:
        # Update default symbols to use Alpaca format (USD instead of USDT)
        default_symbols = [s.replace('USDT', 'USD') for s in DEFAULT_SYMBOLS]
        symbols_input = st.text_input("Trading Symbols (comma-separated)", value=",".join(default_symbols))
        profit_target = st.number_input("Profit Target (%)", value=PROFIT_TARGET_PERCENT, format="%.2f", key="settings_profit_target")
    
    with auto_col2:
        daily_profit_target = st.number_input("Daily Profit Target ($)", value=DAILY_PROFIT_TARGET_PERCENT, format="%.2f", key="settings_daily_target")
    
    # Advanced settings
    show_advanced = st.checkbox("Show Advanced Settings")
    
    if show_advanced:
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            use_news = st.checkbox("Use News Strategy", value=True)
            news_weight = st.slider("News Impact Weight", 0.0, 1.0, 0.5, 0.1, key="settings_news_weight")
        
        with adv_col2:
            use_earnings = st.checkbox("Use Earnings Reports", value=True)
            earnings_weight = st.slider("Earnings Impact Weight", 0.0, 1.0, 0.6, 0.1, key="settings_earnings_weight")
    else:
        use_news = True
        news_weight = 0.5
        use_earnings = True
        earnings_weight = 0.6
    
    # Save auto trading settings
    if st.button("Save Auto Trading Settings"):
        symbols_list = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        st.session_state.symbols = symbols_list
        st.session_state.profit_target = profit_target
        st.session_state.daily_profit_target = daily_profit_target
        st.session_state.use_news = use_news
        st.session_state.news_weight = news_weight
        st.session_state.use_earnings = use_earnings
        st.session_state.earnings_weight = earnings_weight
        
        st.success("Auto trading settings saved!")
    
    # Notification settings
    st.subheader("Notification Settings")
    notification_options = st.multiselect(
        "Select notification channels",
        options=["Telegram", "Discord", "Slack", "Console"],
        default=st.session_state.get("notification_options", ["Console"]),
        key="settings_notification_options"  # Add a unique key to fix duplicate element ID
    )
    
    if st.button("Save Notification Settings"):
        st.session_state.notification_options = notification_options
        st.success("Notification settings saved!")
    
    # Strategy selection
    st.subheader("Strategy Settings")
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        selected_strategy = st.selectbox(
            "Trading Strategy",
            options=["Combined Strategy", "Bollinger Bands + RSI", "EMA Crossover", "Breakout Detection"],
            index=0,
            key="settings_strategy"
        )
        
        # Update session state
        st.session_state.selected_strategy = selected_strategy
        
    with strategy_col2:
        # Strategy description based on selection
        strategy_descriptions = {
            "Combined Strategy": "Uses multiple indicators and sentiment analysis to generate stronger signals",
            "Bollinger Bands + RSI": "Combines Bollinger Bands for volatility with RSI for momentum",
            "EMA Crossover": "Uses exponential moving average crossovers for trend following",
            "Breakout Detection": "Identifies price breakouts from consolidation patterns"
        }
        
        st.info(strategy_descriptions.get(selected_strategy, ""))
    
    # Strategy parameters section based on selected strategy
    if selected_strategy == "Bollinger Bands + RSI":
        st.subheader("Bollinger Bands + RSI Parameters")
        bb_col1, bb_col2 = st.columns(2)
        
        with bb_col1:
            bb_length = st.number_input("Bollinger Length", min_value=5, max_value=50, value=20, step=1, key="bb_length")
            bb_std = st.number_input("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="bb_std")
        
        with bb_col2:
            rsi_length = st.number_input("RSI Length", min_value=2, max_value=30, value=14, step=1, key="rsi_length")
            rsi_ob = st.number_input("RSI Overbought", min_value=50, max_value=90, value=70, step=1, key="rsi_ob")
            rsi_os = st.number_input("RSI Oversold", min_value=10, max_value=50, value=30, step=1, key="rsi_os")
    
    elif selected_strategy == "EMA Crossover":
        st.subheader("EMA Crossover Parameters")
        ema_col1, ema_col2 = st.columns(2)
        
        with ema_col1:
            ema_fast = st.number_input("Fast EMA", min_value=5, max_value=100, value=50, step=1, key="ema_fast")
        
        with ema_col2:
            ema_slow = st.number_input("Slow EMA", min_value=20, max_value=500, value=200, step=10, key="ema_slow")
    
    elif selected_strategy == "Breakout Detection":
        st.subheader("Breakout Detection Parameters")
        breakout_col1, breakout_col2 = st.columns(2)
        
        with breakout_col1:
            consol_periods = st.number_input("Consolidation Periods", min_value=5, max_value=30, value=14, step=1, key="consol_periods")
            bb_threshold = st.number_input("Squeeze Threshold", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="bb_threshold")
        
        with breakout_col2:
            volume_increase = st.number_input("Volume Increase (%)", min_value=50, max_value=300, value=150, step=10, key="volume_increase")
    
    else:  # Combined strategy
        st.subheader("Combined Strategy Parameters")
        st.info("This strategy automatically combines multiple indicators and sentiment analysis for stronger signals.")
        
        # Specify the weights
        st.write("Component Weights")
        weights_col1, weights_col2, weights_col3 = st.columns(3)
        
        with weights_col1:
            tech_weight = st.slider("Technical Weight", min_value=0.2, max_value=1.0, value=0.5, step=0.1, key="combined_tech_weight")
        
        with weights_col2:
            sentiment_weight = st.slider("Sentiment Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="combined_sentiment_weight")
        
        with weights_col3:
            earnings_weight = st.slider("Earnings Weight", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="combined_earnings_weight")
        
        # Normalize weights
        total_weight = tech_weight + sentiment_weight + earnings_weight
        if total_weight > 0:
            norm_tech = tech_weight / total_weight
            norm_sentiment = sentiment_weight / total_weight
            norm_earnings = earnings_weight / total_weight
            
            st.caption(f"Normalized weights: Technical {norm_tech:.2f}, Sentiment {norm_sentiment:.2f}, Earnings {norm_earnings:.2f}")
    
    # Save strategy settings
    if st.button("Save Strategy Settings", key="save_strategy_settings"):
        # Save the selected strategy
        st.session_state.selected_strategy = selected_strategy
        
        # Save parameters specific to the selected strategy
        if selected_strategy == "Bollinger Bands + RSI":
            st.session_state.bb_length = bb_length
            st.session_state.bb_std = bb_std
            st.session_state.rsi_length = rsi_length
            st.session_state.rsi_ob = rsi_ob
            st.session_state.rsi_os = rsi_os
        
        elif selected_strategy == "EMA Crossover":
            st.session_state.ema_fast = ema_fast
            st.session_state.ema_slow = ema_slow
        
        elif selected_strategy == "Breakout Detection":
            st.session_state.consol_periods = consol_periods
            st.session_state.bb_threshold = bb_threshold
            st.session_state.volume_increase = volume_increase
        
        else:  # Combined strategy
            st.session_state.combined_tech_weight = norm_tech
            st.session_state.combined_sentiment_weight = norm_sentiment
            st.session_state.combined_earnings_weight = norm_earnings
        
        st.success("Strategy settings saved!")
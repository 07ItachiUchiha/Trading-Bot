import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.export import export_to_excel, export_to_google_sheets
from config import CAPITAL, RISK_PERCENT, MAX_CAPITAL_PER_TRADE, DEFAULT_SYMBOLS, PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT

# Import local modules
from .trading import (
    calculate_signals, plot_candlestick_chart
)
from .database import get_trades_from_db

def render_live_trading_tab(data, signals=None):
    """Render a prediction-only live tab with no execution controls."""
    st.subheader("Live Prediction")

    if data is None or len(data) < 2:
        st.error("Insufficient data to display prediction interface")
        return

    if signals is None:
        signals, data_with_indicators = calculate_signals(data)
        chart_data = data_with_indicators
    else:
        chart_data = data

    last_price = data.iloc[-1]["close"]
    prev_price = data.iloc[-2]["close"]
    price_change_pct = ((last_price - prev_price) / prev_price) * 100 if prev_price else 0.0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Market Status")
        st.metric(
            label=f"{st.session_state.symbol} Price",
            value=f"${last_price:.2f}",
            delta=f"{price_change_pct:.2f}%",
        )

    combined_signal = signals.get(
        "combined_signal",
        {"signal": "neutral", "confidence": 0.0, "reasoning": "No signal data"},
    )
    signal_name = str(combined_signal.get("signal", "neutral")).upper()
    signal_confidence = float(combined_signal.get("confidence", 0.0))
    signal_reasoning = combined_signal.get("reasoning", "No reasoning provided")

    with col2:
        st.subheader("Signal")
        signal_color = {"BUY": "green", "SELL": "red", "NEUTRAL": "gray"}.get(signal_name, "gray")
        st.markdown(f"<h3 style='color: {signal_color};'>{signal_name}</h3>", unsafe_allow_html=True)

    with col3:
        st.subheader("Confidence")
        st.metric("Model Confidence", f"{signal_confidence:.2f}")

    st.subheader("Signal Analysis")
    info_col1, info_col2 = st.columns([1, 2])
    with info_col1:
        st.write("**Signal Reasoning**")
        st.caption(signal_reasoning)
    with info_col2:
        sentiment = combined_signal.get("sentiment_signal", {})
        if sentiment:
            st.write("**Sentiment Snapshot**")
            st.caption(
                f"Score: {float(sentiment.get('score', 0.0)):.2f} | "
                f"Articles: {sentiment.get('news_count', 0)} | "
                f"Sources: {sentiment.get('source_count', 0)}"
            )
        else:
            st.write("**Sentiment Snapshot**")
            st.caption("No live sentiment details available for this symbol.")

    chart = plot_candlestick_chart(chart_data, signals)
    st.plotly_chart(chart, use_container_width=True)

def render_prediction_log_tab():
    """Render the prediction outcome log tab content."""
    st.subheader("Prediction Outcome Log")
    
    # Load historical outcomes from local database
    trades_df = get_trades_from_db()
    
    if trades_df.empty:
        st.info("No historical prediction outcomes available")
    else:
        # Create filters
        col1, col2, col3 = st.columns(3)
        with col1:
            symbols = ['All'] + trades_df['symbol'].unique().tolist()
            symbol_filter = st.selectbox("Filter by Symbol", symbols)
            
        with col2:
            statuses = ['All'] + trades_df['status'].unique().tolist()
            status_filter = st.selectbox("Filter by Outcome Status", statuses)
            
        with col3:
            directions = ['All'] + trades_df['direction'].unique().tolist()
            direction_filter = st.selectbox("Filter by Predicted Direction", directions)
        
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
                    st.metric("Net Outcome", f"${total_pnl:.2f}")
                with col2:
                    st.metric("Positive Rate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("Positive Outcomes", f"{len(win_trades)}")
                with col4:
                    st.metric("Negative Outcomes", f"{len(loss_trades)}")
        
        # Format data for display
        display_df = filtered_df.copy()
        
        # Format targets
        display_df['targets'] = display_df['targets'].apply(
            lambda x: ", ".join([f"{target:.2f}" for target in x]) if isinstance(x, list) else ""
        )
        
        # Format times
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Reorder and relabel columns for prediction context
        display_df = display_df.rename(columns={
            'direction': 'predicted_direction',
            'status': 'outcome_status',
            'entry_price': 'baseline_price',
            'exit_price': 'resolved_price',
            'pnl': 'outcome_delta',
            'entry_time': 'signal_time',
            'exit_time': 'resolution_time',
        })
        columns_order = [
            'id', 'symbol', 'predicted_direction', 'outcome_status',
            'baseline_price', 'resolved_price', 'outcome_delta',
            'signal_time', 'resolution_time'
        ]
        display_df = display_df[columns_order]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to Excel"):
                file_path = export_to_excel(filtered_df, "prediction_outcome_log")
                st.success(f"Exported to {file_path}")
                
        with col2:
            if st.button("Export to Google Sheets"):
                sheet_url = export_to_google_sheets(filtered_df, "Market Prediction - Outcome Log")
                if sheet_url:
                    st.success(f"Exported to Google Sheets")
                else:
                    st.error("Failed to export to Google Sheets")

def render_performance_tab():
    """Render prediction performance analytics tab content."""
    st.subheader("Prediction Outcome Analytics")
    
    # Get trades data
    trades_df = get_trades_from_db()
    closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
    
    if closed_trades.empty:
        st.info("No resolved outcomes available for analysis")
        return
    
    # Calculate daily PnL
    closed_trades['date'] = closed_trades['exit_time'].dt.date
    daily_pnl = closed_trades.groupby('date')['pnl'].sum().reset_index()
    daily_pnl['cumulative_pnl'] = daily_pnl['pnl'].cumsum()
    
    # Plot cumulative outcome
    st.subheader("Cumulative Outcome")
    import plotly.express as px
    
    fig = px.line(
        daily_pnl,
        x='date',
        y='cumulative_pnl',
        title='Cumulative Outcome Over Time'
    )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Outcome ($)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Outcome Statistics")
        
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
                'Net Outcome', 'Positive Rate', 'Total Resolved Signals',
                'Positive Outcomes', 'Negative Outcomes', 'Avg Positive Outcome',
                'Avg Negative Outcome', 'Outcome Factor', 'Max Drawdown'
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
        st.subheader("Outcome by Symbol")
        
        symbol_perf = closed_trades.groupby('symbol').agg({
            'pnl': 'sum',
            'id': 'count'
        }).reset_index()
        
        symbol_perf.columns = ['Symbol', 'Net Outcome', 'Resolved Signals']
        symbol_perf['Win Rate'] = [
            f"{len(closed_trades[(closed_trades['symbol'] == symbol) & (closed_trades['pnl'] > 0)]) / count * 100:.1f}%"
            for symbol, count in zip(symbol_perf['Symbol'], symbol_perf['Resolved Signals'])
        ]
        
        symbol_perf['Net Outcome'] = symbol_perf['Net Outcome'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(symbol_perf, use_container_width=True, hide_index=True)
    
    # Display monthly performance
    st.subheader("Monthly Outcome")
    closed_trades['month'] = closed_trades['exit_time'].dt.strftime('%Y-%m')
    monthly_perf = closed_trades.groupby('month').agg({
        'pnl': 'sum',
        'id': 'count'
    }).reset_index()
    
    monthly_perf.columns = ['Month', 'Net Outcome', 'Resolved Signals']
    monthly_perf['Win Rate'] = [
        f"{len(closed_trades[(closed_trades['month'] == month) & (closed_trades['pnl'] > 0)]) / count * 100:.1f}%"
        for month, count in zip(monthly_perf['Month'], monthly_perf['Resolved Signals'])
    ]
    
    monthly_perf['Net Outcome'] = monthly_perf['Net Outcome'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(monthly_perf, use_container_width=True, hide_index=True)

def render_settings_tab():
    """Render the Settings tab content"""
    st.subheader("Application Settings")
    
    # Guardrail settings
    st.subheader("Model Guardrails")
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        capital = st.number_input("Reference Capital ($)", value=CAPITAL, format="%.2f", key="settings_capital")
        risk_percent = st.number_input("Max Risk Budget (%)", value=RISK_PERCENT, format="%.2f", key="settings_risk_percent")
    
    with risk_col2:
        max_capital_per_trade = st.number_input("Max Allocation Per Signal (%)", value=MAX_CAPITAL_PER_TRADE*100, format="%.2f", key="settings_max_capital")
        st.info(f"Max notional per signal: ${capital * max_capital_per_trade/100:.2f}")
    
    # Save guardrails to session state
    if st.button("Save Guardrails"):
        st.session_state.capital = capital
        st.session_state.risk_percent = risk_percent
        st.session_state.max_capital_per_trade = max_capital_per_trade / 100
        st.success("Guardrail settings saved!")
    
    # Prediction settings
    st.subheader("Prediction Settings")
    
    auto_col1, auto_col2 = st.columns(2)
    
    with auto_col1:
        # Update default symbols to use Alpaca format (USD instead of USDT)
        default_symbols = [s.replace('USDT', 'USD') for s in DEFAULT_SYMBOLS]
        symbols_input = st.text_input("Watchlist Symbols (comma-separated)", value=",".join(default_symbols))
        profit_target = st.number_input("Signal Target Threshold (%)", value=PROFIT_TARGET_PERCENT, format="%.2f", key="settings_profit_target")
    
    with auto_col2:
        daily_profit_target = st.number_input("Daily Signal Budget", value=DAILY_PROFIT_TARGET_PERCENT, format="%.2f", key="settings_daily_target")
    
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
    
    # Save prediction settings
    if st.button("Save Prediction Settings"):
        symbols_list = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        st.session_state.symbols = symbols_list
        st.session_state.profit_target = profit_target
        st.session_state.daily_profit_target = daily_profit_target
        st.session_state.use_news = use_news
        st.session_state.news_weight = news_weight
        st.session_state.use_earnings = use_earnings
        st.session_state.earnings_weight = earnings_weight
        
        st.success("Prediction settings saved!")
    
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
    
    # Ensure these options match exactly with those in strategy_selector.py
    strategy_options = ["Combined Strategy", "Bollinger Bands + RSI", "EMA Crossover", "Breakout Detection"]
    
    with strategy_col1:
        selected_strategy = st.selectbox(
            "Prediction Strategy",
            options=strategy_options,
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
        
        # Initialize session state values for weights if they don't exist
        if 'combined_tech_weight_value' not in st.session_state:
            st.session_state.combined_tech_weight_value = 0.5
        if 'combined_sentiment_weight_value' not in st.session_state:
            st.session_state.combined_sentiment_weight_value = 0.3
        if 'combined_earnings_weight_value' not in st.session_state:
            st.session_state.combined_earnings_weight_value = 0.2
            
        with weights_col1:
            tech_weight = st.slider("Technical Weight", min_value=0.2, max_value=1.0, 
                                   value=st.session_state.combined_tech_weight_value, 
                                   step=0.1, key="tech_weight_slider")  # Changed key to avoid conflict
        
        with weights_col2:
            sentiment_weight = st.slider("Sentiment Weight", min_value=0.0, max_value=1.0, 
                                        value=st.session_state.combined_sentiment_weight_value, 
                                        step=0.1, key="sentiment_weight_slider")  # Changed key
            
        with weights_col3:
            earnings_weight = st.slider("Earnings Weight", min_value=0.0, max_value=1.0,
                                       value=st.session_state.combined_earnings_weight_value,
                                       step=0.1, key="earnings_weight_slider")  # Changed key

        # Save button and normalization moved here
        if st.button("Save Strategy Settings", key="save_strategy_settings"):
            # Save the selected strategy to session state
            st.session_state.selected_strategy = selected_strategy
            
            # Save parameters specific to the selected strategy
            if selected_strategy == "Bollinger Bands + RSI":
                # ...existing code...
                pass
            elif selected_strategy == "EMA Crossover":
                # ...existing code...
                pass
            else:  # Combined strategy
                # Normalize weights to ensure they sum to 1.0
                total_weight = tech_weight + sentiment_weight + earnings_weight
                if total_weight > 0:  # Avoid division by zero
                    st.session_state.combined_tech_weight_value = tech_weight / total_weight
                    st.session_state.combined_sentiment_weight_value = sentiment_weight / total_weight
                    st.session_state.combined_earnings_weight_value = earnings_weight / total_weight
                
            st.success("Strategy settings saved!")

# Backward-compatible alias for older imports
def render_trade_history_tab():
    return render_prediction_log_tab()

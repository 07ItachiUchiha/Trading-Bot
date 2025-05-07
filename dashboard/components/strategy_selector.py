import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import strategies
from strategy.multiple_strategies import (
    get_strategy_by_name,
    get_available_strategy_names,
    get_available_strategy_display_names
)

def display_strategy_selector(data, key_prefix=""):
    """
    Display strategy selector dropdown and strategy information
    
    Args:
        data (pd.DataFrame): Price data with OHLC columns
        key_prefix (str): Optional prefix for Streamlit widget keys
        
    Returns:
        tuple: (selected_strategy_name, signals)
    """
    st.subheader("📊 Strategy Selection")
    
    # Get available strategies
    strategy_names = get_available_strategy_names()
    strategy_display_names = get_available_strategy_display_names()
    
    # Create a dictionary mapping display names to internal names
    strategy_map = dict(zip(strategy_display_names, strategy_names))
    
    # Initialize session state if needed
    if f"{key_prefix}selected_strategy" not in st.session_state:
        st.session_state[f"{key_prefix}selected_strategy"] = strategy_display_names[0]
        
    if f"{key_prefix}live_execution" not in st.session_state:
        st.session_state[f"{key_prefix}live_execution"] = False
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Strategy selector dropdown
        selected_strategy_display = st.selectbox(
            "Select Trading Strategy",
            options=strategy_display_names,
            index=strategy_display_names.index(st.session_state[f"{key_prefix}selected_strategy"]),
            key=f"{key_prefix}strategy_selector"
        )
        st.session_state[f"{key_prefix}selected_strategy"] = selected_strategy_display
    
    with col2:
        # Live execution toggle
        live_execution = st.toggle(
            "Live Execution",
            value=st.session_state[f"{key_prefix}live_execution"],
            key=f"{key_prefix}live_execution_toggle",
            help="Enable or disable live trading with this strategy"
        )
        st.session_state[f"{key_prefix}live_execution"] = live_execution
    
    # Get the internal strategy name
    selected_strategy_name = strategy_map[selected_strategy_display]
    
    # Get the strategy instance
    strategy = get_strategy_by_name(selected_strategy_name)
    
    # Calculate signals using the selected strategy
    signals = strategy.generate_signals(data)
    
    # Display strategy explanation
    with st.expander("Strategy Explanation", expanded=False):
        st.markdown(strategy.get_visual_explanation())
        
        # Display current signal from the strategy
        if 'combined_signal' in signals:
            combined_signal = signals['combined_signal']
            
            signal_color = {
                'buy': 'green',
                'sell': 'red',
                'neutral': 'gray'
            }.get(combined_signal['signal'], 'gray')
            
            st.markdown(f"""
            <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 5px;'>
                <h3 style='color: {signal_color}; margin:0;'>Current Signal: {combined_signal['signal'].upper()}</h3>
                <p>Confidence: {combined_signal['confidence']:.2f}</p>
                <p>Reasoning: {combined_signal['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)
        
    # Return the selected strategy name and signals
    return selected_strategy_name, signals, live_execution

def display_strategy_performance(strategy_name, data, signals):
    """
    Display performance metrics for the selected strategy
    
    Args:
        strategy_name (str): Name of the selected strategy
        data (pd.DataFrame): Price data
        signals (dict): Trading signals
    """
    st.subheader("Strategy Performance Metrics")
    
    # Check if we have buy/sell signals
    if 'buy_signals' not in signals or 'sell_signals' not in signals:
        st.warning("No trading signals available for this strategy.")
        return
    
    buy_signals = signals['buy_signals']
    sell_signals = signals['sell_signals']
    
    # Count signals
    buy_count = buy_signals.sum()
    sell_count = sell_signals.sum()
    
    # Calculate simple backtest metrics
    initial_price = data['close'].iloc[0]
    final_price = data['close'].iloc[-1]
    buy_and_hold_return = (final_price - initial_price) / initial_price * 100
    
    # Simple strategy return calculation
    # (Note: This is a simplified calculation for demonstration)
    strategy_return = 0
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    
    for i in range(len(data)):
        if buy_signals.iloc[i] and position <= 0:
            # Buy signal when not in long position
            position = 1
            entry_price = data['close'].iloc[i]
        elif sell_signals.iloc[i] and position >= 0:
            # Sell signal when not in short position
            if position == 1:
                # Close long position
                strategy_return += (data['close'].iloc[i] - entry_price) / entry_price * 100
            position = -1
            entry_price = data['close'].iloc[i]
    
    # Close any open position at the end
    if position == 1:
        strategy_return += (final_price - entry_price) / entry_price * 100
    elif position == -1:
        strategy_return += (entry_price - final_price) / entry_price * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Buy Signals", f"{buy_count}")
    
    with col2:
        st.metric("Sell Signals", f"{sell_count}")
    
    with col3:
        st.metric("Buy & Hold", f"{buy_and_hold_return:.2f}%")
    
    with col4:
        st.metric("Strategy Return", f"{strategy_return:.2f}%", delta=f"{strategy_return - buy_and_hold_return:.2f}%")

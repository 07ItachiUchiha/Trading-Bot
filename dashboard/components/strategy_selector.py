import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from trading.py for indicator calculations
from dashboard.components.trading import (
    calculate_bollinger_bands, 
    calculate_ema, 
    calculate_rsi, 
    calculate_signals
)

def display_strategy_selector(data, key_prefix=""):
    """Show the strategy picker dropdown and run the selected strategy."""
    st.subheader("📊 Strategy Selection")
    
    # Ensure this list matches exactly with the options in settings
    strategy_options = [
        "Combined Strategy", 
        "Bollinger Bands + RSI", 
        "EMA Crossover", 
        "Breakout Detection"
    ]
    
    # Initialize session state if needed
    if f"{key_prefix}selected_strategy" not in st.session_state:
        st.session_state[f"{key_prefix}selected_strategy"] = "Combined Strategy"
        
    selected_strategy = st.selectbox(
        "Select Prediction Strategy",
        options=strategy_options,
        index=strategy_options.index(st.session_state.get(f"{key_prefix}selected_strategy", "Combined Strategy")),
        key=f"{key_prefix}strategy_selector"
    )
    st.session_state[f"{key_prefix}selected_strategy"] = selected_strategy
    
    # Calculate signals using the selected strategy
    signals = apply_strategy(data, selected_strategy)
    
    # Display strategy explanation
    with st.expander("Strategy Explanation", expanded=False):
        st.markdown(get_strategy_explanation(selected_strategy))
        
        # Display current signal from the strategy
        if signals and 'combined_signal' in signals:
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
    return selected_strategy, signals, False

def get_strategy_explanation(strategy_name):
    """Return a markdown blurb explaining how the strategy works."""
    explanations = {
        "Combined Strategy": """
        ## Combined Strategy
        
        The Combined Strategy integrates multiple technical indicators and sentiment analysis to generate stronger, more reliable signals.
        
        **Key Components:**
        - Technical indicators (Bollinger Bands, EMAs, RSI)
        - Volume analysis
        - Pattern recognition
        - News sentiment (when available)
        """,
        
        "Bollinger Bands + RSI": """
        ## Bollinger Bands + RSI Strategy
        
        This strategy combines Bollinger Bands to identify volatility with the Relative Strength Index (RSI) for momentum confirmation.
        
        **Buy Signal:** Price touches lower Bollinger Band while RSI is oversold (below 30)
        **Sell Signal:** Price touches upper Bollinger Band while RSI is overbought (above 70)
        """,
        
        "EMA Crossover": """
        ## EMA Crossover Strategy
        
        The EMA Crossover strategy uses exponential moving averages to identify trend changes.
        
        **Buy Signal:** Fast EMA (50) crosses above Slow EMA (200) - Golden Cross
        **Sell Signal:** Fast EMA (50) crosses below Slow EMA (200) - Death Cross
        """,
        
        "Breakout Detection": """
        ## Breakout Detection Strategy
        
        This strategy identifies price breakouts from consolidation patterns with volume confirmation.
        
        **Buy Signal:** Price breaks above upper Bollinger Band after consolidation, with increased volume
        **Sell Signal:** Price breaks below lower Bollinger Band after consolidation, with increased volume
        """
    }
    
    return explanations.get(strategy_name, "No explanation available for this strategy.")

def apply_strategy(data, strategy_name):
    """Apply the selected strategy to generate signals"""
    if data is None or len(data) < 20:
        return None
    
    # Create a copy of data to avoid modifying the original
    df = data.copy()
    
    # Initialize signals dictionary
    signals = {}
    
    # Apply strategy based on selection
    if strategy_name == "Combined Strategy":
        # Use the combined signals from calculate_signals
        signals, _ = calculate_signals(df)
        
    elif strategy_name == "Bollinger Bands + RSI":
        # Calculate Bollinger Bands
        length = st.session_state.get('bb_length', 20)
        std_dev = st.session_state.get('bb_std', 2.0)
        rsi_length = st.session_state.get('rsi_length', 14)
        rsi_ob = st.session_state.get('rsi_ob', 70)
        rsi_os = st.session_state.get('rsi_os', 30)
        
        upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'], length, std_dev)
        signals['upper_band'] = upper_band
        signals['middle_band'] = middle_band
        signals['lower_band'] = lower_band
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], rsi_length)
        signals['rsi'] = df['rsi']
        
        # Generate buy signals (price touches lower band and RSI is oversold)
        buy_signals = (df['close'] <= lower_band * 1.01) & (df['rsi'] <= rsi_os)
        
        # Generate sell signals (price touches upper band and RSI is overbought)
        sell_signals = (df['close'] >= upper_band * 0.99) & (df['rsi'] >= rsi_ob)
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        
    elif strategy_name == "EMA Crossover":
        # Get parameters from session state or use defaults
        fast_length = st.session_state.get('ema_fast', 50)
        slow_length = st.session_state.get('ema_slow', 200)
        
        # Calculate EMAs
        df['ema_fast'] = calculate_ema(df['close'], fast_length)
        df['ema_slow'] = calculate_ema(df['close'], slow_length)
        
        # Generate signals
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)
        
        # Golden Cross (fast EMA crosses above slow EMA)
        for i in range(1, len(df)):
            if df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1] and \
               df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:
                buy_signals.iloc[i] = True
                
            # Death Cross (fast EMA crosses below slow EMA)
            if df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1] and \
               df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]:
                sell_signals.iloc[i] = True
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        signals['ema_fast'] = df['ema_fast']
        signals['ema_slow'] = df['ema_slow']
        
    elif strategy_name == "Breakout Detection":
        # Get parameters from session state or use defaults
        consol_periods = st.session_state.get('consol_periods', 14)
        bb_threshold = st.session_state.get('bb_threshold', 0.1)
        volume_increase = st.session_state.get('volume_increase', 150) / 100
        
        # Calculate Bollinger Band width
        upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'], consol_periods)
        bb_width = (upper_band - lower_band) / middle_band
        
        # Detect consolidation (tight Bollinger Bands)
        consolidation = bb_width < bb_threshold
        
        # Calculate volume moving average
        volume_ma = df['volume'].rolling(window=consol_periods).mean()
        
        # Generate buy signals (breakout from consolidation with volume increase)
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)
        
        for i in range(consol_periods, len(df)):
            # Check for consolidation followed by breakout
            if consolidation.iloc[i-1]:
                # Bullish breakout
                if df['close'].iloc[i] > upper_band.iloc[i-1] and \
                   df['volume'].iloc[i] > volume_ma.iloc[i-1] * volume_increase:
                    buy_signals.iloc[i] = True
                
                # Bearish breakout
                if df['close'].iloc[i] < lower_band.iloc[i-1] and \
                   df['volume'].iloc[i] > volume_ma.iloc[i-1] * volume_increase:
                    sell_signals.iloc[i] = True
        
        signals['upper_band'] = upper_band
        signals['middle_band'] = middle_band
        signals['lower_band'] = lower_band
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
    
    # Add a combined signal evaluation
    if 'buy_signals' in signals and 'sell_signals' in signals:
        # Check recent signals
        recent_buys = signals['buy_signals'].iloc[-5:].any()
        recent_sells = signals['sell_signals'].iloc[-5:].any()
        
        if recent_buys and not recent_sells:
            signal = "buy"
            confidence = 0.7
            reasoning = f"{strategy_name} generated a buy signal"
        elif recent_sells and not recent_buys:
            signal = "sell"
            confidence = 0.7
            reasoning = f"{strategy_name} generated a sell signal"
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning = "No clear signals detected"
            
        signals['combined_signal'] = {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    return signals

def display_strategy_performance(strategy_name, data, signals):
    """Show performance stats for the selected strategy."""
    if data is None or signals is None or len(data) < 20:
        st.warning("Insufficient data to analyze strategy performance")
        return
        
    st.subheader("Strategy Performance")
    
    # Check if we have buy/sell signals
    if 'buy_signals' not in signals or 'sell_signals' not in signals:
        st.warning("No trading signals available for this strategy.")
        return
    
    # Simulate strategy performance
    performance = simulate_strategy(data, signals)
    
    # Display performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{performance['total_return']:.2f}%")
        
    with col2:
        st.metric("Win Rate", f"{performance['win_rate']:.2f}%")
        
    with col3:
        st.metric("Profit Factor", f"{performance['profit_factor']:.2f}")
    
    # Display trade count
    st.caption(f"Based on {performance['trade_count']} simulated trades")
    
    # Display performance chart
    if not performance['equity_curve'].empty:
        st.line_chart(performance['equity_curve'])

def simulate_strategy(data, signals):
    """Simulate strategy performance based on signals with optimized calculations."""
    if 'buy_signals' not in signals or 'sell_signals' not in signals:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0,
            'equity_curve': pd.DataFrame()
        }

    df = data.copy()
    buy_signals = signals['buy_signals']
    sell_signals = signals['sell_signals']

    # Vectorized calculation of trades
    buy_prices = df['close'][buy_signals].values
    sell_prices = df['close'][sell_signals].values[:len(buy_prices)]  # Ensure matching lengths
    profits = (sell_prices - buy_prices) / buy_prices * 100

    # Calculate performance metrics
    win_trades = profits[profits > 0]
    loss_trades = profits[profits <= 0]
    win_rate = len(win_trades) / len(profits) * 100 if len(profits) > 0 else 0
    profit_factor = abs(win_trades.sum() / loss_trades.sum()) if loss_trades.sum() != 0 else float('inf')
    total_return = profits.sum()

    # Create equity curve
    equity_curve = pd.DataFrame({'equity': 100 + np.cumsum(profits)})

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trade_count': len(profits),
        'equity_curve': equity_curve
    }

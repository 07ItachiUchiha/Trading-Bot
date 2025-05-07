import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import utility functions
from utils.signal_combiner import get_combined_signals

def display_market_data(symbol='BTC/USD', interval='1h', show_combined_signals=False):
    """Display market data with technical indicators and signals"""
    st.subheader(f"{symbol} Market Data")
    
    # Create tabs for different chart types
    tab1, tab2, tab3 = st.tabs(["Price Chart", "Technical Indicators", "Signal Analysis"])
    
    with tab1:
        # Display price chart (candlestick)
        display_price_chart(symbol, interval)
    
    with tab2:
        # Display technical indicators
        display_technical_indicators(symbol, interval)
    
    with tab3:
        # Display signal analysis with combined signals
        if show_combined_signals:
            display_combined_signals(symbol)
        else:
            display_technical_signals(symbol)

def display_price_chart(symbol, interval):
    """Display price chart with candlesticks"""
    # For demo, generate sample price data
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days).tolist()
    
    # Base price and random walk
    if 'BTC' in symbol:
        base_price = 65000
    elif 'ETH' in symbol:
        base_price = 3500
    else:
        base_price = 200
    
    np.random.seed(42)  # For reproducibility
    price_changes = np.random.normal(0, base_price * 0.02, days)
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLC data
    data = []
    for i in range(days):
        open_price = prices[i] * (1 - 0.005 * np.random.random())
        close_price = prices[i]
        high_price = max(open_price, close_price) * (1 + 0.01 * np.random.random())
        low_price = min(open_price, close_price) * (1 - 0.01 * np.random.random())
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        marker={
            'color': 'rgba(100, 100, 100, 0.3)',
        },
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title=f"{symbol} Price ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_indicators(symbol, interval):
    """Display technical indicators for a symbol"""
    # Mock data for indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Momentum Indicators")
        
        # RSI
        rsi_value = 42 if 'BTC' in symbol else 58
        rsi_color = "green" if rsi_value < 50 else "red"
        st.metric("RSI (14)", f"{rsi_value}", delta="Bullish" if rsi_value < 50 else "Bearish")
        
        # MACD
        macd_value = 0.5 if 'BTC' in symbol else -0.3
        macd_signal = 0.2 if 'BTC' in symbol else -0.1
        st.metric("MACD", f"{macd_value:.2f}", delta=f"Signal: {macd_signal:.2f}")
        
        # Stochastic
        stoch_k = 65 if 'BTC' in symbol else 35
        stoch_d = 60 if 'BTC' in symbol else 40
        st.metric("Stochastic", f"K: {stoch_k}", delta=f"D: {stoch_d}")
    
    with col2:
        st.subheader("Trend Indicators")
        
        # Moving Averages
        ma_50 = 64200 if 'BTC' in symbol else 3450
        ma_200 = 61500 if 'BTC' in symbol else 3200
        st.metric("MA 50/200", f"${ma_50:,}", delta=f"{(ma_50/ma_200-1)*100:.1f}%")
        
        # Bollinger Bands
        bb_upper = 67500 if 'BTC' in symbol else 3650
        bb_lower = 63000 if 'BTC' in symbol else 3400
        current = 65000 if 'BTC' in symbol else 3500
        bb_width = (bb_upper - bb_lower) / current * 100
        st.metric("BB Width", f"{bb_width:.1f}%", delta="Expanding" if bb_width > 4 else "Contracting")
        
        # ADX
        adx_value = 28 if 'BTC' in symbol else 15
        st.metric("ADX", f"{adx_value}", delta="Strong trend" if adx_value > 25 else "Weak trend")

def display_technical_signals(symbol):
    """Display technical signals for a symbol"""
    st.subheader("Technical Signals")
    
    # Mock technical signals
    signals = {
        "MA Crossover": "Bullish" if 'BTC' in symbol else "Bearish",
        "RSI": "Oversold" if 'BTC' in symbol else "Neutral",
        "MACD": "Buy signal" if 'BTC' in symbol else "Sell signal",
        "Bollinger Bands": "Upper band test" if 'BTC' in symbol else "Lower band test",
        "Support/Resistance": "Above support" if 'BTC' in symbol else "Below resistance"
    }
    
    # Display signals
    for indicator, signal in signals.items():
        col1, col2 = st.columns([1, 3])
        col1.write(f"**{indicator}:**")
        
        # Color-code signals
        if "Bullish" in signal or "Buy" in signal or "Oversold" in signal:
            col2.markdown(f"<span style='color:green'>{signal}</span>", unsafe_allow_html=True)
        elif "Bearish" in signal or "Sell" in signal:
            col2.markdown(f"<span style='color:red'>{signal}</span>", unsafe_allow_html=True)
        else:
            col2.write(signal)
    
    # Overall signal
    st.subheader("Overall Signal")
    signal = "BUY" if 'BTC' in symbol else "SELL"
    confidence = 72 if 'BTC' in symbol else 68
    
    if signal == "BUY":
        st.markdown(f"<h3 style='color:green'>{signal} with {confidence}% confidence</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red'>{signal} with {confidence}% confidence</h3>", unsafe_allow_html=True)

def display_combined_signals(symbol):
    """Display combined technical and sentiment signals"""
    st.subheader("Combined Signal Analysis")
    
    # Get combined signals
    combined_data = get_combined_signals(symbol)
    signal_data = combined_data[symbol]
    
    # Create columns for technical, sentiment, and combined signals
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Technical")
        tech_signal = signal_data['technical_signal'].upper()
        signal_color = "green" if tech_signal in ["BUY", "STRONG BUY"] else "red" if tech_signal in ["SELL", "STRONG SELL"] else "orange"
        st.markdown(f"<h3 style='color:{signal_color}'>{tech_signal}</h3>", unsafe_allow_html=True)
        st.caption("Based on technical indicators")
        
    with col2:
        st.subheader("Sentiment")
        sent_signal = signal_data['sentiment_signal'].upper()
        signal_color = "green" if sent_signal in ["BUY", "STRONG BUY"] else "red" if sent_signal in ["SELL", "STRONG SELL"] else "orange"
        st.markdown(f"<h3 style='color:{signal_color}'>{sent_signal}</h3>", unsafe_allow_html=True)
        st.caption("Based on news & social media")
        
    with col3:
        st.subheader("Combined")
        comb_signal = signal_data['signal'].upper()
        signal_color = "green" if comb_signal in ["BUY", "STRONG BUY"] else "red" if comb_signal in ["SELL", "STRONG SELL"] else "orange"
        st.markdown(f"<h3 style='color:{signal_color}'>{comb_signal}</h3>", unsafe_allow_html=True)
        st.caption("Weighted technical & sentiment")
    
    # Display confidence meter
    st.subheader("Signal Confidence")
    confidence = signal_data['confidence']
    st.progress(confidence)
    
    # Format confidence as percentage
    conf_pct = int(confidence * 100)
    
    # Store confidence in session state for use by risk management
    st.session_state.signal_confidence = confidence
    
    # Confidence text
    if conf_pct >= 80:
        st.success(f"High confidence signal ({conf_pct}%) - Consider larger position size")
    elif conf_pct >= 60:
        st.info(f"Medium confidence signal ({conf_pct}%) - Use standard position size")
    else:
        st.warning(f"Low confidence signal ({conf_pct}%) - Use reduced position size")
        
    # Display signal explanation
    st.subheader("Signal Explanation")
    
    if comb_signal == "BUY" or comb_signal == "STRONG BUY":
        st.write("""
        This **BUY** signal is based on positive technical indicators and favorable sentiment analysis.
        
        - Technical indicators show upward momentum
        - News sentiment is positive
        - Combined analysis suggests good entry point
        """)
    elif comb_signal == "SELL" or comb_signal == "STRONG SELL":
        st.write("""
        This **SELL** signal is based on bearish technical indicators and negative sentiment analysis.
        
        - Technical indicators show downward momentum
        - News sentiment is negative
        - Combined analysis suggests reducing exposure
        """)
    else:
        st.write("""
        This **HOLD** signal indicates mixed or neutral indicators.
        
        - Technical indicators show sideways movement
        - News sentiment is neutral or conflicting
        - Wait for a stronger signal before taking action
        """)

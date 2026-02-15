import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

def calculate_position_size(account_balance, confidence_score, risk_percentage=1.0, max_alloc_percentage=5.0):
    """Figure out how much to allocate to a trade given the confidence level."""
    # Base allocation on confidence
    if confidence_score > 0.8:  # High confidence
        allocation_percentage = min(3.0 + (confidence_score - 0.8) * 10, max_alloc_percentage)
        risk_level = "High confidence"
    elif confidence_score > 0.6:  # Medium confidence
        allocation_percentage = 1.0 + (confidence_score - 0.6) * 10
        risk_level = "Medium confidence"
    else:  # Low confidence
        allocation_percentage = min(1.0, confidence_score)
        risk_level = "Low confidence"
    
    # Calculate position size
    position_size = account_balance * (allocation_percentage / 100)
    
    # Adjust based on risk percentage
    max_risk_amount = account_balance * (risk_percentage / 100)
    if position_size * 0.05 > max_risk_amount:  # If potential 5% loss exceeds max risk
        position_size = max_risk_amount / 0.05
    
    return {
        "position_size": position_size,
        "percentage": allocation_percentage,
        "risk_level": risk_level,
        "max_loss_amount": position_size * 0.05
    }

def calculate_stop_loss(entry_price, asset_volatility, direction="buy", risk_tolerance="medium"):
    """Calculate a stop loss level based on volatility and risk tolerance."""
    # Adjust stop distance based on risk tolerance
    if risk_tolerance == "low":
        multiplier = 1.5
    elif risk_tolerance == "medium":
        multiplier = 2.0
    else:  # high
        multiplier = 3.0
    
    # Calculate stop distance as percentage of price
    stop_percentage = min(asset_volatility * multiplier, 5.0)  # Cap at 5%
    
    # Calculate stop price
    if direction == "buy":
        stop_price = entry_price * (1 - stop_percentage / 100)
    else:  # sell
        stop_price = entry_price * (1 + stop_percentage / 100)
    
    return round(stop_price, 2)

def calculate_trailing_stop(current_price, highest_price, direction="buy", trail_percentage=2.0):
    """Get the current trailing stop price."""
    if direction == "buy":
        return round(highest_price * (1 - trail_percentage / 100), 2)
    else:  # sell
        return round(highest_price * (1 + trail_percentage / 100), 2)

def calculate_correlation_matrix(symbols, price_data):
    """Build a correlation matrix from price returns."""
    returns_data = {}
    
    # Calculate returns for each symbol
    for symbol in symbols:
        if symbol in price_data and len(price_data[symbol]) > 1:
            prices = price_data[symbol]['close']
            returns = np.diff(prices) / prices[:-1]
            returns_data[symbol] = returns
    
    # Create a DataFrame of returns
    if not returns_data:
        return pd.DataFrame()
        
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    return returns_df.corr()

def should_avoid_correlated_trade(symbol, portfolio_symbols, correlation_matrix, threshold=0.7):
    """Check if adding this symbol would create too much correlation in the portfolio."""
    if symbol not in correlation_matrix.columns or not portfolio_symbols:
        return False, []
    
    correlated_symbols = []
    
    for port_sym in portfolio_symbols:
        if port_sym in correlation_matrix.columns:
            corr = correlation_matrix.loc[symbol, port_sym]
            if abs(corr) >= threshold:
                correlated_symbols.append((port_sym, corr))
    
    return len(correlated_symbols) > 0, correlated_symbols

def get_asset_volatility(symbol, price_data, days=14):
    """Get recent volatility (daily std dev of returns) for a symbol."""
    if symbol not in price_data or len(price_data[symbol]) < days:
        # Default volatility estimates if data not available
        volatility_map = {
            "BTC": 4.2,
            "ETH": 5.1,
            "SOL": 7.3,
            "ADA": 6.2,
            "DOGE": 8.5,
            "AAPL": 1.8,
            "MSFT": 1.9,
            "AMZN": 2.3,
            "TSLA": 4.1
        }
        
        # Extract base symbol from trading pair
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol.replace('USD', '')
        
        # Return default volatility or generic estimate
        return volatility_map.get(base_symbol, 5.0)
    
    # Calculate daily returns
    prices = price_data[symbol]['close'][-days:]
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate standard deviation and convert to percentage
    return float(np.std(returns) * 100)

def display_risk_management_controls(symbol, account_balance=10000, confidence=0.7):
    """Display risk management controls in the UI"""
    st.subheader("Risk Management")
    
    with st.expander("Position Sizing & Risk Management", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk percentage setting
            risk_percentage = st.slider(
                "Risk per Trade (%)", 
                min_value=0.5, 
                max_value=5.0, 
                value=1.0, 
                step=0.1,
                help="Maximum percentage of portfolio to risk on this trade"
            )
            
        with col2:
            # Risk tolerance setting
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Low", "Medium", "High"],
                value="Medium",
                help="Low: tighter stops, smaller positions. High: wider stops, larger positions"
            )
        
        # Calculate position size
        position_info = calculate_position_size(
            account_balance=account_balance,
            confidence_score=confidence,
            risk_percentage=risk_percentage
        )
        
        # Display position size recommendation
        st.info(f"**Recommended Position Size:** ${position_info['position_size']:.2f} ({position_info['percentage']:.1f}% of portfolio)")
        st.caption(f"Based on {position_info['risk_level']} ({confidence*100:.0f}%) and {risk_percentage}% max risk per trade")
    
    with st.expander("Stop Loss & Take Profit", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Stop loss settings
            use_auto_stop = st.checkbox("Auto Stop-Loss", value=True, 
                                         help="Automatically calculate stop loss based on volatility")
            
            # Get current price for the symbol - placeholder for demo
            current_price = 65000.0  # Would be replaced with actual price
            
            # Volatility estimate - placeholder for demo
            volatility = 4.5  # Would be calculated based on price history
            
            if use_auto_stop:
                stop_price = calculate_stop_loss(
                    entry_price=current_price,
                    asset_volatility=volatility,
                    direction="buy",
                    risk_tolerance=risk_tolerance.lower()
                )
                st.number_input("Stop Loss Price", value=stop_price, step=1.0, format="%.2f", disabled=True)
            else:
                stop_price = st.number_input("Stop Loss Price", 
                                             value=current_price * 0.95,  # Default 5% below
                                             step=1.0, format="%.2f")
            
            stop_percentage = abs(stop_price - current_price) / current_price * 100
            st.caption(f"Stop Loss: {stop_percentage:.1f}% from entry price")
            
        with col2:
            # Take profit settings
            take_profit_multiplier = st.slider(
                "Risk-Reward Ratio", 
                min_value=1.0, 
                max_value=5.0, 
                value=2.0, 
                step=0.5,
                help="Target profit as multiple of risk (2.0 = 2:1 reward:risk)"
            )
            
            # Calculate take profit price
            take_profit_distance = abs(current_price - stop_price) * take_profit_multiplier
            take_profit_price = current_price + take_profit_distance  # For buy
            
            st.number_input("Take Profit Price", value=take_profit_price, step=1.0, format="%.2f", disabled=True)
            
            tp_percentage = abs(take_profit_price - current_price) / current_price * 100
            st.caption(f"Take Profit: {tp_percentage:.1f}% from entry price")
    
    with st.expander("Advanced Risk Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Trailing stop settings
            use_trailing_stop = st.checkbox("Use Trailing Stop", value=False, 
                                             help="Automatically adjust stop loss as price moves in your favor")
            if use_trailing_stop:
                trail_percentage = st.slider(
                    "Trailing Stop %", 
                    min_value=1.0, 
                    max_value=10.0, 
                    value=2.0, 
                    step=0.5,
                    help="Distance of trailing stop from highest price"
                )
                
            # Max loss exit settings
            use_max_loss_exit = st.checkbox("Auto-Close on Loss", value=False,
                                         help="Automatically close position if loss reaches threshold")
            if use_max_loss_exit:
                max_loss_percent = st.slider(
                    "Max Loss %", 
                    min_value=1.0, 
                    max_value=15.0, 
                    value=5.0, 
                    step=0.5,
                    help="Maximum loss percentage before automatically closing position"
                )
                
        with col2:
            # Time-based exit
            use_time_exit = st.checkbox("Time-Based Exit", value=False,
                                         help="Automatically exit position after specific time period")
            if use_time_exit:
                exit_hours = st.number_input(
                    "Exit After (hours)", 
                    min_value=1, 
                    max_value=168,  # 1 week
                    value=24,
                    help="Exit position after this many hours if target not reached"
                )
    
    # Return the risk settings
    return {
        "position_size": position_info['position_size'],
        "stop_loss": stop_price,
        "take_profit": take_profit_price,
        "use_trailing_stop": use_trailing_stop if 'use_trailing_stop' in locals() else False,
        "trailing_stop_percentage": trail_percentage if 'trail_percentage' in locals() else 2.0,
        "use_time_exit": use_time_exit if 'use_time_exit' in locals() else False,
        "exit_hours": exit_hours if 'exit_hours' in locals() else 24,
        # Include max loss exit settings
        "use_max_loss_exit": use_max_loss_exit if 'use_max_loss_exit' in locals() else False,
        "max_loss_percent": max_loss_percent if 'max_loss_percent' in locals() else 5.0
    }

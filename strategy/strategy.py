import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger('strategy')

def detect_consolidation(df, lookback_period=20, threshold=0.03):
    """
    Detect price consolidation periods
    
    Args:
        df (pd.DataFrame): Price data with OHLC columns
        lookback_period (int): Period for rolling calculations
        threshold (float): Threshold for BB width to consider a period as consolidation
        
    Returns:
        tuple: (consolidation, bb_width, atr) - boolean series for consolidation periods,
               BB width values, and ATR values
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate Bollinger Bands
        rolling_mean = df['close'].rolling(window=lookback_period).mean()
        rolling_std = df['close'].rolling(window=lookback_period).std()
        
        # Handle NaN values in rolling calculations
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        # Calculate Bollinger Bands width as a percentage of the middle band
        # Add small constant to avoid division by zero
        epsilon = 1e-9
        bb_width = ((rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)) / (rolling_mean + epsilon)
        
        # Calculate Average True Range (ATR) for additional volatility measure
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        
        # Fill NaN values from shift operation
        df['high_close'] = df['high_close'].fillna(0)
        df['low_close'] = df['low_close'].fillna(0)
        
        # Calculate True Range and ATR
        df['tr'] = np.maximum.reduce([df['high_low'], df['high_close'], df['low_close']])
        atr = df['tr'].rolling(window=lookback_period).mean().fillna(0)
        
        # Initialize consolidation series with all False values
        consolidation = pd.Series(False, index=df.index)
        
        # Consider price in consolidation when BB width is below threshold
        # Ensure we have valid data by checking for NaN and None values
        valid_index = (~pd.isna(bb_width)) & (bb_width < threshold)
        consolidation.loc[valid_index] = True
        
        # Clean up any remaining NaN values
        consolidation = consolidation.fillna(False)
        bb_width = bb_width.fillna(0)
        atr = atr.fillna(0)
        
        return consolidation, bb_width, atr
        
    except Exception as e:
        logger.error(f"Error in detect_consolidation: {e}")
        # Return fallback values if function fails
        default_values = pd.Series(False, index=df.index)
        return default_values, default_values.astype(float), default_values.astype(float)

def detect_breakout(df, consolidation, bb_width, lookback=20):
    """
    Detect breakouts from consolidation periods
    
    Args:
        df (pd.DataFrame): Price data with OHLC columns
        consolidation (pd.Series): Boolean series indicating consolidation periods
        bb_width (pd.Series): Bollinger Band width values
        lookback (int): Period for looking back for consolidation
        
    Returns:
        tuple: (buy_signals, sell_signals) - Boolean series for buy and sell signals
    """
    try:
        # Validate inputs
        if df is None or consolidation is None or bb_width is None:
            logger.warning("Invalid inputs to detect_breakout")
            # Return empty signals series with same index as df
            empty_signals = pd.Series(False, index=df.index)
            return empty_signals, empty_signals
            
        # Initialize signal series with False values
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)
        
        # Ensure we have enough data
        if len(df) <= lookback:
            return buy_signals, sell_signals
        
        # Find periods where price was in consolidation and now breaking out
        for i in range(lookback, len(df)):
            # Check if there's any consolidation in the lookback period
            consolidation_window = consolidation.iloc[i-lookback:i]
            if consolidation_window.any():
                # Safety check for NaN or None values
                current_bb = bb_width.iloc[i] if not pd.isna(bb_width.iloc[i]) else 0
                prev_bb = bb_width.iloc[i-1] if not pd.isna(bb_width.iloc[i-1]) else 0
                
                # Check for breakout with increasing BB width (volatility expansion)
                if current_bb > 0 and prev_bb > 0 and current_bb > prev_bb * 1.5:
                    # Determine breakout direction using price movement
                    current_close = df['close'].iloc[i]
                    prev_close = df['close'].iloc[i-1]
                    
                    # Ensure we have valid prices
                    if pd.isna(current_close) or pd.isna(prev_close):
                        continue
                        
                    if current_close > prev_close:
                        buy_signals.iloc[i] = True
                    else:
                        sell_signals.iloc[i] = True
        
        return buy_signals, sell_signals
        
    except Exception as e:
        logger.error(f"Error in detect_breakout: {e}")
        # Return fallback values if function fails
        empty_signals = pd.Series(False, index=df.index)
        return empty_signals, empty_signals

def calculate_targets_and_stop(entry_price, direction, atr_value, risk_reward_ratios=[1.5, 2.5, 4.0]):
    """
    Calculate stop loss and take profit targets based on ATR
    
    Args:
        entry_price (float): Entry price
        direction (str): 'long' or 'short'
        atr_value (float): ATR value
        risk_reward_ratios (list): List of risk-reward ratios for targets
        
    Returns:
        tuple: (stop_loss, targets)
    """
    # Calculate stop distance (1 ATR)
    stop_distance = atr_value
    
    if direction == 'long':
        stop_loss = entry_price - stop_distance
        targets = [entry_price + (stop_distance * rr) for rr in risk_reward_ratios]
    else:  # short
        stop_loss = entry_price + stop_distance
        targets = [entry_price - (stop_distance * rr) for rr in risk_reward_ratios]
    
    return stop_loss, targets

def check_entry(df):
    """
    Check for entry signals based on multiple technical indicators
    
    Args:
        df (pd.DataFrame): Price data with technical indicators
        
    Returns:
        tuple: (signal, confidence) where signal is 'BUY', 'SELL', or 'NEUTRAL'
    """
    signal = 'NEUTRAL'
    confidence = 0.0
    
    # Get the last row for analysis
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Initialize signal scoring
    buy_signals = 0
    sell_signals = 0
    total_signals = 0
    
    # Check Bollinger Bands
    if 'upper_band' in df.columns and 'lower_band' in df.columns:
        # Ensure values are not None
        if (current['upper_band'] is not None and current['lower_band'] is not None and
            previous['upper_band'] is not None and previous['lower_band'] is not None):
            # Bollinger Band signals
            if current['close'] > current['upper_band'] and previous['close'] <= previous['upper_band']:
                buy_signals += 1
            elif current['close'] < current['lower_band'] and previous['close'] >= previous['lower_band']:
                sell_signals += 1
            total_signals += 1
    
    # Check RSI
    if 'rsi' in df.columns and current['rsi'] is not None and previous['rsi'] is not None:
        # RSI signals
        if current['rsi'] < 30 and previous['rsi'] >= 30:  # Oversold
            buy_signals += 1
        elif current['rsi'] > 70 and previous['rsi'] <= 70:  # Overbought
            sell_signals += 1
        total_signals += 1
    
    # Check moving averages
    if 'sma20' in df.columns and current['sma20'] is not None and previous['sma20'] is not None:
        # Moving average crossovers
        if current['close'] > current['sma20'] and previous['close'] <= previous['sma20']:
            buy_signals += 1
        elif current['close'] < current['sma20'] and previous['close'] >= previous['sma20']:
            sell_signals += 1
        
        # Check for ema indicators
        if 'ema50' in df.columns and 'ema200' in df.columns:
            # Check trend direction
            if (current['ema50'] is not None and current['ema200'] is not None):
                if current['ema50'] > current['ema200']:
                    buy_signals += 0.5
                elif current['ema50'] < current['ema200']:
                    sell_signals += 0.5
        
        total_signals += 1
    
    # Check for consolidation breakouts if we have atr
    if 'atr' in df.columns and current['atr'] is not None:
        # Simple volatility breakout
        price_change = abs(current['close'] - previous['close'])
        if price_change > current['atr'] * 1.5:  # significant move
            if current['close'] > previous['close']:
                buy_signals += 1
            else:
                sell_signals += 1
        total_signals += 1
    
    # Determine signal direction
    if total_signals > 0:
        buy_confidence = buy_signals / total_signals
        sell_confidence = sell_signals / total_signals
        
        if buy_confidence > 0.5 and buy_confidence > sell_confidence:
            signal = 'BUY'
            confidence = buy_confidence
        elif sell_confidence > 0.5 and sell_confidence > buy_confidence:
            signal = 'SELL'
            confidence = sell_confidence
    
    return signal, confidence
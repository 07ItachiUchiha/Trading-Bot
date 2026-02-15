import pandas as pd
import numpy as np
from datetime import datetime

def append_to_dataframe(df, new_data):
    """Append data to a DataFrame using concat (since .append is deprecated)."""
    # Convert dict to DataFrame if needed
    if isinstance(new_data, dict):
        new_row = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        new_row = new_data
    else:
        raise ValueError("new_data must be a dictionary or DataFrame")
    
    # Use concat instead of deprecated append
    return pd.concat([df, new_row], ignore_index=True)

def update_ohlc_candle(df, new_candle, match_column='time'):
    """Update an existing candle in the df or append if it's new."""
    if df.empty:
        return pd.DataFrame([new_candle])
    
    # make sure we can actually match
    if match_column not in df.columns or match_column not in new_candle:
        # Just append if we can't match
        return append_to_dataframe(df, new_candle)
    
    # Convert to comparable format if timestamps
    if isinstance(new_candle[match_column], str):
        new_time = pd.to_datetime(new_candle[match_column])
    else:
        new_time = new_candle[match_column]
        
    # Find matching row
    match_idx = df[df[match_column] == new_time].index
    
    if len(match_idx) > 0:
        # Update existing candle
        idx = match_idx[0]
        
        # Update high/low
        if 'high' in new_candle and 'high' in df.columns:
            df.loc[idx, 'high'] = max(df.loc[idx, 'high'], new_candle['high'])
        
        if 'low' in new_candle and 'low' in df.columns:
            df.loc[idx, 'low'] = min(df.loc[idx, 'low'], new_candle['low'])
            
        # Update close price and volume 
        if 'close' in new_candle and 'close' in df.columns:
            df.loc[idx, 'close'] = new_candle['close']
            
        if 'volume' in new_candle and 'volume' in df.columns:
            df.loc[idx, 'volume'] = new_candle['volume']
            
        return df
    else:
        # Add as new candle
        return append_to_dataframe(df, new_candle)

def clean_dataframe(df):
    """Handle NaN values and ensure correct types in OHLCV data."""
    # Make a copy
    result = df.copy()
    
    # Convert timestamp if needed
    if 'time' in result.columns and not pd.api.types.is_datetime64_any_dtype(result['time']):
        try:
            result['time'] = pd.to_datetime(result['time'])
        except:
            pass
            
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in result.columns:
            if col == 'volume':
                result[col] = result[col].fillna(0)
            else:
                # ffill then bfill for price data
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Ensure numeric type
            result[col] = pd.to_numeric(result[col], errors='coerce')
    
    return result

def validate_ohlc_data(data):
    """Check OHLC data for basic correctness (missing cols, negative prices, outliers)."""
    issues = []
    is_valid = True
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        is_valid = False
    
    if is_valid:
        # Check for logical consistency: high >= max(open, close) and low <= min(open, close)
        for idx, row in data.iterrows():
            if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                continue  # Skip rows with NaN values
                
            if row['high'] < max(row['open'], row['close']):
                issues.append(f"Row {idx}: High price is less than max(open, close)")
                is_valid = False
                
            if row['low'] > min(row['open'], row['close']):
                issues.append(f"Row {idx}: Low price is greater than min(open, close)")
                is_valid = False
        
        # Check for negative prices
        for col in required_columns:
            if (data[col] < 0).any():
                issues.append(f"Column {col} contains negative values")
                is_valid = False
                
        # Check for extreme outliers (1000%+ move between candles)
        if len(data) > 1:
            for col in required_columns:
                price_changes = data[col].pct_change().abs()
                extreme_changes = price_changes > 10  # 1000% change
                if extreme_changes.any():
                    extreme_indices = data.index[extreme_changes].tolist()
                    issues.append(f"Column {col} has extreme price changes at indices: {extreme_indices}")
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'summary': f"{'Valid' if is_valid else 'Invalid'} OHLC data with {len(issues)} issues"
    }

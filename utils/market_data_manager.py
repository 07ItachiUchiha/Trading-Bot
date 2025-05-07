import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
import time
import alpaca_trade_api as tradeapi

class MarketDataManager:
    """
    Manages market data including fetching, caching, and preprocessing.
    Supports multiple data sources and timeframes.
    """
    
    def __init__(self, api_keys=None, config=None):
        """
        Initialize the market data manager with API keys and configuration
        
        Args:
            api_keys (dict): API keys for different data sources
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'data_source': 'alpaca',      # Default data source
            'default_timeframe': '1h',    # Default timeframe
            'cache_duration': {
                '1m': 30,                 # Cache 1-minute data for 30 minutes
                '5m': 60,                 # Cache 5-minute data for 1 hour
                '15m': 120,               # Cache 15-minute data for 2 hours
                '1h': 480,                # Cache 1-hour data for 8 hours
                '1d': 1440                # Cache daily data for 24 hours
            },
            'max_bars': {
                '1m': 1000,
                '5m': 1000,
                '15m': 1000,
                '1h': 750,
                '1d': 500
            },
            'timeframe_minutes': {
                '1m': 1, 
                '3m': 3, 
                '5m': 5, 
                '15m': 15, 
                '30m': 30,
                '1h': 60, 
                '2h': 120, 
                '4h': 240, 
                '1d': 1440
            },
            'cache_dir': 'cache/market_data'
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # API keys for different services
        self.api_keys = {
            'alpaca': {
                'key': None,
                'secret': None
            },
            'polygon': None,
            'binance': None
        }
        
        if api_keys:
            if 'alpaca_key' in api_keys and 'alpaca_secret' in api_keys:
                self.api_keys['alpaca']['key'] = api_keys['alpaca_key']
                self.api_keys['alpaca']['secret'] = api_keys['alpaca_secret']
            if 'polygon' in api_keys:
                self.api_keys['polygon'] = api_keys['polygon']
            if 'binance' in api_keys:
                self.api_keys['binance'] = api_keys['binance']
        
        # Initialize Alpaca API
        if self.api_keys['alpaca']['key'] and self.api_keys['alpaca']['secret']:
            self.alpaca = tradeapi.REST(
                self.api_keys['alpaca']['key'],
                self.api_keys['alpaca']['secret'],
                base_url='https://paper-api.alpaca.markets'
            )
        else:
            self.alpaca = None
        
        # Initialize data cache
        self.data_cache = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('MarketDataManager')
    
    def get_historical_data(self, symbol, timeframe=None, limit=None):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data (e.g., '1h', '1d')
            limit (int): Maximum number of bars to fetch
            
        Returns:
            pd.DataFrame: Historical price data
        """
        # Use default timeframe if none provided
        timeframe = timeframe or self.config['default_timeframe']
        
        # Use default limit if none provided
        limit = limit or self.config['max_bars'].get(timeframe, 500)
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            cache_age_minutes = (datetime.now() - cached_data['timestamp']).seconds / 60
            
            if cache_age_minutes < self.config['cache_duration'].get(timeframe, 60):
                data = cached_data['data']
                
                # If we need more data than cached, refetch
                if limit and len(data) < limit:
                    self.logger.info(f"Cached data for {symbol} {timeframe} has fewer bars ({len(data)}) than requested ({limit}). Refetching.")
                else:
                    # Return the most recent bars up to the limit
                    if limit and len(data) > limit:
                        return data.iloc[-limit:]
                    return data
        
        # Fetch data from source
        if self.config['data_source'] == 'alpaca':
            data = self._fetch_from_alpaca(symbol, timeframe, limit)
        else:
            self.logger.error(f"Unsupported data source: {self.config['data_source']}")
            return None
        
        # Cache the data
        if data is not None:
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            # Also save to disk cache
            self._save_to_disk_cache(symbol, timeframe, data)
        
        return data
    
    def _fetch_from_alpaca(self, symbol, timeframe, limit):
        """
        Fetch historical data from Alpaca
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            limit (int): Maximum number of bars
            
        Returns:
            pd.DataFrame: Historical price data
        """
        if self.alpaca is None:
            self.logger.error("Alpaca API not initialized")
            return None
        
        try:
            # Map to Alpaca timeframe format
            timeframe_map = {
                '1m': '1Min',
                '3m': '3Min',
                '5m': '5Min',
                '15m': '15Min',
                '30m': '30Min',
                '1h': '1Hour',
                '2h': '2Hour',
                '4h': '4Hour',
                '1d': '1Day'
            }
            
            alpaca_timeframe = timeframe_map.get(timeframe, '1Hour')
            
            # Calculate start and end times based on limit and timeframe
            end_date = datetime.now()
            minutes = self.config['timeframe_minutes'].get(timeframe, 60) * limit
            start_date = end_date - timedelta(minutes=minutes)
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Determine if symbol is crypto
            is_crypto = 'USD' in symbol and not symbol.endswith('USD=X')
            
            if is_crypto:
                # Get crypto data
                bars = self.alpaca.get_crypto_bars(
                    symbol=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_str,
                    end=end_str
                ).df
            else:
                # Get stock data
                bars = self.alpaca.get_bars(
                    symbol=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_str,
                    end=end_str
                ).df
            
            # Check if we got any data
            if bars is None or len(bars) == 0:
                self.logger.warning(f"No data returned from Alpaca for {symbol} {timeframe}")
                return None
            
            # Format the data
            df = pd.DataFrame({
                'time': bars.index,
                'open': bars['open'],
                'high': bars['high'],
                'low': bars['low'],
                'close': bars['close'],
                'volume': bars['volume']
            })
            
            # Ensure time column is datetime, reindex and sort
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            df.reset_index(drop=True, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Alpaca for {symbol} {timeframe}: {e}")
            
            # Try to load from disk cache as fallback
            return self._load_from_disk_cache(symbol, timeframe)
    
    def _save_to_disk_cache(self, symbol, timeframe, data):
        """
        Save market data to disk cache
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            data (pd.DataFrame): Market data
        """
        try:
            # Create cache file path
            cache_file = os.path.join(
                self.config['cache_dir'], 
                f"{symbol}_{timeframe}.csv"
            )
            
            # Save to CSV
            data.to_csv(cache_file, index=False)
        except Exception as e:
            self.logger.error(f"Error saving to disk cache: {e}")
    
    def _load_from_disk_cache(self, symbol, timeframe):
        """
        Load market data from disk cache
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            
        Returns:
            pd.DataFrame: Market data or None
        """
        try:
            # Create cache file path
            cache_file = os.path.join(
                self.config['cache_dir'], 
                f"{symbol}_{timeframe}.csv"
            )
            
            # Check if file exists
            if not os.path.exists(cache_file):
                return None
            
            # Load CSV
            data = pd.read_csv(cache_file)
            
            # Ensure time column is datetime
            data['time'] = pd.to_datetime(data['time'])
            
            return data
        except Exception as e:
            self.logger.error(f"Error loading from disk cache: {e}")
            return None
    
    def update_live_data(self, symbol, timeframe=None, latest_bar=None):
        """
        Update cached data with latest bar
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            latest_bar (dict): Latest bar data
            
        Returns:
            pd.DataFrame: Updated data
        """
        if latest_bar is None:
            return None
            
        # Use default timeframe if none provided
        timeframe = timeframe or self.config['default_timeframe']
        
        # Get cached data
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self.data_cache:
            # Fetch historical data first
            self.get_historical_data(symbol, timeframe)
            
            if cache_key not in self.data_cache:
                return None
        
        # Get cached data
        cached_data = self.data_cache[cache_key]['data'].copy()
        
        # Create new bar DataFrame
        new_bar = pd.DataFrame({
            'time': [pd.to_datetime(latest_bar.get('time', datetime.now()))],
            'open': [float(latest_bar.get('open', 0))],
            'high': [float(latest_bar.get('high', 0))],
            'low': [float(latest_bar.get('low', 0))],
            'close': [float(latest_bar.get('close', 0))],
            'volume': [float(latest_bar.get('volume', 0))]
        })
        
        # Check if we already have this timestamp
        if new_bar['time'].iloc[0] in cached_data['time'].values:
            # Update the existing bar
            idx = cached_data['time'] == new_bar['time'].iloc[0]
            cached_data.loc[idx, 'open'] = new_bar['open'].iloc[0]
            cached_data.loc[idx, 'high'] = max(cached_data.loc[idx, 'high'].iloc[0], new_bar['high'].iloc[0])
            cached_data.loc[idx, 'low'] = min(cached_data.loc[idx, 'low'].iloc[0], new_bar['low'].iloc[0])
            cached_data.loc[idx, 'close'] = new_bar['close'].iloc[0]
            cached_data.loc[idx, 'volume'] += new_bar['volume'].iloc[0]
        else:
            # Append the new bar
            cached_data = pd.concat([cached_data, new_bar], ignore_index=True)
        
        # Limit the size of the dataframe
        max_bars = self.config['max_bars'].get(timeframe, 500)
        if len(cached_data) > max_bars:
            cached_data = cached_data.iloc[-max_bars:]
        
        # Update cache
        self.data_cache[cache_key] = {
            'data': cached_data,
            'timestamp': datetime.now()
        }
        
        return cached_data
    
    def get_current_price(self, symbol):
        """
        Get current price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price or None
        """
        try:
            if self.alpaca is not None:
                # Determine if symbol is crypto
                is_crypto = 'USD' in symbol and not symbol.endswith('USD=X')
                
                if is_crypto:
                    # Get crypto price
                    last_trade = self.alpaca.get_latest_crypto_trade(symbol)
                    return last_trade.p
                else:
                    # Get stock price
                    last_trade = self.alpaca.get_latest_trade(symbol)
                    return last_trade.p
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """
        Calculate common technical indicators for a price dataframe
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        if data is None or len(data) < 20:
            return data
            
        df = data.copy()
        
        # Add basic moving averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['middle_band'] + (df['std'] * 2)
        df['lower_band'] = df['middle_band'] - (df['std'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Calculate MACD
        df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate volume indicators
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df

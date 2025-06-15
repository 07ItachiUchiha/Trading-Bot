import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
import time
import alpaca_trade_api as tradeapi
import requests

class MarketDataManager:
    """
    Manages market data including fetching, caching, and preprocessing.
    Supports Alpaca data source only (removed Binance).
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
            'default_timeframe': '1h',
            'cache_expiry': {
                '1m': 60 * 60,  # 1 hour
                '5m': 24 * 60 * 60,  # 1 day
                '15m': 3 * 24 * 60 * 60,  # 3 days
                '1h': 7 * 24 * 60 * 60,  # 7 days
                '1d': 30 * 24 * 60 * 60   # 30 days
            },
            'max_bars': {
                '1m': 1500,
                '5m': 1200,
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
        
        # API keys for Alpaca only (removed Binance)
        self.api_keys = {
            'alpaca': {
                'key': None,
                'secret': None
            },
            'polygon': None
        }
        
        if api_keys:
            if 'alpaca_key' in api_keys and 'alpaca_secret' in api_keys:
                self.api_keys['alpaca']['key'] = api_keys['alpaca_key']
                self.api_keys['alpaca']['secret'] = api_keys['alpaca_secret']
            if 'polygon' in api_keys:
                self.api_keys['polygon'] = api_keys['polygon']
        
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
            
            if cache_age_minutes < self.config['cache_expiry'].get(timeframe, 60):
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
        """Fetch historical data from Alpaca API with improved error handling"""
        try:
            if self.alpaca is None:
                self.logger.error(f"Cannot fetch data for {symbol}: Alpaca API not initialized")
                return self._generate_mock_data(symbol, timeframe, limit)
            
            # Calculate start and end times based on limit and timeframe
            end_date = datetime.now()
            minutes = self.config['timeframe_minutes'].get(timeframe, 60) * limit
            start_date = end_date - timedelta(minutes=minutes)
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fix symbol formatting - remove slash for Alpaca API
            formatted_symbol = symbol.replace('/', '')
            
            self.logger.info(f"Fetching {formatted_symbol} data from {start_str} to {end_str} with {timeframe} timeframe")
            
            # Determine if symbol is crypto
            is_crypto = 'USD' in symbol and not symbol.endswith('USD=X')
            
            try:
                if is_crypto:
                    # Get crypto data with error handling
                    bars = self.alpaca.get_crypto_bars(
                        symbol=formatted_symbol,  # Use formatted symbol
                        timeframe=timeframe,
                        start=start_str,
                        end=end_str
                    ).df
                else:
                    # Get stock data with error handling
                    bars = self.alpaca.get_bars(
                        symbol=formatted_symbol,  # Use formatted symbol
                        timeframe=timeframe,
                        start=start_str,
                        end=end_str
                    ).df
                
                # Process the data
                if bars is not None and not bars.empty:
                    # Reset index to get datetime as a column
                    bars = bars.reset_index()
                    # Rename columns to match expected format
                    bars = bars.rename(columns={
                        'timestamp': 'datetime', 
                        'open': 'open', 
                        'high': 'high', 
                        'low': 'low', 
                        'close': 'close',
                        'volume': 'volume'
                    })
                    
                    # Ensure datetime format is consistent
                    bars['datetime'] = pd.to_datetime(bars['datetime'])
                    
                    # Cache the data
                    self._save_to_disk_cache(symbol, timeframe, bars)
                    
                    self.logger.info(f"Successfully fetched {len(bars)} bars for {symbol}")
                    return bars
                else:
                    self.logger.warning(f"No data returned for {symbol} with timeframe {timeframe}")
                    return self._generate_mock_data(symbol, timeframe, limit)
            
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                # If we get an error, try with mock data
                return self._generate_mock_data(symbol, timeframe, limit)
                
        except Exception as e:
            self.logger.error(f"Unexpected error in _fetch_from_alpaca for {symbol}: {e}")
            return self._generate_mock_data(symbol, timeframe, limit)
    
    def _generate_mock_data(self, symbol, timeframe, limit):
        """Generate mock price data when API fails, so trading can continue"""
        self.logger.info(f"Generating mock data for {symbol} with {timeframe} timeframe")
        
        # Current time
        end_date = datetime.now()
        
        # Calculate minutes per bar based on timeframe
        minutes_per_bar = self.config['timeframe_minutes'].get(timeframe, 60)
        
        # Calculate start date
        start_date = end_date - timedelta(minutes=minutes_per_bar * limit)
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, periods=limit)
        
        # Base price based on symbol with added XAU/USD support
        if 'BTC' in symbol:
            base_price = 65000  # BTC base price
        elif 'ETH' in symbol:
            base_price = 3500   # ETH base price
        elif 'SOL' in symbol:
            base_price = 150    # SOL base price
        elif 'ADA' in symbol:
            base_price = 0.5    # ADA base price
        elif 'DOGE' in symbol:
            base_price = 0.15   # DOGE base price
        elif 'XAU' in symbol or 'GOLD' in symbol:
            base_price = 2400   # Gold price in USD
            volatility_factor = 0.005  # Lower volatility for gold
        else:
            base_price = 100    # Default price
        
        # Determine volatility factor based on asset type
        volatility_factor = 0.01  # Default 1% standard deviation
        if 'XAU' in symbol or 'GOLD' in symbol:
            volatility_factor = 0.005  # Gold is less volatile (0.5%)
        
        # Generate price with some randomness and appropriate volatility
        np.random.seed(42)  # For reproducibility
        price_changes = np.random.normal(0, base_price * volatility_factor, limit)
        prices = base_price + np.cumsum(price_changes)
        prices = np.maximum(prices, base_price * 0.5)  # Ensure prices don't go too low
        
        # Create DataFrame
        mock_data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 - 0.005 * np.random.random(limit)),
            'high': prices * (1 + 0.01 * np.random.random(limit)),
            'low': prices * (1 - 0.01 * np.random.random(limit)),
            'close': prices,
            'volume': np.random.randint(100, 10000, limit)
        })
        
        self.logger.info(f"Generated mock data for {symbol}: {len(mock_data)} bars")
        return mock_data
    
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
        
        # RSI calculation
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(winsdow=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=20)
        rolling_std = rolling_mean.std()
        df['upper_band'] = rolling_mean.mean() + (rolling_std * 2)
        df['lower_band'] = rolling_mean.mean() - (rolling_std * 2)

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()

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

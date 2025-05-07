import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import logging
import os
import threading
import traceback
import alpaca_trade_api as tradeapi

# Ensure numpy.NaN is available
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

from strategy.strategy import check_entry
from strategy.news_strategy import NewsBasedStrategy
from strategy.earnings_report_strategy import EarningsReportStrategy
from utils.risk_management import calculate_position_size, manage_open_position
from utils.telegram_alert import send_alert
from utils.news_fetcher import NewsFetcher
from config import (API_KEY, API_SECRET, CAPITAL, RISK_PERCENT, MAX_CAPITAL_PER_TRADE, 
                   DEFAULT_SYMBOLS, PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT,
                   SENTIMENT_REFRESH_INTERVAL)

class AutoTradingManager:
    """Manager for automated trading across multiple symbols"""
    
    def __init__(self, symbols, timeframe, capital=10000, risk_percent=1.0, profit_target_percent=3.0, 
                 daily_profit_target=5.0, use_news=True, news_weight=0.5, use_earnings=True, 
                 earnings_weight=0.6):
        """Initialize the AutoTradingManager"""
        self.logger = logging.getLogger('auto_trading_manager')
        
        # Format symbols correctly
        self.symbols = []
        for symbol in symbols:
            # For Alpaca, use "BTC/USD" format for crypto
            if '/' not in symbol and symbol.endswith('USD'):
                formatted_symbol = f"{symbol[:-3]}/USD"
                self.symbols.append(formatted_symbol)
            else:
                self.symbols.append(symbol)
        
        # Set up API with more robust error handling
        try:
            # Try paper trading URL first
            self.api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
            try:
                # Validate API key with a simple account query
                self.api.get_account()
                self.logger.info("Successfully connected to Alpaca API")
                self.using_binance = False
            except Exception as e:
                self.logger.error(f"Failed to connect to Alpaca API: {e}")
                # Try the live API URL if paper trading fails
                try:
                    self.api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://api.alpaca.markets')
                    self.api.get_account()
                    self.logger.info("Successfully connected to Alpaca Live API")
                    self.using_binance = False
                except:
                    # Try to use Binance if Alpaca fails
                    try:
                        from binance.client import Client
                        self.binance_client = Client()
                        self.logger.info("Using Binance as fallback for data")
                        self.using_binance = True
                    except:
                        self.logger.warning("Unable to connect to Binance either")
                        self.using_binance = False
        except Exception as e:
            self.logger.error(f"Error initializing trading API: {e}")
            self.using_binance = False
        
        self.timeframe = timeframe
        self.capital = capital
        self.risk_percent = risk_percent
        self.profit_target_percent = profit_target_percent
        self.daily_profit_target = daily_profit_target
        
        # Strategy weights
        self.use_news = use_news
        self.news_weight = news_weight
        self.use_earnings = use_earnings
        self.earnings_weight = earnings_weight
        
        # Trading state
        self.running = False
        self.dataframes = {}
        self.active_trades = {}
        self.daily_pnl = 0
        self.total_pnl = 0
        self.last_signal = {}
        self.symbols_trading_halted = {}
        
        # Initialize API client using config values
        self.client = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize strategies
        self.news_strategy = NewsBasedStrategy() if use_news else None
        self.earnings_strategy = EarningsReportStrategy() if use_earnings else None
        
        self.logger.info(f"Auto trading manager initialized with {len(self.symbols)} symbols")
    
    def setup_logging(self):
        """Setup the logger"""
        self.logger = logging.getLogger("auto_trading_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create directory for logs if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(f"logs/auto_trading_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def run(self):
        """Run the auto trading manager"""
        self.running = True
        self.logger.info("Starting auto trading manager")
        
        # Load initial historical data
        for symbol in self.symbols:
            self._load_historical_data(symbol)
        
        # Main loop
        while self.running:
            try:
                for symbol in self.symbols:
                    if self.symbols_trading_halted.get(symbol, False):
                        self.logger.warning(f"Trading for {symbol} is halted, skipping")
                        continue
                    
                    # Update candles
                    self._update_candles(symbol)
                    
                    # Calculate indicators
                    self._calculate_indicators(symbol)
                    
                    # Update sentiment if applicable
                    if self.use_news:
                        self._update_sentiment(symbol)
                    
                    # Process earnings signals if applicable
                    if self.use_earnings:
                        self._process_earnings_signals(symbol)
                    
                    # Generate trading signals
                    self._analyze_symbol(symbol)
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in auto trading manager: {e}")
                traceback.print_exc()
                time.sleep(60)  # Wait before trying again
        
        self.logger.info("Auto trading manager stopped")
    
    def stop(self):
        """Stop the auto trading manager"""
        self.running = False
        self.logger.info("Stopping auto trading manager")
    
    def _convert_timeframe(self, timeframe):
        """Convert user-friendly timeframe to Alpaca format"""
        # Already mapped in start_auto_trader function
        return timeframe
    
    def _load_historical_data(self, symbol):
        """Load historical data for a specific symbol"""
        try:
            end = datetime.now()
            start = end - timedelta(days=7)  # Get 7 days of data
            
            # Convert crypto symbol format for Alpaca API
            if symbol.endswith('USD') and '/' not in symbol:
                base = symbol[:-3]
                quote = symbol[-3:]
                alpaca_symbol = f"{base}/{quote}"
            else:
                alpaca_symbol = symbol
            
            self.logger.info(f"Requesting historical data for {alpaca_symbol} with {self.timeframe}")
            
            bars = self.client.get_crypto_bars(
                symbol=alpaca_symbol,
                timeframe=self.timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d')
            ).df
            
            # Convert to pandas DataFrame with expected format
            df = pd.DataFrame({
                'open': bars['open'],
                'high': bars['high'],
                'low': bars['low'],
                'close': bars['close'],
                'volume': bars['volume']
            })
            df.index.name = 'timestamp'
            
            self.dataframes[symbol] = df
            
            # Initialize trading state for this symbol
            if symbol not in self.symbols_trading_halted:
                self.symbols_trading_halted[symbol] = False
            
            self.logger.info(f"Loaded {len(df)} candles of historical data for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            self.symbols_trading_halted[symbol] = True
            return False
    
    def _update_candles(self, symbol):
        """Update candle data for a specific symbol"""
        try:
            if symbol not in self.dataframes:
                return self._load_historical_data(symbol)
            
            # Get the latest candle from Alpaca
            end = datetime.now()
            start = end - timedelta(minutes=60)  # Get the most recent hour
            
            latest_bars = self.client.get_crypto_bars(
                symbol=symbol,
                timeframe=self.timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d')
            ).df
            
            if len(latest_bars) > 0:
                # Convert to expected format
                latest_df = pd.DataFrame({
                    'open': latest_bars['open'],
                    'high': latest_bars['high'],
                    'low': latest_bars['low'],
                    'close': latest_bars['close'],
                    'volume': latest_bars['volume']
                })
                latest_df.index.name = 'timestamp'
                
                # Merge with existing dataframe
                if self.dataframes[symbol] is not None:
                    # Remove old data to keep dataframe size manageable
                    if len(self.dataframes[symbol]) > 1000:
                        self.dataframes[symbol] = self.dataframes[symbol].iloc[-500:]
                    
                    # Append new data
                    self.dataframes[symbol] = pd.concat([self.dataframes[symbol], latest_df])
                    self.dataframes[symbol] = self.dataframes[symbol][~self.dataframes[symbol].index.duplicated(keep='last')]
                else:
                    self.dataframes[symbol] = latest_df
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating candles for {symbol}: {e}")
            return False
    
    def _calculate_indicators(self, symbol):
        """Calculate technical indicators for a given symbol"""
        # Implementation depends on what indicators you want to use
        # This is a simplified version
        df = self.dataframes.get(symbol)
        if df is None or len(df) < 20:
            return
        
        # Calculate basic indicators
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['middle_band'] + (df['std'] * 2)
        df['lower_band'] = df['middle_band'] - (df['std'] * 2)
        
        self.dataframes[symbol] = df

    def get_historical_data(self, symbol, timeframe, days_back=30):
        """Get historical price data with retry and error handling"""
        self.logger.info(f"Requesting historical data for {symbol} with {timeframe}")
        
        # Try multiple times with exponential backoff
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if hasattr(self, 'using_binance') and self.using_binance:
                    # Use Binance for data
                    return self._get_binance_data(symbol, timeframe, days_back)
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Format dates for API request
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Ensure symbol is formatted correctly for API request
                api_symbol = symbol.replace('/', '')
                
                # Get historical data from Alpaca
                bars = self.api.get_crypto_bars(
                    api_symbol, 
                    timeframe,
                    start=start_str,
                    end=end_str
                ).df
                
                if bars.empty:
                    self.logger.warning(f"No data returned for {symbol}")
                    return None
                    
                # Format the dataframe for our use
                bars = bars.reset_index()
                bars = bars.rename(columns={
                    'timestamp': 'time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low', 
                    'close': 'close',
                    'volume': 'volume'
                })
                return bars
                
            except Exception as e:
                self.logger.error(f"Error loading historical data for {symbol.replace('/', '')}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Generate synthetic data as last resort
                    self.logger.info(f"Using synthetic data for {symbol}")
                    return self._generate_synthetic_data(symbol)
        
        return None

    def _generate_synthetic_data(self, symbol):
        """Generate synthetic data when API fails"""
        self.logger.info(f"Generating synthetic price data for {symbol}")
        
        # Create a date range for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Base price depends on the symbol
        if 'BTC' in symbol:
            base_price = 30000
        elif 'ETH' in symbol:
            base_price = 2000
        elif 'BNB' in symbol:
            base_price = 500
        elif 'ADA' in symbol:
            base_price = 0.6
        elif 'SOL' in symbol:
            base_price = 140
        elif 'DOGE' in symbol:
            base_price = 0.15
        else:
            base_price = 100
        
        # Generate realistic price data
        close_prices = [base_price]
        for i in range(1, len(dates)):
            # Add some randomness, momentum, and mean reversion
            momentum = 0.2
            mean_reversion = 0.1
            volatility = base_price * 0.01
            
            price_change = np.random.normal(0, volatility)
            if i > 1:
                prev_change = close_prices[-1] - close_prices[-2]
                price_change += momentum * prev_change
            price_change -= mean_reversion * (close_prices[-1] - base_price)
            
            close_prices.append(close_prices[-1] + price_change)
        
        # Create dataframe with OHLC data
        df = pd.DataFrame({
            'time': dates,
            'close': close_prices
        })
        
        # Generate open/high/low from close
        df['open'] = df['close'].shift(1)
        df.loc[0, 'open'] = df.loc[0, 'close'] * (1 + np.random.uniform(-0.005, 0.005))
        
        df['high'] = df.apply(
            lambda x: max(x['open'], x['close']) * (1 + abs(np.random.normal(0, 0.005))), 
            axis=1
        )
        
        df['low'] = df.apply(
            lambda x: min(x['open'], x['close']) * (1 - abs(np.random.normal(0, 0.005))),
            axis=1
        )
        
        df['volume'] = np.random.normal(base_price * 100, base_price * 20, len(df))
        df['volume'] = df['volume'].astype(int).clip(lower=1)
        
        return df

    def _get_binance_data(self, symbol, timeframe, days_back):
        """Get historical data from Binance"""
        try:
            # Map timeframe to Binance format
            tf_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1Hour': '1h',
                '4Hour': '4h',
                '1Day': '1d'
            }
            binance_tf = tf_map.get(timeframe, '1h')
            
            # Format symbol for Binance (e.g., BTC/USD -> BTCUSD)
            binance_symbol = symbol.replace('/', '')
            
            # Calculate time range
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            # Get klines data from Binance
            klines = self.binance_client.get_historical_klines(
                binance_symbol,
                binance_tf,
                start_time,
                end_time
            )
            
            # Convert to dataframe
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Format data types
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Error getting Binance data: {e}")
            return None

    def get_active_trades(self):
        """Get list of active trades"""
        return list(self.active_trades.values())
    
    def get_last_signal(self):
        """Get the last generated signal"""
        return self.last_signal
    
    def get_current_pnl(self):
        """Get current total PnL"""
        return self.total_pnl
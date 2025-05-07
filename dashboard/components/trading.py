import pandas as pd
import numpy as np
import sys
from pathlib import Path
import streamlit as st
import traceback
import threading
import plotly.graph_objects as go
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import os
from functools import lru_cache

# Add parent directory to path to access both config and strategy modules
sys.path.append(str(Path(__file__).parent.parent))

# Import API keys from config file
try:
    from config import API_KEY, API_SECRET, NEWS_API_KEY, ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY, FINNHUB_WEBHOOK_SECRET
except ImportError:
    # Fallback to environment variables if config not found
    API_KEY = os.environ.get("ALPACA_API_KEY", "")
    API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
    FINNHUB_WEBHOOK_SECRET = os.environ.get("FINNHUB_WEBHOOK_SECRET", "d0csqrpr01ql2j3fac90")
    if not API_KEY or not API_SECRET:
        print("Warning: API keys not found in config or environment variables")

# Import trading bot modules
try:
    from strategy.strategy import detect_consolidation, detect_breakout
    from strategy.auto_trading_manager import AutoTradingManager
    from utils.sentiment_analyzer import SentimentAnalyzer
    from utils.signal_processor import SignalProcessor
    from utils.finnhub_webhook import subscribe_to_event
except ImportError:
    print("Warning: Unable to import some strategy or utility modules")

# Auto trading state variables
auto_trader_thread = None
auto_trader_running = False
auto_trader_instance = None

# Initialize global sentiment analyzer and signal processor
sentiment_analyzer = None
signal_processor = None

def initialize_analyzers():
    """Initialize the sentiment analyzer and signal processor if not already initialized"""
    global sentiment_analyzer, signal_processor
    
    if sentiment_analyzer is None:
        api_keys = {
            'newsapi': NEWS_API_KEY,
            'alphavantage': ALPHAVANTAGE_API_KEY,
            'finnhub': FINNHUB_API_KEY,
            'finnhub_webhook_secret': FINNHUB_WEBHOOK_SECRET
        }
        sentiment_analyzer = SentimentAnalyzer(api_keys=api_keys)
        
        # Subscribe to Finnhub news events if available
        try:
            subscribe_to_event('news', sentiment_analyzer.process_finnhub_news)
        except:
            print("Warning: Could not subscribe to Finnhub news events")
        
    if signal_processor is None:
        signal_processor = SignalProcessor()

# Cache API responses to reduce API calls
@lru_cache(maxsize=32)
def fetch_historical_data(symbol, interval, limit=100, provider='alpaca'):
    """Fetch historical price data from different providers"""
    try:
        if provider == 'alpaca':
            return fetch_alpaca_data(symbol, interval, limit)
        elif provider == 'binance':
            return fetch_binance_data(symbol, interval, limit)
        else:
            st.error(f"Unsupported provider: {provider}")
            return generate_demo_data(symbol, interval, limit)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        print(f"Detailed error when fetching data: {traceback.format_exc()}")
        return generate_demo_data(symbol, interval, limit)

def fetch_alpaca_data(symbol, interval, limit=100):
    """Fetch historical price data from Alpaca"""
    try:
        # Initialize connection to Alpaca with more robust handling
        api = None
        try:
            # Try paper trading API first
            api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
            api.get_account()  # Test connection
        except Exception as e:
            # Fall back to live API if paper fails
            try:
                api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://api.alpaca.markets')
                api.get_account()
            except Exception as e2:
                # Both APIs failed
                raise Exception(f"Failed to connect to Alpaca API: {str(e2)}")
        
        # Map interval to Alpaca timeframe format
        timeframe_dict = {
            '1m': '1Min', '3m': '3Min', '5m': '5Min', '15m': '15Min', '30m': '30Min',
            '1h': '1Hour', '2h': '2Hour', '4h': '4Hour', '6h': '6Hour', '12h': '12Hour',
            '1d': '1Day'
        }
        alpaca_timeframe = timeframe_dict.get(interval, '1Hour')
        
        # Calculate time range
        end_date = datetime.now()
        interval_mins = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        minutes = interval_mins.get(interval, 60) * limit
        start_date = end_date - timedelta(minutes=minutes)
        
        # Format symbol and fetch data
        api_symbol = symbol.replace('/', '')
        bars = api.get_crypto_bars(
            api_symbol,
            alpaca_timeframe,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        ).df
        
        if not bars.empty:
            df = bars.reset_index().rename(columns={
                'timestamp': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            return df
            
        # Fall back to demo data if we got empty results
        return generate_demo_data(symbol, interval, limit, start_date, end_date)
        
    except Exception as e:
        print(f"Error fetching Alpaca data: {e}")
        return generate_demo_data(symbol, interval, limit)

def fetch_binance_data(symbol, interval, limit=100):
    """Fetch historical price data from Binance"""
    try:
        # Try to import binance client
        try:
            from binance.client import Client
            binance_available = True
        except ImportError:
            binance_available = False
            st.warning("Binance API not available. Install with: pip install python-binance")
            
        if binance_available:
            # Format symbol for Binance (e.g., BTCUSDT)
            formatted_symbol = symbol.replace('/', '').upper()
            
            # Map Streamlit interval options to Binance format
            timeframe_dict = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '6h': Client.KLINE_INTERVAL_6HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            binance_interval = timeframe_dict.get(interval, Client.KLINE_INTERVAL_1HOUR)
            
            # Initialize Binance client (no API keys needed for public data)
            client = Client()
            
            # Fetch historical klines
            klines = client.get_historical_klines(
                formatted_symbol, 
                binance_interval,
                int((datetime.now() - timedelta(days=3)).timestamp() * 1000),  # Start from 3 days ago
                int(datetime.now().timestamp() * 1000)  # Now
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_asset_volume', 
                'number_of_trades', 'taker_buy_base_asset_volume', 
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Keep only required columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Limit to the requested number of rows
            if len(df) > limit:
                df = df.tail(limit)
                
            return df
        else:
            # Fall back to demo data
            print("Binance API not available, generating demo data")
            return generate_demo_data(symbol, interval, limit)
            
    except Exception as e:
        st.error(f"Error fetching data from Binance: {e}")
        print(f"Detailed error when fetching data: {traceback.format_exc()}")
        return generate_demo_data(symbol, interval, limit)

def generate_demo_data(symbol, interval, limit=100, start_date=None, end_date=None):
    """Generate demo data for testing"""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        # Calculate start date based on interval and limit
        interval_mins = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440
        }
        minutes = interval_mins.get(interval, 60) * limit
        start_date = end_date - timedelta(minutes=minutes)
    
    print(f"Generating demo data for {symbol} from {start_date} to {end_date}")
    dates = pd.date_range(start=start_date, end=end_date, periods=limit)
    
    # Start with a base price that's somewhat realistic for the given symbol
    base_price = 100
    if "BTC" in symbol or "btc" in symbol:
        base_price = 30000
    elif "ETH" in symbol or "eth" in symbol:
        base_price = 2000
    elif "AAPL" in symbol or "aapl" in symbol:
        base_price = 180
    elif "MSFT" in symbol or "msft" in symbol:
        base_price = 350
    
    # Generate more realistic price movements
    close_prices = [base_price]
    for i in range(1, limit):
        # Random walk with momentum and some mean reversion
        momentum = 0.2  # Positive means trends tend to continue
        mean_reversion = 0.1  # How strongly prices return to the mean
        volatility = base_price * 0.01  # 1% daily volatility
        
        price_change = np.random.normal(0, volatility)
        # Add momentum (previous change affects next change)
        if i > 1:
            prev_change = close_prices[-1] - close_prices[-2]
            price_change += momentum * prev_change
        # Add mean reversion
        price_change -= mean_reversion * (close_prices[-1] - base_price)
        
        new_price = close_prices[-1] + price_change
        close_prices.append(new_price)
    
    # Generate OHLC data based on close prices
    df = pd.DataFrame()
    df['time'] = dates
    df['close'] = close_prices
    
    # Generate realistic open/high/low based on close
    for i in range(len(df)):
        # For first row, base on the close price
        if i == 0:
            df.loc[i, 'open'] = df.loc[i, 'close'] * (1 + np.random.uniform(-0.005, 0.005))
        else:
            # Open is usually near previous close
            df.loc[i, 'open'] = df.loc[i-1, 'close'] * (1 + np.random.uniform(-0.002, 0.002))
        
        # High is above both open and close
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close']) * (1 + abs(np.random.normal(0, 0.005)))
        
        # Low is below both open and close
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close']) * (1 - abs(np.random.normal(0, 0.005)))
        
    # Generate volume
    avg_volume = base_price * 1000  # Higher price, higher volume
    df['volume'] = np.random.normal(avg_volume, avg_volume * 0.3, len(df))
    df['volume'] = df['volume'].astype(int).clip(lower=1)  # Ensure positive integers
    
    print(f"Generated demo data with {len(df)} rows")
    return df

def calculate_bollinger_bands(close_prices, length=20, num_std=2.0):
    """Calculate Bollinger Bands"""
    rolling_mean = close_prices.rolling(window=length).mean()
    rolling_std = close_prices.rolling(window=length).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    middle_band = rolling_mean
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_ema(close_prices, length=50):
    """Calculate EMA"""
    return close_prices.ewm(span=length, adjust=False).mean()

def calculate_signals(df):
    """Calculate trading signals from price data including sentiment analysis"""
    # Safety check for input data
    if df is None or len(df) < 20:  # Need at least 20 candles for most indicators
        return {
            'buy_signals': pd.Series(False, index=range(len(df) if df is not None else 0)),
            'sell_signals': pd.Series(False, index=range(len(df) if df is not None else 0)),
            'combined_signal': {'signal': 'neutral', 'confidence': 0, 'reasoning': 'Insufficient data'}
        }, df
    
    # Initialize analyzers
    initialize_analyzers()
    
    # Copy dataframe to avoid modifying original
    data = df.copy(deep=True)
    
    # Calculate indicators
    signals = {}
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data['close'], length=20)
    signals['upper_band'] = upper_band
    signals['middle_band'] = middle_band
    signals['lower_band'] = lower_band
    
    # Calculate EMAs
    data['ema50'] = calculate_ema(data['close'], length=50)
    data['ema200'] = calculate_ema(data['close'], length=200)
    
    # Detect consolidation and breakout
    try:
        consolidation, bb_width, atr = detect_consolidation(data)
        buy_signals, sell_signals = detect_breakout(data, consolidation, bb_width)
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
    except Exception as e:
        # Create empty signals if detection fails
        signals['buy_signals'] = pd.Series(False, index=data.index)
        signals['sell_signals'] = pd.Series(False, index=data.index)
    
    # Add RSI
    data['rsi'] = calculate_rsi(data['close'])
    signals['rsi'] = data['rsi']
    
    # Generate combined signal
    technical_signal = {
        'signal': 'neutral',
        'confidence': 0.0,
        'reasoning': 'No clear signal'
    }
    
    # Check recent buy/sell signals
    recent_buys = signals['buy_signals'].iloc[-5:].any()
    recent_sells = signals['sell_signals'].iloc[-5:].any()
    
    if recent_buys and not recent_sells:
        technical_signal['signal'] = 'buy'
        technical_signal['confidence'] = 0.7
        technical_signal['reasoning'] = 'Breakout detected'
    elif recent_sells and not recent_buys:
        technical_signal['signal'] = 'sell'
        technical_signal['confidence'] = 0.7
        technical_signal['reasoning'] = 'Breakdown detected'
    
    # Add sentiment if available
    sentiment_signal = None
    if 'symbol' in st.session_state and sentiment_analyzer:
        try:
            sentiment = sentiment_analyzer.get_sentiment(st.session_state.symbol)
            if sentiment:
                sentiment_signal = {
                    'signal': sentiment['signal'],
                    'confidence': sentiment['confidence'],
                    'score': sentiment['score'],
                    'news_count': sentiment['news_count'],
                    'source_count': sentiment.get('source_count', 0)
                }
        except Exception:
            pass
    
    # Combine signals or use technical signal
    if signal_processor and sentiment_signal:
        signals['combined_signal'] = signal_processor.process_signals(
            symbol=st.session_state.get('symbol', 'unknown'),
            technical_signal=technical_signal,
            sentiment_signal=sentiment_signal
        )
    else:
        signals['combined_signal'] = technical_signal
    
    return signals, data

def calculate_rsi(close_prices, length=14):
    """Calculate Relative Strength Index"""
    # Get price changes
    delta = close_prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def plot_candlestick_chart(df, signals=None):
    """Plot interactive candlestick chart with indicators"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    ))
    
    # Add EMA indicators if available
    if 'ema50' in df.columns and 'ema200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['ema50'],
            line=dict(color='blue', width=1),
            name='EMA 50'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['ema200'],
            line=dict(color='red', width=1),
            name='EMA 200'
        ))
    
    # Add Bollinger Bands if signals is provided
    if signals is not None and 'upper_band' in signals and 'middle_band' in signals and 'lower_band' in signals:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=signals['upper_band'],
            line=dict(color='rgba(250, 120, 120, 0.5)', width=1),
            name='Upper BB'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=signals['middle_band'],
            line=dict(color='rgba(120, 120, 250, 0.5)', width=1),
            name='Middle BB'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=signals['lower_band'],
            line=dict(color='rgba(120, 250, 120, 0.5)', width=1),
            name='Lower BB'
        ))
        
        # Add buy/sell signals
        if 'buy_signals' in signals:
            buy_indices = signals['buy_signals'][signals['buy_signals']].index
            if not buy_indices.empty:
                buy_times = df.loc[buy_indices, 'time']
                buy_prices = df.loc[buy_indices, 'low'] * 0.99  # Slightly below the candle
                
                fig.add_trace(go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='Buy Signal'
                ))
            
        if 'sell_signals' in signals:
            sell_indices = signals['sell_signals'][signals['sell_signals']].index
            if not sell_indices.empty:
                sell_times = df.loc[sell_indices, 'time']
                sell_prices = df.loc[sell_indices, 'high'] * 1.01  # Slightly above the candle
                
                fig.add_trace(go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='Sell Signal'
                ))
    
    # Add RSI subplot if available
    if signals is not None and 'rsi' in signals:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=signals['rsi'],
            line=dict(color='purple', width=1),
            name='RSI',
            yaxis="y2"
        ))
        
        # Add RSI reference lines (30 and 70)
        fig.add_shape(
            type="line",
            x0=df['time'].iloc[0],
            y0=30,
            x1=df['time'].iloc[-1],
            y1=30,
            line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
            yref="y2"
        )
        
        fig.add_shape(
            type="line",
            x0=df['time'].iloc[0],
            y0=70,
            x1=df['time'].iloc[-1],
            y1=70,
            line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
            yref="y2"
        )
    
    # Update layout for secondary y-axis
    fig.update_layout(
        title='Price Chart with Indicators',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        yaxis2=dict(
            title="RSI",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
    )
    
    return fig

def calculate_pnl(entry_price, exit_price, direction, size):
    """Calculate profit/loss for a trade"""
    if direction == 'long':
        pnl = (exit_price - entry_price) * size
    else:  # short
        pnl = (entry_price - exit_price) * size
    
    return pnl

# Auto trading functions
def start_auto_trader(symbols, timeframe, capital, risk_percent, profit_target, daily_profit_target, 
                     use_news=True, news_weight=0.5, use_earnings=True, earnings_weight=0.6):
    """Start the auto trader in a background thread"""
    global auto_trader_thread, auto_trader_running, auto_trader_instance
    
    # Convert Streamlit timeframe format to Alpaca format for AutoTradingManager
    timeframe_dict = {
        '1m': '1Min',
        '3m': '3Min',
        '5m': '5Min',
        '15m': '15Min',
        '30m': '30Min',
        '1h': '1Hour',
        '2h': '2Hour',
        '4h': '4Hour',
        '6h': '6Hour',
        '12h': '12Hour',
        '1d': '1Day'
    }
    alpaca_timeframe = timeframe_dict.get(timeframe, '1Hour')
    
    if auto_trader_running:
        return False, "Auto trader is already running"
    
    try:
        # Create an instance of the auto trading manager without sending API keys
        # The AutoTradingManager will use the keys from config.py
        auto_trader_instance = AutoTradingManager(
            symbols=symbols,
            timeframe=alpaca_timeframe,
            capital=capital,
            risk_percent=risk_percent,
            profit_target_percent=profit_target,
            daily_profit_target=daily_profit_target,
            use_news=use_news,
            news_weight=news_weight,
            use_earnings=use_earnings,
            earnings_weight=earnings_weight
        )
        
        # Define the thread function
        def run_trader():
            try:
                auto_trader_instance.run()
            except Exception as e:
                print(f"Error in auto trader thread: {e}")
                traceback.print_exc()
        
        # Create and start the thread
        auto_trader_thread = threading.Thread(target=run_trader, daemon=True)
        auto_trader_thread.start()
        auto_trader_running = True
        
        return True, "Auto trader started successfully"
    except Exception as e:
        error_msg = f"Failed to start auto trader: {str(e)}"
        traceback.print_exc()
        return False, error_msg

def stop_auto_trader():
    """Stop the running auto trader"""
    global auto_trader_running, auto_trader_instance
    
    if not auto_trader_running:
        return False, "Auto trader is not running"
    
    try:
        # Tell the auto trader to stop
        if auto_trader_instance:
            auto_trader_instance.stop()
        
        auto_trader_running = False
        return True, "Auto trader stopped successfully"
    except Exception as e:
        error_msg = f"Failed to stop auto trader: {str(e)}"
        return False, error_msg

def get_auto_trader_status():
    """Get the current status of the auto trader"""
    global auto_trader_running, auto_trader_instance
    
    status = {
        "running": auto_trader_running,
        "trades": []
    }
    
    if auto_trader_running and auto_trader_instance:
        # Get information from auto trader instance
        status["active_trades"] = auto_trader_instance.get_active_trades() if hasattr(auto_trader_instance, 'get_active_trades') else []
        status["last_signal"] = auto_trader_instance.get_last_signal() if hasattr(auto_trader_instance, 'get_last_signal') else None
        status["pnl"] = auto_trader_instance.get_current_pnl() if hasattr(auto_trader_instance, 'get_current_pnl') else 0
    
    return status

# Force a rerun periodically to refresh the UI
if not hasattr(st.session_state, 'last_rerun') or \
   (datetime.datetime.now() - st.session_state.last_rerun).total_seconds() > 5:
    st.session_state.last_rerun = datetime.datetime.now()
    st.rerun()

from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np

# fix for pandas_ta needing np.NaN which got removed
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import time
from datetime import datetime
import sys
import traceback
import pandas_ta as ta

from strategy.strategy import check_entry
from strategy.news_strategy import NewsBasedStrategy
from strategy.earnings_report_strategy import EarningsReportStrategy
from utils.risk_management import calculate_position_size, manage_open_position
from utils.telegram_alert import send_alert
from config_manager import get_api_key, get_trading_config, ConfigManager
from utils.llm_integration import get_llm_response

config_manager = ConfigManager()
trading_config = config_manager.get_trading_config()

class VolatilityBreakoutBot:
    def __init__(self, symbol='BTCUSDT', timeframe='1h', capital=None, risk_percent=None, 
                 use_news=True, news_weight=0.5, use_earnings=True, earnings_weight=0.6):
        
        # Use config manager for defaults
        if capital is None:
            capital = trading_config.get('CAPITAL', 10000)
        if risk_percent is None:
            risk_percent = trading_config.get('RISK_PERCENT', 1.0)
            
        # Initialize connection to exchange
        self.client = Client(get_api_key('alpaca_key'), get_api_key('alpaca_secret'))
        self.symbol = symbol
        self.timeframe = self._convert_timeframe(timeframe)
        self.capital = capital
        self.risk_percent = risk_percent
        self.candles = []
        self.positions = {}  # Track open positions
        self.last_checked = None
        
        self.use_news = use_news
        self.news_weight = news_weight
        
        self.use_earnings = use_earnings
        self.earnings_weight = earnings_weight
        
        if self.use_news:
            # Configure news-based strategy
            news_config = {
                'news_lookback_days': 2,            # Days to look back for news
                'sentiment_threshold_buy': 0.2,     # Minimum sentiment score to consider buying
                'sentiment_threshold_sell': -0.15,  # Maximum sentiment score to consider selling
                'min_news_count': 3,                # Minimum number of news items required                'use_keyword_boost': True,          # Boost sentiment based on keywords
                'api_keys': {
                    'news_api': get_api_key('news_api_key'),
                    'alphavantage': get_api_key('alphavantage_key'),
                    'finnhub': get_api_key('finnhub_key')
                },
                'cache_expiry_minutes': 30,         # Cache news for 30 minutes
            }
            self.news_strategy = NewsBasedStrategy(news_config)
            
            # Add company-specific keywords to improve sentiment analysis
            base_currency = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol[:-3]
            if base_currency == 'BTC':
                self.news_strategy.set_company_keywords('BTC', 
                    positive_words=['adoption', 'institutional', 'ETF', 'lightning', 'halving'],
                    negative_words=['ban', 'regulation', 'hack', 'security breach', 'Mt. Gox']
                )
            elif base_currency == 'ETH':
                self.news_strategy.set_company_keywords('ETH',
                    positive_words=['Ethereum 2.0', 'proof of stake', 'scaling', 'DeFi', 'NFT'],
                    negative_words=['51% attack', 'gas fees', 'congestion', 'fork', 'vulnerability']
                )
        
        if self.use_earnings:
            # Configure earnings report strategy
            earnings_config = {
                'pre_event_days': 3,              # Days before event to start monitoring
                'post_event_days': 2,             # Days after event to continue monitoring
                'confidence_threshold': 0.7,      # Minimum confidence to execute a trade
                'stop_loss_percent': 3.0,         # Stop loss percentage for event trades
                'take_profit_percent': 5.0,       # Take profit percentage for event trades                'max_position_size': 0.1,         # Maximum position size as fraction of portfolio
                'api_keys': {
                    'news_api': get_api_key('news_api_key'),
                    'alphavantage': get_api_key('alphavantage_key'),
                    'finnhub': get_api_key('finnhub_key')
                },
                'use_sentiment_boost': True       # Use sentiment analysis to boost signals
            }
            self.earnings_strategy = EarningsReportStrategy(earnings_config)
            # Schedule initial calendar update
            self.next_calendar_update = datetime.now()
        
        # Test API connection
        try:
            self.client.ping()
            print(f"Successfully connected to Binance API")
        except Exception as e:
            print(f"Error connecting to Binance API: {e}")
            sys.exit(1)
            
        # Load initial historical data
        self._load_historical_data()
        
    def _convert_timeframe(self, timeframe):
        # map our timeframe strings to binance constants
        timeframe_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }
        return timeframe_map.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
        
    def _load_historical_data(self):
        try:
            klines = self.client.get_historical_klines(
                self.symbol, self.timeframe, "100 hours ago UTC")
            
            # Convert to pandas DataFrame
            self.df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                            'close_time', 'quote_asset_volume', 'number_of_trades',
                                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].astype(float)
                
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            
            # Set timestamp as index
            self.df.set_index('timestamp', inplace=True)
            
            print(f"Loaded {len(self.df)} candles of historical data")
        except Exception as e:
            print(f"Error loading historical data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _update_candles(self):
        try:
            # Get latest klines
            klines = self.client.get_klines(
                symbol=self.symbol, interval=self.timeframe, limit=2)
            
            latest_candle = {
                'timestamp': pd.to_datetime(klines[-1][0], unit='ms'),
                'open': float(klines[-1][1]),
                'high': float(klines[-1][2]),
                'low': float(klines[-1][3]),
                'close': float(klines[-1][4]),
                'volume': float(klines[-1][5])
            }
            
            # If we have a new candle, append it
            if latest_candle['timestamp'] not in self.df.index:
                self.df = self.df.append(pd.Series(latest_candle, name=latest_candle['timestamp']))
                self.df = self.df.iloc[-100:]  # Keep only most recent 100 candles
                print(f"New candle added: {latest_candle['timestamp']}")
            # Otherwise update the current candle
            else:
                self.df.loc[latest_candle['timestamp']] = pd.Series(latest_candle)
                print(f"Updated candle: {latest_candle['timestamp']}")
            
            return True
        except Exception as e:
            print(f"Error updating candles: {e}")
            traceback.print_exc()
            return False
    
    def _process_news_signals(self):
        if not self.use_news:
            return None
            
        try:
            # Extract base currency/stock ticker from symbol
            base_currency = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol[:-3]
            
            # Get news sentiment
            sentiment_score, news_confidence, news_count = self.news_strategy.get_sentiment(base_currency)
            print(f"News sentiment for {base_currency}: Score={sentiment_score:.2f}, Confidence={news_confidence:.2f}, Count={news_count}")
            
            # Generate signal based on news sentiment
            if news_count >= self.news_strategy.min_news_count:
                if sentiment_score > self.news_strategy.sentiment_threshold_buy and news_confidence > 0.6:
                    return {'signal': 'BUY', 'score': sentiment_score, 'confidence': news_confidence}
                elif sentiment_score < self.news_strategy.sentiment_threshold_sell and news_confidence > 0.6:
                    return {'signal': 'SELL', 'score': sentiment_score, 'confidence': news_confidence}
                
            return {'signal': 'NEUTRAL', 'score': sentiment_score, 'confidence': news_confidence}
            
        except Exception as e:
            print(f"Error processing news signals: {e}")
            traceback.print_exc()
            return None
    
    def _process_earnings_signals(self):
        if not self.use_earnings:
            return None
            
        try:
            # Update earnings calendar if needed (once per day)
            current_time = datetime.now()
            if current_time > self.next_calendar_update:
                print("Updating earnings calendar...")
                self.earnings_strategy.update_calendar()
                # Schedule next update for tomorrow
                self.next_calendar_update = current_time.replace(hour=0, minute=0, second=0) + pd.Timedelta(days=1)
            
            # Extract ticker from symbol
            ticker = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol[:-3]
            
            # Get earnings signal
            signal = self.earnings_strategy.check_earnings_events(ticker)
            
            if signal:
                print(f"Earnings signal for {ticker}: {signal['signal']} (Confidence: {signal['confidence']:.2f})")
                return signal
            
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
            
        except Exception as e:
            print(f"Error processing earnings signals: {e}")
            traceback.print_exc()
            return None
    
    def _execute_trade(self, direction, confidence, source='technical'):
        try:
            # figure out position size based on risk/confidence
            position_size = calculate_position_size(
                self.capital, 
                self.risk_percent, 
                trading_config.get('MAX_CAPITAL_PER_TRADE', 5000), 
                confidence,
                self.df['close'].iloc[-1]
            )
            
            # Simulate order (would place real orders in production)
            current_price = self.df['close'].iloc[-1]
            trade_info = {
                'symbol': self.symbol,
                'direction': direction,
                'entry_price': current_price,
                'position_size': position_size,
                'timestamp': datetime.now(),
                'source': source
            }
            
            # set SL/TP
            if direction == 'BUY':
                trade_info['stop_loss'] = current_price * 0.97
                trade_info['take_profit'] = current_price * 1.06
            else:
                trade_info['stop_loss'] = current_price * 1.03
                trade_info['take_profit'] = current_price * 0.94
            
            # store position
            position_id = f"{self.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.positions[position_id] = trade_info
            
            # Send alert
            alert_message = f"🚨 {source.upper()} SIGNAL: {direction} {self.symbol}\n" + \
                          f"Price: ${current_price:.2f}\n" + \
                          f"Position Size: {position_size:.2f}\n" + \
                          f"Stop Loss: ${trade_info['stop_loss']:.2f}\n" + \
                          f"Take Profit: ${trade_info['take_profit']:.2f}\n" + \
                          f"Confidence: {confidence:.2f}"
            
            send_alert(alert_message)
            print(alert_message)
            return True
        except Exception as e:
            print(f"Error executing trade: {e}")
            traceback.print_exc()
            return False
    
    def analyze_market_with_llm(self, market_data):
        # send market snapshot to LLM for a second opinion
        prompt = f"""Analyze these market conditions and suggest trading strategy:
        
        {market_data}
        
        Provide recommendations in this format:
        1. Primary trend: ...
        2. Key indicators: ...
        3. Recommended action: ...
        4. Risk considerations: ..."""
        
        return get_llm_response(prompt, self.llm_model)
    
    def run(self):
        # main loop - runs until killed
        print(f"Starting VolatilityBreakoutBot for {self.symbol}")
        print(f"News-based strategy enabled: {self.use_news}")
        print(f"Earnings strategy enabled: {self.use_earnings}")
        
        while True:
            try:
                # Update candlestick data
                if not self._update_candles():
                    print("Failed to update candles, retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                
                # Calculate technical indicators
                self.df['atr'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
                self.df['sma20'] = ta.sma(self.df['close'], length=20)
                self.df['rsi'] = ta.rsi(self.df['close'], length=14)
                
                # Fix for Bollinger Bands - pandas_ta returns a DataFrame, not three separate values
                bbands = ta.bbands(self.df['close'], length=20)
                self.df['upper_band'] = bbands['BBU_20_2.0']
                self.df['middle_band'] = bbands['BBM_20_2.0']
                self.df['lower_band'] = bbands['BBL_20_2.0']
                
                # Check for technical trading signals
                tech_signal, tech_confidence = check_entry(self.df)
                print(f"Technical signal: {tech_signal}, confidence: {tech_confidence:.2f}")
                
                # Initialize combined signals with technical analysis
                combined_signal = tech_signal
                combined_confidence = tech_confidence
                
                # Process news-based signals
                news_signal_data = self._process_news_signals()
                if news_signal_data:
                    news_signal = news_signal_data['signal']
                    news_confidence = news_signal_data['confidence']
                    
                    # Combine technical and news signals using weighted approach
                    if news_signal != 'NEUTRAL':
                        # If signals agree, boost confidence
                        if (tech_signal == 'BUY' and news_signal == 'BUY') or (tech_signal == 'SELL' and news_signal == 'SELL'):
                            combined_signal = tech_signal
                            combined_confidence = (tech_confidence * (1 - self.news_weight)) + (news_confidence * self.news_weight)
                            print(f"Technical and news signals AGREE: {combined_signal} with confidence {combined_confidence:.2f}")
                        # If signals conflict, use the one with higher confidence * weight
                        elif tech_signal != 'NEUTRAL' and news_signal != 'NEUTRAL':
                            tech_weighted = tech_confidence * (1 - self.news_weight)
                            news_weighted = news_confidence * self.news_weight
                            
                            if tech_weighted > news_weighted:
                                combined_signal = tech_signal
                                combined_confidence = tech_weighted
                                print(f"Technical signal overrides news: {combined_signal} with confidence {combined_confidence:.2f}")
                            else:
                                combined_signal = news_signal
                                combined_confidence = news_weighted
                                print(f"News signal overrides technical: {combined_signal} with confidence {combined_confidence:.2f}")
                        # If only one has a signal, use that with reduced confidence
                        elif tech_signal != 'NEUTRAL':
                            combined_signal = tech_signal
                            combined_confidence = tech_confidence * (1 - self.news_weight/2)
                            print(f"Using technical signal only: {combined_signal} with confidence {combined_confidence:.2f}")
                        elif news_signal != 'NEUTRAL':
                            combined_signal = news_signal
                            combined_confidence = news_confidence * self.news_weight
                            print(f"Using news signal only: {combined_signal} with confidence {combined_confidence:.2f}")
                
                # Process earnings signals
                earnings_signal_data = self._process_earnings_signals()
                if earnings_signal_data and earnings_signal_data['signal'] != 'NEUTRAL':
                    earnings_signal = earnings_signal_data['signal']
                    earnings_confidence = earnings_signal_data['confidence']
                    
                    # High-confidence earnings signals can override other signals
                    if earnings_confidence > 0.8:
                        combined_signal = earnings_signal
                        combined_confidence = earnings_confidence
                        print(f"High-confidence earnings signal: {combined_signal} with confidence {combined_confidence:.2f}")
                    # Otherwise blend with existing signal
                    elif combined_signal != 'NEUTRAL':
                        # If signals agree, boost confidence
                        if combined_signal == earnings_signal:
                            combined_confidence = combined_confidence * (1 - self.earnings_weight) + earnings_confidence * self.earnings_weight
                            print(f"Earnings signal boosts existing signal: {combined_signal} with confidence {combined_confidence:.2f}")
                        # If signals conflict, use the higher confidence one
                        else:
                            earnings_weighted = earnings_confidence * self.earnings_weight
                            existing_weighted = combined_confidence * (1 - self.earnings_weight)
                            
                            if earnings_weighted > existing_weighted:
                                combined_signal = earnings_signal
                                combined_confidence = earnings_weighted
                                print(f"Earnings signal overrides: {combined_signal} with confidence {combined_confidence:.2f}")
                    # If no signal yet, use earnings signal
                    elif combined_signal == 'NEUTRAL' and earnings_signal != 'NEUTRAL':
                        combined_signal = earnings_signal
                        combined_confidence = earnings_confidence * self.earnings_weight
                        print(f"Using earnings signal: {combined_signal} with confidence {combined_confidence:.2f}")
                
                # Execute trade if signal confidence is high enough
                if combined_signal != 'NEUTRAL' and combined_confidence > 0.6:
                    signal_source = "combined"
                    self._execute_trade(combined_signal, combined_confidence, signal_source)
                
                # Manage open positions (check stop loss/take profit)
                for pos_id in list(self.positions.keys()):
                    manage_open_position(self.positions, pos_id, self.df['close'].iloc[-1])
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(60)  # Wait before retrying


if __name__ == "__main__":    # Initialize the bot with both news and earnings strategies
    bot = VolatilityBreakoutBot(
        symbol='BTCUSDT',
        timeframe='1h',
        capital=trading_config.get('CAPITAL', 10000),
        risk_percent=trading_config.get('RISK_PERCENT', 1.0),
        use_news=True,
        news_weight=0.5,
        use_earnings=True,
        earnings_weight=0.6
    )
    
    # Run the bot
    bot.run()

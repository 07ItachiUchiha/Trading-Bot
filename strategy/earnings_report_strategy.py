import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from utils.news_fetcher import NewsFetcher
from strategy.news_strategy import NewsBasedStrategy
import requests
import json
import time
import logging

logger = logging.getLogger('earnings_report_strategy')

class EarningsReportStrategy:
    """
    A strategy specifically designed to trade around earnings reports and economic data releases.
    
    This strategy:
    1. Tracks upcoming earnings releases and economic events
    2. Analyzes the impact of the report when released (beat/miss/in-line)
    3. Executes trades based on predefined rules
    4. Can implement pre-event and post-event trading strategies
    """
    
    def __init__(self, config=None):
        """
        Initialize the earnings report strategy
        
        Args:
            config (dict): Strategy configuration
        """
        # Default configuration
        self.config = {
            'pre_event_days': 3,              # Days before event to start monitoring
            'post_event_days': 2,             # Days after event to continue monitoring
            'confidence_threshold': 0.7,      # Minimum confidence to execute a trade
            'stop_loss_percent': 3.0,         # Stop loss percentage for event trades
            'take_profit_percent': 5.0,       # Take profit percentage for event trades
            'max_position_size': 0.1,         # Maximum position size as fraction of portfolio
            'api_keys': {
                'news_api': None,
                'alphavantage': None,
                'finnhub': None
            },
            'use_sentiment_boost': True,      # Use sentiment analysis to boost signals
            'earnings_keywords': [
                'earnings', 'revenue', 'profit', 'loss', 'EPS', 'guidance',
                'forecast', 'outlook', 'beat', 'miss', 'estimate', 'exceeded',
                'quarterly', 'annual', 'report', 'financial results'
            ],
            'economic_event_keywords': [
                'fed', 'interest rate', 'fomc', 'inflation', 'cpi', 'ppi',
                'unemployment', 'jobs', 'nonfarm', 'gdp', 'economic', 'economy'
            ]
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize news fetcher
        self.news_fetcher = NewsFetcher()
        if self.config['api_keys']['news_api']:
            self.news_fetcher.set_api_keys(
                news_api_key=self.config['api_keys']['news_api'],
                alphavantage_key=self.config['api_keys']['alphavantage'],
                finnhub_key=self.config['api_keys']['finnhub']
            )
        
        # Initialize news strategy for sentiment analysis
        self.news_strategy = NewsBasedStrategy(config)
        
        # Track currently monitored events
        self.monitored_events = {}
        
        # Cache for earnings calendars
        self.earnings_calendar_cache = {}
        self.economic_calendar_cache = {}
        self.last_cache_update = 0
        self.cache_expiry = 6 * 60 * 60  # 6 hours
    
    def update_calendars(self):
        """
        Update the earnings and economic calendars
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Check if cache is still valid
        if time.time() - self.last_cache_update < self.cache_expiry:
            return True
        
        success = True
        
        # Try to update earnings calendar
        if self.config['api_keys']['finnhub']:
            try:
                # Get earnings calendar for the next 30 days
                today = datetime.now().strftime("%Y-%m-%d")
                next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                
                url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={next_month}&token={self.config['api_keys']['finnhub']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    self.earnings_calendar_cache = data.get("earningsCalendar", [])
                else:
                    print(f"Error fetching earnings calendar: {response.status_code} - {response.text}")
                    success = False
                    
            except Exception as e:
                print(f"Error updating earnings calendar: {e}")
                success = False
        else:
            # No API key for earnings calendar
            success = False
        
        # Try to update economic calendar
        if self.config['api_keys']['finnhub']:
            try:
                # Get economic calendar for the next 30 days
                today = datetime.now().strftime("%Y-%m-%d")
                next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                
                url = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={next_month}&token={self.config['api_keys']['finnhub']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    self.economic_calendar_cache = data.get("economicCalendar", [])
                else:
                    print(f"Error fetching economic calendar: {response.status_code} - {response.text}")
                    success = False
                    
            except Exception as e:
                print(f"Error updating economic calendar: {e}")
                success = False
        else:
            # No API key for economic calendar
            success = False
        
        if success:
            self.last_cache_update = time.time()
        
        return success
    
    def get_upcoming_events(self, symbol=None, days_ahead=7):
        """
        Get upcoming earnings and economic events
        
        Args:
            symbol (str): Optional stock symbol to filter by
            days_ahead (int): How many days ahead to look
            
        Returns:
            dict: Dictionary containing earnings and economic events
        """
        # Update calendars if needed
        self.update_calendars()
        
        # Filter earnings events
        upcoming_earnings = []
        target_date = datetime.now() + timedelta(days=days_ahead)
        
        for event in self.earnings_calendar_cache:
            try:
                event_date = datetime.strptime(event.get("date", ""), "%Y-%m-%d")
                if datetime.now() <= event_date <= target_date:
                    if symbol is None or event.get("symbol") == symbol:
                        upcoming_earnings.append(event)
            except:
                pass
        
        # Filter economic events
        upcoming_economic = []
        for event in self.economic_calendar_cache:
            try:
                event_date = datetime.strptime(event.get("date", ""), "%Y-%m-%d")
                if datetime.now() <= event_date <= target_date:
                    if event.get("impact") == "high":  # Only high-impact events
                        upcoming_economic.append(event)
            except:
                pass
        
        return {
            "earnings": upcoming_earnings,
            "economic": upcoming_economic
        }
    
    def detect_earnings_event(self, symbol, news_items):
        """
        Detect if earnings or significant report was released
        
        Args:
            symbol (str): Stock symbol
            news_items (list): List of news items
            
        Returns:
            dict: Event details or None if no event detected
        """
        for item in news_items:
            title = item.get("title", "").lower()
            description = item.get("description", "").lower()
            combined_text = f"{title} {description}"
            
            # Check for earnings keywords
            keyword_matches = 0
            for keyword in self.config['earnings_keywords']:
                if keyword.lower() in combined_text:
                    keyword_matches += 1
            
            # If multiple earnings keywords found, likely an earnings report
            if keyword_matches >= 3:
                # Detect if beat, miss, or in-line
                result = "unknown"
                if any(word in combined_text for word in ["beat", "exceed", "better", "above"]):
                    result = "beat"
                elif any(word in combined_text for word in ["miss", "below", "disappoint", "short"]):
                    result = "miss"
                elif any(word in combined_text for word in ["in-line", "in line", "matches", "expected", "met"]):
                    result = "in-line"
                
                return {
                    "type": "earnings",
                    "symbol": symbol,
                    "result": result,
                    "news_item": item,
                    "detected_at": datetime.now().isoformat(),
                }
        
        return None
    
    def detect_economic_event(self, news_items):
        """
        Detect if a significant economic event occurred
        
        Args:
            news_items (list): List of news items
            
        Returns:
            dict: Event details or None if no event detected
        """
        for item in news_items:
            title = item.get("title", "").lower()
            description = item.get("description", "").lower()
            combined_text = f"{title} {description}"
            
            # Check for economic event keywords
            keyword_matches = 0
            matched_keywords = []
            for keyword in self.config['economic_event_keywords']:
                if keyword.lower() in combined_text:
                    keyword_matches += 1
                    matched_keywords.append(keyword)
            
            # If multiple economic keywords found, likely an economic event
            if keyword_matches >= 2:
                # Determine the event type
                event_type = "economic"
                if "fed" in matched_keywords or "interest rate" in matched_keywords or "fomc" in matched_keywords:
                    event_type = "fed_decision"
                elif "inflation" in matched_keywords or "cpi" in matched_keywords or "ppi" in matched_keywords:
                    event_type = "inflation_report"
                elif "jobs" in matched_keywords or "unemployment" in matched_keywords or "nonfarm" in matched_keywords:
                    event_type = "jobs_report"
                elif "gdp" in matched_keywords:
                    event_type = "gdp_report"
                
                return {
                    "type": event_type,
                    "result": "unknown",  # Would need more complex logic to determine if good/bad
                    "news_item": item,
                    "detected_at": datetime.now().isoformat(),
                }
        
        return None
    
    def generate_signal(self, symbol, data=None, technical_signal=None):
        """
        Generate a trading signal based on earnings/economic events
        
        Args:
            symbol (str): Trading symbol
            data (pd.DataFrame): Optional price data
            technical_signal (str): Optional technical signal
            
        Returns:
            dict: Trading signal information
        """
        # Fetch recent news
        news = self.news_fetcher.get_news_for_symbol(
            symbol, 
            days=1,  # Focus on very recent news
            max_items=20,
            use_cache=False  # Always get fresh news for event detection
        )
        
        if not news:
            return {
                'signal': 'neutral',
                'confidence': 0,
                'reasoning': 'No recent news found',
                'event_detected': False
            }
        
        # Detect earnings event
        earnings_event = self.detect_earnings_event(symbol, news)
        
        # Detect economic event
        economic_event = self.detect_economic_event(news)
        
        # If no events detected, return neutral signal
        if not earnings_event and not economic_event:
            # Check for upcoming events
            upcoming = self.get_upcoming_events(symbol, days_ahead=self.config['pre_event_days'])
            
            if upcoming['earnings']:
                # Upcoming earnings event - could implement pre-event strategy here
                days_to_event = 0  # Calculate days to earnings
                return {
                    'signal': 'neutral',
                    'confidence': 0.3,
                    'reasoning': f'Upcoming earnings in {days_to_event} days',
                    'event_detected': False,
                    'upcoming_event': upcoming['earnings'][0]
                }
            
            return {
                'signal': 'neutral',
                'confidence': 0,
                'reasoning': 'No earnings or economic events detected',
                'event_detected': False
            }
        
        # Process earnings event
        if earnings_event:
            signal = 'neutral'
            confidence = 0.5
            reasoning = f"Earnings report detected: {earnings_event['result']}"
            
            # Generate signal based on earnings result
            if earnings_event['result'] == 'beat':
                signal = 'buy'
                confidence = 0.8
                reasoning = "Earnings beat detected"
            elif earnings_event['result'] == 'miss':
                signal = 'sell'
                confidence = 0.8
                reasoning = "Earnings miss detected"
            
            # If we have a news strategy and sentiment boost is enabled, use it
            if self.config['use_sentiment_boost'] and hasattr(self, 'news_strategy'):
                # Get sentiment from news strategy
                sentiment_result = self.news_strategy.generate_signal(symbol)
                sentiment_signal = sentiment_result['signal']
                sentiment_confidence = sentiment_result['confidence']
                
                # Adjust signal if sentiment strongly contradicts earnings result
                if sentiment_signal != signal and sentiment_confidence > 0.7:
                    # Reduce confidence in our signal
                    confidence *= 0.7
                    reasoning += f" but sentiment analysis suggests {sentiment_signal} ({sentiment_confidence:.2f})"
                elif sentiment_signal == signal:
                    # Increase confidence if sentiment agrees
                    confidence = min(0.95, confidence * 1.2)
                    reasoning += f" confirmed by sentiment analysis ({sentiment_confidence:.2f})"
            
            # Add stop loss and take profit values
            stop_loss = None
            take_profit = None
            
            if data is not None and len(data) > 0:
                current_price = data.iloc[-1]['close']
                if signal == 'buy':
                    stop_loss = current_price * (1 - self.config['stop_loss_percent']/100)
                    take_profit = current_price * (1 + self.config['take_profit_percent']/100)
                elif signal == 'sell':
                    stop_loss = current_price * (1 + self.config['stop_loss_percent']/100)
                    take_profit = current_price * (1 - self.config['take_profit_percent']/100)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning,
                'event_detected': True,
                'event_type': 'earnings',
                'event_details': earnings_event,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': self.config['max_position_size'] * confidence
            }
        
        # Process economic event
        if economic_event:
            # For economic events, we rely more on sentiment analysis
            if self.config['use_sentiment_boost'] and hasattr(self, 'news_strategy'):
                sentiment_result = self.news_strategy.generate_signal(symbol)
                signal = sentiment_result['signal']
                confidence = sentiment_result['confidence'] * 0.8  # Slightly reduced confidence 
                reasoning = f"{economic_event['type']} detected, sentiment suggests {signal}"
            else:
                signal = 'neutral'
                confidence = 0.3
                reasoning = f"{economic_event['type']} detected but unable to determine direction"
            
            # Add stop loss and take profit values
            stop_loss = None
            take_profit = None
            
            if data is not None and len(data) > 0:
                current_price = data.iloc[-1]['close']
                if signal == 'buy':
                    stop_loss = current_price * (1 - self.config['stop_loss_percent']/100)
                    take_profit = current_price * (1 + self.config['take_profit_percent']/100)
                elif signal == 'sell':
                    stop_loss = current_price * (1 + self.config['stop_loss_percent']/100)
                    take_profit = current_price * (1 - self.config['take_profit_percent']/100)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning,
                'event_detected': True,
                'event_type': economic_event['type'],
                'event_details': economic_event,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': self.config['max_position_size'] * confidence
            }
    
    def should_exit_position(self, symbol, entry_price, position_type, entry_time, current_price, current_time):
        """
        Determine if we should exit an event-based position
        
        Args:
            symbol (str): Trading symbol
            entry_price (float): Entry price
            position_type (str): 'long' or 'short'
            entry_time (datetime): Entry time
            current_price (float): Current price
            current_time (datetime): Current time
            
        Returns:
            tuple: (should_exit, reason)
        """
        # Calculate holding time
        holding_time = current_time - entry_time
        max_holding_days = self.config['post_event_days']
        
        # If we've held the position for the maximum time, exit
        if holding_time.days >= max_holding_days:
            return (True, f"Maximum holding period ({max_holding_days} days) reached")
        
        # Calculate profit/loss
        if position_type == 'long':
            pnl_percent = (current_price / entry_price - 1) * 100
            
            # Stop loss hit
            if pnl_percent <= -self.config['stop_loss_percent']:
                return (True, f"Stop loss triggered at {pnl_percent:.2f}% loss")
            
            # Take profit hit
            if pnl_percent >= self.config['take_profit_percent']:
                return (True, f"Take profit triggered at {pnl_percent:.2f}% gain")
                
        elif position_type == 'short':
            pnl_percent = (1 - current_price / entry_price) * 100
            
            # Stop loss hit
            if pnl_percent <= -self.config['stop_loss_percent']:
                return (True, f"Stop loss triggered at {pnl_percent:.2f}% loss")
            
            # Take profit hit
            if pnl_percent >= self.config['take_profit_percent']:
                return (True, f"Take profit triggered at {pnl_percent:.2f}% gain")
        
        # Check for reversal signals in newer news
        news = self.news_fetcher.get_news_for_symbol(
            symbol, 
            days=1,
            max_items=10,
            use_cache=False  # Get fresh news
        )
        
        if news:
            if self.config['use_sentiment_boost'] and hasattr(self, 'news_strategy'):
                sentiment_result = self.news_strategy.generate_signal(symbol)
                sentiment_signal = sentiment_result['signal']
                sentiment_confidence = sentiment_result['confidence']
                
                # If sentiment strongly contradicts our position, consider exiting
                if (position_type == 'long' and sentiment_signal == 'sell' and sentiment_confidence > 0.8) or \
                   (position_type == 'short' and sentiment_signal == 'buy' and sentiment_confidence > 0.8):
                    return (True, f"Sentiment reversal detected: {sentiment_signal} with {sentiment_confidence:.2f} confidence")
        
        return (False, "Maintain position")
    
    def update_calendar(self, force=False):
        """
        Update the economic calendar data
        
        Args:
            force (bool): Force update even if the data is recent
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            # Skip update if we've updated recently and not forcing
            if not force and self.last_cache_update and datetime.now() - self.last_cache_update < self.cache_expiry:
                logger.debug("Using cached calendar data")
                return True
            
            logger.info("Updating economic calendar data")
            
            # If we have an API key, fetch real data
            if self.config['api_keys']['finnhub']:
                # Start and end dates for the calendar (2 weeks range)
                start_date = datetime.now() - timedelta(days=7)
                end_date = datetime.now() + timedelta(days=7)
                
                # Format dates for API
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Example API call to financial calendar provider
                # This is a placeholder - replace with actual API endpoint
                url = f"https://api.example.com/calendar?apikey={self.config['api_keys']['finnhub']}&from={start_str}&to={end_str}"
                
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        self.calendar_data = response.json()
                        self.last_cache_update = datetime.now()
                        logger.info(f"Calendar updated successfully with {len(self.calendar_data.get('data', []))} events")
                        return True
                    else:
                        logger.warning(f"Failed to update calendar: {response.status_code}")
                        # Continue with mock data if API fails
                except Exception as e:
                    logger.error(f"Error connecting to calendar API: {e}")
                    # Continue with mock data if API fails
            
            # Generate mock calendar data if no API key or API failed
            self.calendar_data = self._generate_mock_calendar()
            self.last_cache_update = datetime.now()
            logger.info("Using mock calendar data")
            return True
            
        except Exception as e:
            logger.error(f"Error updating calendar: {e}")
            return False
    
    def _generate_mock_calendar(self):
        """Generate mock calendar data for testing"""
        # Current date for reference
        now = datetime.now()
        
        # Sample company symbols from different sectors
        companies = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com, Inc.',
            'META': 'Meta Platforms, Inc.',
            'TSLA': 'Tesla, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp',
            'WMT': 'Walmart Inc.'
        }
        
        # Generate events across next 7 days
        events = []
        
        for i, (symbol, name) in enumerate(companies.items()):
            # Create an event date that's -3 to +10 days from now
            event_date = now + timedelta(days=i-3)
            
            events.append({
                'date': event_date.strftime('%Y-%m-%d'),
                'time': '16:30:00',
                'symbol': symbol,
                'name': name,
                'event': 'Earnings Report',
                'expected': f'${(i+1)*0.25:.2f}',
                'previous': f'${(i+1)*0.23:.2f}',
                'importance': 'high' if i < 5 else 'medium'
            })
            
        return {'data': events}
    
    def get_upcoming_events(self, symbol=None, days_ahead=7):
        """
        Get upcoming calendar events filtered by symbol
        
        Args:
            symbol (str): Symbol to filter events for, or None for all events
            days_ahead (int): Number of days ahead to look for events
            
        Returns:
            list: List of upcoming events
        """
        # Update calendar if needed
        self.update_calendar()
        
        # Filter events
        upcoming_events = []
        now = datetime.now()
        cutoff_date = now + timedelta(days=days_ahead)
        
        for event in self.calendar_data.get('data', []):
            try:
                # Skip if not an earnings event
                if 'event' not in event or 'earnings' not in event['event'].lower():
                    continue
                    
                # Filter by symbol if provided
                if symbol and event.get('symbol') != symbol:
                    continue
                
                # Check if event is within the time window
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                if now <= event_date <= cutoff_date:
                    upcoming_events.append(event)
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                continue
                
        return upcoming_events
    
    def get_signals(self, symbol, timeframe='1d', lookback_days=30):
        """
        Get trading signals based on earnings reports
        
        Args:
            symbol (str): Symbol to generate signals for
            timeframe (str): Timeframe for analysis
            lookback_days (int): Days to look back for historical earnings
            
        Returns:
            dict: Signal information including direction, confidence, and reasoning
        """
        try:
            # Ensure calendar is updated
            self.update_calendar()
            
            # Get upcoming events for this symbol
            upcoming_events = self.get_upcoming_events(symbol)
            
            # Default signal
            signal = {
                'signal': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No significant earnings events found'
            }
            
            # Check if there are upcoming earnings
            if upcoming_events:
                event = upcoming_events[0]
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                days_until = (event_date - datetime.now()).days
                
                if days_until <= 2:
                    # Very close to earnings - high volatility expected
                    signal['signal'] = 'neutral'
                    signal['confidence'] = 0.7
                    signal['reasoning'] = f"Earnings report in {days_until} days. Avoiding position due to potential volatility."
                elif days_until <= 5:
                    # Pre-earnings run-up often happens
                    signal['signal'] = 'buy'
                    signal['confidence'] = 0.6
                    signal['reasoning'] = f"Potential pre-earnings run-up, report in {days_until} days."
                else:
                    # More distant earnings
                    signal['signal'] = 'neutral'
                    signal['confidence'] = 0.3
                    signal['reasoning'] = f"Earnings report in {days_until} days."
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating earnings signals: {e}")
            return {'signal': 'neutral', 'confidence': 0.0, 'reasoning': f"Error analyzing earnings: {str(e)}"}
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import os

from strategy.news_strategy import NewsBasedStrategy
from strategy.earnings_report_strategy import EarningsReportStrategy
from strategy.strategy import detect_consolidation, detect_breakout, calculate_targets_and_stop
from utils.risk_management import RiskManager

class AutoTrader:
    """
    Automated trading system that combines news, earnings reports, and technical analysis
    to execute trades with risk management.
    
    This autotrader:
    1. Monitors market for trading opportunities using multiple strategies
    2. Evaluates signals and executes trades based on confidence and risk parameters
    3. Manages open positions with trailing stops and profit targets
    4. Logs trading activity and performance metrics
    """
    
    def __init__(self, broker_api=None, config=None):
        """
        Initialize the auto trader with broker API connection and configuration
        
        Args:
            broker_api: API object for the broker to execute trades
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'trading_enabled': False,        # Safety switch to enable/disable actual trading
            'trading_hours': {
                'start': '09:30',            # Market open time (EST)
                'end': '16:00'               # Market close time (EST)
            },
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],  # Default watch list
            'max_positions': 5,              # Maximum concurrent positions
            'max_position_size': 0.1,        # Maximum position size as fraction of portfolio
            'min_confidence': 0.7,           # Minimum confidence threshold to execute a trade
            'signal_aging_hours': 2,         # How long a signal remains valid
            'multi_signal_boost': 0.1,       # Boost confidence if multiple signals align
            'scan_interval_seconds': 300,    # How often to scan for new opportunities (5 minutes)
            'api_keys': {
                'news_api': None,
                'alphavantage': None,
                'finnhub': None
            },
            'profit_taking': {
                'partial_at_target1': 0.3,   # Take 30% off at first target
                'partial_at_target2': 0.3,   # Take another 30% off at second target
                'trailing_stop_after_target1': True,  # Enable trailing stop after hitting first target
                'trailing_stop_percent': 2.0  # Trailing stop percentage
            },
            'risk_management': {
                'max_daily_loss': 3.0,       # Maximum daily loss as % of portfolio
                'max_trade_loss': 1.0,       # Maximum loss per trade as % of portfolio
                'max_correlated_exposure': 15.0,  # Maximum exposure to correlated assets
            }
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize broker API connection
        self.broker = broker_api
        
        # Initialize strategies
        self.news_strategy = NewsBasedStrategy(self.config)
        self.earnings_strategy = EarningsReportStrategy(self.config)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config['risk_management'])
        
        # Track active positions and signals
        self.active_positions = {}
        self.active_signals = {}
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join('logs', f'autotrader_{datetime.now().strftime("%Y%m%d")}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AutoTrader')
        self.logger.info("AutoTrader initialized with configuration: %s", json.dumps(self.config, indent=2))
    
    def is_trading_hours(self):
        """
        Check if current time is within trading hours
        
        Returns:
            bool: True if within trading hours, False otherwise
        """
        now = datetime.now().time()
        start_time = datetime.strptime(self.config['trading_hours']['start'], '%H:%M').time()
        end_time = datetime.strptime(self.config['trading_hours']['end'], '%H:%M').time()
        
        return start_time <= now <= end_time
    
    def get_market_data(self, symbol, timeframe='1h', bars=100):
        """
        Get market data for a symbol
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for the data
            bars (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: Market data with OHLCV
        """
        if self.broker is None:
            self.logger.warning("No broker API connected, cannot fetch market data")
            return None
        
        try:
            # This would be implemented based on the specific broker API
            data = self.broker.get_historical_data(symbol, timeframe, bars)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def scan_for_signals(self):
        """
        Scan all symbols for trading signals
        
        Returns:
            dict: Dictionary of new trading signals by symbol
        """
        new_signals = {}
        
        for symbol in self.config['symbols']:
            try:
                # Skip if we already have an active position for this symbol
                if symbol in self.active_positions:
                    continue
                
                # Get market data
                data = self.get_market_data(symbol)
                if data is None:
                    continue
                
                # Apply technical analysis
                consolidation, bb_width, atr = detect_consolidation(data)
                buy_signals, sell_signals = detect_breakout(data, consolidation, bb_width)
                
                # Get news-based signal
                news_signal = self.news_strategy.generate_signal(symbol)
                
                # Get earnings-based signal
                earnings_signal = self.earnings_strategy.generate_signal(symbol, data)
                
                # Combine signals
                final_signal = self._combine_signals(
                    symbol, 
                    data,
                    technical_buy=buy_signals.iloc[-1],
                    technical_sell=sell_signals.iloc[-1],
                    news_signal=news_signal,
                    earnings_signal=earnings_signal,
                    atr=atr.iloc[-1]
                )
                
                # Add to new signals if confidence is above threshold
                if final_signal['confidence'] >= self.config['min_confidence']:
                    new_signals[symbol] = final_signal
                    self.active_signals[symbol] = final_signal
                    self.active_signals[symbol]['timestamp'] = datetime.now()
                    
                    self.logger.info(f"New trading signal for {symbol}: {final_signal['signal']} with confidence {final_signal['confidence']:.2f}")
                    self.logger.info(f"Reasoning: {final_signal['reasoning']}")
                
            except Exception as e:
                self.logger.error(f"Error scanning for signals for {symbol}: {e}")
        
        return new_signals
    
    def _combine_signals(self, symbol, data, technical_buy, technical_sell, news_signal, earnings_signal, atr):
        """
        Combine different signal sources into a final trading signal
        
        Args:
            symbol (str): Trading symbol
            data (pd.DataFrame): Market data
            technical_buy (bool): Technical buy signal
            technical_sell (bool): Technical sell signal
            news_signal (dict): Signal from news strategy
            earnings_signal (dict): Signal from earnings strategy
            atr (float): ATR value for position sizing
            
        Returns:
            dict: Combined signal information
        """
        # Start with neutral signal
        signal = 'neutral'
        confidence = 0.0
        reasoning = []
        
        # Technical signal
        if technical_buy:
            signal = 'buy'
            confidence = 0.6  # Base confidence for technical signal
            reasoning.append("Technical breakout detected")
        elif technical_sell:
            signal = 'sell'
            confidence = 0.6  # Base confidence for technical signal
            reasoning.append("Technical breakdown detected")
        
        # News signal
        if news_signal['confidence'] > 0.5:
            if signal == 'neutral':
                # No technical signal yet, use news
                signal = news_signal['signal']
                confidence = news_signal['confidence']
                reasoning.append(f"News sentiment: {news_signal['reasoning']}")
            elif news_signal['signal'] == signal:
                # News confirms technical, boost confidence
                confidence += self.config['multi_signal_boost']
                reasoning.append(f"News confirms: {news_signal['reasoning']}")
            else:
                # News contradicts technical, reduce confidence
                confidence -= self.config['multi_signal_boost']
                reasoning.append(f"News contradicts: {news_signal['reasoning']}")
        
        # Earnings signal has highest priority
        if earnings_signal['confidence'] > 0.6:
            if earnings_signal['event_detected']:
                # Strong earnings signal overrides everything
                signal = earnings_signal['signal']
                confidence = earnings_signal['confidence']
                reasoning = [f"Earnings event: {earnings_signal['reasoning']}"]
            elif signal != 'neutral':
                # No strong earnings event but use it to adjust confidence
                if earnings_signal['signal'] == signal:
                    # Earnings confirms, boost confidence
                    confidence += self.config['multi_signal_boost']
                    reasoning.append(f"Earnings outlook confirms: {earnings_signal['reasoning']}")
                else:
                    # Earnings contradicts, reduce confidence
                    confidence -= self.config['multi_signal_boost']
                    reasoning.append(f"Earnings outlook contradicts: {earnings_signal['reasoning']}")
        
        # Cap confidence
        confidence = min(confidence, 0.95)
        
        # Calculate entry price, stop loss and targets
        current_price = data.iloc[-1]['close']
        
        if signal == 'buy':
            stop_loss, targets = calculate_targets_and_stop(
                current_price, 'long', atr
            )
            position_size = self._calculate_position_size(symbol, current_price, stop_loss, confidence)
        elif signal == 'sell':
            stop_loss, targets = calculate_targets_and_stop(
                current_price, 'short', atr
            )
            position_size = self._calculate_position_size(symbol, current_price, stop_loss, confidence)
        else:
            stop_loss = None
            targets = None
            position_size = 0
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'reasoning': '; '.join(reasoning),
            'entry_price': current_price if signal != 'neutral' else None,
            'stop_loss': stop_loss,
            'targets': targets,
            'position_size': position_size,
            'atr': atr
        }
    
    def _calculate_position_size(self, symbol, entry_price, stop_loss, confidence):
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            symbol (str): Trading symbol
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            confidence (float): Signal confidence
            
        Returns:
            float: Position size as fraction of portfolio
        """
        # If no risk management or broker, use default sizing
        if self.broker is None or self.risk_manager is None:
            return self.config['max_position_size'] * confidence
        
        # Get account value
        try:
            account_value = self.broker.get_account_value()
        except:
            account_value = 100000  # Default assumption if can't get actual value
        
        # Calculate risk amount (% of portfolio to risk)
        risk_percent = self.config['risk_management']['max_trade_loss'] * confidence
        risk_amount = account_value * (risk_percent / 100)
        
        # Calculate position size based on stop distance
        if entry_price == stop_loss:  # Avoid division by zero
            return 0
            
        stop_distance_percent = abs(entry_price - stop_loss) / entry_price
        position_value = risk_amount / stop_distance_percent
        
        # Convert to position size as fraction of portfolio
        position_size = position_value / account_value
        
        # Cap position size
        max_size = self.config['max_position_size']
        position_size = min(position_size, max_size)
        
        # Apply risk manager limits (e.g., correlation limits)
        if self.risk_manager:
            # This would check for correlated positions and adjust sizing accordingly
            position_size = self.risk_manager.adjust_position_size(symbol, position_size)
        
        return position_size
    
    def execute_signals(self):
        """
        Execute pending trading signals
        
        Returns:
            int: Number of trades executed
        """
        if not self.config['trading_enabled']:
            self.logger.info("Trading is disabled, skipping signal execution")
            return 0
        
        if not self.is_trading_hours():
            self.logger.info("Outside trading hours, skipping signal execution")
            return 0
        
        # Check if we can take more positions
        current_positions = len(self.active_positions)
        remaining_slots = self.config['max_positions'] - current_positions
        
        if remaining_slots <= 0:
            self.logger.info(f"Maximum positions ({self.config['max_positions']}) reached, skipping signal execution")
            return 0
        
        # Sort signals by confidence
        sorted_signals = sorted(
            [(k, v) for k, v in self.active_signals.items() if k not in self.active_positions],
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        executed = 0
        
        # Execute highest confidence signals first
        for symbol, signal_data in sorted_signals:
            if executed >= remaining_slots:
                break
                
            # Skip expired signals
            signal_age = datetime.now() - signal_data['timestamp']
            if signal_age > timedelta(hours=self.config['signal_aging_hours']):
                self.logger.info(f"Signal for {symbol} expired, skipping execution")
                del self.active_signals[symbol]
                continue
            
            # Skip neutral signals
            if signal_data['signal'] == 'neutral':
                continue
            
            # Execute the trade
            try:
                # This would be implemented based on the specific broker API
                if self.broker is not None:
                    result = self.execute_trade(
                        symbol=symbol,
                        direction=signal_data['signal'],
                        quantity=self._calculate_order_quantity(symbol, signal_data),
                        stop_loss=signal_data['stop_loss'],
                        targets=signal_data['targets']
                    )
                    
                    if result['success']:
                        self.active_positions[symbol] = {
                            'symbol': symbol,
                            'direction': signal_data['signal'],
                            'entry_price': result.get('entry_price', signal_data['entry_price']),
                            'quantity': result['quantity'],
                            'stop_loss': signal_data['stop_loss'],
                            'targets': signal_data['targets'],
                            'entry_time': datetime.now(),
                            'order_ids': result.get('order_ids', []),
                            'reasoning': signal_data['reasoning']
                        }
                        
                        executed += 1
                        self.logger.info(f"Executed {signal_data['signal']} for {symbol} at {result.get('entry_price', 'market price')}")
            except Exception as e:
                self.logger.error(f"Error executing trade for {symbol}: {e}")
        
        return executed
    
    def execute_trade(self, symbol, direction, quantity, stop_loss, targets):
        """
        Execute a trade through the broker API
        
        Args:
            symbol (str): Trading symbol
            direction (str): 'buy' or 'sell'
            quantity (float): Number of shares/contracts
            stop_loss (float): Stop loss price
            targets (list): Take profit target prices
            
        Returns:
            dict: Trade execution result
        """
        if self.broker is None:
            return {'success': False, 'message': 'No broker connected'}
        
        try:
            # This would be implemented based on the specific broker API
            if direction == 'buy':
                # Long position
                result = self.broker.open_long_position(
                    symbol=symbol,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=targets[0] if targets else None
                )
            else:
                # Short position
                result = self.broker.open_short_position(
                    symbol=symbol,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=targets[0] if targets else None
                )
                
            return {
                'success': True,
                'entry_price': result.get('entry_price'),
                'quantity': result.get('quantity'),
                'order_ids': result.get('order_ids', [])
            }
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {'success': False, 'message': str(e)}
    
    def _calculate_order_quantity(self, symbol, signal_data):
        """
        Calculate order quantity based on position size and current price
        
        Args:
            symbol (str): Trading symbol
            signal_data (dict): Signal data including position size
            
        Returns:
            float: Order quantity
        """
        if self.broker is None:
            return 0
            
        try:
            # Get account value
            account_value = self.broker.get_account_value()
            
            # Calculate position value
            position_value = account_value * signal_data['position_size']
            
            # Get current price
            current_price = signal_data['entry_price']
            
            # Calculate quantity
            quantity = position_value / current_price
            
            # Round to appropriate number of shares/contracts
            quantity = round(quantity, 0)
            
            return max(1, quantity)  # Minimum 1 share/contract
            
        except Exception as e:
            self.logger.error(f"Error calculating order quantity: {e}")
            return 0
    
    def manage_positions(self):
        """
        Manage open positions (trailing stops, partial profits, etc.)
        
        Returns:
            int: Number of positions updated
        """
        if not self.config['trading_enabled'] or self.broker is None:
            return 0
            
        if not self.is_trading_hours():
            return 0
        
        updated = 0
        positions_to_remove = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                current_price = self.broker.get_current_price(symbol)
                
                if current_price is None:
                    continue
                
                # Check if position is still open
                position_status = self.broker.get_position_status(position.get('order_ids', []))
                
                if position_status == 'closed':
                    positions_to_remove.append(symbol)
                    self.logger.info(f"Position for {symbol} closed, removing from tracking")
                    continue
                
                # Update trailing stops if needed
                direction = position['direction']
                entry_price = position['entry_price']
                current_stop = position['stop_loss']
                
                # For long positions
                if direction == 'buy':
                    # Check if we hit any target
                    targets_hit = 0
                    for i, target in enumerate(position['targets']):
                        if current_price >= target:
                            targets_hit = i + 1
                    
                    # If we hit at least first target and trailing stop is enabled
                    if targets_hit > 0 and self.config['profit_taking']['trailing_stop_after_target1']:
                        # Calculate new trailing stop
                        atr = position.get('atr', 0)
                        trailing_stop = current_price * (1 - self.config['profit_taking']['trailing_stop_percent']/100)
                        
                        # Only move stop up, never down
                        if trailing_stop > current_stop:
                            # Update stop loss order
                            result = self.broker.update_stop_loss(
                                position.get('order_ids', []), 
                                new_stop=trailing_stop
                            )
                            
                            if result.get('success'):
                                position['stop_loss'] = trailing_stop
                                updated += 1
                                self.logger.info(f"Updated trailing stop for {symbol} to {trailing_stop:.2f}")
                
                # For short positions
                elif direction == 'sell':
                    # Check if we hit any target
                    targets_hit = 0
                    for i, target in enumerate(position['targets']):
                        if current_price <= target:
                            targets_hit = i + 1
                    
                    # If we hit at least first target and trailing stop is enabled
                    if targets_hit > 0 and self.config['profit_taking']['trailing_stop_after_target1']:
                        # Calculate new trailing stop
                        atr = position.get('atr', 0)
                        trailing_stop = current_price * (1 + self.config['profit_taking']['trailing_stop_percent']/100)
                        
                        # Only move stop down, never up
                        if trailing_stop < current_stop:
                            # Update stop loss order
                            result = self.broker.update_stop_loss(
                                position.get('order_ids', []), 
                                new_stop=trailing_stop
                            )
                            
                            if result.get('success'):
                                position['stop_loss'] = trailing_stop
                                updated += 1
                                self.logger.info(f"Updated trailing stop for {symbol} to {trailing_stop:.2f}")
            
            except Exception as e:
                self.logger.error(f"Error managing position for {symbol}: {e}")
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del self.active_positions[symbol]
        
        return updated
    
    def cleanup_expired_signals(self):
        """
        Remove expired signals from tracking
        
        Returns:
            int: Number of signals removed
        """
        removed = 0
        symbols_to_remove = []
        
        for symbol, signal in self.active_signals.items():
            signal_age = datetime.now() - signal['timestamp']
            if signal_age > timedelta(hours=self.config['signal_aging_hours']):
                symbols_to_remove.append(symbol)
                removed += 1
        
        for symbol in symbols_to_remove:
            del self.active_signals[symbol]
        
        return removed
    
    def run_trading_cycle(self):
        """
        Run a complete trading cycle (scan, execute, manage)
        
        Returns:
            dict: Summary of actions taken
        """
        self.logger.info("Starting trading cycle")
        
        # Cleanup old signals
        removed = self.cleanup_expired_signals()
        if removed > 0:
            self.logger.info(f"Removed {removed} expired signals")
        
        # Scan for new signals
        new_signals = self.scan_for_signals()
        self.logger.info(f"Found {len(new_signals)} new trading signals")
        
        # Execute signals
        executed = self.execute_signals()
        self.logger.info(f"Executed {executed} new trades")
        
        # Manage positions
        updated = self.manage_positions()
        self.logger.info(f"Updated {updated} existing positions")
        
        self.logger.info("Trading cycle completed")
        
        return {
            'signals_removed': removed,
            'signals_found': len(new_signals),
            'trades_executed': executed,
            'positions_updated': updated,
            'active_positions': len(self.active_positions),
            'active_signals': len(self.active_signals)
        }
    
    def run(self):
        """
        Main execution loop for the auto trader
        """
        self.logger.info("Starting auto trader")
        
        try:
            while True:
                if not self.is_trading_hours():
                    self.logger.info("Outside trading hours, waiting...")
                    time.sleep(60)  # Check every minute outside trading hours
                    continue
                
                # Run a complete trading cycle
                self.run_trading_cycle()
                
                # Wait for next scan
                time.sleep(self.config['scan_interval_seconds'])
                
        except KeyboardInterrupt:
            self.logger.info("Auto trader stopped by user")
        except Exception as e:
            self.logger.error(f"Error in auto trader main loop: {e}")
        finally:
            self.logger.info("Auto trader shutdown complete")
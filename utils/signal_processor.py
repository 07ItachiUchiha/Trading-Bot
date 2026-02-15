import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

class SignalProcessor:
    """Combines technical, sentiment, and earnings signals into one decision."""
    
    def __init__(self, config=None):
        # default config
        self.config = {
            'min_confidence': 0.7,
            'signal_aging_hours': 2,
            'multi_signal_boost': 0.1,
            'signal_weights': {
                'technical': 0.5,
                'sentiment': 0.3,
                'earnings': 0.2
            }
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # signal tracking
        self.active_signals = {}
        
        self.logger = logging.getLogger('SignalProcessor')
    
    def process_signals(self, symbol, technical_signal=None, sentiment_signal=None, 
                       earnings_signal=None, price_data=None):
        """Merge all signal sources into a single buy/sell/neutral with confidence."""
        # Start with neutral signal
        combined_signal = {
            'symbol': symbol,
            'signal': 'neutral',
            'confidence': 0.0,
            'reasoning': [],
            'timestamp': datetime.now()
        }
        
        signals_received = []
        signal_weights = []
        signal_confidences = []
        
        # Process technical signal
        if technical_signal and 'signal' in technical_signal:
            signals_received.append(technical_signal['signal'])
            signal_weights.append(self.config['signal_weights']['technical'])
            signal_confidences.append(technical_signal.get('confidence', 0.6))
            
            combined_signal['reasoning'].append(
                f"Technical: {technical_signal['signal'].upper()} "
                f"({technical_signal.get('reasoning', 'technical pattern')})"
            )
        
        # Process sentiment signal
        if sentiment_signal and 'signal' in sentiment_signal:
            signals_received.append(sentiment_signal['signal'])
            signal_weights.append(self.config['signal_weights']['sentiment'])
            signal_confidences.append(sentiment_signal.get('confidence', 0.5))
            
            news_count = sentiment_signal.get('news_count', 0)
            source_count = sentiment_signal.get('source_count', 0)
            combined_signal['reasoning'].append(
                f"News sentiment: {sentiment_signal['signal'].upper()} "
                f"(score: {sentiment_signal.get('score', 0):.2f}, "
                f"{news_count} stories from {source_count} sources)"
            )
        
        # Process earnings signal
        if earnings_signal and 'signal' in earnings_signal:
            signals_received.append(earnings_signal['signal'])
            signal_weights.append(self.config['signal_weights']['earnings'])
            signal_confidences.append(earnings_signal.get('confidence', 0.7))
            
            combined_signal['reasoning'].append(
                f"Earnings: {earnings_signal['signal'].upper()} "
                f"({earnings_signal.get('reasoning', 'earnings analysis')})"
            )
        
        # If no signals, nothing to do
        if not signals_received:
            return combined_signal
        
        # Calculate signal votes
        buy_weight = sum(w * c for s, w, c in zip(signals_received, signal_weights, signal_confidences) if s == 'buy')
        sell_weight = sum(w * c for s, w, c in zip(signals_received, signal_weights, signal_confidences) if s == 'sell')
        
        # normalize
        total_weight = sum(signal_weights)
        if total_weight > 0:
            if buy_weight > sell_weight:
                combined_signal['signal'] = 'buy'
                combined_signal['confidence'] = buy_weight / total_weight
            elif sell_weight > buy_weight:
                combined_signal['signal'] = 'sell'
                combined_signal['confidence'] = sell_weight / total_weight
            else:
                combined_signal['signal'] = 'neutral'
                combined_signal['confidence'] = 0.0
        
        # boost if multiple signals agree
        buy_count = sum(1 for s in signals_received if s == 'buy')
        sell_count = sum(1 for s in signals_received if s == 'sell')
        
        if buy_count > 1 and combined_signal['signal'] == 'buy':
            boost = self.config['multi_signal_boost'] * (buy_count - 1)
            combined_signal['confidence'] = min(0.95, combined_signal['confidence'] + boost)
            combined_signal['reasoning'].append(f"{buy_count} confirming BUY signals")
        
        if sell_count > 1 and combined_signal['signal'] == 'sell':
            boost = self.config['multi_signal_boost'] * (sell_count - 1)
            combined_signal['confidence'] = min(0.95, combined_signal['confidence'] + boost)
            combined_signal['reasoning'].append(f"{sell_count} confirming SELL signals")
        
        # Format reasoning
        combined_signal['reasoning'] = "; ".join(combined_signal['reasoning'])
        
        # track it
        self.active_signals[symbol] = combined_signal
        
        return combined_signal
    
    def get_all_active_signals(self):
        """Return all non-expired signals."""
        self.cleanup_expired_signals()
        return self.active_signals
    
    def cleanup_expired_signals(self):
        """Drop signals older than the aging threshold."""
        removed = 0
        current_time = datetime.now()
        symbols_to_remove = []
        
        for symbol, signal in self.active_signals.items():
            signal_age = current_time - signal['timestamp']
            if signal_age > timedelta(hours=self.config['signal_aging_hours']):
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.active_signals[symbol]
            removed += 1
        
        return removed

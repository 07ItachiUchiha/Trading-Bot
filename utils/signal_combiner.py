import numpy as np
import pandas as pd
from datetime import datetime
import logging

class SignalCombiner:
    """
    Combines technical and sentiment signals to produce more reliable trading signals
    """
    
    def __init__(self, technical_weight=0.6, sentiment_weight=0.4):
        """
        Initialize signal combiner with configurable weights
        
        Args:
            technical_weight (float): Weight for technical signals (0-1)
            sentiment_weight (float): Weight for sentiment signals (0-1)
        """
        # Ensure weights sum to 1
        total = technical_weight + sentiment_weight
        self.technical_weight = technical_weight / total
        self.sentiment_weight = sentiment_weight / total
        
        # Initialize logger
        self.logger = logging.getLogger('SignalCombiner')
    
    def combine_signals(self, technical_signals, sentiment_signals):
        """
        Combine technical and sentiment signals
        
        Args:
            technical_signals (dict): Dictionary of technical signals (-1 to 1 scale)
            sentiment_signals (dict): Dictionary of sentiment signals (-1 to 1 scale)
        
        Returns:
            dict: Combined signals with confidence scores
        """
        combined_signals = {}
        
        # Process each symbol
        for symbol in technical_signals.keys():
            if symbol not in sentiment_signals:
                # If we only have technical signals, use those
                combined_signals[symbol] = {
                    'signal': technical_signals[symbol]['signal'],
                    'strength': technical_signals[symbol]['strength'],
                    'confidence': technical_signals[symbol].get('confidence', 0.6),
                    'source': 'technical_only'
                }
                continue
                
            # Get signals and strengths
            tech_signal = technical_signals[symbol]['signal']
            tech_strength = technical_signals[symbol]['strength']
            sent_signal = sentiment_signals[symbol]['signal']
            sent_strength = sentiment_signals[symbol]['strength']
            
            # Convert string signals to numeric if needed
            tech_value = self._signal_to_value(tech_signal, tech_strength)
            sent_value = self._signal_to_value(sent_signal, sent_strength)
            
            # Calculate weighted average
            combined_value = (tech_value * self.technical_weight) + (sent_value * self.sentiment_weight)
            
            # Convert back to signal and strength
            combined_signal, combined_strength = self._value_to_signal(combined_value)
            
            # Calculate confidence
            tech_conf = technical_signals[symbol].get('confidence', 0.6)
            sent_conf = sentiment_signals[symbol].get('confidence', 0.5)
            
            # Higher confidence when signals agree, lower when they conflict
            if tech_signal == sent_signal:
                combined_conf = (tech_conf * self.technical_weight + sent_conf * self.sentiment_weight) * 1.2
            else:
                combined_conf = (tech_conf * self.technical_weight + sent_conf * self.sentiment_weight) * 0.8
                
            # Cap confidence at 0.95
            combined_conf = min(0.95, combined_conf)
            
            # Store combined signal
            combined_signals[symbol] = {
                'signal': combined_signal,
                'strength': combined_strength,
                'confidence': combined_conf,
                'technical_signal': tech_signal,
                'sentiment_signal': sent_signal,
                'source': 'combined'
            }
        
        return combined_signals
    
    def _signal_to_value(self, signal, strength):
        """Convert signal string and strength to numeric value (-1 to 1)"""
        if isinstance(signal, (int, float)):
            return signal  # Already numeric
            
        signal_map = {
            'strong buy': 1.0,
            'buy': 0.5,
            'hold': 0.0,
            'sell': -0.5,
            'strong sell': -1.0
        }
        
        base_value = signal_map.get(signal.lower(), 0)
        # Adjust by strength (0-1)
        return base_value * min(1.0, max(0.1, strength))
    
    def _value_to_signal(self, value):
        """Convert numeric value to signal string and strength"""
        abs_value = abs(value)
        strength = min(1.0, abs_value * 1.5)  # Scale up strength for better readability
        
        if value > 0.7:
            return 'strong buy', strength
        elif value > 0.2:
            return 'buy', strength
        elif value < -0.7:
            return 'strong sell', strength
        elif value < -0.2:
            return 'sell', strength
        else:
            return 'hold', abs_value

def get_combined_signals(symbol, technical_data=None, sentiment_data=None):
    """
    Helper function to get combined signals for the UI
    
    Args:
        symbol (str): Trading symbol
        technical_data (dict): Technical signals dict
        sentiment_data (dict): Sentiment signals dict
    
    Returns:
        dict: Combined signals with metadata
    """
    # Create default data if not provided (for demo purposes)
    if technical_data is None:
        technical_data = {
            symbol: {
                'signal': 'buy',
                'strength': 0.65,
                'confidence': 0.72,
                'indicators': {
                    'rsi': 42,
                    'macd': 'bullish',
                    'sma': 'uptrend'
                }
            }
        }
    
    if sentiment_data is None:
        # Base sentiment on time of day for demo variation
        hour = datetime.now().hour
        if hour < 12:
            sentiment = 'positive'
            signal = 'buy'
            strength = 0.6
        else:
            sentiment = 'negative'
            signal = 'sell'
            strength = 0.7
            
        sentiment_data = {
            symbol: {
                'signal': signal,
                'strength': strength,
                'confidence': 0.68,
                'sentiment': sentiment,
                'news_count': 7
            }
        }
    
    # Combine signals
    combiner = SignalCombiner()
    return combiner.combine_signals(technical_data, sentiment_data)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from utils.enhanced_sentiment import EnhancedSentimentAnalyzer

class NewsTradingAdvisor:
    """
    Provides trading recommendations based on news sentiment 
    and market analysis, without auto trading functionality
    """
    
    def __init__(self, api_keys=None):
        """Initialize the news trading advisor"""
        self.sentiment_analyzer = EnhancedSentimentAnalyzer(api_keys)
        self.logger = logging.getLogger('NewsTradingAdvisor')
        self.tracked_symbols = set()
        self.analysis_cache = {}
        
    def track_symbol(self, symbol):
        """Add a symbol to the tracking list"""
        self.tracked_symbols.add(symbol)
        
    def untrack_symbol(self, symbol):
        """Remove a symbol from the tracking list"""
        if symbol in self.tracked_symbols:
            self.tracked_symbols.remove(symbol)
            
    def get_recommendation(self, symbol, force_refresh=False):
        """Get a trading recommendation for a symbol"""
        if force_refresh or symbol not in self.analysis_cache:
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(symbol)
            self.analysis_cache[symbol] = {
                'timestamp': datetime.now(),
                'data': self._generate_recommendation(symbol, sentiment_result)
            }
        elif (datetime.now() - self.analysis_cache[symbol]['timestamp']).total_seconds() > 1800:
            # Refresh if older than 30 minutes
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(symbol)
            self.analysis_cache[symbol] = {
                'timestamp': datetime.now(),
                'data': self._generate_recommendation(symbol, sentiment_result)
            }
            
        return self.analysis_cache[symbol]['data']
    
    def _generate_recommendation(self, symbol, sentiment_result):
        """Generate a detailed trading recommendation based on sentiment analysis"""
        signal = sentiment_result['signal']
        sentiment_score = sentiment_result['sentiment_score']
        confidence = sentiment_result['confidence']
        
        # Determine trade action
        if signal == 'strong buy':
            action = "BUY"
            urgency = "High"
            risk_level = "Medium"
        elif signal == 'buy':
            action = "BUY"
            urgency = "Medium"
            risk_level = "Medium"
        elif signal == 'hold':
            action = "HOLD"
            urgency = "Low"
            risk_level = "Low"
        elif signal == 'sell':
            action = "SELL"
            urgency = "Medium" 
            risk_level = "Medium"
        elif signal == 'strong sell':
            action = "SELL"
            urgency = "High"
            risk_level = "Medium"
        else:
            action = "HOLD"
            urgency = "Low"
            risk_level = "Low"
            
        # Adjust risk level based on confidence
        if confidence < 0.6:
            risk_level = "High"
        elif confidence > 0.8:
            risk_level = "Low"
            
        # Calculate recommended position size
        if confidence > 0.8:
            position_size = "3-5% of portfolio"
        elif confidence > 0.65:
            position_size = "1-3% of portfolio"
        else:
            position_size = "Up to 1% of portfolio"
            
        # Custom recommendation text
        if action == "BUY":
            recommendation = f"Consider opening a {position_size} position in {symbol}. "
            recommendation += f"News sentiment is {sentiment_result['sentiment']} with {confidence*100:.1f}% confidence. "
            
            if urgency == "High":
                recommendation += "Current sentiment suggests strong upward momentum."
            else:
                recommendation += "Current sentiment is favorable."
                
        elif action == "SELL":
            recommendation = f"Consider reducing or closing positions in {symbol}. "
            recommendation += f"News sentiment is {sentiment_result['sentiment']} with {confidence*100:.1f}% confidence. "
            
            if urgency == "High":
                recommendation += "Current sentiment suggests strong downward pressure."
            else:
                recommendation += "Current sentiment is unfavorable."
                
        else:  # HOLD
            recommendation = f"No clear action needed for {symbol} at this time. "
            recommendation += f"Current sentiment is {sentiment_result['sentiment']} with {confidence*100:.1f}% confidence. "
            recommendation += "Monitor for changing conditions."
            
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'urgency': urgency,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'position_size': position_size,
            'timestamp': datetime.now().isoformat(),
            'sentiment_data': sentiment_result,
            'analysis_summary': sentiment_result['analysis']
        }
        
    def get_all_tracked_recommendations(self):
        """Get recommendations for all tracked symbols"""
        results = {}
        for symbol in self.tracked_symbols:
            results[symbol] = self.get_recommendation(symbol)
        return results
    
    def process_news_update(self, news_data):
        """Process news updates from webhooks or other sources"""
        self.sentiment_analyzer.process_news_event(news_data)
        
        # Clear cache for affected symbols to force refresh
        if isinstance(news_data, dict):
            if 'symbols' in news_data:
                for symbol in news_data['symbols']:
                    if symbol in self.analysis_cache:
                        del self.analysis_cache[symbol]
            elif 'symbol' in news_data:
                symbol = news_data['symbol']
                if symbol in self.analysis_cache:
                    del self.analysis_cache[symbol]

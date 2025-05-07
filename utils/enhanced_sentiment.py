import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import re
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import threading
import time

# Download NLTK resources if not available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analysis for trading decisions with improved signal generation
    """
    
    def __init__(self, api_keys=None):
        """Initialize the sentiment analyzer with API keys"""
        self.api_keys = {
            'newsapi': None,
            'alphavantage': None,
            'finnhub': None
        }
        
        if api_keys:
            self.api_keys.update(api_keys)
        
        # Initialize sentiment analysis tools
        self.vader = SentimentIntensityAnalyzer()
        
        # Configure custom sentiment dictionary for financial terms
        self.vader.lexicon.update({
            # Positive terms
            'bull': 3.0, 'bullish': 3.0, 'rally': 2.5, 'uptick': 1.5, 
            'outperform': 2.0, 'upbeat': 1.8, 'upgrade': 2.0,
            'buy': 1.5, 'strong buy': 2.5, 'beat': 1.7,
            'growth': 1.5, 'profit': 1.8, 'gain': 1.5,
            
            # Negative terms
            'bear': -3.0, 'bearish': -3.0, 'plunge': -2.5, 'downtick': -1.5,
            'underperform': -2.0, 'downbeat': -1.8, 'downgrade': -2.0,
            'sell': -1.5, 'strong sell': -2.5, 'miss': -1.7,
            'decline': -1.5, 'loss': -1.8, 'recession': -2.5
        })
        
        # Cache for storing sentiment results
        self.sentiment_cache = {}
        
        # Set up logging
        self.logger = logging.getLogger('EnhancedSentimentAnalyzer')
        
        # Tracked symbols and their sentiment history
        self.sentiment_history = {}
        
        # News buffer for events
        self.news_buffer = []
        self.news_lock = threading.Lock()
    
    def fetch_news(self, symbol, days_back=3):
        """Fetch news for a specific symbol from multiple sources"""
        results = []
        
        # Remove USD suffix if present (for crypto)
        search_term = symbol.replace('/USD', '')
        if search_term.endswith('USD'):
            search_term = search_term[:-3]
        
        # Try News API
        if self.api_keys['newsapi']:
            try:
                today = datetime.now()
                from_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                url = f"https://newsapi.org/v2/everything?q={search_term}&from={from_date}&language=en&sortBy=publishedAt&apiKey={self.api_keys['newsapi']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    news_data = response.json()
                    if news_data['status'] == 'ok' and len(news_data['articles']) > 0:
                        for article in news_data['articles']:
                            results.append({
                                'title': article['title'],
                                'description': article['description'],
                                'content': article['content'],
                                'source': article['source']['name'],
                                'published_at': article['publishedAt'],
                                'url': article['url']
                            })
            except Exception as e:
                self.logger.error(f"Error fetching news from News API: {e}")
        
        # Try Alpha Vantage
        if self.api_keys['alphavantage']:
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={search_term}&apikey={self.api_keys['alphavantage']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'feed' in data:
                        for item in data['feed']:
                            results.append({
                                'title': item['title'],
                                'description': item.get('summary', ''),
                                'content': item.get('summary', ''),
                                'source': item['source'],
                                'published_at': item['time_published'],
                                'url': item['url'],
                                'sentiment_score': item.get('overall_sentiment_score', None)
                            })
            except Exception as e:
                self.logger.error(f"Error fetching news from Alpha Vantage: {e}")
        
        # Try Finnhub
        if self.api_keys['finnhub']:
            try:
                today = datetime.now()
                from_timestamp = int((today - timedelta(days=days_back)).timestamp())
                to_timestamp = int(today.timestamp())
                
                url = f"https://finnhub.io/api/v1/company-news?symbol={search_term}&from={from_timestamp}&to={to_timestamp}&token={self.api_keys['finnhub']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        results.append({
                            'title': item['headline'],
                            'description': item.get('summary', ''),
                            'content': item.get('summary', ''),
                            'source': item['source'],
                            'published_at': datetime.fromtimestamp(item['datetime']).isoformat(),
                            'url': item['url']
                        })
            except Exception as e:
                self.logger.error(f"Error fetching news from Finnhub: {e}")
        
        # If no news from APIs, generate mock news for testing
        if not results:
            results = self._generate_mock_news(symbol)
            
        return results
    
    def _generate_mock_news(self, symbol):
        """Generate mock news articles for testing when APIs fail"""
        mock_news = []
        today = datetime.now()
        
        # Define sentiment conditions based on time of day for testing
        hour = today.hour
        if hour < 12:  # Morning - bullish sentiment
            sentiment = "positive"
        else:  # Afternoon/evening - bearish sentiment
            sentiment = "negative"
            
        search_term = symbol.replace('/USD', '')
        if search_term.endswith('USD'):
            search_term = search_term[:-3]
            
        # Generate mock articles with appropriate sentiment
        if sentiment == "positive":
            headlines = [
                f"{search_term} poised for growth amid positive market trends",
                f"Analysts upgrade {search_term}, citing strong fundamentals",
                f"New developments could boost {search_term} price, experts say"
            ]
            
            content = [
                f"Market analysts are bullish on {search_term} following recent developments. Technical indicators suggest a potential breakout.",
                f"Several major investment firms have upgraded their outlook on {search_term}, citing improved market conditions and strong growth potential.",
                f"Recent developments in the {search_term} ecosystem point to increased adoption and value, according to industry experts."
            ]
        else:
            headlines = [
                f"{search_term} faces pressure as markets remain uncertain",
                f"Analysts cautious on {search_term} amid broader market concerns",
                f"Regulatory challenges could impact {search_term}, report suggests"
            ]
            
            content = [
                f"Market sentiment has turned bearish on {search_term} as broader economic concerns impact investment outlook.",
                f"Several analysts have expressed caution regarding {search_term}, citing potential market volatility and uncertain growth prospects.",
                f"Emerging regulatory challenges could pose risks for {search_term} in the coming months, according to industry observers."
            ]
            
        # Create mock news items
        for i in range(len(headlines)):
            pub_date = today - timedelta(hours=i*6)  # Space out the articles
            mock_news.append({
                'title': headlines[i],
                'description': content[i][:100] + "...",
                'content': content[i],
                'source': 'Mock Financial News',
                'published_at': pub_date.isoformat(),
                'url': '#'
            })
            
        return mock_news
    
    def analyze_sentiment(self, symbol, news_items=None, days_back=3):
        """
        Analyze sentiment for a symbol based on recent news
        Returns a detailed sentiment analysis with buy/sell signal
        """
        # Check cache first
        cache_key = f"{symbol}_{days_back}"
        current_time = time.time()
        
        # Return cached result if available and not expired (30 minutes)
        if cache_key in self.sentiment_cache:
            timestamp, result = self.sentiment_cache[cache_key]
            if current_time - timestamp < 1800:  # 30 minutes
                return result
        
        # Fetch news if not provided
        if not news_items:
            news_items = self.fetch_news(symbol, days_back)
        
        if not news_items:
            # No news found
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'confidence': 0.3,  # Low confidence due to lack of data
                'signal': 'hold',
                'strength': 0,
                'news_count': 0,
                'analysis': "Insufficient news data for sentiment analysis.",
                'sources': []
            }
        
        # Analyze sentiment for each news item
        article_sentiments = []
        total_score = 0
        sources = set()
        all_text = ""
        
        for item in news_items:
            text = f"{item['title']} {item.get('description', '')}"
            all_text += text + " "
            sources.add(item['source'])
            
            if 'sentiment_score' in item and item['sentiment_score'] is not None:
                # Use pre-calculated sentiment if available
                sentiment_score = item['sentiment_score']
            else:
                # Calculate sentiment using VADER
                vader_score = self.vader.polarity_scores(text)
                sentiment_score = vader_score['compound']
            
            # Calculate recency weight (newer articles have more weight)
            try:
                published_at = item.get('published_at')
                if published_at:
                    if isinstance(published_at, str):
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    else:
                        pub_date = published_at
                        
                    days_old = (datetime.now() - pub_date).total_seconds() / 86400
                    recency_weight = max(0.5, 1.0 - (days_old / (days_back * 2)))
                else:
                    recency_weight = 0.75  # Default weight if no date
            except Exception:
                recency_weight = 0.75
            
            article_sentiments.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'sentiment_score': sentiment_score,
                'recency_weight': recency_weight
            })
            
            total_score += sentiment_score * recency_weight
        
        # Calculate overall sentiment
        if article_sentiments:
            avg_sentiment = total_score / len(article_sentiments)
        else:
            avg_sentiment = 0
        
        # Track sentiment history
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []
            
        self.sentiment_history[symbol].append({
            'timestamp': datetime.now(),
            'sentiment': avg_sentiment
        })
        
        # Keep only last 10 days of history
        self.sentiment_history[symbol] = self.sentiment_history[symbol][-10:]
        
        # Calculate sentiment trend
        sentiment_trend = 0
        history = self.sentiment_history[symbol]
        if len(history) >= 2:
            sentiment_trend = history[-1]['sentiment'] - history[0]['sentiment']
        
        # Determine sentiment category and signal
        if avg_sentiment >= 0.5:
            sentiment = 'very positive'
            signal = 'strong buy'
            strength = 1.0
            confidence = min(0.9, 0.7 + (avg_sentiment - 0.5) * 0.4)
        elif avg_sentiment >= 0.2:
            sentiment = 'positive'
            signal = 'buy'
            strength = 0.7
            confidence = min(0.85, 0.6 + (avg_sentiment - 0.2) * 0.5)
        elif avg_sentiment > -0.2:
            sentiment = 'neutral'
            
            # Consider sentiment trend for neutral overall sentiment
            if sentiment_trend > 0.2:
                signal = 'buy'
                strength = 0.3
                confidence = 0.6
            elif sentiment_trend < -0.2:
                signal = 'sell'
                strength = 0.3
                confidence = 0.6
            else:
                signal = 'hold'
                strength = 0
                confidence = 0.5
        elif avg_sentiment >= -0.5:
            sentiment = 'negative'
            signal = 'sell'
            strength = 0.7
            confidence = min(0.85, 0.6 + (abs(avg_sentiment) - 0.2) * 0.5)
        else:
            sentiment = 'very negative'
            signal = 'strong sell'
            strength = 1.0
            confidence = min(0.9, 0.7 + (abs(avg_sentiment) - 0.5) * 0.4)
            
        # Adjust confidence based on news count
        news_count = len(news_items)
        if news_count < 3:
            confidence *= 0.7  # Reduced confidence with few news items
        elif news_count > 10:
            confidence = min(0.95, confidence * 1.1)  # Higher confidence with many news items
        
        # Create detailed result
        result = {
            'symbol': symbol,
            'sentiment_score': avg_sentiment,
            'sentiment': sentiment,
            'confidence': confidence,
            'signal': signal,
            'strength': strength * (1 if signal in ['buy', 'strong buy'] else -1),
            'news_count': news_count,
            'sentiment_trend': sentiment_trend,
            'article_sentiments': article_sentiments[:5],  # Top 5 articles
            'analysis': self._generate_analysis_summary(avg_sentiment, news_count, sentiment_trend, signal),
            'sources': list(sources)
        }
        
        # Cache the result
        self.sentiment_cache[cache_key] = (current_time, result)
        
        return result
    
    def _generate_analysis_summary(self, sentiment_score, news_count, trend, signal):
        """Generate a human-readable analysis summary"""
        if news_count == 0:
            return "Insufficient news data available for analysis."
        
        summary = f"Based on analysis of {news_count} recent news articles, "
        
        if sentiment_score > 0.5:
            summary += "market sentiment is extremely positive. "
        elif sentiment_score > 0.2:
            summary += "market sentiment is positive. "
        elif sentiment_score > -0.2:
            summary += "market sentiment is relatively neutral. "
        elif sentiment_score > -0.5:
            summary += "market sentiment is negative. "
        else:
            summary += "market sentiment is extremely negative. "
            
        if abs(trend) > 0.3:
            if trend > 0:
                summary += "Sentiment is improving significantly over time. "
            else:
                summary += "Sentiment is deteriorating over time. "
        elif abs(trend) > 0.1:
            if trend > 0:
                summary += "Sentiment is showing slight improvement. "
            else:
                summary += "Sentiment is showing slight deterioration. "
        else:
            summary += "Sentiment remains stable. "
            
        # Add signal recommendation
        if signal == 'strong buy':
            summary += "Strong buy signal based on highly positive sentiment."
        elif signal == 'buy':
            summary += "Consider buying based on positive market sentiment."
        elif signal == 'hold':
            summary += "Consider holding current positions until clearer signals emerge."
        elif signal == 'sell':
            summary += "Consider selling based on negative market sentiment."
        elif signal == 'strong sell':
            summary += "Strong sell signal based on highly negative sentiment."
            
        return summary
    
    def process_news_event(self, news_data):
        """Process incoming news events from webhooks or other sources"""
        with self.news_lock:
            self.news_buffer.append({
                'timestamp': datetime.now(),
                'data': news_data
            })
            
        # Process the latest news for any tracked symbols
        self._process_news_buffer()
        
    def _process_news_buffer(self):
        """Process news buffer for signals"""
        with self.news_lock:
            if not self.news_buffer:
                return
                
            # Process the latest news
            latest_news = self.news_buffer[-1]['data']
            
            # Clear old news (keep last 100 items maximum)
            if len(self.news_buffer) > 100:
                self.news_buffer = self.news_buffer[-100:]
                
        # Extract symbols and relevant information
        if isinstance(latest_news, dict):
            symbols = set()
            
            # Extract potential symbols from the news
            if 'symbols' in latest_news:
                symbols.update(latest_news['symbols'])
            elif 'symbol' in latest_news:
                symbols.add(latest_news['symbol'])
                
            # For any extracted symbols, update sentiment
            for symbol in symbols:
                if symbol in self.sentiment_history:
                    self.analyze_sentiment(symbol, days_back=1)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import time
from utils.news_fetcher import NewsFetcher

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Download more NLTK resources for topic analysis if needed
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class NewsBasedStrategy:
    """
    A trading strategy that uses news sentiment analysis to generate trading signals.
    
    This strategy:
    1. Fetches recent news for a given asset
    2. Analyzes the sentiment of the news
    3. Generates trading signals based on sentiment and keywords
    4. Can be combined with technical strategies for confirmation
    5. Detects emerging news trends
    """
    
    def __init__(self, config=None):
        """
        Initialize the news-based trading strategy
        
        Args:
            config (dict): Strategy configuration
        """
        # Default configuration
        self.config = {
            'news_lookback_days': 1,            # Days to look back for news
            'sentiment_threshold_buy': 0.2,     # Minimum sentiment score to consider buying
            'sentiment_threshold_sell': -0.15,  # Maximum sentiment score to consider selling
            'min_news_count': 3,                # Minimum number of news items required
            'use_keyword_boost': True,          # Boost sentiment based on keywords
            'api_keys': {
                'news_api': None,
                'alphavantage': None,
                'finnhub': None
            },
            'cache_expiry_minutes': 30,         # Cache news for 30 minutes
            'use_trend_detection': True,        # Enable trend detection in news
            'trend_weight': 0.3,                # Weight to give to trending topics
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
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Keywords that might indicate positive or negative news
        self.positive_keywords = [
            'surge', 'jump', 'soar', 'rise', 'gain', 'bull', 'bullish', 'breakthrough',
            'record high', 'outperform', 'beat', 'exceed', 'positive', 'strong', 
            'upgrade', 'buy', 'recommend', 'partnership', 'collaboration', 'launch', 
            'approval', 'patent', 'success', 'profit', 'growth'
        ]
        
        self.negative_keywords = [
            'crash', 'plunge', 'tumble', 'fall', 'drop', 'decline', 'bear', 'bearish',
            'record low', 'underperform', 'miss', 'below', 'negative', 'weak',
            'downgrade', 'sell', 'avoid', 'violation', 'lawsuit', 'investigation', 
            'recall', 'failure', 'loss', 'bankruptcy', 'debt', 'fine', 'penalty'
        ]
        
        # Extra weight for company/crypto-specific keywords
        self.company_specific_keywords = {}
        
        # Stop words for trend analysis
        self.stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.data.path else set()
    
    def set_company_keywords(self, symbol, positive_words=None, negative_words=None):
        """
        Set company-specific keywords that might impact trading decisions
        
        Args:
            symbol (str): The trading symbol (e.g., AAPL, BTCUSD)
            positive_words (list): List of positive keywords specific to this symbol
            negative_words (list): List of negative keywords specific to this symbol
        """
        if not positive_words:
            positive_words = []
            
        if not negative_words:
            negative_words = []
            
        self.company_specific_keywords[symbol] = {
            'positive': positive_words,
            'negative': negative_words
        }
    
    def generate_signal(self, symbol, current_price=None, technical_signal=None):
        """
        Generate a trading signal based on news sentiment
        
        Args:
            symbol (str): Trading symbol (e.g., AAPL, BTCUSD)
            current_price (float): Current price of the asset (optional)
            technical_signal (str): Signal from technical analysis (optional)
            
        Returns:
            dict: Trading signal with the following keys:
                - 'signal': 'buy', 'sell', or 'neutral'
                - 'confidence': Signal confidence score (0-1)
                - 'reasoning': Explanation for the signal
                - 'news_count': Number of news items analyzed
                - 'sentiment_score': Overall sentiment score
                - 'top_headlines': List of top headlines influencing the decision
                - 'trending_topics': List of trending topics (if enabled)
        """
        # Fetch news for the symbol with automatic sentiment analysis
        news = self.news_fetcher.get_news_for_symbol(
            symbol, 
            days=self.config['news_lookback_days'],
            analyze_sentiment=True
        )
        
        # If not enough news, return neutral signal
        if len(news) < self.config['min_news_count']:
            return {
                'signal': 'neutral',
                'confidence': 0,
                'reasoning': f'Not enough news items found ({len(news)} < {self.config["min_news_count"]})',
                'news_count': len(news),
                'sentiment_score': 0,
                'top_headlines': [item['title'] for item in news[:3]]
            }
            
        # Apply keyword boost if enabled
        if self.config['use_keyword_boost']:
            for item in news:
                sentiment = item.get('sentiment', 0)
                item['sentiment'] = self._apply_keyword_boost(item, symbol, sentiment)
        
        # Calculate overall sentiment score (weighted by recency)
        total_weight = 0
        weighted_sentiment = 0
        
        for i, item in enumerate(news):
            # More recent news gets higher weight
            weight = len(news) - i
            weighted_sentiment += item.get('sentiment', 0) * weight
            total_weight += weight
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Detect trending topics if enabled
        trending_topics = []
        if self.config['use_trend_detection']:
            trending_topics = self._detect_trending_topics(news)
            
            # Adjust sentiment based on trending topics
            if trending_topics:
                trend_sentiment = self._analyze_trend_sentiment(trending_topics, news)
                # Blend the trend sentiment with the overall sentiment
                overall_sentiment = (overall_sentiment * (1 - self.config['trend_weight']) + 
                                    trend_sentiment * self.config['trend_weight'])
        
        # Generate signal based on sentiment
        if overall_sentiment >= self.config['sentiment_threshold_buy']:
            signal = 'buy'
            confidence = min(1.0, overall_sentiment * 2)  # Scale to 0-1
            reasoning = f"Positive news sentiment ({overall_sentiment:.2f})"
        elif overall_sentiment <= self.config['sentiment_threshold_sell']:
            signal = 'sell'
            confidence = min(1.0, abs(overall_sentiment) * 2)  # Scale to 0-1
            reasoning = f"Negative news sentiment ({overall_sentiment:.2f})"
        else:
            signal = 'neutral'
            confidence = 0.5 - abs(overall_sentiment) * 2  # Higher for more neutral
            reasoning = f"Neutral news sentiment ({overall_sentiment:.2f})"
        
        # If technical signal is provided, adjust confidence
        if technical_signal:
            if technical_signal == signal:
                confidence = min(1.0, confidence * 1.5)  # Boost confidence
                reasoning += f" - Confirmed by technical analysis ({technical_signal})"
            else:
                confidence *= 0.7  # Reduce confidence
                reasoning += f" - Contradicted by technical analysis ({technical_signal})"
        
        # Add trending topics to reasoning if available
        if trending_topics:
            trending_str = ", ".join([f"{topic} ({count})" for topic, count in trending_topics[:3]])
            reasoning += f" - Trending topics: {trending_str}"
        
        # Sort news by sentiment (most extreme first)
        sorted_news = sorted(news, key=lambda x: abs(x.get('sentiment', 0)), reverse=True)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'news_count': len(news),
            'sentiment_score': overall_sentiment,
            'top_headlines': [f"{item['title']} (sentiment: {item.get('sentiment', 0):.2f})" for item in sorted_news[:3]],
            'trending_topics': trending_topics[:5] if trending_topics else []
        }
    
    def filter_news_by_category(self, symbol, categories=None, days=1):
        """
        Get news filtered by specific categories
        
        Args:
            symbol (str): Trading symbol
            categories (list): Categories to filter by (e.g., 'earnings', 'merger', 'product')
            days (int): Number of days to look back
            
        Returns:
            dict: Dictionary mapping categories to news items
        """
        if categories is None:
            categories = ['earnings', 'product', 'merger', 'legal', 'management']
            
        news = self.news_fetcher.get_news_for_symbol(symbol, days=days, analyze_sentiment=True)
        
        # Define category keywords
        category_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'financial results'],
            'product': ['product', 'launch', 'release', 'new', 'feature', 'update'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'consolidation'],
            'legal': ['lawsuit', 'legal', 'regulation', 'compliance', 'fine', 'settlement'],
            'management': ['CEO', 'executive', 'board', 'management', 'leadership', 'resign'],
            'partnerships': ['partnership', 'collaborate', 'alliance', 'joint venture'],
            'market': ['market share', 'competition', 'industry', 'sector', 'trend']
        }
        
        filtered_news = {category: [] for category in categories}
        
        for item in news:
            text = f"{item['title']} {item.get('description', '')}"
            
            for category in categories:
                if category in category_keywords:
                    for keyword in category_keywords[category]:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                            filtered_news[category].append(item)
                            break
        
        return filtered_news
    
    def get_sentiment_summary(self, symbol, days=7):
        """
        Get a summary of sentiment trends over time
        
        Args:
            symbol (str): Trading symbol
            days (int): Number of days to analyze
            
        Returns:
            dict: Sentiment summary statistics and trends
        """
        news = self.news_fetcher.get_news_for_symbol(
            symbol, days=days, max_items=100, analyze_sentiment=True
        )
        
        if not news:
            return {"error": "No news found"}
            
        # Group by date
        date_sentiment = {}
        for item in news:
            if 'publishedAt' in item:
                try:
                    date_str = item['publishedAt'].split('T')[0]
                    if date_str not in date_sentiment:
                        date_sentiment[date_str] = []
                    date_sentiment[date_str].append(item.get('sentiment', 0))
                except (AttributeError, IndexError):
                    pass
        
        # Calculate daily averages
        daily_avg = {date: sum(scores)/len(scores) for date, scores in date_sentiment.items()}
        
        # Get overall stats
        all_sentiment = [item.get('sentiment', 0) for item in news]
        
        return {
            'overall_average': sum(all_sentiment) / len(all_sentiment) if all_sentiment else 0,
            'min': min(all_sentiment) if all_sentiment else 0,
            'max': max(all_sentiment) if all_sentiment else 0,
            'count': len(news),
            'daily_trend': sorted([(date, avg) for date, avg in daily_avg.items()]),
            'positive_count': sum(1 for s in all_sentiment if s > 0.2),
            'negative_count': sum(1 for s in all_sentiment if s < -0.2),
            'neutral_count': sum(1 for s in all_sentiment if -0.2 <= s <= 0.2)
        }
    
    def _apply_keyword_boost(self, news_item, symbol, sentiment=None):
        """Apply keyword-based sentiment boost"""
        text = f"{news_item['title']} {news_item.get('description', '')}"
        if sentiment is None:
            sentiment = news_item.get('sentiment', 0)
        
        # Check general positive/negative keywords
        for keyword in self.positive_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                sentiment += 0.05  # Small boost for each positive keyword
                
        for keyword in self.negative_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                sentiment -= 0.05  # Small penalty for each negative keyword
        
        # Check company-specific keywords
        if symbol in self.company_specific_keywords:
            for keyword in self.company_specific_keywords[symbol]['positive']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    sentiment += 0.1  # Larger boost for company-specific positives
                    
            for keyword in self.company_specific_keywords[symbol]['negative']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    sentiment -= 0.1  # Larger penalty for company-specific negatives
        
        # Ensure sentiment stays in [-1, 1] range
        return max(-1, min(1, sentiment))
        
    def _detect_trending_topics(self, news_items, min_occurrences=2):
        """
        Detect trending topics in news articles
        
        Args:
            news_items (list): List of news items
            min_occurrences (int): Minimum occurrences to consider a topic trending
            
        Returns:
            list: List of (topic, count) tuples sorted by count
        """
        # Combine all text
        all_text = ""
        for item in news_items:
            text = f"{item['title']} {item.get('description', '')}"
            all_text += " " + text
            
        # Tokenize and filter words
        try:
            words = word_tokenize(all_text.lower())
            words = [word for word in words if word is not None and word.isalnum() and 
                    word not in self.stop_words and len(word) > 3]
                    
            # Count occurrences
            word_counts = Counter(words)
            
            # Filter by minimum occurrences and sort
            trending = [(word, count) for word, count in word_counts.items() 
                        if count >= min_occurrences]
            trending.sort(key=lambda x: x[1], reverse=True)
            
            return trending
        except:
            # Fall back if NLTK resources aren't available
            return []
            
    def _analyze_trend_sentiment(self, trending_topics, news_items):
        """
        Analyze sentiment of trending topics
        
        Args:
            trending_topics (list): List of (topic, count) tuples
            news_items (list): List of news items
            
        Returns:
            float: Sentiment score for trending topics
        """
        # Extract topics
        topics = [topic for topic, _ in trending_topics]
        
        # Find news items containing trending topics
        topic_sentiment = []
        
        for item in news_items:
            text = f"{item['title']} {item.get('description', '')}"
            text_lower = text.lower()
            
            # Check if this news item contains trending topics
            contains_topics = False
            for topic in topics:
                if topic in text_lower:
                    contains_topics = True
                    break
                    
            if contains_topics:
                topic_sentiment.append(item.get('sentiment', 0))
                
        # Return average sentiment for trending topics
        if topic_sentiment:
            return sum(topic_sentiment) / len(topic_sentiment)
        else:
            return 0.0
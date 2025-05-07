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

class SentimentAnalyzer:
    """
    A class for analyzing market sentiment from news articles and social media.
    Supports multiple data sources and combines them into a unified sentiment score.
    """
    
    def __init__(self, api_keys=None, config=None):
        """
        Initialize the sentiment analyzer with API keys and configuration
        
        Args:
            api_keys (dict): API keys for different news sources
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'sources': ['alphavantage', 'finnhub', 'newsapi'],
            'lookback_days': 3,
            'sentiment_weights': {
                'title': 0.6,
                'description': 0.3,
                'content': 0.1
            },
            'source_weights': {
                'alphavantage': 1.0,
                'finnhub': 0.8,
                'newsapi': 0.7,
                'twitter': 0.5
            },
            'cache_duration_hours': 4
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # API keys for different services
        self.api_keys = {
            'alphavantage': None,
            'finnhub': None,
            'newsapi': None,
            'twitter': None
        }
        
        if api_keys:
            self.api_keys.update(api_keys)
        
        # Initialize sentiment analysis tools
        self.vader = SentimentIntensityAnalyzer()
        
        # Cache for storing sentiment results
        self.sentiment_cache = {}
        
        # Set up logging
        self.logger = logging.getLogger('SentimentAnalyzer')
        
        self.finnhub_news_buffer = []  # Buffer for news from Finnhub webhooks
        self.finnhub_news_lock = threading.Lock()  # Lock for thread safety

    def get_sentiment(self, symbol, force_refresh=False):
        """
        Get sentiment analysis for a specific symbol
        
        Args:
            symbol (str): Trading symbol
            force_refresh (bool): Whether to force a cache refresh
            
        Returns:
            dict: Sentiment analysis results
        """
        # Check cache first (unless force refresh is requested)
        cache_key = f"{symbol.lower()}"
        if not force_refresh and cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            # Check if cache is still valid
            if datetime.now() - cached['timestamp'] < timedelta(hours=self.config['cache_duration_hours']):
                return cached['data']
        
        # Fetch and analyze news from all configured sources
        news_items = []
        
        for source in self.config['sources']:
            if self.api_keys.get(source):
                source_items = self._fetch_news_from_source(symbol, source)
                news_items.extend(source_items)
        
        # Also check our Finnhub webhook buffer for relevant news
        webhook_news = []
        with self.finnhub_news_lock:
            # Filter news that's recent and related to this symbol
            cutoff_time = time.time() - (24 * 60 * 60)  # Last 24 hours
            for item in self.finnhub_news_buffer:
                if item['received_at'] < cutoff_time:
                    continue
                    
                # Check if this item mentions our symbol
                if (symbol.upper() in [rel.upper() for rel in item['related']] or
                    symbol.upper() in item['headline'].upper() or
                    symbol.upper() in item['summary'].upper()):
                    webhook_news.append(item)
        
        # Combine news from API and webhook
        news_items.extend(webhook_news)

        # Analyze sentiment from collected news
        if not news_items:
            sentiment_result = {
                'symbol': symbol,
                'score': 0,
                'magnitude': 0,
                'signal': 'neutral',
                'confidence': 0,
                'news_count': 0,
                'latest_news': None,
                'timestamp': datetime.now()
            }
        else:
            # Analyze sentiment for each news item
            for item in news_items:
                item['sentiment_scores'] = self._analyze_text_sentiment(item)
            
            # Calculate aggregate sentiment
            sentiment_result = self._calculate_aggregate_sentiment(symbol, news_items)
        
        # Update cache
        self.sentiment_cache[cache_key] = {
            'data': sentiment_result,
            'timestamp': datetime.now()
        }
        
        return sentiment_result
    
    def _fetch_news_from_source(self, symbol, source):
        """
        Fetch news from a specific source
        
        Args:
            symbol (str): Trading symbol
            source (str): News source identifier
            
        Returns:
            list: List of news items
        """
        try:
            if source == 'alphavantage':
                return self._fetch_from_alphavantage(symbol)
            elif source == 'finnhub':
                return self._fetch_from_finnhub(symbol)
            elif source == 'newsapi':
                return self._fetch_from_newsapi(symbol)
            else:
                self.logger.warning(f"Unknown news source: {source}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching news from {source}: {e}")
            return []
    
    def _fetch_from_alphavantage(self, symbol):
        """
        Fetch news from Alpha Vantage
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            list: List of news items
        """
        api_key = self.api_keys.get('alphavantage')
        if not api_key:
            return []
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract news items
                news_items = []
                for item in data.get('feed', []):
                    # Check publication date
                    pub_date = datetime.strptime(item.get('time_published', '')[:8], "%Y%m%d")
                    if (datetime.now() - pub_date).days > self.config['lookback_days']:
                        continue
                    
                    news_items.append({
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'content': item.get('summary', ''),  # Alpha Vantage doesn't provide full content
                        'url': item.get('url', ''),
                        'published_at': pub_date,
                        'source': item.get('source', 'Alpha Vantage'),
                        'source_type': 'alphavantage',
                        'source_weight': self.config['source_weights']['alphavantage'],
                        'pre_analyzed_sentiment': item.get('overall_sentiment_score', None)
                    })
                
                return news_items
            else:
                self.logger.error(f"Alpha Vantage API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching news from Alpha Vantage: {e}")
            return []
    
    def _fetch_from_finnhub(self, symbol):
        """
        Fetch news from Finnhub
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            list: List of news items
        """
        api_key = self.api_keys.get('finnhub')
        if not api_key:
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['lookback_days'])
        
        url = (f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
               f"&from={start_date.strftime('%Y-%m-%d')}"
               f"&to={end_date.strftime('%Y-%m-%d')}"
               f"&token={api_key}")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract news items
                news_items = []
                for item in data:
                    news_items.append({
                        'title': item.get('headline', ''),
                        'description': item.get('summary', ''),
                        'content': item.get('summary', ''),
                        'url': item.get('url', ''),
                        'published_at': datetime.fromtimestamp(item.get('datetime', 0)),
                        'source': item.get('source', 'Finnhub'),
                        'source_type': 'finnhub',
                        'source_weight': self.config['source_weights']['finnhub'],
                        'pre_analyzed_sentiment': None
                    })
                
                return news_items
            else:
                self.logger.error(f"Finnhub API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching news from Finnhub: {e}")
            return []
    
    def _fetch_from_newsapi(self, symbol):
        """
        Fetch news from NewsAPI
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            list: List of news items
        """
        api_key = self.api_keys.get('newsapi')
        if not api_key:
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['lookback_days'])
        
        # Add company name for better search results if available
        company_mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Facebook OR Meta',
            'TSLA': 'Tesla',
            'NFLX': 'Netflix',
            'BTCUSD': 'Bitcoin',
            'ETHUSD': 'Ethereum'
        }
        
        query = company_mapping.get(symbol, symbol)
        
        url = (f"https://newsapi.org/v2/everything?q={query}"
               f"&from={start_date.strftime('%Y-%m-%d')}"
               f"&to={end_date.strftime('%Y-%m-%d')}"
               f"&language=en&sortBy=publishedAt"
               f"&apiKey={api_key}")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract news items
                news_items = []
                for item in data.get('articles', []):
                    news_items.append({
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'content': item.get('content', ''),
                        'url': item.get('url', ''),
                        'published_at': datetime.strptime(item.get('publishedAt', ''), "%Y-%m-%dT%H:%M:%SZ") 
                                       if item.get('publishedAt') else datetime.now(),
                        'source': item.get('source', {}).get('name', 'NewsAPI'),
                        'source_type': 'newsapi',
                        'source_weight': self.config['source_weights']['newsapi'],
                        'pre_analyzed_sentiment': None
                    })
                
                return news_items
            else:
                self.logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def _analyze_text_sentiment(self, news_item):
        """
        Analyze sentiment of a news item using multiple methods
        
        Args:
            news_item (dict): News item data
            
        Returns:
            dict: Sentiment scores
        """
        # If the source already provided sentiment analysis (e.g., Alpha Vantage)
        if news_item['pre_analyzed_sentiment'] is not None:
            return {
                'compound': news_item['pre_analyzed_sentiment'] * 2 - 1,  # Convert 0-1 range to -1 to 1
                'pos': max(0, news_item['pre_analyzed_sentiment']),
                'neg': max(0, -news_item['pre_analyzed_sentiment']),
                'neu': 1 - abs(news_item['pre_analyzed_sentiment'])
            }
        
        # Prepare text for analysis
        title = news_item.get('title', '')
        description = news_item.get('description', '')
        content = news_item.get('content', '')
        
        # Analyze using VADER
        title_sentiment = self.vader.polarity_scores(title) if title else {'compound': 0}
        desc_sentiment = self.vader.polarity_scores(description) if description else {'compound': 0}
        content_sentiment = self.vader.polarity_scores(content) if content else {'compound': 0}
        
        # Combine scores using configured weights
        weights = self.config['sentiment_weights']
        compound_score = (
            title_sentiment['compound'] * weights['title'] +
            desc_sentiment['compound'] * weights['description'] +
            content_sentiment['compound'] * weights['content']
        ) / sum(weights.values())
        
        # Analyze using TextBlob as a second opinion
        textblob_title = TextBlob(title).sentiment.polarity if title else 0
        textblob_desc = TextBlob(description).sentiment.polarity if description else 0
        
        # Average VADER and TextBlob
        final_compound = (compound_score + (textblob_title + textblob_desc)/2) / 2
        
        return {
            'compound': final_compound,
            'vader_compound': compound_score,
            'textblob': (textblob_title + textblob_desc) / 2,
            'pos': max(0, final_compound),
            'neg': max(0, -final_compound),
            'neu': 1 - abs(final_compound)
        }
    
    def _calculate_aggregate_sentiment(self, symbol, news_items):
        """
        Calculate aggregate sentiment from all news items
        
        Args:
            symbol (str): Trading symbol
            news_items (list): List of news items with sentiment scores
            
        Returns:
            dict: Aggregate sentiment analysis
        """
        if not news_items:
            return {
                'symbol': symbol,
                'score': 0,
                'magnitude': 0,
                'signal': 'neutral',
                'confidence': 0,
                'news_count': 0,
                'latest_news': None,
                'timestamp': datetime.now()
            }
        
        # Sort by date, newest first
        news_items.sort(key=lambda x: x['published_at'], reverse=True)
        
        # Calculate time-weighted scores (newer articles have more weight)
        total_score = 0
        total_weight = 0
        sentiment_distribution = []
        
        for i, item in enumerate(news_items):
            # Time decay factor (newest articles have more weight)
            recency_factor = 1.0 / (i + 1)
            
            # Source credibility weight
            source_weight = item.get('source_weight', 1.0)
            
            # Combined weight
            weight = recency_factor * source_weight
            
            # Get compound score
            score = item['sentiment_scores']['compound']
            
            total_score += score * weight
            total_weight += weight
            
            # Track distribution for confidence calculation
            sentiment_distribution.append(score)
        
        # Calculate weighted average sentiment
        avg_sentiment = total_score / total_weight if total_weight > 0 else 0
        
        # Calculate sentiment magnitude and confidence
        magnitude = abs(avg_sentiment)
        
        # Standard deviation shows sentiment consistency (lower std = higher confidence)
        std_dev = np.std(sentiment_distribution) if len(sentiment_distribution) > 1 else 1.0
        confidence = min(1.0, (1.0 - std_dev) * (len(news_items) / 10))
        
        # Adjust confidence based on number of sources
        sources = set(item['source'] for item in news_items)
        source_diversity_factor = min(1.0, len(sources) / 5)  # Max boost with 5+ different sources
        confidence = min(0.95, confidence + (source_diversity_factor * 0.2))
        
        # Determine signal
        if avg_sentiment > 0.2 and confidence > 0.4:
            signal = 'buy'
        elif avg_sentiment < -0.2 and confidence > 0.4:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Get latest news for reference
        latest_news = {
            'title': news_items[0]['title'],
            'url': news_items[0]['url'],
            'source': news_items[0]['source'],
            'published_at': news_items[0]['published_at']
        } if news_items else None
        
        return {
            'symbol': symbol,
            'score': avg_sentiment,
            'magnitude': magnitude,
            'signal': signal,
            'confidence': confidence,
            'news_count': len(news_items),
            'source_count': len(sources),
            'latest_news': latest_news,
            'timestamp': datetime.now()
        }

    def process_finnhub_news(self, payload):
        """
        Process news data received from Finnhub webhook
        
        Args:
            payload (dict): Webhook payload from Finnhub
        """
        if payload.get('type') != 'news':
            return
            
        try:
            # Extract news data
            data = payload.get('data', [])
            if not data:
                return
                
            # Store in buffer with timestamp
            with self.finnhub_news_lock:
                for item in data:
                    if 'headline' in item and 'summary' in item:
                        self.finnhub_news_buffer.append({
                            'headline': item['headline'],
                            'summary': item['summary'],
                            'source': item.get('source', 'Finnhub'),
                            'url': item.get('url', ''),
                            'datetime': item.get('datetime', int(time.time())),
                            'related': item.get('related', '').split(','),
                            'sentiment': self._analyze_text(f"{item['headline']} {item['summary']}"),
                            'received_at': time.time()
                        })
                
                # Keep buffer from growing too large (keep last 100 news items)
                if len(self.finnhub_news_buffer) > 100:
                    self.finnhub_news_buffer = self.finnhub_news_buffer[-100:]
                    
        except Exception as e:
            self.logger.error(f"Error processing Finnhub news: {e}")

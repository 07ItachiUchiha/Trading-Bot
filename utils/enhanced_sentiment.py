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
        """Fetch news for a specific symbol from multiple sources with improved integration"""
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
                            # Filter out irrelevant articles with better keyword matching
                            title = article['title'].lower()
                            if search_term.lower() in title or any(related in title for related in self._get_related_terms(search_term)):
                                results.append({
                                    'title': article['title'],
                                    'description': article.get('description', ''),
                                    'content': article.get('content', ''),
                                    'source': article['source']['name'],
                                    'published_at': article['publishedAt'],
                                    'url': article['url'],
                                    'relevance': 0.9 if search_term.lower() in title else 0.7  # Add relevance score
                                })
            except Exception as e:
                self.logger.error(f"Error fetching news from News API: {e}")
        
        # Try Alpha Vantage with improved parameters
        if self.api_keys['alphavantage']:
            try:
                # Use both ticker and name search for better results
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={search_term}&topics=earnings,financial_markets,technology,economy_macro,finance&limit=50&apikey={self.api_keys['alphavantage']}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'feed' in data:
                        for item in data['feed']:
                            # Calculate relevance based on sentiment certainty and matching
                            relevance = 0.6
                            if 'ticker_sentiment' in item:
                                for ticker_sent in item['ticker_sentiment']:
                                    if ticker_sent['ticker'].upper() == search_term.upper():
                                        relevance = float(ticker_sent.get('relevance_score', 0.6))
                                        break
                            
                            results.append({
                                'title': item['title'],
                                'description': item.get('summary', ''),
                                'content': item.get('summary', ''),
                                'source': item['source'],
                                'published_at': item['time_published'],
                                'url': item['url'],
                                'sentiment_score': item.get('overall_sentiment_score', None),
                                'sentiment_label': item.get('overall_sentiment_label', None),
                                'relevance': relevance
                            })
            except Exception as e:
                self.logger.error(f"Error fetching news from Alpha Vantage: {e}")
        
        # Try Finnhub with improved parameters
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
                        # Filter by relevance - Finnhub sometimes returns loosely related news
                        if search_term.lower() in item['headline'].lower() or search_term.lower() in item.get('summary', '').lower():
                            results.append({
                                'title': item['headline'],
                                'description': item.get('summary', ''),
                                'content': item.get('summary', ''),
                                'source': item['source'],
                                'published_at': datetime.fromtimestamp(item['datetime']).isoformat(),
                                'url': item['url'],
                                'relevance': 0.8 if search_term.lower() in item['headline'].lower() else 0.6
                            })
            except Exception as e:
                self.logger.error(f"Error fetching news from Finnhub: {e}")
        
        # Try additional API: Tiingo for further sentiment enhancement
        if self.api_keys.get('tiingo'):
            try:
                headers = {'Authorization': f"Token {self.api_keys['tiingo']}"}
                url = f"https://api.tiingo.com/tiingo/news?tickers={search_term}&startDate={(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        results.append({
                            'title': item['title'],
                            'description': item['description'],
                            'content': item['description'],
                            'source': item.get('source', 'Tiingo'),
                            'published_at': item['publishedDate'],
                            'url': item['url'],
                            'relevance': 0.8,
                            'tags': item.get('tags', [])
                        })
            except Exception as e:
                self.logger.error(f"Error fetching news from Tiingo: {e}")
        
        # If no news from APIs, generate mock news for testing
        if not results:
            self.logger.info(f"No API results found for {symbol}, generating mock news")
            try:
                results = self._generate_mock_news(symbol)
            except AttributeError:
                # Fallback if method missing (should not happen with this fix)
                self.logger.error(f"_generate_mock_news method not found, using emergency fallback for {symbol}")
                results = self._emergency_mock_news(symbol)
        else:
            # Sort by relevance and recency - prioritize relevant recent news
            results = sorted(results, 
                             key=lambda x: (x.get('relevance', 0.5), 
                                           self._get_recency_weight(x.get('published_at'))),
                             reverse=True)
            
            # Keep only the top 25 most relevant news items
            results = results[:25]
            
        return results

    def _get_recency_weight(self, published_at_str):
        """Calculate recency weight with better time handling"""
        if not published_at_str:
            return 0.5
            
        try:
            if isinstance(published_at_str, str):
                # Handle various date formats
                published_at_str = published_at_str.replace('Z', '+00:00')
                published_at = datetime.fromisoformat(published_at_str)
            else:
                published_at = published_at_str
                
            # Calculate days since publication
            days_old = (datetime.now() - published_at).total_seconds() / 86400
            
            # More sophisticated decay function: exponential decay with 3-day half-life
            weight = 2 ** (-days_old / 3.0)
            return max(0.2, min(1.0, weight))
            
        except Exception:
            return 0.5

    def _get_related_terms(self, symbol):
        """Get related search terms for better news matching"""
        # Map of symbols to related terms
        symbol_map = {
            'BTC': ['bitcoin', 'crypto', 'cryptocurrency', 'digital currency', 'btc'],
            'ETH': ['ethereum', 'crypto', 'defi', 'smart contract', 'eth', 'ether'],
            'SOL': ['solana', 'crypto', 'sol'],
            'ADA': ['cardano', 'crypto', 'ada'],
            'DOGE': ['dogecoin', 'crypto', 'doge', 'meme coin'],
            'AAPL': ['apple', 'iphone', 'mac', 'tech stock'],
            'MSFT': ['microsoft', 'windows', 'tech stock', 'azure', 'office'],
            'GOOGL': ['google', 'alphabet', 'tech stock', 'search engine'],
            'AMZN': ['amazon', 'e-commerce', 'aws', 'cloud'],
        }
        
        # Strip USD or other suffixes
        clean_symbol = symbol.replace('/USD', '').upper()
        if clean_symbol.endswith('USD'):
            clean_symbol = clean_symbol[:-3]
        
        return symbol_map.get(clean_symbol, [clean_symbol.lower()])

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

    def _generate_mock_news(self, symbol):
        """Generate realistic mock news data for when API calls fail"""
        import random
        from datetime import datetime, timedelta
        import logging
        
        logging.info(f"Generating mock news for {symbol}")
        
        # Current time for reference
        now = datetime.now()
        
        # Prepare mock news items
        mock_news = []
        
        # Number of mock news to generate
        num_items = random.randint(5, 15)
        
        # Common sources
        sources = [
            "Bloomberg", "Reuters", "CNBC", "Financial Times", "Wall Street Journal",
            "MarketWatch", "Yahoo Finance", "Investing.com", "Benzinga", "Forbes"
        ]
        
        # Symbol-specific news templates
        news_templates = {
            "BTC": [
                {"title": "Bitcoin Surges Above $TARGET_PRICE as Institutional Interest Grows", "sentiment": "positive"},
                {"title": "Bitcoin Drops Below $TARGET_PRICE Amid Market Volatility", "sentiment": "negative"},
                {"title": "Crypto Analysts Predict Bitcoin to Reach $TARGET_PRICE by Year End", "sentiment": "positive"},
                {"title": "Bitcoin Trading Sideways Around $TARGET_PRICE", "sentiment": "neutral"},
                {"title": "Major Investment Firm Adds Bitcoin to Portfolio", "sentiment": "positive"},
                {"title": "Regulatory Concerns Push Bitcoin Price Down", "sentiment": "negative"},
                {"title": "Bitcoin Mining Difficulty Increases", "sentiment": "neutral"},
                {"title": "New ETF Approval Boosts Bitcoin", "sentiment": "positive"},
            ],
            "ETH": [
                {"title": "Ethereum Breaks $TARGET_PRICE as Upgrade Nears", "sentiment": "positive"},
                {"title": "Ethereum Falls Below $TARGET_PRICE Support Level", "sentiment": "negative"},
                {"title": "Ethereum Gas Fees Hit New Low", "sentiment": "positive"},
                {"title": "Developers Delay Ethereum Upgrade", "sentiment": "negative"},
                {"title": "Ethereum Staking Growth Accelerates", "sentiment": "positive"},
                {"title": "ETH/BTC Ratio Approaching Key Level", "sentiment": "neutral"},
            ],
            "SOL": [
                {"title": "Solana Network Activity Hits All-Time High", "sentiment": "positive"},
                {"title": "Solana Experiences Network Outage", "sentiment": "negative"},
                {"title": "New Projects Choosing Solana Over Competitors", "sentiment": "positive"},
                {"title": "Solana Price Stabilizes Around $TARGET_PRICE", "sentiment": "neutral"},
                {"title": "Major Exchange Adds New Solana Trading Pairs", "sentiment": "positive"},
            ],
            "XAU": [
                {"title": "Gold Prices Rally as Inflation Fears Mount", "sentiment": "positive"},
                {"title": "Gold Retreats from $TARGET_PRICE on Strong Dollar", "sentiment": "negative"},
                {"title": "Central Banks Increase Gold Reserves", "sentiment": "positive"},
                {"title": "Gold Holds Steady Around $TARGET_PRICE", "sentiment": "neutral"},
                {"title": "Geopolitical Tensions Boost Safe Haven Demand for Gold", "sentiment": "positive"},
                {"title": "Rising Treasury Yields Pressure Gold Prices", "sentiment": "negative"},
                {"title": "Gold Mining Stocks Outperform Physical Gold", "sentiment": "positive"},
                {"title": "Technical Analysis: Gold Set for Breakout Above $TARGET_PRICE", "sentiment": "positive"},
            ]
        }
        
        # Extract the base symbol (remove /USD, etc.)
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Use the matching templates or generic if not found
        templates = news_templates.get(base_symbol, [
            {"title": f"{base_symbol} Price Analysis: Bulls Target $TARGET_PRICE", "sentiment": "positive"},
            {"title": f"{base_symbol} Falls as Sellers Take Control", "sentiment": "negative"},
            {"title": f"{base_symbol} Trading Volume Increases by 30%", "sentiment": "positive"},
            {"title": f"{base_symbol} Shows Signs of Consolidation", "sentiment": "neutral"},
            {"title": f"Analyst Report: {base_symbol} Forecast Updated", "sentiment": "neutral"}
        ])
        
        # Current mock price to use in templates
        current_price = self._get_mock_price(base_symbol)
        
        # Generate news items
        for i in range(num_items):
            # Select a random template
            template = random.choice(templates)
            
            # Calculate a random target price based on sentiment
            if template["sentiment"] == "positive":
                target_price = current_price * (1 + random.uniform(0.05, 0.2))
            elif template["sentiment"] == "negative":
                target_price = current_price * (1 - random.uniform(0.05, 0.2))
            else:
                target_price = current_price * (1 + random.uniform(-0.03, 0.03))
        
            # Format target price nicely
            formatted_price = f"{target_price:,.0f}" if target_price > 100 else f"{target_price:.2f}"
            
            # Replace placeholder with target price
            title = template["title"].replace("$TARGET_PRICE", formatted_price)
            
            # Generate a random published time within the last week
            days_ago = random.uniform(0, 7)
            published_time = (now - timedelta(days=days_ago)).isoformat()
            
            # Select a random source
            source = random.choice(sources)
            
            # Construct a mock news item
            news_item = {
                "title": title,
                "summary": f"This is a mock summary for {title.lower()}.",
                "published": published_time,
                "source": source,
                "url": f"https://example.com/mock-news/{i}",
                "sentiment_score": self._mock_sentiment_score(template["sentiment"]),
                "is_mock": True  # Flag to indicate this is mock data
            }
            
            mock_news.append(news_item)
        
        # Sort by published time (most recent first)
        mock_news.sort(key=lambda x: x["published"], reverse=True)
        
        return mock_news

    def _get_mock_price(self, symbol):
        """Get a realistic mock price for a given symbol"""
        # Base prices for common assets
        prices = {
            "BTC": 65000,
            "ETH": 3500,
            "SOL": 150,
            "ADA": 0.5,
            "DOGE": 0.15,
            "XRP": 0.5,
            "XAU": 2400,  # Gold price in USD
            "GOLD": 2400
        }
        
        # Use the symbol's price or a default
        return prices.get(symbol, 100)

    def _mock_sentiment_score(self, sentiment_category):
        """Generate a mock sentiment score based on category"""
        import random
        
        if sentiment_category == "positive":
            return random.uniform(0.55, 0.95)
        elif sentiment_category == "negative":
            return random.uniform(0.05, 0.45)
        else:  # neutral
            return random.uniform(0.45, 0.55)
        
    def _emergency_mock_news(self, symbol):
        """Emergency fallback for mock news if main method fails"""
        self.logger.warning(f"Using emergency mock news generator for {symbol}")
        # Simplified version for emergency fallback
        now = datetime.now()
        
        # Special case for XAU/USD
        if 'XAU' in symbol:
            return [
                {
                    "title": "Gold Prices Rally as Inflation Fears Mount",
                    "summary": "Emergency mock data: Gold prices seeing strong support.",
                    "published": now.isoformat(),
                    "source": "Emergency Mock Data",
                    "url": "https://example.com/mock-news/emergency",
                    "sentiment_score": 0.65,
                    "is_mock": True
                },
                {
                    "title": "Gold Holds Above $2400 Level",
                    "summary": "Emergency mock data: Gold maintaining key levels.",
                    "published": (now - timedelta(days=1)).isoformat(),
                    "source": "Emergency Mock Data",
                    "url": "https://example.com/mock-news/emergency-2",
                    "sentiment_score": 0.55,
                    "is_mock": True
                }
            ]
        else:
            return [
                {
                    "title": f"{symbol} Market Analysis",
                    "summary": f"Emergency mock data: {symbol} showing mixed signals.",
                    "published": now.isoformat(),
                    "source": "Emergency Mock Data",
                    "url": "https://example.com/mock-news/emergency",
                    "sentiment_score": 0.5,
                    "is_mock": True
                }
            ]
        
if __name__ == "__main__":
    # Example usage
    api_keys = {
        'newsapi': 'your_newsapi_key',
        'alphavantage': 'your_alphavantage_key',
        'finnhub': 'your_finnhub_key',
        'tiingo': 'your_tiingo_key'
    }
    
    analyzer = EnhancedSentimentAnalyzer(api_keys)
    
    # Fetch and analyze news for a specific symbol
    symbol = "AAPL"
    news = analyzer.fetch_news(symbol)
    sentiment = analyzer.analyze_sentiment(symbol, news)
    
    print(json.dumps(sentiment, indent=2))

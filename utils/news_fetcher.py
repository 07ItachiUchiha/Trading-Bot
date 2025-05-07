import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import time
import json
from urllib.parse import quote

class NewsFetcher:
    """
    A class to fetch financial news from multiple sources
    """
    
    def __init__(self, api_key=None):
        """Initialize with API credentials"""
        # NewsAPI API key
        self.news_api_key = api_key
        self.alphavantage_key = None
        self.finnhub_key = None
        self.cache_expiry = 30  # Cache expires after 30 minutes
        self.cached_news = {}
        self.last_fetch_time = {}

    def set_api_keys(self, news_api_key=None, alphavantage_key=None, finnhub_key=None):
        """Set API keys for different news sources"""
        if news_api_key:
            self.news_api_key = news_api_key
        if alphavantage_key:
            self.alphavantage_key = alphavantage_key
        if finnhub_key:
            self.finnhub_key = finnhub_key
        
    def get_news_for_symbol(self, symbol, days=1, max_items=10, use_cache=True, analyze_sentiment=True):
        """
        Get recent news for a specific symbol from multiple sources
        
        Args:
            symbol (str): Stock symbol (e.g., AAPL, BTCUSDT)
            days (int): Number of days to look back
            max_items (int): Maximum number of news items to return
            use_cache (bool): Whether to use cached news if available
            analyze_sentiment (bool): Whether to analyze sentiment of news articles
            
        Returns:
            list: List of news items as dictionaries
        """
        # Clean up the symbol - remove USD/USDT for cryptocurrencies
        search_term = symbol
        if "USD" in symbol:
            search_term = symbol.split("USD")[0]
        
        # Check if we have cached news and it's still valid
        cache_key = f"{search_term}_{days}"
        if use_cache and cache_key in self.cached_news:
            cache_time = self.last_fetch_time.get(cache_key, 0)
            if time.time() - cache_time < 60 * self.cache_expiry:  # 30 minutes cache
                return self.cached_news[cache_key]
        
        # Collect news from all available sources
        all_news = []
        
        # Try NewsAPI first
        if self.news_api_key:
            news_api_items = self._get_from_newsapi(search_term, days)
            all_news.extend(news_api_items)
        
        # Try Alpha Vantage news
        if self.alphavantage_key:
            alpha_vantage_items = self._get_from_alphavantage(search_term, days)
            all_news.extend(alpha_vantage_items)
        
        # Try Finnhub news
        if self.finnhub_key:
            finnhub_items = self._get_from_finnhub(symbol, days)
            all_news.extend(finnhub_items)
            
        # If no API keys set or no results, try public RSS feeds
        if not all_news:
            # Try Crypto-specific news for crypto symbols
            if any(crypto in symbol for crypto in ["BTC", "ETH", "XRP", "SOL", "ADA", "BNB"]):
                crypto_news = self._get_crypto_news(search_term, days)
                all_news.extend(crypto_news)
            
            # Try Yahoo Finance for all symbols
            yahoo_news = self._get_from_yahoo_finance(symbol, days)
            all_news.extend(yahoo_news)
        
        # Sort by date (newest first)
        all_news.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
        
        # Analyze sentiment if requested
        if analyze_sentiment:
            all_news = self.analyze_sentiment(all_news)
        
        # Limit results
        result = all_news[:max_items] if max_items else all_news
        
        # Cache the results
        self.cached_news[cache_key] = result
        self.last_fetch_time[cache_key] = time.time()
        
        return result

    def _get_from_newsapi(self, search_term, days=1):
        """Get news from NewsAPI"""
        if not self.news_api_key:
            return []
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Encode search term for URL
            search_query = f"{search_term} OR {search_term.lower()} OR {search_term.upper()}"
            
            # Make API request
            url = f"https://newsapi.org/v2/everything?q={quote(search_query)}&from={start_date}&to={end_date}&language=en&sortBy=publishedAt&apiKey={self.news_api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                # Format the news articles
                news_items = []
                for article in articles:
                    news_items.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "publishedAt": article.get("publishedAt", ""),
                        "source": f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}"
                    })
                
                return news_items
            else:
                print(f"NewsAPI Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def _get_from_alphavantage(self, search_term, days=1):
        """Get news from Alpha Vantage"""
        if not self.alphavantage_key:
            return []
            
        try:
            # Make API request
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={search_term}&limit=50&apikey={self.alphavantage_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("feed", [])
                
                # Format the news articles
                news_items = []
                for article in articles:
                    # Check if the article is within the date range
                    try:
                        pub_date = datetime.strptime(article.get("time_published", "")[:8], "%Y%m%d")
                        if (datetime.now() - pub_date).days > days:
                            continue
                    except:
                        # If date parsing fails, include it anyway
                        pass
                    
                    news_items.append({
                        "title": article.get("title", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "publishedAt": article.get("time_published", ""),
                        "source": f"Alpha Vantage - {article.get('source', 'Unknown')}",
                        "sentiment": article.get("overall_sentiment_score", 0)
                    })
                
                return news_items
            else:
                print(f"Alpha Vantage Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching news from Alpha Vantage: {e}")
            return []
    
    def _get_from_finnhub(self, symbol, days=1):
        """Get news from Finnhub"""
        if not self.finnhub_key:
            return []
            
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Make API request
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={self.finnhub_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                articles = response.json()
                
                # Format the news articles
                news_items = []
                for article in articles:
                    # Convert timestamp to datetime
                    try:
                        pub_date = datetime.fromtimestamp(article.get("datetime", 0))
                        published_at = pub_date.isoformat()
                    except:
                        published_at = ""
                    
                    news_items.append({
                        "title": article.get("headline", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "publishedAt": published_at,
                        "source": f"Finnhub - {article.get('source', 'Unknown')}",
                    })
                
                return news_items
            else:
                print(f"Finnhub Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching news from Finnhub: {e}")
            return []
    
    def _get_crypto_news(self, search_term, days=1):
        """
        Get cryptocurrency news from public sources like CryptoCompare
        
        Args:
            search_term (str): Cryptocurrency symbol/name (e.g., BTC, ETH)
            days (int): Number of days to look back
            
        Returns:
            list: List of news items as dictionaries
        """
        try:
            # Use CryptoCompare's free news API
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={search_term.lower()},Blockchain&excludeCategories=Sponsored"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("Data", [])
                
                # Calculate the cutoff date
                cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
                
                # Format the news articles
                news_items = []
                for article in articles:
                    # Skip if older than requested days
                    if article.get("published_on", 0) < cutoff_timestamp:
                        continue
                        
                    # Convert timestamp to ISO format
                    try:
                        pub_date = datetime.fromtimestamp(article.get("published_on", 0))
                        published_at = pub_date.isoformat()
                    except:
                        published_at = ""
                    
                    news_items.append({
                        "title": article.get("title", ""),
                        "description": article.get("body", ""),
                        "url": article.get("url", ""),
                        "publishedAt": published_at,
                        "source": f"CryptoCompare - {article.get('source_info', {}).get('name', 'Unknown')}",
                        "sentiment": 0  # Neutral by default
                    })
                
                return news_items
            else:
                print(f"CryptoCompare API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching crypto news: {e}")
            return []
    
    def _get_from_yahoo_finance(self, symbol, days=1):
        """
        Get news from Yahoo Finance using web scraping
        
        Args:
            symbol (str): Stock symbol (e.g., AAPL, MSFT)
            days (int): Number of days to look back
            
        Returns:
            list: List of news items as dictionaries
        """
        try:
            # We need BeautifulSoup for parsing HTML
            from bs4 import BeautifulSoup
            
            # Yahoo Finance URL for the symbol's news page
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            
            # Set a user agent to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                # Calculate the cutoff date
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find all news articles
                articles = soup.find_all("li", {"class": "js-stream-content"})
                
                # Format the news articles
                news_items = []
                for article in articles:
                    try:
                        # Extract the title and link
                        title_elem = article.find("h3")
                        link_elem = article.find("a")
                        
                        if not title_elem or not link_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        url = link_elem.get("href")
                        if url and not url.startswith("http"):
                            url = "https://finance.yahoo.com" + url
                            
                        # Extract the publisher and date
                        publisher_elem = article.find("div", {"class": "C(#959595)"})
                        publisher = "Yahoo Finance"
                        pub_date_str = ""
                        
                        if publisher_elem:
                            pub_info = publisher_elem.text.strip().split("·")
                            if len(pub_info) >= 2:
                                publisher = pub_info[0].strip()
                                pub_date_str = pub_info[1].strip()
                                
                        # Parse the date
                        pub_date = datetime.now()
                        if "hour" in pub_date_str or "minute" in pub_date_str:
                            # It's today
                            pass
                        elif "yesterday" in pub_date_str.lower():
                            pub_date = datetime.now() - timedelta(days=1)
                        elif "day" in pub_date_str:
                            days_ago = int(pub_date_str.split()[0])
                            pub_date = datetime.now() - timedelta(days=days_ago)
                            
                        # Skip if older than requested days
                        if (datetime.now() - pub_date).days > days:
                            continue
                            
                        # Extract description (summary)
                        desc_elem = article.find("p")
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                        news_items.append({
                            "title": title,
                            "description": description,
                            "url": url,
                            "publishedAt": pub_date.isoformat(),
                            "source": f"Yahoo Finance - {publisher}"
                        })
                        
                    except Exception as e:
                        print(f"Error parsing Yahoo Finance article: {e}")
                        continue
                
                return news_items
            else:
                print(f"Yahoo Finance Error: {response.status_code}")
                return []
                
        except ImportError:
            print("BeautifulSoup is required for Yahoo Finance scraping. Install it with 'pip install beautifulsoup4'")
            return []
        except Exception as e:
            print(f"Error fetching news from Yahoo Finance: {e}")
            return []

    def analyze_sentiment(self, news_items):
        """
        Analyze sentiment in news articles
        
        Args:
            news_items (list): List of news item dictionaries
            
        Returns:
            list: Same list with added sentiment scores
        """
        # Try to use NLTK for sentiment analysis if available
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Initialize the analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Analyze each news item
            for item in news_items:
                # Skip if sentiment is already provided
                if "sentiment" in item and item["sentiment"] != 0:
                    continue
                    
                # Get text to analyze (title + description)
                text = f"{item.get('title', '')} {item.get('description', '')}"
                
                # Perform sentiment analysis
                sentiment = sia.polarity_scores(text)
                
                # Add compound sentiment score
                item["sentiment"] = sentiment["compound"]
                
                # Add sentiment labels for easy filtering
                if sentiment["compound"] >= 0.05:
                    item["sentiment_label"] = "positive"
                elif sentiment["compound"] <= -0.05:
                    item["sentiment_label"] = "negative"
                else:
                    item["sentiment_label"] = "neutral"
                    
            return news_items
            
        except ImportError:
            # Fallback to simple keyword-based sentiment analysis
            print("NLTK not available. Using simple keyword-based sentiment analysis.")
            
            # Define positive and negative keywords
            positive_keywords = [
                "bullish", "buy", "positive", "gain", "growth", "profit", "success", "rally", "up",
                "grow", "strong", "surge", "jump", "advantage", "support", "alliance", "partnership",
                "good", "great", "excellent", "promising", "potential", "high", "best", "beat", "exceed",
                "breakthrough", "innovative", "opportunity", "progress", "approval", "launch"
            ]
            
            negative_keywords = [
                "bearish", "sell", "negative", "loss", "decline", "risk", "fail", "drop", "down",
                "weak", "plunge", "fall", "slump", "disadvantage", "opposition", "against", "controversy",
                "bad", "poor", "terrible", "challenging", "uncertain", "low", "worst", "miss", "underperform",
                "setback", "problem", "issue", "concern", "delay", "suspend", "lawsuit", "investigation"
            ]
            
            # Analyze each news item
            for item in news_items:
                # Skip if sentiment is already provided
                if "sentiment" in item and item["sentiment"] != 0:
                    continue
                    
                # Get text to analyze (title + description)
                text = f"{item.get('title', '')} {item.get('description', '')}".lower()
                
                # Count occurrences of positive and negative keywords
                positive_count = sum(1 for word in positive_keywords if word in text)
                negative_count = sum(1 for word in negative_keywords if word in text)
                
                # Calculate simple sentiment score (-1 to 1)
                if positive_count == 0 and negative_count == 0:
                    sentiment_score = 0
                else:
                    total = positive_count + negative_count
                    sentiment_score = (positive_count - negative_count) / total
                
                # Add sentiment score
                item["sentiment"] = sentiment_score
                
                # Add sentiment labels for easy filtering
                if sentiment_score > 0.1:
                    item["sentiment_label"] = "positive"
                elif sentiment_score < -0.1:
                    item["sentiment_label"] = "negative"
                else:
                    item["sentiment_label"] = "neutral"
            
            return news_items

    def filter_news_by_sentiment(self, news_items, sentiment_filter="all", min_sentiment=None, max_sentiment=None):
        """
        Filter news based on sentiment
        
        Args:
            news_items (list): List of news articles (usually from get_news_for_symbol)
            sentiment_filter (str): Filter by sentiment label ('positive', 'negative', 'neutral', or 'all')
            min_sentiment (float): Minimum sentiment score to include (scale -1 to 1)
            max_sentiment (float): Maximum sentiment score to include (scale -1 to 1)
            
        Returns:
            list: Filtered list of news items
        """
        # First ensure all items have sentiment scores
        news_items = self.analyze_sentiment(news_items)
        
        # Filter by sentiment label if specified
        if sentiment_filter != "all":
            news_items = [item for item in news_items 
                         if item.get("sentiment_label", "neutral") == sentiment_filter]
        
        # Filter by sentiment score range if specified
        if min_sentiment is not None:
            news_items = [item for item in news_items 
                         if item.get("sentiment", 0) >= min_sentiment]
            
        if max_sentiment is not None:
            news_items = [item for item in news_items 
                         if item.get("sentiment", 0) <= max_sentiment]
                         
        return news_items
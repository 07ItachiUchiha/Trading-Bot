import streamlit as st
import os
import json
from pathlib import Path
import base64
import requests
import time

def display_api_settings():
    """Display API configuration settings for exchanges and data providers"""
    st.header("🔑 API Configuration")
    
    # Load existing API settings
    api_settings = load_api_settings()
    
    # Create tabs for different API providers
    tabs = st.tabs(["Alpaca", "Binance", "NewsAPI", "Other Providers"])
    
    with tabs[0]:
        api_settings = display_alpaca_settings(api_settings)
    
    with tabs[1]:
        api_settings = display_binance_settings(api_settings)
    
    with tabs[2]:
        api_settings = display_news_api_settings(api_settings)
    
    with tabs[3]:
        api_settings = display_other_api_settings(api_settings)
    
    # Save settings button
    if st.button("Save API Settings", type="primary"):
        save_api_settings(api_settings)
        st.success("API settings saved successfully!")

def display_alpaca_settings(api_settings):
    """Display Alpaca API settings"""
    st.subheader("Alpaca Markets API")
    st.write("Configure Alpaca API access for stocks and crypto trading")
    
    # Get current Alpaca settings
    alpaca = api_settings.get('alpaca', {})
    
    # Check if we have existing credentials
    has_credentials = bool(alpaca.get('api_key', '')) and bool(alpaca.get('api_secret', ''))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use password input for API keys to hide them
        api_key = st.text_input(
            "API Key", 
            value=alpaca.get('api_key', ''),
            type="password" if has_credentials else "default",
            help="Your Alpaca API key"
        )
        
        # Only update if changed from placeholder
        if api_key != "••••••••" or not has_credentials:
            alpaca['api_key'] = api_key
    
    with col2:
        api_secret = st.text_input(
            "API Secret", 
            value=alpaca.get('api_secret', ''),
            type="password",
            help="Your Alpaca API secret"
        )
        
        # Only update if changed from placeholder
        if api_secret != "••••••••" or not has_credentials:
            alpaca['api_secret'] = api_secret
    
    # Endpoint selection
    alpaca['endpoint'] = st.selectbox(
        "Trading Environment",
        options=["Paper Trading", "Live Trading"],
        index=0 if alpaca.get('endpoint', 'paper') == 'paper' else 1,
        help="Select paper trading for testing, live for real money"
    )
    
    # Convert the select box value to the actual endpoint string
    alpaca['endpoint'] = 'paper' if alpaca['endpoint'] == 'Paper Trading' else 'live'
    
    # Test connection button
    if st.button("Test Alpaca Connection"):
        if api_key and api_secret:
            with st.spinner("Testing connection..."):
                success, message = test_alpaca_connection(api_key, api_secret, alpaca['endpoint'])
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.warning("Please enter API key and secret before testing")
    
    # Display resources
    with st.expander("Alpaca API Resources"):
        st.markdown("""
        - [Create an Alpaca account](https://app.alpaca.markets/signup)
        - [Alpaca API Documentation](https://alpaca.markets/docs/api-documentation/)
        - [Paper Trading Environment](https://paper-api.alpaca.markets)
        - [Live Trading Environment](https://api.alpaca.markets)
        """)
    
    # Update settings
    api_settings['alpaca'] = alpaca
    return api_settings

def display_binance_settings(api_settings):
    """Display Binance API settings"""
    st.subheader("Binance API")
    st.write("Configure Binance API access for cryptocurrency trading")
    
    # Get current Binance settings
    binance = api_settings.get('binance', {})
    
    # Check if we have existing credentials
    has_credentials = bool(binance.get('api_key', '')) and bool(binance.get('api_secret', ''))
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input(
            "API Key", 
            value=binance.get('api_key', ''),
            type="password" if has_credentials else "default",
            help="Your Binance API key",
            key="binance_api_key"
        )
        
        if api_key != "••••••••" or not has_credentials:
            binance['api_key'] = api_key
    
    with col2:
        api_secret = st.text_input(
            "API Secret", 
            value=binance.get('api_secret', ''),
            type="password",
            help="Your Binance API secret",
            key="binance_api_secret"
        )
        
        if api_secret != "••••••••" or not has_credentials:
            binance['api_secret'] = api_secret
    
    # Choose US or global Binance
    binance['endpoint'] = st.selectbox(
        "Binance Region",
        options=["Binance.US", "Binance Global"],
        index=0 if binance.get('endpoint', 'us') == 'us' else 1,
        help="Select the appropriate Binance platform based on your region"
    )
    
    # Convert the select box value
    binance['endpoint'] = 'us' if binance['endpoint'] == 'Binance.US' else 'global'
    
    # Test connection button
    if st.button("Test Binance Connection"):
        if api_key and api_secret:
            with st.spinner("Testing connection..."):
                success, message = test_binance_connection(api_key, api_secret, binance['endpoint'])
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.warning("Please enter API key and secret before testing")
    
    # Display resources
    with st.expander("Binance API Resources"):
        st.markdown("""
        - [Create a Binance.US account](https://accounts.binance.us/en/register)
        - [Create a Binance Global account](https://www.binance.com/en/register)
        - [Binance API Documentation](https://binance-docs.github.io/apidocs/)
        - [API Management](https://www.binance.com/en/my/settings/api-management)
        """)
    
    # Update settings
    api_settings['binance'] = binance
    return api_settings

def display_news_api_settings(api_settings):
    """Display NewsAPI settings"""
    st.subheader("News API for Sentiment Analysis")
    st.write("Configure News API access for market sentiment analysis")
    
    # Get current News API settings
    news_api = api_settings.get('news_api', {})
    
    # Check if we have existing credentials
    has_key = bool(news_api.get('api_key', ''))
    
    # API key input
    api_key = st.text_input(
        "API Key", 
        value=news_api.get('api_key', ''),
        type="password" if has_key else "default",
        help="Your News API key"
    )
    
    if api_key != "••••••••" or not has_key:
        news_api['api_key'] = api_key
    
    # Display additional options
    col1, col2 = st.columns(2)
    
    with col1:
        news_api['news_sources'] = st.text_input(
            "News Sources (comma-separated)",
            value=news_api.get('news_sources', 'bloomberg,reuters,cnbc,wsj'),
            help="News sources to use for sentiment analysis"
        )
    
    with col2:
        news_api['sentiment_update_interval'] = st.number_input(
            "Update Interval (minutes)",
            min_value=5,
            max_value=1440,
            value=int(news_api.get('sentiment_update_interval', 60)),
            step=5,
            help="How frequently to update sentiment analysis"
        )
    
    # News categories
    news_api['categories'] = st.multiselect(
        "News Categories",
        options=["business", "finance", "economy", "markets", "technology", "crypto", "commodities"],
        default=news_api.get('categories', ["business", "finance", "markets"]),
        help="Categories of news to analyze"
    )
    
    # Test connection button
    if st.button("Test News API Connection"):
        if api_key:
            with st.spinner("Testing connection..."):
                success, message = test_news_api_connection(api_key)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.warning("Please enter an API key before testing")
    
    # Display resources
    with st.expander("News API Resources"):
        st.markdown("""
        - [Create a NewsAPI account](https://newsapi.org/register)
        - [NewsAPI Documentation](https://newsapi.org/docs)
        - [Pricing](https://newsapi.org/pricing)
        """)
    
    # Update settings
    api_settings['news_api'] = news_api
    return api_settings

def display_other_api_settings(api_settings):
    """Display settings for other API providers"""
    st.subheader("Additional API Providers")
    st.write("Configure other API services for enhanced trading capabilities")
    
    # Get current other API settings
    other = api_settings.get('other', {})
    
    # Create tabs for different providers
    other_tabs = st.tabs(["CoinMarketCap", "TradingView", "Alpha Vantage"])
    
    with other_tabs[0]:
        st.subheader("CoinMarketCap API")
        
        # CoinMarketCap API key
        cmc_api_key = st.text_input(
            "API Key",
            value=other.get('coinmarketcap_api_key', ''),
            type="password" if other.get('coinmarketcap_api_key') else "default",
            help="Your CoinMarketCap API key",
            key="cmc_api_key"
        )
        
        if cmc_api_key != "••••••••" or not other.get('coinmarketcap_api_key'):
            other['coinmarketcap_api_key'] = cmc_api_key
            
        st.markdown("[Get a CoinMarketCap API key](https://coinmarketcap.com/api/)")
    
    with other_tabs[1]:
        st.subheader("TradingView")
        st.info("TradingView integration requires Pine Script export or webhook capabilities.")
        
        # TradingView webhook URL
        other['tradingview_webhook_enabled'] = st.checkbox(
            "Enable TradingView Webhook",
            value=other.get('tradingview_webhook_enabled', False),
            help="Enable webhook endpoint to receive signals from TradingView"
        )
        
        if other['tradingview_webhook_enabled']:
            other['tradingview_webhook_key'] = st.text_input(
                "Webhook Security Key",
                value=other.get('tradingview_webhook_key', ''),
                help="Security key to validate TradingView webhook requests"
            )
            
            if not other.get('tradingview_webhook_key'):
                # Generate a secure random key if none exists
                import secrets
                suggested_key = secrets.token_urlsafe(16)
                st.code(f"Suggested key: {suggested_key}")
                
                if st.button("Use Suggested Key"):
                    other['tradingview_webhook_key'] = suggested_key
        
        st.markdown("[TradingView Documentation](https://www.tradingview.com/support/)")
    
    with other_tabs[2]:
        st.subheader("Alpha Vantage API")
        
        # Alpha Vantage API key
        alpha_api_key = st.text_input(
            "API Key",
            value=other.get('alphavantage_api_key', ''),
            type="password" if other.get('alphavantage_api_key') else "default",
            help="Your Alpha Vantage API key",
            key="alpha_api_key"
        )
        
        if alpha_api_key != "••••••••" or not other.get('alphavantage_api_key'):
            other['alphavantage_api_key'] = alpha_api_key
            
        # Alpha Vantage settings
        other['alphavantage_data_types'] = st.multiselect(
            "Data Types to Fetch",
            options=["TIME_SERIES_DAILY", "TIME_SERIES_INTRADAY", "GLOBAL_QUOTE", "TECHNICAL_INDICATORS"],
            default=other.get('alphavantage_data_types', ["GLOBAL_QUOTE"]),
            help="Types of data to fetch from Alpha Vantage"
        )
        
        st.markdown("[Get an Alpha Vantage API key](https://www.alphavantage.co/support/#api-key)")
    
    # Update settings
    api_settings['other'] = other
    return api_settings

def load_api_settings():
    """Load API settings from storage"""
    try:
        config_dir = Path(__file__).parent.parent.parent / "config"
        api_config_file = config_dir / "api_settings.json"
        
        if api_config_file.exists():
            with open(api_config_file, 'r') as f:
                return json.load(f)
        else:
            # Return default empty settings if file doesn't exist
            return {
                'alpaca': {},
                'binance': {},
                'news_api': {},
                'other': {}
            }
    except Exception as e:
        st.error(f"Error loading API settings: {str(e)}")
        return {
            'alpaca': {},
            'binance': {},
            'news_api': {},
            'other': {}
        }

def save_api_settings(settings):
    """Save API settings to storage"""
    try:
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        api_config_file = config_dir / "api_settings.json"
        
        with open(api_config_file, 'w') as f:
            json.dump(settings, f, indent=4)
            
        # Also save individual provider credentials to environment variables
        # This is useful for deployment scenarios
        if 'alpaca' in settings and settings['alpaca'].get('api_key'):
            os.environ["ALPACA_API_KEY"] = settings['alpaca']['api_key']
            os.environ["ALPACA_API_SECRET"] = settings['alpaca'].get('api_secret', '')
            
        if 'binance' in settings and settings['binance'].get('api_key'):
            os.environ["BINANCE_API_KEY"] = settings['binance']['api_key']
            os.environ["BINANCE_API_SECRET"] = settings['binance'].get('api_secret', '')
        
        return True
    except Exception as e:
        st.error(f"Error saving API settings: {str(e)}")
        return False

def test_alpaca_connection(api_key, api_secret, endpoint):
    """Test connection to Alpaca API"""
    try:
        # Construct API URL based on endpoint
        base_url = "https://paper-api.alpaca.markets" if endpoint == 'paper' else "https://api.alpaca.markets"
        url = f"{base_url}/v2/account"
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_info = response.json()
            account_status = account_info.get('status', 'Unknown')
            buying_power = float(account_info.get('buying_power', 0))
            
            return True, f"Connected to Alpaca {endpoint} API. Account status: {account_status}, Buying power: ${buying_power:,.2f}"
        else:
            return False, f"Connection failed. HTTP status: {response.status_code}. {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_binance_connection(api_key, api_secret, endpoint):
    """Test connection to Binance API"""
    try:
        # Determine base URL based on endpoint
        base_url = "https://api.binance.us" if endpoint == 'us' else "https://api.binance.com"
        url = f"{base_url}/api/v3/account"
        
        # Create timestamp for signing
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        
        # Create signature
        import hmac
        import hashlib
        
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Combine URL and parameters
        url = f"{url}?{query_string}&signature={signature}"
        
        headers = {
            "X-MBX-APIKEY": api_key
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_info = response.json()
            return True, f"Connected to Binance API successfully. Account has {len(account_info.get('balances', []))} assets."
        else:
            return False, f"Connection failed. HTTP status: {response.status_code}. {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_news_api_connection(api_key):
    """Test connection to NewsAPI"""
    try:
        url = "https://newsapi.org/v2/top-headlines"
        
        params = {
            "apiKey": api_key,
            "category": "business",
            "pageSize": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, f"Connected to News API successfully. {data.get('totalResults', 0)} business headlines available."
        else:
            return False, f"Connection failed. HTTP status: {response.status_code}. {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

if __name__ == "__main__":
    # Test the component
    st.set_page_config(page_title="API Settings Test", layout="wide")
    display_api_settings()

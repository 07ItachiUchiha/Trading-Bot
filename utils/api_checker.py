import os
import json
import logging
from pathlib import Path
import requests

logger = logging.getLogger("api_checker")

def check_api_credentials():
    """Check if API credentials are properly configured"""
    # Try to load API credentials from config
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    
    # Check if credentials are set
    if not api_key or not api_secret:
        logger.warning("API credentials are not set in environment variables.")
        
        # Try to load from config file
        try:
            config_path = Path(__file__).parent.parent / "config.py"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_content = f.read()
                    if "API_KEY" in config_content and "API_SECRET" in config_content:
                        logger.info("API credentials found in config.py, but may not be loaded correctly.")
                    else:
                        logger.warning("API credentials not found in config.py")
            else:
                logger.warning("config.py file not found")
        except Exception as e:
            logger.error(f"Error checking config file: {e}")
    
    # Test API connection
    try:
        # Alpaca API check
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        response = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
        
        if response.status_code == 200:
            logger.info("Alpaca API connection successful!")
            return True
        else:
            logger.error(f"API connection failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing API connection: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    # Run the check
    check_api_credentials()

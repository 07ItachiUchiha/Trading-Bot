import requests
import os
import logging

# Configure logging
logger = logging.getLogger('telegram_alert')

# Default values (should be overridden in config.py)
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_alert(message):
    """Send a message to Telegram. Returns True on success."""
    try:
        # Check if token and chat_id are available
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured. Message not sent.")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully")
            return True
        else:
            logger.error(f"Failed to send Telegram alert: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {str(e)}")
        return False

import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logger = logging.getLogger('slack_webhook')

# Default token (should be overridden in config.py or dashboard settings)
SLACK_TOKEN = os.environ.get('SLACK_TOKEN', '')
SLACK_CHANNEL = os.environ.get('SLACK_CHANNEL', '#trading-alerts')

def send_slack_alert(message):
    """Send an alert to a Slack channel."""
    try:
        # Check if token is available
        if not SLACK_TOKEN:
            logger.warning("Slack token not configured. Message not sent.")
            return False
            
        client = WebClient(token=SLACK_TOKEN)
        
        # color by message type
        color = "#36C5F0"
        if "NEW MANUAL TRADE" in message or "BUY SIGNAL" in message:
            color = "#2EB67D"
        elif "POSITION CLOSED" in message or "SELL SIGNAL" in message:
            color = "#E01E5A"
        
        # Send message with attachment
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL,
            text="Trading Bot Alert",
            attachments=[
                {
                    "color": color,
                    "text": message,
                    "fallback": message
                }
            ]
        )
        
        logger.info("Slack alert sent successfully")
        return True
            
    except SlackApiError as e:
        logger.error(f"Error sending Slack alert: {e.response['error']}")
        return False
    except Exception as e:
        logger.error(f"Error sending Slack alert: {str(e)}")
        return False

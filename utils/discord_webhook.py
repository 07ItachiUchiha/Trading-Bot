import os
import logging
from discord_webhook import DiscordWebhook, DiscordEmbed

# Configure logging
logger = logging.getLogger('discord_webhook')

# Default webhook URL (should be overridden in config.py or dashboard settings)
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

def send_discord_alert(message, title="Trading Bot Alert"):
    """Send an embed message to Discord via webhook."""
    try:
        # Check if webhook URL is available
        if not DISCORD_WEBHOOK_URL:
            logger.warning("Discord webhook URL not configured. Message not sent.")
            return False
            
        # Initialize webhook
        webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
        
        embed = DiscordEmbed(
            title=title, 
            description=message, 
            color="03b2f8"  # Blue color
        )
        
        # color by alert type
        if "NEW MANUAL TRADE" in message or "BUY SIGNAL" in message:
            embed.set_color("18c651")  # Green color
        elif "POSITION CLOSED" in message or "SELL SIGNAL" in message:
            embed.set_color("e74c3c")  # Red color
            
        embed.set_timestamp()
        webhook.add_embed(embed)
        response = webhook.execute()
        
        if response:
            logger.info("Discord alert sent successfully")
            return True
        else:
            logger.error("Failed to send Discord alert")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Discord alert: {str(e)}")
        return False

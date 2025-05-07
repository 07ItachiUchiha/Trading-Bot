import json
import logging
import hmac
import hashlib
import threading
from flask import Flask, request, jsonify
from datetime import datetime

# Get the configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import FINNHUB_WEBHOOK_SECRET

# Configure logging
logger = logging.getLogger('finnhub_webhook')

app = Flask(__name__)

# Event subscribers for different types of events
event_subscribers = {
    'trade': [],
    'news': [],
    'earnings': [],
    'price_target': [],
    'technical_signals': []
}

def verify_webhook_signature(payload, signature):
    """
    Verify the webhook signature from Finnhub
    
    Args:
        payload: The webhook payload (raw bytes)
        signature: The X-Finnhub-Signature header
    
    Returns:
        bool: Whether the signature is valid
    """
    if not FINNHUB_WEBHOOK_SECRET:
        logger.warning("No Finnhub webhook secret configured, skipping signature verification")
        return True
        
    try:
        # Create expected signature
        expected = hmac.new(
            FINNHUB_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare with provided signature
        return hmac.compare_digest(expected, signature)
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {e}")
        return False

@app.route('/webhook/finnhub', methods=['POST'])
def finnhub_webhook():
    """
    Handle incoming webhooks from Finnhub
    
    This endpoint receives various events from Finnhub including:
    - Earnings reports
    - Price targets
    - News
    - Technical signals
    """
    # Verify webhook signature
    signature = request.headers.get('X-Finnhub-Signature', '')
    if not verify_webhook_signature(request.data, signature):
        logger.warning(f"Invalid webhook signature received: {signature}")
        return jsonify({"status": "error", "message": "Invalid signature"}), 401
    
    try:
        # Parse the payload
        payload = request.json
        
        if not payload:
            return jsonify({"status": "error", "message": "Empty payload"}), 400
        
        # Log the event
        logger.info(f"Received Finnhub webhook: {payload.get('type', 'unknown')}")
        
        # Process different types of events
        event_type = payload.get('type', 'unknown')
        if event_type in event_subscribers:
            # Notify subscribers
            for callback in event_subscribers[event_type]:
                # Use a thread to prevent blocking the response
                threading.Thread(
                    target=callback,
                    args=(payload,),
                    daemon=True
                ).start()
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def subscribe_to_event(event_type, callback):
    """
    Subscribe to a specific type of Finnhub event
    
    Args:
        event_type (str): The event type ('trade', 'news', 'earnings', etc.)
        callback (function): The function to call when the event occurs
        
    Returns:
        bool: Whether the subscription was successful
    """
    if event_type not in event_subscribers:
        logger.error(f"Unknown event type: {event_type}")
        return False
    
    event_subscribers[event_type].append(callback)
    logger.info(f"Subscribed to {event_type} events")
    return True

def start_webhook_server(host='0.0.0.0', port=8000):
    """
    Start the Flask server to receive webhooks
    
    Args:
        host (str): Host to bind to
        port (int): Port to listen on
        
    Returns:
        threading.Thread: The server thread
    """
    def run_server():
        app.run(host=host, port=port)
    
    # Start Flask in a separate thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    logger.info(f"Finnhub webhook server started on {host}:{port}")
    return thread

if __name__ == "__main__":
    # Configure logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    def handle_news_event(payload):
        print(f"News received at {datetime.now()}: {payload}")
    
    # Subscribe to news events
    subscribe_to_event('news', handle_news_event)
    
    # Start the server
    start_webhook_server()
    
    # Keep the main thread running
    import time
    while True:
        time.sleep(1)

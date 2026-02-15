import hashlib
import hmac
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
import sys

from flask import Flask, jsonify, request

sys.path.append(str(Path(__file__).parent.parent))
from config import FINNHUB_WEBHOOK_SECRET

logger = logging.getLogger("finnhub_webhook")
app = Flask(__name__)

event_subscribers = {
    "trade": [],
    "news": [],
    "earnings": [],
    "price_target": [],
    "technical_signals": [],
}


def verify_webhook_signature(payload, signature):
    """
    Verify the webhook signature from Finnhub.

    This function is intentionally fail-closed: if secret is missing,
    all webhook requests are rejected.
    """
    if not FINNHUB_WEBHOOK_SECRET:
        logger.error("No Finnhub webhook secret configured; rejecting webhook request")
        return False

    try:
        expected = hmac.HMAC(
            FINNHUB_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature or "")
    except Exception as exc:
        logger.error("Error verifying webhook signature: %s", exc)
        return False


@app.route("/webhook/finnhub", methods=["POST"])
def finnhub_webhook():
    """Handle incoming webhooks from Finnhub."""
    signature = request.headers.get("X-Finnhub-Signature", "")
    if not verify_webhook_signature(request.data, signature):
        logger.warning("Invalid webhook signature received: %s", signature)
        return jsonify({"status": "error", "message": "Invalid signature"}), 401

    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"status": "error", "message": "Empty payload"}), 400

        event_type = payload.get("type", "unknown")
        logger.info("Received Finnhub webhook: %s", event_type)

        callbacks = event_subscribers.get(event_type, [])
        for callback in callbacks:
            threading.Thread(target=callback, args=(payload,), daemon=True).start()

        return jsonify({"status": "success"}), 200
    except Exception as exc:
        logger.error("Error processing webhook: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


def subscribe_to_event(event_type, callback):
    """Subscribe a callback to a webhook event type."""
    if event_type not in event_subscribers:
        logger.error("Unknown event type: %s", event_type)
        return False

    event_subscribers[event_type].append(callback)
    logger.info("Subscribed to %s events", event_type)
    return True


def start_webhook_server(host="0.0.0.0", port=8000):
    """Start the Flask webhook server in a daemon thread."""

    def run_server():
        app.run(host=host, port=port)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info("Finnhub webhook server started on %s:%s", host, port)
    return thread


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def handle_news_event(payload):
        print(f"News received at {datetime.now()}: {payload}")

    subscribe_to_event("news", handle_news_event)
    start_webhook_server()
    while True:
        time.sleep(1)

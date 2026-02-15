#!/usr/bin/env python
import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our components
from strategy.prediction_runtime_manager import PredictionRuntimeManager
from utils.finnhub_webhook import start_webhook_server, subscribe_to_event
from config import (
    API_KEY, API_SECRET, DEFAULT_SYMBOLS, CAPITAL, RISK_PERCENT,
    PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT, FINNHUB_WEBHOOK_SECRET
)

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"prediction_{time.strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for this module
    logger = logging.getLogger("run_bot")
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Run the prediction runtime')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=DEFAULT_SYMBOLS,
                        help='Symbols to analyze')
    parser.add_argument('--timeframe', type=str, default='30Min',
                        help='Prediction timeframe')
    parser.add_argument('--capital', type=float, default=CAPITAL,
                        help='Reference capital')
    parser.add_argument('--risk-percent', type=float, default=RISK_PERCENT,
                        help='Risk budget percentage per signal')
    parser.add_argument('--profit-target', type=float, default=PROFIT_TARGET_PERCENT,
                        help='Target threshold percentage')
    parser.add_argument('--daily-profit-target', type=float, default=DAILY_PROFIT_TARGET_PERCENT,
                        help='Daily target threshold percentage')
    parser.add_argument('--use-news', action='store_true', default=True,
                        help='Use news sentiment analysis')
    parser.add_argument('--use-earnings', action='store_true', default=True,
                        help='Use earnings reports for model signals')
    parser.add_argument('--webhook-host', type=str, default='0.0.0.0',
                        help='Host for the webhook server')
    parser.add_argument('--webhook-port', type=int, default=8000,
                        help='Port for the webhook server')
    parser.add_argument('--no-webhook', action='store_true',
                        help='Disable webhook server')
    
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info(f"Starting prediction runtime with symbols: {args.symbols}")
    
    # start webhook server if enabled and securely configured
    webhook_enabled = not args.no_webhook
    if webhook_enabled and not FINNHUB_WEBHOOK_SECRET:
        logger.error(
            "Webhook server disabled: FINNHUB_WEBHOOK_SECRET is not configured."
        )
        webhook_enabled = False

    if webhook_enabled:
        logger.info("Starting Finnhub webhook server")
        webhook_thread = start_webhook_server(args.webhook_host, args.webhook_port)
        time.sleep(1)  # give the server a sec to start up
    
    # Create runtime manager
    try:
        runtime_manager = PredictionRuntimeManager(
            symbols=args.symbols,
            timeframe=args.timeframe,
            capital=args.capital,
            risk_percent=args.risk_percent,
            profit_target_percent=args.profit_target,
            daily_profit_target=args.daily_profit_target,
            use_news=args.use_news,
            use_earnings=args.use_earnings
        )
        
        # hook up webhook events to manager handlers
        if webhook_enabled:
            logger.info("Wiring up webhook events")
            try:
                subscribe_to_event('news', runtime_manager.process_news_event)
                subscribe_to_event('earnings', runtime_manager.process_earnings_event)
            except Exception as e:
                logger.error(f"Error connecting webhook events: {e}")
                
        # Run the runtime manager (this will block)
        runtime_manager.run()
        
    except KeyboardInterrupt:
        logger.info("Prediction runtime interrupted by user")
        if 'runtime_manager' in locals():
            runtime_manager.stop()
    except Exception as e:
        logger.exception(f"Error running prediction runtime: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

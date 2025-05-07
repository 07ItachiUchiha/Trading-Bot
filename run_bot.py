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
from strategy.auto_trading_manager import AutoTradingManager
from utils.finnhub_webhook import start_webhook_server, subscribe_to_event
from config import (
    API_KEY, API_SECRET, DEFAULT_SYMBOLS, CAPITAL, RISK_PERCENT,
    PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT
)

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"trading_{time.strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for this module
    logger = logging.getLogger("run_bot")
    logger.info("Logging initialized")
    
    return logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the trading bot')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=DEFAULT_SYMBOLS,
                        help='Symbols to trade')
    parser.add_argument('--timeframe', type=str, default='30Min',
                        help='Timeframe for trading')
    parser.add_argument('--capital', type=float, default=CAPITAL,
                        help='Trading capital')
    parser.add_argument('--risk-percent', type=float, default=RISK_PERCENT,
                        help='Risk percentage per trade')
    parser.add_argument('--profit-target', type=float, default=PROFIT_TARGET_PERCENT,
                        help='Profit target percentage')
    parser.add_argument('--daily-profit-target', type=float, default=DAILY_PROFIT_TARGET_PERCENT,
                        help='Daily profit target percentage')
    parser.add_argument('--use-news', action='store_true', default=True,
                        help='Use news sentiment analysis')
    parser.add_argument('--use-earnings', action='store_true', default=True,
                        help='Use earnings reports for trading signals')
    parser.add_argument('--webhook-host', type=str, default='0.0.0.0',
                        help='Host for the webhook server')
    parser.add_argument('--webhook-port', type=int, default=8000,
                        help='Port for the webhook server')
    parser.add_argument('--no-webhook', action='store_true',
                        help='Disable webhook server')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info(f"Starting trading bot with symbols: {args.symbols}")
    
    # Start webhook server if enabled
    if not args.no_webhook:
        logger.info("Starting Finnhub webhook server")
        webhook_thread = start_webhook_server(args.webhook_host, args.webhook_port)
        
        # Wait a moment to ensure the server starts
        time.sleep(1)
    
    # Create trading manager
    try:
        trading_manager = AutoTradingManager(
            symbols=args.symbols,
            timeframe=args.timeframe,
            capital=args.capital,
            risk_percent=args.risk_percent,
            profit_target_percent=args.profit_target,
            daily_profit_target=args.daily_profit_target,
            use_news=args.use_news,
            use_earnings=args.use_earnings
        )
        
        # Connect webhook events if available
        if not args.no_webhook and trading_manager.news_strategy:
            logger.info("Connecting webhook events to news strategy")
            try:
                subscribe_to_event('news', trading_manager.news_strategy.process_news)
                subscribe_to_event('earnings', trading_manager.process_earnings_event)
            except Exception as e:
                logger.error(f"Error connecting webhook events: {e}")
                
        # Run the trading manager (this will block)
        trading_manager.run()
        
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user")
        if 'trading_manager' in locals():
            trading_manager.stop()
    except Exception as e:
        logger.exception(f"Error running trading bot: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

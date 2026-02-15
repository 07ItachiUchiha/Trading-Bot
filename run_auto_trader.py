#!/usr/bin/env python
import argparse
import sys
import os
import time
from datetime import datetime
import json
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.auto_trading_manager import AutoTradingManager
from config import DEFAULT_SYMBOLS, CAPITAL, RISK_PERCENT, PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT

def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'auto_trader_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('run_auto_trader')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run auto trading algorithm.')
    
    parser.add_argument('--symbols', nargs='+', default=None,
                      help='Trading symbols to use (e.g., BTCUSDT ETHUSDT)')
    
    parser.add_argument('--timeframe', default='1h',
                      help='Trading timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    
    parser.add_argument('--capital', type=float, default=CAPITAL,
                      help=f'Initial capital in USD (default: {CAPITAL})')
    
    parser.add_argument('--risk-percent', type=float, default=RISK_PERCENT,
                      help=f'Risk percentage per trade (default: {RISK_PERCENT}%)')
    
    parser.add_argument('--profit-target', type=float, default=PROFIT_TARGET_PERCENT,
                      help=f'Profit target percentage to stop trading (default: {PROFIT_TARGET_PERCENT}%)')
    
    parser.add_argument('--daily-profit-target', type=float, default=DAILY_PROFIT_TARGET_PERCENT,
                      help=f'Daily profit target percentage (default: {DAILY_PROFIT_TARGET_PERCENT}%)')
    
    parser.add_argument('--use-news', action='store_true', default=True,
                      help='Use news-based strategy (default: True)')
    
    parser.add_argument('--news-weight', type=float, default=0.5,
                      help='Weight for news signals (default: 0.5)')
    
    parser.add_argument('--use-earnings', action='store_true', default=True,
                      help='Use earnings-based strategy (default: True)')
    
    parser.add_argument('--earnings-weight', type=float, default=0.6,
                      help='Weight for earnings signals (default: 0.6)')
    
    return parser.parse_args()

def main():
    logger = setup_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    
    logger.info(f"Starting auto trader")
    logger.info(f"Symbols: {symbols} | Capital: ${args.capital} | Risk: {args.risk_percent}%")
    
    try:
        trader = AutoTradingManager(
            symbols=symbols,
            timeframe=args.timeframe,
            capital=args.capital,
            risk_percent=args.risk_percent,
            profit_target_percent=args.profit_target,
            daily_profit_target=args.daily_profit_target,
            use_news=args.use_news,
            news_weight=args.news_weight,
            use_earnings=args.use_earnings,
            earnings_weight=args.earnings_weight
        )
        
        trader.run()
        
    except KeyboardInterrupt:
        logger.info("Auto trader stopped by user")
    except Exception as e:
        logger.error(f"Error running auto trader: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

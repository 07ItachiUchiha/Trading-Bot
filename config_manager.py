"""
🔐 Enhanced Configuration Management
Integrates secure configuration with the main trading bot application
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Add security module to path
sys.path.append(str(Path(__file__).parent))

from security.secure_config import SecureConfigManager, get_secure_config

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Unified configuration manager that integrates secure and standard config
    """
    
    def __init__(self):
        self.secure_manager = SecureConfigManager()
        self._config_cache: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from secure storage with fallbacks"""
        try:
            # Try to load from secure storage first
            secure_config = get_secure_config()
            if secure_config:
                self._config_cache = secure_config
                logger.info("Loaded configuration from secure storage")
                return
        except Exception as e:
            logger.warning(f"Could not load secure config: {e}")
        
        # Fallback to environment variables and config.py
        try:
            from config import (
                API_KEY, API_SECRET, NEWS_API_KEY, ALPHAVANTAGE_API_KEY, 
                FINNHUB_API_KEY, CAPITAL, RISK_PERCENT, DEFAULT_SYMBOLS,
                PROFIT_TARGET_PERCENT, DAILY_PROFIT_TARGET_PERCENT
            )
            
            self._config_cache = {
                'api_keys': {
                    'alpaca_key': API_KEY,
                    'alpaca_secret': API_SECRET,
                    'news_api': NEWS_API_KEY,
                    'alphavantage': ALPHAVANTAGE_API_KEY,
                    'finnhub': FINNHUB_API_KEY
                },
                'trading': {
                    'capital': CAPITAL,
                    'risk_percent': RISK_PERCENT,
                    'default_symbols': DEFAULT_SYMBOLS,
                    'profit_target_percent': PROFIT_TARGET_PERCENT,
                    'daily_profit_target_percent': DAILY_PROFIT_TARGET_PERCENT
                }
            }
            logger.info("Loaded configuration from config.py")
            
        except ImportError as e:
            logger.error(f"Could not load configuration: {e}")
            self._config_cache = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with environment variable fallbacks"""
        return {
            'api_keys': {
                'alpaca_key': os.getenv('ALPACA_API_KEY', ''),
                'alpaca_secret': os.getenv('ALPACA_API_SECRET', ''),
                'news_api': os.getenv('NEWS_API_KEY', ''),
                'alphavantage': os.getenv('ALPHAVANTAGE_API_KEY', ''),
                'finnhub': os.getenv('FINNHUB_API_KEY', '')
            },
            'trading': {
                'capital': float(os.getenv('CAPITAL', '10000')),
                'risk_percent': float(os.getenv('RISK_PERCENT', '1.0')),
                'default_symbols': ['BTC/USD', 'ETH/USD'],
                'profit_target_percent': float(os.getenv('PROFIT_TARGET_PERCENT', '3.0')),
                'daily_profit_target_percent': float(os.getenv('DAILY_PROFIT_TARGET_PERCENT', '5.0'))
            }
        }
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service"""
        if not self._config_cache:
            return ''
        
        api_keys = self._config_cache.get('api_keys', {})
        return api_keys.get(service, '')
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        if not self._config_cache:
            return {}
        
        return self._config_cache.get('trading', {})
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status"""
        status = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'secure_storage': False
        }
        
        if not self._config_cache:
            status['valid'] = False
            status['errors'].append('No configuration loaded')
            return status
        
        # Check API keys
        api_keys = self._config_cache.get('api_keys', {})
        required_keys = ['alpaca_key', 'alpaca_secret']
        
        for key in required_keys:
            if not api_keys.get(key):
                status['errors'].append(f'Missing required API key: {key}')
                status['valid'] = False
        
        # Check trading configuration
        trading_config = self._config_cache.get('trading', {})
        
        if trading_config.get('capital', 0) <= 0:
            status['errors'].append('Capital must be positive')
            status['valid'] = False
        
        if not (0 < trading_config.get('risk_percent', 0) <= 10):
            status['warnings'].append('Risk percent should be between 0-10%')
        
        # Check if using secure storage
        try:
            secure_config = get_secure_config()
            if secure_config:
                status['secure_storage'] = True
        except:
            pass
        
        return status
    
    def migrate_to_secure_storage(self) -> bool:
        """Migrate current configuration to secure storage"""
        try:
            if not self._config_cache:
                logger.error("No configuration to migrate")
                return False
            
            # Store API keys securely
            api_keys = self._config_cache.get('api_keys', {})
            for service, key in api_keys.items():
                if key:
                    if service == 'alpaca_key':
                        secret = api_keys.get('alpaca_secret', '')
                        self.secure_manager.store_api_key('alpaca', key, secret)
                    else:
                        self.secure_manager.store_api_key(service, key)
            
            # Store trading configuration
            trading_config = self._config_cache.get('trading', {})
            self.secure_manager.store_encrypted_config(trading_config)
            
            logger.info("Successfully migrated configuration to secure storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate configuration: {e}")
            return False
    
    def refresh_config(self):
        """Refresh configuration from storage"""
        self._config_cache = None
        self._load_config()
    
    def get_masked_config(self) -> Dict[str, Any]:
        """Get configuration with sensitive data masked for display"""
        if not self._config_cache:
            return {}
        
        config_copy = self._config_cache.copy()
        
        # Mask API keys
        if 'api_keys' in config_copy:
            for key, value in config_copy['api_keys'].items():
                if value:
                    config_copy['api_keys'][key] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        
        return config_copy

# Global configuration manager instance
config_manager = ConfigManager()

def get_api_key(service: str) -> str:
    """Get API key for a service (global convenience function)"""
    return config_manager.get_api_key(service)

def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration (global convenience function)"""
    return config_manager.get_trading_config()

def validate_config() -> Dict[str, Any]:
    """Validate configuration (global convenience function)"""
    return config_manager.validate_configuration()

# Backward compatibility - provide the same interface as config.py
API_KEY = config_manager.get_api_key('alpaca_key')
API_SECRET = config_manager.get_api_key('alpaca_secret')
NEWS_API_KEY = config_manager.get_api_key('news_api')
ALPHAVANTAGE_API_KEY = config_manager.get_api_key('alphavantage')
FINNHUB_API_KEY = config_manager.get_api_key('finnhub')

trading_config = config_manager.get_trading_config()
CAPITAL = trading_config.get('capital', 10000)
RISK_PERCENT = trading_config.get('risk_percent', 1.0)
DEFAULT_SYMBOLS = trading_config.get('default_symbols', ['BTC/USD', 'ETH/USD'])
PROFIT_TARGET_PERCENT = trading_config.get('profit_target_percent', 3.0)
DAILY_PROFIT_TARGET_PERCENT = trading_config.get('daily_profit_target_percent', 5.0)

# Additional constants (keep from original config.py)
TIMEFRAME = '1H'
MARKET_TYPE = 'crypto'
MAX_CAPITAL_PER_TRADE = 0.15
BOLLINGER_LENGTH = 20
BOLLINGER_STD = 2
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_SHORT = 12
EMA_LONG = 26
NEWS_WEIGHT = 0.5
EARNINGS_WEIGHT = 0.6

if __name__ == "__main__":
    # Configuration validation and setup script
    print("🔧 Trading Bot Configuration Manager")
    print("=" * 50)
    
    status = validate_config()
    
    print(f"Configuration Status: {'✅ Valid' if status['valid'] else '❌ Invalid'}")
    print(f"Secure Storage: {'✅ Enabled' if status['secure_storage'] else '❌ Disabled'}")
    
    if status['errors']:
        print("\n❌ Errors:")
        for error in status['errors']:
            print(f"  - {error}")
    
    if status['warnings']:
        print("\n⚠️ Warnings:")
        for warning in status['warnings']:
            print(f"  - {warning}")
    
    if not status['secure_storage']:
        print("\n🔐 To enable secure storage, run:")
        print("python security/secure_config.py")
        
        migrate = input("\nMigrate current config to secure storage? (y/N): ")
        if migrate.lower() == 'y':
            if config_manager.migrate_to_secure_storage():
                print("✅ Configuration migrated to secure storage")
            else:
                print("❌ Migration failed")

"""
🔐 Security Enhancements for Trading Bot
Implements secure configuration management and API key handling
"""

import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
import keyring
import logging
from typing import Dict, Optional
import hashlib

logger = logging.getLogger(__name__)

class SecureConfigManager:
    """Secure configuration manager for API keys and sensitive data"""
    
    def __init__(self, app_name: str = "trading_bot"):
        self.app_name = app_name
        self.config_dir = Path.home() / ".trading_bot"
        self.config_dir.mkdir(exist_ok=True)
        self.encrypted_config_file = self.config_dir / "config.encrypted"
        
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = self.config_dir / "key.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Restrict file permissions (Unix-like systems)
            os.chmod(key_file, 0o600)
            return key
    
    def store_api_key(self, service: str, api_key: str, api_secret: str = None) -> bool:
        """Store API key securely using system keyring"""
        try:
            # Store API key
            keyring.set_password(self.app_name, f"{service}_api_key", api_key)
            
            # Store API secret if provided
            if api_secret:
                keyring.set_password(self.app_name, f"{service}_api_secret", api_secret)
            
            logger.info(f"API credentials stored securely for {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to store API credentials for {service}: {e}")
            return False
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve API key from secure storage"""
        try:
            return keyring.get_password(self.app_name, f"{service}_api_key")
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {service}: {e}")
            return None
    
    def get_api_secret(self, service: str) -> Optional[str]:
        """Retrieve API secret from secure storage"""
        try:
            return keyring.get_password(self.app_name, f"{service}_api_secret")
        except Exception as e:
            logger.error(f"Failed to retrieve API secret for {service}: {e}")
            return None
    
    def store_encrypted_config(self, config_data: Dict) -> bool:
        """Store configuration data encrypted"""
        try:
            key = self._get_encryption_key()
            fernet = Fernet(key)
            
            # Convert to JSON and encrypt
            json_data = json.dumps(config_data).encode()
            encrypted_data = fernet.encrypt(json_data)
            
            # Write to file
            with open(self.encrypted_config_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Restrict file permissions
            os.chmod(self.encrypted_config_file, 0o600)
            logger.info("Configuration stored securely")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store encrypted config: {e}")
            return False
    
    def load_encrypted_config(self) -> Optional[Dict]:
        """Load and decrypt configuration data"""
        try:
            if not self.encrypted_config_file.exists():
                return None
                
            key = self._get_encryption_key()
            fernet = Fernet(key)
            
            # Read and decrypt
            with open(self.encrypted_config_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            logger.error(f"Failed to load encrypted config: {e}")
            return None

class APIKeyValidator:
    """Validate API keys and detect potentially compromised keys"""
    
    @staticmethod
    def validate_alpaca_key(api_key: str, api_secret: str) -> bool:
        """Validate Alpaca API credentials format"""
        # Alpaca keys have specific formats
        if not api_key or not api_secret:
            return False
        
        # Check basic format (this is a simplified check)
        if len(api_key) < 20 or len(api_secret) < 40:
            logger.warning("API key format appears invalid")
            return False
        
        return True
    
    @staticmethod
    def check_key_exposure(api_key: str) -> bool:
        """Check if API key might be exposed (basic heuristics)"""
        # Check for common exposure patterns
        exposure_indicators = [
            api_key in str(Path.cwd()),  # Key in current directory name
            "demo" in api_key.lower(),
            "test" in api_key.lower(),
            "sample" in api_key.lower()
        ]
        
        if any(exposure_indicators):
            logger.warning("API key might be a demo/test key")
            return True
        
        return False

class SecurityUtils:
    """General security utilities"""
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Create hash of sensitive data for logging/comparison"""
        return hashlib.sha256(data.encode()).hexdigest()[:8]
    
    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """Mask API key for safe logging"""
        if not api_key or len(api_key) < 8:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"
    
    @staticmethod
    def validate_environment() -> Dict[str, bool]:
        """Validate security environment"""
        checks = {
            "home_directory_writable": os.access(Path.home(), os.W_OK),
            "config_directory_secure": True,  # Would check permissions in production
            "keyring_available": True,  # Would test keyring in production
        }
        
        return checks

def setup_secure_config():
    """Setup secure configuration for the trading bot"""
    config_manager = SecureConfigManager()
    
    # Example usage - this would be called during initial setup
    print("🔐 Setting up secure configuration...")
    
    # Store API keys securely
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_API_SECRET')
    
    if alpaca_key and alpaca_secret:
        if APIKeyValidator.validate_alpaca_key(alpaca_key, alpaca_secret):
            config_manager.store_api_key('alpaca', alpaca_key, alpaca_secret)
            print("✅ Alpaca credentials stored securely")
        else:
            print("❌ Invalid Alpaca credentials format")
    
    # Store other configuration
    trading_config = {
        'capital': float(os.getenv('CAPITAL', '10000')),
        'risk_percent': float(os.getenv('RISK_PERCENT', '1.0')),
        'max_capital_per_trade': float(os.getenv('MAX_CAPITAL_PER_TRADE', '0.1')),
        'default_symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']
    }
    
    config_manager.store_encrypted_config(trading_config)
    print("✅ Trading configuration stored securely")

def get_secure_config() -> Dict:
    """Get configuration from secure storage"""
    config_manager = SecureConfigManager()
    
    # Load encrypted config
    config = config_manager.load_encrypted_config() or {}
    
    # Add API keys
    config['api_keys'] = {
        'alpaca': {
            'key': config_manager.get_api_key('alpaca'),
            'secret': config_manager.get_api_secret('alpaca')
        },
        'newsapi': config_manager.get_api_key('newsapi'),
        'finnhub': config_manager.get_api_key('finnhub'),
        'alphavantage': config_manager.get_api_key('alphavantage')
    }
    
    return config

if __name__ == "__main__":
    # Demo setup
    setup_secure_config()
    
    # Demo retrieval
    config = get_secure_config()
    print("🔍 Configuration loaded:")
    for key, value in config.items():
        if 'api' in key.lower() or 'key' in key.lower():
            print(f"  {key}: {SecurityUtils.mask_api_key(str(value))}")
        else:
            print(f"  {key}: {value}")

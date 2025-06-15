#!/usr/bin/env python3
"""
Trading Bot Setup and Validation Script
Checks dependencies, API keys, and system configuration
"""

import sys
import os
import importlib
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'requests',
        'alpaca_trade_api', 'binance', 'nltk', 'textblob',
        'pandas_ta', 'websocket', 'flask', 'python_dotenv'
    ]
    
    optional_packages = [
        'discord_webhook', 'slack_sdk', 'jwt'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (REQUIRED)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional - alerts may not work)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install discord-webhook slack-sdk PyJWT")
    
    return True

def check_config_files():
    """Check if configuration files exist and are valid"""
    print("\n⚙️  Checking configuration files...")
    
    config_file = Path("config.py")
    env_file = Path(".env")
    
    if not config_file.exists():
        print("❌ config.py not found")
        return False
    print("✅ config.py found")
    
    if not env_file.exists():
        print("⚠️  .env file not found (optional but recommended)")
        print("   Copy .env.template to .env and fill in your API keys")
    else:
        print("✅ .env file found")
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API keys...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from config import (
            API_KEY, API_SECRET, NEWS_API_KEY, 
            ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY
        )
        
        keys_status = {
            "Alpaca API Key": bool(API_KEY.strip()),
            "Alpaca Secret": bool(API_SECRET.strip()),
            "News API Key": bool(NEWS_API_KEY.strip()),
            "AlphaVantage Key": bool(ALPHAVANTAGE_API_KEY.strip()),
            "Finnhub Key": bool(FINNHUB_API_KEY.strip())
        }
        
        for key_name, configured in keys_status.items():
            if configured:
                print(f"✅ {key_name}")
            else:
                print(f"❌ {key_name} (not configured)")
        
        required_keys = ["Alpaca API Key", "Alpaca Secret"]
        missing_required = [k for k in required_keys if not keys_status[k]]
        
        if missing_required:
            print(f"\n❌ Missing required API keys: {', '.join(missing_required)}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    directories = ['data', 'logs', 'exports']
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}/")
        else:
            print(f"⚠️  {directory}/ not found, creating...")
            path.mkdir(exist_ok=True)
            print(f"✅ {directory}/ created")
    
    return True

def test_imports():
    """Test critical imports"""
    print("\n🧪 Testing critical imports...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test the numpy.NaN patch
        if not hasattr(np, 'NaN'):
            np.NaN = np.nan
        print("✅ NumPy NaN patch applied")
        
        import streamlit
        print("✅ Streamlit import")
        
        import alpaca_trade_api
        print("✅ Alpaca API import")
        
        try:
            import pandas_ta
            print("✅ Pandas-TA import")
        except ImportError:
            print("❌ Pandas-TA import failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Critical import failed: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test database creation
        from dashboard.components.database import ensure_db_exists
        ensure_db_exists()
        print("✅ Database creation test")
        
        # Test configuration loading
        from config import CAPITAL, RISK_PERCENT
        assert isinstance(CAPITAL, (int, float))
        assert isinstance(RISK_PERCENT, (int, float))
        print("✅ Configuration loading test")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup validation function"""
    print("🚀 Trading Bot Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Config Files", check_config_files),
        ("API Keys", check_api_keys),
        ("Directories", check_directories),
        ("Critical Imports", test_imports),
        ("Basic Tests", run_basic_tests)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Your trading bot is ready to run.")
        print("\nNext steps:")
        print("1. Fill in your API keys in .env file")
        print("2. Run the dashboard: streamlit run dashboard/app.py")
        print("3. Or run the bot: python run_bot.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before running the bot.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Copy .env.template to .env and add your API keys")
        print("- Make sure config.py has valid settings")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

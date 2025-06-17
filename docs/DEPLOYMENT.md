#  Trading Bot Deployment Guide

## **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- Git (for cloning repository)
- API keys for trading and news services

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run setup validation
python setup_check.py
```

### **Configuration**

1. **Copy the example configuration:**
   ```bash
   cp config_example.py config.py
   ```

2. **Edit configuration with your API keys:**
   ```python
   # API Keys (Required)
   API_KEY = "your_alpaca_api_key"
   API_SECRET = "your_alpaca_secret_key"
   NEWS_API_KEY = "your_news_api_key"
   FINNHUB_API_KEY = "your_finnhub_api_key"
   
   # Trading Settings
   CAPITAL = 10000.0           # Starting capital
   RISK_PERCENT = 1.0          # Risk per trade (%)
   DEFAULT_SYMBOLS = ['BTC/USD', 'ETH/USD']
   ```

3. **Set up secure configuration (recommended):**
   ```python
   from security.secure_config import setup_secure_config
   setup_secure_config()
   ```

### **Running the Bot**

#### **Option 1: Dashboard Mode**
```bash
streamlit run dashboard/app.py
```
Access at: http://localhost:8501

#### **Option 2: Automated Trading**
```bash
python run_auto_trader.py --symbols BTC/USD ETH/USD --capital 10000
```

#### **Option 3: Manual Bot**
```bash
python main.py
```

---

## **Production Deployment**

### **Environment Setup**

#### **1. Server Requirements**
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for logs and data
- **Network**: Stable internet connection

#### **2. Python Environment**
```bash
# Install Python 3.8+
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# Create production user
sudo useradd -m -s /bin/bash tradingbot
sudo su - tradingbot

# Setup application
git clone <repository-url> /home/tradingbot/trading-bot
cd /home/tradingbot/trading-bot
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **3. Configuration Management**
```bash
# Create configuration directory
mkdir -p /etc/tradingbot

# Copy and secure configuration
sudo cp config_example.py /etc/tradingbot/config.py
sudo chown tradingbot:tradingbot /etc/tradingbot/config.py
sudo chmod 600 /etc/tradingbot/config.py

# Set environment variables
cat > /etc/tradingbot/environment << EOF
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
NEWS_API_KEY=your_news_key
FINNHUB_API_KEY=your_finnhub_key
CAPITAL=10000
RISK_PERCENT=1.0
EOF

sudo chmod 600 /etc/tradingbot/environment
```

### **Service Configuration**

#### **1. Systemd Service**
Create `/etc/systemd/system/tradingbot.service`:

```ini
[Unit]
Description=Automated Trading Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=tradingbot
WorkingDirectory=/home/tradingbot/trading-bot
EnvironmentFile=/etc/tradingbot/environment
ExecStart=/home/tradingbot/trading-bot/venv/bin/python run_auto_trader.py
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=tradingbot

[Install]
WantedBy=multi-user.target
```

#### **2. Enable and Start Service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot

# Check status
sudo systemctl status tradingbot
```

#### **3. Dashboard Service** (Optional)
Create `/etc/systemd/system/tradingbot-dashboard.service`:

```ini
[Unit]
Description=Trading Bot Dashboard
After=network.target

[Service]
Type=simple
Restart=always
User=tradingbot
WorkingDirectory=/home/tradingbot/trading-bot
EnvironmentFile=/etc/tradingbot/environment
ExecStart=/home/tradingbot/trading-bot/venv/bin/python -m streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=tradingbot-dashboard

[Install]
WantedBy=multi-user.target
```

### **Logging Configuration**

#### **1. Centralized Logging**
```bash
# Create log directory
sudo mkdir -p /var/log/tradingbot
sudo chown tradingbot:tradingbot /var/log/tradingbot

# Configure log rotation
cat > /etc/logrotate.d/tradingbot << EOF
/var/log/tradingbot/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 tradingbot tradingbot
}
EOF
```

#### **2. Update Logging Config**
Edit `utils/logging_config.py`:

```python
def setup():
    """Set up logging with file and console handlers"""
    # Use production log directory
    logs_dir = Path("/var/log/tradingbot")
    logs_dir.mkdir(exist_ok=True)
    
    # Rest of configuration...
```

### **Security Hardening**

#### **1. Firewall Configuration**
```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow dashboard (if needed)
sudo ufw allow 8501/tcp

# Check status
sudo ufw status
```

#### **2. API Key Security**
```bash
# Use encrypted configuration
python -c "
from security.secure_config import setup_secure_config
setup_secure_config()
"

# Remove plain text keys
sudo rm /etc/tradingbot/environment
```

#### **3. File Permissions**
```bash
# Secure application files
chmod -R 755 /home/tradingbot/trading-bot
chmod 600 /home/tradingbot/trading-bot/config.py

# Secure log files
chmod 755 /var/log/tradingbot
chmod 644 /var/log/tradingbot/*.log
```

### **Monitoring Setup**

#### **1. Health Check Script**
Create `/home/tradingbot/health_check.py`:

```python
#!/usr/bin/env python3
"""Health check script for trading bot"""

import sys
import requests
import subprocess
from datetime import datetime, timedelta

def check_service_status():
    """Check if trading bot service is running"""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'tradingbot'], 
                              capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except:
        return False

def check_dashboard():
    """Check if dashboard is accessible"""
    try:
        response = requests.get('http://localhost:8501', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_recent_logs():
    """Check for recent log entries"""
    try:
        with open('/var/log/tradingbot/trading.log', 'r') as f:
            lines = f.readlines()
            if lines:
                # Check if last log entry is within last 10 minutes
                # Implementation depends on log format
                return True
    except:
        pass
    return False

def main():
    checks = {
        'service': check_service_status(),
        'dashboard': check_dashboard(),
        'logs': check_recent_logs()
    }
    
    all_healthy = all(checks.values())
    
    print(f"Health Check - {datetime.now()}")
    for check, status in checks.items():
        print(f"  {check}: {'✅' if status else '❌'}")
    
    sys.exit(0 if all_healthy else 1)

if __name__ == "__main__":
    main()
```

#### **2. Cron Jobs for Monitoring**
```bash
# Add to crontab
crontab -e

# Health check every 5 minutes
*/5 * * * * /home/tradingbot/trading-bot/venv/bin/python /home/tradingbot/health_check.py >> /var/log/tradingbot/health.log 2>&1

# Daily log rotation
0 0 * * * /usr/sbin/logrotate /etc/logrotate.d/tradingbot
```

### **Backup Strategy**

#### **1. Configuration Backup**
```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/home/tradingbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    /etc/tradingbot/ \
    /home/tradingbot/trading-bot/config.py \
    /home/tradingbot/trading-bot/data/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "config_*.tar.gz" -mtime +7 -delete
```

#### **2. Database Backup**
```bash
#!/bin/bash
# backup_data.sh

BACKUP_DIR="/home/tradingbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup trading database
cp /home/tradingbot/trading-bot/dashboard/data/trading_bot.db \
   $BACKUP_DIR/trading_bot_$DATE.db

# Compress old backups
find $BACKUP_DIR -name "trading_bot_*.db" -mtime +1 -exec gzip {} \;

# Keep only last 30 days
find $BACKUP_DIR -name "trading_bot_*.db.gz" -mtime +30 -delete
```

### **Performance Optimization**

#### **1. System Tuning**
```bash
# Increase file descriptor limits
echo "tradingbot soft nofile 65536" >> /etc/security/limits.conf
echo "tradingbot hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.rmem_default = 262144" >> /etc/sysctl.conf
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_default = 262144" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf

sysctl -p
```

#### **2. Python Optimization**
```bash
# Install performance packages
pip install cython numba

# Set Python optimizations
export PYTHONOPTIMIZE=2
echo "export PYTHONOPTIMIZE=2" >> /home/tradingbot/.bashrc
```

### **Troubleshooting**

#### **Common Issues**

1. **Service won't start**
   ```bash
   sudo journalctl -u tradingbot -f
   sudo systemctl status tradingbot
   ```

2. **API connection errors**
   ```bash
   # Test API connectivity
   python -c "
   from config import API_KEY, API_SECRET
   import alpaca_trade_api as tradeapi
   api = tradeapi.REST(API_KEY, API_SECRET)
   print(api.get_account())
   "
   ```

3. **High memory usage**
   ```bash
   # Monitor memory usage
   ps aux | grep python
   top -p $(pgrep -f tradingbot)
   ```

4. **Log file issues**
   ```bash
   # Check log permissions
   ls -la /var/log/tradingbot/
   
   # Check disk space
   df -h
   ```

#### **Emergency Procedures**

1. **Stop all trading**
   ```bash
   sudo systemctl stop tradingbot
   # Manually close positions via dashboard or API
   ```

2. **Rollback deployment**
   ```bash
   cd /home/tradingbot/trading-bot
   git checkout previous-version-tag
   sudo systemctl restart tradingbot
   ```

3. **Restore from backup**
   ```bash
   cd /home/tradingbot/backups
   tar -xzf config_YYYYMMDD_HHMMSS.tar.gz -C /
   sudo systemctl restart tradingbot
   ```

---

## **Maintenance**

### **Regular Tasks**

#### **Daily**
- Monitor log files for errors
- Check trading performance
- Verify API connectivity
- Review risk metrics

#### **Weekly**
- Update market data cache
- Review and optimize strategies
- Check system resource usage
- Test backup and restore procedures

#### **Monthly**
- Update dependencies
- Security audit
- Performance optimization
- Documentation updates

### **Updates and Upgrades**

```bash
# Update trading bot
cd /home/tradingbot/trading-bot
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl restart tradingbot
sudo systemctl restart tradingbot-dashboard
```

### **Monitoring Dashboards**

Access monitoring information:
- **Trading Dashboard**: http://your-server:8501
- **System Logs**: `/var/log/tradingbot/`
- **Service Status**: `systemctl status tradingbot`
- **Health Checks**: `/var/log/tradingbot/health.log`

---

This deployment guide ensures a robust, secure, and maintainable production environment for the trading bot.

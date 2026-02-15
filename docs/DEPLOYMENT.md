# Deployment

How to get the bot running locally or on a server.

## Local setup

```bash
git clone <repo-url>
cd trading-bot
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python setup_check.py
```

### Config

1. Copy the example config:
   ```bash
   cp config_example.py config.py
   ```

2. Fill in your API keys in `config.py`:
   ```python
   API_KEY = "your_alpaca_key"
   API_SECRET = "your_alpaca_secret"
   NEWS_API_KEY = "your_news_key"       # optional
   FINNHUB_API_KEY = "your_finnhub_key" # optional
   ```

3. Or use the secure config manager:
   ```python
   from security.secure_config import setup_secure_config
   setup_secure_config()
   ```

### Running

Dashboard mode (recommended for getting started):
```bash
streamlit run dashboard/app.py
```

Automated trading:
```bash
python run_auto_trader.py --symbols BTC/USD ETH/USD --capital 10000
```

---

## Production (Linux server)

### Requirements
- 2+ cores, 4GB RAM minimum
- Python 3.8+
- Stable internet

### Setup

```bash
sudo useradd -m -s /bin/bash tradingbot
sudo su - tradingbot

git clone <repo-url> ~/trading-bot
cd ~/trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Systemd service

Create `/etc/systemd/system/tradingbot.service`:
```ini
[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=1
User=tradingbot
WorkingDirectory=/home/tradingbot/trading-bot
ExecStart=/home/tradingbot/trading-bot/venv/bin/python run_auto_trader.py

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot
```

You can do the same for the dashboard if you want it running as a service.

### Logging

```bash
sudo mkdir -p /var/log/tradingbot
sudo chown tradingbot:tradingbot /var/log/tradingbot
```

Set up logrotate at `/etc/logrotate.d/tradingbot`:
```
/var/log/tradingbot/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### Firewall

```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8501/tcp  # dashboard
```

### Backups

Back up config and the SQLite DB regularly:
```bash
tar -czf ~/backups/config_$(date +%Y%m%d).tar.gz config.py data/
cp dashboard/data/trading_bot.db ~/backups/trading_bot_$(date +%Y%m%d).db
```

---

## Troubleshooting

**Service won't start:**
```bash
sudo journalctl -u tradingbot -f
```

**API errors:**
Double check your keys in config.py. Try:
```bash
python -c "from config import API_KEY; print(bool(API_KEY))"
```

**No trades happening:**
Check signal thresholds and risk limits. The bot won't trade if confidence is too low or daily limits are hit.

**Logs:**
```bash
tail -f logs/trading_*.log
```

## Maintenance

- Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- Rotate API keys periodically
- Monitor disk space (logs and cache can grow)
- Check the dashboard for performance metrics

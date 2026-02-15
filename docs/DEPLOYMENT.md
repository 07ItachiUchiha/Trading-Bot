# Deployment

How to run the prediction platform locally or on a server.

## Local setup

```bash
git clone <repo-url>
cd market-prediction-platform
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

Headless prediction runtime:

```bash
python run_prediction_engine.py --symbols BTC/USD ETH/USD --capital 10000
```

---

## Production (Linux server)

### Requirements

- 2+ cores, 4GB RAM minimum
- Python 3.8+
- Stable internet

### Setup

```bash
sudo useradd -m -s /bin/bash predictionbot
sudo su - predictionbot

git clone <repo-url> ~/market-prediction-platform
cd ~/market-prediction-platform
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Systemd service

Create `/etc/systemd/system/predictionbot.service`:

```ini
[Unit]
Description=Market Prediction Service
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=1
User=predictionbot
WorkingDirectory=/home/predictionbot/market-prediction-platform
ExecStart=/home/predictionbot/market-prediction-platform/venv/bin/python run_prediction_engine.py

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable predictionbot
sudo systemctl start predictionbot
```

You can do the same for the dashboard if you want it running as a service.

### Logging

```bash
sudo mkdir -p /var/log/predictionbot
sudo chown predictionbot:predictionbot /var/log/predictionbot
```

Set up logrotate at `/etc/logrotate.d/predictionbot`:

```
/var/log/predictionbot/*.log {
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
cp dashboard/data/prediction_platform.db ~/backups/prediction_runtime_$(date +%Y%m%d).db
```

---

## Troubleshooting

**Service won't start:**

```bash
sudo journalctl -u predictionbot -f
```

**API errors:**
Double check your keys in config.py. Try:

```bash
python -c "from config import API_KEY; print(bool(API_KEY))"
```

**No actionable signals:**
Check signal thresholds and data connectivity. The runtime will stay neutral if confidence is too low.

**Logs:**

```bash
tail -f logs/prediction_*.log
```

## Maintenance

- Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- Rotate API keys periodically
- Monitor disk space (logs and cache can grow)
- Check the dashboard for performance metrics

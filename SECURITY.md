# Security Policy

## Reporting a Vulnerability

If you find a security issue (leaked keys, auth bypass, etc.), please open a private issue or email me directly.
Don't post it publicly until there's a fix.

## API Key Safety

- Never commit real API keys to the repo
- Use `.env` or the secure config manager (`security/secure_config.py`)
- The `.gitignore` already covers `.env` and `config.py`

## What's covered

- Alpaca / Binance credential handling
- Dashboard session auth
- Config encryption via keyring

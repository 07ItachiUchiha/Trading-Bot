# Automated Crypto Trading Bot

An algorithmic trading bot for cryptocurrency and stock markets with multiple trading strategy support, technical analysis, sentiment analysis, and a Streamlit dashboard for visualization and control.

## Features

- 📊 **Multiple Trading Strategies**:
  - RSI + EMA Crossover
  - Bollinger Bands + RSI
  - Breakout Detection
  - Switch between strategies from the dashboard

- 🔍 **Advanced Analysis**:
  - Technical indicators (RSI, EMA, Bollinger Bands)
  - Sentiment analysis from news sources
  - Earnings report integration
  - Real-time market data via websockets

- 💰 **Risk Management**:
  - Position sizing based on risk percentage
  - Stop-loss and take-profit management
  - Trailing stops
  - Daily profit/loss limits

- 📈 **Comprehensive Dashboard**:
  - Streamlit-based web interface
  - Interactive charts with indicators
  - PnL visualization and performance metrics
  - Trade history with filtering capabilities
  - Strategy selection and configuration

- 🔄 **Backtesting**:
  - Test strategies on historical data
  - Performance comparison
  - Optimization options

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Copy `config_example.py` to `config.py`
   - Add your API keys for Alpaca, Binance, etc.

## Usage

1. Start the Streamlit dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

2. Run the automated trader:
   ```bash
   python run_bot.py
   ```

3. Access the dashboard at http://localhost:8501

## Configuration

Edit `config.py` to customize:
- Trading pairs
- Risk parameters
- API endpoints
- Strategy parameters
- Notification settings

## Project Structure

```
trading-bot/
├── dashboard/            # Streamlit dashboard
│   ├── app.py            # Main dashboard application
│   ├── pages/            # Dashboard pages
│   └── components/       # UI components
├── strategy/             # Trading strategies
│   ├── auto_trader.py    # Automated trading logic
│   └── multiple_strategies.py # Strategy implementations
├── utils/                # Utility functions
│   ├── risk_manager.py   # Risk management
│   └── sentiment_analyzer.py # News sentiment analysis
├── logs/                 # Log files
├── exports/              # Exported data
├── data/                 # Historical data
├── config.py             # Configuration
└── run_bot.py            # Entry point script
```

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Alpaca Trade API
- Python-Binance

## License

MIT License

## Disclaimer

This software is for educational purposes only. Use it at your own risk. The authors are not responsible for any financial losses incurred from using this software.

# NIFTY Scalping Bot

An automated **NIFTY50 Options Scalping Bot** built using [Fyers API](https://myapi.fyers.in/docs/) (WebSocket + REST).  
The bot constructs real-time 5-second OHLC candles, applies EMA/SMA crossover strategies, and executes trades on At-The-Money (ATM) options.

## 🚀 Features
- Fetches historical and live data using Fyers API.
- Builds real-time 5-second OHLC candles from WebSocket ticks.
- Incremental **Fast Moving Average** (e.g., EMA(9)) and **Slow Moving Average** (e.g., SMA(9)) updates. 
- Scalping strategy with CE/PE entries and exits.
- Cooldown logic to prevent over-trading.
- Trade logs stored in CSV.
- ✅ **Paper Trading Only** (no live orders are placed).

## 📂 Project Structure
- scalping_bot.py # Main trading bot code
- README.md

## ⚡ Requirements
- Python 3.9+
- [Fyers API v3](https://myapi.fyers.in/docs/)
- [TA-Lib](https://github.com/ta-lib/ta-lib-python)
- pandas, numpy, pytz

Install dependencies:
```bash
pip install fyers-apiv3 pandas numpy pytz TA-Lib
```

## ▶️ Usage
1. Generate your Fyers access token and save it in access_token.txt.
2. Update credentials.py with your client_id.
3. Run the bot:
```bash
python scalping_bot.py
```

## ⚠️ Disclaimer

This project is for educational purposes only.
It performs **paper trading only** and does not place real trades on the exchange.
Trading in financial markets involves substantial risk of loss and is not suitable for all investors.
Use this strategy at your own risk.

## ⚠️ Important
- This bot currently performs **paper trading only** and does **not** place live orders.  
- To enable live trading, you must refer to the official [Fyers API Documentation](https://myapi.fyers.in/docs/) and update the order placement logic.  
- Do not copy the SMA/FMA values blindly. Always perform your own research and update the strategy parameters before live trading.

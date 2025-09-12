"""
NIFTY Options Scalping Bot using Fyers API

This script connects to the Fyers WebSocket, retrieves live NIFTY50 data, builds 5-second OHLC candles,
and applies EMA(9) / SMA(9) strategies to generate BUY/SELL signals on ATM options.

Main Features:
- Historical data retrieval for indices and options.
- Real-time candle construction (5s) from WebSocket ticks.
- Incremental EMA and SMA updates.
- Automated entry/exit logic with cooldown periods.
- Trade logging into a CSV file.

Requirements:
- Fyers API v3 (REST + WebSocket)
- TOTP-enabled access token stored in `access_token.txt`
- TA-Lib, Pandas, NumPy, Pytz

Author: Vaibhav Saxena
"""

from fyers_apiv3.FyersWebsocket import data_ws
import datetime as dt
import pandas as pd
import pytz
from fyers_apiv3 import fyersModel
from credentials import client_id
import talib as ta
import numpy as np
import math

# -------------------------------
# Utility Functions
# -------------------------------

def historical_data(symbol, today):
    """
    Fetch historical candle data for a given symbol.

    Parameters:
        symbol (str): Trading symbol (e.g., "NSE:NIFTY50-INDEX").
        today (datetime.date): Current date.

    Returns:
        DataFrame: OHLC data indexed by datetime with no volume column.
    """
    data = {
        "symbol":symbol,
        "resolution":"5S",                              # 5-second candles
        "date_format":"1",                              # Epoch format
        "range_from":today,
        "range_to":today,
        "cont_flag":"1"
    }

    response = fyers.history(data=data)
    data = response['candles']
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='s')   # Convert epoch -> datetime

    # Convert UTC -> IST -> Naive datetime
    ist = pytz.timezone('Asia/Kolkata')
    df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(ist).dt.tz_localize(None)

    df.set_index('date', inplace=True)                  # Set date as index
    df.drop('volume', axis=1, inplace=True)             # Drop the volume column

    return df

def update_moving_averages(df, live_candle, period):
    """
    Incrementally update EMA and SMA using the last values from historical CSV and a new live candle.

    Parameters:
        df (data frame): Historical candles with columns ['open','high','low','close','ema_9','ema_15']
        live_candle (dict): New candle data {'open':..., 'high':..., 'low':..., 'close':...}

    Returns:
        updated_ema (float): Updated EMA 9
        updated_ema (float): Updated EMA 15
    """
    # Ensure df has enough rows for SMA calculation
    if len(df) < period:
        raise ValueError(f"CSV must have at least 15 rows for EMA calculation.")

    # Get last EMA
    last_ema = df[f'ema_{period}'].iloc[-2]

    # EMA update formula
    alpha = 2 / (period + 1)
    updated_ema = alpha * live_candle['close'] + (1 - alpha) * last_ema
    
    return updated_ema

def new_candle(df, message):
    """
    Convert incoming Fyers WS tick into a 5s candle, and update EMA(9)/EMA(15).

    Parameters:
        df (DataFrame): Candle DataFrame with OHLC + ema/sma.
        message (dict): Fyers WebSocket message containing 'ltp' and optional 'exch_feed_time'.

    Returns:
        DataFrame: Updated DataFrame with the new/updated candle.
    """
    # Parse price
    try:
        ltp = float(message["ltp"])  # WebSocket gives price as string
    except KeyError:
        print("Invalid message:", message)
        return df

    # Parse time from exchange if available (preferred)
    if "exch_feed_time" in message:
        try:
            timestamp = pd.to_datetime(int(message["exch_feed_time"]), unit="s")
        except Exception:
            timestamp = pd.Timestamp.utcnow()
    else:
        timestamp = pd.Timestamp.utcnow()

    # Normalize to IST and floor to 5s
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = timestamp.tz_localize("UTC").tz_convert(ist).tz_localize(None)
    timestamp_5s = timestamp.floor("5s")

    if timestamp_5s in df.index:
        # Update existing candle
        df.at[timestamp_5s, "close"] = ltp
        df.at[timestamp_5s, "high"] = max(df.at[timestamp_5s, "high"], ltp)
        df.at[timestamp_5s, "low"] = min(df.at[timestamp_5s, "low"], ltp)
    else:
        # Add new row
        df.loc[timestamp_5s] = [ltp, ltp, ltp, ltp, np.nan, np.nan]

    # Update EMA & SMA for the last candle
    updated_ema_9 = update_moving_averages(df, {"close": df.at[timestamp_5s, "close"]}, period=9)
    updated_ema_15 = update_moving_averages(df, {"close": df.at[timestamp_5s, "close"]}, period=15)
    df.at[timestamp_5s, "ema_9"] = updated_ema_9
    df.at[timestamp_5s, "ema_15"] = updated_ema_15

    return df

def get_atm_option(option_type, nifty_price):
    """
    Get At-The-Money (ATM) option symbol for NIFTY50.

    Parameters:
        option_type (str): "CE" or "PE".
        nifty_price (float): Current NIFTY50 price.

    Returns:
        str: ATM option symbol.
    """
    data = {
        "symbol":"NSE:NIFTY50-INDEX",
        "strikecount":1,
        "timestamp": ""
    }
    response = fyers.optionchain(data=data)
    
    # Filter CE or PE and find closest strike
    type_options = [option for option in response['data']['optionsChain'] if option['option_type'] == option_type]  # Getting options
    atm_option = min(type_options, key=lambda option: abs(option['strike_price'] - nifty_price))                        # ATM option
    
    return atm_option['symbol']

def log_trade(action, symbol, price):
    """
    Append trade details to a CSV log.

    Parameters:
        action (str): "BUY" or "SELL".
        symbol (str): Option symbol.
        price (float): Execution price.
    """
    with open("trades_log.csv", "a") as file:
        file.write(f"{dt.datetime.now()},{action},{symbol},{price}\n")

# -------------------------------
# Live WebSocket Handler Class
# -------------------------------
class LiveOHLC:
    """
    Handles live WebSocket data, builds candles, applies EMA/SMA strategy, and manages trades.
    """

    def __init__(self, dfs_hist):
        """
        Parameters:
            dfs_hist (dict): Mapping of {symbol: historical_dataframe}.
        """
        self.dfs = {sym: df for sym, df in dfs_hist.items()}
        self.trend = 0
        self.option = None
        self.position = False
        self.last_exit_time = None
        self.cooldown_seconds = 5   # Prevent re-entry for 5s after exit
        self.sl = None
        self.tp = None
        
    def onmessage(self, message):
        """
        Callback function to handle incoming messages from the FyersDataSocket WebSocket.

        Parameters:
            message (dict): The received message from the WebSocket.
        
        Implements scalping strategy logic.

        """
        symbol = message.get("symbol")
        if not symbol:
            return  # skip heartbeat/system messages
        
        if symbol not in self.dfs:
            return

        # Update NIFTY50 candles and trend
        if symbol == "NSE:NIFTY50-INDEX":
            self.dfs[symbol] = new_candle(self.dfs[symbol], message)
            if (self.dfs[symbol]['ema_9'].iloc[-2] > self.dfs[symbol]['ema_15'].iloc[-2] and
                5 * (self.dfs[symbol]['ema_15'].iloc[-2] - self.dfs[symbol]['ema_15'].iloc[-3]) > 1 / math.sqrt(3)):
                self.trend = 100
            elif (self.dfs[symbol]['ema_15'].iloc[-2] > self.dfs[symbol]['ema_9'].iloc[-2] and
                  5 * (self.dfs[symbol]['ema_15'].iloc[-3] - self.dfs[symbol]['ema_15'].iloc[-2]) > 1 / math.sqrt(3)):
                self.trend = -100
            else:
                self.trend = 0
        
        # Cooldown check
        if self.last_exit_time is not None:
            elapsed = (dt.datetime.now() - self.last_exit_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                print(f"Cooldown active, skipping entry. Wait {self.cooldown_seconds - int(elapsed)}s")
                return

        # Entry Logic (when no position is open)
        if not self.position:
            option_df = None
            """
            Attempt to enter CE or PE position based on trend and moving averages.
            """
            candle = (ta.CDLMARUBOZU(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']) +
                      ta.CDLCLOSINGMARUBOZU(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']) +
                      ta.CDLDRAGONFLYDOJI(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']) +
                      ta.CDLGRAVESTONEDOJI(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']) +
                      ta.CDLHAMMER(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']) -
                      ta.CDLINVERTEDHAMMER(self.dfs[symbol]['open'], self.dfs[symbol]['high'], self.dfs[symbol]['low'], self.dfs[symbol]['close']))
            
            if (self.trend > 10 and candle.iloc[-2] >= 1 and
                self.dfs[symbol]['close'].iloc[-2] > self.dfs[symbol]['ema_9'].iloc[-2] and
                self.dfs[symbol]['close'].iloc[-2] > self.dfs[symbol]['ema_15'].iloc[-2]):
                """
                Buy ATM option and subscribe to its live feed.
                """
                self.option = get_atm_option(option_type="CE", nifty_price=message.get("ltp"))
                response = fyers.quotes(data={"symbols":self.option})
                entry = response["d"][0]["v"]["ask"]

                print(entry)
                log_trade("BUY", self.option, entry)

                self.position = True
                fyers_socket.subscribe(symbols=[self.option], data_type="SymbolUpdate")

                # Initialize option DataFrame with historical + indicators
                option_df = historical_data(symbol=self.option, today=dt.date.today())
                self.sl = option_df["low"].iloc[-1]
                self.tp = entry + 1 * (entry - self.sl)
                option_df['ema_9'] = ta.EMA(option_df['close'], 9)
                option_df['ema_15'] = ta.EMA(option_df['close'], 15)
                self.dfs[self.option] = option_df
            elif (self.trend < -10 and candle.iloc[-2] <= -1 and
                  self.dfs[symbol]['close'].iloc[-2] < self.dfs[symbol]['ema_9'].iloc[-2] and
                  self.dfs[symbol]['close'].iloc[-2] < self.dfs[symbol]['ema_15'].iloc[-2]):
                """
                Buy ATM option and subscribe to its live feed.
                """
                self.option = get_atm_option(option_type="PE", nifty_price=message.get("ltp"))
                response = fyers.quotes(data={"symbols":self.option})
                entry = response["d"][0]["v"]["ask"]

                print(entry)
                log_trade("BUY", self.option, entry)
                
                self.position = True
                fyers_socket.subscribe(symbols=[self.option], data_type="SymbolUpdate")
                
                # Initialize option DataFrame with historical + indicators
                option_df = historical_data(symbol=self.option, today=dt.date.today())
                self.sl = option_df["low"].iloc[-1]
                self.tp = entry + 2 * (entry - self.sl)
                option_df['ema_9'] = ta.EMA(option_df['close'], 9)
                option_df['ema_15'] = ta.EMA(option_df['close'], 15)
                self.dfs[self.option] = option_df
        
        # Exit Logic (if in position)
        if self.option and symbol == self.option:
            """Attempt to exit position based on price action or MA crossover."""
            self.dfs[symbol] = new_candle(self.dfs[symbol], message)
            
            print(self.dfs[symbol].tail())
            print(self.trend)
            
            if self.option.endswith("CE"):
                """Exit logic for Call Options."""
                if self.trend > 10:
                    print("Price Action")

                    if message.get("ltp") <= self.sl or message.get("ltp") >= self.tp:
                        
                        response = fyers.quotes(data={"symbols":self.option})
                        print(response["d"][0]["v"]["bid"])
                        log_trade("SELL", self.option, response["d"][0]["v"]["bid"])
                        
                        if self.option in self.dfs:
                            fyers_socket.unsubscribe(symbols=[self.option], data_type="SymbolUpdate")
                            self.dfs.pop(self.option)
                        
                        self.position = False
                        self.option = None
                        self.last_exit_time = dt.datetime.now()
                    else:
                        print('holding')
                
                else:
                    print('MA crossover')

                    if message.get("ltp") <= self.sl or message.get("ltp") >= self.tp:
                        
                        response = fyers.quotes(data={"symbols":self.option})
                        print(response["d"][0]["v"]["bid"])
                        log_trade("SELL", self.option, response["d"][0]["v"]["bid"])
                        
                        if self.option in self.dfs:
                            fyers_socket.unsubscribe(symbols=[self.option], data_type="SymbolUpdate")
                            self.dfs.pop(self.option)
                        
                        self.position = False
                        self.option = None
                        self.last_exit_time = dt.datetime.now()
                    else:
                        print('holding')
            elif self.option.endswith("PE"):
                """Exit logic for Put Options."""
                if self.trend < -10:
                    print("Price Action")
                    
                    if message.get("ltp") <= self.sl or message.get("ltp") >= self.tp:
                        
                        response = fyers.quotes(data={"symbols":self.option})
                        print(response["d"][0]["v"]["bid"])
                        log_trade("SELL", self.option, response["d"][0]["v"]["bid"])
                        
                        if self.option in self.dfs:
                            fyers_socket.unsubscribe(symbols=[self.option], data_type="SymbolUpdate")
                            self.dfs.pop(self.option)
                        
                        self.position = False
                        self.option = None
                        self.last_exit_time = dt.datetime.now()
                    else:
                        print('holding')
                
                else:
                    print('MA crossover')

                    if message.get("ltp") <= self.sl or message.get("ltp") >= self.tp:
                        
                        response = fyers.quotes(data={"symbols":self.option})
                        print(response["d"][0]["v"]["bid"])
                        log_trade("SELL", self.option, response["d"][0]["v"]["bid"])
                        
                        if self.option in self.dfs:
                            fyers_socket.unsubscribe(symbols=[self.option], data_type="SymbolUpdate")
                            self.dfs.pop(self.option)
                        
                        self.position = False
                        self.option = None
                        self.last_exit_time = dt.datetime.now()
                    else:
                        print('holding')


    def onerror(self, message):
        """
        Callback function to handle WebSocket errors.

        Parameters:
            message (dict): The error message received from the WebSocket.


        """
        print("Error:", message)


    def onclose(self, message):
        """
        Callback function to handle WebSocket connection close events.
        """
        print("Connection closed:", message)


    def onopen(self):
        """
        Callback function to subscribe to data type and symbols upon WebSocket connection.

        """
        # Specify the data type and symbols you want to subscribe to
        data_type = "SymbolUpdate"

        # Subscribe to the specified symbols and data type
        symbols = ["NSE:NIFTY50-INDEX"]
        fyers_socket.subscribe(symbols=symbols, data_type=data_type)

        # Keep the socket running to receive real-time data
        fyers_socket.keep_running()

# -------------------------------
# Script Execution
# -------------------------------
if __name__ == "__main__":
    # Retreiving access token stored in access_token.txt
    with open('access_token.txt', 'r') as file:
        access_token = file.read()

    # Initialize the FyersModel instance with your client_id, access_token, and enable async mode
    fyers = fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        is_async=False,
        log_path=''
        )

    # Fetch initial historical data for NIFTY50
    nifty_df = historical_data(symbol="NSE:NIFTY50-INDEX", today=dt.date.today())
    nifty_df['ema_9'] = ta.EMA(nifty_df['close'], 9)    # Calculate EMA using 'close' as the source and a length of 9
    nifty_df['ema_15'] = ta.EMA(nifty_df['close'], 15)  # Calculate SMA using 'close' as the source and a length of 9

    df_hist = {"NSE:NIFTY50-INDEX": nifty_df}

    # Initialize strategy
    live = LiveOHLC(df_hist)

    # Create a FyersDataSocket instance with the provided parameters
    fyers_socket = data_ws.FyersDataSocket(
        access_token=access_token,  # Access token in the format "appid:accesstoken"
        log_path="",                # Path to save logs. Leave empty to auto-create logs in the current directory.
        litemode=False,             # Lite mode disabled. Set to True if you want a lite response.
        write_to_file=False,        # Save response in a log file instead of printing it.
        reconnect=True,             # Enable auto-reconnection to WebSocket on disconnection.
        on_connect=live.onopen,     # Callback function to subscribe to data upon connection.
        on_close=live.onclose,      # Callback function to handle WebSocket connection close events.
        on_error=live.onerror,      # Callback function to handle WebSocket errors.
        on_message=live.onmessage   # Callback function to handle incoming messages from the WebSocket.
    )

    # Establish a connection to the Fyers WebSocket
    fyers_socket.connect()

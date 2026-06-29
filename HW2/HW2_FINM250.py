#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# In[33]:


from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# In[35]:


allowed_tickers = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]

while True:
    user_input = input("Enter a ticker symbol: ").strip().upper()
    
    if user_input in allowed_tickers:
        print(f"Ticker {user_input} selected successfully!")
        ticker = user_input
        break
    else:
        print("Invalid ticker. Please try again.")


# In[41]:


# Get historical data for requested ticker
request = StockBarsRequest(
    symbol_or_symbols=ticker,
    timeframe=TimeFrame.Day,
    start=datetime(2021, 6, 1),
    end=datetime(2026, 6, 1),
)

bars = client.get_stock_bars(request).df


# In[51]:


# Building techincal indicators so that they can be recalled in the various strategies 
df = bars.reset_index()

# SMA short and long
df['SMA_short'] = df['close'].rolling(window=20).mean()
df['SMA_long'] = df['close'].rolling(window=200).mean()

# EMA short and long
df['EMA_short'] = df['close'].ewm(span=20, adjust=False).mean()
df['EMA_long'] = df['close'].ewm(span=200, adjust=False).mean()

# MACD
df['MACD'] = df['EMA_short'] - df['EMA_long']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# ADX (14-period)
plus_dm = df['high'].diff()
minus_dm = -df['low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm < 0] = 0
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift(1)).abs(),
    (df['low'] - df['close'].shift(1)).abs()
], axis=1).max(axis=1)
atr_14 = tr.ewm(span=14, adjust=False).mean()
plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14)
minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14)
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
df['ADX'] = dx.ewm(span=14, adjust=False).mean()

# RSI (14-day period)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

# Bollinger Bands (20-period, 2 std)
df['BB_mid'] = df['SMA_short']
df['BB_upper'] = df['SMA_short'] + 2 * df['close'].rolling(window=20).std()
df['BB_lower'] = df['SMA_short'] - 2 * df['close'].rolling(window=20).std()

# Williams %R 
high_14 = df['high'].rolling(window=14).max()
low_14 = df['low'].rolling(window=14).min()
df['WILLIAMS_R'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

# Chaikin Money Flow (20-period)
mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
mf_volume = mf_multiplier * df['volume']
df['CMF'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()


# In[53]:


#Strategy 1: Trend Following
# Buy: MACD > Signal, ADX > 25
# Sell: MACD < Signal

df['signal_1'] = 0

for i in range(1, len(df)):
    if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['ADX'].iloc[i] > 25:
        df.loc[df.index[i], 'signal_1'] = 1
    elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
        df.loc[df.index[i], 'signal_1'] = -1

df['position_1'] = df['signal_1'].replace(0, np.nan).ffill().fillna(0)

df['market_return'] = df['close'].pct_change()
df['strategy_1_return'] = df['position_1'].shift(1) * df['market_return']

df['cumulative_market'] = (1 + df['market_return']).cumprod()
df['cumulative_strategy_1'] = (1 + df['strategy_1_return']).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['cumulative_market'], label='Buy & Hold', alpha=0.7)
plt.plot(df['timestamp'], df['cumulative_strategy_1'], label='Strategy 1: Trend Following', alpha=0.7)
plt.title(f'Strategy 1: Trend Following vs Buy & Hold ({ticker})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

total_return = df['cumulative_strategy_1'].iloc[-1] - 1
buy_hold_return = df['cumulative_market'].iloc[-1] - 1
num_trades = (df['position_1'].diff() != 0).sum()
print(f"Strategy 1 Total Return: {total_return:.2%}")
print(f"Buy & Hold Return: {buy_hold_return:.2%}")
print(f"Number of trades: {num_trades}")


# In[55]:


#Strategy 2: Mean Reversion
# Buy: RSI < 40 AND price crosses BELOW BB_lower (oversold bounce)
# Sell: RSI > 60 AND price crosses ABOVE BB_upper (overbought fade)
# Exit to flat: price returns to BB_mid (mean achieved)

df['BB_pct'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

df['signal_2'] = 0

for i in range(1, len(df)):
    # Buy: oversold + breaking below lower band
    if df['RSI'].iloc[i] < 40 and df['close'].iloc[i] < df['BB_lower'].iloc[i]:
        df.loc[df.index[i], 'signal_2'] = 1
    # Sell: overbought + breaking above upper band
    elif df['RSI'].iloc[i] > 60 and df['close'].iloc[i] > df['BB_upper'].iloc[i]:
        df.loc[df.index[i], 'signal_2'] = -1
    # Exit: price reverts to the mean (middle band)
    elif (df['close'].iloc[i-1] < df['BB_mid'].iloc[i-1] and df['close'].iloc[i] >= df['BB_mid'].iloc[i]) or \
         (df['close'].iloc[i-1] > df['BB_mid'].iloc[i-1] and df['close'].iloc[i] <= df['BB_mid'].iloc[i]):
        df.loc[df.index[i], 'signal_2'] = 0

df['position_2'] = df['signal_2'].replace(0, np.nan).ffill().fillna(0)

df['strategy_2_return'] = df['position_2'].shift(1) * df['market_return']
df['cumulative_strategy_2'] = (1 + df['strategy_2_return']).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['cumulative_market'], label='Buy & Hold', alpha=0.7)
plt.plot(df['timestamp'], df['cumulative_strategy_2'], label='Strategy 2: Mean Reversion', alpha=0.7)
plt.title(f'Strategy 2: Mean Reversion vs Buy & Hold ({ticker})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

total_return_2 = df['cumulative_strategy_2'].iloc[-1] - 1
num_trades_2 = (df['position_2'].diff() != 0).sum()
print(f"Strategy 2 Total Return: {total_return_2:.2%}")
print(f"Buy & Hold Return: {buy_hold_return:.2%}")
print(f"Number of trades: {num_trades_2}")


# In[61]:


#Strategy 3: Jesse Livermore Pivotal Points using SMA, Williams, and CMF
# Jesse Livermore believed that he could identify prices where stocks reached what he called pivotal points and either broke up or down out of their averages.
# The idea is to wait for price to consolidate near SMA, then enter when momentum + volume (Williams and CMF) both confirm a directional break
# Buy: Price crosses above SMA_long AND Williams %R > -20 (strength) AND CMF > 0 (buying pressure)
# Sell: Price crosses below SMA_long AND Williams %R < -80 (weakness) AND CMF < 0 (selling pressure)

df['signal_3'] = 0

for i in range(1, len(df)):
    price = df['close'].iloc[i]
    sma = df['SMA_short'].iloc[i]
    wr = df['WILLIAMS_R'].iloc[i]
    cmf = df['CMF'].iloc[i]

    # Buy: price above short SMA + momentum not oversold + buying pressure
    if price > sma and wr > -50 and cmf > 0:
        df.loc[df.index[i], 'signal_3'] = 1
    # Exit to cash (not short): any two of three turn bearish
    elif (price < sma and wr < -50) or (price < sma and cmf < 0) or (wr < -50 and cmf < 0):
        df.loc[df.index[i], 'signal_3'] = 0
        
df['position_3'] = df['signal_3'].replace(0, np.nan).ffill().fillna(0)

df['strategy_3_return'] = df['position_3'].shift(1) * df['market_return']
df['cumulative_strategy_3'] = (1 + df['strategy_3_return']).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['cumulative_market'], label='Buy & Hold', alpha=0.7)
plt.plot(df['timestamp'], df['cumulative_strategy_3'], label='Strategy 3: Livermore Breakout', alpha=0.7)
plt.title(f'Strategy 3: Livermore Breakout vs Buy & Hold ({ticker})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

total_return_3 = df['cumulative_strategy_3'].iloc[-1] - 1
num_trades_3 = (df['position_3'].diff() != 0).sum()
print(f"Strategy 3 Total Return: {total_return_3:.2%}")
print(f"Buy & Hold Return: {buy_hold_return:.2%}")
print(f"Number of trades: {num_trades_3}")


# In[ ]:





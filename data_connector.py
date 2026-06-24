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

# Request 30 days of 5-minute bars for AAPL
request = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Minute,  # 1-minute bars
    start=datetime.now() - timedelta(days=30),
    end=datetime.now()
)

bars = client.get_stock_bars(request)
df = bars.df

print(df.head())
print(f"\nTotal bars: {len(df)}")

import plotly.graph_objects as go

# Reset index so symbol and timestamp become columns
df = df.reset_index()

fig = go.Figure(data=[go.Candlestick(
    x=df['timestamp'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

fig.update_layout(title='AAPL - 30 Day OHLCV', xaxis_title='Date', yaxis_title='Price')
fig.show()
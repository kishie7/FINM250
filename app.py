import time
import streamlit as st
import os

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrameUnit

from data_connector import get_historical_bars

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

st.set_page_config(page_title="Mini Market Data Terminal", layout="wide")
st.title("Mini Market Data Terminal")

ticker = st.sidebar.text_input("Ticker symbol:", value="AAPL").upper()
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)
auto_refresh = st.sidebar.checkbox("Auto-update quotes", value=True)

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

st.subheader(f"Live Quote: {ticker}")

quote_placeholder = st.empty()


def render_quote():
    quote_request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
    quote = client.get_stock_latest_quote(quote_request)[ticker]

    trade_request = StockLatestTradeRequest(symbol_or_symbols=ticker)
    trade = client.get_stock_latest_trade(trade_request)[ticker]

    with quote_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bid", f"${quote.bid_price:.2f}")
        col2.metric("Ask", f"${quote.ask_price:.2f}")
        col3.metric("Spread", f"${quote.ask_price - quote.bid_price:.2f}")
        col4.metric("Last Trade", f"${trade.price:.2f}")
        st.caption(f"Last updated: {quote.timestamp}")


render_quote()

st.subheader(f"Historical Data: {ticker}")

days = st.sidebar.slider("Days of history", 5, 60, 30)
bar_minutes = st.sidebar.selectbox("Bar size (minutes)", [1, 5, 15], index=1)

bars = get_historical_bars(
    ticker, days=days, timeframe_amount=bar_minutes, timeframe_unit=TimeFrameUnit.Minute
)

if bars.empty:
    st.warning("No historical data returned for this symbol/time range.")
else:
    st.line_chart(bars[["open", "high", "low", "close"]])
    st.bar_chart(bars["volume"])
    with st.expander("Show raw OHLCV data"):
        st.dataframe(bars)


if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()

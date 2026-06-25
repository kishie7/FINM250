import os
from datetime import datetime
import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

# ── Load .env file ──────────────────────────────────────────────────
load_dotenv()

# ── Page setup ──────────────────────────────────────────────────────
st.set_page_config(page_title="Real-Time Quote UI", layout="centered")
st.title("Real-Time Quote UI")

# ── Auth ────────────────────────────────────────────────────────────
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    st.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in your .env file.")
    st.stop()

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

ticker = st.text_input("Ticker symbol:", value="AAPL").upper().strip()

# ── Initialize price history in session state ───────────────────────
if "price_history" not in st.session_state:
    st.session_state.price_history = []
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ticker

# Clear history when the ticker changes
if ticker != st.session_state.last_ticker:
    st.session_state.price_history = []
    st.session_state.last_ticker = ticker

# ── Live quote fragment (refreshes every 2s without freezing the UI)
@st.fragment(run_every=2)
def live_quote():
    try:
        quote = client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker)
        )[ticker]
        trade = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=ticker)
        )[ticker]

        # ── Metrics ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Bid", f"${quote.bid_price:.2f}")
        col2.metric("Ask", f"${quote.ask_price:.2f}")
        col3.metric("Last Trade", f"${trade.price:.2f}")
        st.caption(f"Last updated: {quote.timestamp}")

        # ── Accumulate data point ───────────────────────────────
        st.session_state.price_history.append({
            "time": datetime.now(),
            "Bid": quote.bid_price,
            "Ask": quote.ask_price,
            "Last Trade": trade.price,
        })

        # Keep the last 150 points so the chart doesn't grow forever
        st.session_state.price_history = st.session_state.price_history[-150:]

        # ── Chart ───────────────────────────────────────────────
        df = pd.DataFrame(st.session_state.price_history)
        df_melted = df.melt(id_vars="time", var_name="Series", value_name="Price")

        y_min = df[["Bid", "Ask", "Last Trade"]].min().min()
        y_max = df[["Bid", "Ask", "Last Trade"]].max().max()
        padding = max((y_max - y_min) * 0.3, 0.05)

        chart = (
            alt.Chart(df_melted)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("Price:Q", title="Price ($)",
                         scale=alt.Scale(domain=[y_min - padding, y_max + padding])),
                color=alt.Color("Series:N", title=""),
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch quote for {ticker}: {e}")

if ticker:
    live_quote()
else:
    st.info("Enter a ticker symbol above to get started.")

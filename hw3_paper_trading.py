import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data import StockHistoricalDataClient
from alpaca.data.enums import Adjustment
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import feature_cols, compute_features

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

TRADE_NOTIONAL = 5_000  # demo paper-trading position size, in dollars

allowed_tickers = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]

while True:
    user_input = input("Enter a ticker symbol: ").strip().upper()
    if user_input in allowed_tickers:
        print(f"Ticker {user_input} selected successfully!")
        ticker = user_input
        break
    else:
        print("Invalid ticker. Please try again.")


history_request = StockBarsRequest(
    symbol_or_symbols=ticker,
    timeframe=TimeFrame.Day,
    start=datetime(2021, 6, 1),
    end=datetime.now() - timedelta(days=1),
    adjustment=Adjustment.ALL,
)
history_bars = data_client.get_stock_bars(history_request).df
history_df = compute_features(history_bars.reset_index())
history_df["market_return"] = history_df["close"].pct_change()
history_df["target"] = (history_df["market_return"].shift(-1) > 0).astype(int)

ml_df = history_df.dropna(subset=feature_cols + ["target"]).copy()
X = ml_df[feature_cols]
y = ml_df["target"]

prod_scaler = StandardScaler()
X_scaled_full = prod_scaler.fit_transform(X)

prod_pca = PCA(n_components=0.80)
X_pca_full = prod_pca.fit_transform(X_scaled_full)

prod_model = LogisticRegression(max_iter=1000)
prod_model.fit(X_pca_full, y)

print(f"Production model trained on {len(ml_df)} days of history "
      f"({prod_pca.n_components_} PCA components, "
      f"{prod_pca.explained_variance_ratio_.sum():.1%} variance explained)")


live_request = StockBarsRequest(
    symbol_or_symbols=ticker,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=730),
    end=datetime.now() - timedelta(days=1),
    adjustment=Adjustment.ALL,
)
live_bars = data_client.get_stock_bars(live_request).df
live_df = live_bars.reset_index()


live_df = compute_features(live_df)
latest_row = live_df.dropna(subset=feature_cols).iloc[[-1]]


latest_scaled = prod_scaler.transform(latest_row[feature_cols])
latest_pca = prod_pca.transform(latest_scaled)


latest_probability = prod_model.predict_proba(latest_pca)[:, 1][0]
latest_signal = "Long" if latest_probability > 0.60 else "Flat"
latest_date = latest_row["timestamp"].iloc[0]
latest_close = latest_row["close"].iloc[0]

print(f"\nTicker: {ticker}")
print(f"Latest bar date: {latest_date}")
print(f"Latest close: ${latest_close:.2f}")
print(f"Model probability (P[next-day return > 0]): {latest_probability:.4f}")
print(f"Signal: {latest_signal}")


try:
    current_position = trading_client.get_open_position(ticker)
    current_qty = float(current_position.qty)
except Exception:
    current_qty = 0.0

order = None
action = "HOLD (no order needed)"

if latest_signal == "Long" and current_qty <= 0:
    order_request = MarketOrderRequest(
        symbol=ticker,
        notional=TRADE_NOTIONAL,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(order_request)
    action = f"BUY submitted (${TRADE_NOTIONAL} notional)"

elif latest_signal == "Flat" and current_qty > 0:
    order = trading_client.close_position(ticker)
    action = "SELL submitted (closed existing position)"

print(f"Action: {action}")
if order is not None:
    print(f"Order ID: {order.id} | Status: {order.status}")


log_entry = pd.DataFrame([{
    "timestamp": datetime.now(),
    "ticker": ticker,
    "latest_bar_date": latest_date,
    "latest_close": latest_close,
    "probability": latest_probability,
    "signal": latest_signal,
    "action": action,
    "order_id": getattr(order, "id", None),
    "order_status": getattr(order, "status", None),
}])

log_path = "paper_trading_log.csv"
log_entry.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
print(f"\nLogged signal + order to {log_path}")

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from config_loader import load_config, load_credentials
from data.historical import HistoricalDataService
from engine.trading_engine import PaperTradingEngine
from execution.alpaca_broker import AlpacaPaperBroker
from main import build_components, log_signals

SETTINGS_PATH = Path("config/settings.yaml")
SIGNALS_PATH = Path("logs/signals.csv")

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Algorithmic Trading Dashboard")

config = load_config(SETTINGS_PATH)

try:
    credentials = load_credentials(require=True)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "engine" not in st.session_state:
    strategy, risk = build_components(config)
    broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    st.session_state.engine = PaperTradingEngine(
        strategy, risk, broker, config["execution"]["rebalance_threshold_pct"]
    )
    st.session_state.broker = broker
    st.session_state.service = HistoricalDataService(credentials.api_key, credentials.secret_key)

engine = st.session_state.engine
broker = st.session_state.broker
service = st.session_state.service

cycle_seconds = config["execution"].get("cycle_seconds", 300)
st_autorefresh(interval=cycle_seconds * 1000, key="refresh")

if engine.enabled:
    try:
        minimum_history = engine.strategy.minimum_history
        lookback_start = date.today() - timedelta(days=minimum_history * 2 + 10)
        bars = service.get_daily_bars(config["symbols"], start=lookback_start.isoformat())
        signals = engine.strategy.latest_signals(bars)
        log_signals(signals, SIGNALS_PATH)
        engine.run_cycle(bars)
    except Exception as exc:
        st.error(f"Trading cycle failed: {exc}")

st.header("System Status")
account = broker.get_account()
col1, col2, col3 = st.columns(3)
col1.metric("Equity", f"${float(account.equity):,.2f}")
col2.metric("Cash", f"${float(account.cash):,.2f}")
col3.metric("Buying Power", f"${float(account.buying_power):,.2f}")

status_col1, status_col2 = st.columns(2)
status_col1.metric("Engine", "Running" if engine.enabled else "Stopped")
status_col2.metric("Last Update", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

st.divider()

st.header("Portfolio")
positions = broker.get_positions()
rows = [
    {
        "Ticker": symbol,
        "Qty": position.quantity,
        "Avg Price": position.average_entry_price,
        "Current Price": position.current_price,
        "PnL": round(position.unrealized_pnl, 2),
    }
    for symbol, position in positions.items()
]
st.dataframe(pd.DataFrame(rows), width="stretch")

st.header("Recent Orders")
orders = broker.get_recent_orders()
order_rows = [
    {
        "Time": order.submitted_at,
        "Symbol": order.symbol,
        "Side": order.side,
        "Qty": order.qty,
        "Status": order.status,
    }
    for order in orders
]
st.dataframe(pd.DataFrame(order_rows), width="stretch")

st.header("Recent Signals")
try:
    signals_df = pd.read_csv(SIGNALS_PATH)
    st.dataframe(signals_df.tail(10), width="stretch")
except FileNotFoundError:
    st.info("No signals have been generated yet.")

st.header("Controls")
control_col1, control_col2 = st.columns(2)
with control_col1:
    if st.button("Start Strategy", disabled=engine.enabled):
        engine.start()
        st.rerun()
with control_col2:
    if st.button("Stop Strategy", disabled=not engine.enabled):
        engine.stop()
        st.rerun()

st.header("Risk Settings")
risk_cfg = config["risk"]
with st.form("risk_settings_form"):
    max_position_pct = st.number_input(
        "Max Position Weight (%)", min_value=0.0, max_value=100.0,
        value=risk_cfg["max_position_pct"] * 100, step=1.0,
    ) / 100
    stop_loss_pct = st.number_input(
        "Stop Loss (%)", min_value=0.0, max_value=100.0,
        value=risk_cfg["stop_loss_pct"] * 100, step=0.5,
    ) / 100
    take_profit_pct = st.number_input(
        "Take Profit (%)", min_value=0.0, max_value=100.0,
        value=risk_cfg["take_profit_pct"] * 100, step=0.5,
    ) / 100
    max_daily_drawdown_pct = st.number_input(
        "Max Daily Drawdown (%)", min_value=0.0, max_value=100.0,
        value=risk_cfg["max_daily_drawdown_pct"] * 100, step=0.5,
    ) / 100
    submitted = st.form_submit_button("Save Configuration")

if submitted:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    settings.setdefault("risk", {})
    settings["risk"]["max_position_pct"] = max_position_pct
    settings["risk"]["stop_loss_pct"] = stop_loss_pct
    settings["risk"]["take_profit_pct"] = take_profit_pct
    settings["risk"]["max_daily_drawdown_pct"] = max_daily_drawdown_pct
    with SETTINGS_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(settings, file, sort_keys=False)

    engine.risk.max_position_pct = max_position_pct
    engine.risk.stop_loss_pct = stop_loss_pct
    engine.risk.take_profit_pct = take_profit_pct
    engine.risk.max_daily_drawdown_pct = max_daily_drawdown_pct
    engine.strategy.max_position_weight = max_position_pct

    st.success("Configuration saved and applied to the running engine.")

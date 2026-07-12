from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtest.runner import BacktestRunner
from config_loader import load_config, load_credentials
from data.historical import HistoricalDataService
from data.live_stream import LiveMarketDataStream
from data.store import SQLiteBarStore
from engine.trading_engine import PaperTradingEngine
from execution.alpaca_broker import AlpacaPaperBroker
from execution.simulated_broker import SimulatedBroker
from logging_setup import configure_logging
from models import Signal
from risk.manager import RiskManager
from strategy.trend_following import MovingAverageTrendStrategy

LOGGER = logging.getLogger(__name__)


def build_components(config: dict):
    strategy_cfg = config["strategy"]
    risk_cfg = config["risk"]
    execution_cfg = config["execution"]

    strategy = MovingAverageTrendStrategy(
        fast_window=strategy_cfg["fast_window"],
        slow_window=strategy_cfg["slow_window"],
        volatility_window=strategy_cfg["volatility_window"],
        annualized_target_volatility=strategy_cfg["annualized_target_volatility"],
        max_position_weight=risk_cfg["max_position_pct"],
        minimum_history=strategy_cfg.get("minimum_history"),
    )
    risk = RiskManager(
        max_position_pct=risk_cfg["max_position_pct"],
        max_gross_exposure_pct=risk_cfg["max_gross_exposure_pct"],
        max_order_notional_pct=risk_cfg["max_order_notional_pct"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        take_profit_pct=risk_cfg["take_profit_pct"],
        max_daily_drawdown_pct=risk_cfg["max_daily_drawdown_pct"],
        allow_fractional_shares=execution_cfg["allow_fractional_shares"],
    )
    return strategy, risk


def run_backtest(config: dict) -> None:
    credentials = load_credentials(require=True)
    service = HistoricalDataService(credentials.api_key, credentials.secret_key)
    bars = service.get_daily_bars(
        config["symbols"], config["backtest"]["start"], config["backtest"]["end"]
    )
    strategy, risk = build_components(config)
    broker = SimulatedBroker(
        initial_cash=config["risk"]["initial_capital"],
        slippage_bps=config["backtest"]["slippage_bps"],
        commission=config["backtest"]["commission_per_trade"],
    )
    result = BacktestRunner(
        strategy,
        risk,
        broker,
        config["execution"]["rebalance_threshold_pct"],
    ).run(bars)
    print("Backtest metrics")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    result.equity_curve.to_csv("data/equity_curve.csv")
    result.trades.to_csv("data/trades.csv", index=False)


def validate_paper_connection() -> None:
    credentials = load_credentials(require=True)
    broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    account = broker.get_account()
    print(f"Connected to Alpaca paper account. Equity: ${float(account.equity):,.2f}")


def log_signals(signals: dict[str, Signal], path: str | Path = "logs/signals.csv") -> None:
    if not signals:
        return
    signals_path = Path(path)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "price": signal.price,
            "target_weight": signal.target_weight,
            "fast_ma": signal.fast_ma,
            "slow_ma": signal.slow_ma,
            "annualized_volatility": signal.annualized_volatility,
        }
        for signal in signals.values()
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(signals_path, mode="a", header=not signals_path.exists(), index=False)


def run_paper_trade(config: dict) -> None:
    credentials = load_credentials(require=True)
    strategy, risk = build_components(config)
    broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    engine = PaperTradingEngine(strategy, risk, broker, config["execution"]["rebalance_threshold_pct"])
    service = HistoricalDataService(credentials.api_key, credentials.secret_key)

    symbols = config["symbols"]
    minimum_history = strategy.minimum_history
    cycle_seconds = config["execution"].get("cycle_seconds", 300)

    store = SQLiteBarStore()
    stream = LiveMarketDataStream(credentials.api_key, credentials.secret_key, symbols)

    async def _on_bar(bar) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            ]
        )
        store.append_bars(frame)

    stream.subscribe_bars(_on_bar)
    stream_thread = threading.Thread(target=stream.run, daemon=True)
    stream_thread.start()

    engine.start()
    try:
        while engine.enabled:
            lookback_start = date.today() - timedelta(days=minimum_history * 2 + 10)
            bars = service.get_daily_bars(symbols, start=lookback_start.isoformat())

            signals = strategy.latest_signals(bars)
            log_signals(signals)

            executions = engine.run_cycle(bars)
            if executions:
                LOGGER.info("Cycle executions: %s", executions)

            time.sleep(cycle_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Paper trading loop interrupted by user")
    finally:
        engine.stop()
        stream.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["backtest", "paper-check", "paper-trade"], default="backtest"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config)
    if args.mode == "backtest":
        run_backtest(config)
    elif args.mode == "paper-check":
        validate_paper_connection()
    else:
        run_paper_trade(config)


if __name__ == "__main__":
    main()

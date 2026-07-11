from __future__ import annotations

import argparse
import logging

from backtest.runner import BacktestRunner
from config_loader import load_config, load_credentials
from data.historical import HistoricalDataService
from execution.alpaca_broker import AlpacaPaperBroker
from execution.simulated_broker import SimulatedBroker
from logging_setup import configure_logging
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "paper-check"], default="backtest")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config)
    if args.mode == "backtest":
        run_backtest(config)
    else:
        validate_paper_connection()


if __name__ == "__main__":
    main()

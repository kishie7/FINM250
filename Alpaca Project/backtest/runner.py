from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from execution.simulated_broker import SimulatedBroker
from models import Signal, SignalDirection
from risk.manager import RiskManager
from strategy.trend_following import MovingAverageTrendStrategy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class BacktestRunner:
    def __init__(
        self,
        strategy: MovingAverageTrendStrategy,
        risk: RiskManager,
        broker: SimulatedBroker,
        rebalance_threshold_pct: float,
    ) -> None:
        self.strategy = strategy
        self.risk = risk
        self.broker = broker
        self.rebalance_threshold_pct = rebalance_threshold_pct

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        enriched = self.strategy.enrich(bars)
        dates = sorted(pd.to_datetime(enriched.timestamp).unique())
        records: list[dict[str, float | pd.Timestamp]] = []
        initial_equity = self.broker.cash
        start_of_day_equity = initial_equity
        current_trading_date = None
        kill_switch_active = False

        for timestamp in dates:
            timestamp_value = pd.Timestamp(timestamp)
            trading_date = timestamp_value.date()
            snapshot = enriched[enriched.timestamp == timestamp]
            prices = dict(zip(snapshot.symbol, snapshot.close, strict=False))
            equity = self.broker.mark_to_market(prices)

            if trading_date != current_trading_date:
                current_trading_date = trading_date
                start_of_day_equity = equity
                kill_switch_active = False

            if kill_switch_active or self.risk.kill_switch_triggered(equity, start_of_day_equity):
                kill_switch_active = True
                records.append({"timestamp": timestamp, "equity": equity})
                continue

            exited_symbols: set[str] = set()
            for symbol, position in list(self.broker.positions.items()):
                exit_intent = self.risk.protective_exit(position)
                if exit_intent:
                    try:
                        fill = self.broker.execute(exit_intent)
                    except Exception:
                        LOGGER.exception("Protective exit failed for %s; skipping", symbol)
                        fill = None
                    if fill is not None:
                        exited_symbols.add(symbol)

            gross = self.broker.gross_exposure()
            for row in snapshot.itertuples():
                if pd.isna(row.fast_ma) or pd.isna(row.slow_ma) or pd.isna(row.annualized_volatility):
                    continue
                if row.symbol in exited_symbols:
                    # No-churn cooldown: don't immediately re-enter a position
                    # that was just stopped out / took profit on this bar.
                    continue

                signal = Signal(
                    symbol=row.symbol,
                    timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                    direction=SignalDirection(int(row.direction)),
                    price=float(row.close),
                    target_weight=float(row.target_weight),
                    fast_ma=float(row.fast_ma),
                    slow_ma=float(row.slow_ma),
                    annualized_volatility=float(row.annualized_volatility),
                )
                current_qty = self.broker.positions.get(
                    row.symbol,
                    type("Empty", (), {"quantity": 0.0})(),
                ).quantity
                intent = self.risk.generate_order_intent(
                    signal,
                    current_qty,
                    equity,
                    gross,
                    self.rebalance_threshold_pct,
                )
                if intent:
                    try:
                        fill = self.broker.execute(intent)
                    except Exception:
                        LOGGER.exception("Order execution failed for %s; skipping", row.symbol)
                        fill = None
                    if fill is not None:
                        gross = self.broker.gross_exposure()

            equity = self.broker.mark_to_market(prices)
            records.append({"timestamp": timestamp, "equity": equity})

        curve = pd.DataFrame(records).drop_duplicates("timestamp").set_index("timestamp")
        curve["return"] = curve.equity.pct_change().fillna(0.0)
        curve["cumulative_return"] = curve.equity / initial_equity - 1.0
        curve["running_peak"] = curve.equity.cummax()
        curve["drawdown"] = curve.equity / curve.running_peak - 1.0

        trades = pd.DataFrame([fill.__dict__ for fill in self.broker.fills])
        metrics = self._metrics(curve, trades)
        return BacktestResult(curve, trades, metrics)

    @staticmethod
    def _metrics(curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
        if curve.empty:
            return {}
        returns = curve["return"]
        volatility = returns.std(ddof=0) * np.sqrt(252)
        sharpe = returns.mean() * 252 / volatility if volatility > 0 else 0.0
        closed_trades = trades[trades["side"] == "sell"] if not trades.empty else trades
        hit_rate = (
            float((closed_trades["realized_pnl"] > 0).mean())
            if not closed_trades.empty and "realized_pnl" in closed_trades
            else 0.0
        )
        return {
            "final_equity": float(curve.equity.iloc[-1]),
            "cumulative_return": float(curve.cumulative_return.iloc[-1]),
            "max_drawdown": float(curve.drawdown.min()),
            "annualized_volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "number_of_trades": float(len(trades)),
            "hit_rate": hit_rate,
        }

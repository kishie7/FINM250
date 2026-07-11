from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from models import OrderExecution, OrderIntent, PositionState
from risk.manager import RiskManager
from strategy.trend_following import MovingAverageTrendStrategy

LOGGER = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("America/New_York")


class BrokerProtocol(Protocol):
    def get_positions(self) -> dict[str, PositionState]: ...
    def submit_market_order(self, intent: OrderIntent) -> OrderExecution: ...
    def get_account(self): ...


class PaperTradingEngine:
    def __init__(
        self,
        strategy: MovingAverageTrendStrategy,
        risk_manager: RiskManager,
        broker: BrokerProtocol,
        rebalance_threshold_pct: float,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk_manager
        self.broker = broker
        self.rebalance_threshold_pct = rebalance_threshold_pct
        self.enabled = True
        self.start_of_day_equity: float | None = None
        self.start_of_day_date: date | None = None
        self._now_provider = now_provider or (lambda: datetime.now(MARKET_TIMEZONE))

    def start(self) -> None:
        self.enabled = True
        LOGGER.info("Trading engine started")

    def stop(self) -> None:
        self.enabled = False
        LOGGER.warning("Trading engine stopped")

    def _refresh_start_of_day_equity(self, equity: float) -> None:
        today = self._now_provider().astimezone(MARKET_TIMEZONE).date()
        if self.start_of_day_date != today:
            self.start_of_day_date = today
            self.start_of_day_equity = equity
            LOGGER.info("Start-of-day equity reset for %s: $%.2f", today, equity)

    @staticmethod
    def _gross_exposure(positions: dict[str, PositionState]) -> float:
        return sum(abs(position.market_value) for position in positions.values())

    def run_cycle(self, bars: pd.DataFrame) -> list[OrderExecution]:
        if not self.enabled:
            return []

        account = self.broker.get_account()
        equity = float(account.equity)
        self._refresh_start_of_day_equity(equity)
        assert self.start_of_day_equity is not None
        if self.risk.kill_switch_triggered(equity, self.start_of_day_equity):
            self.stop()
            raise RuntimeError("Daily drawdown kill switch triggered")

        positions = self.broker.get_positions()
        executions: list[OrderExecution] = []
        cooldown_symbols: set[str] = set()

        # Protective exits are processed first. A submitted exit that remains
        # open is also placed on cooldown so the strategy cannot send a
        # conflicting buy while the sell is still working.
        for position in list(positions.values()):
            exit_intent = self.risk.protective_exit(position)
            if exit_intent is None:
                continue

            result = self._submit_safely(exit_intent)
            if result is None:
                continue
            executions.append(result)

            if result.submitted and not result.permanently_failed:
                cooldown_symbols.add(position.symbol)
            if result.has_fill:
                positions = self.broker.get_positions()

        signals = self.strategy.latest_signals(bars)
        gross_exposure = self._gross_exposure(positions)

        for symbol, signal in signals.items():
            if symbol in cooldown_symbols:
                continue

            current_qty = positions[symbol].quantity if symbol in positions else 0.0
            intent = self.risk.generate_order_intent(
                signal=signal,
                current_quantity=current_qty,
                equity=equity,
                current_gross_exposure=gross_exposure,
                rebalance_threshold_pct=self.rebalance_threshold_pct,
            )
            if intent is None:
                continue

            result = self._submit_safely(intent)
            if result is None:
                continue
            executions.append(result)

            # Only a confirmed fill changes the position book. Re-fetching
            # avoids incorrect gross-exposure arithmetic for sells, partial
            # fills, slippage, and broker-side rounding.
            if result.has_fill:
                positions = self.broker.get_positions()
                gross_exposure = self._gross_exposure(positions)

        return executions

    def _submit_safely(self, intent: OrderIntent) -> OrderExecution | None:
        try:
            return self.broker.submit_market_order(intent)
        except Exception:
            LOGGER.exception(
                "Order submission failed for %s (%s); continuing cycle",
                intent.symbol,
                intent.reason,
            )
            return None

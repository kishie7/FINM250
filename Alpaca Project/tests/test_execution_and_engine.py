from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from engine.trading_engine import PaperTradingEngine
from models import OrderExecution, PositionState, Signal, SignalDirection
from risk.manager import RiskManager


class StaticStrategy:
    def __init__(self, signals):
        self.signals = signals

    def latest_signals(self, bars):
        return self.signals


class FakeBroker:
    def __init__(self, positions, results):
        self.positions = positions
        self.results = list(results)
        self.intents = []
        self.position_reads = 0

    def get_account(self):
        return SimpleNamespace(equity="100000")

    def get_positions(self):
        self.position_reads += 1
        return dict(self.positions)

    def submit_market_order(self, intent):
        self.intents.append(intent)
        result = self.results.pop(0)
        if result.has_fill and intent.side == "sell":
            self.positions.pop(intent.symbol, None)
        return result


def manager():
    return RiskManager(0.2, 0.9, 0.1, 0.05, 0.10, 0.03, True)


def signal(symbol="AAPL"):
    return Signal(
        symbol=symbol,
        timestamp=datetime.now(),
        direction=SignalDirection.LONG,
        price=100.0,
        target_weight=0.1,
        fast_ma=105.0,
        slow_ma=100.0,
        annualized_volatility=0.2,
    )


def execution(status, filled=0.0, terminal=True):
    return OrderExecution(
        symbol="AAPL",
        side="sell",
        requested_quantity=10.0,
        filled_quantity=filled,
        status=status,
        submitted=True,
        terminal=terminal,
        order_id="abc",
    )


def test_rejected_protective_exit_does_not_create_cooldown():
    position = PositionState("AAPL", 10.0, 100.0, 90.0)
    broker = FakeBroker(
        {"AAPL": position},
        [
            execution("rejected"),
            OrderExecution(
                symbol="AAPL",
                side="buy",
                requested_quantity=10.0,
                filled_quantity=10.0,
                status="filled",
                submitted=True,
                terminal=True,
                order_id="buy-1",
            ),
        ],
    )
    engine = PaperTradingEngine(
        StaticStrategy({"AAPL": signal()}), manager(), broker, 0.001
    )

    engine.run_cycle(pd.DataFrame())

    assert len(broker.intents) == 2
    assert broker.intents[0].reason == "stop_loss"
    assert broker.intents[1].side == "buy"


def test_open_partial_protective_exit_prevents_conflicting_reentry():
    position = PositionState("AAPL", 10.0, 100.0, 90.0)
    broker = FakeBroker(
        {"AAPL": position},
        [execution("partially_filled", filled=2.0, terminal=False)],
    )
    engine = PaperTradingEngine(
        StaticStrategy({"AAPL": signal()}), manager(), broker, 0.001
    )

    results = engine.run_cycle(pd.DataFrame())

    assert len(results) == 1
    assert len(broker.intents) == 1
    assert broker.position_reads >= 2


def test_start_of_day_equity_resets_only_when_market_date_changes():
    times = iter(
        [
            datetime(2026, 7, 10, 9, 30, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 7, 10, 15, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 7, 11, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        ]
    )
    broker = FakeBroker({}, [])
    engine = PaperTradingEngine(
        StaticStrategy({}), manager(), broker, 0.001, now_provider=lambda: next(times)
    )

    engine._refresh_start_of_day_equity(100000)
    engine._refresh_start_of_day_equity(99000)
    assert engine.start_of_day_equity == 100000
    engine._refresh_start_of_day_equity(98000)
    assert engine.start_of_day_equity == 98000

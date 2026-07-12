from __future__ import annotations

from types import SimpleNamespace

from alpaca.trading.enums import OrderStatus

from execution.alpaca_broker import AlpacaPaperBroker


def test_partial_fill_is_not_terminal(monkeypatch):
    broker = object.__new__(AlpacaPaperBroker)
    broker.fill_poll_seconds = 0.0
    broker.fill_timeout_seconds = 0.0
    partial = SimpleNamespace(
        status=OrderStatus.PARTIALLY_FILLED,
        filled_qty="2",
        filled_avg_price="101.5",
    )
    broker.client = SimpleNamespace(get_order_by_id=lambda order_id: partial)

    order, terminal = broker._wait_for_terminal_status("order-1")

    assert order is partial
    assert terminal is False

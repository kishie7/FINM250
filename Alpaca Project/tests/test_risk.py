from datetime import datetime

from models import Signal, SignalDirection
from risk.manager import RiskManager


def make_manager():
    return RiskManager(0.2, 0.9, 0.1, 0.05, 0.10, 0.03, True)


def test_order_notional_is_capped():
    signal = Signal("AAPL", datetime.now(), SignalDirection.LONG, 100, 0.2, 101, 99, 0.2)
    intent = make_manager().generate_order_intent(signal, 0, 100_000, 0, 0.001)
    assert intent is not None
    assert intent.quantity * intent.reference_price <= 10_000 + 1e-6


def test_kill_switch():
    manager = make_manager()
    assert manager.kill_switch_triggered(96_000, 100_000)

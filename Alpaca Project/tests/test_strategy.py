import pandas as pd

from strategy.trend_following import MovingAverageTrendStrategy


def test_uptrend_generates_long_signal():
    dates = pd.date_range("2025-01-01", periods=80, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {"symbol": "TEST", "timestamp": dates, "close": range(100, 180)}
    )
    strategy = MovingAverageTrendStrategy(10, 30, 10, 0.15, 0.20)
    signal = strategy.latest_signals(bars)["TEST"]
    assert signal.direction.value == 1
    assert 0 < signal.target_weight <= 0.20


def test_downtrend_generates_flat_signal():
    dates = pd.date_range("2025-01-01", periods=80, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {"symbol": "TEST", "timestamp": dates, "close": range(180, 100, -1)}
    )
    strategy = MovingAverageTrendStrategy(10, 30, 10, 0.15, 0.20)
    signal = strategy.latest_signals(bars)["TEST"]
    assert signal.direction.value == 0
    assert signal.target_weight == 0

from __future__ import annotations

from datetime import datetime
import math

import numpy as np
import pandas as pd

from models import Signal, SignalDirection


class MovingAverageTrendStrategy:
    """Long/flat trend strategy with inverse-volatility position sizing."""

    def __init__(
        self,
        fast_window: int,
        slow_window: int,
        volatility_window: int,
        annualized_target_volatility: float,
        max_position_weight: float,
        minimum_history: int | None = None,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.volatility_window = volatility_window
        self.target_volatility = annualized_target_volatility
        self.max_position_weight = max_position_weight
        self.minimum_history = minimum_history or max(slow_window, volatility_window) + 1

    def enrich(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "timestamp", "close"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        frame = bars.copy().sort_values(["symbol", "timestamp"])
        grouped = frame.groupby("symbol", group_keys=False)
        frame["return"] = grouped["close"].pct_change()
        frame["fast_ma"] = grouped["close"].transform(
            lambda s: s.rolling(self.fast_window, min_periods=self.fast_window).mean()
        )
        frame["slow_ma"] = grouped["close"].transform(
            lambda s: s.rolling(self.slow_window, min_periods=self.slow_window).mean()
        )
        frame["annualized_volatility"] = grouped["return"].transform(
            lambda s: s.rolling(self.volatility_window, min_periods=self.volatility_window).std()
            * math.sqrt(252)
        )
        frame["direction"] = np.where(frame["fast_ma"] > frame["slow_ma"], 1, 0)
        raw_weight = self.target_volatility / frame["annualized_volatility"].replace(0, np.nan)
        frame["target_weight"] = (
            raw_weight.clip(lower=0, upper=self.max_position_weight) * frame["direction"]
        ).fillna(0.0)
        return frame

    def latest_signals(self, bars: pd.DataFrame) -> dict[str, Signal]:
        enriched = self.enrich(bars)
        signals: dict[str, Signal] = {}
        for symbol, group in enriched.groupby("symbol"):
            if len(group) < self.minimum_history:
                continue
            row = group.iloc[-1]
            required_values = [row.fast_ma, row.slow_ma, row.annualized_volatility, row.close]
            if any(pd.isna(value) for value in required_values):
                continue
            signals[symbol] = Signal(
                symbol=symbol,
                timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                direction=SignalDirection(int(row.direction)),
                price=float(row.close),
                target_weight=float(row.target_weight),
                fast_ma=float(row.fast_ma),
                slow_ma=float(row.slow_ma),
                annualized_volatility=float(row.annualized_volatility),
            )
        return signals

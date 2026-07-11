from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class SignalDirection(int, Enum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: datetime
    direction: SignalDirection
    price: float
    target_weight: float
    fast_ma: float
    slow_ma: float
    annualized_volatility: float


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    quantity: float
    side: str
    reference_price: float
    reason: str


@dataclass(frozen=True)
class OrderExecution:
    """Normalized broker result used by the engine.

    `submitted` means Alpaca accepted the API request and created an order.
    It does not imply a fill. `terminal` indicates that the broker has stopped
    working the order. A timed-out order may remain open with terminal=False.
    """

    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    status: str
    submitted: bool
    terminal: bool
    order_id: str | None = None
    average_fill_price: float | None = None
    raw_order: Any | None = None

    @property
    def has_fill(self) -> bool:
        return self.filled_quantity > 1e-9

    @property
    def fully_filled(self) -> bool:
        return self.status == "filled" and self.has_fill

    @property
    def permanently_failed(self) -> bool:
        return self.status in {"rejected", "canceled", "expired"} and not self.has_fill


@dataclass
class PositionState:
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.average_entry_price)

    @property
    def return_pct(self) -> float:
        if self.average_entry_price <= 0:
            return 0.0
        return self.current_price / self.average_entry_price - 1.0

from __future__ import annotations

import logging
from dataclasses import dataclass

from models import OrderIntent, PositionState

LOGGER = logging.getLogger(__name__)


@dataclass
class Fill:
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    realized_pnl: float = 0.0


class SimulatedBroker:
    def __init__(self, initial_cash: float, slippage_bps: float = 0.0, commission: float = 0.0) -> None:
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.commission = commission
        self.positions: dict[str, PositionState] = {}
        self.fills: list[Fill] = []

    def mark_to_market(self, prices: dict[str, float]) -> float:
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        return self.cash + sum(position.market_value for position in self.positions.values())

    def gross_exposure(self) -> float:
        return sum(abs(position.market_value) for position in self.positions.values())

    def execute(self, intent: OrderIntent) -> Fill | None:
        slip = self.slippage_bps / 10_000
        price = intent.reference_price * (1 + slip if intent.side == "buy" else 1 - slip)
        quantity = intent.quantity
        current = self.positions.get(
            intent.symbol,
            PositionState(intent.symbol, 0.0, 0.0, price),
        )
        realized_pnl = 0.0

        if intent.side == "buy":
            max_affordable = max((self.cash - self.commission) / price, 0.0)
            quantity = min(quantity, max_affordable)
            if quantity <= 1e-9:
                LOGGER.warning("Skipping buy for %s: insufficient cash", intent.symbol)
                return None
            total_cost = price * quantity + self.commission
            new_qty = current.quantity + quantity
            new_avg = (
                current.average_entry_price * current.quantity + price * quantity
            ) / new_qty
            self.cash -= total_cost
            self.positions[intent.symbol] = PositionState(intent.symbol, new_qty, new_avg, price)
        elif intent.side == "sell":
            quantity = min(quantity, current.quantity)
            if quantity <= 1e-9:
                LOGGER.warning("Skipping sell for %s: no position available", intent.symbol)
                return None
            realized_pnl = (price - current.average_entry_price) * quantity - self.commission
            self.cash += price * quantity - self.commission
            remaining = current.quantity - quantity
            if remaining <= 1e-9:
                self.positions.pop(intent.symbol, None)
            else:
                self.positions[intent.symbol] = PositionState(
                    intent.symbol, remaining, current.average_entry_price, price
                )
        else:
            raise ValueError(f"Unsupported order side: {intent.side}")

        fill = Fill(intent.symbol, intent.side, quantity, price, self.commission, realized_pnl)
        self.fills.append(fill)
        return fill

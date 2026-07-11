from __future__ import annotations

import logging
import math

from models import OrderIntent, PositionState, Signal

LOGGER = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        max_position_pct: float,
        max_gross_exposure_pct: float,
        max_order_notional_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_daily_drawdown_pct: float,
        allow_fractional_shares: bool = True,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_order_notional_pct = max_order_notional_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.allow_fractional_shares = allow_fractional_shares

    def kill_switch_triggered(self, equity: float, start_of_day_equity: float) -> bool:
        if start_of_day_equity <= 0:
            return True
        drawdown = 1.0 - equity / start_of_day_equity
        return drawdown >= self.max_daily_drawdown_pct

    def target_quantity(self, signal: Signal, equity: float) -> float:
        capped_weight = min(max(signal.target_weight, 0.0), self.max_position_pct)
        quantity = equity * capped_weight / signal.price
        return quantity if self.allow_fractional_shares else float(math.floor(quantity))

    def generate_order_intent(
        self,
        signal: Signal,
        current_quantity: float,
        equity: float,
        current_gross_exposure: float,
        rebalance_threshold_pct: float,
    ) -> OrderIntent | None:
        target_qty = self.target_quantity(signal, equity)
        delta_qty = target_qty - current_quantity
        delta_notional = abs(delta_qty * signal.price)

        if delta_notional < equity * rebalance_threshold_pct:
            return None

        max_order_notional = equity * self.max_order_notional_pct
        if delta_notional > max_order_notional:
            delta_qty = math.copysign(max_order_notional / signal.price, delta_qty)
            delta_notional = max_order_notional

        projected_gross = current_gross_exposure + delta_notional
        if delta_qty > 0 and projected_gross > equity * self.max_gross_exposure_pct:
            remaining = max(equity * self.max_gross_exposure_pct - current_gross_exposure, 0.0)
            delta_qty = remaining / signal.price

        if not self.allow_fractional_shares:
            delta_qty = math.copysign(math.floor(abs(delta_qty)), delta_qty)
        if abs(delta_qty) <= 1e-9:
            return None

        return OrderIntent(
            symbol=signal.symbol,
            quantity=abs(delta_qty),
            side="buy" if delta_qty > 0 else "sell",
            reference_price=signal.price,
            reason=f"rebalance_to_weight={signal.target_weight:.4f}",
        )

    def protective_exit(self, position: PositionState) -> OrderIntent | None:
        if position.quantity <= 0:
            return None
        if position.return_pct <= -self.stop_loss_pct:
            return OrderIntent(
                symbol=position.symbol,
                quantity=position.quantity,
                side="sell",
                reference_price=position.current_price,
                reason="stop_loss",
            )
        if position.return_pct >= self.take_profit_pct:
            return OrderIntent(
                symbol=position.symbol,
                quantity=position.quantity,
                side="sell",
                reference_price=position.current_price,
                reason="take_profit",
            )
        return None

from __future__ import annotations

import logging
import time
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from models import OrderExecution, OrderIntent, PositionState

LOGGER = logging.getLogger(__name__)

# A partial fill is deliberately not terminal: the remaining quantity may
# still be active and fill later.
_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}


class AlpacaPaperBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        fill_poll_seconds: float = 1.0,
        fill_timeout_seconds: float = 10.0,
    ) -> None:
        self.client = TradingClient(api_key, secret_key, paper=True)
        self.fill_poll_seconds = fill_poll_seconds
        self.fill_timeout_seconds = fill_timeout_seconds

    def get_account(self) -> Any:
        return self.client.get_account()

    def get_positions(self) -> dict[str, PositionState]:
        positions: dict[str, PositionState] = {}
        for item in self.client.get_all_positions():
            positions[item.symbol] = PositionState(
                symbol=item.symbol,
                quantity=float(item.qty),
                average_entry_price=float(item.avg_entry_price),
                current_price=float(item.current_price),
            )
        return positions

    @staticmethod
    def _status_text(order: Any) -> str:
        status = getattr(order, "status", "unknown")
        return str(getattr(status, "value", status)).lower()

    @staticmethod
    def _filled_quantity(order: Any) -> float:
        value = getattr(order, "filled_qty", 0) or 0
        return float(value)

    @staticmethod
    def _average_fill_price(order: Any) -> float | None:
        value = getattr(order, "filled_avg_price", None)
        return float(value) if value not in (None, "") else None

    def _wait_for_terminal_status(self, order_id: str) -> tuple[Any, bool]:
        deadline = time.monotonic() + self.fill_timeout_seconds
        order = self.client.get_order_by_id(order_id)

        while order.status not in _TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(self.fill_poll_seconds)
            order = self.client.get_order_by_id(order_id)

        terminal = order.status in _TERMINAL_STATUSES
        if terminal:
            LOGGER.info(
                "Order %s reached terminal status=%s filled_qty=%s",
                order_id,
                self._status_text(order),
                self._filled_quantity(order),
            )
        else:
            LOGGER.warning(
                "Order %s remains open after %.1fs: status=%s filled_qty=%s. "
                "The engine will not assume a full fill.",
                order_id,
                self.fill_timeout_seconds,
                self._status_text(order),
                self._filled_quantity(order),
            )
        return order, terminal

    def submit_market_order(self, intent: OrderIntent, retries: int = 3) -> OrderExecution:
        if intent.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if intent.side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {intent.side}")

        order_request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=round(intent.quantity, 6),
            side=OrderSide.BUY if intent.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"system-{intent.symbol}-{int(time.time() * 1000)}",
        )

        for attempt in range(1, retries + 1):
            try:
                submitted = self.client.submit_order(order_data=order_request)
                order_id = str(submitted.id)
                LOGGER.info(
                    "Order submitted: symbol=%s side=%s qty=%s id=%s reason=%s",
                    intent.symbol,
                    intent.side,
                    intent.quantity,
                    order_id,
                    intent.reason,
                )
                final_order, terminal = self._wait_for_terminal_status(order_id)
                result = OrderExecution(
                    symbol=intent.symbol,
                    side=intent.side,
                    requested_quantity=intent.quantity,
                    filled_quantity=self._filled_quantity(final_order),
                    status=self._status_text(final_order),
                    submitted=True,
                    terminal=terminal,
                    order_id=order_id,
                    average_fill_price=self._average_fill_price(final_order),
                    raw_order=final_order,
                )
                if result.permanently_failed:
                    LOGGER.error(
                        "Order failed permanently: symbol=%s status=%s id=%s",
                        intent.symbol,
                        result.status,
                        order_id,
                    )
                return result
            except Exception:
                LOGGER.exception(
                    "Order attempt %d/%d failed before a usable result for %s",
                    attempt,
                    retries,
                    intent.symbol,
                )
                if attempt == retries:
                    raise
                time.sleep(2 ** (attempt - 1))

        raise RuntimeError("Unreachable order submission state")

    def cancel_all_orders(self) -> None:
        self.client.cancel_orders()
        LOGGER.warning("All open paper orders canceled")

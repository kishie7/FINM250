from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote

LOGGER = logging.getLogger(__name__)


class LiveMarketDataStream:
    def __init__(self, api_key: str, secret_key: str, symbols: list[str]) -> None:
        self.symbols = symbols
        self.stream = StockDataStream(api_key, secret_key)

    def subscribe_bars(self, handler: Callable[[Bar], Awaitable[None]]) -> None:
        self.stream.subscribe_bars(handler, *self.symbols)

    def subscribe_quotes(self, handler: Callable[[Quote], Awaitable[None]]) -> None:
        self.stream.subscribe_quotes(handler, *self.symbols)

    def run(self) -> None:
        LOGGER.info("Starting Alpaca stock stream for %s", ", ".join(self.symbols))
        self.stream.run()

    def stop(self) -> None:
        LOGGER.info("Stopping Alpaca stock stream")
        self.stream.stop()

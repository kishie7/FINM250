from __future__ import annotations

from datetime import datetime
import logging
from typing import Iterable

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

LOGGER = logging.getLogger(__name__)


class HistoricalDataService:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self.client.get_stock_bars(request).df
        if bars.empty:
            raise RuntimeError("Alpaca returned no historical bars.")

        bars = bars.reset_index()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        LOGGER.info("Downloaded %d historical bars for %d symbols", len(bars), bars.symbol.nunique())
        return bars

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


class SQLiteBarStore:
    def __init__(self, database_path: str = "data/market_data.db") -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_bars(self, bars: pd.DataFrame) -> None:
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"Bars missing required columns: {sorted(missing)}")
        payload = bars.copy()
        payload["timestamp"] = pd.to_datetime(payload["timestamp"], utc=True).astype(str)
        with sqlite3.connect(self.path) as connection:
            payload.to_sql("bars", connection, if_exists="append", index=False)

    def read_bars(self, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM bars"
        parameters: tuple[str, ...] = ()
        if symbol:
            query += " WHERE symbol = ?"
            parameters = (symbol,)
        query += " ORDER BY timestamp"
        with sqlite3.connect(self.path) as connection:
            result = pd.read_sql_query(query, connection, params=parameters)
        if not result.empty:
            result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        return result

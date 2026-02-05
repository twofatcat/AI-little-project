"""
Abstract base: fetch_hourly(symbol, start_dt, end_dt) -> DataFrame with unified column names.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


# Unified column names for all adapters
DATETIME_COL = "datetime"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
EXTRA_COLS = ["adj_close", "dividend"]
EXPECTED_COLS = [DATETIME_COL] + OHLCV_COLS + EXTRA_COLS


class BaseHourlySource(ABC):
    """Abstract data source for hourly OHLCV. All adapters return same column set."""

    @abstractmethod
    def fetch_hourly(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly bars for symbol from start_dt to end_dt (inclusive where applicable).
        Return DataFrame with columns: datetime, open, high, low, close, volume, adj_close, dividend.
        datetime: timezone-aware or naive consistently. Return None on failure.
        """
        pass

"""
US market: yfinance 1h intraday. Max 60 days history.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseHourlySource
from date_compat import safe_date_ymd

# yfinance intraday is limited to last 60 days
MAX_DAYS_US = 60
logger = logging.getLogger(__name__)


def _normalize_us(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """Map yfinance result to unified columns. Single-ticker columns may be top-level or under ticker."""
    if df is None or df.empty:
        return None
    df = df.copy()
    # Single ticker: columns can be Open, High, ... or (Open, symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Standard names
    col_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adj_close": "adj_close",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    df = df.rename(columns={"adj close": "adj_close"}) if "adj close" in df.columns else df
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    df["dividend"] = None
    # Keep only expected columns
    want = ["datetime", "open", "high", "low", "close", "volume", "adj_close", "dividend"]
    have = [c for c in want if c in df.columns]
    return df[have]


class USYFinanceSource(BaseHourlySource):
    """Hourly data from Yahoo Finance for US symbols. 60-day max window."""

    def __init__(self) -> None:
        self.source_name = "yfinance"
        self._date_compat_logged = False

    def fetch_hourly(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception:
            # Dependency missing or import failure
            return None
        # Clamp to last 60 days (use naive UTC for comparison)
        now = datetime.utcnow()
        max_start = now - timedelta(days=MAX_DAYS_US)
        start_naive = start_dt.replace(tzinfo=None) if start_dt.tzinfo else start_dt
        end_naive = end_dt.replace(tzinfo=None) if end_dt.tzinfo else end_dt
        if start_naive < max_start:
            start_dt = max_start
        if end_naive > now:
            end_dt = now
        if start_dt >= end_dt:
            return None
        # yfinance `end` is exclusive; add 1 day to include the last trading day
        start_s, start_norm = safe_date_ymd(start_dt)
        end_s, end_norm = safe_date_ymd(end_dt + timedelta(days=1))
        if (start_norm or end_norm) and not self._date_compat_logged:
            # One-time hint for systems with localized calendar/digits.
            logger.warning("date_compat: normalized localized date format for yfinance requests")
            self._date_compat_logged = True
        try:
            data = yf.download(
                symbol,
                interval="1h",
                start=start_s,
                end=end_s,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            return None
        return _normalize_us(data, symbol)

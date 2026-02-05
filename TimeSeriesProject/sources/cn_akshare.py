"""
CN A-share: AKShare 东财 60 分钟 K 线. Max ~3 months (90 days) history.
"""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseHourlySource
from zoneinfo import ZoneInfo

# 东财分钟数据约 3 个月
MAX_DAYS_CN = 90

# AKShare column names (Chinese)
COL_TIME = "时间"
COL_OPEN = "开盘"
COL_CLOSE = "收盘"
COL_HIGH = "最高"
COL_LOW = "最低"
COL_VOL = "成交量"
COL_AMOUNT = "成交额"


def _normalize_cn(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map AKShare result to unified columns."""
    if df is None or df.empty:
        return None
    # 时间/开盘/收盘/最高/最低/成交量
    col_map = {
        COL_TIME: "datetime",
        COL_OPEN: "open",
        COL_CLOSE: "close",
        COL_HIGH: "high",
        COL_LOW: "low",
        COL_VOL: "volume",
        COL_AMOUNT: "amount",
    }
    out = df.copy()
    for cn_name, en_name in col_map.items():
        if cn_name in out.columns:
            out = out.rename(columns={cn_name: en_name})
    out["datetime"] = pd.to_datetime(out["datetime"])
    # 东财复权时 收盘 已是复权价，用作 adj_close
    if "close" in out.columns:
        out["adj_close"] = out["close"]
    out["dividend"] = None
    want = ["datetime", "open", "high", "low", "close", "volume", "amount", "adj_close", "dividend"]
    have = [c for c in want if c in out.columns]
    return out[have]


class CNAKShareSource(BaseHourlySource):
    """Hourly (60-min) data from AKShare 东财 for A-share. ~90-day max window."""

    def __init__(self, adjust: str = "qfq"):
        # qfq=前复权, hfq=后复权, ""=不复权
        self.adjust = adjust
        self.source_name = "akshare_eastmoney"

    def fetch_hourly(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[pd.DataFrame]:
        # Clamp to ~90 days
        # IMPORTANT: input start_dt/end_dt are expected in China local time (Asia/Shanghai)
        now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        max_start = now - timedelta(days=MAX_DAYS_CN)
        if start_dt < max_start:
            start_dt = max_start
        if end_dt > now:
            end_dt = now
        if start_dt >= end_dt:
            return None
        start_s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            import akshare as ak
            data = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=start_s,
                end_date=end_s,
                period="60",
                adjust=self.adjust,
            )
        except Exception:
            return None
        df = _normalize_cn(data)
        return df


def cn_code_to_prefix(code6: str) -> str:
    """
    Convert 6-digit A-share code to akshare/Sina-style symbol with exchange prefix.
    - 6xxxx/68xxx/9xxxx -> sh
    - 0xxxx/3xxxx -> sz
    - 8xxxx/4xxxx -> bj (Beijing Stock Exchange)
    """
    code6 = str(code6).zfill(6)
    if code6.startswith(("60", "68", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99")) or code6.startswith(
        "6"
    ):
        return f"sh{code6}"
    if code6.startswith(("00", "30")) or code6.startswith(("0", "3")):
        return f"sz{code6}"
    if code6.startswith(("8", "4")):
        return f"bj{code6}"
    # fallback
    return f"sz{code6}"


class CNAKShareSinaFallbackSource(BaseHourlySource):
    """
    Fallback: AKShare 新浪分时数据 stock_zh_a_minute.
    Note: may be rate-limited; caller should throttle/retry.
    """

    def __init__(self, adjust: str = "qfq") -> None:
        self.adjust = adjust
        self.source_name = "akshare_sina"

    def fetch_hourly(self, symbol: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak

            sina_symbol = cn_code_to_prefix(symbol)
            data = ak.stock_zh_a_minute(symbol=sina_symbol, period="60", adjust=self.adjust)
        except Exception:
            return None
        if data is None or data.empty:
            return None
        # 新浪返回列: day, open, high, low, close, volume
        df = data.copy()
        if "day" in df.columns:
            df = df.rename(columns={"day": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        # filter range
        df = df[(df["datetime"] >= pd.to_datetime(start_dt)) & (df["datetime"] <= pd.to_datetime(end_dt))]
        df["adj_close"] = df["close"]
        df["dividend"] = None
        # amount not provided; leave missing
        want = ["datetime", "open", "high", "low", "close", "volume", "amount", "adj_close", "dividend"]
        for c in want:
            if c not in df.columns:
                df[c] = None
        return df[want]

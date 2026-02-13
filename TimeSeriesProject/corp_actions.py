"""
Corporate actions / adjustment factor utilities.

We store a daily adjustment factor adj_factor(trade_date) and apply it to intraday bars:
  price_adj = price_raw * adj_factor

For US (yfinance):
  adj_factor = Adj Close / Close (daily)

For CN:
  robust adj_factor requires both raw and adjusted prices. This repo provides an
  optional dual-fetch mode (unadjusted + qfq) to compute daily factors.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from date_compat import safe_date_ymd


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_yf_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out


def fetch_us_daily_adj_factors(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch US daily Close/Adj Close from yfinance and compute adj_factor per trade_date.
    Returns DataFrame columns: symbol, trade_date, adj_factor, source, updated_at_utc
    """
    import yfinance as yf

    if start_date > end_date:
        return pd.DataFrame()

    start_s, _ = safe_date_ymd(start_date)
    end_s, _ = safe_date_ymd(end_date + timedelta(days=1))  # yfinance end is exclusive
    try:
        raw = yf.download(
            symbol,
            interval="1d",
            start=start_s,
            end=end_s,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    df = _normalize_yf_daily(raw)
    if df.empty:
        return pd.DataFrame()

    close = pd.to_numeric(df.get("close"), errors="coerce")
    adj = pd.to_numeric(df.get("adj_close"), errors="coerce")
    if close is None or adj is None:
        return pd.DataFrame()

    factor = adj / close
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # trade_date as YYYY-MM-DD (yfinance daily index is date-like)
    idx = pd.to_datetime(df.index, errors="coerce")
    trade_date = pd.Series(idx.date).astype(str)
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": trade_date.values,
            "adj_factor": factor.values.astype(float),
            "source": "yfinance_daily",
            "updated_at_utc": _now_utc_iso(),
        }
    )
    out = out.dropna(subset=["trade_date", "adj_factor"])
    # guardrails
    out.loc[~np.isfinite(out["adj_factor"]), "adj_factor"] = 1.0
    out.loc[out["adj_factor"] <= 0, "adj_factor"] = 1.0
    return out.reset_index(drop=True)


def compute_cn_daily_adj_factors_from_dual_intraday(
    *,
    unadjusted_v2: pd.DataFrame,
    adjusted_v2: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Compute CN daily adj_factor from dual intraday series (raw + qfq).

    Method:
    - align by ts_end_utc
    - factor_bar = close_adj / close_raw
    - per trade_date, use the last available factor_bar as daily adj_factor
    """
    if unadjusted_v2 is None or unadjusted_v2.empty or adjusted_v2 is None or adjusted_v2.empty:
        return pd.DataFrame()

    u = unadjusted_v2.copy()
    a = adjusted_v2.copy()
    for df in (u, a):
        if "flag_off_calendar" in df.columns:
            df.drop(df[df["flag_off_calendar"].fillna(0).astype(int) != 0].index, inplace=True)

    if u.empty or a.empty:
        return pd.DataFrame()

    u = u[["ts_end_utc", "dt_end_local", "close"]].rename(columns={"close": "close_raw"})
    a = a[["ts_end_utc", "dt_end_local", "close"]].rename(columns={"close": "close_adj"})
    m = u.merge(a, on="ts_end_utc", how="inner", suffixes=("", ""))
    if m.empty:
        return pd.DataFrame()

    m["trade_date"] = m["dt_end_local"].astype(str).str[:10]
    m["close_raw"] = pd.to_numeric(m["close_raw"], errors="coerce")
    m["close_adj"] = pd.to_numeric(m["close_adj"], errors="coerce")
    m["factor_bar"] = (m["close_adj"] / m["close_raw"]).replace([np.inf, -np.inf], np.nan)
    m = m.dropna(subset=["trade_date", "factor_bar"])
    if m.empty:
        return pd.DataFrame()

    # pick the last factor within each day (session close bucket)
    m = m.sort_values(["trade_date", "ts_end_utc"])
    daily = m.groupby("trade_date", as_index=False).tail(1)[["trade_date", "factor_bar"]].rename(
        columns={"factor_bar": "adj_factor"}
    )
    daily["symbol"] = symbol
    daily["source"] = "akshare_dual_intraday"
    daily["updated_at_utc"] = _now_utc_iso()
    daily.loc[~np.isfinite(daily["adj_factor"]), "adj_factor"] = 1.0
    daily.loc[daily["adj_factor"] <= 0, "adj_factor"] = 1.0
    return daily[["symbol", "trade_date", "adj_factor", "source", "updated_at_utc"]].reset_index(drop=True)


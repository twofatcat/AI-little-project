"""
Qlib-style canonicalization & cleaning utilities for hourly bar data.

Goal:
- unify timestamps (compute bar_end, store ts_end_utc as the primary time index)
- store both UTC and exchange-local end timestamps
- dedupe/sort/validate OHLCV
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from trading_calendar import generate_intraday_buckets

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass(frozen=True)
class MarketTimeSpec:
    exchange_tz: str
    input_tz: str
    timestamp_kind: str  # "bar_start" or "bar_end"


def _require_zoneinfo() -> None:
    if ZoneInfo is None:
        raise RuntimeError("Python zoneinfo not available; please use Python 3.9+")


def iso_z(dt_utc: pd.Timestamp) -> str:
    """UTC timestamp to ISO string with Z."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.tz_localize("UTC")
    dt_utc = dt_utc.tz_convert("UTC")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_unix_seconds(ts: pd.Series) -> pd.Series:
    """
    Convert timezone-aware datetime Series to Unix seconds robustly.
    Works across datetime64 units (ns/us/ms/s), unlike raw int64 // 1e9.
    """
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    return ((ts - epoch) / pd.Timedelta(seconds=1)).astype("int64")


def canonicalize_hourly_bars(
    df: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    source: str,
    spec: MarketTimeSpec,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Convert raw hourly bars into canonical schema.

    Input df must include:
      datetime, open, high, low, close, volume
    Optional:
      amount, adj_close, dividend

    Returns a DataFrame with canonical columns:
      symbol, ts_end_utc, dt_end_utc, dt_end_local, exchange_tz, interval,
      open, high, low, close, volume, amount, adj_close, dividend, source, updated_at_utc
    """
    _require_zoneinfo()
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("Input df missing required column: datetime")

    # Parse datetime column
    dt = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.assign(_dt=dt).dropna(subset=["_dt"])

    # Attach/convert timezone for input timestamps
    if getattr(out["_dt"].dt, "tz", None) is None:
        out["_dt"] = out["_dt"].dt.tz_localize(ZoneInfo(spec.input_tz))
    else:
        out["_dt"] = out["_dt"].dt.tz_convert(ZoneInfo(spec.input_tz))

    step = pd.Timedelta(hours=1)
    if spec.timestamp_kind == "bar_start":
        dt_end_in_input_tz = out["_dt"] + step
    elif spec.timestamp_kind == "bar_end":
        dt_end_in_input_tz = out["_dt"]
    else:
        raise ValueError(f"Unknown timestamp_kind: {spec.timestamp_kind}")

    # Canonical end time (UTC)
    dt_end_utc = dt_end_in_input_tz.dt.tz_convert(ZoneInfo("UTC"))
    dt_end_local = dt_end_utc.dt.tz_convert(ZoneInfo(spec.exchange_tz))

    # Primary key: ts_end_utc (seconds since epoch)
    ts_end_utc = _to_unix_seconds(dt_end_utc)

    # Ensure numeric columns
    for c in ["open", "high", "low", "close", "volume", "amount", "adj_close", "dividend"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Amount fallback (US often lacks amount)
    if "amount" not in out.columns or out["amount"].isna().all():
        if "volume" in out.columns and "close" in out.columns:
            out["amount"] = out["volume"] * out["close"]
        else:
            out["amount"] = pd.NA

    # adj_close fallback
    if "adj_close" not in out.columns or out["adj_close"].isna().all():
        out["adj_close"] = out.get("close")

    if "dividend" not in out.columns:
        out["dividend"] = pd.NA

    # Basic validity: prices must be > 0; high/low must bound open/close
    price_cols = ["open", "high", "low", "close"]
    for c in price_cols:
        if c not in out.columns:
            out[c] = pd.NA
    valid_price = (out["close"] > 0) & (out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0)
    bounded = (out["high"] >= out[["open", "close", "low"]].max(axis=1)) & (
        out["low"] <= out[["open", "close", "high"]].min(axis=1)
    )
    out = out[valid_price & bounded].copy()
    if out.empty:
        return pd.DataFrame()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out2 = pd.DataFrame(
        {
            "symbol": symbol,
            "ts_end_utc": ts_end_utc.loc[out.index].astype("int64"),
            "dt_end_utc": dt_end_utc.loc[out.index].map(iso_z),
            "dt_end_local": dt_end_local.loc[out.index].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "exchange_tz": spec.exchange_tz,
            "interval": interval,
            "open": out["open"].astype(float),
            "high": out["high"].astype(float),
            "low": out["low"].astype(float),
            "close": out["close"].astype(float),
            "volume": out.get("volume").astype(float) if "volume" in out.columns else pd.NA,
            "amount": out.get("amount").astype(float) if "amount" in out.columns else pd.NA,
            "adj_close": out.get("adj_close").astype(float) if "adj_close" in out.columns else pd.NA,
            "dividend": out.get("dividend").astype(float) if "dividend" in out.columns else pd.NA,
            "source": source,
            "updated_at_utc": now_utc,
        }
    )

    # Dedup/sort by ts_end_utc
    out2 = out2.drop_duplicates(subset=["symbol", "ts_end_utc"]).sort_values("ts_end_utc")
    return out2.reset_index(drop=True)


def canonicalize_hourly_bars_v2(
    df: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    source: str,
    spec: MarketTimeSpec,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Calendar-aware canonicalization (Silver layer).

    Differences vs v1 (`canonicalize_hourly_bars`):
    - Derives (ts_start_utc, ts_end_utc) by aligning input timestamps onto the
      official trading-calendar buckets (early closes, lunch breaks, DST).
    - Does NOT drop invalid rows; instead produces QC flags that downstream
      (Gold) can use to isolate/NULL-out bars and produce masks.

    Output columns (Silver v2):
      symbol,
      ts_start_utc, ts_end_utc,
      dt_start_utc, dt_end_utc,
      dt_start_local, dt_end_local,
      exchange_tz, interval,
      open, high, low, close, volume, amount, adj_close, dividend,
      source, updated_at_utc,
      flag_off_calendar, flag_any_null_price, flag_nonpositive_price,
      flag_ohlc_inconsistent, flag_volume_negative
    """
    _require_zoneinfo()
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("Input df missing required column: datetime")

    # Parse datetime column
    dt = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.assign(_dt=dt)

    # Attach/convert timezone for input timestamps
    # Note: for robust alignment we always interpret incoming timestamps in spec.input_tz.
    if getattr(out["_dt"].dt, "tz", None) is None:
        out["_dt"] = out["_dt"].dt.tz_localize(ZoneInfo(spec.input_tz))
    else:
        out["_dt"] = out["_dt"].dt.tz_convert(ZoneInfo(spec.input_tz))

    # Numeric conversions (keep NaNs; we will flag instead of dropping)
    for c in ["open", "high", "low", "close", "volume", "amount", "adj_close", "dividend"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Amount fallback (US often lacks amount)
    if "amount" not in out.columns or out["amount"].isna().all():
        if "volume" in out.columns and "close" in out.columns:
            out["amount"] = out["volume"] * out["close"]
        else:
            out["amount"] = pd.NA

    # adj_close fallback
    if "adj_close" not in out.columns or out["adj_close"].isna().all():
        out["adj_close"] = out.get("close")

    if "dividend" not in out.columns:
        out["dividend"] = pd.NA

    # Key timestamp to align onto calendar buckets
    if spec.timestamp_kind == "bar_start":
        dt_key_utc = out["_dt"].dt.tz_convert("UTC")
        key_side = "start"
    elif spec.timestamp_kind == "bar_end":
        dt_key_utc = out["_dt"].dt.tz_convert("UTC")
        key_side = "end"
    else:
        raise ValueError(f"Unknown timestamp_kind: {spec.timestamp_kind}")

    # Generate trading-calendar buckets that cover the data range.
    dt_min = dt_key_utc.min()
    dt_max = dt_key_utc.max()
    if pd.isna(dt_min) or pd.isna(dt_max):
        return pd.DataFrame()

    # Expand window so buckets are fully included (bucket generator keeps buckets fully inside window).
    start_utc = (dt_min - pd.Timedelta(days=3)).floor("D")
    end_utc = (dt_max + pd.Timedelta(days=3)).ceil("D")
    buckets = generate_intraday_buckets(market=market, start_utc=start_utc, end_utc=end_utc, interval=interval)

    # Compute integer key in seconds (robust to datetime precision unit)
    ts_key_utc = _to_unix_seconds(dt_key_utc)
    out["_ts_key_utc"] = ts_key_utc

    # Align to buckets by ts_start_utc (bar_start) or ts_end_utc (bar_end)
    bucket_key = "ts_start_utc" if key_side == "start" else "ts_end_utc"
    merged = out.merge(buckets, left_on="_ts_key_utc", right_on=bucket_key, how="left", suffixes=("", "_b"))

    flag_off_calendar = merged["ts_start_utc"].isna() | merged["ts_end_utc"].isna()

    # Fallback timestamps for off-calendar rows (kept, but flagged).
    step_s = 3600
    if key_side == "start":
        ts_start_utc = merged["_ts_key_utc"].astype("int64")
        ts_end_utc = merged["ts_end_utc"].fillna(ts_start_utc + step_s).astype("int64")
        ts_start_utc = merged["ts_start_utc"].fillna(ts_start_utc).astype("int64")
    else:
        ts_end_utc = merged["_ts_key_utc"].astype("int64")
        ts_start_utc = merged["ts_start_utc"].fillna(ts_end_utc - step_s).astype("int64")
        ts_end_utc = merged["ts_end_utc"].fillna(ts_end_utc).astype("int64")

    def _dt_from_ts_s(ts_s: pd.Series) -> pd.Series:
        # Force Series output (avoid DatetimeIndex edge cases).
        return pd.to_datetime(pd.Series(ts_s, index=ts_s.index), unit="s", utc=True)

    dt_start_utc_ts = _dt_from_ts_s(ts_start_utc)
    dt_end_utc_ts = _dt_from_ts_s(ts_end_utc)

    # Use bucket-provided dt strings when available; otherwise compute from timestamps.
    dt_start_utc = merged["dt_start_utc"].fillna(dt_start_utc_ts.map(iso_z))
    dt_end_utc = merged["dt_end_utc"].fillna(dt_end_utc_ts.map(iso_z))

    # local strings
    tz_ex = ZoneInfo(spec.exchange_tz)
    dt_start_local = merged["dt_start_local"].fillna(
        dt_start_utc_ts.dt.tz_convert(tz_ex).dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    dt_end_local = merged["dt_end_local"].fillna(
        dt_end_utc_ts.dt.tz_convert(tz_ex).dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # QC flags (do not drop)
    price_cols = ["open", "high", "low", "close"]
    for c in price_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    flag_any_null_price = merged[price_cols].isna().any(axis=1)
    flag_nonpositive_price = (merged[price_cols] <= 0).any(axis=1)
    # bounded only meaningful when prices present; if NaNs, mark inconsistent as False and rely on null flag
    hi = merged["high"]
    lo = merged["low"]
    oc_max = merged[["open", "close", "low"]].max(axis=1)
    oc_min = merged[["open", "close", "high"]].min(axis=1)
    flag_ohlc_inconsistent = (hi < oc_max) | (lo > oc_min)
    flag_volume_negative = (merged.get("volume") < 0) if "volume" in merged.columns else False

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out2 = pd.DataFrame(
        {
            "symbol": symbol,
            "ts_start_utc": ts_start_utc.astype("int64"),
            "ts_end_utc": ts_end_utc.astype("int64"),
            "dt_start_utc": dt_start_utc,
            "dt_end_utc": dt_end_utc,
            "dt_start_local": dt_start_local,
            "dt_end_local": dt_end_local,
            "exchange_tz": spec.exchange_tz,
            "interval": interval,
            "open": pd.to_numeric(merged.get("open"), errors="coerce").astype(float),
            "high": pd.to_numeric(merged.get("high"), errors="coerce").astype(float),
            "low": pd.to_numeric(merged.get("low"), errors="coerce").astype(float),
            "close": pd.to_numeric(merged.get("close"), errors="coerce").astype(float),
            "volume": pd.to_numeric(merged.get("volume"), errors="coerce").astype(float) if "volume" in merged.columns else pd.NA,
            "amount": pd.to_numeric(merged.get("amount"), errors="coerce").astype(float) if "amount" in merged.columns else pd.NA,
            "adj_close": pd.to_numeric(merged.get("adj_close"), errors="coerce").astype(float)
            if "adj_close" in merged.columns
            else pd.NA,
            "dividend": pd.to_numeric(merged.get("dividend"), errors="coerce").astype(float)
            if "dividend" in merged.columns
            else pd.NA,
            "source": source,
            "updated_at_utc": now_utc,
            "flag_off_calendar": flag_off_calendar.astype(int),
            "flag_any_null_price": flag_any_null_price.astype(int),
            "flag_nonpositive_price": flag_nonpositive_price.astype(int),
            "flag_ohlc_inconsistent": flag_ohlc_inconsistent.fillna(True).astype(int),
            "flag_volume_negative": (
                pd.to_numeric(merged["volume"], errors="coerce").fillna(0) < 0
            ).astype(int)
            if "volume" in merged.columns
            else 0,
        }
    )

    # Dedup/sort by ts_end_utc (prefer in-calendar; then fewer nulls; then latest updated_at)
    out2["_null_cnt"] = out2[["open", "high", "low", "close"]].isna().sum(axis=1)
    out2 = out2.sort_values(["flag_off_calendar", "_null_cnt", "updated_at_utc"], ascending=[True, True, False])
    out2 = out2.drop_duplicates(subset=["symbol", "ts_end_utc"]).sort_values("ts_end_utc")
    out2 = out2.drop(columns=["_null_cnt"])
    return out2.reset_index(drop=True)


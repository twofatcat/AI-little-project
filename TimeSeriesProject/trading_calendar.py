"""
Trading calendar utilities (industrial-grade time semantics).

This module is the foundation for:
- calendar-aware bar bucket generation (early closes, lunch breaks, DST)
- grid alignment (expected buckets vs. observed bars)

We use `exchange_calendars` (the same family used by Zipline) to obtain official
exchange sessions and intraday breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

import config
import warnings

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass(frozen=True)
class BucketSpec:
    market: str
    interval: str = "1h"

    @property
    def exchange_tz(self) -> str:
        if self.market == config.MARKET_US:
            return config.TZ_US
        if self.market == config.MARKET_CN:
            return config.TZ_CN
        raise ValueError(f"Unknown market: {self.market}")

    @property
    def calendar_name(self) -> str:
        # NOTE: for US equities, major venues (NYSE/Nasdaq) share the same
        # RTH schedule and early closes for most practical purposes.
        if self.market == config.MARKET_US:
            return "XNYS"
        if self.market == config.MARKET_CN:
            return "XSHG"
        raise ValueError(f"Unknown market: {self.market}")


def _require_zoneinfo() -> None:
    if ZoneInfo is None:
        raise RuntimeError("Python zoneinfo not available; please use Python 3.9+")


def _ensure_timestamp(ts) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _ensure_timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _iso_z(ts_utc: pd.Timestamp) -> str:
    ts_utc = _ensure_utc(ts_utc)
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _series_to_unix_seconds(ts: pd.Series) -> pd.Series:
    """
    Convert UTC datetime Series to Unix seconds robustly across datetime units.
    """
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    return ((ts - epoch) / pd.Timedelta(seconds=1)).astype("int64")


@lru_cache(maxsize=8)
def _get_calendar(calendar_name: str):
    try:
        import exchange_calendars as xcals
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `exchange_calendars`. Install with `pip install -r requirements.txt`."
        ) from e
    return xcals.get_calendar(calendar_name)


def get_exchange_calendar(market: str):
    spec = BucketSpec(market=market)
    return _get_calendar(spec.calendar_name)


def _session_break_start(cal, session) -> Optional[pd.Timestamp]:
    # exchange_calendars API includes these; keep defensive fallback.
    fn = getattr(cal, "session_break_start", None)
    if fn is None:
        return None
    try:
        return fn(session)
    except Exception:  # pragma: no cover
        return None


def _session_break_end(cal, session) -> Optional[pd.Timestamp]:
    fn = getattr(cal, "session_break_end", None)
    if fn is None:
        return None
    try:
        return fn(session)
    except Exception:  # pragma: no cover
        return None


def _sessions_in_range(cal, start_d: date, end_d: date) -> pd.DatetimeIndex:
    # Accept date objects and convert to ISO strings.
    return cal.sessions_in_range(start_d.isoformat(), end_d.isoformat())


def generate_intraday_buckets(
    *,
    market: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Generate official intraday buckets for a market between [start_utc, end_utc].

    Buckets are returned with UTC + local timestamps:
      - ts_start_utc, ts_end_utc (int seconds)
      - dt_start_utc, dt_end_utc (ISO with Z)
      - dt_start_local, dt_end_local (exchange local, no offset; timezone stored separately)
      - trade_date (local YYYY-MM-DD)

    Notes:
    - Handles US early closes automatically (last bucket may be < 60min).
    - Handles CN lunch break via calendar break times.
    """
    _require_zoneinfo()

    if interval != "1h":
        raise NotImplementedError("Only interval='1h' is supported for now.")

    start_utc = _ensure_utc(_ensure_timestamp(start_utc))
    end_utc = _ensure_utc(_ensure_timestamp(end_utc))
    if start_utc >= end_utc:
        return pd.DataFrame()

    spec = BucketSpec(market=market, interval=interval)
    exchange_tz = spec.exchange_tz
    try:
        cal = get_exchange_calendar(market)
    except Exception as e:  # pragma: no cover
        # Fallback mode: no exchange_calendars installed. We still generate a best-effort
        # weekday-only schedule with fixed session hours. This preserves core semantics
        # (e.g. US last half hour ends at 16:00 local) and DST handling via zoneinfo,
        # but will NOT be accurate for holidays / special sessions / early closes.
        warnings.warn(
            f"exchange_calendars unavailable ({e}); using simplified weekday-only schedule. "
            "Holidays/early-closes may be incorrect.",
            RuntimeWarning,
        )
        return _generate_intraday_buckets_fallback(
            market=market, start_utc=start_utc, end_utc=end_utc, interval=interval
        )

    # Expand session label range slightly to ensure we cover the bucket that
    # might start before start_utc but end after it (and vice versa).
    start_local_d = (start_utc.tz_convert(exchange_tz).date() - timedelta(days=2))
    end_local_d = (end_utc.tz_convert(exchange_tz).date() + timedelta(days=2))
    sessions = _sessions_in_range(cal, start_local_d, end_local_d)
    if sessions.empty:
        return pd.DataFrame()

    step = pd.Timedelta(hours=1)
    rows: list[dict] = []
    for sess in sessions:
        open_utc = cal.session_open(sess)
        close_utc = cal.session_close(sess)
        bstart = _session_break_start(cal, sess)
        bend = _session_break_end(cal, sess)

        segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if bstart is not None and bend is not None and pd.notna(bstart) and pd.notna(bend):
            segments.append((open_utc, bstart))
            segments.append((bend, close_utc))
        else:
            segments.append((open_utc, close_utc))

        for seg_start, seg_end in segments:
            t = seg_start
            while t < seg_end:
                t_end = min(t + step, seg_end)
                # Keep only buckets fully inside the requested UTC window.
                if t >= start_utc and t_end <= end_utc:
                    t_local = t.tz_convert(exchange_tz)
                    t_end_local = t_end.tz_convert(exchange_tz)
                    rows.append(
                        {
                            "dt_start_utc": t,
                            "dt_end_utc": t_end,
                            "dt_start_local": t_local.strftime("%Y-%m-%dT%H:%M:%S"),
                            "dt_end_local": t_end_local.strftime("%Y-%m-%dT%H:%M:%S"),
                            "exchange_tz": exchange_tz,
                            "interval": interval,
                            "trade_date": t_end_local.strftime("%Y-%m-%d"),
                        }
                    )
                t = t + step

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["ts_start_utc"] = _series_to_unix_seconds(out["dt_start_utc"])
    out["ts_end_utc"] = _series_to_unix_seconds(out["dt_end_utc"])
    out["dt_start_utc"] = out["dt_start_utc"].map(_iso_z)
    out["dt_end_utc"] = out["dt_end_utc"].map(_iso_z)
    out = out.sort_values(["ts_end_utc", "ts_start_utc"]).drop_duplicates(subset=["ts_start_utc", "ts_end_utc"])
    return out.reset_index(drop=True)


def _generate_intraday_buckets_fallback(
    *,
    market: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fallback bucket generation without exchange_calendars.

    Rules:
    - trading days = weekdays (Mon-Fri) in exchange local time
    - US: 09:30-16:00
    - CN: 09:30-11:30 and 13:00-15:00

    Does NOT handle holidays / early closes.
    """
    _require_zoneinfo()
    if interval != "1h":
        raise NotImplementedError("Only interval='1h' is supported for now.")

    spec = BucketSpec(market=market, interval=interval)
    tz = ZoneInfo(spec.exchange_tz)
    start_utc = _ensure_utc(_ensure_timestamp(start_utc))
    end_utc = _ensure_utc(_ensure_timestamp(end_utc))
    if start_utc >= end_utc:
        return pd.DataFrame()

    start_local = start_utc.tz_convert(tz)
    end_local = end_utc.tz_convert(tz)
    start_d = start_local.date()
    end_d = end_local.date()

    if market == config.MARKET_US:
        segments = [("09:30:00", "16:00:00")]
    elif market == config.MARKET_CN:
        segments = [("09:30:00", "11:30:00"), ("13:00:00", "15:00:00")]
    else:
        raise ValueError(f"Unknown market: {market}")

    step = pd.Timedelta(hours=1)
    rows: list[dict] = []
    d = start_d
    while d <= end_d:
        if d.weekday() >= 5:
            d = d + timedelta(days=1)
            continue
        for seg_start_s, seg_end_s in segments:
            seg_start_local = pd.Timestamp(f"{d.isoformat()}T{seg_start_s}").tz_localize(tz)
            seg_end_local = pd.Timestamp(f"{d.isoformat()}T{seg_end_s}").tz_localize(tz)
            seg_start = seg_start_local.tz_convert("UTC")
            seg_end = seg_end_local.tz_convert("UTC")
            t = seg_start
            while t < seg_end:
                t_end = min(t + step, seg_end)
                if t >= start_utc and t_end <= end_utc:
                    t_local = t.tz_convert(tz)
                    t_end_local = t_end.tz_convert(tz)
                    rows.append(
                        {
                            "dt_start_utc": t,
                            "dt_end_utc": t_end,
                            "dt_start_local": t_local.strftime("%Y-%m-%dT%H:%M:%S"),
                            "dt_end_local": t_end_local.strftime("%Y-%m-%dT%H:%M:%S"),
                            "exchange_tz": spec.exchange_tz,
                            "interval": interval,
                            "trade_date": t_end_local.strftime("%Y-%m-%d"),
                        }
                    )
                t = t + step
        d = d + timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["ts_start_utc"] = _series_to_unix_seconds(out["dt_start_utc"])
    out["ts_end_utc"] = _series_to_unix_seconds(out["dt_end_utc"])
    out["dt_start_utc"] = out["dt_start_utc"].map(_iso_z)
    out["dt_end_utc"] = out["dt_end_utc"].map(_iso_z)
    out = out.sort_values(["ts_end_utc", "ts_start_utc"]).drop_duplicates(subset=["ts_start_utc", "ts_end_utc"])
    return out.reset_index(drop=True)


def expected_buckets_per_day(
    *,
    market: str,
    start_local_date: date,
    end_local_date: date,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Return expected bucket counts per local trade_date between [start_local_date, end_local_date].

    Output columns: trade_date, expected_buckets
    """
    spec = BucketSpec(market=market, interval=interval)
    tz = spec.exchange_tz
    # Convert local date range to UTC bounds generously, then bucketize.
    start_utc = pd.Timestamp(datetime.combine(start_local_date, datetime.min.time())).tz_localize(tz).tz_convert("UTC")
    end_utc = (
        pd.Timestamp(datetime.combine(end_local_date + timedelta(days=1), datetime.min.time()))
        .tz_localize(tz)
        .tz_convert("UTC")
    )
    buckets = generate_intraday_buckets(market=market, start_utc=start_utc, end_utc=end_utc, interval=interval)
    if buckets.empty:
        return pd.DataFrame(columns=["trade_date", "expected_buckets"])
    g = buckets.groupby("trade_date", as_index=False).size().rename(columns={"size": "expected_buckets"})
    return g


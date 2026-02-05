"""
Industrial cleaning pipeline (Silver v2 -> Gold clean).

This module takes calendar-aware canonical bars (Silver v2) and produces a grid-aligned
Gold table with:
- missing placeholders + flags
- adjusted prices (via adj_factor when available)
- anomaly flags and isolation policy
- feature/label masks for training
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from trading_calendar import BucketSpec, generate_intraday_buckets


DEFAULT_LABEL_HORIZONS: tuple[int, ...] = (1, 4, 7)


def _local_date_range_from_dt_end_local(dt_end_local: pd.Series) -> tuple[date, date]:
    d = pd.to_datetime(dt_end_local, errors="coerce")
    if d.isna().all():
        raise ValueError("dt_end_local is all NaT; cannot infer date range")
    start = d.min().date()
    end = d.max().date()
    return start, end


def grid_align_v2_to_calendar(
    v2: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Build a full calendar grid (expected buckets) and left-join observed v2 bars onto it.

    Output is the base of Gold table:
    - one row per expected bucket
    - missing placeholders (is_missing=1) with missing_reason
    - is_no_trade flag (volume==0 and flat price)
    - suspected_halt heuristic for gaps inside a session day
    """
    if v2 is None or v2.empty:
        return pd.DataFrame()

    use = v2.copy()
    # Only keep in-calendar bars for Gold grid (off-calendar stays in Silver for audit).
    if "flag_off_calendar" in use.columns:
        use = use[use["flag_off_calendar"].fillna(0).astype(int) == 0].copy()
    if use.empty:
        return pd.DataFrame()

    # Determine local trade date range and generate expected buckets.
    start_d, end_d = _local_date_range_from_dt_end_local(use["dt_end_local"])
    spec = BucketSpec(market=market, interval=interval)
    tz = spec.exchange_tz

    start_utc = pd.Timestamp(datetime.combine(start_d, datetime.min.time())).tz_localize(tz).tz_convert("UTC")
    end_utc = pd.Timestamp(datetime.combine(end_d + timedelta(days=1), datetime.min.time())).tz_localize(tz).tz_convert("UTC")

    buckets = generate_intraday_buckets(market=market, start_utc=start_utc, end_utc=end_utc, interval=interval)
    if buckets.empty:
        return pd.DataFrame()
    buckets = buckets.copy()
    buckets["symbol"] = symbol

    # Join by bucket end timestamp (primary key in our schema).
    merged = buckets.merge(use, on=["ts_end_utc"], how="left", suffixes=("_grid", ""))
    # Keep grid timestamps authoritative.
    merged["ts_start_utc"] = merged["ts_start_utc_grid"].fillna(merged.get("ts_start_utc"))
    merged["dt_start_utc"] = merged["dt_start_utc_grid"].fillna(merged.get("dt_start_utc"))
    merged["dt_end_utc"] = merged["dt_end_utc_grid"].fillna(merged.get("dt_end_utc"))
    merged["dt_start_local"] = merged["dt_start_local_grid"].fillna(merged.get("dt_start_local"))
    merged["dt_end_local"] = merged["dt_end_local_grid"].fillna(merged.get("dt_end_local"))
    merged["exchange_tz"] = merged["exchange_tz_grid"].fillna(merged.get("exchange_tz"))
    merged["interval"] = merged["interval_grid"].fillna(merged.get("interval"))
    merged["trade_date"] = merged["trade_date"].fillna(
        pd.to_datetime(merged["dt_end_local"], errors="coerce").dt.strftime("%Y-%m-%d")
    )

    # Indicator: missing if there was no observed row joined.
    merged["is_missing"] = merged["source"].isna().astype(int)
    merged["missing_reason"] = np.where(merged["is_missing"] == 1, "vendor_gap", None)

    # No-trade heuristic (bar exists but no volume and flat prices)
    price_cols = ["open", "high", "low", "close"]
    for c in price_cols + ["volume", "amount"]:
        if c not in merged.columns:
            merged[c] = np.nan
    flat_price = (
        (merged["open"] == merged["close"])
        & (merged["open"] == merged["high"])
        & (merged["open"] == merged["low"])
    )
    merged["is_no_trade"] = ((merged["is_missing"] == 0) & flat_price & (merged["volume"].fillna(-1) == 0)).astype(int)

    # Suspected halt heuristic: missing buckets inside a day (flanked by present bars)
    merged = merged.sort_values(["trade_date", "ts_end_utc"]).reset_index(drop=True)
    is_missing = merged["is_missing"].astype(bool)
    present = (~is_missing)

    def _prev_has_present(s: pd.Series) -> pd.Series:
        return s.shift(1).fillna(False).cummax()

    def _next_has_present(s: pd.Series) -> pd.Series:
        return s.iloc[::-1].shift(1).fillna(False).cummax().iloc[::-1]

    prev_present = present.groupby(merged["trade_date"], group_keys=False).apply(_prev_has_present)
    next_present = present.groupby(merged["trade_date"], group_keys=False).apply(_next_has_present)
    merged["suspected_halt"] = (is_missing & prev_present & next_present).astype(int)

    # Carry QC flags from v2 (default 0 for missing placeholders).
    qc_cols = [
        "flag_off_calendar",
        "flag_any_null_price",
        "flag_nonpositive_price",
        "flag_ohlc_inconsistent",
        "flag_volume_negative",
    ]
    for c in qc_cols:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0).astype(int)

    # Drop duplicated grid columns.
    drop_cols = [c for c in merged.columns if c.endswith("_grid")]
    merged = merged.drop(columns=drop_cols)
    return merged.reset_index(drop=True)


def apply_daily_adj_factors(
    base: pd.DataFrame,
    *,
    adj_factors: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Add adj_factor + adjusted OHLC columns to a grid-aligned DataFrame.

    adj_factors columns (from DB or fetch): trade_date, adj_factor
    """
    use = base.copy()
    if adj_factors is None or adj_factors.empty:
        use["adj_factor"] = 1.0
    else:
        fac = adj_factors.copy()
        if "trade_date" not in fac.columns or "adj_factor" not in fac.columns:
            use["adj_factor"] = 1.0
        else:
            fac = fac[["trade_date", "adj_factor"]].dropna(subset=["trade_date", "adj_factor"])
            fac["trade_date"] = fac["trade_date"].astype(str)
            fac["adj_factor"] = pd.to_numeric(fac["adj_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fac = fac.dropna(subset=["adj_factor"])
            use = use.merge(fac, on="trade_date", how="left")
            use["adj_factor"] = use["adj_factor"].fillna(1.0)

    # Guardrails
    use["adj_factor"] = pd.to_numeric(use["adj_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    use.loc[use["adj_factor"] <= 0, "adj_factor"] = 1.0

    for c in ["open", "high", "low", "close"]:
        if c not in use.columns:
            use[c] = np.nan
        use[f"{c}_adj"] = pd.to_numeric(use[c], errors="coerce") * use["adj_factor"]
    return use


def flag_corp_action_days(df: pd.DataFrame, *, eps: float = 1e-6) -> pd.Series:
    """
    Heuristic: mark days where adj_factor changes materially vs previous day.
    """
    if "trade_date" not in df.columns or "adj_factor" not in df.columns:
        return pd.Series(0, index=df.index)
    d = df[["trade_date", "adj_factor"]].copy()
    d["trade_date"] = d["trade_date"].astype(str)
    d["adj_factor"] = pd.to_numeric(d["adj_factor"], errors="coerce").fillna(1.0)
    daily = d.groupby("trade_date", as_index=False)["adj_factor"].last().sort_values("trade_date")
    daily["prev"] = daily["adj_factor"].shift(1)
    ratio = (daily["adj_factor"] / daily["prev"]).replace([np.inf, -np.inf], np.nan)
    changed = ratio.notna() & (np.abs(np.log(ratio)) > eps)
    daily["flag_corp_action"] = changed.astype(int)
    return df["trade_date"].astype(str).map(dict(zip(daily["trade_date"], daily["flag_corp_action"]))).fillna(0).astype(int)


def detect_anomalies(
    df: pd.DataFrame,
    *,
    abs_logret_threshold: float = 0.20,
    robust_z_threshold: float = 10.0,
    min_abs_logret_for_z: float = 0.05,
    range_pct_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Add anomaly flags based on adjusted prices.

    Outputs:
      flag_jump_unexplained, flag_range_unusual
    """
    use = df.copy()
    use["flag_jump_unexplained"] = 0
    use["flag_range_unusual"] = 0
    if "is_missing" not in use.columns:
        use["is_missing"] = 0

    # Range-based flag on adjusted OHLC (when available)
    if all(c in use.columns for c in ["high_adj", "low_adj", "close_adj"]):
        close = pd.to_numeric(use["close_adj"], errors="coerce")
        span = pd.to_numeric(use["high_adj"], errors="coerce") - pd.to_numeric(use["low_adj"], errors="coerce")
        pct = (span / close).replace([np.inf, -np.inf], np.nan)
        use["flag_range_unusual"] = ((use["is_missing"] == 0) & (pct > range_pct_threshold)).fillna(False).astype(int)

    # Return-based flag (avoid overnight false positives by resetting at trade_date boundary)
    if "close_adj" in use.columns and "trade_date" in use.columns:
        close = pd.to_numeric(use["close_adj"], errors="coerce")
        ret = np.log(close) - np.log(close.shift(1))
        boundary = use["trade_date"].astype(str) != use["trade_date"].astype(str).shift(1)
        ret = ret.mask(boundary, np.nan)
        abs_ret = ret.abs()

        # robust z-score (rolling median/MAD)
        win = 50
        minp = 20
        med = ret.rolling(window=win, min_periods=minp).median()
        mad = (ret - med).abs().rolling(window=win, min_periods=minp).median()
        robust_z = (ret - med).abs() / (1.4826 * mad.replace(0, np.nan))

        jump_abs = abs_ret > abs_logret_threshold
        jump_z = (robust_z > robust_z_threshold) & (abs_ret > min_abs_logret_for_z)
        use["flag_jump_unexplained"] = (
            (use["is_missing"] == 0) & (jump_abs | jump_z)
        ).fillna(False).astype(int)

    return use


def apply_isolation_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation policy for Gold:
    - missing buckets -> keep flags, keep NaNs
    - QC failures / anomalies -> NULL-out raw + adjusted fields and set masks later
    """
    use = df.copy()
    # Define bad bars
    bad = (
        (use.get("is_missing", 0).fillna(0).astype(int) == 1)
        | (use.get("flag_any_null_price", 0).fillna(0).astype(int) == 1)
        | (use.get("flag_nonpositive_price", 0).fillna(0).astype(int) == 1)
        | (use.get("flag_ohlc_inconsistent", 0).fillna(0).astype(int) == 1)
        | (use.get("flag_volume_negative", 0).fillna(0).astype(int) == 1)
        | (use.get("flag_jump_unexplained", 0).fillna(0).astype(int) == 1)
    )
    use["_is_bad_bar"] = bad.astype(int)

    # NULL-out numerical bar fields for bad bars (keep dividend as event field)
    null_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
    ]
    for c in null_cols:
        if c in use.columns:
            use.loc[use["_is_bad_bar"] == 1, c] = np.nan
    return use


def compute_masks(df: pd.DataFrame, *, label_horizons: Sequence[int]) -> pd.DataFrame:
    """
    Add feature_mask and label_mask_{h} columns.
    label_mask_{h} is 1 only if the current bar and the next h bars are all feature_mask==1.
    """
    use = df.copy()
    # feature is usable when not missing and not isolated and has adjusted close
    usable = (
        (use.get("is_missing", 0).fillna(0).astype(int) == 0)
        & (use.get("_is_bad_bar", 0).fillna(0).astype(int) == 0)
        & pd.to_numeric(use.get("close_adj"), errors="coerce").notna()
    )
    use["feature_mask"] = usable.astype(int)

    for h in label_horizons:
        col = f"label_mask_{int(h)}"
        m = use["feature_mask"].copy().astype(int)
        for k in range(1, int(h) + 1):
            m = m & use["feature_mask"].shift(-k).fillna(0).astype(int)
        use[col] = m.astype(int)
    return use


def build_gold_clean_bars(
    v2: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    adj_factors: pd.DataFrame | None,
    label_horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS,
    abs_logret_threshold: float = 0.20,
    robust_z_threshold: float = 10.0,
    min_abs_logret_for_z: float = 0.05,
    range_pct_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    End-to-end: Silver v2 -> Gold clean.
    """
    base = grid_align_v2_to_calendar(v2, market=market, symbol=symbol, interval="1h")
    if base is None or base.empty:
        return pd.DataFrame()
    out = apply_daily_adj_factors(base, adj_factors=adj_factors)
    out["flag_corp_action"] = flag_corp_action_days(out)
    out = detect_anomalies(
        out,
        abs_logret_threshold=abs_logret_threshold,
        robust_z_threshold=robust_z_threshold,
        min_abs_logret_for_z=min_abs_logret_for_z,
        range_pct_threshold=range_pct_threshold,
    )
    out = apply_isolation_policy(out)
    out = compute_masks(out, label_horizons=label_horizons)
    return out.reset_index(drop=True)


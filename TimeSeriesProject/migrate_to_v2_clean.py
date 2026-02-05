"""
Migration: rebuild Silver v2 + Gold clean tables from existing raw `ohlcv_*` tables.

Why:
- v1 bars_hourly_* used a simplified +1h bar_end logic for US; v2 uses official calendars.
- Gold clean adds grid alignment, missing placeholders, anomaly flags, and masks.

Run:
  python3 migrate_to_v2_clean.py
  python3 migrate_to_v2_clean.py --market us
  python3 migrate_to_v2_clean.py --market cn

Options:
  --no-us-factor-fetch   Do not fetch US daily adj factors from yfinance (offline mode)
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Optional

import pandas as pd

import config
from cleaning import MarketTimeSpec, canonicalize_hourly_bars_v2
from clean_pipeline import build_gold_clean_bars
from corp_actions import fetch_us_daily_adj_factors
from db.storage import (
    load_adj_factors_df,
    save_adj_factors_df,
    save_cleaned_hourly_clean_df,
    save_cleaned_hourly_v2_df,
)


def list_ohlcv_tables(conn: sqlite3.Connection, market: Optional[str]):
    like = f"ohlcv_{market}_%" if market else "ohlcv_%"
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
        (like,),
    )
    return [r[0] for r in cur.fetchall()]


def infer_market_from_table(table: str) -> Optional[str]:
    # ohlcv_{market}_YYYYQq
    parts = table.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def build_spec(market: str) -> MarketTimeSpec:
    if market == config.MARKET_US:
        return MarketTimeSpec(exchange_tz=config.TZ_US, input_tz="UTC", timestamp_kind="bar_start")
    return MarketTimeSpec(exchange_tz=config.TZ_CN, input_tz=config.TZ_CN, timestamp_kind="bar_end")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "cn", "all"], default="all")
    parser.add_argument("--no-us-factor-fetch", action="store_true", help="Do not fetch US daily adj factors (offline)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print lightweight sanity checks (session close times, missing placeholders, anomaly isolation).",
    )
    args = parser.parse_args()

    market = None if args.market == "all" else args.market

    conn = sqlite3.connect(config.DB_PATH)
    try:
        tables = list_ohlcv_tables(conn, market)
        if not tables:
            print("No ohlcv_* tables found; nothing to migrate.")
            return

        for t in tables:
            m = infer_market_from_table(t)
            if market and m != market:
                continue
            if m not in ("us", "cn"):
                continue

            spec = build_spec(m)
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
            if df.empty:
                continue

            total_v2 = 0
            total_gold = 0
            for sym, g in df.groupby("symbol"):
                sym = str(sym)
                v2 = canonicalize_hourly_bars_v2(
                    g,
                    market=m,
                    symbol=sym,
                    source="migrate_ohlcv_v2",
                    spec=spec,
                    interval="1h",
                )
                if v2.empty:
                    continue
                total_v2 += save_cleaned_hourly_v2_df(conn, m, v2)

                # factors (US only by default; CN optional dual-fetch not supported in offline migration)
                adj_factors = None
                if m == config.MARKET_US and config.US_FETCH_DAILY_ADJ_FACTORS and not args.no_us_factor_fetch:
                    d = pd.to_datetime(v2["dt_end_local"], errors="coerce")
                    if not d.isna().all():
                        start_d = d.min().date()
                        end_d = d.max().date()
                        fac = fetch_us_daily_adj_factors(sym, start_d, end_d)
                        if fac is not None and not fac.empty:
                            save_adj_factors_df(conn, m, fac)
                        adj_factors = load_adj_factors_df(conn, m, sym, start_d.isoformat(), end_d.isoformat())

                gold = build_gold_clean_bars(
                    v2,
                    market=m,
                    symbol=sym,
                    adj_factors=adj_factors,
                    label_horizons=config.LABEL_HORIZONS_BARS,
                    abs_logret_threshold=config.ANOMALY_ABS_LOGRET_THRESHOLD,
                    robust_z_threshold=config.ANOMALY_ROBUST_Z_THRESHOLD,
                    min_abs_logret_for_z=config.ANOMALY_MIN_ABS_LOGRET_FOR_Z,
                    range_pct_threshold=config.ANOMALY_RANGE_PCT_THRESHOLD,
                )
                if gold is not None and not gold.empty:
                    total_gold += save_cleaned_hourly_clean_df(conn, m, gold)
                    if args.verify:
                        # 1) Session close sanity (US should not exceed 16:00 local in RTH buckets)
                        times = gold["dt_end_local"].astype(str).str[11:19].value_counts().head(5).to_dict()
                        max_t = gold["dt_end_local"].astype(str).str[11:19].max()
                        print(f"  [verify] {m} {sym}: top_end_times={times}, max_end_time={max_t}")
                        if m == config.MARKET_US and max_t > "16:00:00":
                            print(f"  [verify][WARN] {m} {sym}: max_end_time={max_t} > 16:00:00 (check calendar alignment)")

                        # 2) Missing placeholders
                        miss = int(pd.to_numeric(gold.get("is_missing"), errors="coerce").fillna(0).sum())
                        total = int(len(gold))
                        print(f"  [verify] {m} {sym}: missing_buckets={miss}/{total}")

                        # 3) Anomaly isolation: flagged jump bars should be NULLed (close_adj NaN)
                        jump = gold.get("flag_jump_unexplained")
                        if jump is not None:
                            jump_idx = (pd.to_numeric(jump, errors="coerce").fillna(0).astype(int) == 1)
                            if jump_idx.any():
                                still_has_price = pd.to_numeric(gold.loc[jump_idx, "close_adj"], errors="coerce").notna().sum()
                                print(
                                    f"  [verify] {m} {sym}: jump_flags={int(jump_idx.sum())}, "
                                    f"jump_with_nonnull_close_adj={int(still_has_price)}"
                                )

            print(f"{t}: v2_saved_rows={total_v2}, gold_saved_rows={total_gold}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


"""
One-time migration: convert legacy raw tables (ohlcv_*) into cleaned canonical tables (bars_hourly_*).

This is useful if you already fetched data before the qlib-style cleaning/storage upgrade.

Run:
  python migrate_ohlcv_to_bars.py
  python migrate_ohlcv_to_bars.py --market cn
  python migrate_ohlcv_to_bars.py --market us
"""

import argparse
import sqlite3
from typing import Optional

import pandas as pd

import config
from cleaning import MarketTimeSpec, canonicalize_hourly_bars
from db.storage import save_cleaned_hourly_df


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "cn", "all"], default="all")
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

            if m == "us":
                spec = MarketTimeSpec(exchange_tz=config.TZ_US, input_tz="UTC", timestamp_kind="bar_start")
            else:
                spec = MarketTimeSpec(exchange_tz=config.TZ_CN, input_tz=config.TZ_CN, timestamp_kind="bar_end")

            df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
            if df.empty:
                continue
            # per symbol canonicalize + save
            total = 0
            for sym, g in df.groupby("symbol"):
                cleaned = canonicalize_hourly_bars(
                    g,
                    market=m,
                    symbol=str(sym),
                    source="migrate_ohlcv",
                    spec=spec,
                    interval="1h",
                )
                total += save_cleaned_hourly_df(conn, m, cleaned)
            print(f"{t}: migrated saved_rows={total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


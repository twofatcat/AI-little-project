"""
Quality report for cleaned hourly bars tables (bars_hourly_*).

Run:
  python quality_report.py
  python quality_report.py --market cn
  python quality_report.py --market us
"""

import argparse
import sqlite3
from datetime import datetime
from typing import Optional

import config
from trading_calendar import expected_buckets_per_day


def list_bars_tables(conn: sqlite3.Connection, market: Optional[str]):
    # Include v1/v2/clean: bars_hourly_{market}_*, bars_hourly_v2_{market}_*, bars_hourly_clean_{market}_*
    if market:
        like = f"bars_hourly%_{market}_%"
    else:
        like = "bars_hourly%"
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
        (like,),
    )
    return [r[0] for r in cur.fetchall()]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [row[1] for row in cur.fetchall()]


def _parse_local_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return str(s)[:10]


def report_table(conn: sqlite3.Connection, table: str, market: str):
    n = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
    sym_n = conn.execute(f'SELECT count(DISTINCT symbol) FROM "{table}"').fetchone()[0]
    mn, mx = conn.execute(f'SELECT min(dt_end_local), max(dt_end_local) FROM "{table}"').fetchone()
    print(f"\n{table}: rows={n}, symbols={sym_n}, dt_end_local=[{mn} .. {mx}]")
    start_d = _parse_local_date(mn)
    end_d = _parse_local_date(mx)
    if not start_d or not end_d:
        return
    cols = set(_table_columns(conn, table))

    # Expected buckets per day from official exchange calendar
    exp_df = expected_buckets_per_day(
        market=market,
        start_local_date=datetime.fromisoformat(start_d).date(),
        end_local_date=datetime.fromisoformat(end_d).date(),
        interval="1h",
    )
    exp_map = {r["trade_date"]: int(r["expected_buckets"]) for _, r in exp_df.iterrows()} if not exp_df.empty else {}

    # Symbol total bars
    top = conn.execute(
        f'SELECT symbol, count(*) c FROM "{table}" GROUP BY symbol ORDER BY c DESC LIMIT 5'
    ).fetchall()
    bottom = conn.execute(
        f'SELECT symbol, count(*) c FROM "{table}" GROUP BY symbol ORDER BY c ASC LIMIT 5'
    ).fetchall()
    print("  top5 symbols:", top)
    print("  bottom5 symbols:", bottom)

    # Day-level counts per symbol
    day_rows = conn.execute(
        f"""
        SELECT symbol, substr(dt_end_local, 1, 10) AS day, count(*) AS bars
        FROM "{table}"
        GROUP BY symbol, day
        ORDER BY symbol, day
        """
    ).fetchall()

    odd_days = []
    for sym, day, bars in day_rows:
        exp = exp_map.get(day)
        if exp is None:
            continue
        if int(bars) != int(exp):
            odd_days.append((sym, day, int(bars), int(exp)))

    if not odd_days:
        print("  day-bar-count check: OK (all days match calendar expected buckets)")
    else:
        per_symbol = {}
        for sym, _, _, _ in odd_days:
            per_symbol[sym] = per_symbol.get(sym, 0) + 1
        worst = sorted(per_symbol.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  day-bar-count check: {len(odd_days)} odd day records (calendar expected)")
        print("  worst symbols (odd_days_count):", worst)
        print("  sample odd days (sym, day, bars, expected):", odd_days[:10])

    # For Gold clean tables, additionally summarize missing placeholders
    if "is_missing" in cols:
        miss_rows = conn.execute(
            f"""
            SELECT symbol, substr(dt_end_local, 1, 10) AS day,
                   sum(CASE WHEN is_missing=1 THEN 1 ELSE 0 END) AS missing,
                   count(*) AS total
            FROM "{table}"
            GROUP BY symbol, day
            ORDER BY missing DESC, total DESC
            LIMIT 10
            """
        ).fetchall()
        if miss_rows:
            print("  top missing days (sym, day, missing, total):", miss_rows)


def infer_market_from_table(table: str) -> Optional[str]:
    # bars_hourly_{market}_YYYYQq
    # bars_hourly_v2_{market}_YYYYQq
    # bars_hourly_clean_{market}_YYYYQq
    parts = table.split("_")
    if len(parts) < 3:
        return None
    if parts[0] != "bars" or parts[1] != "hourly":
        return None
    if parts[2] in ("us", "cn"):
        return parts[2]
    if parts[2] in ("v2", "clean") and len(parts) >= 4:
        return parts[3]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "cn", "all"], default="all")
    args = parser.parse_args()

    conn = sqlite3.connect(config.DB_PATH)
    try:
        market = None if args.market == "all" else args.market
        tables = list_bars_tables(conn, market)
        if not tables:
            print("No bars_hourly_* tables found yet. Run `python main.py` first (it will create cleaned tables).")
            return
        for t in tables:
            m = infer_market_from_table(t) or (market or "")
            report_table(conn, t, m)
    finally:
        conn.close()


if __name__ == "__main__":
    main()


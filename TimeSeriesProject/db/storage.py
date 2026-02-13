"""
Storage: get max(datetime) per (market, symbol); ensure table; save by quarter (upsert).
"""
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from .schema import (
    adj_factors_table_name,
    bars_table_name_for_quarter,
    bars_v2_table_name_for_quarter,
    bars_clean_table_name_for_quarter,
    create_adj_factors_table_sql,
    create_bars_table_sql,
    create_bars_v2_table_sql,
    create_bars_clean_table_sql,
    create_table_sql,
    quarter_from_datetime,
    table_name_for_quarter,
)

# Standard columns we expect on the DataFrame (after adapter normalizes)
OHLCV_COLS = [
    "datetime", "open", "high", "low", "close",
    "volume", "adj_close", "dividend",
]


def get_last_datetime(conn: sqlite3.Connection, market: str, symbol: str) -> Optional[datetime]:
    """
    For (market, symbol), query all ohlcv_{market}_* tables and return max(datetime).
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (f"ohlcv_{market}_%",),
    )
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        return None
    last_ts = None
    for t in tables:
        cur = conn.execute(
            f'SELECT max(datetime) FROM "{t}" WHERE symbol = ?',
            (symbol,),
        )
        row = cur.fetchone()
        if row and row[0]:
            try:
                dt = pd.Timestamp(row[0]).to_pydatetime()
                if last_ts is None or dt > last_ts:
                    last_ts = dt
            except Exception:
                pass
    return last_ts


def ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Create table if it does not exist."""
    conn.execute(create_table_sql(table_name))
    conn.commit()


def ensure_bars_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Create cleaned bars table if it does not exist."""
    conn.execute(create_bars_table_sql(table_name))
    conn.commit()


def ensure_bars_v2_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Create Silver v2 bars table if it does not exist."""
    conn.execute(create_bars_v2_table_sql(table_name))
    conn.commit()


def ensure_bars_clean_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Create Gold clean bars table if it does not exist."""
    conn.execute(create_bars_clean_table_sql(table_name))
    conn.commit()


def ensure_adj_factors_table(conn: sqlite3.Connection, market: str) -> str:
    """Create daily adj_factors table for market if needed. Returns table name."""
    tname = adj_factors_table_name(market)
    conn.execute(create_adj_factors_table_sql(tname))
    conn.commit()
    return tname


def get_last_ts_end_utc(conn: sqlite3.Connection, market: str, symbol: str) -> Optional[int]:
    """
    For (market, symbol), query all bars_hourly_{market}_* tables and return max(ts_end_utc).
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (f"bars_hourly_{market}_%",),
    )
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        return None
    last_ts = None
    for t in tables:
        row = conn.execute(
            f'SELECT max(ts_end_utc) FROM "{t}" WHERE symbol = ?',
            (symbol,),
        ).fetchone()
        if row and row[0] is not None:
            ts = int(row[0])
            if last_ts is None or ts > last_ts:
                last_ts = ts
    return last_ts


def save_quarter_df(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    symbol: str,
) -> int:
    """
    Upsert rows into the quarter table. df must have datetime + OHLCV columns.
    datetime stored as ISO string. Returns number of rows written.
    """
    if df is None or df.empty:
        return 0
    ensure_table(conn, table_name)
    # Normalize: ensure we have required cols; map datetime to string
    use = df.copy()
    if "adj_close" not in use.columns and "close" in use.columns:
        use["adj_close"] = use["close"]
    if "dividend" not in use.columns:
        use["dividend"] = None
    use["symbol"] = symbol
    if "datetime" in use.columns:
        use["datetime"] = pd.to_datetime(use["datetime"]).astype(str).str.replace(" ", "T").str[:19]
    cols = ["symbol", "datetime", "open", "high", "low", "close", "volume", "adj_close", "dividend"]
    existing = [c for c in cols if c in use.columns]
    use = use[existing].dropna(subset=["datetime"], how="all")
    if use.empty:
        return 0
    cur = conn.cursor()
    n = 0
    for _, row in use.iterrows():
        cur.execute(
            f'INSERT OR REPLACE INTO "{table_name}" (symbol, datetime, open, high, low, close, volume, adj_close, dividend) '
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol,
                str(row["datetime"]),
                float(row.get("open")) if pd.notna(row.get("open")) else None,
                float(row.get("high")) if pd.notna(row.get("high")) else None,
                float(row.get("low")) if pd.notna(row.get("low")) else None,
                float(row.get("close")) if pd.notna(row.get("close")) else None,
                float(row.get("volume")) if pd.notna(row.get("volume")) else None,
                float(row.get("adj_close")) if pd.notna(row.get("adj_close")) else None,
                float(row.get("dividend")) if pd.notna(row.get("dividend")) else None,
            ),
        )
        n += 1
    conn.commit()
    return n


def save_fetched_df(
    conn: sqlite3.Connection,
    market: str,
    symbol: str,
    df: pd.DataFrame,
) -> int:
    """
    Split df by quarter and save into ohlcv_{market}_{year}Q{q}. Returns total rows written.
    """
    if df is None or df.empty:
        return 0
    total = 0
    use = df.copy()
    use["datetime"] = pd.to_datetime(use["datetime"])
    use["_year"] = use["datetime"].apply(lambda x: quarter_from_datetime(x)[0])
    use["_q"] = use["datetime"].apply(lambda x: quarter_from_datetime(x)[1])
    for (y, q), g in use.groupby(["_year", "_q"]):
        tname = table_name_for_quarter(market, int(y), int(q))
        drop = g.drop(columns=["_year", "_q"])
        total += save_quarter_df(conn, tname, drop, symbol)
    return total


def save_bars_df(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> int:
    """
    Upsert canonical bars into a cleaned table. Expects columns:
      symbol, ts_end_utc, dt_end_utc, dt_end_local, exchange_tz, interval,
      open, high, low, close, volume, amount, adj_close, dividend, source, updated_at_utc
    """
    if df is None or df.empty:
        return 0
    ensure_bars_table(conn, table_name)
    cols = [
        "symbol",
        "ts_end_utc",
        "dt_end_utc",
        "dt_end_local",
        "exchange_tz",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_close",
        "dividend",
        "source",
        "updated_at_utc",
    ]
    use = df.copy()
    # Ensure all cols exist
    for c in cols:
        if c not in use.columns:
            use[c] = None
    use = use[cols].dropna(subset=["symbol", "ts_end_utc"])
    if use.empty:
        return 0
    # Convert ts_end_utc to int
    use["ts_end_utc"] = use["ts_end_utc"].astype("int64")
    rows = list(use.itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(
        f'INSERT OR REPLACE INTO "{table_name}" ('
        "symbol, ts_end_utc, dt_end_utc, dt_end_local, exchange_tz, interval, "
        "open, high, low, close, volume, amount, adj_close, dividend, source, updated_at_utc"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def save_bars_v2_df(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> int:
    """
    Upsert Silver v2 canonical bars into a cleaned table.
    Expects columns:
      symbol, ts_start_utc, ts_end_utc, dt_start_utc, dt_end_utc, dt_start_local, dt_end_local,
      exchange_tz, interval, open, high, low, close, volume, amount, adj_close, dividend,
      source, updated_at_utc, flag_off_calendar, flag_any_null_price, flag_nonpositive_price,
      flag_ohlc_inconsistent, flag_volume_negative
    """
    if df is None or df.empty:
        return 0
    ensure_bars_v2_table(conn, table_name)
    cols = [
        "symbol",
        "ts_start_utc",
        "ts_end_utc",
        "dt_start_utc",
        "dt_end_utc",
        "dt_start_local",
        "dt_end_local",
        "exchange_tz",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_close",
        "dividend",
        "source",
        "updated_at_utc",
        "flag_off_calendar",
        "flag_any_null_price",
        "flag_nonpositive_price",
        "flag_ohlc_inconsistent",
        "flag_volume_negative",
    ]
    use = df.copy()
    for c in cols:
        if c not in use.columns:
            use[c] = None
    use = use[cols].dropna(subset=["symbol", "ts_end_utc"])
    if use.empty:
        return 0
    use["ts_start_utc"] = use["ts_start_utc"].astype("int64")
    use["ts_end_utc"] = use["ts_end_utc"].astype("int64")
    rows = list(use.itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(
        f'INSERT OR REPLACE INTO "{table_name}" ('
        "symbol, ts_start_utc, ts_end_utc, dt_start_utc, dt_end_utc, dt_start_local, dt_end_local, "
        "exchange_tz, interval, open, high, low, close, volume, amount, adj_close, dividend, "
        "source, updated_at_utc, flag_off_calendar, flag_any_null_price, flag_nonpositive_price, "
        "flag_ohlc_inconsistent, flag_volume_negative"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def save_bars_clean_df(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> int:
    """
    Upsert Gold clean bars (grid-aligned) into a cleaned table.
    """
    if df is None or df.empty:
        return 0
    ensure_bars_clean_table(conn, table_name)
    cols = [
        "symbol",
        "ts_start_utc",
        "ts_end_utc",
        "dt_start_utc",
        "dt_end_utc",
        "dt_start_local",
        "dt_end_local",
        "trade_date",
        "exchange_tz",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "dividend",
        "source",
        "updated_at_utc",
        "is_missing",
        "missing_reason",
        "suspected_halt",
        "is_no_trade",
        "flag_off_calendar",
        "flag_any_null_price",
        "flag_nonpositive_price",
        "flag_ohlc_inconsistent",
        "flag_volume_negative",
        "flag_jump_unexplained",
        "flag_range_unusual",
        "flag_corp_action",
        "feature_mask",
        "label_mask_1",
        "label_mask_4",
        "label_mask_7",
    ]
    use = df.copy()
    for c in cols:
        if c not in use.columns:
            use[c] = None
    use = use[cols].dropna(subset=["symbol", "ts_end_utc"])
    if use.empty:
        return 0
    use["ts_start_utc"] = use["ts_start_utc"].astype("int64")
    use["ts_end_utc"] = use["ts_end_utc"].astype("int64")
    rows = list(use.itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(
        f'INSERT OR REPLACE INTO "{table_name}" ('
        "symbol, ts_start_utc, ts_end_utc, dt_start_utc, dt_end_utc, dt_start_local, dt_end_local, trade_date, "
        "exchange_tz, interval, open, high, low, close, volume, amount, adj_factor, open_adj, high_adj, low_adj, close_adj, "
        "dividend, source, updated_at_utc, is_missing, missing_reason, suspected_halt, is_no_trade, "
        "flag_off_calendar, flag_any_null_price, flag_nonpositive_price, flag_ohlc_inconsistent, flag_volume_negative, "
        "flag_jump_unexplained, flag_range_unusual, flag_corp_action, feature_mask, label_mask_1, label_mask_4, label_mask_7"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def save_cleaned_hourly_df(conn: sqlite3.Connection, market: str, df: pd.DataFrame) -> int:
    """
    Split canonical bars df by quarter (based on dt_end_local) and save into bars_hourly_{market}_{year}Q{q}.
    """
    if df is None or df.empty:
        return 0
    use = df.copy()
    # quarter based on local end time (consistent with how users think about quarters per market)
    use["_dt_local"] = pd.to_datetime(use["dt_end_local"])
    use["_year"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[0])
    use["_q"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[1])
    total = 0
    for (y, q), g in use.groupby(["_year", "_q"]):
        tname = bars_table_name_for_quarter(market, int(y), int(q))
        drop = g.drop(columns=["_dt_local", "_year", "_q"])
        total += save_bars_df(conn, tname, drop)
    return total


def save_cleaned_hourly_v2_df(conn: sqlite3.Connection, market: str, df: pd.DataFrame) -> int:
    """
    Split Silver v2 df by quarter (based on dt_end_local) and save into bars_hourly_v2_{market}_{year}Q{q}.
    """
    if df is None or df.empty:
        return 0
    use = df.copy()
    use["_dt_local"] = pd.to_datetime(use["dt_end_local"])
    use["_year"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[0])
    use["_q"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[1])
    total = 0
    for (y, q), g in use.groupby(["_year", "_q"]):
        tname = bars_v2_table_name_for_quarter(market, int(y), int(q))
        drop = g.drop(columns=["_dt_local", "_year", "_q"])
        total += save_bars_v2_df(conn, tname, drop)
    return total


def save_cleaned_hourly_clean_df(conn: sqlite3.Connection, market: str, df: pd.DataFrame) -> int:
    """
    Split Gold clean df by quarter (based on dt_end_local) and save into bars_hourly_clean_{market}_{year}Q{q}.
    """
    if df is None or df.empty:
        return 0
    use = df.copy()
    use["_dt_local"] = pd.to_datetime(use["dt_end_local"])
    use["_year"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[0])
    use["_q"] = use["_dt_local"].apply(lambda x: quarter_from_datetime(x)[1])
    total = 0
    for (y, q), g in use.groupby(["_year", "_q"]):
        tname = bars_clean_table_name_for_quarter(market, int(y), int(q))
        drop = g.drop(columns=["_dt_local", "_year", "_q"])
        total += save_bars_clean_df(conn, tname, drop)
    return total


def save_adj_factors_df(conn: sqlite3.Connection, market: str, df: pd.DataFrame) -> int:
    """
    Upsert daily adjustment factors for a market.
    df columns: symbol, trade_date (YYYY-MM-DD), adj_factor, source, updated_at_utc
    """
    if df is None or df.empty:
        return 0
    tname = ensure_adj_factors_table(conn, market)
    use = df.copy()
    cols = ["symbol", "trade_date", "adj_factor", "source", "updated_at_utc"]
    for c in cols:
        if c not in use.columns:
            use[c] = None
    use = use[cols].dropna(subset=["symbol", "trade_date", "adj_factor"])
    if use.empty:
        return 0
    rows = list(use.itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(
        f'INSERT OR REPLACE INTO "{tname}" (symbol, trade_date, adj_factor, source, updated_at_utc) VALUES (?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    return len(rows)


def load_adj_factors_df(
    conn: sqlite3.Connection,
    market: str,
    symbol: str,
    start_trade_date: str,
    end_trade_date: str,
) -> pd.DataFrame:
    """
    Load daily adjustment factors from DB for [start_trade_date, end_trade_date] (inclusive).
    """
    tname = ensure_adj_factors_table(conn, market)
    q = (
        f'SELECT symbol, trade_date, adj_factor, source, updated_at_utc FROM "{tname}" '
        "WHERE symbol = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date"
    )
    return pd.read_sql_query(q, conn, params=(symbol, start_trade_date, end_trade_date))

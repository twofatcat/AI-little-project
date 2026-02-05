"""
查看已抓取的股票数据：表列表、行数、抽样预览。
用法:
  python view_data.py              # 总览 + 各表行数 + 每市场抽样几条
  python view_data.py --symbol AAPL  # 只看某只股票
  python view_data.py --table ohlcv_us_2025Q1  # 只看某张季度表
  python view_data.py --head 20    # 每表预览行数，默认 5
"""
import argparse
import sqlite3
import sys

import pandas as pd

import config


def _order_col_for_table(conn: sqlite3.Connection, table: str) -> str:
    """清洗表 bars_hourly_* 用 ts_end_utc，原始表 ohlcv_* 用 datetime。"""
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    cols = [row[1] for row in cur.fetchall()]
    if "ts_end_utc" in cols:
        return "ts_end_utc"
    return "datetime"


def list_tables(conn: sqlite3.Connection):
    """列出所有 ohlcv_* 与 bars_hourly_* 表及行数。"""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE 'ohlcv_%' OR name LIKE 'bars_hourly_%') ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        print("当前数据库中没有 ohlcv_* 或 bars_hourly_* 表（尚未抓取或 DB 路径不对）")
        return []
    print("表名\t行数")
    print("-" * 50)
    total = 0
    for t in tables:
        n = conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        total += n
        print(f"{t}\t{n}")
    print("-" * 50)
    print(f"合计\t{total}")
    return tables


def sample_table(conn: sqlite3.Connection, table: str, limit: int = 5) -> pd.DataFrame:
    """从指定表取前 limit 条。bars_hourly_* 按 ts_end_utc，ohlcv_* 按 datetime。"""
    order_col = _order_col_for_table(conn, table)
    return pd.read_sql_query(f'SELECT * FROM "{table}" ORDER BY {order_col} LIMIT {limit}', conn)


def sample_symbol(conn: sqlite3.Connection, symbol: str, market: str = None, limit: int = 20) -> pd.DataFrame:
    """从某市场的 ohlcv_* 表中取该 symbol 的数据（按时间排序取 limit 条）。"""
    if market:
        pattern = f"ohlcv_{market}_%"
    else:
        pattern = "ohlcv_%"
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
        (pattern,),
    )
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        return pd.DataFrame()
    dfs = []
    for t in tables:
        df = pd.read_sql_query(
            f'SELECT * FROM "{t}" WHERE symbol = ? ORDER BY datetime',
            conn,
            params=(symbol,),
        )
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["symbol", "datetime"])
    out = out.sort_values("datetime").tail(limit)
    return out


def main():
    parser = argparse.ArgumentParser(description="查看 TimeSeriesProject 抓取的股票数据")
    parser.add_argument("--symbol", "-s", type=str, help="只看指定代码，如 AAPL 或 300750")
    parser.add_argument("--table", "-t", type=str, help="只看指定表，如 ohlcv_us_2025Q1")
    parser.add_argument("--head", "-n", type=int, default=5, help="每表/抽样显示行数，默认 5")
    parser.add_argument("--market", "-m", type=str, choices=["us", "cn"], help="与 --symbol 合用，限定市场")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(config.DB_PATH)
    except Exception as e:
        print(f"无法打开数据库 {config.DB_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.table:
            tables = [args.table]
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                (args.table,),
            )
            if not cur.fetchone():
                print(f"表不存在: {args.table}")
                return
            n = conn.execute(f'SELECT count(*) FROM "{args.table}"').fetchone()[0]
            print(f"表 {args.table} 共 {n} 行，预览前 {args.head} 条:\n")
            df = sample_table(conn, args.table, limit=args.head)
            print(df.to_string())
            return

        if args.symbol:
            df = sample_symbol(conn, args.symbol, market=args.market, limit=args.head)
            if df.empty:
                print(f"未找到 symbol={args.symbol} 的数据（可加 --market us 或 --market cn）")
            else:
                print(f"symbol={args.symbol} 最近 {len(df)} 条:\n")
                print(df.to_string())
            return

        # 默认：总览 + 每市场抽样
        tables = list_tables(conn)
        if not tables:
            return
        us_tables = [t for t in tables if t.startswith("ohlcv_us_")]
        cn_tables = [t for t in tables if t.startswith("ohlcv_cn_")]
        print()
        if us_tables:
            # 取最新的 US 表，预览一条
            t = sorted(us_tables)[-1]
            df = sample_table(conn, t, limit=args.head)
            print(f"[US] 表 {t} 预览前 {args.head} 条:")
            print(df.to_string())
            print()
        if cn_tables:
            t = sorted(cn_tables)[-1]
            df = sample_table(conn, t, limit=args.head)
            print(f"[CN] 表 {t} 预览前 {args.head} 条:")
            print(df.to_string())
    finally:
        conn.close()


if __name__ == "__main__":
    main()

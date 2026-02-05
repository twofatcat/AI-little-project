"""
Entry: read progress -> fetch by market/symbol (incremental) -> save by quarter.
Run from project root: python main.py
"""
import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from zoneinfo import ZoneInfo

import pandas as pd

import config
from cleaning import MarketTimeSpec, canonicalize_hourly_bars, canonicalize_hourly_bars_v2
from clean_pipeline import build_gold_clean_bars
from corp_actions import (
    compute_cn_daily_adj_factors_from_dual_intraday,
    fetch_us_daily_adj_factors,
)
from db.storage import (
    get_last_datetime,
    get_last_ts_end_utc,
    load_adj_factors_df,
    save_adj_factors_df,
    save_cleaned_hourly_clean_df,
    save_cleaned_hourly_df,
    save_cleaned_hourly_v2_df,
    save_fetched_df,
)
from symbols import get_symbols
from sources.us_yfinance import USYFinanceSource
from sources.cn_akshare import CNAKShareSource, CNAKShareSinaFallbackSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _free_start_dt(market: str) -> datetime:
    """Earliest allowed start for free hourly data."""
    now = datetime.now(timezone.utc)
    if market == config.MARKET_US:
        return (now - timedelta(days=config.FREE_HOURLY_DAYS_US)).replace(tzinfo=timezone.utc)
    return (now - timedelta(days=config.FREE_HOURLY_DAYS_CN)).replace(tzinfo=timezone.utc)


def _sleep_backoff(attempt: int) -> None:
    sleep_s = config.FETCH_RETRY_BACKOFF_SECONDS * (2**attempt)
    time.sleep(sleep_s)


def _fetch_with_retries(source, market: str, symbol: str, start_dt, end_dt):
    for attempt in range(config.FETCH_RETRY_TIMES):
        try:
            df = source.fetch_hourly(symbol, start_dt, end_dt)
        except Exception:
            df = None
        if df is not None and not df.empty:
            return df
        logger.warning("%s %s: empty/no data (attempt %d/%d)", market, symbol, attempt + 1, config.FETCH_RETRY_TIMES)
        _sleep_backoff(attempt)
    return None


def run_market(conn: sqlite3.Connection, market: str, theme: str = "ai_robotics") -> None:
    symbols = get_symbols(market, theme)
    if market == config.MARKET_US:
        source = USYFinanceSource()
        delay = config.FETCH_DELAY_SECONDS_US
        spec = MarketTimeSpec(exchange_tz=config.TZ_US, input_tz="UTC", timestamp_kind="bar_start")
    else:
        source = CNAKShareSource(adjust="qfq")
        delay = config.FETCH_DELAY_SECONDS_CN
        spec = MarketTimeSpec(exchange_tz=config.TZ_CN, input_tz=config.TZ_CN, timestamp_kind="bar_end")
        fallback = CNAKShareSinaFallbackSource(adjust="qfq") if config.CN_ENABLE_FALLBACK_SINA else None
        source_unadj = CNAKShareSource(adjust="") if config.CN_COMPUTE_ADJ_FACTORS_DUAL_FETCH else None

    end_dt_utc = datetime.now(timezone.utc)
    free_start_utc = _free_start_dt(market)
    for i, symbol in enumerate(symbols):
        try:
            last_ts_end = get_last_ts_end_utc(conn, market, symbol)
            if last_ts_end is not None:
                last_end_dt_utc = datetime.fromtimestamp(int(last_ts_end), tz=timezone.utc)
                start_dt_utc = max(free_start_utc, last_end_dt_utc - config.RESUME_OVERLAP)
            else:
                start_dt_utc = free_start_utc

            if start_dt_utc >= end_dt_utc:
                logger.info("%s %s: no new range (start >= end), skip", market, symbol)
                if delay > 0:
                    time.sleep(delay)
                continue

            # Pass datetimes to adapters in their preferred frame
            if market == config.MARKET_CN:
                tz = ZoneInfo(config.TZ_CN)
                start_arg = start_dt_utc.astimezone(tz).replace(tzinfo=None)
                end_arg = end_dt_utc.astimezone(tz).replace(tzinfo=None)
            else:
                start_arg = start_dt_utc.replace(tzinfo=None)
                end_arg = end_dt_utc.replace(tzinfo=None)

            used_source_name = getattr(source, "source_name", "unknown")
            df = _fetch_with_retries(source, market, symbol, start_arg, end_arg)
            if (df is None or df.empty) and market == config.MARKET_CN and config.CN_ENABLE_FALLBACK_SINA:
                df = _fetch_with_retries(fallback, market, symbol, start_arg, end_arg)  # type: ignore[arg-type]
                if df is not None and not df.empty:
                    used_source_name = getattr(fallback, "source_name", used_source_name)  # type: ignore[union-attr]

            if df is None or df.empty:
                logger.warning("%s %s: no data (final)", market, symbol)
            else:
                # Save raw (legacy) if enabled
                if config.WRITE_RAW_TABLES:
                    save_fetched_df(conn, market, symbol, df)

                # Canonicalize + clean (qlib-style spirit) and save into bars_hourly_* tables
                cleaned = canonicalize_hourly_bars(
                    df,
                    market=market,
                    symbol=symbol,
                    source=used_source_name,
                    spec=spec,
                    interval="1h",
                )
                n2 = save_cleaned_hourly_df(conn, market, cleaned)

                # Silver v2 + Gold clean (calendar-aware). Keep v1 pipeline intact if v2 deps missing.
                if config.WRITE_V2_TABLES or config.WRITE_GOLD_TABLES:
                    try:
                        v2 = canonicalize_hourly_bars_v2(
                            df,
                            market=market,
                            symbol=symbol,
                            source=used_source_name,
                            spec=spec,
                            interval="1h",
                        )
                        n_v2 = save_cleaned_hourly_v2_df(conn, market, v2) if config.WRITE_V2_TABLES else 0

                        # Corporate actions factors (daily)
                        adj_factors = None
                        if config.US_FETCH_DAILY_ADJ_FACTORS and market == config.MARKET_US and not v2.empty:
                            d = pd.to_datetime(v2["dt_end_local"], errors="coerce")
                            if not d.isna().all():
                                start_d = d.min().date()
                                end_d = d.max().date()
                                fac = fetch_us_daily_adj_factors(symbol, start_d, end_d)
                                if fac is not None and not fac.empty:
                                    save_adj_factors_df(conn, market, fac)
                                adj_factors = load_adj_factors_df(
                                    conn, market, symbol, start_d.isoformat(), end_d.isoformat()
                                )

                        if (
                            config.CN_COMPUTE_ADJ_FACTORS_DUAL_FETCH
                            and market == config.MARKET_CN
                            and source_unadj is not None
                        ):
                            # Dual fetch (unadjusted + qfq) to compute daily factors (rate-limit prone; optional).
                            df_unadj = _fetch_with_retries(source_unadj, market, symbol, start_arg, end_arg)
                            if df_unadj is not None and not df_unadj.empty:
                                v2_unadj = canonicalize_hourly_bars_v2(
                                    df_unadj,
                                    market=market,
                                    symbol=symbol,
                                    source="akshare_eastmoney_unadj",
                                    spec=spec,
                                    interval="1h",
                                )
                                fac = compute_cn_daily_adj_factors_from_dual_intraday(
                                    unadjusted_v2=v2_unadj, adjusted_v2=v2, symbol=symbol
                                )
                                if fac is not None and not fac.empty:
                                    save_adj_factors_df(conn, market, fac)
                                    s0, s1 = fac["trade_date"].min(), fac["trade_date"].max()
                                    adj_factors = load_adj_factors_df(conn, market, symbol, str(s0), str(s1))

                        # Gold clean: grid align + adjusted + anomalies + masks
                        if config.WRITE_GOLD_TABLES and v2 is not None and not v2.empty:
                            gold = build_gold_clean_bars(
                                v2,
                                market=market,
                                symbol=symbol,
                                adj_factors=adj_factors,
                                label_horizons=config.LABEL_HORIZONS_BARS,
                                abs_logret_threshold=config.ANOMALY_ABS_LOGRET_THRESHOLD,
                                robust_z_threshold=config.ANOMALY_ROBUST_Z_THRESHOLD,
                                min_abs_logret_for_z=config.ANOMALY_MIN_ABS_LOGRET_FOR_Z,
                                range_pct_threshold=config.ANOMALY_RANGE_PCT_THRESHOLD,
                            )
                            n_gold = save_cleaned_hourly_clean_df(conn, market, gold)
                        else:
                            n_gold = 0
                    except Exception as e:
                        logger.warning("%s %s: v2/gold skipped (%s)", market, symbol, e)
                        n_v2 = 0
                        n_gold = 0
                else:
                    n_v2 = 0
                    n_gold = 0

                logger.info("%s %s: raw=%d cleaned_saved=%d", market, symbol, len(df), n2)
                if config.WRITE_V2_TABLES:
                    logger.info("%s %s: v2_saved=%d", market, symbol, n_v2)
                if config.WRITE_GOLD_TABLES:
                    logger.info("%s %s: gold_saved=%d", market, symbol, n_gold)
        except Exception as e:
            logger.exception("%s %s: %s", market, symbol, e)
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--market", choices=["us", "cn", "all"], default="all")
        args, _ = parser.parse_known_args()

        if args.market in ("us", "all"):
            logger.info("Starting US market")
            run_market(conn, config.MARKET_US)
        if args.market in ("cn", "all"):
            logger.info("Starting CN market")
            run_market(conn, config.MARKET_CN)
    finally:
        conn.close()
    logger.info("Done")


if __name__ == "__main__":
    main()

"""
Table name rules (ohlcv_{market}_{year}Q{q}) and standard CREATE TABLE SQL.
"""


def table_name_for_quarter(market: str, year: int, quarter: int) -> str:
    """e.g. ohlcv_us_2024Q1, ohlcv_cn_2024Q1"""
    return f"ohlcv_{market}_{year}Q{quarter}"


def bars_table_name_for_quarter(market: str, year: int, quarter: int) -> str:
    """
    Cleaned, canonical hourly bars table.
    e.g. bars_hourly_us_2024Q1, bars_hourly_cn_2024Q1
    """
    return f"bars_hourly_{market}_{year}Q{quarter}"


def bars_v2_table_name_for_quarter(market: str, year: int, quarter: int) -> str:
    """
    Silver v2 (calendar-aware canonical bars + QC flags).
    e.g. bars_hourly_v2_us_2024Q1
    """
    return f"bars_hourly_v2_{market}_{year}Q{quarter}"


def bars_clean_table_name_for_quarter(market: str, year: int, quarter: int) -> str:
    """
    Gold clean table (grid-aligned + missing placeholders + adjusted + masks).
    e.g. bars_hourly_clean_us_2024Q1
    """
    return f"bars_hourly_clean_{market}_{year}Q{quarter}"


def quarter_from_datetime(dt) -> tuple:
    """
    Return (year, quarter) for a datetime-like value.
    quarter: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec.
    """
    if hasattr(dt, "year") and hasattr(dt, "month"):
        y, m = dt.year, dt.month
    else:
        # pandas Timestamp or datetime
        y, m = dt.year, dt.month
    q = (m - 1) // 3 + 1
    return (y, q)


def create_table_sql(table_name: str) -> str:
    """
    Standard OHLCV hourly table: symbol, datetime, open, high, low, close,
    volume, adj_close, dividend. UNIQUE(symbol, datetime) for upsert.
    """
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        " symbol TEXT NOT NULL,"
        " datetime TEXT NOT NULL,"
        " open REAL,"
        " high REAL,"
        " low REAL,"
        " close REAL,"
        " volume REAL,"
        " adj_close REAL,"
        " dividend REAL,"
        " PRIMARY KEY (symbol, datetime)"
        ")"
    )


def create_bars_table_sql(table_name: str) -> str:
    """
    Canonical hourly bars schema (qlib-style spirit):
    - primary key: (symbol, ts_end_utc)
    - store both UTC and exchange-local end timestamps
    - store richer fields: amount, source, updated_at_utc
    """
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        " symbol TEXT NOT NULL,"
        " ts_end_utc INTEGER NOT NULL,"
        " dt_end_utc TEXT NOT NULL,"
        " dt_end_local TEXT NOT NULL,"
        " exchange_tz TEXT NOT NULL,"
        " interval TEXT NOT NULL,"
        " open REAL,"
        " high REAL,"
        " low REAL,"
        " close REAL,"
        " volume REAL,"
        " amount REAL,"
        " adj_close REAL,"
        " dividend REAL,"
        " source TEXT,"
        " updated_at_utc TEXT,"
        " PRIMARY KEY (symbol, ts_end_utc)"
        ")"
    )


def create_bars_v2_table_sql(table_name: str) -> str:
    """
    Silver v2 canonical bars schema.
    Primary key remains (symbol, ts_end_utc) for idempotent upsert.
    """
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        " symbol TEXT NOT NULL,"
        " ts_start_utc INTEGER NOT NULL,"
        " ts_end_utc INTEGER NOT NULL,"
        " dt_start_utc TEXT NOT NULL,"
        " dt_end_utc TEXT NOT NULL,"
        " dt_start_local TEXT NOT NULL,"
        " dt_end_local TEXT NOT NULL,"
        " exchange_tz TEXT NOT NULL,"
        " interval TEXT NOT NULL,"
        " open REAL,"
        " high REAL,"
        " low REAL,"
        " close REAL,"
        " volume REAL,"
        " amount REAL,"
        " adj_close REAL,"
        " dividend REAL,"
        " source TEXT,"
        " updated_at_utc TEXT,"
        " flag_off_calendar INTEGER,"
        " flag_any_null_price INTEGER,"
        " flag_nonpositive_price INTEGER,"
        " flag_ohlc_inconsistent INTEGER,"
        " flag_volume_negative INTEGER,"
        " PRIMARY KEY (symbol, ts_end_utc)"
        ")"
    )


def create_bars_clean_table_sql(table_name: str) -> str:
    """
    Gold clean table schema:
    - grid-aligned buckets (one row per expected bucket)
    - missing placeholders + flags
    - adjusted prices + masks for training/label availability
    """
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        " symbol TEXT NOT NULL,"
        " ts_start_utc INTEGER NOT NULL,"
        " ts_end_utc INTEGER NOT NULL,"
        " dt_start_utc TEXT NOT NULL,"
        " dt_end_utc TEXT NOT NULL,"
        " dt_start_local TEXT NOT NULL,"
        " dt_end_local TEXT NOT NULL,"
        " trade_date TEXT NOT NULL,"
        " exchange_tz TEXT NOT NULL,"
        " interval TEXT NOT NULL,"
        " open REAL,"
        " high REAL,"
        " low REAL,"
        " close REAL,"
        " volume REAL,"
        " amount REAL,"
        " adj_factor REAL,"
        " open_adj REAL,"
        " high_adj REAL,"
        " low_adj REAL,"
        " close_adj REAL,"
        " dividend REAL,"
        " source TEXT,"
        " updated_at_utc TEXT,"
        " is_missing INTEGER NOT NULL,"
        " missing_reason TEXT,"
        " suspected_halt INTEGER,"
        " is_no_trade INTEGER,"
        " flag_off_calendar INTEGER,"
        " flag_any_null_price INTEGER,"
        " flag_nonpositive_price INTEGER,"
        " flag_ohlc_inconsistent INTEGER,"
        " flag_volume_negative INTEGER,"
        " flag_jump_unexplained INTEGER,"
        " flag_range_unusual INTEGER,"
        " flag_corp_action INTEGER,"
        " feature_mask INTEGER,"
        " label_mask_1 INTEGER,"
        " label_mask_4 INTEGER,"
        " label_mask_7 INTEGER,"
        " PRIMARY KEY (symbol, ts_end_utc)"
        ")"
    )


def adj_factors_table_name(market: str) -> str:
    """
    Daily adjustment factors table.
    e.g. adj_factors_daily_us, adj_factors_daily_cn
    """
    return f"adj_factors_daily_{market}"


def create_adj_factors_table_sql(table_name: str) -> str:
    """
    Daily adjustment factors used to transform raw prices -> adjusted prices.
    Factor is typically: AdjClose / Close on the same trade_date.
    """
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        " symbol TEXT NOT NULL,"
        " trade_date TEXT NOT NULL,"  # YYYY-MM-DD in exchange local calendar
        " adj_factor REAL NOT NULL,"
        " source TEXT,"
        " updated_at_utc TEXT,"
        " PRIMARY KEY (symbol, trade_date)"
        ")"
    )

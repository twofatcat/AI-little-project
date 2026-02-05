"""
Configuration: DB path, market list, free hourly history limits.
"""
import os
from datetime import timedelta

# SQLite database path (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "stocks.db")

# Markets
MARKET_US = "us"
MARKET_CN = "cn"
MARKETS = [MARKET_US, MARKET_CN]

# Exchange timezones (for canonicalization/cleaning)
TZ_US = "America/New_York"
TZ_CN = "Asia/Shanghai"

# Free tier: max days of hourly history allowed by data source
# US: yfinance intraday limited to last 60 days
FREE_HOURLY_DAYS_US = 60
# CN: AKShare 东财 minute limited to ~3 months
FREE_HOURLY_DAYS_CN = 90

# Fetch throttling (CN sources are more likely to rate-limit / temporarily block)
FETCH_DELAY_SECONDS_US = 0.4
FETCH_DELAY_SECONDS_CN = 2.0

# Retry when API returns empty / transient failures
FETCH_RETRY_TIMES = 3
FETCH_RETRY_BACKOFF_SECONDS = 3.0  # exponential backoff base

# When resuming, re-download a small overlap window for safety (INSERT OR REPLACE makes it idempotent)
RESUME_OVERLAP = timedelta(hours=48)

# Write raw (legacy) tables `ohlcv_*` in addition to cleaned tables `bars_hourly_*`
WRITE_RAW_TABLES = True

# Use a fallback CN source (Sina) if Eastmoney returns no data
CN_ENABLE_FALLBACK_SINA = True

# ---------------------------------------------------------------------------
# Industrial cleaning pipeline (v2 + Gold)
# ---------------------------------------------------------------------------

# Write Silver v2 canonical tables `bars_hourly_v2_*`
WRITE_V2_TABLES = True

# Write Gold clean tables `bars_hourly_clean_*`
WRITE_GOLD_TABLES = True

# Label mask horizons (in number of trading buckets/bars)
LABEL_HORIZONS_BARS = [1, 4, 7]

# Corporate actions / adjustment factors
# - US: fetch daily Close/AdjClose from yfinance and store adj_factor (AdjClose/Close) per trade_date.
US_FETCH_DAILY_ADJ_FACTORS = True
# - CN: computing adj_factor robustly requires both raw and adjusted prices; this can be expensive / rate-limited.
CN_COMPUTE_ADJ_FACTORS_DUAL_FETCH = False

# Anomaly detection thresholds (on adjusted prices)
ANOMALY_ABS_LOGRET_THRESHOLD = 0.20  # ~22% move in one bar
ANOMALY_ROBUST_Z_THRESHOLD = 10.0
ANOMALY_MIN_ABS_LOGRET_FOR_Z = 0.05
ANOMALY_RANGE_PCT_THRESHOLD = 0.20  # (high-low)/close

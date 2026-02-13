"""
CN-specific validation for v2/Gold pipeline fields and behaviors.

This script does NOT modify production code or database tables.
It builds synthetic CN market data (bar_end semantics) and writes:
  - test_outputs/v2_gold_field_validation_cn.log
  - test_outputs/v2_gold_field_validation_cn_report.md
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from cleaning import MarketTimeSpec, canonicalize_hourly_bars_v2
from clean_pipeline import build_gold_clean_bars


OUTPUT_DIR = PROJECT_ROOT / "test_outputs"
LOG_PATH = OUTPUT_DIR / "v2_gold_field_validation_cn.log"
REPORT_PATH = OUTPUT_DIR / "v2_gold_field_validation_cn_report.md"


EXPECTED_V2_FIELDS = [
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

EXPECTED_GOLD_EXTRA_FIELDS = [
    "trade_date",
    "adj_factor",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
    "is_missing",
    "missing_reason",
    "suspected_halt",
    "is_no_trade",
    "flag_jump_unexplained",
    "flag_range_unusual",
    "flag_corp_action",
    "feature_mask",
    "label_mask_1",
    "label_mask_4",
    "label_mask_7",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v2_gold_field_validation_cn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _build_synthetic_cn_raw() -> pd.DataFrame:
    # CN uses bar_end semantics in project config:
    # expected 1h bucket ends are generally: 10:30, 11:30, 14:00, 15:00 local.
    rows = [
        # Off-calendar (lunch break end-time; should be flagged off-calendar in v2)
        {"datetime": "2026-01-05 12:00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000, "amount": 10100},
        # Valid session bars (day 1)
        {"datetime": "2026-01-05 10:30:00", "open": 10.0, "high": 10.3, "low": 9.8, "close": 10.2, "volume": 12000, "amount": 122000},
        {"datetime": "2026-01-05 11:30:00", "open": 10.2, "high": 10.2, "low": 10.2, "close": 10.2, "volume": 0, "amount": 0},
        # Skip 14:00 to force missing bucket in Gold
        {"datetime": "2026-01-05 15:00:00", "open": 10.1, "high": 12.5, "low": 8.0, "close": 10.0, "volume": 8000, "amount": 82000},
        # Day 2 abnormal rows
        {"datetime": "2026-01-06 10:30:00", "open": None, "high": 10.6, "low": 10.2, "close": 10.4, "volume": 11000, "amount": 114000},
        {"datetime": "2026-01-06 11:30:00", "open": 10.4, "high": 10.3, "low": 10.1, "close": 10.5, "volume": 9000, "amount": 93000},
        {"datetime": "2026-01-06 14:00:00", "open": -1.0, "high": 10.8, "low": 10.0, "close": -1.0, "volume": 9500, "amount": 92000},
        {"datetime": "2026-01-06 15:00:00", "open": 10.6, "high": 10.7, "low": 10.5, "close": 10.6, "volume": -200, "amount": 2000},
    ]
    return pd.DataFrame(rows)


def _run_check(name: str, func: Callable[[], tuple[bool, str]], out: List[CheckResult], logger: logging.Logger) -> None:
    try:
        passed, details = func()
    except Exception as e:  # pragma: no cover
        passed, details = False, f"exception: {e}"
    out.append(CheckResult(name=name, passed=passed, details=details))
    level = logging.INFO if passed else logging.ERROR
    logger.log(level, "[%s] %s | %s", "PASS" if passed else "FAIL", name, details)


def main() -> None:
    logger = _setup_logging()
    logger.info("Starting CN v2/gold field validation with synthetic data")

    raw = _build_synthetic_cn_raw()
    logger.info("Synthetic CN raw rows: %d", len(raw))

    spec = MarketTimeSpec(exchange_tz=config.TZ_CN, input_tz=config.TZ_CN, timestamp_kind="bar_end")
    v2 = canonicalize_hourly_bars_v2(
        raw,
        market=config.MARKET_CN,
        symbol="CNTEST",
        source="synthetic_cn",
        spec=spec,
        interval="1h",
    )
    logger.info("v2 rows: %d", len(v2))

    # Simulate one corporate action day
    adj_factors = pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "adj_factor": 1.0},
            {"trade_date": "2026-01-06", "adj_factor": 1.2},
        ]
    )
    gold = build_gold_clean_bars(
        v2,
        market=config.MARKET_CN,
        symbol="CNTEST",
        adj_factors=adj_factors,
        label_horizons=(1, 4, 7),
        abs_logret_threshold=0.08,
        robust_z_threshold=5.0,
        min_abs_logret_for_z=0.02,
        range_pct_threshold=0.20,
    )
    logger.info("gold rows: %d", len(gold))

    checks: List[CheckResult] = []
    _run_check(
        "v2 required fields present",
        lambda: (
            set(EXPECTED_V2_FIELDS).issubset(set(v2.columns)),
            f"missing={sorted(set(EXPECTED_V2_FIELDS) - set(v2.columns))}",
        ),
        checks,
        logger,
    )
    _run_check(
        "gold required fields present",
        lambda: (
            set(EXPECTED_V2_FIELDS + EXPECTED_GOLD_EXTRA_FIELDS).issubset(set(gold.columns)),
            f"missing={sorted(set(EXPECTED_V2_FIELDS + EXPECTED_GOLD_EXTRA_FIELDS) - set(gold.columns))}",
        ),
        checks,
        logger,
    )
    _run_check(
        "v2 has off-calendar flag",
        lambda: (int(v2["flag_off_calendar"].sum()) >= 1, f"count={int(v2['flag_off_calendar'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "v2 has null-price flag",
        lambda: (int(v2["flag_any_null_price"].sum()) >= 1, f"count={int(v2['flag_any_null_price'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "v2 has nonpositive-price flag",
        lambda: (int(v2["flag_nonpositive_price"].sum()) >= 1, f"count={int(v2['flag_nonpositive_price'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "v2 has ohlc-inconsistent flag",
        lambda: (int(v2["flag_ohlc_inconsistent"].sum()) >= 1, f"count={int(v2['flag_ohlc_inconsistent'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "v2 has volume-negative flag",
        lambda: (int(v2["flag_volume_negative"].sum()) >= 1, f"count={int(v2['flag_volume_negative'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold has missing placeholders",
        lambda: (int(gold["is_missing"].sum()) >= 1, f"count={int(gold['is_missing'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold missing_reason vendor_gap present",
        lambda: (
            "vendor_gap" in set(gold.loc[gold["is_missing"] == 1, "missing_reason"].dropna().astype(str)),
            f"values={sorted(set(gold.loc[gold['is_missing'] == 1, 'missing_reason'].dropna().astype(str)))}",
        ),
        checks,
        logger,
    )
    _run_check(
        "gold has suspected_halt",
        lambda: (int(gold["suspected_halt"].sum()) >= 1, f"count={int(gold['suspected_halt'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold has is_no_trade",
        lambda: (int(gold["is_no_trade"].sum()) >= 1, f"count={int(gold['is_no_trade'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold has range anomaly flag",
        lambda: (int(gold["flag_range_unusual"].sum()) >= 1, f"count={int(gold['flag_range_unusual'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold has corp-action flag",
        lambda: (int(gold["flag_corp_action"].sum()) >= 1, f"count={int(gold['flag_corp_action'].sum())}"),
        checks,
        logger,
    )
    _run_check(
        "gold masks are binary",
        lambda: (
            all(set(gold[c].dropna().astype(int).unique()).issubset({0, 1}) for c in ["feature_mask", "label_mask_1", "label_mask_4", "label_mask_7"]),
            "checked feature_mask/label_mask_1/4/7",
        ),
        checks,
        logger,
    )
    _run_check(
        "CN lunch break buckets respected",
        lambda: (
            bool((gold["dt_end_local"].astype(str).str.endswith("10:30:00")).any())
            and bool((gold["dt_end_local"].astype(str).str.endswith("11:30:00")).any())
            and bool((gold["dt_end_local"].astype(str).str.endswith("14:00:00")).any())
            and bool((gold["dt_end_local"].astype(str).str.endswith("15:00:00")).any()),
            "expect end times include 10:30/11:30/14:00/15:00",
        ),
        checks,
        logger,
    )

    passed = sum(1 for c in checks if c.passed)
    failed = [c for c in checks if not c.passed]
    logger.info("Finished checks: total=%d pass=%d fail=%d", len(checks), passed, len(failed))

    lines: list[str] = []
    lines.append("# CN v2/Gold 字段与行为测试报告")
    lines.append("")
    lines.append("## 测试范围")
    lines.append("- 验证 CN 市场 v2/Gold 字段与行为。")
    lines.append("- 重点覆盖 CN 特性：午休断档、`bar_end` 时间语义、交易日内缺失占位。")
    lines.append("- 本次仅测试，不修改生产代码。")
    lines.append("")
    lines.append("## 检查结果")
    lines.append(f"- 总检查项：{len(checks)}")
    lines.append(f"- 通过：{passed}")
    lines.append(f"- 失败：{len(failed)}")
    lines.append("")
    lines.append("| 检查项 | 结果 | 说明 |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.details} |")
    lines.append("")
    lines.append("## 发现的问题")
    if not failed:
        lines.append("- 本次覆盖范围内未发现功能性失败。")
        lines.append("- 残余风险：真实行情下的长停牌、节假日临时调整、供应商复权口径差异仍需在线回归验证。")
    else:
        for c in failed:
            lines.append(f"- `{c.name}`: {c.details}")
    lines.append("")
    lines.append("## 建议改进（仅建议）")
    lines.append("- 增加 CN 专属回归集：春节前后、节前半天、长假后首日等样本。")
    lines.append("- 增加 source 对比测试（Sina/Eastmoney）并输出差异统计。")
    lines.append("- 在 CI 增加 Gold masks 稳定性监控（各 symbol 的 mask 比例阈值报警）。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append(f"- 日志：`{LOG_PATH}`")
    lines.append(f"- 报告：`{REPORT_PATH}`")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", REPORT_PATH)
    print(f"log: {LOG_PATH}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""
Validate v2/Gold pipeline fields and behaviors using synthetic problematic data.

This script does NOT modify production code or database tables.
It only generates synthetic in-memory data and writes test outputs:
  - test_outputs/v2_gold_field_validation.log
  - test_outputs/v2_gold_field_validation_report.md
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import pandas as pd

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from cleaning import MarketTimeSpec, canonicalize_hourly_bars_v2
from clean_pipeline import build_gold_clean_bars


OUTPUT_DIR = PROJECT_ROOT / "test_outputs"
LOG_PATH = OUTPUT_DIR / "v2_gold_field_validation.log"
REPORT_PATH = OUTPUT_DIR / "v2_gold_field_validation_report.md"


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
    logger = logging.getLogger("v2_gold_field_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _build_synthetic_us_raw() -> pd.DataFrame:
    # US market 1h bars, input timestamp is bar_start in UTC for this project.
    # Includes:
    # - off-calendar row
    # - null price row
    # - nonpositive price row
    # - inconsistent OHLC row
    # - negative volume row
    # - range outlier row
    # - large return jump row
    # - skipped bucket to create missing vendor_gap in Gold
    # - no-trade row (flat OHLC + volume 0)
    rows = [
        # Off-calendar: before RTH open (should become flag_off_calendar=1 in v2)
        {"datetime": "2026-01-02T13:00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        # RTH rows (bar_start UTC)
        {"datetime": "2026-01-02T14:30:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10000},
        # no-trade candidate
        {"datetime": "2026-01-02T15:30:00", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 0},
        # range outlier candidate
        {"datetime": "2026-01-02T16:30:00", "open": 101, "high": 140, "low": 80, "close": 102, "volume": 5000},
        # large jump candidate
        {"datetime": "2026-01-02T17:30:00", "open": 102, "high": 210, "low": 101, "close": 205, "volume": 6000},
        # Skip 18:30 to force missing bucket in Gold
        {"datetime": "2026-01-02T19:30:00", "open": 103, "high": 104, "low": 102, "close": 103, "volume": -50},
        {"datetime": "2026-01-02T20:30:00", "open": 0, "high": 103, "low": 98, "close": -1, "volume": 7000},
        # Next trading day: null + inconsistent OHLC
        {"datetime": "2026-01-05T14:30:00", "open": None, "high": 110, "low": 105, "close": 108, "volume": 9000},
        {"datetime": "2026-01-05T15:30:00", "open": 109, "high": 108, "low": 107, "close": 110, "volume": 9500},
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
    logger.info("Starting v2/gold field validation with synthetic data")

    raw = _build_synthetic_us_raw()
    logger.info("Synthetic raw rows: %d", len(raw))

    spec = MarketTimeSpec(exchange_tz=config.TZ_US, input_tz="UTC", timestamp_kind="bar_start")
    v2 = canonicalize_hourly_bars_v2(
        raw,
        market=config.MARKET_US,
        symbol="TEST",
        source="synthetic",
        spec=spec,
        interval="1h",
    )
    logger.info("v2 rows: %d", len(v2))

    adj_factors = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "adj_factor": 1.00},
            {"trade_date": "2026-01-05", "adj_factor": 1.15},
        ]
    )
    gold = build_gold_clean_bars(
        v2,
        market=config.MARKET_US,
        symbol="TEST",
        adj_factors=adj_factors,
        label_horizons=(1, 4, 7),
        abs_logret_threshold=0.12,  # easier trigger for synthetic jump
        robust_z_threshold=5.0,
        min_abs_logret_for_z=0.03,
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

    # v2 flag checks
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

    # Gold checks
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
        "gold has jump anomaly flag",
        lambda: (int(gold["flag_jump_unexplained"].sum()) >= 1, f"count={int(gold['flag_jump_unexplained'].sum())}"),
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

    # summary
    passed = sum(1 for c in checks if c.passed)
    failed = [c for c in checks if not c.passed]
    logger.info("Finished checks: total=%d pass=%d fail=%d", len(checks), passed, len(failed))

    # Build report markdown
    lines: list[str] = []
    lines.append("# v2/Gold 字段与行为测试报告")
    lines.append("")
    lines.append("## 测试范围")
    lines.append("- 依据文档 `数据处理说明_中美数据下载与清洗.md` 的 v2 + Gold 字段与行为定义进行验证。")
    lines.append("- 使用合成的 US 1h 样本，覆盖正常与异常场景（off-calendar、缺失、空值、负值、异常波动、停牌猜测等）。")
    lines.append("- 本次仅测试，不修改生产逻辑。")
    lines.append("")
    lines.append("## 数据设计（虚拟问题数据）")
    lines.append("- 包含 off-calendar 时间点（盘前）。")
    lines.append("- 包含空价格、非正价格、OHLC 不一致、负成交量。")
    lines.append("- 人为跳过一个中间 bucket，验证 missing/vendor_gap/suspected_halt。")
    lines.append("- 包含 no-trade bar（OHLC 平坦且 volume=0）。")
    lines.append("- 人工注入 daily adj_factor 变化，验证 corp_action 标记。")
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
        lines.append("- 本次脚本覆盖范围内，未发现功能性失败项。")
        lines.append("- 仍有残余风险：真实供应商数据可能出现更复杂异常（停牌跨日、复权源延迟、节假日临时变更）。")
    else:
        for c in failed:
            lines.append(f"- `{c.name}`: {c.details}")
    lines.append("")
    lines.append("## 建议的改进计划（仅建议，不在本次实现）")
    lines.append("- 增加参数化测试（按 US/CN、不同 horizon、不同阈值组合自动跑）。")
    lines.append("- 增加回归基线：把关键统计（flag 比例、mask 比例）做快照，CI 中比较漂移。")
    lines.append("- 增加 edge-case 用例：跨季度边界、早收盘日、连续停牌多天、adj_factor 缺失与异常值。")
    lines.append("- 增加自动数据体检脚本入口，定期对最新季度表执行 quality + schema 双重检查。")
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

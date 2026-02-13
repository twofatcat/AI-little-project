"""
Date formatting compatibility helpers for locale/calendar-sensitive systems.

Some OS locale/calendar settings (for example Arabic locale with Hijri calendar)
can produce localized digits or non-Gregorian years when using strftime/date APIs.
Data vendors typically require ASCII Gregorian date strings, so we normalize here.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime
from typing import Tuple

_YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YMD_HMS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_ISLAMIC_HINTS = ("islamic", "hijri", "ummalqura", "umalqura", "arabic", "ar_sa", "sa")


def _to_ascii_digits(text: str) -> str:
    out = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except Exception:
                pass
        out.append(ch)
    return "".join(out)


def _looks_plausible_gregorian_year(year: int) -> bool:
    return 1900 <= year <= 2100


def detect_islamic_like_calendar_env() -> bool:
    """
    Best-effort check for locale/calendar environments likely to localize dates.
    Returns True only when there are signs of non-default date calendar formatting.
    """
    locale_blob = " ".join(
        str(os.environ.get(k, "")).lower()
        for k in ("LC_ALL", "LC_TIME", "LANG", "LANGUAGE")
    )
    has_locale_hint = any(h in locale_blob for h in _ISLAMIC_HINTS)

    sample = datetime.now().strftime("%Y-%m-%d")
    sample_ascii = _to_ascii_digits(sample)
    has_non_ascii_digits = sample != sample_ascii

    year_plausible = False
    if _YMD_RE.match(sample_ascii):
        try:
            year_plausible = _looks_plausible_gregorian_year(int(sample_ascii[:4]))
        except Exception:
            year_plausible = False

    # Treat as problematic only when hints exist together with formatting oddities.
    return has_locale_hint and (has_non_ascii_digits or not year_plausible)


def safe_date_ymd(dt: date | datetime) -> Tuple[str, bool]:
    """
    Return API-safe YYYY-MM-DD (ASCII Gregorian), plus whether normalization applied.
    """
    raw = dt.strftime("%Y-%m-%d")
    normalized = _to_ascii_digits(raw)
    if _YMD_RE.match(normalized):
        try:
            y = int(normalized[:4])
            if _looks_plausible_gregorian_year(y):
                return normalized, normalized != raw
        except Exception:
            pass
    # Hard fallback from numeric fields (locale-independent)
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}", True


def safe_datetime_ymd_hms(dt: datetime) -> Tuple[str, bool]:
    """
    Return API-safe YYYY-MM-DD HH:MM:SS (ASCII Gregorian), plus normalization flag.
    """
    raw = dt.strftime("%Y-%m-%d %H:%M:%S")
    normalized = _to_ascii_digits(raw)
    if _YMD_HMS_RE.match(normalized):
        try:
            y = int(normalized[:4])
            if _looks_plausible_gregorian_year(y):
                return normalized, normalized != raw
        except Exception:
            pass
    return (
        f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} "
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}",
        True,
    )

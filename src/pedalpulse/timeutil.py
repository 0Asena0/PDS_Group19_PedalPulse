"""Local New York wall time to UTC, with explicit rules for the two DST edge cases.

Raw trip times are naive local wall-clock times. Two cases need a rule:
- Spring forward: local 02:00-02:59 does not exist. Such times are read with the offset
  before the change (EST), which places them at 03:00-03:59 EDT.
- Fall back: local 01:00-01:59 occurs twice. Such times are assigned to the first
  occurrence (EDT, UTC-4); the second occurrence (EST, UTC-5) receives no trips.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

NY = ZoneInfo(config.TIMEZONE)
AMBIGUOUS_FOLD = 0  # 0 = first occurrence (EDT), 1 = second occurrence (EST)


def local_to_utc(local: datetime) -> datetime:
    """Map a naive local wall time to naive UTC using the rules above."""
    aware = local.replace(tzinfo=NY, fold=AMBIGUOUS_FOLD)
    return aware.astimezone(timezone.utc).replace(tzinfo=None)


def utc_to_local(utc: datetime) -> datetime:
    return utc.replace(tzinfo=timezone.utc).astimezone(NY).replace(tzinfo=None)


def wall_time_kind(local: datetime) -> str:
    first = local.replace(tzinfo=NY, fold=0).astimezone(timezone.utc)
    second = local.replace(tzinfo=NY, fold=1).astimezone(timezone.utc)
    if first == second:
        return "regular"
    return "ambiguous" if first < second else "nonexistent"


def local_hour_map(first_local: datetime, last_local: datetime) -> pd.DataFrame:
    """One row per naive local wall-clock hour with the UTC hour it maps to."""
    rows = []
    hour = first_local.replace(minute=0, second=0, microsecond=0)
    while hour <= last_local:
        rows.append((hour, local_to_utc(hour), wall_time_kind(hour)))
        hour += timedelta(hours=1)
    return pd.DataFrame(rows, columns=["local_hour", "utc_hour", "kind"])


def local_midnight_utc(day) -> datetime:
    """Forecast origin: 00:00 America/New_York on the given local date, as naive UTC."""
    return local_to_utc(datetime(day.year, day.month, day.day))


def grid_bounds_utc() -> tuple[datetime, datetime]:
    """First and last UTC hour start of the study period (inclusive)."""
    start = local_midnight_utc(config.STUDY_START)
    end = local_midnight_utc(config.STUDY_END + timedelta(days=1)) - timedelta(hours=1)
    return start, end

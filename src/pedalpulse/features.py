"""History features for day-ahead forecasts, built the same way for backtests and the app.

Every history feature points to a source: a local day before the target day, and for hourly lags a local hour.
A value is used only if the source is complete at the forecast origin:

    source end + latency_h <= origin_utc + origin_shift_h

The defaults follow the data contract (origin 00:00 local on the target day, no latency). origin_shift_h=-1 and
latency_h=1 give the roadmap variant (origin 23:00 on the day before, one hour latency); values that are not
available then become missing.

Lags match the local date and local hour, not a fixed UTC shift. On a fall-back day the local hour 1 occurs twice;
the first occurrence is used (trips are assigned to it, see data contract). On a spring-forward day the local hour
2 does not exist, so lags pointing to it are missing. Missing history (before the station's first hour, or a
missing source hour) stays missing and is never filled with 0.

`status` is not used: it is computed from later days.
"""
from __future__ import annotations

import pandas as pd

HOUR_LAG_DAYS = (1, 2, 7, 14, 21, 28)
CALENDAR_FEATURES = ["hour_local", "weekday", "month", "is_weekend", "is_holiday", "is_dst_change_day"]
HISTORY_FEATURES = [f"lag_{k}d" for k in HOUR_LAG_DAYS] + [
    "mean_same_hour_7d",
    "mean_same_weekday_hour_4w",
    "day_total_1d",
    "day_total_7d",
]
KEY = ["station_id", "date_local", "hour_local"]


def _hours(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station, local date and local hour (first occurrence), with the end of the source hour."""
    # On the fall-back day, local hour 1 exists twice. I keep the first one, because Benjamin's pipeline
    # puts all trips of that hour there (the second one is always 0).
    src = df.sort_values("hour_utc").drop_duplicates(KEY)
    return src[KEY + ["pickups"]].assign(source_end=src["hour_utc"] + pd.Timedelta(hours=1))


def _days(df: pd.DataFrame) -> pd.DataFrame:
    """Pickups per station and local day, with the end of the local day in UTC."""
    days = df.groupby(["station_id", "date_local"], as_index=False).agg(
        pickups=("pickups", "sum"), source_end=("hour_utc", "max")
    )
    days["source_end"] += pd.Timedelta(hours=1)
    return days


def _cutoff(out: pd.DataFrame, origin_shift_h: int, latency_h: int) -> pd.Series:
    # The latest moment a value may come from. With the defaults this is midnight before the target day.
    # I made shift and latency parameters because the roadmap says 23:00 + 1 h and the data contract says 00:00,
    # so later I can measure what the difference costs.
    return out["origin_utc"] + pd.Timedelta(hours=origin_shift_h - latency_h)


def _shift_days(table: pd.DataFrame, days: int) -> pd.DataFrame:
    return table.assign(date_local=table["date_local"] + pd.Timedelta(days=days))


def add_history_features(df: pd.DataFrame, *, origin_shift_h: int = 0, latency_h: int = 0) -> pd.DataFrame:
    """Add lags, rolling means and daily totals of past pickups. Needs origin_utc (from the calendar)."""
    if "origin_utc" not in df.columns:
        raise KeyError("origin_utc missing; load the station-hours with the calendar")
    out = df.copy()
    cutoff = _cutoff(out, origin_shift_h, latency_h)
    hours = _hours(df)

    # I match lags on local date and local hour, not with shift(24): rows can be missing,
    # and on DST days 24 UTC hours back is not the same local hour.
    def lag(days: int) -> pd.Series:
        src = _shift_days(hours, days)
        merged = out[KEY].merge(src, on=KEY, how="left")
        merged.index = out.index
        # Only keep the value if its hour was over at the forecast origin. Otherwise it stays empty (NaN), never 0.
        ok = merged["source_end"] <= cutoff
        return merged["pickups"].where(ok)

    for k in HOUR_LAG_DAYS:
        out[f"lag_{k}d"] = lag(k)
    last_week = pd.concat([out[f"lag_{k}d"] if k in HOUR_LAG_DAYS else lag(k) for k in range(1, 8)], axis=1)
    out["mean_same_hour_7d"] = last_week.mean(axis=1)
    # I build the averages only from the lags above, so they can never see more than the lags are allowed to.
    out["mean_same_weekday_hour_4w"] = out[[f"lag_{k}d" for k in (7, 14, 21, 28)]].mean(axis=1)

    days = _days(df)
    for k in (1, 7):
        src = _shift_days(days, k)
        merged = out[["station_id", "date_local"]].merge(src, on=["station_id", "date_local"], how="left")
        merged.index = out.index
        out[f"day_total_{k}d"] = merged["pickups"].where(merged["source_end"] <= cutoff)
    return out

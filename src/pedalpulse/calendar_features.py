"""Calendar features per UTC hour, derived in America/New_York local time."""

from datetime import timedelta

import holidays
import pandas as pd

from . import config
from .timeutil import grid_bounds_utc

CALENDAR_FILE = "calendar.parquet"


def holiday_calendar(years) -> holidays.HolidayBase:
    return holidays.country_holidays("US", subdiv="NY", years=list(years), observed=True)


def build_calendar_frame() -> pd.DataFrame:
    first, last = grid_bounds_utc()
    hours_utc = pd.date_range(first, last, freq="h", tz="UTC")
    local = hours_utc.tz_convert(config.TIMEZONE)
    frame = pd.DataFrame({"hour_utc": hours_utc})
    frame["local_time"] = local.tz_localize(None)
    frame["date_local"] = local.date
    frame["hour_local"] = local.hour.astype("int8")
    frame["utc_offset_h"] = (local.map(lambda ts: ts.utcoffset().total_seconds() / 3600)).astype("int8")
    frame["weekday"] = local.day_of_week.astype("int8")  # 0 = Monday
    frame["month"] = local.month.astype("int8")
    frame["year"] = local.year.astype("int16")
    frame["is_weekend"] = frame["weekday"] >= 5

    calendar = holiday_calendar(range(config.STUDY_START.year - 1, config.STUDY_END.year + 2))
    names = frame["date_local"].map(lambda day: calendar.get(day))
    frame["is_holiday"] = names.notna()
    frame["holiday_name"] = names.astype("string")
    frame["is_observed_holiday"] = names.fillna("").str.contains("(observed)", regex=False)
    federal = holidays.country_holidays("US", years=list(range(config.STUDY_START.year - 1, config.STUDY_END.year + 2)), observed=True)
    frame["is_federal_holiday"] = frame["date_local"].map(lambda day: day in federal)

    hours_per_day = frame.groupby("date_local")["hour_utc"].transform("size").astype("int8")
    frame["local_hours_in_day"] = hours_per_day
    frame["is_dst_change_day"] = hours_per_day != 24

    origin = pd.to_datetime(frame["date_local"]).dt.tz_localize(config.TIMEZONE).dt.tz_convert("UTC")
    frame["origin_utc"] = origin
    return frame


def validate_calendar(frame: pd.DataFrame) -> None:
    if not frame["hour_utc"].is_unique:
        raise AssertionError("calendar hour_utc is not unique")
    step = frame["hour_utc"].diff().dropna()
    if not (step == timedelta(hours=1)).all():
        raise AssertionError("calendar has gaps")
    days = frame.groupby("date_local").size()
    if not days.isin([23, 24, 25]).all():
        raise AssertionError("local day with an impossible number of hours")


def build_calendar(force: bool = False) -> None:
    target = config.PROCESSED_DIR / CALENDAR_FILE
    if target.exists() and not force:
        print("skip calendar (done)")
        return
    frame = build_calendar_frame()
    validate_calendar(frame)
    frame.to_parquet(target, index=False, compression="zstd")
    holiday_days = frame.loc[frame["is_holiday"], "date_local"].nunique()
    print(f"calendar: {len(frame):,} hours, {holiday_days} holiday dates")

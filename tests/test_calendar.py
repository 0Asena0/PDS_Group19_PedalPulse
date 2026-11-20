from datetime import date
import pandas as pd

from pedalpulse.calendar_features import build_calendar_frame, validate_calendar


def day(frame, value):
    return frame[frame["date_local"] == value]


def test_calendar_is_complete_and_unique():
    frame = build_calendar_frame()
    validate_calendar(frame)
    assert frame["hour_utc"].is_unique


def test_dst_days_have_23_and_25_local_hours():
    frame = build_calendar_frame()
    assert len(day(frame, date(2024, 3, 10))) == 23
    assert len(day(frame, date(2024, 11, 3))) == 25
    assert len(day(frame, date(2024, 7, 1))) == 24
    hours_per_day = frame.groupby("date_local").size()
    changes = hours_per_day[hours_per_day != 24]
    assert len(changes) == 7  # three spring and three fall changes 2023-2025, spring 2026
    assert set(changes.values) == {23, 25}


def test_holiday_samples():
    frame = build_calendar_frame()
    thanksgiving = day(frame, date(2024, 11, 28))
    assert thanksgiving["is_holiday"].all() and thanksgiving["holiday_name"].iloc[0] == "Thanksgiving Day"
    # Independence Day falls on weekdays 2023-2025, so the observed case is New Year's Day 2023 (a Sunday).
    observed = day(frame, date(2023, 1, 2))
    assert observed["is_holiday"].all() and observed["is_observed_holiday"].all()
    independence = day(frame, date(2024, 7, 4))
    assert independence["is_holiday"].all() and independence["weekday"].iloc[0] == 3
    assert not day(frame, date(2024, 7, 5))["is_holiday"].any()


def test_weekend_flag():
    frame = build_calendar_frame()
    saturday = day(frame, date(2024, 7, 6))
    assert saturday["is_weekend"].all() and saturday["weekday"].iloc[0] == 5


def lag_source_complete(frame, source_hours):
    return source_hours + pd.Timedelta(hours=1) <= frame["origin_utc"]


def test_local_day_lag_respects_origin_but_utc_shift_does_not():
    frame = build_calendar_frame()
    local_key = frame.set_index(["date_local", "hour_local"])["hour_utc"]
    local_key = local_key[~local_key.index.duplicated()]
    previous = pd.MultiIndex.from_arrays([frame["date_local"] - pd.Timedelta(days=1), frame["hour_local"]])
    local_lag = pd.Series(local_key.reindex(previous).to_numpy(), index=frame.index)
    has_lag = local_lag.notna()
    assert lag_source_complete(frame[has_lag], local_lag[has_lag]).all()

    utc_lag = frame["hour_utc"] - pd.Timedelta(hours=24)
    leaking = frame.loc[~lag_source_complete(frame, utc_lag), "date_local"]
    assert sorted(map(str, leaking)) == ["2023-11-05", "2024-11-03", "2025-11-02"]

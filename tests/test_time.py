from datetime import date, datetime

import pandas as pd

from pedalpulse.timeutil import grid_bounds_utc, local_hour_map, local_midnight_utc, local_to_utc, wall_time_kind


def test_regular_winter_and_summer_offsets():
    assert local_to_utc(datetime(2024, 1, 15, 8, 30)) == datetime(2024, 1, 15, 13, 30)
    assert local_to_utc(datetime(2024, 7, 15, 8, 30)) == datetime(2024, 7, 15, 12, 30)


def test_fall_back_hour_goes_to_first_occurrence():
    # 2024-11-03 01:30 happens twice; the rule maps it to the first one (EDT, UTC-4).
    assert wall_time_kind(datetime(2024, 11, 3, 1, 30)) == "ambiguous"
    assert local_to_utc(datetime(2024, 11, 3, 1, 30)) == datetime(2024, 11, 3, 5, 30)
    assert local_to_utc(datetime(2024, 11, 3, 2, 30)) == datetime(2024, 11, 3, 7, 30)


def test_spring_forward_gap_is_shifted_forward():
    # 2024-03-10 02:30 does not exist; read with the EST offset it lands at 03:30 EDT.
    assert wall_time_kind(datetime(2024, 3, 10, 2, 30)) == "nonexistent"
    assert local_to_utc(datetime(2024, 3, 10, 2, 30)) == datetime(2024, 3, 10, 7, 30)
    assert local_to_utc(datetime(2024, 3, 10, 3, 30)) == datetime(2024, 3, 10, 7, 30)


def test_hour_map_never_skips_utc_hours_on_fall_back():
    hours = local_hour_map(datetime(2024, 11, 2, 23), datetime(2024, 11, 3, 4))
    assert list(hours["utc_hour"].diff().dropna().dt.total_seconds() / 3600) == [1, 1, 2, 1, 1]


def test_forecast_origin_is_local_midnight():
    assert local_midnight_utc(date(2024, 7, 4)) == datetime(2024, 7, 4, 4)
    assert local_midnight_utc(date(2024, 12, 25)) == datetime(2024, 12, 25, 5)


def test_grid_bounds_cover_the_study_period():
    first, last = grid_bounds_utc()
    assert first == datetime(2023, 1, 1, 5)
    assert last == datetime(2026, 4, 1, 3)
    assert pd.Timestamp(last) - pd.Timestamp(first) == pd.Timedelta(hours=28462)

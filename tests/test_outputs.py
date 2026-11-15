"""Checks on the real pipeline outputs. They skip when the outputs are not built."""

import json

import duckdb
import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.months import study_months
from pedalpulse.timeutil import grid_bounds_utc


def parquet(path) -> str:
    return f"read_parquet('{path.as_posix()}')"


def release_file(name):
    path = config.RELEASE_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not exported")
    return path


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute("SET memory_limit = '3GB'")
    yield connection
    connection.close()


def test_every_study_month_is_conserved(con):
    logs = sorted((config.PROCESSED_DIR / "filter_log").glob("*.json"))
    if not logs:
        pytest.skip("no months loaded")
    checked = []
    for path in logs:
        log = json.loads(path.read_text())
        month = log["month"]
        assert log["raw_rows"] == log["kept_rows"] + sum(log["dropped"].values()), month
        pickups = con.execute(
            f"SELECT sum(pickups) FROM {parquet(config.PROCESSED_DIR / 'pickups' / f'{month}.parquet')}"
        ).fetchone()[0]
        assert pickups == log["kept_rows"], month
        checked.append(month)
    if len(checked) == len(study_months()) or release_file("station_hours_dev.parquet").exists():
        assert checked == study_months()


def test_release_filter_log_adds_up():
    path = release_file("filter_log.parquet")
    if not path.exists():
        pytest.skip("release not built or downloaded")
    log = pd.read_parquet(path)
    dropped = ["duplicate_ride_id", "missing_start_station", "unparsable_time", "time_out_of_range",
               "end_before_start", "start_outside_study_period"]
    assert list(log["month"]) == study_months()
    assert (log["raw_rows"] == log["kept_rows"] + log[dropped].sum(axis=1)).all()


def test_ride_ids_are_unique_across_months(con):
    trips = config.PROCESSED_DIR / "trips"
    if len(list(trips.glob("*.parquet"))) != len(study_months()):
        pytest.skip("not all months loaded")
    duplicates = con.execute(f"""
        SELECT count(*) FROM (SELECT ride_id FROM {parquet(trips / '*.parquet')}
                              WHERE ride_id IS NOT NULL GROUP BY 1 HAVING count(*) > 1)
    """).fetchone()[0]
    assert duplicates == 0


def test_station_hour_keys_are_unique(con):
    for name in ("station_hours_dev.parquet", "station_hours_holdout.parquet"):
        rows, keys = con.execute(f"""
            SELECT count(*), count(DISTINCT (station_id, hour_utc)) FROM {parquet(release_file(name))}
        """).fetchone()
        assert rows == keys, name


def test_dev_and_holdout_split_cleanly_and_cover_the_period(con):
    dev = parquet(release_file("station_hours_dev.parquet"))
    holdout = parquet(release_file("station_hours_holdout.parquet"))
    dev_last, dev_last_date = con.execute(f"SELECT max(hour_utc)::TIMESTAMP, max(date_local) FROM {dev}").fetchone()
    hold_first, hold_first_date, hold_last = con.execute(
        f"SELECT min(hour_utc)::TIMESTAMP, min(date_local), max(hour_utc)::TIMESTAMP FROM {holdout}"
    ).fetchone()
    first_hour, last_hour = grid_bounds_utc()
    assert dev_last < hold_first
    assert dev_last_date == config.DEV_END
    assert hold_first_date == config.HOLDOUT_START
    assert con.execute(f"SELECT min(hour_utc)::TIMESTAMP FROM {dev}").fetchone()[0] == first_hour
    assert hold_last == last_hour
    overlap = con.execute(f"""
        SELECT count(*) FROM {dev} d JOIN {holdout} h USING (station_id, hour_utc)
    """).fetchone()[0]
    assert overlap == 0


def test_every_cohort_station_has_every_hour_after_its_start(con):
    stations = parquet(release_file("stations.parquet"))
    dev = parquet(release_file("station_hours_dev.parquet"))
    holdout = parquet(release_file("station_hours_holdout.parquet"))
    first_hour, last_hour = grid_bounds_utc()
    result = con.execute(f"""
        WITH grid AS (SELECT station_id, hour_utc FROM {dev} UNION ALL SELECT station_id, hour_utc FROM {holdout}),
        per_station AS (
            SELECT station_id, count(*) AS n, min(hour_utc) AS first_hour, max(hour_utc) AS last_hour
            FROM grid GROUP BY 1
        )
        SELECT s.station_id, s.first_seen_hour_utc, p.n, p.first_hour, p.last_hour
        FROM {stations} s LEFT JOIN per_station p USING (station_id)
        WHERE s.is_cohort
    """).df()
    assert result["n"].notna().all()
    expected_first = result["first_seen_hour_utc"].clip(lower=pd.Timestamp(first_hour, tz="UTC"))
    assert (result["first_hour"] == expected_first).all(), "rows before station start or missing first hours"
    assert (result["last_hour"] == pd.Timestamp(last_hour, tz="UTC")).all()
    expected_rows = (result["last_hour"] - result["first_hour"]) / pd.Timedelta(hours=1) + 1
    assert (result["n"] == expected_rows).all(), "gaps in the station-hour grid"


def test_joins_with_calendar_and_weather_do_not_multiply_rows(con):
    dev = parquet(release_file("dev_sample_manhattan_2024_summer.parquet"))
    calendar = parquet(release_file("calendar.parquet"))
    forecast = parquet(release_file("weather_forecast.parquet"))
    stations = parquet(release_file("stations.parquet"))
    base = con.execute(f"SELECT count(*) FROM {dev}").fetchone()[0]
    joined = con.execute(f"""
        SELECT count(*) FROM {dev} d
        LEFT JOIN {calendar} c USING (hour_utc)
        LEFT JOIN {forecast} w USING (hour_utc)
        LEFT JOIN {stations} s USING (station_id)
    """).fetchone()[0]
    assert base == joined
    missing_calendar = con.execute(f"""
        SELECT count(*) FROM {dev} d ANTI JOIN {calendar} c USING (hour_utc)
    """).fetchone()[0]
    assert missing_calendar == 0


def test_inactive_status_only_on_days_without_trips(con):
    dev = parquet(release_file("station_hours_dev.parquet"))
    busy_inactive_days = con.execute(f"""
        SELECT count(*) FROM (
            SELECT station_id, date_local FROM {dev} WHERE status = 'inactive'
            GROUP BY ALL HAVING sum(pickups + dropoffs) > 0
        )
    """).fetchone()[0]
    assert busy_inactive_days == 0
    assert set(r[0] for r in con.execute(f"SELECT DISTINCT status FROM {dev}").fetchall()) <= {"observed", "inactive", "dst_repeat"}


def test_daily_system_has_every_study_day():
    daily = pd.read_parquet(release_file("daily_system.parquet"))
    days = pd.date_range(config.STUDY_START, config.STUDY_END, freq="D").date
    assert list(daily["date_local"]) == list(days)


def test_dev_sample_is_small_enough():
    path = release_file("dev_sample_manhattan_2024_summer.parquet")
    assert path.stat().st_size < 50 * 1024 * 1024


def test_checksums_match():
    import hashlib

    sums = release_file("SHA256SUMS").read_text().split("\n")
    for line in filter(None, sums):
        digest, name = line.split("  ")
        assert hashlib.sha256((config.RELEASE_DIR / name).read_bytes()).hexdigest() == digest, name

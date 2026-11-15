"""Cleaning rules and archive handling on a small synthetic month."""

import io
import json
import zipfile

import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.hourly import aggregate_month, output_path
from pedalpulse.trips import EXPECTED_COLUMNS, extract_month_csvs, filter_log_path, load_month, trips_path

ROWS = [
    # ride_id, rideable_type, started_at, ended_at, start name, start id, end name, end id, lat/lng x4, member
    ("a", "classic_bike", "2024-11-03 00:30:00", "2024-11-03 00:50:00", "S1", "6140.05", "S2", "6140.1", 40.7, -73.9, 40.7, -73.9, "member"),
    ("b", "electric_bike", "2024-11-03 01:30:00.123", "2024-11-03 01:45:00", "S1", "6140.05", "S2", "6140.10", 40.7, -73.9, 40.7, -73.9, "casual"),
    ("b", "electric_bike", "2024-11-03 01:30:00.123", "2024-11-03 01:45:00", "S1", "6140.05", "S2", "6140.10", 40.7, -73.9, 40.7, -73.9, "casual"),
    ("c", "classic_bike", "2024-11-03 02:10:00", "2024-11-03 02:20:00", "", "", "S2", "6140.10", 40.7, -73.9, 40.7, -73.9, "member"),
    ("d", "classic_bike", "not a time", "2024-11-03 02:20:00", "S1", "6140.05", "S2", "6140.10", 40.7, -73.9, 40.7, -73.9, "member"),
    ("e", "classic_bike", "2024-11-03 03:00:00", "2024-11-03 02:00:00", "S1", "6140.05", "S2", "6140.10", 40.7, -73.9, 40.7, -73.9, "member"),
    ("f", "classic_bike", "2024-11-03 05:00:00", "2024-11-03 05:00:30", "S1", "6140.05", "", "", 40.7, -73.9, None, None, "member"),
    ("g", "classic_bike", "2024-11-30 23:50:00", "2024-12-01 00:10:00", "S2", "6140.10_", "S1", "6140.05", 40.7, -73.9, 40.7, -73.9, "member"),
]


def csv_bytes(rows) -> bytes:
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    return frame.to_csv(index=False).encode()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RELEASE_DIR", tmp_path / "release")
    monkeypatch.setattr(config, "DUCKDB_TEMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "TRACKED_DIR", tmp_path / "tracked")
    config.ensure_dirs()
    with zipfile.ZipFile(config.RAW_DIR / "202411-citibike-tripdata.zip", "w") as archive:
        archive.writestr("202411-citibike-tripdata_1.csv", csv_bytes(ROWS[:4]))
        archive.writestr("202411-citibike-tripdata_2.csv", csv_bytes(ROWS[4:]))
        archive.writestr("__MACOSX/._202411-citibike-tripdata_1.csv", b"junk")
    return tmp_path


def test_filter_log_counts_every_rule(sandbox):
    load_month("2024-11")
    log = json.loads(filter_log_path("2024-11").read_text())
    assert log["raw_rows"] == 8
    assert log["dropped"] == {
        "duplicate_ride_id": 1,
        "missing_start_station": 1,
        "unparsable_time": 1,
        "time_out_of_range": 0,
        "end_before_start": 1,
        "start_outside_study_period": 0,
    }
    assert log["kept_rows"] == 4
    assert log["kept_start_ambiguous_hour"] == 1
    assert log["kept_missing_end_station"] == 1
    assert log["kept_duration_under_1min"] == 1
    assert log["csv_parts"] == ["202411-citibike-tripdata_1.csv", "202411-citibike-tripdata_2.csv"]


def test_station_ids_are_normalised_and_times_converted(sandbox):
    load_month("2024-11")
    trips = pd.read_parquet(trips_path("2024-11")).set_index("ride_id")
    assert trips.loc["a", "end_station_id"] == "6140.10"
    assert trips.loc["g", "start_station_id"] == "6140.10"
    assert str(trips.loc["b", "started_utc"]) == "2024-11-03 05:30:00.123000"
    assert str(trips.loc["a", "started_utc"]) == "2024-11-03 04:30:00"


def test_aggregation_preserves_trip_counts(sandbox):
    load_month("2024-11")
    aggregate_month("2024-11")
    pickups = pd.read_parquet(output_path("pickups", "2024-11"))
    dropoffs = pd.read_parquet(output_path("dropoffs", "2024-11"))
    assert pickups["pickups"].sum() == 4
    assert dropoffs["dropoffs"].sum() == 3  # one trip has no end station
    assert not pickups.duplicated(["station_id", "hour_utc"]).any()
    ended_next_month = dropoffs[dropoffs["hour_utc"] == pd.Timestamp("2024-12-01 05:00")]
    assert ended_next_month["dropoffs"].sum() == 1


def test_nested_yearly_archive_is_unpacked(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("202301-citibike-tripdata_1.csv", csv_bytes(ROWS[:1]))
        archive.writestr("202301-citibike-tripdata_2.csv", csv_bytes(ROWS[1:2]))
    outer_path = tmp_path / "2023-citibike-tripdata.zip"
    with zipfile.ZipFile(outer_path, "w") as archive:
        archive.writestr("2023-citibike-tripdata/202301-citibike-tripdata.zip", inner.getvalue())
        archive.writestr("__MACOSX/2023-citibike-tripdata/._202301-citibike-tripdata.zip", b"junk")
        archive.writestr("2023-citibike-tripdata/.DS_Store", b"junk")
        archive.writestr("2023-citibike-tripdata/202302-citibike-tripdata.zip", inner.getvalue())
    csvs, nested = extract_month_csvs(outer_path, "2023-01", tmp_path / "out")
    assert nested == "2023-citibike-tripdata/202301-citibike-tripdata.zip"
    assert [path.name for path in csvs] == ["202301-citibike-tripdata_1.csv", "202301-citibike-tripdata_2.csv"]

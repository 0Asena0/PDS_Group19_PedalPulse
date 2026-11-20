"""Load the release tables with calendar and forecast weather attached.

Only the day-ahead forecast is joined. Observed weather stays out on purpose: it is not known at the forecast origin.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import config

SPLITS = {
    "dev": "station_hours_dev.parquet",
    "holdout": "station_hours_holdout.parquet",
    "sample": "dev_sample_manhattan_2024_summer.parquet",
}
CALENDAR_COLUMNS = ["weekday", "month", "is_weekend", "is_holiday", "is_federal_holiday", "is_dst_change_day", "origin_utc"]
WEATHER_COLUMNS = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]
# data-v1 contains 19 forecast values just outside the physical range (cloud cover -1 or 101 %, precipitation -0.1 mm),
# rounding artefacts of the weather models. data-v2 is clipped at export; the loader clips too, so both versions agree.
WEATHER_SQL = {
    "temperature_2m": "w.temperature_2m",
    # CASE keeps missing values missing; DuckDB's greatest/least would turn NULL into the bound.
    "precipitation": "CASE WHEN w.precipitation < 0 THEN 0 ELSE w.precipitation END AS precipitation",
    "wind_speed_10m": "CASE WHEN w.wind_speed_10m < 0 THEN 0 ELSE w.wind_speed_10m END AS wind_speed_10m",
    "cloud_cover": "CASE WHEN w.cloud_cover < 0 THEN 0 WHEN w.cloud_cover > 100 THEN 100 ELSE w.cloud_cover END AS cloud_cover",
}


def release_path(name: str, release_dir: Path | None = None) -> str:
    path = (release_dir or config.RELEASE_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: gh release download {config.DATA_VERSION} --dir data/release")
    return path.as_posix()


def load_station_hours(
    split: str = "sample",
    *,
    with_calendar: bool = True,
    with_weather: bool = True,
    stations: list[str] | None = None,
    start_local: str | None = None,
    end_local: str | None = None,
    observed_only: bool = False,
    release_dir: Path | None = None,
) -> pd.DataFrame:
    """Station-hours of one split, optionally filtered by station and local date (inclusive)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    # data-v1 marks the repeated 01:00 hour of fall-back nights as 'observed', but no pickup can fall into it
    # (all starts in 01:00-01:59 are assigned to the first occurrence). The loader labels it 'dst_repeat'.
    calendar = release_path("calendar.parquet", release_dir)
    status = f"""CASE WHEN s.status = 'observed' AND s.hour_utc IN (
            SELECT hour_utc FROM read_parquet('{calendar}')
            WHERE hour_local = 1 AND local_hours_in_day = 25 AND utc_offset_h = -5)
        THEN 'dst_repeat' ELSE s.status END"""
    select = ["s.* EXCLUDE (status)", f"{status} AS status"]
    joins = []
    if with_calendar:
        select += [f"c.{col}" for col in CALENDAR_COLUMNS]
        joins.append(f"JOIN read_parquet('{release_path('calendar.parquet', release_dir)}') c USING (hour_utc)")
    if with_weather:
        select += [WEATHER_SQL[col] for col in WEATHER_COLUMNS]
        joins.append(f"LEFT JOIN read_parquet('{release_path('weather_forecast.parquet', release_dir)}') w USING (hour_utc)")
    where, params = [], []
    if stations is not None:
        where.append("s.station_id IN (SELECT unnest(?))")
        params.append(list(stations))
    if start_local is not None:
        where.append("s.date_local >= ?::DATE")
        params.append(start_local)
    if end_local is not None:
        where.append("s.date_local <= ?::DATE")
        params.append(end_local)
    if observed_only:
        where.append(f"({status}) = 'observed'")
    query = f"""
        SELECT {', '.join(select)}
        FROM read_parquet('{release_path(SPLITS[split], release_dir)}') s
        {' '.join(joins)}
        {'WHERE ' + ' AND '.join(where) if where else ''}
        ORDER BY s.station_id, s.hour_utc
    """
    with duckdb.connect() as con:
        con.execute("SET enable_progress_bar = false")
        return con.execute(query, params).df()

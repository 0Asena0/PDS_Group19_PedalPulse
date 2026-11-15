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
# rounding artefacts of the weather models. The loader clips them so every user gets the same values.
WEATHER_SQL = {
    "temperature_2m": "w.temperature_2m",
    "precipitation": "greatest(w.precipitation, 0) AS precipitation",
    "wind_speed_10m": "greatest(w.wind_speed_10m, 0) AS wind_speed_10m",
    "cloud_cover": "least(greatest(w.cloud_cover, 0), 100) AS cloud_cover",
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
    select = ["s.*"]
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
        where.append("s.status = 'observed'")
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

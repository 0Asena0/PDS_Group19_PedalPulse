"""Hourly pickups and dropoffs per station, plus a small daily system summary."""

from pathlib import Path

from . import config
from .db import connect, copy_to_parquet as copy, sql_path
from .timeutil import grid_bounds_utc
from .trips import trips_path


def output_path(kind: str, month: str) -> Path:
    return config.PROCESSED_DIR / kind / f"{month}.parquet"


def aggregate_month(month: str, force: bool = False) -> None:
    kinds = ("pickups", "dropoffs", "daily_system")
    if all(output_path(kind, month).exists() for kind in kinds) and not force:
        print(f"skip aggregate {month} (done)")
        return
    source = trips_path(month)
    if not source.exists():
        raise FileNotFoundError(f"{source.name} missing; run the load step first")
    first_hour, last_hour = grid_bounds_utc()
    con = connect()
    trips = f"read_parquet('{sql_path(source)}')"
    copy(con, f"""
        SELECT start_station_id AS station_id, date_trunc('hour', started_utc) AS hour_utc,
               CAST(count(*) AS INTEGER) AS pickups
        FROM {trips} GROUP BY ALL ORDER BY ALL
    """, output_path("pickups", month))
    # Dropoffs are counted at the end station and end hour; ends after the study period are out of range.
    copy(con, f"""
        SELECT end_station_id AS station_id, date_trunc('hour', ended_utc) AS hour_utc,
               CAST(count(*) AS INTEGER) AS dropoffs
        FROM {trips}
        WHERE end_station_id IS NOT NULL
          AND ended_utc >= TIMESTAMP '{first_hour}' AND ended_utc < TIMESTAMP '{last_hour}' + INTERVAL 1 HOUR
        GROUP BY ALL ORDER BY ALL
    """, output_path("dropoffs", month))
    copy(con, f"""
        SELECT CAST(started_local AS DATE) AS date_local,
               count(*) AS trips,
               count(*) FILTER (WHERE rideable_type = 'electric_bike') AS ebike_trips,
               count(*) FILTER (WHERE member_casual = 'member') AS member_trips,
               count(*) FILTER (WHERE duration_s BETWEEN 60 AND 86400) AS trips_1min_24h,
               count(*) FILTER (WHERE end_station_id IS NULL) AS trips_without_end_station,
               count(DISTINCT start_station_id) AS active_start_stations
        FROM {trips} GROUP BY ALL ORDER BY ALL
    """, output_path("daily_system", month))
    con.close()
    print(f"aggregate {month}: done")


def aggregate_months(months: list[str], force: bool = False) -> None:
    for month in months:
        aggregate_month(month, force)

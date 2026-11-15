"""Complete station x UTC-hour grid for the cohort, with inactive periods marked."""

import shutil

import pandas as pd

from . import config
from .cohort import COHORT_FILE
from .db import connect, sql_path
from .stations import STATIONS_BASE

GRID_DIR = "grid"
INACTIVE_DAYS = "inactive_days.parquet"


def register_inputs(con) -> None:
    cohort = pd.read_csv(config.TRACKED_DIR / COHORT_FILE, dtype={"station_id": "string"})
    stations = pd.read_parquet(config.PROCESSED_DIR / STATIONS_BASE, columns=["station_id", "first_seen_hour_utc"])
    members = stations[stations["station_id"].isin(cohort["station_id"])]
    if len(members) != len(cohort):
        raise AssertionError("cohort stations missing from station metadata")
    con.register("members_df", members)
    con.execute("CREATE OR REPLACE TEMP TABLE members AS SELECT station_id, first_seen_hour_utc::TIMESTAMP AS first_hour FROM members_df")
    calendar = sql_path(config.PROCESSED_DIR / "calendar.parquet")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE hours AS
        SELECT hour_utc::TIMESTAMP AS hour_utc, date_local, hour_local FROM read_parquet('{calendar}')
    """)
    for kind in ("pickups", "dropoffs"):
        source = sql_path(config.PROCESSED_DIR / kind / "*.parquet")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {kind} AS
            SELECT a.station_id, a.hour_utc, CAST(sum(a.{kind}) AS INTEGER) AS {kind}
            FROM read_parquet('{source}') a JOIN members USING (station_id)
            GROUP BY ALL
        """)


def build_inactive_days(con) -> None:
    """Local days that belong to a run of >= INACTIVE_MIN_DAYS days without any pickup or dropoff."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE inactive_days AS
        WITH activity AS (
            SELECT station_id, h.date_local, sum(n) AS trips
            FROM (SELECT station_id, hour_utc, pickups AS n FROM pickups
                  UNION ALL SELECT station_id, hour_utc, dropoffs FROM dropoffs) x
            JOIN hours h USING (hour_utc)
            GROUP BY ALL
        ),
        days AS (
            SELECT f.station_id, d.date_local
            FROM (SELECT m.station_id, h.date_local AS first_date
                  FROM members m JOIN hours h ON h.hour_utc = m.first_hour) f
            JOIN (SELECT DISTINCT date_local FROM hours) d ON d.date_local >= f.first_date
        ),
        empty AS (
            SELECT d.station_id, d.date_local,
                   d.date_local - CAST(row_number() OVER (PARTITION BY d.station_id ORDER BY d.date_local) AS INTEGER) AS run_id
            FROM days d LEFT JOIN activity a USING (station_id, date_local)
            WHERE coalesce(a.trips, 0) = 0
        ),
        runs AS (
            SELECT station_id, run_id, count(*) AS run_days FROM empty GROUP BY ALL
        )
        SELECT e.station_id, e.date_local, r.run_days
        FROM empty e JOIN runs r USING (station_id, run_id)
        WHERE r.run_days >= {config.INACTIVE_MIN_DAYS}
    """)


def month_chunks(con) -> list[tuple]:
    return con.execute("""
        SELECT date_trunc('month', date_local) AS m, min(hour_utc), max(hour_utc)
        FROM hours GROUP BY 1 ORDER BY 1
    """).fetchall()


def write_chunk(con, month_start, first_hour, last_hour) -> None:
    target = config.PROCESSED_DIR / GRID_DIR / f"{month_start:%Y-%m}.parquet"
    con.execute(f"""
        COPY (
            SELECT m.station_id, h.hour_utc, h.date_local, h.hour_local,
                   coalesce(p.pickups, 0) AS pickups,
                   coalesce(d.dropoffs, 0) AS dropoffs,
                   CASE WHEN i.station_id IS NULL THEN 'observed' ELSE 'inactive' END AS status
            FROM members m
            JOIN hours h ON h.hour_utc >= m.first_hour
            LEFT JOIN pickups p ON p.station_id = m.station_id AND p.hour_utc = h.hour_utc
            LEFT JOIN dropoffs d ON d.station_id = m.station_id AND d.hour_utc = h.hour_utc
            LEFT JOIN inactive_days i ON i.station_id = m.station_id AND i.date_local = h.date_local
            WHERE h.hour_utc BETWEEN TIMESTAMP '{first_hour}' AND TIMESTAMP '{last_hour}'
            ORDER BY m.station_id, h.hour_utc
        ) TO '{sql_path(target)}' (FORMAT parquet, COMPRESSION zstd)
    """)


def build_grid(force: bool = False) -> None:
    grid_dir = config.PROCESSED_DIR / GRID_DIR
    if grid_dir.exists() and any(grid_dir.iterdir()) and not force:
        print("skip grid (done)")
        return
    shutil.rmtree(grid_dir, ignore_errors=True)
    grid_dir.mkdir(parents=True)
    con = connect()
    register_inputs(con)
    build_inactive_days(con)
    con.execute(f"COPY inactive_days TO '{sql_path(config.PROCESSED_DIR / INACTIVE_DAYS)}' (FORMAT parquet)")
    for month_start, first_hour, last_hour in month_chunks(con):
        write_chunk(con, month_start, first_hour, last_hour)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(grid_dir / '*.parquet')}')").fetchone()[0]
    con.close()
    print(f"grid: {rows:,} station-hours")

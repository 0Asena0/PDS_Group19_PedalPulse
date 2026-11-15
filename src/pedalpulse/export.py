"""Write the release files (data-v1) and their SHA-256 checksums."""

import hashlib
import json
import shutil

import pandas as pd

from . import config
from .cohort import COHORT_FILE, non_public_reason
from .db import connect, sql_path
from .grid import GRID_DIR
from .stations import NAME_CHANGES, STATIONS_BASE
from .trips import DROP_RULES, read_filter_logs

STATION_HOURS_COLUMNS = """
    station_id,
    hour_utc::TIMESTAMPTZ AS hour_utc,
    date_local,
    CAST(hour_local AS TINYINT) AS hour_local,
    pickups,
    dropoffs,
    status
"""


def release_path(name: str):
    return config.RELEASE_DIR / name


def copy_query(con, query: str, name: str) -> None:
    con.execute(f"COPY ({query}) TO '{sql_path(release_path(name))}' (FORMAT parquet, COMPRESSION zstd)")


def export_station_hours(con) -> dict:
    grid = f"read_parquet('{sql_path(config.PROCESSED_DIR / GRID_DIR / '*.parquet')}')"
    dev_end = config.DEV_END.isoformat()
    copy_query(con, f"SELECT {STATION_HOURS_COLUMNS} FROM {grid} WHERE date_local <= DATE '{dev_end}' ORDER BY station_id, hour_utc",
               "station_hours_dev.parquet")
    copy_query(con, f"SELECT {STATION_HOURS_COLUMNS} FROM {grid} WHERE date_local > DATE '{dev_end}' ORDER BY station_id, hour_utc",
               "station_hours_holdout.parquet")
    first, last = config.DEV_SAMPLE_WINDOW
    stations = sql_path(release_path("stations.parquet"))
    copy_query(con, f"""
        SELECT {STATION_HOURS_COLUMNS} FROM {grid}
        WHERE date_local BETWEEN DATE '{first}' AND DATE '{last}'
          AND station_id IN (SELECT station_id FROM read_parquet('{stations}')
                             WHERE is_cohort AND borough = '{config.DEV_SAMPLE_BOROUGH}')
        ORDER BY station_id, hour_utc
    """, "dev_sample_manhattan_2024_summer.parquet")
    counts = {
        name: con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(release_path(name))}')").fetchone()[0]
        for name in ("station_hours_dev.parquet", "station_hours_holdout.parquet")
    }
    total = con.execute(f"SELECT count(*) FROM {grid}").fetchone()[0]
    if sum(counts.values()) != total:
        raise AssertionError("dev + holdout rows do not equal the grid")
    return counts


def export_stations() -> None:
    stations = pd.read_parquet(config.PROCESSED_DIR / STATIONS_BASE)
    cohort = pd.read_csv(config.TRACKED_DIR / COHORT_FILE, dtype={"station_id": "string"})
    stations["is_cohort"] = stations["station_id"].isin(cohort["station_id"])
    stations["non_public_reason"] = non_public_reason(stations)
    columns = [
        "station_id", "name", "n_names", "lat", "lng", "coord_source", "borough", "borough_method",
        "first_seen_utc", "first_seen_hour_utc", "last_seen_utc", "pickups_total", "non_public_reason", "is_cohort",
    ]
    stations = stations[columns].copy()
    for column in ("first_seen_utc", "first_seen_hour_utc", "last_seen_utc"):
        stations[column] = pd.to_datetime(stations[column]).dt.tz_localize("UTC")
    stations.to_parquet(release_path("stations.parquet"), index=False, compression="zstd")


def export_daily_system(con) -> None:
    """Daily totals for every local date of the study period; days without any trip are rows with 0."""
    source = sql_path(config.PROCESSED_DIR / "daily_system" / "*.parquet")
    copy_query(con, f"""
        WITH days AS (
            SELECT CAST(range AS DATE) AS date_local
            FROM range(DATE '{config.STUDY_START}', DATE '{config.STUDY_END}' + INTERVAL 1 DAY, INTERVAL 1 DAY)
        ), totals AS (
            SELECT date_local, sum(trips) AS trips, sum(ebike_trips) AS ebike_trips, sum(member_trips) AS member_trips,
                   sum(trips_1min_24h) AS trips_1min_24h, sum(trips_without_end_station) AS trips_without_end_station
            FROM read_parquet('{source}') GROUP BY ALL
        )
        SELECT date_local, coalesce(trips, 0) AS trips, coalesce(ebike_trips, 0) AS ebike_trips,
               coalesce(member_trips, 0) AS member_trips, coalesce(trips_1min_24h, 0) AS trips_1min_24h,
               coalesce(trips_without_end_station, 0) AS trips_without_end_station
        FROM days LEFT JOIN totals USING (date_local) ORDER BY date_local
    """, "daily_system.parquet")


def export_filter_log() -> None:
    rows = []
    for log in read_filter_logs():
        row = {"month": log["month"], "raw_rows": log["raw_rows"], "kept_rows": log["kept_rows"]}
        row |= {rule: log["dropped"][rule] for rule in DROP_RULES}
        row |= {key: value for key, value in log.items() if key.startswith("kept_") and key != "kept_rows"}
        rows.append(row)
    pd.DataFrame(rows).to_parquet(release_path("filter_log.parquet"), index=False, compression="zstd")


def copy_processed() -> None:
    for name in ("calendar.parquet", "weather_forecast.parquet", "weather_observed.parquet"):
        shutil.copy2(config.PROCESSED_DIR / name, release_path(name))
    shutil.copy2(config.PROCESSED_DIR / NAME_CHANGES, release_path(NAME_CHANGES))
    shutil.copy2(config.TRACKED_DIR / COHORT_FILE, release_path(COHORT_FILE))


def write_checksums() -> None:
    lines = []
    for path in sorted(config.RELEASE_DIR.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (config.RELEASE_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify_release() -> None:
    """Compare the downloaded release files with SHA256SUMS; works without sha256sum (Windows)."""
    sums = config.RELEASE_DIR / "SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"{sums} not found. Run: gh release download {config.DATA_VERSION} --dir data/release")
    failed = []
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = config.RELEASE_DIR / name
        ok = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        print(f"{name}: {'OK' if ok else 'MISSING' if not path.exists() else 'FAILED'}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit(f"{len(failed)} file(s) missing or changed")


def export_release(force: bool = False) -> None:
    if (config.RELEASE_DIR / "SHA256SUMS").exists() and not force:
        print("skip export (done)")
        return
    shutil.rmtree(config.RELEASE_DIR, ignore_errors=True)
    config.RELEASE_DIR.mkdir(parents=True)
    export_stations()
    copy_processed()
    con = connect()
    counts = export_station_hours(con)
    export_daily_system(con)
    con.close()
    export_filter_log()
    write_checksums()
    print(f"export: {json.dumps(counts)}")

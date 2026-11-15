"""Load monthly trip CSVs, apply the documented cleaning rules and keep a filter log."""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import pandas as pd

from . import config
from .db import connect, copy_to_parquet, sql_path
from .download import load_log
from .months import archive_for_month, month_key, study_months
from .timeutil import local_hour_map

EXPECTED_COLUMNS = (
    "ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
    "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
)
DROP_RULES = (
    "duplicate_ride_id",
    "missing_start_station",
    "unparsable_time",
    "time_out_of_range",
    "end_before_start",
    "start_outside_study_period",
)
MAP_RANGE = (datetime(2020, 1, 1), datetime(2027, 12, 31, 23))


def station_id_sql(column: str) -> str:
    """Normalise station IDs to the 'NNNN.NN' text form.

    Some files carry float-formatted IDs ('6313.1' for '6313.10') or a trailing underscore.
    """
    cleaned = f"regexp_replace(nullif(trim({column}), ''), '_+$', '')"
    return f"CASE WHEN regexp_full_match({cleaned}, '\\d+\\.\\d') THEN {cleaned} || '0' ELSE {cleaned} END"


def trips_path(month: str) -> Path:
    return config.PROCESSED_DIR / "trips" / f"{month}.parquet"


def filter_log_path(month: str) -> Path:
    return config.PROCESSED_DIR / "filter_log" / f"{month}.json"


def is_junk(name: str) -> bool:
    path = PurePosixPath(name)
    return name.endswith("/") or "__MACOSX" in path.parts or path.name.startswith(".")


def extract_members(archive: zipfile.ZipFile, members: list[str], dest: Path) -> list[Path]:
    paths = []
    for member in members:
        target = dest / PurePosixPath(member).name
        with archive.open(member) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, 16 * 1024 * 1024)
        paths.append(target)
    return paths


def month_csv_members(archive: zipfile.ZipFile, month: str) -> list[str]:
    key = month_key(month)
    return sorted(
        name for name in archive.namelist()
        if not is_junk(name) and name.endswith(".csv") and PurePosixPath(name).name.startswith(key)
    )


def extract_month_csvs(archive_path: Path, month: str, dest: Path) -> tuple[list[Path], str]:
    """Extract the CSV parts of one month; handles nested monthly ZIPs inside a yearly ZIP."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as outer:
        nested = [
            name for name in outer.namelist()
            if not is_junk(name) and name.endswith(".zip") and PurePosixPath(name).name.startswith(month_key(month))
        ]
        if len(nested) > 1:
            raise ValueError(f"several nested archives for {month}: {nested}")
        if nested:
            inner_path = extract_members(outer, nested, dest)[0]
            with zipfile.ZipFile(inner_path) as inner:
                csvs = extract_members(inner, month_csv_members(inner, month), dest)
            inner_path.unlink()
            return csvs, nested[0]
        return extract_members(outer, month_csv_members(outer, month), dest), ""


def read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return tuple(column.strip().strip('"') for column in handle.readline().strip().split(","))


def check_schema(csvs: list[Path]) -> str:
    if not csvs:
        raise ValueError("no CSV parts found")
    headers = {read_header(path) for path in csvs}
    if len(headers) != 1 or set(next(iter(headers))) != set(EXPECTED_COLUMNS):
        raise ValueError(f"unexpected schema: {headers}")
    return hashlib.sha256(",".join(next(iter(headers))).encode()).hexdigest()[:12]


def classify_sql(files_sql: str) -> str:
    start = config.STUDY_START.isoformat()
    end = (config.STUDY_END + timedelta(days=1)).isoformat()
    return f"""
    CREATE OR REPLACE TEMP TABLE classified AS
    WITH typed AS (
        SELECT
            nullif(trim(ride_id), '') AS ride_id,
            nullif(trim(rideable_type), '') AS rideable_type,
            nullif(trim(member_casual), '') AS member_casual,
            {station_id_sql("start_station_id")} AS start_station_id,
            nullif(trim(start_station_id), '') AS start_station_id_raw,
            nullif(trim(start_station_name), '') AS start_station_name,
            {station_id_sql("end_station_id")} AS end_station_id,
            nullif(trim(end_station_id), '') AS end_station_id_raw,
            nullif(trim(end_station_name), '') AS end_station_name,
            TRY_CAST(start_lat AS DOUBLE) AS start_lat,
            TRY_CAST(start_lng AS DOUBLE) AS start_lng,
            TRY_CAST(end_lat AS DOUBLE) AS end_lat,
            TRY_CAST(end_lng AS DOUBLE) AS end_lng,
            started_at AS started_raw,
            ended_at AS ended_raw,
            TRY_CAST(started_at AS TIMESTAMP) AS started_local,
            TRY_CAST(ended_at AS TIMESTAMP) AS ended_local,
            filename
        FROM read_csv([{files_sql}], header = true, all_varchar = true, filename = true)
    ),
    mapped AS (
        SELECT t.*,
            t.started_local + (ms.utc_hour - ms.local_hour) AS started_utc,
            t.ended_local + (me.utc_hour - me.local_hour) AS ended_utc,
            ms.kind AS start_kind,
            row_number() OVER (
                PARTITION BY t.ride_id
                ORDER BY t.filename, t.started_raw, t.ended_raw, t.start_station_id, t.end_station_id
            ) AS ride_rank
        FROM typed t
        LEFT JOIN hour_map ms ON ms.local_hour = date_trunc('hour', t.started_local)
        LEFT JOIN hour_map me ON me.local_hour = date_trunc('hour', t.ended_local)
    )
    SELECT *,
        CASE
            WHEN ride_id IS NOT NULL AND ride_rank > 1 THEN 'duplicate_ride_id'
            WHEN start_station_id IS NULL THEN 'missing_start_station'
            WHEN started_local IS NULL OR ended_local IS NULL THEN 'unparsable_time'
            WHEN started_utc IS NULL OR ended_utc IS NULL THEN 'time_out_of_range'
            WHEN ended_utc < started_utc THEN 'end_before_start'
            WHEN started_local < TIMESTAMP '{start}' OR started_local >= TIMESTAMP '{end}'
                THEN 'start_outside_study_period'
        END AS drop_reason
    FROM mapped
    """


KEEP_COLUMNS = """
    ride_id, rideable_type, member_casual,
    start_station_id, start_station_name, end_station_id, end_station_name,
    start_lat, start_lng, end_lat, end_lng,
    started_local, ended_local, started_utc, ended_utc,
    CAST(epoch(ended_utc) - epoch(started_utc) AS INTEGER) AS duration_s
"""


def month_stats(con) -> dict:
    counts = dict(con.execute(
        "SELECT coalesce(drop_reason, 'kept'), count(*) FROM classified GROUP BY 1"
    ).fetchall())
    extra = con.execute("""
        SELECT
            count(*),
            count(*) FILTER (WHERE ride_id IS NULL),
            min(started_local), max(started_local),
            count(*) FILTER (WHERE drop_reason IS NULL AND end_station_id IS NULL),
            count(*) FILTER (WHERE drop_reason IS NULL AND start_kind = 'ambiguous'),
            count(*) FILTER (WHERE drop_reason IS NULL AND start_kind = 'nonexistent'),
            count(*) FILTER (WHERE drop_reason IS NULL AND epoch(ended_utc) - epoch(started_utc) < 60),
            count(*) FILTER (WHERE drop_reason IS NULL AND epoch(ended_utc) - epoch(started_utc) > 86400),
            count(*) FILTER (WHERE drop_reason IS NULL AND (start_station_id <> start_station_id_raw
                                                            OR end_station_id <> end_station_id_raw))
        FROM classified
    """).fetchone()
    return {
        "raw_rows": extra[0],
        "dropped": {rule: counts.get(rule, 0) for rule in DROP_RULES},
        "kept_rows": counts.get("kept", 0),
        "null_ride_id": extra[1],
        "min_started_local": str(extra[2]),
        "max_started_local": str(extra[3]),
        "kept_missing_end_station": extra[4],
        "kept_start_ambiguous_hour": extra[5],
        "kept_start_nonexistent_hour": extra[6],
        "kept_duration_under_1min": extra[7],
        "kept_duration_over_24h": extra[8],
        "kept_station_id_normalised": extra[9],
    }


def load_month(month: str, force: bool = False) -> None:
    target = trips_path(month)
    if target.exists() and filter_log_path(month).exists() and not force:
        print(f"skip load {month} (done)")
        return
    archive = config.RAW_DIR / archive_for_month(month)
    if not archive.exists():
        raise FileNotFoundError(f"{archive.name} missing; run the download step first")
    work = config.RAW_DIR / "_extract" / month
    shutil.rmtree(work, ignore_errors=True)
    try:
        csvs, nested = extract_month_csvs(archive, month, work)
        schema_hash = check_schema(csvs)
        con = connect()
        con.register("hour_map_df", local_hour_map(*MAP_RANGE))
        con.execute("CREATE TEMP TABLE hour_map AS SELECT * FROM hour_map_df")
        files_sql = ", ".join(f"'{sql_path(path)}'" for path in csvs)
        con.execute(classify_sql(files_sql))
        stats = month_stats(con)
        if stats["raw_rows"] != stats["kept_rows"] + sum(stats["dropped"].values()):
            raise AssertionError(f"filter log does not add up for {month}")
        copy_to_parquet(con, f"SELECT {KEEP_COLUMNS} FROM classified WHERE drop_reason IS NULL", target)
        con.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stats.update({
        "month": month,
        "archive": archive.name,
        "nested_archive": nested,
        "csv_parts": [path.name for path in csvs],
        "schema_hash": schema_hash,
    })
    filter_log_path(month).parent.mkdir(parents=True, exist_ok=True)
    filter_log_path(month).write_text(json.dumps(stats, indent=2))
    print(f"load {month}: {stats['raw_rows']:,} raw, {stats['kept_rows']:,} kept")


def read_filter_logs() -> list[dict]:
    paths = sorted((config.PROCESSED_DIR / "filter_log").glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def write_manifest() -> Path:
    downloads = load_log()
    logs = {entry["month"]: entry for entry in read_filter_logs()}
    rows = []
    for month in study_months():
        archive = archive_for_month(month)
        info = downloads.get(archive, {})
        log = logs.get(month, {})
        rows.append({
            "month": month,
            "status": "included" if log else "not_loaded",
            "url": info.get("url", ""),
            "retrieved_at": info.get("retrieved_at", ""),
            "archive_size_bytes": info.get("size_bytes"),
            "archive_sha256": info.get("sha256", ""),
            "nested_archive": log.get("nested_archive", ""),
            "csv_parts": len(log["csv_parts"]) if log else None,
            "rows": log.get("raw_rows"),
            "min_started_local": log.get("min_started_local", ""),
            "max_started_local": log.get("max_started_local", ""),
            "schema_hash": log.get("schema_hash", ""),
            "note": "",
        })
    for archive, reason in config.EXCLUDED_ARCHIVES.items():
        info = downloads.get(archive, {})
        rows.append({
            "month": f"{archive[:4]}-{archive[4:6]}",
            "status": "excluded",
            "url": info.get("url", ""),
            "archive_size_bytes": info.get("size_bytes"),
            "note": reason,
        })
    path = config.TRACKED_DIR / "manifest.csv"
    manifest = pd.DataFrame(rows).astype({"csv_parts": "Int64", "rows": "Int64", "archive_size_bytes": "Int64"})
    manifest.to_csv(path, index=False)
    return path


def load_months(months: list[str], force: bool = False) -> None:
    for month in months:
        load_month(month, force)
    write_manifest()

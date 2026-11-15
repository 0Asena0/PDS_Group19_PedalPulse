"""Frozen station cohort, chosen only from data up to the selection cutoff."""

import pandas as pd

from . import config
from .db import connect, sql_path
from .stations import STATIONS_BASE

COHORT_FILE = "cohort.csv"
COHORT_RULE = (
    "at least one pickup on >= 80% of local days 2024-07-01..2024-09-30; "
    "test, workshop and depot stations excluded"
)

# Operational locations that are not public docking stations (reviewed against all station IDs and names).
NON_PUBLIC_ID_PATTERN = r"^(SYS\d+|Shop .*|Lab .*|LA Metro Demo \d+|190 Morgan)$"
NON_PUBLIC_NAME_PATTERN = r"(?i)(?:mechanic|workshop|\bshop\b|\blab\b|depot|warehouse|loading dock|\btest\b|\bdemo\b|monolith)"


def non_public_reason(stations: pd.DataFrame) -> pd.Series:
    by_id = stations["station_id"].str.match(NON_PUBLIC_ID_PATTERN)
    by_name = stations["name"].fillna("").str.contains(NON_PUBLIC_NAME_PATTERN, regex=True)
    reason = pd.Series(pd.NA, index=stations.index, dtype="string")
    reason[by_name] = "name pattern"
    reason[by_id] = "id pattern"
    return reason


def active_days(con) -> pd.DataFrame:
    first, last = config.COHORT_WINDOW
    pickups = sql_path(config.PROCESSED_DIR / "pickups" / "*.parquet")
    calendar = sql_path(config.PROCESSED_DIR / "calendar.parquet")
    return con.execute(f"""
        SELECT p.station_id, count(DISTINCT c.date_local) AS active_days
        FROM read_parquet('{pickups}') p
        JOIN read_parquet('{calendar}') c ON c.hour_utc::TIMESTAMP = p.hour_utc
        WHERE c.date_local BETWEEN DATE '{first}' AND DATE '{last}' AND p.pickups > 0
        GROUP BY ALL
    """).df()


def cohort_table() -> tuple[pd.DataFrame, dict]:
    first, last = config.COHORT_WINDOW
    window_days = (last - first).days + 1
    stations = pd.read_parquet(config.PROCESSED_DIR / STATIONS_BASE)
    needed = [config.PROCESSED_DIR / "pickups" / f"2024-{month:02d}.parquet" for month in (7, 8, 9)]
    if not all(path.exists() for path in needed):
        raise RuntimeError("the cohort needs aggregated pickups for 2024-07..2024-09")
    con = connect()
    days = active_days(con)
    con.close()
    table = stations[["station_id", "name", "borough"]].merge(days, on="station_id", how="inner")
    table["window_days"] = window_days
    table["active_share"] = table["active_days"] / window_days
    table["excluded_reason"] = non_public_reason(table)
    public = table[table["excluded_reason"].isna()]
    counts = {f"{share:.0%}": int((public["active_share"] >= share).sum()) for share in config.COHORT_SHARES_REPORTED}
    summary = {
        "stations_with_pickup_in_window": int(len(table)),
        "excluded_non_public": table.loc[table["excluded_reason"].notna(), ["station_id", "name", "active_days"]],
        "cohort_size_by_threshold": counts,
    }
    cohort = public[public["active_share"] >= config.COHORT_SHARE].copy()
    cohort["rule"] = COHORT_RULE
    cohort["cutoff_date"] = config.SELECTION_CUTOFF.isoformat()
    cohort["data_version"] = config.DATA_VERSION
    cohort = cohort.drop(columns="excluded_reason").sort_values("station_id").reset_index(drop=True)
    cohort["active_share"] = cohort["active_share"].round(4)
    return cohort, summary


def build_cohort(force: bool = False) -> None:
    target = config.TRACKED_DIR / COHORT_FILE
    if target.exists() and not force:
        print("skip cohort (frozen file exists; use --force to rebuild)")
        return
    cohort, summary = cohort_table()
    cohort.to_csv(target, index=False)
    summary["excluded_non_public"].to_csv(config.PROCESSED_DIR / "cohort_excluded_non_public.csv", index=False)
    pd.Series(summary["cohort_size_by_threshold"]).to_json(config.PROCESSED_DIR / "cohort_thresholds.json")
    print(f"cohort: {len(cohort):,} stations; thresholds {summary['cohort_size_by_threshold']}")


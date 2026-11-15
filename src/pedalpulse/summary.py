"""Collect the figures for the data quality report into reports/data_summary.json."""

import json

import pandas as pd

from . import config
from .db import connect, sql_path
from .trips import DROP_RULES, read_filter_logs


def parquet(name: str) -> str:
    return f"read_parquet('{sql_path(config.RELEASE_DIR / name)}')"


def trip_numbers() -> dict:
    logs = pd.DataFrame(read_filter_logs())
    logs["year"] = logs["month"].str[:4]
    for rule in DROP_RULES:
        logs[rule] = logs["dropped"].map(lambda dropped: dropped[rule])
    sums = ["raw_rows", "kept_rows", *DROP_RULES, "kept_missing_end_station", "kept_duration_under_1min",
            "kept_duration_over_24h", "kept_start_ambiguous_hour", "kept_start_nonexistent_hour",
            "kept_station_id_normalised"]
    by_year = logs.groupby("year")[sums].sum()
    total = logs[sums].sum()
    kept = int(total["kept_rows"])
    outside = int(total["kept_duration_under_1min"] + total["kept_duration_over_24h"])
    return {
        "months": int(len(logs)),
        "by_year": {year: {k: int(v) for k, v in row.items()} for year, row in by_year.iterrows()},
        "total": {k: int(v) for k, v in total.items()},
        "duration_sensitivity": {
            "outside_1min_24h": outside,
            "share_outside": round(outside / kept, 5),
        },
    }


def station_numbers(con) -> dict:
    stations = pd.read_parquet(config.RELEASE_DIR / "stations.parquet")
    changes = pd.read_parquet(config.RELEASE_DIR / "station_name_changes.parquet")
    thresholds = json.loads((config.PROCESSED_DIR / "cohort_thresholds.json").read_text())
    excluded = pd.read_csv(config.PROCESSED_DIR / "cohort_excluded_non_public.csv", dtype=str)
    cohort = stations[stations["is_cohort"]]
    return {
        "station_ids_total": int(len(stations)),
        "non_public_ids": stations.loc[stations["non_public_reason"].notna(), ["station_id", "name"]].values.tolist(),
        "cohort_size_by_threshold": thresholds,
        "cohort_size": int(len(cohort)),
        "cohort_excluded_non_public": excluded[["station_id", "name"]].values.tolist(),
        "cohort_by_borough": cohort["borough"].value_counts().to_dict(),
        "cohort_coord_source": cohort["coord_source"].value_counts().to_dict(),
        "borough_method_all": stations["borough_method"].value_counts().to_dict(),
        "ids_with_several_names": int(changes.loc[changes["change_type"] == "id_with_several_names", "station_id"].nunique()),
        "names_with_several_ids": int(changes.loc[changes["change_type"] == "name_with_several_ids", "name"].nunique()),
        "cohort_first_seen_after_january_2023": int((cohort["first_seen_hour_utc"] >= pd.Timestamp("2023-02-01", tz="UTC")).sum()),
    }


def grid_numbers(con) -> dict:
    out = {}
    for split in ("dev", "holdout"):
        name = f"station_hours_{split}.parquet"
        row = con.execute(f"""
            SELECT count(*), count(DISTINCT station_id), min(hour_utc)::TIMESTAMP, max(hour_utc)::TIMESTAMP,
                   avg((status = 'inactive')::INT),
                   avg((status = 'observed' AND pickups = 0)::INT) FILTER (WHERE status = 'observed'),
                   sum(pickups), sum(dropoffs),
                   quantile_cont(pickups, 0.5) FILTER (WHERE status = 'observed'),
                   quantile_cont(pickups, 0.99) FILTER (WHERE status = 'observed'),
                   max(pickups)
            FROM {parquet(name)}
        """).fetchone()
        out[split] = {
            "rows": row[0], "stations": row[1], "first_hour_utc": str(row[2]), "last_hour_utc": str(row[3]),
            "inactive_share": round(row[4], 5), "zero_share_observed": round(row[5], 4),
            "pickups": int(row[6]), "dropoffs": int(row[7]),
            "pickups_median_observed": row[8], "pickups_p99_observed": row[9], "pickups_max": row[10],
            "file_mb": round((config.RELEASE_DIR / name).stat().st_size / 1e6, 1),
        }
    sample = "dev_sample_manhattan_2024_summer.parquet"
    rows, stations = con.execute(f"SELECT count(*), count(DISTINCT station_id) FROM {parquet(sample)}").fetchone()
    out["dev_sample"] = {"rows": rows, "stations": stations,
                         "file_mb": round((config.RELEASE_DIR / sample).stat().st_size / 1e6, 1)}
    inactive = pd.read_parquet(config.PROCESSED_DIR / "inactive_days.parquet")
    out["inactive_runs"] = {
        "stations_with_inactive_days": int(inactive["station_id"].nunique()),
        "inactive_station_days": int(len(inactive)),
    }
    return out


def weather_numbers() -> dict:
    models = pd.read_parquet(config.PROCESSED_DIR / "weather_forecast_models.parquet")
    forecast = pd.read_parquet(config.RELEASE_DIR / "weather_forecast.parquet")
    observed = pd.read_parquet(config.RELEASE_DIR / "weather_observed.parquet")
    variables = list(config.WEATHER_VARIABLES)
    models["year"] = models["hour_utc"].dt.tz_convert(config.TIMEZONE).dt.year
    forecast["year"] = forecast["hour_utc"].dt.tz_convert(config.TIMEZONE).dt.year
    coverage = models.groupby(["model", "year"])[variables].apply(lambda g: g.notna().mean().round(4))
    chosen = forecast.groupby(["year", forecast["model"].fillna("none")]).size().unstack(fill_value=0)
    errors = forecast.merge(observed, on="hour_utc", suffixes=("_fc", "_obs"))
    mae = {v: round(float((errors[f"{v}_fc"] - errors[f"{v}_obs"]).abs().mean()), 2) for v in variables}
    return {
        "model_coverage": {f"{m}|{y}": row.to_dict() for (m, y), row in coverage.iterrows()},
        "chosen_model_hours_by_year": {str(y): row.to_dict() for y, row in chosen.iterrows()},
        "forecast_hours_missing": int(forecast[variables].isna().any(axis=1).sum()),
        "observed_hours_missing": int(observed[variables].isna().any(axis=1).sum()),
        "forecast_vs_era5_mae": mae,
    }


def edge_numbers(con) -> dict:
    first = config.PROCESSED_DIR / "filter_log" / "2023-01.json"
    early_dropoffs = con.execute(f"""
        SELECT sum(dropoffs) FILTER (WHERE hour_utc < TIMESTAMPTZ '2023-01-01 06:00:00+00'),
               sum(dropoffs) FILTER (WHERE hour_utc >= TIMESTAMPTZ '2023-01-08 05:00:00+00'
                                      AND hour_utc < TIMESTAMPTZ '2023-01-08 06:00:00+00')
        FROM {parquet('station_hours_dev.parquet')}
    """).fetchone()
    return {
        "trips_2023_01_file_starting_2022": json.loads(first.read_text())["dropped"]["start_outside_study_period"],
        "cohort_dropoffs_first_hour_2023": early_dropoffs[0],
        "cohort_dropoffs_same_hour_one_week_later": early_dropoffs[1],
    }


def write_summary() -> None:
    con = connect()
    summary = {
        "data_version": config.DATA_VERSION,
        "trips": trip_numbers(),
        "stations": station_numbers(con),
        "grid": grid_numbers(con),
        "weather": weather_numbers(),
        "edges": edge_numbers(con),
    }
    con.close()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.REPORTS_DIR / "data_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary written to {path.relative_to(config.REPO_ROOT)}")

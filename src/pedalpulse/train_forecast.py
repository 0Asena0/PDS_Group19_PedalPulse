cat << 'EOF' > src/pedalpulse/train_forecast.py
import argparse
import gc
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

REPO_OWNER = "0Asena0"
REPO_NAME = "PDS_Group19_PedalPulse"
RELEASE_TAG = "data-v1"
RELEASE_BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

EPOCH_DATE = pd.Timestamp("2023-01-01")
EPOCH_UTC = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
N_DAYS = 1210
N_HOURS_GRID = 29000

FEATURE_COLS = [
    "hour_local",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "lat",
    "lng",
    "borough_code",
    "station_age_days",
    "lag_1d_pickups",
    "lag_2d_pickups",
    "lag_7d_pickups",
    "lag_14d_pickups",
    "lag_1d_dropoffs",
    "prev_day_station_pickups",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]


def ensure_release_file(filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = PROCESSED_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{RELEASE_BASE_URL}/{filename}"
    print(f"Lade {filename} aus Release {RELEASE_TAG} herunter...")
    urlretrieve(url, target)
    return target


def load_lookup_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], list[str]]:
    cal_path = ensure_release_file("calendar.parquet")
    weather_path = ensure_release_file("weather_forecast.parquet")
    stations_path = ensure_release_file("stations.parquet")

    cal = pd.read_parquet(cal_path)[
        ["hour_utc", "weekday", "month", "is_weekend", "is_holiday", "origin_utc"]
    ].copy()
    cal["hour_utc"] = pd.to_datetime(cal["hour_utc"], utc=True)
    cal["origin_utc"] = pd.to_datetime(cal["origin_utc"], utc=True)
    cal["hour_idx"] = ((cal["hour_utc"] - EPOCH_UTC).dt.total_seconds() // 3600).astype(np.int32)
    cal["origin_hour_idx"] = ((cal["origin_utc"] - EPOCH_UTC).dt.total_seconds() // 3600).astype(np.int32)

    weather = pd.read_parquet(weather_path)[
        [
            "hour_utc",
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover",
            "available_at_utc",
        ]
    ].copy()
    weather["hour_utc"] = pd.to_datetime(weather["hour_utc"], utc=True)
    weather["available_at_utc"] = pd.to_datetime(weather["available_at_utc"], utc=True)
    weather["cloud_cover"] = weather["cloud_cover"].clip(lower=0.0, upper=100.0)
    weather["precipitation"] = weather["precipitation"].clip(lower=0.0)

    cw = cal.merge(weather, on="hour_utc", how="left")
    leak_mask = cw["available_at_utc"] > cw["origin_utc"]
    if leak_mask.any():
        cw.loc[leak_mask, ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]] = np.nan

    hour_lookup = {
        "weekday": np.zeros(N_HOURS_GRID, dtype=np.float32),
        "month": np.zeros(N_HOURS_GRID, dtype=np.float32),
        "is_weekend": np.zeros(N_HOURS_GRID, dtype=np.float32),
        "is_holiday": np.zeros(N_HOURS_GRID, dtype=np.float32),
        "temperature_2m": np.full(N_HOURS_GRID, np.nan, dtype=np.float32),
        "precipitation": np.full(N_HOURS_GRID, np.nan, dtype=np.float32),
        "wind_speed_10m": np.full(N_HOURS_GRID, np.nan, dtype=np.float32),
        "cloud_cover": np.full(N_HOURS_GRID, np.nan, dtype=np.float32),
        "origin_hour_idx": np.zeros(N_HOURS_GRID, dtype=np.int32),
    }

    h_idx = cw["hour_idx"].to_numpy()
    for col in [
        "weekday",
        "month",
        "is_weekend",
        "is_holiday",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "cloud_cover",
    ]:
        hour_lookup[col][h_idx] = cw[col].to_numpy(dtype=np.float32)
    hour_lookup["origin_hour_idx"][h_idx] = cw["origin_hour_idx"].to_numpy(dtype=np.int32)

    stations = pd.read_parquet(stations_path)[
        ["station_id", "lat", "lng", "borough", "first_seen_hour_utc"]
    ].copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.drop_duplicates(subset=["station_id"]).reset_index(drop=True)

    idx_to_station = stations["station_id"].tolist()
    station_to_idx = {sid: idx for idx, sid in enumerate(idx_to_station)}
    first_seen_utc = pd.to_datetime(stations["first_seen_hour_utc"], utc=True)

    station_lookup = {
        "n_stations": len(idx_to_station),
        "lat": stations["lat"].to_numpy(dtype=np.float32),
        "lng": stations["lng"].to_numpy(dtype=np.float32),
        "borough_code": stations["borough"].fillna("Unknown").astype("category").cat.codes.to_numpy(dtype=np.float32),
        "first_seen_hour_idx": ((first_seen_utc - EPOCH_UTC).dt.total_seconds() // 3600).to_numpy(dtype=np.float32),
    }

    return hour_lookup, station_lookup, station_to_idx, idx_to_station


def stream_parquet_into_grids(
    filename: str,
    station_to_idx: dict[str, int],
    pickups_grid: np.ndarray,
    dropoffs_grid: np.ndarray,
    min_day_idx: int,
    keep_rows: bool,
    only_observed: bool = False,
) -> dict[str, np.ndarray] | None:
    path = ensure_release_file(filename)
    pf = pq.ParquetFile(path)
    cols = ["station_id", "hour_utc", "date_local", "hour_local", "pickups", "dropoffs", "status"]

    chunks: dict[str, list[np.ndarray]] = {
        "station_idx": [],
        "hour_idx": [],
        "day_idx": [],
        "hour_local": [],
        "pickups": [],
        "is_observed": [],
    }

    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        d_idx = (pd.to_datetime(df["date_local"]) - EPOCH_DATE).dt.days.to_numpy(dtype=np.int16)

        date_mask = d_idx >= min_day_idx
        if not date_mask.any():
            continue

        df = df[date_mask]
        d_idx = d_idx[date_mask]

        s_idx = df["station_id"].astype(str).map(station_to_idx)
        valid_mask = s_idx.notna().to_numpy()
        if not valid_mask.any():
            continue

        df = df[valid_mask]
        d_idx = d_idx[valid_mask]
        s_arr = s_idx[valid_mask].to_numpy(dtype=np.int16)
        h_loc = df["hour_local"].to_numpy(dtype=np.int8)
        p_arr = df["pickups"].to_numpy(dtype=np.float32)
        drop_arr = df["dropoffs"].to_numpy(dtype=np.float32)

        pickups_grid[s_arr, d_idx, h_loc] = p_arr
        dropoffs_grid[s_arr, d_idx, h_loc] = drop_arr

        if keep_rows:
            obs_arr = (df["status"] == "observed").to_numpy()
            keep_mask = obs_arr if only_observed else np.ones(len(df), dtype=bool)

            if keep_mask.any():
                h_utc = pd.to_datetime(df["hour_utc"], utc=True)
                h_idx = ((h_utc - EPOCH_UTC).dt.total_seconds() // 3600).to_numpy(dtype=np.int32)
                chunks["station_idx"].append(s_arr[keep_mask])
                chunks["hour_idx"].append(h_idx[keep_mask])
                chunks["day_idx"].append(d_idx[keep_mask])
                chunks["hour_local"].append(h_loc[keep_mask])
                chunks["pickups"].append(p_arr[keep_mask])
                chunks["is_observed"].append(obs_arr[keep_mask])

    gc.collect()
    if not keep_rows:
        return None
    return {k: np.concatenate(v) for k, v in chunks.items()}


def build_feature_matrix(
    rows: dict[str, np.ndarray],
    pickups_grid: np.ndarray,
    dropoffs_grid: np.ndarray,
    daily_pickups: np.ndarray,
    hour_lookup: dict[str, np.ndarray],
    station_lookup: dict[str, np.ndarray],
) -> np.ndarray:
    n = len(rows["station_idx"])
    X = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)

    s = rows["station_idx"]
    h_idx = rows["hour_idx"]
    d = rows["day_idx"]
    hl = rows["hour_local"]

    X[:, 0] = hl.astype(np.float32)
    X[:, 1] = hour_lookup["weekday"][h_idx]
    X[:, 2] = hour_lookup["month"][h_idx]
    X[:, 3] = hour_lookup["is_weekend"][h_idx]
    X[:, 4] = hour_lookup["is_holiday"][h_idx]
    X[:, 5] = station_lookup["lat"][s]
    X[:, 6] = station_lookup["lng"][s]
    X[:, 7] = station_lookup["borough_code"][s]
    X[:, 8] = np.maximum(0.0, (h_idx.astype(np.float32) - station_lookup["first_seen_hour_idx"][s]) / 24.0)

    X[:, 9] = pickups_grid[s, np.maximum(0, d - 1), hl]
    X[:, 10] = pickups_grid[s, np.maximum(0, d - 2), hl]
    X[:, 11] = pickups_grid[s, np.maximum(0, d - 7), hl]
    X[:, 12] = pickups_grid[s, np.maximum(0, d - 14), hl]
    X[:, 13] = dropoffs_grid[s, np.maximum(0, d - 1), hl]
    X[:, 14] = daily_pickups[s, np.maximum(0, d - 1)]

    X[:, 15] = hour_lookup["temperature_2m"][h_idx]
    X[:, 16] = hour_lookup["precipitation"][h_idx]
    X[:, 17] = hour_lookup["wind_speed_10m"][h_idx]
    X[:, 18] = hour_lookup["cloud_cover"][h_idx]
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="PedalPulse Station 10: Modelltraining & Prognose")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Schnelllauf auf dem Manhattan-Sommersample statt vollem dev-Datensatz",
    )
    parser.add_argument(
        "--export-days",
        type=int,
        default=31,
        help="Anzahl der letzten Holdout-Tage für das Dashboard-Parquet (Standard: 31)",
    )
    args = parser.parse_args()

    hour_lookup, station_lookup, station_to_idx, idx_to_station = load_lookup_arrays()
    n_stations = station_lookup["n_stations"]

    pickups_grid = np.full((n_stations, N_DAYS, 24), np.nan, dtype=np.float32)
    dropoffs_grid = np.full((n_stations, N_DAYS, 24), np.nan, dtype=np.float32)

    if args.quick:
        print("Starte Schnelllauf (--quick) auf dev_sample_manhattan_2024_summer.parquet...")
        dev_rows = stream_parquet_into_grids(
            "dev_sample_manhattan_2024_summer.parquet",
            station_to_idx,
            pickups_grid,
            dropoffs_grid,
            min_day_idx=0,
            keep_rows=True,
            only_observed=True,
        )
        data_mode = "holdout_from_quick_dev"
        model_version = "hist_gbdt_quick_v1"
    else:
        min_dev_day = int((pd.Timestamp("2024-09-15") - EPOCH_DATE).days)
        print("Lade dev-Trainingsdaten (ab 2024-09-15) speichersparend in Batches...")
        dev_rows = stream_parquet_into_grids(
            "station_hours_dev.parquet",
            station_to_idx,
            pickups_grid,
            dropoffs_grid,
            min_day_idx=min_dev_day,
            keep_rows=True,
            only_observed=True,
        )
        data_mode = "holdout_backtest"
        model_version = "hist_gbdt_v1"

    assert dev_rows is not None
    daily_pickups = np.nansum(pickups_grid, axis=2).astype(np.float32)

    min_train_day = int(dev_rows["day_idx"].min()) + 14
    valid_train = dev_rows["day_idx"] >= min_train_day
    for k in dev_rows:
        dev_rows[k] = dev_rows[k][valid_train]

    max_train_rows = 5_000_000
    if len(dev_rows["station_idx"]) > max_train_rows:
        rng = np.random.default_rng(42)
        sub_idx = np.sort(rng.choice(len(dev_rows["station_idx"]), size=max_train_rows, replace=False))
        for k in dev_rows:
            dev_rows[k] = dev_rows[k][sub_idx]

    print(f"Erzeuge Trainingsmatrix für {len(dev_rows['station_idx']):,} beobachtete Stunden...")
    X_train = build_feature_matrix(
        dev_rows, pickups_grid, dropoffs_grid, daily_pickups, hour_lookup, station_lookup
    )
    y_train = np.clip(dev_rows["pickups"], 0.0, None)
    del dev_rows
    gc.collect()

    print("Trainiere HistGradientBoostingRegressor (Poisson-Loss)...")
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    print("Lade Holdout-Testzeitraum (2025-10-01 bis 2026-03-31)...")
    holdout_rows = stream_parquet_into_grids(
        "station_hours_holdout.parquet",
        station_to_idx,
        pickups_grid,
        dropoffs_grid,
        min_day_idx=0,
        keep_rows=True,
        only_observed=False,
    )
    assert holdout_rows is not None
    daily_pickups = np.nansum(pickups_grid, axis=2).astype(np.float32)

    n_holdout = len(holdout_rows["station_idx"])
    preds = np.empty(n_holdout, dtype=np.float32)
    baseline = np.empty(n_holdout, dtype=np.float32)

    chunk_size = 1_000_000
    for start in range(0, n_holdout, chunk_size):
        end = min(start + chunk_size, n_holdout)
        sub_rows = {k: v[start:end] for k, v in holdout_rows.items()}
        X_chunk = build_feature_matrix(
            sub_rows, pickups_grid, dropoffs_grid, daily_pickups, hour_lookup, station_lookup
        )
        preds[start:end] = np.clip(model.predict(X_chunk), 0.0, None)
        b_chunk = np.where(np.isnan(X_chunk[:, 11]), X_chunk[:, 9], X_chunk[:, 11])
        baseline[start:end] = np.clip(np.nan_to_num(b_chunk, nan=0.0), 0.0, None)
        del X_chunk

    del pickups_grid, dropoffs_grid, daily_pickups
    gc.collect()

    obs_mask = holdout_rows["is_observed"]
    y_true_obs = holdout_rows["pickups"][obs_mask].astype(np.float64)
    y_pred_obs = preds[obs_mask].astype(np.float64)
    y_base_obs = baseline[obs_mask].astype(np.float64)

    baseline_metrics = {
        "rows_observed": int(obs_mask.sum()),
        "mae": float(mean_absolute_error(y_true_obs, y_base_obs)),
        "rmse": float(root_mean_squared_error(y_true_obs, y_base_obs)),
        "bias": float(np.mean(y_base_obs - y_true_obs)),
    }
    model_metrics = {
        "rows_observed": int(obs_mask.sum()),
        "mae": float(mean_absolute_error(y_true_obs, y_pred_obs)),
        "rmse": float(root_mean_squared_error(y_true_obs, y_pred_obs)),
        "bias": float(np.mean(y_pred_obs - y_true_obs)),
    }

    print("\n=== Holdout-Auswertung (2025-10-01 bis 2026-03-31, status == observed) ===")
    print(
        f"Baseline (7-Tage-Lag): MAE = {baseline_metrics['mae']:.3f} | "
        f"RMSE = {baseline_metrics['rmse']:.3f} | Bias = {baseline_metrics['bias']:+.3f}"
    )
    print(
        f"ML-Modell ({model_version}): MAE = {model_metrics['mae']:.3f} | "
        f"RMSE = {model_metrics['rmse']:.3f} | Bias = {model_metrics['bias']:+.3f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_version": model_version,
        "data_mode": data_mode,
        "features": FEATURE_COLS,
        "baseline_weekly_lag": baseline_metrics,
        "model": model_metrics,
    }
    (REPORTS_DIR / "holdout_metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    max_day = int(holdout_rows["day_idx"].max())
    export_start_day = max_day - (args.export_days - 1)
    exp_mask = holdout_rows["day_idx"] >= export_start_day

    station_arr = np.array(idx_to_station, dtype=object)
    exp_station_ids = station_arr[holdout_rows["station_idx"][exp_mask]]
    exp_h_idx = holdout_rows["hour_idx"][exp_mask]

    target_utc = EPOCH_UTC + pd.to_timedelta(exp_h_idx, unit="h")
    origin_utc = EPOCH_UTC + pd.to_timedelta(hour_lookup["origin_hour_idx"][exp_h_idx], unit="h")

    preds_out = pd.DataFrame(
        {
            "station_id": exp_station_ids,
            "origin_utc": origin_utc,
            "target_start_utc": target_utc,
            "prediction": preds[exp_mask],
            "baseline_prediction": baseline[exp_mask],
            "model_version": model_version,
            "data_mode": data_mode,
        }
    )

    exp_pickups = holdout_rows["pickups"][exp_mask].copy()
    exp_obs = holdout_rows["is_observed"][exp_mask]
    exp_pickups[~exp_obs] = np.nan
    exp_status = np.where(exp_obs, "observed", "missing")

    actuals_out = pd.DataFrame(
        {
            "station_id": exp_station_ids,
            "target_start_utc": target_utc,
            "pickups": exp_pickups,
            "status": exp_status,
        }
    )

    preds_path = PROCESSED_DIR / "predictions.parquet"
    actuals_path = PROCESSED_DIR / "actuals.parquet"
    preds_out.to_parquet(preds_path, index=False)
    actuals_out.to_parquet(actuals_path, index=False)

    start_date_str = (EPOCH_DATE + pd.Timedelta(days=export_start_day)).date()
    end_date_str = (EPOCH_DATE + pd.Timedelta(days=max_day)).date()
    print(
        f"\nGespeichert ({start_date_str} bis {end_date_str}): "
        f"{len(preds_out):,} Prognosezeilen -> {preds_path}"
    )


if __name__ == "__main__":
    main()
EOF
python src/pedalpulse/train_forecast.py

import argparse
import gc
import json
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

REPO_OWNER = "0Asena0"
REPO_NAME = "PDS_Group19_PedalPulse"
RELEASE_TAG = "data-v1"
RELEASE_BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
EPOCH_DATE = pd.Timestamp("2023-01-01")

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


def load_supporting_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    cal_path = ensure_release_file("calendar.parquet")
    weather_path = ensure_release_file("weather_forecast.parquet")
    stations_path = ensure_release_file("stations.parquet")

    cal = pd.read_parquet(cal_path)[
        ["hour_utc", "weekday", "month", "is_weekend", "is_holiday", "origin_utc"]
    ].copy()
    cal["hour_utc"] = pd.to_datetime(cal["hour_utc"], utc=True)
    cal["origin_utc"] = pd.to_datetime(cal["origin_utc"], utc=True)
    cal["weekday"] = cal["weekday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["is_weekend"] = cal["is_weekend"].astype("int8")
    cal["is_holiday"] = cal["is_holiday"].astype("int8")

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
    # Laut data_contract.md Rundungsartefakte begrenzen und kompakt speichern
    weather["cloud_cover"] = weather["cloud_cover"].clip(lower=0.0, upper=100.0).astype("float32")
    weather["precipitation"] = weather["precipitation"].clip(lower=0.0).astype("float32")
    weather["temperature_2m"] = weather["temperature_2m"].astype("float32")
    weather["wind_speed_10m"] = weather["wind_speed_10m"].astype("float32")

    stations = pd.read_parquet(stations_path)[
        ["station_id", "lat", "lng", "borough", "first_seen_hour_utc"]
    ].copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.drop_duplicates(subset=["station_id"]).reset_index(drop=True)

    idx_to_station = stations["station_id"].tolist()
    station_to_idx = {sid: idx for idx, sid in enumerate(idx_to_station)}
    stations["station_idx"] = np.arange(len(stations), dtype="int16")
    stations["lat"] = stations["lat"].astype("float32")
    stations["lng"] = stations["lng"].astype("float32")
    stations["first_seen_hour_utc"] = pd.to_datetime(stations["first_seen_hour_utc"], utc=True)
    stations["borough_code"] = (
        stations["borough"].fillna("Unknown").astype("category").cat.codes.astype("int16")
    )

    return cal, weather, stations, station_to_idx, idx_to_station


def load_station_hours(
    filename: str,
    station_to_idx: dict[str, int],
    min_date: date | None = None,
) -> pd.DataFrame:
    path = ensure_release_file(filename)
    cols = ["station_id", "hour_utc", "date_local", "hour_local", "pickups", "dropoffs", "status"]
    if min_date is not None:
        try:
            df = pd.read_parquet(path, columns=cols, filters=[("date_local", ">=", min_date)])
        except Exception:
            df = pd.read_parquet(path, columns=cols)
            df = df[pd.to_datetime(df["date_local"]).dt.date >= min_date]
    else:
        df = pd.read_parquet(path, columns=cols)

    mapped_idx = df["station_id"].astype(str).map(station_to_idx)
    valid_mask = mapped_idx.notna()
    if not valid_mask.all():
        df = df[valid_mask].copy()
        mapped_idx = mapped_idx[valid_mask]

    out = pd.DataFrame(
        {
            "station_idx": mapped_idx.astype("int16").to_numpy(),
            "hour_utc": pd.to_datetime(df["hour_utc"], utc=True).to_numpy(),
            "day_idx": (pd.to_datetime(df["date_local"]) - EPOCH_DATE).dt.days.astype("int16").to_numpy(),
            "hour_local": df["hour_local"].astype("int8").to_numpy(),
            "pickups": df["pickups"].astype("float32").to_numpy(),
            "dropoffs": df["dropoffs"].astype("float32").to_numpy(),
            "is_observed": (df["status"] == "observed").to_numpy(),
        }
    )
    del df, mapped_idx
    gc.collect()
    return out


def add_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    cal: pd.DataFrame,
    weather: pd.DataFrame,
    stations: pd.DataFrame,
    filter_for_training: bool = False,
) -> pd.DataFrame:
    df = target_df.copy()

    # Kompakte Lookup-Tabelle auf kleinen Integer-Schlüsseln (verhindert DST-Probleme & spart RAM)
    hourly_hist = history_df[["station_idx", "day_idx", "hour_local", "pickups", "dropoffs"]].drop_duplicates(
        subset=["station_idx", "day_idx", "hour_local"]
    )

    for lag_days in [1, 2, 7, 14]:
        shifted = hourly_hist[["station_idx", "day_idx", "hour_local", "pickups"]].copy()
        shifted["day_idx"] = (shifted["day_idx"] + lag_days).astype("int16")
        shifted = shifted.rename(columns={"pickups": f"lag_{lag_days}d_pickups"})
        df = df.merge(shifted, on=["station_idx", "day_idx", "hour_local"], how="left")
        del shifted
        gc.collect()

    shifted_drop = hourly_hist[["station_idx", "day_idx", "hour_local", "dropoffs"]].copy()
    shifted_drop["day_idx"] = (shifted_drop["day_idx"] + 1).astype("int16")
    shifted_drop = shifted_drop.rename(columns={"dropoffs": "lag_1d_dropoffs"})
    df = df.merge(shifted_drop, on=["station_idx", "day_idx", "hour_local"], how="left")
    del shifted_drop, hourly_hist
    gc.collect()

    # Beim Training nicht beobachtete / Warmup-Zeilen sofort entfernen, bevor weitere Joins folgen
    if filter_for_training:
        df = df[df["is_observed"] & df["lag_7d_pickups"].notna()].copy()
        gc.collect()

    # Vortages-Gesamtausleihen der Station (bis Ende D-1 bekannt)
    daily_hist = (
        history_df.groupby(["station_idx", "day_idx"], as_index=False)["pickups"]
        .sum()
        .rename(columns={"pickups": "prev_day_station_pickups"})
    )
    daily_hist["day_idx"] = (daily_hist["day_idx"] + 1).astype("int16")
    daily_hist["prev_day_station_pickups"] = daily_hist["prev_day_station_pickups"].astype("float32")
    df = df.merge(daily_hist, on=["station_idx", "day_idx"], how="left")
    del daily_hist
    gc.collect()

    # Kalender-, Wetter- und statische Stationsmerkmale anspielen
    df = df.merge(cal, on="hour_utc", how="left")
    df = df.merge(weather, on="hour_utc", how="left")
    df = df.merge(
        stations[["station_idx", "lat", "lng", "borough_code", "first_seen_hour_utc"]],
        on="station_idx",
        how="left",
    )

    # Strikte Prüfung der E1-Informationsgrenze für Wettervorhersagen
    leak_mask = df["available_at_utc"] > df["origin_utc"]
    if leak_mask.any():
        df.loc[leak_mask, ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]] = np.nan

    df["station_age_days"] = (
        ((df["hour_utc"] - df["first_seen_hour_utc"]).dt.total_seconds() / 86400.0)
        .clip(lower=0.0)
        .astype("float32")
    )
    df.drop(columns=["available_at_utc", "first_seen_hour_utc"], inplace=True)

    # Saisonale Wochen-Baseline (gleiche Stunde vor 7 Tagen, Fallback auf Vortag)
    df["baseline_prediction"] = (
        df["lag_7d_pickups"].fillna(df["lag_1d_pickups"]).fillna(0.0).clip(lower=0.0).astype("float32")
    )
    gc.collect()
    return df


def evaluate_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    obs = df[df["is_observed"]]
    y_true = obs["pickups"].to_numpy(dtype=np.float64)
    y_pred = obs[pred_col].to_numpy(dtype=np.float64)
    return {
        "rows_observed": int(len(obs)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


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

    cal, weather, stations, station_to_idx, idx_to_station = load_supporting_tables()

    if args.quick:
        print("Starte Schnelllauf (--quick) auf dev_sample_manhattan_2024_summer.parquet...")
        dev_raw = load_station_hours("dev_sample_manhattan_2024_summer.parquet", station_to_idx)
        data_mode = "holdout_from_quick_dev"
        model_version = "hist_gbdt_quick_v1"
    else:
        print("Lade dev-Trainingsdaten (ab 2024-09-15 für 12 Monate Saisonzyklus + Lags)...")
        dev_raw = load_station_hours(
            "station_hours_dev.parquet", station_to_idx, min_date=date(2024, 9, 15)
        )
        data_mode = "holdout_backtest"
        model_version = "hist_gbdt_v1"

    print("Erzeuge Trainings-Features (speichersparend, strikt bis D-1 23:59 Ortszeit)...")
    train_df = add_features(
        dev_raw, dev_raw, cal, weather, stations, filter_for_training=True
    )

    # Nur die letzten 15 Tage von dev für den Lag-Übergang in den Oktober behalten
    dev_tail = dev_raw[dev_raw["day_idx"] >= int((pd.Timestamp("2025-09-15") - EPOCH_DATE).days)].copy()
    del dev_raw
    gc.collect()

    print(f"Trainiere HistGradientBoostingRegressor auf {len(train_df):,} beobachteten Stunden...")
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = np.clip(train_df["pickups"].to_numpy(dtype=np.float32), 0.0, None)
    del train_df
    gc.collect()

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
    holdout_raw = load_station_hours("station_hours_holdout.parquet", station_to_idx)
    combined_hist = pd.concat([dev_tail, holdout_raw], ignore_index=True)
    del dev_tail
    gc.collect()

    print("Berechne Holdout-Features und Prognosen...")
    holdout_feat = add_features(
        holdout_raw, combined_hist, cal, weather, stations, filter_for_training=False
    )
    del holdout_raw, combined_hist
    gc.collect()

    X_holdout = holdout_feat[FEATURE_COLS].to_numpy(dtype=np.float32)
    holdout_feat["prediction"] = np.clip(model.predict(X_holdout), 0.0, None).astype("float32")
    del X_holdout
    gc.collect()

    baseline_metrics = evaluate_predictions(holdout_feat, "baseline_prediction")
    model_metrics = evaluate_predictions(holdout_feat, "prediction")

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

    # Letzte N Tage des Holdouts für das Dashboard kompakt exportieren
    max_day = int(holdout_feat["day_idx"].max())
    export_start_day = max_day - (args.export_days - 1)
    export_df = holdout_feat[holdout_feat["day_idx"] >= export_start_day].copy()
    del holdout_feat
    gc.collect()

    station_arr = np.array(idx_to_station, dtype=object)
    station_ids = station_arr[export_df["station_idx"].to_numpy()]

    preds_out = pd.DataFrame(
        {
            "station_id": station_ids,
            "origin_utc": export_df["origin_utc"],
            "target_start_utc": export_df["hour_utc"],
            "prediction": export_df["prediction"],
            "baseline_prediction": export_df["baseline_prediction"],
            "model_version": model_version,
            "data_mode": data_mode,
        }
    )

    actuals_pickups = export_df["pickups"].to_numpy(dtype=np.float32)
    actuals_pickups[~export_df["is_observed"].to_numpy()] = np.nan
    actuals_status = np.where(export_df["is_observed"].to_numpy(), "observed", "missing")

    actuals_out = pd.DataFrame(
        {
            "station_id": station_ids,
            "target_start_utc": export_df["hour_utc"],
            "pickups": actuals_pickups,
            "status": actuals_status,
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

import argparse
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


def load_station_hours(filename: str, min_date: date | None = None) -> pd.DataFrame:
    path = ensure_release_file(filename)
    if min_date is not None:
        try:
            df = pd.read_parquet(path, filters=[("date_local", ">=", min_date)])
        except Exception:
            df = pd.read_parquet(path)
            df = df[pd.to_datetime(df["date_local"]).dt.date >= min_date]
    else:
        df = pd.read_parquet(path)

    df["station_id"] = df["station_id"].astype(str)
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    df["date_local_ts"] = pd.to_datetime(df["date_local"])
    df["hour_local"] = df["hour_local"].astype("int16")
    return df


def load_supporting_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal_path = ensure_release_file("calendar.parquet")
    weather_path = ensure_release_file("weather_forecast.parquet")
    stations_path = ensure_release_file("stations.parquet")

    cal = pd.read_parquet(cal_path)[
        ["hour_utc", "weekday", "month", "is_weekend", "is_holiday", "origin_utc"]
    ].copy()
    cal["hour_utc"] = pd.to_datetime(cal["hour_utc"], utc=True)
    cal["origin_utc"] = pd.to_datetime(cal["origin_utc"], utc=True)
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
    # Laut data_contract.md Rundungsartefakte der Wettermodelle begrenzen
    weather["cloud_cover"] = weather["cloud_cover"].clip(lower=0.0, upper=100.0)
    weather["precipitation"] = weather["precipitation"].clip(lower=0.0)

    stations = pd.read_parquet(stations_path)[
        ["station_id", "lat", "lng", "borough", "first_seen_hour_utc"]
    ].copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["first_seen_hour_utc"] = pd.to_datetime(stations["first_seen_hour_utc"], utc=True)
    stations["borough_code"] = stations["borough"].fillna("Unknown").astype("category").cat.codes.astype("int16")

    return cal, weather, stations


def add_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    cal: pd.DataFrame,
    weather: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    df = target_df.copy()

    # Eindeutige Basis pro (station_id, date_local_ts, hour_local) für saubere Lags auch an DST-Tagen
    hourly_hist = (
        history_df.groupby(["station_id", "date_local_ts", "hour_local"], as_index=False)
        .agg(hist_pickups=("pickups", "first"), hist_dropoffs=("dropoffs", "first"))
    )

    for lag_days in [1, 2, 7, 14]:
        shifted = hourly_hist[["station_id", "date_local_ts", "hour_local", "hist_pickups"]].copy()
        shifted["date_local_ts"] = shifted["date_local_ts"] + pd.Timedelta(days=lag_days)
        shifted = shifted.rename(columns={"hist_pickups": f"lag_{lag_days}d_pickups"})
        df = df.merge(shifted, on=["station_id", "date_local_ts", "hour_local"], how="left")

    shifted_drop = hourly_hist[["station_id", "date_local_ts", "hour_local", "hist_dropoffs"]].copy()
    shifted_drop["date_local_ts"] = shifted_drop["date_local_ts"] + pd.Timedelta(days=1)
    shifted_drop = shifted_drop.rename(columns={"hist_dropoffs": "lag_1d_dropoffs"})
    df = df.merge(shifted_drop, on=["station_id", "date_local_ts", "hour_local"], how="left")

    # Vortages-Gesamtausleihen der Station (bis Ende D-1 bekannt)
    daily_hist = (
        history_df.groupby(["station_id", "date_local_ts"], as_index=False)["pickups"]
        .sum()
        .rename(columns={"pickups": "prev_day_station_pickups"})
    )
    daily_hist["date_local_ts"] = daily_hist["date_local_ts"] + pd.Timedelta(days=1)
    df = df.merge(daily_hist, on=["station_id", "date_local_ts"], how="left")

    # Kalender-, Wetter- und statische Stationsmerkmale anspielen
    df = df.merge(cal, on="hour_utc", how="left")
    df = df.merge(weather, on="hour_utc", how="left")
    df = df.merge(
        stations[["station_id", "lat", "lng", "borough_code", "first_seen_hour_utc"]],
        on="station_id",
        how="left",
    )

    # Strikte Prüfung der E1-Informationsgrenze für Wettervorhersagen
    leak_mask = df["available_at_utc"] > df["origin_utc"]
    if leak_mask.any():
        df.loc[leak_mask, ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]] = np.nan

    df["station_age_days"] = (
        (df["hour_utc"] - df["first_seen_hour_utc"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    # Saisonale Wochen-Baseline (gleiche Stunde vor 7 Tagen, Fallback auf Vortag)
    df["baseline_prediction"] = (
        df["lag_7d_pickups"].fillna(df["lag_1d_pickups"]).fillna(0.0).clip(lower=0.0)
    )

    return df


def evaluate_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    obs = df[df["status"] == "observed"]
    y_true = obs["pickups"].to_numpy(dtype=float)
    y_pred = obs[pred_col].to_numpy(dtype=float)
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

    cal, weather, stations = load_supporting_tables()

    if args.quick:
        print("Starte Schnelllauf (--quick) auf dev_sample_manhattan_2024_summer.parquet...")
        dev_raw = load_station_hours("dev_sample_manhattan_2024_summer.parquet")
        data_mode = "holdout_from_quick_dev"
        model_version = "hist_gbdt_quick_v1"
    else:
        print("Lade dev-Trainingsdaten (ab 2024-09-15 für 12 Monate Saisonzyklus + Lags)...")
        dev_raw = load_station_hours("station_hours_dev.parquet", min_date=date(2024, 9, 15))
        data_mode = "holdout_backtest"
        model_version = "hist_gbdt_v1"

    print("Erzeuge Trainings-Features (strikt bis D-1 23:59 Ortszeit)...")
    dev_feat = add_features(dev_raw, dev_raw, cal, weather, stations)

    # Nur beobachtete Stunden und Zeilen mit vorhandenem Wochen-Lag zum Training nutzen
    train_df = dev_feat[
        (dev_feat["status"] == "observed") & (dev_feat["lag_7d_pickups"].notna())
    ].copy()

    print(f"Trainiere HistGradientBoostingRegressor auf {len(train_df):,} beobachteten Stunden...")
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLS], train_df["pickups"].clip(lower=0))

    print("Lade Holdout-Testzeitraum (2025-10-01 bis 2026-03-31)...")
    holdout_raw = load_station_hours("station_hours_holdout.parquet")

    # Ende von dev als Lag-Kontext anhängen, damit auch die ersten Oktobertage Lags haben
    context_cutoff = pd.Timestamp("2025-09-15")
    dev_tail = dev_raw[dev_raw["date_local_ts"] >= context_cutoff]
    combined_hist = pd.concat([dev_tail, holdout_raw], ignore_index=True)

    print("Berechne Holdout-Features und Prognosen...")
    holdout_feat = add_features(holdout_raw, combined_hist, cal, weather, stations)
    holdout_feat["prediction"] = np.clip(model.predict(holdout_feat[FEATURE_COLS]), 0.0, None)

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
    max_date = holdout_feat["date_local_ts"].max()
    export_start = max_date - pd.Timedelta(days=args.export_days - 1)
    export_df = holdout_feat[holdout_feat["date_local_ts"] >= export_start].copy()

    preds_out = export_df[
        [
            "station_id",
            "origin_utc",
            "hour_utc",
            "prediction",
            "baseline_prediction",
        ]
    ].rename(columns={"hour_utc": "target_start_utc"})
    preds_out["model_version"] = model_version
    preds_out["data_mode"] = data_mode

    actuals_out = export_df[["station_id", "hour_utc", "pickups", "status"]].rename(
        columns={"hour_utc": "target_start_utc"}
    )
    # Inaktive Läufe im Dashboard als missing markieren, damit sie die Fehlerstatistik nicht verzerren
    actuals_out.loc[actuals_out["status"] != "observed", "pickups"] = np.nan
    actuals_out["status"] = actuals_out["status"].replace({"inactive": "missing"})

    preds_path = PROCESSED_DIR / "predictions.parquet"
    actuals_path = PROCESSED_DIR / "actuals.parquet"
    preds_out.to_parquet(preds_path, index=False)
    actuals_out.to_parquet(actuals_path, index=False)

    print(
        f"\nGespeichert ({export_start.date()} bis {max_date.date()}): "
        f"{len(preds_out):,} Prognosezeilen -> {preds_path}"
    )


if __name__ == "__main__":
    main()

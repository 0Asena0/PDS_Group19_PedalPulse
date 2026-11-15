from pathlib import Path
import numpy as np
import pandas as pd


def load_cohort(cohort_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)

    # Spaltennamen vereinheitlichen, falls noch leicht abweichend benannt
    rename_map = {}
    if "start_station_id" in df.columns and "station_id" not in df.columns:
        rename_map["start_station_id"] = "station_id"
    if "start_station_name" in df.columns and "station_name" not in df.columns:
        rename_map["start_station_name"] = "station_name"
    if "latitude" in df.columns and "lat" not in df.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in df.columns and "lon" not in df.columns:
        rename_map["longitude"] = "lon"

    df = df.rename(columns=rename_map)
    df["station_id"] = df["station_id"].astype(str)
    return df


def base_demand_profile(hour: int, is_weekend: bool) -> float:
    if is_weekend:
        return 2.5 + 9.0 * np.exp(-0.5 * ((hour - 14.5) / 3.5) ** 2)
    morning_peak = 12.0 * np.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
    evening_peak = 15.0 * np.exp(-0.5 * ((hour - 17.5) / 2.0) ** 2)
    return 1.8 + morning_peak + evening_peak


def generate_dummy_forecast(
    cohort_path: Path = Path("data/cohort.csv"),
    out_dir: Path = Path("data/processed"),
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = load_cohort(cohort_path)
    station_ids = stations["station_id"].tolist()

    # Jede Station bekommt einen eigenen Grundfaktor
    station_scale = {
        sid: float(rng.uniform(0.5, 1.8)) for sid in station_ids
    }

    # 3 Tage im Validierungsfenster; der letzte Tag bleibt ohne Ist-Werte ("Zukunft")
    local_days = ["2025-07-14", "2025-07-15", "2025-07-16"]
    pred_records = []
    actual_records = []

    for day_idx, day_str in enumerate(local_days):
        day_start = pd.Timestamp(day_str, tz="America/New_York")
        day_end = day_start + pd.Timedelta(days=1)
        origin_utc = (day_start - pd.Timedelta(hours=1)).tz_convert("UTC")

        utc_hours = pd.date_range(
            day_start.tz_convert("UTC"),
            day_end.tz_convert("UTC"),
            freq="1h",
            inclusive="left",
        )

        for target_utc in utc_hours:
            target_local = target_utc.tz_convert("America/New_York")
            is_weekend = target_local.weekday() >= 5
            base_rate = base_demand_profile(target_local.hour, is_weekend)

            for sid in station_ids:
                mu = base_rate * station_scale[sid]
                pred_val = max(0.0, mu + rng.normal(0.0, 1.1))

                pred_records.append(
                    {
                        "station_id": sid,
                        "origin_utc": origin_utc,
                        "target_start_utc": target_utc,
                        "prediction": round(float(pred_val), 2),
                        "model_version": "dummy_v0",
                        "data_mode": "dummy_synthetic",
                    }
                )

                if day_idx < 2:
                    if rng.random() < 0.03:
                        actual_records.append(
                            {
                                "station_id": sid,
                                "target_start_utc": target_utc,
                                "pickups": np.nan,
                                "status": "missing",
                            }
                        )
                    else:
                        actual_records.append(
                            {
                                "station_id": sid,
                                "target_start_utc": target_utc,
                                "pickups": int(rng.poisson(max(0.2, mu))),
                                "status": "observed",
                            }
                        )

    preds_df = pd.DataFrame(pred_records)
    actuals_df = pd.DataFrame(actual_records)

    preds_df.to_parquet(out_dir / "dummy_predictions.parquet", index=False)
    actuals_df.to_parquet(out_dir / "dummy_actuals.parquet", index=False)

    print(f"Cohort stations: {len(stations)}")
    print(f"Saved {len(preds_df)} prediction rows and {len(actuals_df)} actual rows.")


if __name__ == "__main__":
    generate_dummy_forecast()

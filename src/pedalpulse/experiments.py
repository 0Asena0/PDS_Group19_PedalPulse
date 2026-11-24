"""Two experiments on the four dev folds, LightGBM with its default settings, 200 stations.

- origin: the data contract's origin (00:00 on the target day) against the roadmap's rule (23:00 the day
  before, plus one hour latency). The stricter rule loses the last hours of the day before.
- weather: with and without the forecast weather.

Usage: uv run python -m pedalpulse.experiments
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .dataset import load_station_hours, release_path
from .evaluate import PREDICTIONS_DIR, RESULTS_DIR, dev_stations
from .features import WEATHER_FEATURES, build_features
from .folds import FOLDS, split
from .metrics import scores
from .models import LightGBM

VARIANTS = {
    "contract": dict(origin_shift_h=0, latency_h=0, weather=True),
    "roadmap_origin": dict(origin_shift_h=-1, latency_h=1, weather=True),
    "no_weather": dict(origin_shift_h=0, latency_h=0, weather=False),
}


def load(stations: list[str], origin_shift_h: int, latency_h: int, weather: bool) -> pd.DataFrame:
    df = build_features(load_station_hours("dev", stations=stations), pd.read_parquet(release_path("stations.parquet")),
                        origin_shift_h=origin_shift_h, latency_h=latency_h)
    if not weather:
        # A column that is always missing gives LightGBM nothing to split on, so this is the same as leaving
        # the weather out, and the model code stays the same for all variants.
        df[WEATHER_FEATURES] = np.nan
    return df


def main() -> None:
    out = RESULTS_DIR / "experiments_lightgbm_200.csv"
    # I keep finished runs, so an interrupted run continues where it stopped.
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["variant"], r["fold"]) for r in rows}
    stations = dev_stations(200)
    for name, variant in VARIANTS.items():
        if all((name, f.name) in done for f in FOLDS):
            continue
        data = load(stations, **variant)
        for fold in FOLDS:
            if (name, fold.name) in done:
                continue
            start = time.time()
            train, valid = split(data, fold)
            valid = valid[valid["status"] == "observed"]
            pred = LightGBM().fit(train).predict(valid)
            rows.append({"variant": name, "fold": fold.name, **scores(valid["pickups"], pred),
                         "seconds": round(time.time() - start, 1)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            if name == "contract":
                # The forecasts of the final setup, kept for the error analysis.
                PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
                valid[["station_id", "hour_utc"]].assign(prediction=pred).to_parquet(
                    PREDICTIONS_DIR / f"lightgbm-final_validation_{fold.name}_200.parquet", index=False)
    table = pd.DataFrame(rows)
    print(table.pivot_table(index="variant", columns="fold", values="wape").round(4).to_string())


if __name__ == "__main__":
    main()

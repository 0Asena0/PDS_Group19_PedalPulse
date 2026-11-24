"""The final test: LightGBM refit on all dev data, one forecast for every holdout hour, scored once.

Usage (only after the team review, when the method is frozen):
    uv run python -m pedalpulse.final_test --unlock

Steps
1. Holdout features. The first holdout days need lags from the last 28 days of dev, so every chunk of stations
   is built from the last HISTORY_DAYS of dev plus the holdout, and then only the holdout rows are kept.
2. Training on all dev rows of all cohort stations (feature store, month by month, as in evaluate.py --full).
3. Forecasts for every holdout station-hour (the dashboard needs inactive hours too), scores on observed
   hours only, like in the folds.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import pandas as pd

from . import config, feature_store
from .dataset import load_station_hours, release_path
from .evaluate import MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR, fit_full
from .features import HOUR_LAG_DAYS, build_features
from .folds import TEST, holdout_split
from .metrics import scores

# The longest lag is 28 days; one extra week so the rolling means at the start are complete too.
HISTORY_DAYS = max(HOUR_LAG_DAYS) + 7


def features_with_history(history: pd.DataFrame, later: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Features for the rows of `later`, with lags reaching back into `history`.

    Building `later` on its own would leave the lags of its first four weeks empty, although that history
    exists and is known at the origin.
    """
    start = pd.to_datetime(later["date_local"]).min()
    both = build_features(pd.concat([history, later], ignore_index=True), stations)
    return both[pd.to_datetime(both["date_local"]) >= start].reset_index(drop=True)


def build_holdout_features(chunk: int = 150) -> None:
    cohort = pd.read_csv(release_path("cohort.csv"), dtype={"station_id": str})["station_id"].sort_values().tolist()
    station_table = pd.read_parquet(release_path("stations.parquet"))
    out_dir = feature_store.FEATURE_DIR / f"holdout_{config.DATA_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)
    first = str(config.DEV_END - timedelta(days=HISTORY_DAYS - 1))
    for i in range(0, len(cohort), chunk):
        part = out_dir / f"part-{i // chunk:03d}.parquet"
        if part.exists():
            continue
        ids = cohort[i:i + chunk]
        history = load_station_hours("dev", stations=ids, start_local=first)
        frame = features_with_history(history, load_station_hours("holdout", stations=ids), station_table)
        floats = frame.select_dtypes("float64").columns
        frame[floats] = frame[floats].astype("float32")
        frame.to_parquet(part, index=False, compression="zstd")
        print(f"holdout {part.name}: {len(frame):,} rows", flush=True)


def run(unlock: bool) -> dict:
    # holdout_split is the lock: without unlock=True it stops here, before anything from the holdout is read.
    holdout_split(pd.DataFrame({"date_local": []}), pd.DataFrame({"date_local": []}), unlock=unlock)
    start = time.time()
    build_holdout_features()
    model = fit_full(TEST.train_start, TEST.train_end)
    model.save(MODELS_DIR / "lightgbm-v1_final")
    test = feature_store.load(["*"], split="holdout")
    holdout_split(pd.DataFrame({"date_local": []}), test, unlock=unlock)  # checks the dates of the test rows
    test["prediction"] = model.predict(test)
    observed = test[test["status"] == "observed"]
    result = {"model": "lightgbm", "fold": "test", **scores(observed["pickups"], observed["prediction"]),
              "train_rows": model.train_rows, "rounds": model.rounds, "seconds": round(time.time() - start, 1)}
    print(result, flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "holdout_metrics.json").write_text(json.dumps(result, indent=2))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "station_id": test["station_id"], "target_start_utc": test["hour_utc"], "origin_utc": test["origin_utc"],
        "target": "pickups", "prediction": test["prediction"].astype("float64"), "model_version": "lightgbm-v1",
        "data_mode": "test", "fold": "test",
    }).to_parquet(PREDICTIONS_DIR / "lightgbm-v1_test_test_all.parquet", index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock", action="store_true", help="open the holdout; only once the method is frozen")
    run(parser.parse_args().unlock)


if __name__ == "__main__":
    main()

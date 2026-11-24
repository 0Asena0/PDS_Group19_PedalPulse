"""Run models over the four dev folds and save the scores.

Usage: uv run python -m pedalpulse.evaluate --stations 200 --models seasonal_naive lightgbm
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config
from .baselines import BASELINES
from .dataset import load_station_hours, release_path
from .features import build_features
from .folds import FOLDS, split
from . import feature_store
from .metrics import scores
from .models import LGB_FEATURES, MODELS, LightGBM, lgb_matrix

ALL_MODELS = {**BASELINES, **MODELS}

RESULTS_DIR = config.REPORTS_DIR / "results"
PREDICTIONS_DIR = config.DATA_DIR / "predictions"
MODELS_DIR = config.DATA_DIR / "models"


def make_model(spec: str):
    """A model by name; "poisson@0.001" sets the penalty alpha of the Poisson regression."""
    name, _, alpha = spec.partition("@")
    if name not in ALL_MODELS:
        raise SystemExit(f"unknown model {name!r}; use one of {list(ALL_MODELS)}")
    return ALL_MODELS[name](alpha=float(alpha)) if alpha else ALL_MODELS[name]()


def dev_stations(n: int | None, seed: int = 19) -> list[str] | None:
    """A fixed random sample of cohort stations, so everyone develops on the same subset. None means all."""
    cohort = pd.read_csv(release_path("cohort.csv"), dtype={"station_id": str})["station_id"].sort_values()
    if n is None or n >= len(cohort):
        return None
    # I draw at random instead of taking the busiest stations: a top list would make the errors look
    # different from the full cohort. The seed keeps the sample the same for everyone.
    return sorted(cohort.sample(n, random_state=seed))


def load_dev(stations: list[str] | None) -> pd.DataFrame:
    df = load_station_hours("dev", stations=stations)
    station_table = pd.read_parquet(release_path("stations.parquet"))
    return build_features(df, station_table)


def run(models: list[str], n_stations: int | None, save_predictions: bool = False) -> pd.DataFrame:
    stations = dev_stations(n_stations)
    data = load_dev(stations)
    results = []
    for fold in FOLDS:
        train, valid = split(data, fold)
        # I forecast every hour (the forecast contract asks for inactive hours too), but score only observed
        # hours, and the same rows for every model.
        observed = (valid["status"] == "observed").to_numpy()
        for name in models:
            start = time.time()
            model = make_model(name).fit(train)
            pred = model.predict(valid)
            results.append({
                "model": name, "fold": fold.name, **scores(valid["pickups"][observed], pred[observed]),
                "seconds": round(time.time() - start, 1),
                "stations": len(stations) if stations else "all",
            })
            print(results[-1])
            if save_predictions:
                _save_predictions(valid, pred, name, fold.name, n_stations)
    return pd.DataFrame(results)


def _save_predictions(valid: pd.DataFrame, pred: np.ndarray, model: str, fold: str, n_stations: int | None) -> None:
    # Same columns as the forecast file the app reads, so Sandra can use these files directly.
    out = pd.DataFrame({
        "station_id": valid["station_id"].to_numpy(),
        "target_start_utc": valid["hour_utc"].to_numpy(),
        "origin_utc": valid["origin_utc"].to_numpy(),
        "target": "pickups",
        "prediction": pred,
        "model_version": f"{model}-v1",
        "data_mode": "validation",
        "fold": fold,
    })
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_DIR / f"{model}-v1_validation_{fold}_{n_stations or 'all'}.parquet", index=False, compression="zstd")


def fit_full(train_start, train_end) -> LightGBM:
    """LightGBM on all cohort stations, trained on the feature store rows between two local dates.

    The training block has up to 47 million rows. I never hold it as a table: it goes month by month
    into one float32 matrix, LightGBM bins it, and then the matrix is freed. Only LightGBM, because the random
    forest needs the raw rows during the whole fit and does not fit into memory at this size.
    """
    stations = pd.read_csv(release_path("cohort.csv"), dtype={"station_id": str})["station_id"].sort_values().tolist()
    boroughs = sorted(pd.read_parquet(release_path("stations.parquet"))["borough"].dropna().unique())
    model = LightGBM()
    model.boroughs, model.stations = boroughs, stations
    x, y, dates = feature_store.load_matrix(lambda c: lgb_matrix(c, boroughs, stations),
                                            str(train_start), str(train_end), len(LGB_FEATURES))
    full = model.dataset(x, y)
    del x  # binned now; the raw matrix is not needed any more
    model.fit_dataset(full, dates)
    model.train_rows = len(y)
    return model


def run_full(folds: list[str], save_predictions: bool = False) -> pd.DataFrame:
    """LightGBM on all cohort stations, one fold at a time. The fitted model is saved, so it can be reused."""
    results = []
    for fold in [f for f in FOLDS if f.name in folds]:
        start = time.time()
        model = fit_full(fold.train_start, fold.train_end)
        model.save(MODELS_DIR / f"lightgbm-v1_{fold.name}_all")
        valid = feature_store.load(["*"], str(fold.valid_start), str(fold.valid_end))
        pred = model.predict(valid)
        observed = (valid["status"] == "observed").to_numpy()
        results.append({"model": "lightgbm", "fold": fold.name, **scores(valid["pickups"][observed], pred[observed]),
                        "seconds": round(time.time() - start, 1), "stations": "all",
                        "train_rows": model.train_rows, "rounds": model.rounds})
        print(results[-1], flush=True)
        if save_predictions:
            _save_predictions(valid, pred, "lightgbm", fold.name, None)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BASELINES),
                        help=f"any of {list(ALL_MODELS)}; poisson@<alpha> sets the penalty")
    parser.add_argument("--stations", type=int, default=200, help="size of the random station sample; 0 = all")
    parser.add_argument("--save-predictions", action="store_true", help="write every forecast to data/predictions")
    parser.add_argument("--full", action="store_true", help="all cohort stations from the feature store")
    parser.add_argument("--folds", nargs="+", default=[f.name for f in FOLDS])
    args = parser.parse_args()
    if args.full:
        if args.models != ["lightgbm"]:
            raise SystemExit("--full runs LightGBM only: use --models lightgbm")
        table = run_full(args.folds, args.save_predictions)
        args.stations = 0
    else:
        table = run(args.models, args.stations or None, args.save_predictions)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{'_'.join(args.models)}_{args.stations or 'all'}.csv"
    table.to_csv(out, index=False)
    summary = table.groupby("model")[["mae", "rmse", "wape"]].mean().round(3)
    print(summary.to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

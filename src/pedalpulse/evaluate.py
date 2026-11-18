"""Run models over the four dev folds and save the scores.

Usage: uv run python -m pedalpulse.evaluate --stations 200 --models seasonal_naive poisson@1e-4
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import config
from .baselines import BASELINES
from .dataset import load_station_hours, release_path
from .features import build_features
from .folds import FOLDS, split
from .metrics import scores

ALL_MODELS = dict(BASELINES)

RESULTS_DIR = config.REPORTS_DIR / "results"


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


def run(models: list[str], n_stations: int | None) -> pd.DataFrame:
    stations = dev_stations(n_stations)
    data = load_dev(stations)
    results = []
    for fold in FOLDS:
        train, valid = split(data, fold)
        # I score only observed hours, and the same rows for every model.
        valid = valid[valid["status"] == "observed"]
        for name in models:
            start = time.time()
            model = make_model(name).fit(train)
            pred = model.predict(valid)
            results.append({
                "model": name, "fold": fold.name, **scores(valid["pickups"], pred),
                "seconds": round(time.time() - start, 1),
                "stations": len(stations) if stations else "all",
            })
            print(results[-1])
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BASELINES),
                        help=f"any of {list(ALL_MODELS)}; poisson@<alpha> sets the penalty")
    parser.add_argument("--stations", type=int, default=200, help="size of the random station sample; 0 = all")
    args = parser.parse_args()
    table = run(args.models, args.stations or None)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{'_'.join(args.models)}_{args.stations or 'all'}.csv"
    table.to_csv(out, index=False)
    summary = table.groupby("model")[["mae", "rmse", "wape"]].mean().round(3)
    print(summary.to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

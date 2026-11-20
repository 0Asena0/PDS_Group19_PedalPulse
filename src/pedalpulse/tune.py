"""Small random search for LightGBM and the random forest on the four dev folds.

Usage: uv run python -m pedalpulse.tune --model lightgbm --trials 12 --stations 200
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .evaluate import RESULTS_DIR, dev_stations, load_dev
from .folds import FOLDS, split
from .metrics import scores
from .models import LightGBM, RandomForest

# I keep the search small on purpose (roadmap p. 16: "kleines Suchbudget"): every extra trial is another look
# at the validation windows, and with enough looks one setting wins by luck.
SPACES = {
    "lightgbm": {
        "num_leaves": [31, 63, 127, 255],
        "min_data_in_leaf": [50, 200, 1000],
        "feature_fraction": [0.6, 0.8, 1.0],
        "lambda_l2": [0.0, 1.0, 10.0],
    },
    "random_forest": {
        "max_features": [0.3, 0.5, 0.7],
        "min_samples_leaf": [5, 20, 50],
        "max_samples": [0.05, 0.1, 0.2],
    },
}
# The search runs at the same learning rate as the final model (0.1), so what wins here is what we use.
SEARCH_FIXED = {"lightgbm": {"learning_rate": 0.1}, "random_forest": {}}
MODEL_CLASS = {"lightgbm": LightGBM, "random_forest": RandomForest}
# The defaults the search started from (trial 0). I write them down here instead of reading MODEL_CLASS.DEFAULTS:
# the forest's default changed after the search, and a rerun must still draw the same 12 trials.
SEARCH_START = {
    "lightgbm": {"num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8, "lambda_l2": 0.0},
    "random_forest": {"max_features": 0.5, "min_samples_leaf": 20, "max_samples": 0.1},
}


def draw_trials(model: str, n: int, seed: int = 19) -> list[dict]:
    """The starting default first (so the search can only confirm or beat it), then n - 1 random settings."""
    rng = np.random.default_rng(seed)
    space = SPACES[model]
    trials = [dict(SEARCH_START[model])]
    while len(trials) < n:
        trial = {k: v[rng.integers(len(v))] for k, v in space.items()}
        trial = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in trial.items()}
        if trial not in trials:
            trials.append(trial)
    return trials


def run(model: str, trials: int, n_stations: int | None) -> pd.DataFrame:
    data = load_dev(dev_stations(n_stations))
    out = RESULTS_DIR / f"tuning_{model}_{n_stations or 'all'}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # I keep finished trials from an earlier run, so an interrupted search continues where it stopped.
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["fold"], r["trial"]) for r in rows}
    for fold in FOLDS:
        todo = [(i, p) for i, p in enumerate(draw_trials(model, trials)) if (fold.name, i) not in done]
        if not todo:
            continue
        train, valid = split(data, fold)
        valid = valid[valid["status"] == "observed"]
        for i, params in todo:
            start = time.time()
            fitted = MODEL_CLASS[model](**SEARCH_FIXED[model], **params).fit(train)
            rows.append({"model": model, "trial": i, "params": json.dumps(params), "fold": fold.name,
                         **scores(valid["pickups"], fitted.predict(valid)),
                         "rounds": getattr(fitted, "rounds", None), "seconds": round(time.time() - start, 1)})
            print(rows[-1], flush=True)
            # I write after every trial, so a crash or a timeout does not lose the finished ones.
            pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean MAE over the folds, and in how many folds each trial beats the default (trial 0)."""
    mae = table.pivot_table(index=["trial", "params"], columns="fold", values="mae")
    default = mae.xs(0, level="trial").iloc[0]
    result = pd.DataFrame({"mean_mae": mae.mean(axis=1), "folds_better_than_default": (mae < default).sum(axis=1)})
    return result.sort_values("mean_mae")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(SPACES), required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--stations", type=int, default=200, help="0 = all")
    args = parser.parse_args()
    table = run(args.model, args.trials, args.stations or None)
    print(summary(table).round(4).to_string())


if __name__ == "__main__":
    main()

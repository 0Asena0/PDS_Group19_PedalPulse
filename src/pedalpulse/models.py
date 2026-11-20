"""Learned models: a random forest as comparison and LightGBM as the main model.

Both get the same features and are trained on the observed hours of a fold's training block only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .baselines import observed
from .features import CALENDAR_FEATURES, HISTORY_FEATURES, WEATHER_FEATURES

NUMERIC_FEATURES = CALENDAR_FEATURES + HISTORY_FEATURES + WEATHER_FEATURES + ["lat", "lng"]
LGB_FEATURES = NUMERIC_FEATURES + ["borough", "station_id"]
SEED = 19


def _matrix(df: pd.DataFrame, boroughs: list[str]) -> pd.DataFrame:
    # float32 instead of float64: half the memory, which matters with 40 million training rows.
    x = df[NUMERIC_FEATURES].astype("float32")
    # Borough as a fixed category list, so a fold that never saw a borough still gets the same columns.
    x["borough"] = pd.Categorical(df["borough"], categories=boroughs).codes.astype("float32")
    return x


def lgb_matrix(df: pd.DataFrame, boroughs: list[str], stations: list[str]) -> np.ndarray:
    """The LightGBM input as one float32 array, with borough and station as category codes (-1 = unknown)."""
    # I fill a single array column by column instead of copying the whole table: on all stations the
    # training block has 42 million rows, and every extra copy of it costs 4 GB.
    x = np.empty((len(df), len(LGB_FEATURES)), dtype="float32")
    for j, col in enumerate(NUMERIC_FEATURES):
        x[:, j] = df[col].to_numpy(dtype="float32", na_value=np.nan)
    x[:, -2] = pd.Categorical(df["borough"], categories=boroughs).codes
    x[:, -1] = pd.Categorical(df["station_id"], categories=stations).codes
    return x


class RandomForest:
    """Many deep trees on random subsets, averaged.

    No station_id: a forest would treat the ID as a number on a scale, which it is not.
    The station is described by its coordinates, its borough and its own past demand (the lags) instead.
    """

    DEFAULTS = dict(n_estimators=100, max_samples=0.1, min_samples_leaf=20, max_features=0.5)

    def __init__(self, **params):
        self.params = {**self.DEFAULTS, **params}

    def fit(self, train: pd.DataFrame) -> "RandomForest":
        rows = observed(train)
        self.boroughs = sorted(rows["borough"].dropna().unique())
        # max_samples=0.1: each tree sees 10 % of the rows. On millions of rows a full bootstrap per tree
        # takes far too long, and the trees still differ enough to average well.
        self.model = RandomForestRegressor(n_jobs=-1, random_state=SEED, **self.params)
        self.model.fit(_matrix(rows, self.boroughs), rows["pickups"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_matrix(df, self.boroughs))


class LightGBM:
    """One global gradient-boosting model for all stations, Poisson objective, station_id as a category."""

    DEFAULTS = dict(objective="poisson", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.0,
                    seed=SEED, verbose=-1)

    def __init__(self, max_rounds: int = 2000, early_stopping_days: int = 28, refit: bool = True, **params):
        self.params = {**self.DEFAULTS, **params}
        self.max_rounds = max_rounds
        self.early_stopping_days = early_stopping_days
        self.refit = refit

    def dataset(self, x: np.ndarray, y: np.ndarray) -> lgb.Dataset:
        """Bin the training rows once. After this the raw array is no longer needed."""
        return lgb.Dataset(x, y, feature_name=LGB_FEATURES, categorical_feature=["borough", "station_id"],
                           params=self.params, free_raw_data=True).construct()

    def fit(self, train: pd.DataFrame) -> "LightGBM":
        rows = observed(train)
        self.boroughs = sorted(rows["borough"].dropna().unique())
        self.stations = sorted(rows["station_id"].unique())
        full = self.dataset(lgb_matrix(rows, self.boroughs, self.stations), rows["pickups"].to_numpy("float32"))
        return self.fit_dataset(full, pd.to_datetime(rows["date_local"]).to_numpy())

    def fit_dataset(self, full: lgb.Dataset, dates: np.ndarray) -> "LightGBM":
        """Train on an already binned dataset; dates are the local dates of its rows, in the same order."""
        # Step 1, how many trees: early stopping needs data the trees did not learn from. I hold back the
        # last 4 weeks of the training block for that, not the fold's validation window: stopping on the
        # validation window would tune the model on exactly the days it is graded on.
        cut = dates.max() - np.timedelta64(self.early_stopping_days, "D")
        fit_set = full.subset(np.flatnonzero(dates <= cut))
        stop_set = full.subset(np.flatnonzero(dates > cut))
        model = lgb.train(self.params, fit_set, num_boost_round=self.max_rounds, valid_sets=[stop_set],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        self.rounds = model.best_iteration
        # Step 2, refit: the last 4 weeks are the most recent and most useful days, so after fixing the number
        # of trees I train again on the whole training block with that number (roadmap p. 6, "Finalisierung").
        self.model = lgb.train(self.params, full, num_boost_round=self.rounds) if self.refit else model
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(lgb_matrix(df, self.boroughs, self.stations), num_iteration=self.rounds)


MODELS = {"random_forest": RandomForest, "lightgbm": LightGBM}

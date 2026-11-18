"""Three simple baselines the main model has to beat.

All of them are fitted on the training rows of one fold only (status observed) and frozen for its validation window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

LEVELS = [["station_id", "weekday", "hour_local"], ["station_id", "hour_local"], ["station_id"]]


def observed(df: pd.DataFrame) -> pd.DataFrame:
    # Inactive hours are closures or data gaps, not real zero demand, so I do not learn from them.
    return df[df["status"] == "observed"]


class GlobalMean:
    """The same number for every station and hour: the mean of all training hours.

    A sanity anchor, not a real competitor. If a model is not clearly better than this, something is broken.
    """

    def fit(self, train: pd.DataFrame) -> "GlobalMean":
        self.mean = float(observed(train)["pickups"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean)


class HistoricalMean:
    """Mean pickups per station, weekday and local hour in the training rows.

    Fallback when a combination was never seen: station and hour, then station, then the overall mean.
    """

    def fit(self, train: pd.DataFrame) -> "HistoricalMean":
        rows = observed(train)
        self.tables = [rows.groupby(cols, observed=True)["pickups"].mean().rename("m").reset_index() for cols in LEVELS]
        self.overall = float(rows["pickups"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = pd.Series(np.nan, index=df.index)
        # I go from the most detailed level to the coarsest and only fill what is still empty.
        for cols, table in zip(LEVELS, self.tables):
            m = df[cols].merge(table, on=cols, how="left")["m"].to_numpy()
            pred = pred.fillna(pd.Series(m, index=df.index))
        return pred.fillna(self.overall).to_numpy()


class SeasonalNaive:
    """Same local hour one week earlier (lag_7d). Fallback: the historical mean."""

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        self.fallback = HistoricalMean().fit(train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # lag_7d is missing for a station's first week and after a spring-forward day.
        # I fill those hours with the historical mean, a fixed rule decided before looking at results.
        lag = df["lag_7d"].to_numpy(dtype=float)
        return np.where(np.isnan(lag), self.fallback.predict(df), lag)


class PoissonBaseline:
    """Poisson regression on calendar and station categories (one-hot, ridge penalty alpha)."""

    CATEGORIES = ["station_id", "hour_type", "weekday", "month", "is_holiday"]

    # I chose alpha on the four folds (data-v2, 200 stations): 1e-2, 1e-3, 1e-4, 1e-5, 1e-6.
    # Less penalty was better in every fold, and below 1e-5 the gain is under 0.001 MAE, so the
    # curve is flat there. With millions of rows and a few thousand dummy columns the model barely overfits.
    def __init__(self, alpha: float = 1e-6):
        self.alpha = alpha

    @staticmethod
    def _frame(df: pd.DataFrame) -> pd.DataFrame:
        # I combine hour and weekend into one category, because the daily curve on weekends looks
        # completely different (no commuter peaks). A plain linear model cannot learn that interaction itself.
        hour_type = df["hour_local"].astype(str) + np.where(df["is_weekend"], "_we", "_wd")
        return pd.DataFrame({
            "station_id": df["station_id"].astype(str),
            "hour_type": hour_type,
            "weekday": df["weekday"].astype(str),
            "month": df["month"].astype(str),
            "is_holiday": df["is_holiday"].astype(str),
        }, index=df.index)

    def fit(self, train: pd.DataFrame) -> "PoissonBaseline":
        rows = observed(train)
        # handle_unknown="ignore": a station or month the fold never saw gets the average effect (all zeros).
        self.encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORIES)])
        x = self.encoder.fit_transform(self._frame(rows))
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=300).fit(x, rows["pickups"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.encoder.transform(self._frame(df)))


BASELINES = {
    "global_mean": GlobalMean,
    "seasonal_naive": SeasonalNaive,
    "historical_mean": HistoricalMean,
    "poisson": PoissonBaseline,
}

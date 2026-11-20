"""Error metrics for count forecasts. Predictions are compared unrounded."""
from __future__ import annotations

import numpy as np
import pandas as pd


def scores(actual, predicted) -> dict[str, float]:
    """MAE, RMSE, WAPE and bias (mean of prediction minus actual) over all rows."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    # I refuse missing predictions: every model has to be scored on exactly the same rows,
    # otherwise one model could look better just by skipping the hard hours.
    if np.isnan(p).any():
        raise ValueError("predictions contain NaN; every model must cover the same rows")
    err = p - y
    total = y.sum()
    return {
        "n": float(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        # WAPE divides by the total demand. With zero demand it is not defined, so I return NaN instead of a number.
        "wape": float(np.abs(err).sum() / total) if total > 0 else float("nan"),
        "bias": float(err.mean()),
    }


def scores_by(df: pd.DataFrame, by: str | list[str], actual: str = "pickups", predicted: str = "prediction") -> pd.DataFrame:
    """The same metrics for each group, for example per fold, borough or hour."""
    rows = [{**dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,))),
             **scores(g[actual], g[predicted])}
            for key, g in df.groupby(by, observed=True)]
    return pd.DataFrame(rows)


def poisson_floor(expected) -> dict[str, float]:
    """The error left over if every hour's true mean were known exactly and pickups were Poisson around it.

    This is a lower bound for any forecast: the randomness of who rents a bike in a given hour cannot be predicted.
    'mae_mean' = floor when forecasting the mean (what our models do); 'mae_median' = floor when forecasting the
    median, which MAE rewards (for small means the median is often 0).
    """
    from scipy.stats import poisson

    lam = np.clip(np.asarray(expected, dtype=float), 1e-9, None)
    # Mean absolute deviation of a Poisson variable around its mean: 2 * lam * P(Y = floor(lam)).
    mae_mean = 2 * lam * poisson.pmf(np.floor(lam), lam)
    # Around the median m: E|Y - m| = lam - m + 2 * (F(0) + ... + F(m - 1)). I compute it once per rounded
    # mean, because millions of rows share a few thousand distinct values.
    grid, inverse = np.unique(np.round(lam, 3), return_inverse=True)
    med = poisson.median(grid)
    cum = np.array([poisson.cdf(np.arange(int(m)), g).sum() for g, m in zip(grid, med)])
    mae_median = (grid - med + 2 * cum)[inverse]
    return {"mae_mean": float(mae_mean.mean()), "mae_median": float(mae_median.mean())}

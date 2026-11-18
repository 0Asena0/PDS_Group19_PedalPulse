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

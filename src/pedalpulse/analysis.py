"""Where does LightGBM go wrong? Error analysis on the four dev folds (200 stations), no holdout.

Needs the forecasts from experiments.py (variant "contract"). Writes tables to reports/results and two
figures to reports/figures.

Usage: uv run python -m pedalpulse.analysis
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .dataset import release_path
from .evaluate import PREDICTIONS_DIR, RESULTS_DIR, dev_stations, load_dev
from .folds import FOLDS
from .metrics import scores_by

FIGURES_DIR = config.FIGURES_DIR
# A forecast hour counts as rainy from 0.2 mm on; below that it is drizzle or forecast noise.
RAIN_MM = 0.2
# Fixed before looking at any forecast: demand terciles of the 200 stations (mean pickups per hour up to
# 2024-09-30, the cohort cut-off), then one station per tercile drawn with seed 19.
EXAMPLE_STATIONS = {"5964.01": "high demand", "5105.09": "medium demand", "6113.07": "low demand"}


def demand_class(data: pd.DataFrame) -> pd.Series:
    """Tercile of each station by its mean pickups up to the cohort cut-off (no information from the folds)."""
    early = data[(data["status"] == "observed") & (pd.to_datetime(data["date_local"]) <= "2024-09-30")]
    mean = early.groupby("station_id")["pickups"].mean()
    return pd.qcut(mean, 3, labels=["low", "medium", "high"])


def forecasts(data: pd.DataFrame) -> pd.DataFrame:
    preds = pd.concat([pd.read_parquet(PREDICTIONS_DIR / f"lightgbm-final_validation_{f.name}_200.parquet")
                       .assign(fold=f.name) for f in FOLDS])
    df = data.merge(preds, on=["station_id", "hour_utc"], how="inner", validate="one_to_one")
    df["demand"] = df["station_id"].map(demand_class(data))
    df["rain"] = np.where(df["precipitation"] >= RAIN_MM, "rain", "dry")
    df["day_type"] = np.select([df["is_holiday"] == 1, df["is_weekend"] == 1], ["holiday", "weekend"], "workday")
    return df


def tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {by: scores_by(df, by) for by in ("fold", "demand", "borough", "rain", "day_type", "hour_local")}
    # Mean demand per group, so MAE and WAPE can be read side by side.
    for by, table in out.items():
        table["mean_pickups"] = df.groupby(by, observed=True)["pickups"].mean().to_numpy()
    return out


def plot_groups(t: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.4, 1, 1]})
    hour = t["hour_local"]
    axes[0].plot(hour["hour_local"], hour["wape"], marker="o", color="#2a6f97")
    axes[0].set(title="WAPE by local hour", xlabel="hour", ylabel="WAPE", xticks=range(0, 24, 3))
    ax2 = axes[0].twinx()
    ax2.bar(hour["hour_local"], hour["mean_pickups"], color="#cccccc", alpha=0.6, zorder=0)
    ax2.set_ylabel("mean pickups per station-hour", color="#777777")
    axes[0].set_zorder(ax2.get_zorder() + 1)
    axes[0].patch.set_visible(False)
    for ax, by, title in ((axes[1], "demand", "by station demand"), (axes[2], "fold", "by fold")):
        axes_table = t[by]
        ax.bar(axes_table[by].astype(str), axes_table["wape"], color="#2a6f97")
        ax.set(title=f"WAPE {title}", ylim=(0, max(0.8, axes_table["wape"].max() * 1.1)))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_error_by_group.png", dpi=150)
    plt.close(fig)


def plot_examples(df: pd.DataFrame) -> None:
    # The second full week (Monday to Sunday) of the summer fold f4, fixed before looking at the forecasts.
    f4 = next(f for f in FOLDS if f.name == "f4")
    start = pd.Timestamp(f4.valid_start)
    first_monday = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
    week_start = first_monday + pd.Timedelta(days=7)
    week = df[(pd.to_datetime(df["date_local"]) >= week_start) & (pd.to_datetime(df["date_local"]) < week_start + pd.Timedelta(days=7))]
    fig, axes = plt.subplots(len(EXAMPLE_STATIONS), 1, figsize=(12, 7.5), sharex=True)
    for ax, (station, label) in zip(axes, EXAMPLE_STATIONS.items()):
        s = week[week["station_id"] == station].sort_values("hour_utc")
        local = s["hour_utc"].dt.tz_convert(config.TIMEZONE) if s["hour_utc"].dt.tz else s["hour_utc"]
        ax.plot(local, s["pickups"], color="#999999", lw=1, label="actual")
        ax.plot(local, s["prediction"], color="#2a6f97", lw=1.6, label="forecast")
        name = s["name"].iloc[0] if "name" in s and len(s) else station
        ax.set_title(f"{name} ({label})", loc="left", fontsize=10)
        ax.set_ylabel("pickups")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Actual against forecast, week from {week_start.date()} (fold f4)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_example_stations.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_dev(dev_stations(200))
    names = pd.read_parquet(release_path("stations.parquet"))[["station_id", "name"]]
    df = forecasts(data).merge(names, on="station_id", how="left")
    t = tables(df)
    for by, table in t.items():
        table.to_csv(RESULTS_DIR / f"errors_by_{by}.csv", index=False)
        print(table.round(3).to_string(index=False), "\n")
    plot_groups(t)
    plot_examples(df)


if __name__ == "__main__":
    main()

"""Time-based folds for model selection and the final test.

All bounds are local dates (America/New_York), inclusive. A forecast for day D uses everything up to the end of
D-1 (data contract, E1), so a fold can train on every day up to train_end and validate from the next day on.

The four rolling-origin folds lie inside the dev file. The holdout period is only reachable through
holdout_split(), which needs an explicit unlock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from . import config


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date


# I use four folds instead of one validation window: one quarter alone could be a lucky or unlucky season.
# With four quarters I see autumn, winter, spring and summer once each.
# Training always starts in January 2023 and grows by one quarter per fold, like in real use:
# the model gets retrained with all history it has at that point.
# I let fold 1 end on 2024-09-30 on purpose: the cohort was chosen with data up to that day,
# so the first fold does not use anything the cohort choice did not already know.
FOLDS = (
    Fold("f1", config.STUDY_START, date(2024, 9, 30), date(2024, 10, 1), date(2024, 12, 31)),
    Fold("f2", config.STUDY_START, date(2024, 12, 31), date(2025, 1, 1), date(2025, 3, 31)),
    Fold("f3", config.STUDY_START, date(2025, 3, 31), date(2025, 4, 1), date(2025, 6, 30)),
    Fold("f4", config.STUDY_START, date(2025, 6, 30), date(2025, 7, 1), date(2025, 9, 30)),
)

# Final model: refit on all dev labels, evaluate once on the holdout.
TEST = Fold("test", config.STUDY_START, config.DEV_END, config.HOLDOUT_START, config.STUDY_END)


def get_fold(name: str) -> Fold:
    for fold in FOLDS:
        if fold.name == name:
            return fold
    raise KeyError(f"unknown fold {name!r}; use one of {[f.name for f in FOLDS]}")


def _local_dates(df: pd.DataFrame) -> pd.Series:
    # I split by local date, not by UTC hour. A forecast covers one local day,
    # so all hours of that day must land on the same side of the split.
    return pd.to_datetime(df["date_local"]).dt.date


def _check_dev_only(dates: pd.Series) -> None:
    # Safety net: if someone loads the holdout by mistake, I stop with an error instead of quietly using it.
    if (dates >= config.HOLDOUT_START).any():
        raise ValueError(f"data reaches into the holdout (from {config.HOLDOUT_START}); use holdout_split() for the final test")


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation rows of a dev fold, by local date."""
    if fold is TEST:
        raise ValueError("the test fold is only available through holdout_split()")
    dates = _local_dates(df)
    _check_dev_only(dates)
    train = df[(dates >= fold.train_start) & (dates <= fold.train_end)]
    valid = df[(dates >= fold.valid_start) & (dates <= fold.valid_end)]
    return train, valid


def holdout_split(dev: pd.DataFrame, holdout: pd.DataFrame, *, unlock: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows (all of dev) and test rows (holdout). Only once the method is frozen."""
    # I treat the test as a one-time look. Tuning anything after seeing it would make the result too optimistic.
    # unlock=True is a conscious step, so nobody opens it by accident.
    if not unlock:
        raise PermissionError("the final test is locked; pass unlock=True only after the method is frozen")
    dev_dates, test_dates = _local_dates(dev), _local_dates(holdout)
    _check_dev_only(dev_dates)
    if (test_dates < TEST.valid_start).any() or (test_dates > TEST.valid_end).any():
        raise ValueError("holdout rows outside the test period")
    return dev[dev_dates >= TEST.train_start], holdout

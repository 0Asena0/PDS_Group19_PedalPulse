from datetime import date, timedelta

import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.folds import FOLDS, TEST, get_fold, holdout_split, split


def days(start, end):
    n = (end - start).days + 1
    return pd.DataFrame({"date_local": [start + timedelta(d) for d in range(n)]})


def test_folds_are_ordered_and_contiguous():
    # I check that each validation window starts the day after training ends: no overlap (that would leak),
    # and no gap (I would silently skip days).
    for fold in FOLDS:
        assert fold.train_start == config.STUDY_START
        assert fold.train_start <= fold.train_end < fold.valid_start <= fold.valid_end
        assert fold.valid_start == fold.train_end + timedelta(1)
    for prev, nxt in zip(FOLDS, FOLDS[1:]):
        assert nxt.train_end == prev.valid_end
        assert nxt.valid_start == prev.valid_end + timedelta(1)


def test_folds_stay_inside_dev():
    # I check that the last fold ends exactly where the holdout begins, so no fold ever touches the test period.
    assert FOLDS[-1].valid_end == config.DEV_END
    assert all(f.valid_end < config.HOLDOUT_START for f in FOLDS)
    assert TEST.train_end == config.DEV_END
    assert TEST.valid_start == config.HOLDOUT_START


def test_cohort_is_chosen_inside_first_training_block():
    # I check that the cohort only uses what fold 1 is allowed to know.
    assert config.SELECTION_CUTOFF <= FOLDS[0].train_end


def test_split_has_no_overlap_and_no_gap():
    # Same check as above, but on a table with one row per day: I count the rows that come out.
    df = days(config.STUDY_START, config.DEV_END)
    for fold in FOLDS:
        train, valid = split(df, fold)
        assert train["date_local"].max() < valid["date_local"].min()
        assert valid["date_local"].min() == fold.valid_start
        assert valid["date_local"].max() == fold.valid_end
        assert len(train) == (fold.train_end - fold.train_start).days + 1


def test_split_keeps_all_hours_of_a_day_together():
    # I took the border between training and validation of fold 1 (30 Sep / 1 Oct 2024)
    # and check that all 24 hours of each day end up on one side.
    hours = pd.DataFrame({"date_local": [date(2024, 9, 30)] * 24 + [date(2024, 10, 1)] * 24})
    train, valid = split(hours, get_fold("f1"))
    assert len(train) == 24 and len(valid) == 24


def test_split_refuses_holdout_rows():
    # I give it a few days that cross into October 2025 (the holdout) and expect an error, not a quiet split.
    df = days(date(2025, 9, 25), date(2025, 10, 2))
    with pytest.raises(ValueError, match="holdout"):
        split(df, get_fold("f4"))
    with pytest.raises(ValueError):
        split(df, TEST)


def test_holdout_split_is_locked():
    # I check that the test data stays closed without unlock=True, and that all rows come back with it.
    dev = days(config.STUDY_START, config.DEV_END)
    holdout = days(config.HOLDOUT_START, config.STUDY_END)
    with pytest.raises(PermissionError):
        holdout_split(dev, holdout)
    train, test = holdout_split(dev, holdout, unlock=True)
    assert len(train) == len(dev) and len(test) == len(holdout)


def test_unknown_fold():
    with pytest.raises(KeyError):
        get_fold("f5")

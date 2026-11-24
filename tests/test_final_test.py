from datetime import timedelta

import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.dataset import load_station_hours, release_path
from pedalpulse.features import build_features
from pedalpulse.final_test import HISTORY_DAYS, features_with_history, run


def test_final_test_is_locked():
    with pytest.raises(PermissionError):
        run(unlock=False)


@pytest.mark.skipif(not (config.RELEASE_DIR / "dev_sample_manhattan_2024_summer.parquet").exists(),
                    reason="release not downloaded")
def test_history_gives_the_same_features_as_one_pass():
    # The holdout is built from the last weeks of dev plus the holdout. I check the idea on the sample without
    # touching the holdout: split the sample in two, build the second part with history, compare with one pass.
    stations = pd.read_parquet(release_path("stations.parquet"))
    sample = load_station_hours("sample")
    dates = pd.to_datetime(sample["date_local"])
    cut = dates.min() + (dates.max() - dates.min()) / 2
    history = sample[(dates < cut) & (dates >= cut - timedelta(days=HISTORY_DAYS))]
    later = sample[dates >= cut]
    whole = build_features(sample, stations)
    whole = whole[pd.to_datetime(whole["date_local"]) >= cut]
    key = ["station_id", "hour_utc"]
    pd.testing.assert_frame_equal(whole.sort_values(key).reset_index(drop=True),
                                  features_with_history(history, later, stations).sort_values(key).reset_index(drop=True),
                                  check_like=True)

import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.dataset import load_station_hours, release_path
from pedalpulse.features import build_features

pytestmark = pytest.mark.skipif(
    not (config.RELEASE_DIR / "dev_sample_manhattan_2024_summer.parquet").exists(), reason="release not downloaded"
)


def test_chunks_give_the_same_features_as_one_pass():
    # The feature store builds 150 stations at a time. That is only allowed if a station's features never
    # depend on other stations. I build the Manhattan sample once in one pass and once in two halves,
    # and check that every value is identical.
    stations = pd.read_parquet(release_path("stations.parquet"))
    whole = build_features(load_station_hours("sample"), stations)
    ids = sorted(whole["station_id"].unique())
    halves = [build_features(load_station_hours("sample", stations=part), stations)
              for part in (ids[: len(ids) // 2], ids[len(ids) // 2:])]
    chunked = pd.concat(halves)
    key = ["station_id", "hour_utc"]
    pd.testing.assert_frame_equal(whole.sort_values(key).reset_index(drop=True),
                                  chunked.sort_values(key).reset_index(drop=True), check_like=True)

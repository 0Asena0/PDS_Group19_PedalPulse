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


def test_load_matrix_crosses_month_borders_without_losing_rows():
    # load_matrix reads one month at a time. Over a month border it must return exactly the rows a single
    # query returns, and nothing twice.
    from pedalpulse import feature_store

    if not list((feature_store.FEATURE_DIR / f"dev_{config.DATA_VERSION}").glob("*.parquet")):
        pytest.skip("feature store not built")
    start, end = "2025-01-25", "2025-02-03"
    x, y, dates = feature_store.load_matrix(lambda c: c[["pickups"]].to_numpy("float32"), start, end, 1)
    direct = feature_store.load(["pickups", "date_local"], start, end, observed_only=True)
    assert len(y) == len(direct)
    assert y.sum() == direct["pickups"].sum()
    assert (x[:, 0] == y).all()
    assert dates.min() == pd.Timestamp(start) and dates.max() == pd.Timestamp(end)

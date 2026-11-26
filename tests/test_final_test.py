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


@pytest.mark.skipif(not (config.RELEASE_DIR / "dev_sample_manhattan_2024_summer.parquet").exists(),
                    reason="release not downloaded")
def test_final_test_runs_end_to_end_on_stand_in_data(tmp_path, monkeypatch):
    # The real run needs the holdout and hours of training, so I can run it only once, at the very end.
    # To catch mistakes before that, I run the whole script with small stand-ins: the sample plays the
    # training data and, moved into the test period, the holdout. Everything around the model is the real code.
    from pedalpulse import final_test
    from pedalpulse.folds import TEST
    from pedalpulse.models import LightGBM

    features = build_features(load_station_hours("sample"), pd.read_parquet(release_path("stations.parquet")))
    dates = pd.to_datetime(features["date_local"])
    stand_in = features[dates >= dates.max() - timedelta(days=6)].copy()
    stand_in["date_local"] = (pd.to_datetime(stand_in["date_local"]) - pd.to_datetime(stand_in["date_local"]).min()
                              + pd.Timestamp(TEST.valid_start)).dt.date
    stand_in.loc[stand_in.index[:24], "status"] = "inactive"  # the forecast file must cover these too

    def small_fit(start, end):
        model = LightGBM(max_rounds=20).fit(features[dates < dates.max() - timedelta(days=6)])
        model.train_rows = len(features)
        return model

    monkeypatch.setattr(final_test, "build_holdout_features", lambda: None)
    # The comparison: the sample stations stand in for the 200, two fast baselines for the five models.
    monkeypatch.setattr(final_test, "dev_stations", lambda n: sorted(stand_in["station_id"].unique()))
    monkeypatch.setattr(final_test, "load_dev", lambda stations: features[dates < dates.max() - timedelta(days=6)])
    monkeypatch.setattr(final_test, "COMPARISON", ["seasonal_naive", "historical_mean"])
    monkeypatch.setattr(final_test, "fit_full", small_fit)
    monkeypatch.setattr(final_test.feature_store, "load", lambda *a, **k: stand_in.copy())
    for name in ("RESULTS_DIR", "PREDICTIONS_DIR", "MODELS_DIR"):
        monkeypatch.setattr(final_test, name, tmp_path / name)  # fresh folders, as in a new clone

    result = final_test.run(unlock=True)

    observed = (stand_in["status"] == "observed").sum()
    assert result["n"] == observed  # scored on observed hours only
    forecasts = pd.read_parquet(tmp_path / "PREDICTIONS_DIR" / "lightgbm-v1_test_test_all.parquet")
    assert len(forecasts) == len(stand_in)  # but a forecast for every hour, as the forecast contract asks
    assert list(forecasts.columns) == ["station_id", "target_start_utc", "origin_utc", "target", "prediction",
                                       "model_version", "data_mode", "fold"]
    assert (forecasts["prediction"] >= 0).all() and set(forecasts["data_mode"]) == {"test"}
    assert (tmp_path / "RESULTS_DIR" / "holdout_metrics.json").exists()
    assert (tmp_path / "MODELS_DIR" / "lightgbm-v1_final" / "model.txt").exists()
    assert len(result["commit"]) == 40
    comparison = pd.read_csv(tmp_path / "RESULTS_DIR" / "holdout_comparison_200.csv")
    assert comparison["model"].tolist() == ["lightgbm (all stations)", "seasonal_naive", "historical_mean"]
    assert (comparison["n"] == observed).all()  # every model scored on the same rows

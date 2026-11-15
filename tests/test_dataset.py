import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.dataset import load_station_hours

pytestmark = pytest.mark.skipif(
    not (config.RELEASE_DIR / "dev_sample_manhattan_2024_summer.parquet").exists(), reason="release not downloaded"
)


def test_joins_keep_one_row_per_station_hour():
    plain = load_station_hours("sample", with_calendar=False, with_weather=False)
    joined = load_station_hours("sample")
    assert len(joined) == len(plain)
    assert not joined.duplicated(["station_id", "hour_utc"]).any()
    assert joined["weekday"].notna().all()
    assert joined["temperature_2m"].notna().all()


def test_no_observed_weather_in_loader():
    joined = load_station_hours("sample")
    assert not any(col.startswith("observed") for col in joined.columns)
    assert "available_at_utc" not in joined.columns


def test_filters():
    station = load_station_hours("sample", with_weather=False)["station_id"].iloc[0]
    one = load_station_hours("sample", stations=[station], start_local="2024-07-01", end_local="2024-07-07",
                             observed_only=True, with_calendar=False, with_weather=False)
    assert set(one["station_id"]) == {station}
    assert pd.to_datetime(one["date_local"]).min() >= pd.Timestamp("2024-07-01")
    assert pd.to_datetime(one["date_local"]).max() <= pd.Timestamp("2024-07-07")
    assert (one["status"] == "observed").all()


def test_loader_clips_forecast_to_physical_range():
    if not (config.RELEASE_DIR / "station_hours_dev.parquet").exists():
        pytest.skip("dev file not downloaded")
    station = load_station_hours("sample", with_calendar=False, with_weather=False)["station_id"].iloc[0]
    frame = load_station_hours("dev", stations=[station], with_calendar=False)
    assert frame["cloud_cover"].between(0, 100).all()
    assert (frame["precipitation"] >= 0).all() and (frame["wind_speed_10m"] >= 0).all()
    raw = pd.read_parquet(config.RELEASE_DIR / "weather_forecast.parquet")
    hours = set(frame["hour_utc"])
    assert raw[raw["hour_utc"].isin(hours)]["cloud_cover"].lt(0).any()  # the raw file has such values in this range

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


def test_loader_weather_comes_from_the_forecast_file():
    joined = load_station_hours("sample", with_calendar=False)
    forecast = pd.read_parquet(config.RELEASE_DIR / "weather_forecast.parquet").set_index("hour_utc")
    observed = pd.read_parquet(config.RELEASE_DIR / "weather_observed.parquet").set_index("hour_utc")
    hours = joined.drop_duplicates("hour_utc").set_index("hour_utc")
    loaded = hours["temperature_2m"].to_numpy()
    assert (loaded == forecast.loc[hours.index, "temperature_2m"].to_numpy()).all()
    assert not (loaded == observed.loc[hours.index, "temperature_2m"].to_numpy()).all()


def test_clipping_keeps_missing_values_missing():
    import duckdb

    from pedalpulse.dataset import WEATHER_SQL

    with duckdb.connect() as con:
        con.execute("CREATE TABLE w AS SELECT * FROM (VALUES (NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE),"
                    " (1.0, -0.1, -1.0, 101.0)) t(temperature_2m, precipitation, wind_speed_10m, cloud_cover)")
        rows = con.execute(f"SELECT {', '.join(WEATHER_SQL.values())} FROM w").fetchall()
    assert rows[0] == (None, None, None, None)
    assert rows[1] == (1.0, 0.0, 0.0, 100.0)


def test_repeated_fall_back_hour_is_not_observed():
    frame = load_station_hours("sample", with_weather=False)
    calendar = pd.read_parquet(config.RELEASE_DIR / "calendar.parquet")
    repeated = calendar[(calendar["hour_local"] == 1) & (calendar["local_hours_in_day"] == 25)
                        & (calendar["utc_offset_h"] == -5)]["hour_utc"]
    assert len(repeated) == 3
    marked = frame[frame["hour_utc"].isin(repeated)]
    assert (marked["status"] != "observed").all()
    assert (frame.loc[frame["status"] == "dst_repeat", "pickups"] == 0).all()

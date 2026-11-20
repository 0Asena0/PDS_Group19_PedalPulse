import pandas as pd
import pytest

from pedalpulse import config
from pedalpulse.weather import LeakageError, assert_available_at_origin, forecast_origin, select_forecast


def forecast_rows(hours_local, lead_hours):
    hour_utc = pd.DatetimeIndex(hours_local).tz_localize(config.TIMEZONE).tz_convert("UTC")
    issued = hour_utc - pd.Timedelta(hours=lead_hours)
    return pd.DataFrame({
        "hour_utc": hour_utc,
        "issued_at_utc": issued,
        "available_at_utc": issued + pd.Timedelta(hours=config.FORECAST_PUBLICATION_DELAY_H),
    })


def test_day2_is_available_for_every_hour_of_the_target_day():
    hours = pd.date_range("2024-07-04 00:00", "2024-07-04 23:00", freq="h")
    assert_available_at_origin(forecast_rows(hours, lead_hours=48))


def test_day1_for_23_local_leaks_and_is_rejected():
    rows = forecast_rows(pd.DatetimeIndex(["2024-07-04 23:00"]), lead_hours=24)
    with pytest.raises(LeakageError):
        assert_available_at_origin(rows)


def test_day2_holds_on_dst_change_days():
    for day in ("2024-03-10", "2024-11-03"):
        hours = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h", tz=config.TIMEZONE)
        utc = pd.Series(hours.tz_convert("UTC")).drop_duplicates()
        rows = pd.DataFrame({"hour_utc": utc})
        rows["available_at_utc"] = rows["hour_utc"] - pd.Timedelta(hours=48 - config.FORECAST_PUBLICATION_DELAY_H)
        assert_available_at_origin(rows)


def test_origin_is_local_midnight_of_target_day():
    hour = pd.Series(pd.to_datetime(["2024-07-05 03:00"]).tz_localize("UTC"))  # 23:00 local on July 4
    assert forecast_origin(hour).iloc[0] == pd.Timestamp("2024-07-04 04:00", tz="UTC")


def test_model_priority_prefers_gfs_and_falls_back_per_hour():
    hour = pd.Timestamp("2024-01-10 12:00", tz="UTC")
    values = {variable: 1.0 for variable in config.WEATHER_VARIABLES}
    models = pd.DataFrame([
        {"hour_utc": hour, "model": "jma_gsm", **values},
        {"hour_utc": hour, "model": "gfs_seamless", **values | {"precipitation": None}},
        {"hour_utc": hour + pd.Timedelta(hours=1), "model": "gfs_seamless", **values},
        {"hour_utc": hour + pd.Timedelta(hours=1), "model": "jma_gsm", **values},
    ])
    chosen = select_forecast(models).set_index("hour_utc")["model"]
    assert chosen[hour] == "jma_gsm"
    assert chosen[hour + pd.Timedelta(hours=1)] == "gfs_seamless"
    assert chosen.isna().sum() == len(chosen) - 2


def test_built_forecast_respects_origin():
    candidates = [config.PROCESSED_DIR / "weather_forecast.parquet", config.RELEASE_DIR / "weather_forecast.parquet"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("weather not built or downloaded")
    forecast = pd.read_parquet(path)
    assert_available_at_origin(forecast)
    assert (forecast["issued_at_utc"] == forecast["hour_utc"] - pd.Timedelta(hours=48)).all()
    assert forecast["hour_utc"].is_unique

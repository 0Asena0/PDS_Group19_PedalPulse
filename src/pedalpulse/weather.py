"""Open-Meteo weather for one point (Central Park): day-ahead forecasts and observed reanalysis."""

import time

import pandas as pd
import requests

from . import config
from .timeutil import grid_bounds_utc

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_FILE = "weather_forecast.parquet"
FORECAST_MODELS_FILE = "weather_forecast_models.parquet"
OBSERVED_FILE = "weather_observed.parquet"


class LeakageError(AssertionError):
    pass


def get_json(url: str, params: dict, attempts: int = 4) -> dict:
    for attempt in range(1, attempts + 1):
        response = requests.get(url, params=params, timeout=120)
        if response.status_code == 200:
            return response.json()
        if attempt == attempts:
            response.raise_for_status()
        time.sleep(20 * attempt)
    raise RuntimeError("unreachable")


def request_window() -> tuple[str, str]:
    first, last = grid_bounds_utc()
    return first.date().isoformat(), last.date().isoformat()


def yearly_windows(start: str, end: str) -> list[tuple[str, str]]:
    windows = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        windows.append((max(start, f"{year}-01-01"), min(end, f"{year}-12-31")))
    return windows


def hourly_frame(payload: dict, suffix: str = "") -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({"hour_utc": pd.to_datetime(hourly["time"]).tz_localize("UTC")})
    for variable in config.WEATHER_VARIABLES:
        frame[variable] = pd.to_numeric(pd.Series(hourly[f"{variable}{suffix}"]), errors="coerce")
    return frame


def point_params() -> dict:
    return {
        "latitude": config.WEATHER_POINT["latitude"],
        "longitude": config.WEATHER_POINT["longitude"],
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def fetch_forecast_model(model: str) -> pd.DataFrame:
    suffix = f"_{config.FORECAST_LEAD}"
    parts = []
    for start, end in yearly_windows(*request_window()):
        params = point_params() | {
            "hourly": ",".join(f"{v}{suffix}" for v in config.WEATHER_VARIABLES),
            "models": model,
            "start_date": start,
            "end_date": end,
        }
        parts.append(hourly_frame(get_json(PREVIOUS_RUNS_URL, params), suffix))
    frame = pd.concat(parts, ignore_index=True)
    frame["model"] = model
    return frame


def fetch_observed() -> pd.DataFrame:
    start, end = request_window()
    params = point_params() | {
        "hourly": ",".join(config.WEATHER_VARIABLES),
        "models": "era5",
        "start_date": start,
        "end_date": end,
    }
    frame = hourly_frame(get_json(ARCHIVE_URL, params))
    frame["source"] = "era5"
    return frame


def select_forecast(models: pd.DataFrame) -> pd.DataFrame:
    """Per hour: the first model in FORECAST_MODELS that has all variables; otherwise empty values."""
    first, last = grid_bounds_utc()
    hours = pd.DataFrame({"hour_utc": pd.date_range(first, last, freq="h", tz="UTC")})
    complete = models.dropna(subset=list(config.WEATHER_VARIABLES)).copy()
    complete["priority"] = complete["model"].map({m: i for i, m in enumerate(config.FORECAST_MODELS)})
    best = complete.sort_values(["hour_utc", "priority"]).drop_duplicates("hour_utc").drop(columns="priority")
    frame = hours.merge(best, on="hour_utc", how="left")
    # A few model values lie just outside the physical range (e.g. cloud cover -1 or 101 %); clip them.
    frame["precipitation"] = frame["precipitation"].clip(lower=0)
    frame["wind_speed_10m"] = frame["wind_speed_10m"].clip(lower=0)
    frame["cloud_cover"] = frame["cloud_cover"].clip(0, 100)
    frame["model"] = frame["model"].astype("string")
    frame["lead"] = config.FORECAST_LEAD
    frame["issued_at_utc"] = frame["hour_utc"] - pd.Timedelta(hours=config.FORECAST_ISSUE_OFFSET_H)
    frame["available_at_utc"] = frame["issued_at_utc"] + pd.Timedelta(hours=config.FORECAST_PUBLICATION_DELAY_H)
    return frame


def forecast_origin(hour_utc: pd.Series) -> pd.Series:
    """00:00 America/New_York of the target hour's local date, in UTC."""
    local_day = hour_utc.dt.tz_convert(config.TIMEZONE).dt.tz_localize(None).dt.normalize()
    return local_day.dt.tz_localize(config.TIMEZONE).dt.tz_convert("UTC")


def assert_available_at_origin(forecast: pd.DataFrame) -> None:
    rows = forecast.dropna(subset=["available_at_utc"])
    late = rows[rows["available_at_utc"] > forecast_origin(rows["hour_utc"])]
    if len(late):
        raise LeakageError(f"{len(late)} forecast rows become available after the forecast origin, e.g. {late.iloc[0].to_dict()}")


def build_weather(force: bool = False) -> None:
    forecast_path = config.PROCESSED_DIR / FORECAST_FILE
    observed_path = config.PROCESSED_DIR / OBSERVED_FILE
    if forecast_path.exists() and observed_path.exists() and not force:
        print("skip weather (done)")
        return
    models = pd.concat([fetch_forecast_model(model) for model in config.FORECAST_MODELS], ignore_index=True)
    models.to_parquet(config.PROCESSED_DIR / FORECAST_MODELS_FILE, index=False, compression="zstd")
    forecast = select_forecast(models)
    assert_available_at_origin(forecast)
    forecast.to_parquet(forecast_path, index=False, compression="zstd")
    observed = fetch_observed()
    first, last = grid_bounds_utc()
    observed = observed[observed["hour_utc"].between(pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC"))]
    observed.to_parquet(observed_path, index=False, compression="zstd")
    print(f"weather: forecast model use {forecast['model'].value_counts(dropna=False).to_dict()}")

"""Central settings. All paths are relative to the repository root."""

import os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked, small files (manifest, frozen cohort) live in the repository.
TRACKED_DIR = REPO_ROOT / "data"

# Large data lives outside git. Override with PEDALPULSE_DATA (relative to the repo root or absolute).
DATA_DIR = (REPO_ROOT / os.environ.get("PEDALPULSE_DATA", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RELEASE_DIR = DATA_DIR / "release"
DUCKDB_TEMP_DIR = DATA_DIR / "duckdb_tmp"

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_VERSION = "data-v2"
TIMEZONE = "America/New_York"

TRIP_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
STUDY_START = date(2023, 1, 1)
STUDY_END = date(2026, 3, 31)  # inclusive, local date
EXCLUDED_ARCHIVES = {
    "202604-citibike-tripdata.zip": "outside study period; unusually small file (0.16 GB vs 0.73 GB for 04/2025)",
}

COHORT_WINDOW = (date(2024, 7, 1), date(2024, 9, 30))
COHORT_SHARE = 0.80
COHORT_SHARES_REPORTED = (0.50, 0.80, 0.95)
SELECTION_CUTOFF = date(2024, 9, 30)  # last local date usable for selection decisions

INACTIVE_MIN_DAYS = 3

DEV_END = date(2025, 9, 30)  # station_hours_dev covers STUDY_START..DEV_END
HOLDOUT_START = date(2025, 10, 1)
EDA_END = date(2025, 6, 30)
DEV_SAMPLE_WINDOW = (date(2024, 6, 1), date(2024, 8, 31))
DEV_SAMPLE_BOROUGH = "Manhattan"

WEATHER_POINT = {"latitude": 40.7812, "longitude": -73.9665, "name": "Central Park"}
WEATHER_VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover")
FORECAST_MODELS = ("gfs_seamless", "jma_gsm")
FORECAST_LEAD = "previous_day2"
FORECAST_ISSUE_OFFSET_H = 48
FORECAST_PUBLICATION_DELAY_H = 6

# NYC Department of City Planning, "Borough Boundaries" (water areas excluded), NYC Open Data gthc-hcne.
BOROUGH_BOUNDARIES_URL = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"
BOROUGH_NEAREST_MAX_M = 500

DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = 3


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RELEASE_DIR, DUCKDB_TEMP_DIR, FIGURES_DIR, TRACKED_DIR):
        path.mkdir(parents=True, exist_ok=True)

"""Features for all cohort stations, built in chunks and stored as Parquet.

Usage: uv run python -m pedalpulse.feature_store

The full dev table has 47 million rows; with all features in memory at once it does not fit on a laptop.
Every feature of a station depends only on that station's own past, so building the stations in chunks
gives exactly the same values as building them all at once.
"""
from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import pandas as pd

from . import config
from .dataset import load_station_hours, release_path
from .features import build_features

FEATURE_DIR = config.DATA_DIR / "features"


def build(split: str = "dev", chunk: int = 150) -> None:
    stations = pd.read_csv(release_path("cohort.csv"), dtype={"station_id": str})["station_id"].sort_values().tolist()
    station_table = pd.read_parquet(release_path("stations.parquet"))
    out_dir = FEATURE_DIR / f"{split}_{config.DATA_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(stations), chunk):
        part = out_dir / f"part-{i // chunk:03d}.parquet"
        if part.exists():
            continue  # resume after an interruption
        start = time.time()
        frame = build_features(load_station_hours(split, stations=stations[i:i + chunk]), station_table)
        # float32 halves the memory; counts and features this small lose nothing at 7 significant digits.
        floats = frame.select_dtypes("float64").columns
        frame[floats] = frame[floats].astype("float32")
        frame.to_parquet(part, index=False, compression="zstd")
        print(f"{part.name}: {len(frame):,} rows, {time.time() - start:.0f} s", flush=True)


def load(columns: list[str], start_local: str | None = None, end_local: str | None = None,
         split: str = "dev", observed_only: bool = False) -> pd.DataFrame:
    """Read only the needed columns and days from the stored features."""
    where = []
    if start_local:
        where.append(f"date_local >= DATE '{start_local}'")
    if end_local:
        where.append(f"date_local <= DATE '{end_local}'")
    if observed_only:
        where.append("status = 'observed'")
    path = (FEATURE_DIR / f"{split}_{config.DATA_VERSION}" / "*.parquet").as_posix()
    query = f"SELECT {', '.join(columns)} FROM read_parquet('{path}') {'WHERE ' + ' AND '.join(where) if where else ''}"
    with duckdb.connect() as con:
        con.execute("SET enable_progress_bar = false")
        return con.execute(query).df()


def load_matrix(to_matrix, start_local: str, end_local: str, n_columns: int, split: str = "dev"):
    """Observed training rows as one float32 matrix, filled month by month so no full table sits in memory.

    to_matrix turns a chunk of the feature table into its rows of the matrix (e.g. models.lgb_matrix).
    Returns the matrix, the pickups and the local dates, all in the same row order.
    """
    n = int(load(["count(*) AS n"], start_local, end_local, split, observed_only=True)["n"].iloc[0])
    x = np.empty((n, n_columns), dtype="float32")
    y = np.empty(n, dtype="float32")
    dates = np.empty(n, dtype="datetime64[D]")
    pos = 0
    for month in pd.period_range(start_local, end_local, freq="M"):
        first = max(month.start_time, pd.Timestamp(start_local)).date()
        last = min(month.end_time, pd.Timestamp(end_local)).date()
        chunk = load(["*"], str(first), str(last), split, observed_only=True)
        x[pos:pos + len(chunk)] = to_matrix(chunk)
        y[pos:pos + len(chunk)] = chunk["pickups"].to_numpy("float32")
        dates[pos:pos + len(chunk)] = pd.to_datetime(chunk["date_local"]).to_numpy("datetime64[D]")
        pos += len(chunk)
    assert pos == n, f"read {pos} rows, expected {n}"
    return x, y, dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=150)
    build(chunk=parser.parse_args().chunk)

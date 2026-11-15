"""Station metadata: names, coordinates, borough, first observation and ID/name changes."""

import json
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import shapely
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from . import config
from .db import connect, sql_path

STATIONS_BASE = "stations_base.parquet"
NAME_CHANGES = "station_name_changes.parquet"
BOROUGHS_FILE = "borough_boundaries.geojson"


def fetch_boroughs() -> Path:
    target = config.RAW_DIR / BOROUGHS_FILE
    if not target.exists():
        response = requests.get(config.BOROUGH_BOUNDARIES_URL, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
    return target


METRES_PER_DEGREE = (111_320 * math.cos(math.radians(40.73)), 110_574)


def to_metres(lng: float, lat: float) -> tuple[float, float]:
    """Local equirectangular projection around New York; accurate to a few metres at city scale."""
    return lng * METRES_PER_DEGREE[0], lat * METRES_PER_DEGREE[1]


def project(geometry):
    return shapely.transform(geometry, lambda coords: coords * METRES_PER_DEGREE)


def assign_boroughs(stations: pd.DataFrame, geojson: Path) -> pd.DataFrame:
    features = json.loads(geojson.read_text())["features"]
    names = [feature["properties"]["boroname"] for feature in features]
    polygons = [project(shape(feature["geometry"])) for feature in features]
    tree = STRtree(polygons)
    boroughs, methods = [], []
    for lng, lat in zip(stations["lng"], stations["lat"]):
        if pd.isna(lng) or pd.isna(lat):
            boroughs.append(None)
            methods.append("no_coordinates")
            continue
        point = Point(*to_metres(lng, lat))
        hits = [index for index in tree.query(point) if polygons[index].contains(point)]
        if hits:
            boroughs.append(names[hits[0]])
            methods.append("inside")
            continue
        nearest = tree.nearest(point)
        if polygons[nearest].distance(point) <= config.BOROUGH_NEAREST_MAX_M:
            boroughs.append(names[nearest])
            methods.append("nearest_within_500m")
        else:
            boroughs.append("Outside NYC")
            methods.append("outside")
    stations = stations.copy()
    stations["borough"] = boroughs
    stations["borough_method"] = methods
    return stations


def month_occurrences(con, path) -> str:
    """Trip starts and ends of one month as (station, name, time, start flag, bike type, coordinates)."""
    source = f"read_parquet('{sql_path(path)}')"
    return f"""
        SELECT start_station_id AS station_id, start_station_name AS name, started_utc AS t_utc,
               started_local AS t_local, TRUE AS is_start, rideable_type, start_lat AS lat, start_lng AS lng
        FROM {source}
        UNION ALL
        SELECT end_station_id, end_station_name, ended_utc, ended_local, FALSE, rideable_type, end_lat, end_lng
        FROM {source}
        WHERE end_station_id IS NOT NULL
    """


def collect_month(con, path, cutoff_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occ = month_occurrences(con, path)
    span = con.execute(f"""
        SELECT station_id, min(t_utc) AS first_seen_utc, max(t_utc) AS last_seen_utc,
               count(*) FILTER (WHERE is_start) AS pickups_total
        FROM ({occ}) GROUP BY ALL
    """).df()
    names = con.execute(f"""
        SELECT station_id, name, count(*) AS n,
               count(*) FILTER (WHERE t_local < TIMESTAMP '{cutoff_end}') AS n_to_cutoff,
               min(t_local) AS first_seen_local, max(t_local) AS last_seen_local
        FROM ({occ}) WHERE name IS NOT NULL GROUP BY ALL
    """).df()
    coords = con.execute(f"""
        SELECT station_id, lat, lng,
               CASE WHEN is_start AND rideable_type = 'classic_bike' AND t_local < TIMESTAMP '{cutoff_end}' THEN 1
                    WHEN is_start AND rideable_type = 'classic_bike' THEN 2
                    ELSE 3 END AS source_rank,
               count(*) AS n
        FROM ({occ}) WHERE lat IS NOT NULL AND lng IS NOT NULL GROUP BY ALL
    """).df()
    return span, names, coords


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = values.argsort()
    values, weights = values.iloc[order].to_numpy(), weights.iloc[order].to_numpy()
    cumulative = weights.cumsum()
    half = cumulative[-1] / 2
    lower = values[(cumulative >= half).argmax()]
    if (cumulative == half).any():
        upper = values[(cumulative > half).argmax()]
        return (lower + upper) / 2
    return float(lower)


COORD_SOURCES = {1: "classic_starts_to_cutoff", 2: "classic_starts_after_cutoff", 3: "all_trip_ends"}


def station_coordinates(coords: pd.DataFrame) -> pd.DataFrame:
    """Median coordinates from the best available source; ranks 1 and 2 are classic-bike starts."""
    coords = coords.groupby(["station_id", "lat", "lng", "source_rank"], as_index=False)["n"].sum()
    rows = []
    for station_id, group in coords.groupby("station_id"):
        best = group["source_rank"].min()
        # rank 3 = all trip starts and ends, so it includes the classic starts of ranks 1 and 2
        chosen = group if best == 3 else group[group["source_rank"] == best]
        rows.append({
            "station_id": station_id,
            "lat": weighted_median(chosen["lat"], chosen["n"]),
            "lng": weighted_median(chosen["lng"], chosen["n"]),
            "coord_source": COORD_SOURCES[best],
        })
    return pd.DataFrame(rows)


def station_frames(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_end = (config.SELECTION_CUTOFF + timedelta(days=1)).isoformat()
    spans, names, coords = [], [], []
    for path in sorted((config.PROCESSED_DIR / "trips").glob("*.parquet")):
        span, name, coord = collect_month(con, path, cutoff_end)
        spans.append(span)
        names.append(name)
        coords.append(coord)
    span = pd.concat(spans).groupby("station_id").agg(
        first_seen_utc=("first_seen_utc", "min"), last_seen_utc=("last_seen_utc", "max"),
        pickups_total=("pickups_total", "sum"),
    ).reset_index()
    names = pd.concat(names).groupby(["station_id", "name"]).agg(
        n=("n", "sum"), n_to_cutoff=("n_to_cutoff", "sum"),
        first_seen_local=("first_seen_local", "min"), last_seen_local=("last_seen_local", "max"),
    ).reset_index()
    base = span.merge(station_coordinates(pd.concat(coords)), on="station_id", how="left")
    return base, names


def most_frequent_names(names: pd.DataFrame) -> pd.DataFrame:
    names = names.copy()
    has_cutoff = names.groupby("station_id")["n_to_cutoff"].transform("sum") > 0
    names["weight"] = names["n_to_cutoff"].where(has_cutoff, names["n"])
    ranked = names.sort_values(["station_id", "weight", "name"], ascending=[True, False, True])
    top = ranked.drop_duplicates("station_id")[["station_id", "name"]]
    counts = names.groupby("station_id")["name"].nunique().rename("n_names")
    return top.merge(counts, on="station_id")


def name_changes(names: pd.DataFrame) -> pd.DataFrame:
    by_id = names[names.groupby("station_id")["name"].transform("nunique") > 1].assign(change_type="id_with_several_names")
    by_name = names[names.groupby("name")["station_id"].transform("nunique") > 1].assign(change_type="name_with_several_ids")
    columns = ["change_type", "station_id", "name", "n", "first_seen_local", "last_seen_local"]
    return pd.concat([by_id, by_name])[columns].sort_values(columns[:2] + ["first_seen_local"]).reset_index(drop=True)


def build_stations(force: bool = False) -> None:
    target = config.PROCESSED_DIR / STATIONS_BASE
    if target.exists() and not force:
        print("skip stations (done)")
        return
    con = connect()
    base, names = station_frames(con)
    con.close()
    stations = base.merge(most_frequent_names(names), on="station_id", how="left")
    stations = assign_boroughs(stations, fetch_boroughs())
    stations["first_seen_hour_utc"] = stations["first_seen_utc"].dt.floor("h")
    stations = stations.sort_values("station_id").reset_index(drop=True)
    stations.to_parquet(target, index=False, compression="zstd")
    name_changes(names).to_parquet(config.PROCESSED_DIR / NAME_CHANGES, index=False, compression="zstd")
    print(f"stations: {len(stations):,} station IDs; boroughs: {stations['borough'].value_counts().to_dict()}")

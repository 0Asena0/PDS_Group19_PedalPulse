# Data contract (data-v1)

This contract describes the files in the GitHub release `data-v1`. Modelling (Asena) and the app and extensions (Sandra) work only against these files and columns. Changes to columns, types or rules require a new data version.

## Conventions

- Key of every hourly table: `hour_utc`, the start of a UTC hour, stored as a timestamp with time zone UTC. Join on `hour_utc` and, for station tables, `station_id`.
- Local time is `America/New_York`. Local columns (`date_local`, `hour_local`) are derived from `hour_utc` and are for grouping and display only.
- `station_id` is a string in the operator's format, for example `"6140.05"`. Never cast it to a number: `"6313.1"` and `"6313.10"` would collide.
- Study period: local dates 2023-01-01 to 2026-03-31, by trip start time. The UTC grid runs from `2023-01-01 05:00` to `2026-04-01 03:00` UTC (28,463 hours).
- All Parquet files use zstd compression. `SHA256SUMS` lists the checksum of every release file.

## Forecast origin (assumption E1)

A forecast for target day D is made at 00:00 America/New_York on day D, with zero latency. Everything up to the end of day D-1 counts as known. This is an assumption: Citi Bike publishes monthly files, not a live feed. The calendar column `origin_utc` gives the origin for every hour.

A value from source hour h may be used for a target hour only if h + 1 hour <= `origin_utc` of the target, that is, the source hour is complete at the origin. Define `lag_24h` as the same local hour on day D-1 (match `date_local` minus one day and `hour_local`); it always satisfies this rule. A plain 24-hour shift in UTC is not the same on DST change days: on the three fall-back days (2023-11-05, 2024-11-03, 2025-11-02) it maps the 23:00 hour to 00:00 local on D, which is not complete at the origin.

## Raw trips to targets

Raw trip times are naive local wall-clock times. They are localised to America/New_York and converted to UTC before aggregation.

- Spring forward: local times 02:00 to 02:59 do not exist. They are read with the offset before the change and land at 03:00 to 03:59 EDT.
- Fall back: local times 01:00 to 01:59 occur twice. They are assigned to the first occurrence (EDT, UTC-4), so the second occurrence (EST) receives no pickups. `tests/test_time.py` checks both rules.
- Trips across the repeated hour: a trip that starts in the first and ends in the second 01:00 hour has a local end time before its start. In `data-v1` these trips were dropped as `end_before_start`: all 1,123 trips of that rule (323 in 2023, 318 in 2024, 482 in 2025) are of this kind, none is a data error. Since the commit after `data-v1`, the pipeline moves such an end to the second occurrence and keeps the trip (`tests/test_trips.py`).
- In `data-v1` the second occurrence of 01:00 on the three fall-back nights is a structural zero for every station (6,069 rows, 5,871 of them `observed`). `load_station_hours` labels these rows `dst_repeat`, later builds store that status directly. Exclude them from training and evaluation like `inactive` rows.

Cleaning drops a trip only for: duplicate `ride_id`, missing start station, unparsable start or end time, end before start, start outside the study period. There is no duration filter. `filter_log.parquet` holds the count per rule and month.

## Files

### `station_hours_dev.parquet` and `station_hours_holdout.parquet`

One row per cohort station and UTC hour, from the station's first observed trip (start or end) to the end of the study period. Hours without trips are rows with 0.

| column | type | meaning |
|---|---|---|
| `station_id` | string | cohort station |
| `hour_utc` | timestamp UTC | start of the hour, key |
| `date_local` | date | local date of the hour |
| `hour_local` | int8 | local hour 0 to 23 |
| `pickups` | int32 | trips starting at the station in this hour (target) |
| `dropoffs` | int32 | trips ending at the station in this hour, by end time |
| `status` | string | `observed` or `inactive`; `load_station_hours` and later builds add `dst_repeat` |

- `dev`: local dates 2023-01-01 to 2025-09-30. `holdout`: 2025-10-01 to 2026-03-31. The files do not overlap and together cover the full grid.
- EDA and every modelling decision use only `dev`. Decisions about the sample itself (cohort, station attributes) use only data up to 2024-09-30.
- `status = observed`: the count is real, including true zeros.
- `status = inactive`: the hour lies on a local day that is part of a run of at least 3 consecutive local days without any pickup or dropoff at the station. These hours stay in the table so that the grid is complete.
- `status = dst_repeat`: second occurrence of the repeated fall-back hour, zero by construction (see above).
- In `data-v1` a run of empty days may cross the split date. This affects one station-day: station `7935.07` on 2025-09-30 (24 development rows) is `inactive` only because the run continues into October. Later builds end every run at the split date, so development status never depends on holdout data.
- `status` is label cleaning, not a feature. It uses information from the whole run, including days after the hour, and must not be used as a model input. Use it to exclude or down-weight training and evaluation rows.
- `pickups` and `dropoffs` of the target hour are labels. Neither is known at the origin. As features, only values of complete source hours are allowed (rule above, for example `lag_24h`); a same-hour `dropoffs` value is leakage. `load_station_hours` returns both columns, so the feature list must name allowed columns explicitly.
- `dropoffs` of trips that end at or after 2026-04-01 04:00 UTC (the end of the last grid hour) are not counted. Trips that started in 2022 are outside the study period, so dropoffs in the first hours of 2023-01-01 are slightly too low.

### `dev_sample_manhattan_2024_summer.parquet`

Same columns as `station_hours_dev`, limited to cohort stations in Manhattan and local dates 2024-06-01 to 2024-08-31. For laptops and quick tests.

### `stations.parquet`

One row per station ID seen in the cleaned trips.

| column | type | meaning |
|---|---|---|
| `station_id` | string | key |
| `name` | string | most frequent name in trip starts and ends up to 2024-09-30 (stations opened later: over all data) |
| `n_names` | int | number of distinct names seen for the ID |
| `lat`, `lng` | float64 | median start coordinates of classic-bike trips up to 2024-09-30 |
| `coord_source` | string | `classic_starts_to_cutoff`; fallback `classic_starts_after_cutoff`, then `all_trip_ends`. Despite its name, the last fallback uses all start and end coordinates of non-classic trips and trip ends (97 stations, none in the cohort) |
| `borough` | string | Manhattan, Brooklyn, Queens, Bronx, Staten Island, or `Outside NYC` |
| `borough_method` | string | `inside` (point in polygon), `nearest_within_500m` (point on a pier or shore line), `outside`, `no_coordinates` |
| `first_seen_utc`, `last_seen_utc` | timestamp UTC | first and last trip start or end at the station |
| `first_seen_hour_utc` | timestamp UTC | first grid hour of the station |
| `pickups_total` | int64 | cleaned trips starting at the station in the study period |
| `non_public_reason` | string | set for test, workshop and depot locations |
| `is_cohort` | bool | member of the frozen cohort |

Allowed as model inputs for cohort stations: `station_id`, `lat`, `lng`, `borough`, `first_seen_hour_utc` (all from data up to 2024-09-30 for cohort stations). `n_names`, `last_seen_utc` and `pickups_total` are computed over the whole study period and are descriptive only; using them as features leaks future information. `name` is for display. `non_public_reason` is empty (null) for public stations.

Borough boundaries: NYC Department of City Planning, Borough Boundaries (water areas excluded), NYC Open Data dataset `gthc-hcne`.

### `cohort.csv`

The frozen cohort (also tracked in the repository as `data/cohort.csv`): `station_id`, `name`, `borough`, `active_days`, `window_days`, `active_share`, `rule`, `cutoff_date`, `data_version`. Rule: at least one pickup on at least 80 % of local days from 2024-07-01 to 2024-09-30; test, workshop and depot stations excluded.

### `station_name_changes.parquet`

Log of IDs with more than one name (`id_with_several_names`) and names with more than one ID (`name_with_several_ids`). Columns: `change_type`, `station_id`, `name` (strings), `n` (int64, number of trip endpoints with this ID and name; a trip that starts and ends at the same station counts twice), `first_seen_local`, `last_seen_local` (naive local timestamps). One row per ID and name pair and change type; no nulls.

### `calendar.parquet`

One row per UTC hour of the study period.

| column | type | meaning |
|---|---|---|
| `hour_utc` | timestamp UTC | key |
| `local_time` | timestamp | local wall-clock time (repeats in the fall-back hour) |
| `date_local`, `hour_local` | date, int8 | local date and hour |
| `utc_offset_h` | int8 | -5 (EST) or -4 (EDT) |
| `weekday` | int8 | 0 = Monday to 6 = Sunday |
| `month`, `year` | int8, int16 | local month and year |
| `is_weekend` | bool | Saturday or Sunday |
| `is_holiday` | bool | US federal or New York State holiday, including observed days (`holidays` package, US, subdivision NY) |
| `holiday_name` | string | name, for example `Thanksgiving Day` or `New Year's Day (observed)` |
| `is_observed_holiday` | bool | holiday moved from a weekend |
| `is_federal_holiday` | bool | federal holiday only |
| `local_hours_in_day` | int8 | 23, 24 or 25 |
| `is_dst_change_day` | bool | local day with 23 or 25 hours |
| `origin_utc` | timestamp UTC | forecast origin of this hour: 00:00 local on `date_local` |

### `weather_forecast.parquet`

Day-ahead forecast for one point, Central Park (40.7812, -73.9665), from the Open-Meteo Previous Runs API with the `_previous_day2` lead. Units: degrees Celsius, mm, km/h, percent.

In `data-v1`, 19 of 28,463 hours carry model values just outside the physical range: `cloud_cover` -1 (5 hours) or 101 (8 hours) and `precipitation` -0.1 (6 hours). These are rounding artefacts of the weather models. `load_station_hours` clips them to 0 to 100 percent and to at least 0 mm and keeps missing values missing; when reading the file directly, clip the same way. The pipeline clips them for later builds.

| column | type | meaning |
|---|---|---|
| `hour_utc` | timestamp UTC | valid time, key |
| `temperature_2m`, `precipitation`, `wind_speed_10m`, `cloud_cover` | float64 | forecast values; empty when no model has all four |
| `model` | string | `gfs_seamless` if it has all four values for the hour, else `jma_gsm` |
| `lead` | string | `previous_day2` |
| `issued_at_utc` | timestamp UTC | `hour_utc` minus 48 h |
| `available_at_utc` | timestamp UTC | `issued_at_utc` plus 6 h (conservative publication delay) |

`available_at_utc <= origin_utc` holds for every row; `tests/test_weather.py` checks it. Whether weather enters the core model is decided in the modelling part.

### `weather_observed.parquet`

Observed weather for the same point from the Open-Meteo Historical Weather API (ERA5): `hour_utc` (timestamp UTC, key), `temperature_2m`, `precipitation`, `wind_speed_10m` (float64), `cloud_cover` (int64, percent), `source` (string). One row per grid hour, no nulls. Only for EDA and the oracle experiment. Never a model input for day-ahead forecasts.

### `daily_system.parquet`

Per local date over all cleaned trips (all stations): `date_local` (date, key), `trips`, `ebike_trips`, `member_trips`, `trips_1min_24h`, `trips_without_end_station`. The counts are stored as float64 but are whole numbers; cast with `astype(int)` if needed. One row for each of the 1,186 study days, including 2026-02-23 with 0 trips; no nulls.

### `filter_log.parquet`

Per month (`month`, string `YYYY-MM`, key), all int64 without nulls: `raw_rows`, `kept_rows`; dropped trips per rule `duplicate_ride_id`, `missing_start_station`, `unparsable_time`, `time_out_of_range`, `end_before_start`, `start_outside_study_period`; descriptive counts of kept trips `kept_missing_end_station`, `kept_start_ambiguous_hour`, `kept_start_nonexistent_hour`, `kept_duration_under_1min`, `kept_duration_over_24h`, `kept_station_id_normalised`. `raw_rows = kept_rows +` the six dropped columns.

## Licences

- Code: MIT (`LICENSE`). The MIT licence covers the code only, not the data files.
- Trip data: Citi Bike System Data, Lyft Bikes and Scooters, LLC, under the Citi Bike Data License Agreement (https://citibikenyc.com/data-sharing-policy). Section 2(b) forbids hosting or distributing the data as a stand-alone dataset and allows them as source material in non-commercial analyses, reports or studies. The release files are derived from the trips and serve only to reproduce this non-commercial course study. Do not pass them on as a dataset of their own. The licence has no exception for private repositories; whether a public release is covered by the study exception is not settled.
- Weather: Open-Meteo (https://open-meteo.com), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: one grid point, model selection per hour (GFS, else JMA), lead `previous_day2`, clipping to the physical range, aggregation to local days in the EDA.
- Borough boundaries: NYC Open Data, NYC Department of City Planning.

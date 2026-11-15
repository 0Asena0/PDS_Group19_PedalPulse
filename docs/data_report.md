# Data quality report (data-v1)

All numbers come from the pipeline run on 2026-11-15 and are reproduced by `uv run python -m pedalpulse report`, which writes them to `reports/data_summary.json`.

## Trips

| year | raw trips | kept trips | dropped |
|---|---:|---:|---:|
| 2023 | 35,106,986 | 35,088,190 | 18,796 |
| 2024 | 44,303,209 | 44,273,056 | 30,153 |
| 2025 | 45,771,102 | 45,750,349 | 20,753 |
| 2026 (Jan to Mar) | 5,981,588 | 5,977,495 | 4,093 |
| total | 131,162,885 | 131,089,090 | 73,795 |

The years are file months. 20,682 kept trips appear in the file of a later month than their start, for example trips from late February 2026 in the March 2026 file; they are kept and counted in the hour they started.

Cleaning removed 0.06 % of the raw rows:

| rule | dropped trips |
|---|---:|
| duplicate `ride_id` | 0 |
| missing start station | 72,397 |
| unparsable start or end time | 0 |
| end before start | 1,123 |
| start outside the study period | 275 |

The 275 trips outside the period are in the January 2023 file and started in December 2022.

Kept trips that need care:

- 347,032 trips (0.26 %) have no end station. They count as pickups and do not appear in `dropoffs`.
- 3,077,947 trips had a station ID written in a second format (`6313.1` for `6313.10`, or a trailing underscore) and were mapped to the operator's format.
- 9,441 trips started in the repeated hour of a fall-back night and were assigned to its first occurrence (EDT). No trip time falls into the non-existent spring-forward hour.

Duration sensitivity: 48,661 kept trips (0.037 %) last less than 1 minute or more than 24 hours, almost all of them more than 24 hours; only 5 trips are shorter than a minute. A 1 minute to 24 hour filter would therefore change the hourly counts very little. The default keeps all of them.

Sum check: for each of the 39 months, raw rows equal kept rows plus the dropped rows of every rule, and the hourly pickups of all stations equal the kept trips (`tests/test_outputs.py`).

## Stations and cohort

- 2,478 station IDs appear in the cleaned trips. 16 of them are test, workshop or depot locations (IDs such as `SYS016` Morgan Bike Mechanics, `Shop Morgan`, `Lab - NYC`, `LA Metro Demo 1`); the full list is in `reports/data_summary.json`.
- Cohort size for the activity rule (at least one pickup on the given share of local days from 2024-07-01 to 2024-09-30): 2,122 stations at 50 %, 2,072 at 80 %, 2,004 at 95 %. The cohort uses 80 %.
- Six non-public locations would have passed the 80 % rule and are excluded: `Lab - NYC`, `Lab - NYC - Monolith`, `SYS016`, `SYS033`, `SYS038`, `Shop Morgan`.
- Cohort by borough: Brooklyn 679, Manhattan 651, Queens 432, Bronx 310. No station in the data lies on Staten Island.
- Borough assignment for all 2,478 IDs: 2,383 inside a borough polygon, 1 within 500 m of one (`SYS028`, a non-public location), 94 outside New York City (90 Jersey City and Hoboken stations reached by trips from New York, plus four non-public IDs). No cohort station is outside the city.
- 75 IDs have more than one name and 37 names have more than one ID over the study period (`station_name_changes.parquet`).
- 465 cohort stations have their first trip after January 2023 and enter the grid later. Coordinates of all cohort stations come from classic-bike starts up to 2024-09-30.
- Cohort stations account for 125.0 million of the 131.1 million kept pickups (95.4 %).

## Grid

| file | rows | stations | first hour (UTC) | last hour (UTC) | size |
|---|---:|---:|---|---|---:|
| `station_hours_dev` | 47,122,121 | 2,072 | 2023-01-01 05:00 | 2025-10-01 03:00 | 167.5 MB |
| `station_hours_holdout` | 9,050,496 | 2,072 | 2025-10-01 04:00 | 2026-04-01 03:00 | 8.8 MB |
| `dev_sample_manhattan_2024_summer` | 1,432,053 | 651 | 2024-06-01 | 2024-08-31 (local) | 2.0 MB |

- Inactive hours: 1.36 % of the dev rows and 4.47 % of the holdout rows. 1,063 cohort stations have at least one inactive run; together these runs cover 43,623 station-days.
- In the holdout the inactive share rises from 3.6 % in October 2025 to 6.3 % in February 2026. 41 cohort stations are inactive for the whole of March 2026 and may have closed; this cannot be checked from the trip files.
- Share of observed hours with zero pickups: 48.2 % in dev and 55.7 % in holdout, which covers only autumn and winter months.
- Pickups per observed station-hour in dev: median 1, 99th percentile 22, maximum 216.
- Dropoffs in the first hour of 2023-01-01 at cohort stations: 678, against 1,048 in the same hour a week later. Part of the gap comes from trips that started in 2022 and are outside the study period.
- No trip starts between 2026-02-22 20:01 and 2026-02-24 05:59 local time. ERA5 shows 34.4 mm of precipitation at temperatures down to -1.5 degrees C and wind up to 35 km/h over these two days, which points to a snowstorm and a suspended system. 2026-02-23 is the only study day without any trip; it is in the holdout, and `daily_system` lists it with 0. Because the gap lasts more than a day but less than three, the hours stay `observed` with zero counts except at stations that were already in an inactive run.

## Weather

The Open-Meteo Previous Runs API returned these shares of hours with values (`_previous_day2`, Central Park):

| model | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| GFS (`gfs_seamless`) | temperature 99 %, other variables 0 % | 95 % for all four | 100 % | 100 % |
| JMA GSM (`jma_gsm`) | 100 % | 100 % | 100 % | 100 % |

GFS has no precipitation, wind or cloud cover before 2024-01-20 11:00 UTC. The forecast file therefore uses JMA GSM for all of 2023 and 463 hours of January 2024, and GFS from then on. No forecast hour and no observed hour is missing.

Mean absolute difference between the day-2 forecast and ERA5 at the same point: 1.48 degrees C for temperature, 0.15 mm for hourly precipitation, 3.88 km/h for wind speed, 22.9 percentage points for cloud cover. The model change in January 2024 means the forecast error is not constant over the period.

## Limits

- The forecast origin (00:00 local on the target day, no latency) is an assumption. The trip files are published monthly.
- The `inactive` rule needs three days without any trip. Shorter outages stay `observed` with zero counts, and a station that moves to a new ID looks like a closure followed by a new station.
- The cohort is fixed on summer 2024 activity. Stations opened later are not in the panel, and stations that closed later stay in it with inactive hours.
- Weather is one point for the whole city.
- Station coordinates and names are taken from trip records, not from an official station register.

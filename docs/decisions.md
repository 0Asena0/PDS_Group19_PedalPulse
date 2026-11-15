# Decision log (data)

The decisions behind the data files, each with its reason. Numbers that follow from them are in `docs/data_report.md`.

E1. Forecast origin. A forecast for day D is made at 00:00 America/New_York on D with zero latency, so all data up to the end of D-1 is known. `lag_24h` means the same local hour on D-1; a plain 24-hour UTC shift would reach into day D on fall-back days (see the data contract). This keeps the day-ahead task simple and identical for every hour. It is an assumption: the trip data arrive as monthly files, not as a live feed.

E2. Study period 2023-01-01 to 2026-03-31 by trip start time, NYC files only (`YYYYMM-citibike-tripdata.zip` and the yearly 2023 archive), no `JC-*` files. From 2023 on, the files share one schema with `ride_id`, and the period gives more than three years with several full seasons. April 2026 is left out: the archive is unusually small (0.16 GB, compared with 0.73 GB for April 2025), which suggests an incomplete month. The manifest lists it as excluded.

E3. Cohort from data up to 2024-09-30 only. A station is in the cohort when it has at least one pickup on at least 80 % of local days from 2024-07-01 to 2024-09-30. Test, workshop and depot locations are excluded after reviewing all station IDs and names. A fixed cohort keeps the panel stable, and choosing it before the evaluation period means no later information enters the sample. The report shows cohort sizes for 50, 80 and 95 %. The result is frozen in `data/cohort.csv`.

E4. Grid and status. Every cohort station gets every UTC hour from its first observed trip (start or end) to 2026-03-31. `observed` counts are real, including zeros. Hours on runs of at least 3 consecutive local days without any pickup or dropoff are `inactive` (closures, construction, missing docks). They stay in the table so the grid is complete. `status` is label cleaning and not a feature, because it uses information from the whole run.

E5. Targets. `pickups` by start station and start hour is the main target. `dropoffs` by end station and end hour are there for the net flow in Sandra's part. Trips that started in 2022 and ended in 2023 are outside the study period, so dropoffs in the first hours of 2023-01-01 are slightly too low.

E6. Time. Raw times are naive local times. They are localised to America/New_York and converted to UTC; the key is the UTC hour start, which avoids duplicate and missing local hours. The repeated local hour in November is assigned to its first occurrence (EDT); times in the missing spring hour are read with the winter offset. Both rules have tests.

E7. Cleaning. Trips are dropped only for duplicate `ride_id`, missing start station, unparsable time, end before start, or start outside the study period, each counted per month. There is no duration filter by default, because very short and very long trips still remove a bike from a dock. The effect of a 1 minute to 24 hour filter is reported as a sensitivity check. Station IDs written as floats (`6313.1` for `6313.10`) or with a trailing underscore are normalised to the operator's format, because otherwise one dock would appear as two stations.

E8. Weather. One point (Central Park, 40.7812, -73.9665), in degrees Celsius, mm and km/h, from Open-Meteo (CC BY 4.0). Observed weather comes from the Historical Weather API (ERA5) and is used only for EDA and an oracle experiment. Forecasts come from the Previous Runs API with the `_previous_day2` lead for every hour. The day-1 value is the forecast made 24 hours before the valid time; for the late hours of day D that run starts after the origin at 00:00 and would leak. The day-2 value is issued 48 hours before the valid time. With a conservative publication delay of 6 hours it is available at least 18 hours before the origin (fall-back days) and usually 24 hours or more. Model choice follows the coverage returned by the API: GFS where it provides all four variables, JMA GSM for the remaining hours (mainly 2023, when the GFS archive lacks precipitation, wind and cloud cover). `issued_at_utc` is valid time minus 48 hours and `available_at_utc` adds 6 hours. Whether weather enters the core model is decided in the modelling part (Asena).

E9. Calendar. Local hour, date, weekday, month, weekend, holiday including observed days (`holidays` package, US with New York State), holiday name, and DST change days.

E10. Station metadata. IDs stay strings. Name is the most frequent name up to 2024-09-30, coordinates are the median start coordinates of classic-bike trips up to that date (in the files almost all trips from a station carry the same dock coordinates, so the median removes the rare deviating points), and the borough comes from a point-in-polygon test against the official NYC borough boundaries (NYC Department of City Planning, NYC Open Data `gthc-hcne`). Points within 500 m of a borough polygon, typically piers and shore lines, take the nearest borough. ID and name changes are logged.

E11. Leakage protection by file split. The full table is split into `station_hours_dev` (2023-01 to 2025-09) and `station_hours_holdout` (2025-10 to 2026-03). EDA and all decisions use only `dev`; selection decisions use only data up to 2024-09-30. A separate file makes the rule hard to break by accident.

E12. Laptop file. Cohort stations in Manhattan, 2024-06-01 to 2024-08-31, below 50 MB, so that Asena and Sandra can work on their laptops without the full data.

E13. Data exchange. All Parquet files (zstd) and `SHA256SUMS` are attached to a GitHub release (`data-v1`, since E16 `data-v2`). No data in the git history; only `data/manifest.csv` and `data/cohort.csv` are tracked.

E14. Language. Code, documentation, commits and notebook in English.

E15. Corrections after `data-v1`. An audit on 2026-11-15 found these problems in the pipeline; they are fixed in the code. `data-v1` can be rebuilt exactly only from the tag `data-v1`.
- Trips across the repeated fall-back hour (local end before local start) were dropped as `end_before_start`: 1,123 trips, 0.0009 % of all trips. The pipeline now moves the end to the second occurrence and keeps them.
- The second occurrence of 01:00 on fall-back nights is a structural zero. It now gets `status = dst_repeat`; `load_station_hours` applies the same label to `data-v1`.
- Inactive runs could cross the split date, so one development station-day depended on holdout data. Runs now end at the split date.
- An interrupted grid build was taken as finished on the next run. The grid now writes a completion marker last and is rebuilt when it is missing. The loader's clipping kept turning missing weather into zeros; it now keeps missing values missing.
E16. Release `data-v2`. The group switches to a release built from the corrected code. Compared with `data-v1` it keeps the 1,123 trips across the repeated hour (+1,070 pickups and +1,099 dropoffs at cohort stations), labels 5,871 rows `dst_repeat`, turns 72 station-hours from `inactive` to `observed` at the split date and clips 19 forecast values. Row counts, cohort, calendar, observed weather and the laptop file are identical. `data-v1` stays published for reproducing earlier results.

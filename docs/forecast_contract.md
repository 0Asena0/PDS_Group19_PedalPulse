# Forecast contract

Format of the forecast files that modelling (B) writes and the app (C) reads. Dummy files and real forecasts use the same format, so the app does not change when real forecasts arrive.

## Conventions

Same as the [data contract](data_contract.md): keys in UTC, `station_id` as string, local time only for display. A forecast for local day D is made at `origin_utc` (00:00 America/New_York on D, assumption E1) and covers every hour of D, so 23, 24 or 25 rows per station and day.

## Columns

| column | type | meaning |
|---|---|---|
| `station_id` | string | cohort station |
| `target_start_utc` | timestamp UTC | start of the forecast hour; joins to `hour_utc` in the station-hour tables |
| `origin_utc` | timestamp UTC | when the forecast was made; same for all hours of a local day |
| `target` | string | `pickups` (main target); `dropoffs` only if a net flow model exists |
| `prediction` | float64 | expected count for the hour, >= 0, not rounded |
| `model_version` | string | for example `seasonal_naive-v1` or `lightgbm-v1` |
| `data_mode` | string | `dummy`, `validation` or `test` |
| `fold` | string | `f1` to `f4` for validation, `test` for the holdout, empty for dummy |

One row per `station_id`, `target_start_utc`, `target` and `model_version`. Forecasts exist for every cohort station hour in the period, also for `inactive` hours; the app takes `pickups`, `dropoffs` and `status` from the station-hour tables via `station_id` and `target_start_utc`.

- `dummy`: made-up numbers for building the app. The app must show that they are not real.
- `validation`: backtest on a validation window of a fold; the model only saw data up to the fold's training end.
- `test`: final forecasts for the holdout period.
- There is no live mode. All forecasts are historical backtests and the app must not call them live.

## Files

One Parquet file (zstd) per model version, mode and fold in `data/predictions/`, for example `lightgbm-v1_validation_f4_200.parquet` (the last part is the number of stations, or `all`). Like all data, the files stay out of git; real forecasts are shared as release assets. A quarter for all cohort stations has about 4.5 million rows per model.

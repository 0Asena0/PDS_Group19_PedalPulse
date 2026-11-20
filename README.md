# PDS_Group19_PedalPulse
PedalPulse – Day-ahead forecasting of hourly Citi Bike demand per station in NYC

## Quickstart

Setup (needs [uv](https://docs.astral.sh/uv/) and, for the download, the GitHub CLI `gh`):

```bash
git clone https://github.com/0Asena0/PDS_Group19_PedalPulse.git && cd PDS_Group19_PedalPulse
uv sync
gh release download data-v2 --dir data/release
```

Without `gh`, download the files from the release `data-v2` in the browser and put them into `data/release/`. Without uv, use `pip install -r requirements.txt` and then `pip install -e .` in a Python 3.12 environment. Check the files with `uv run python -m pedalpulse verify` (works on Windows too). `station_hours_holdout.parquet` can stay on the server until the final evaluation; `verify` and the tests skip files that are not downloaded.

Load the laptop file:

```python
import pandas as pd
df = pd.read_parquet("data/release/dev_sample_manhattan_2024_summer.parquet")
df.head()
```

With calendar and day-ahead forecast weather joined (one row per station and UTC hour; observed weather is never joined):

```python
from pedalpulse.dataset import load_station_hours
df = load_station_hours("dev", start_local="2024-01-01", end_local="2024-12-31", observed_only=True)
```

Splits are `sample`, `dev` and `holdout`. The full `dev` split with all columns needs several GB of RAM, so filter by date or `stations=[...]` where possible.

`load_station_hours` returns `pickups` and `dropoffs` of the target hour; both are labels and must not be used as features. With `observed_only=True` it drops `inactive` hours and the repeated 01:00 hour of fall-back nights (`dst_repeat`).

Columns, types and rules are in [`docs/data_contract.md`](docs/data_contract.md). Work only against this contract. EDA and all modelling decisions use `station_hours_dev`; `station_hours_holdout` is for the final evaluation only.

## Data files (release data-v2)

`data-v2` replaces `data-v1` (decision E16): it keeps the 1,123 trips across the repeated fall-back hour, stores that hour as `dst_repeat`, ends inactive runs at the split date and clips the 19 out-of-range forecast values. Use `data-v2` for all new work.

| file | content | rows | size | SHA-256 (first 12) |
|---|---|---:|---:|---|
| `station_hours_dev.parquet` | cohort station by UTC hour, 2023-01 to 2025-09 | 47,122,121 | 167.5 MB | `b4443af43b51` |
| `station_hours_holdout.parquet` | cohort station by UTC hour, 2025-10 to 2026-03 | 9,050,496 | 8.8 MB | `5be2096eb662` |
| `dev_sample_manhattan_2024_summer.parquet` | Manhattan cohort, 2024-06 to 2024-08 | 1,432,053 | 2.0 MB | `e2a75ad902b6` |
| `stations.parquet` | station metadata and cohort flag | 2,478 | 0.1 MB | `0891c870a37e` |
| `cohort.csv` | frozen cohort with rule and cutoff | 2,072 | 0.4 MB | `796add1cf826` |
| `station_name_changes.parquet` | ID and name changes | 226 | < 0.1 MB | `405b56a61e08` |
| `calendar.parquet` | calendar per UTC hour | 28,463 | 0.5 MB | `55e9c6beb601` |
| `weather_forecast.parquet` | day-ahead forecast, Central Park | 28,463 | 0.8 MB | `5adf5484314c` |
| `weather_observed.parquet` | observed weather (ERA5), EDA only | 28,463 | 0.3 MB | `59fd76c0dada` |
| `daily_system.parquet` | daily totals over all stations | 1,186 | < 0.1 MB | `c0c5dd82d649` |
| `filter_log.parquet` | cleaning counts per month | 39 | < 0.1 MB | `2f967d2ed51a` |

Full checksums are in `SHA256SUMS` in the release. Sources, download dates and archive checksums: [`data/manifest.csv`](data/manifest.csv).

## Rebuilding the data

The pipeline downloads about 20 GB (19 GiB) of trip archives and needs about 6 GB of disk for intermediate files. Data goes to `data/` (ignored by git) unless `PEDALPULSE_DATA` points elsewhere.

```bash
uv run python -m pedalpulse all                      # every step in order
uv run python -m pedalpulse load --months 2024-07    # one step for one month
uv run pytest                                        # tests, including checks on the outputs
```

Steps: `download`, `load`, `aggregate`, `calendar`, `stations`, `cohort`, `grid`, `weather`, `export`, `report` (`all` runs them in this order; `verify` only checks downloaded release files). Each step skips outputs that already exist; `--force` rebuilds them. The code after the tag `data-v1` contains corrections (see `docs/decisions.md`, E15); `git checkout data-v1` rebuilds the release exactly. `scripts/publish_release.sh` uploads the release files.

## Documentation

- [`docs/data_contract.md`](docs/data_contract.md): tables, columns, time convention, status rules
- [`docs/decisions.md`](docs/decisions.md): data decisions with reasons
- [`docs/data_report.md`](docs/data_report.md): data quality report with the numbers of the run
- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): exploratory analysis on the dev file; figures in `reports/figures/`
- [`docs/genai_log.md`](docs/genai_log.md): use of generative AI

## Licence

The code is under the MIT licence. It does not cover the data. Trip data: Citi Bike System Data, Lyft Bikes and Scooters, LLC, under the [Citi Bike Data License Agreement](https://citibikenyc.com/data-sharing-policy); the release holds no trip records, only aggregated counts and station metadata, published as source material of this non-commercial study (Section 2(b)). Do not pass the files on as a dataset of their own. Weather: [Open-Meteo](https://open-meteo.com), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Details in the data contract.

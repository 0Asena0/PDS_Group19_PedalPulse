# GenAI log

## Data part (Benjamin Schwing)

I used Claude (Anthropic) as a coding agent with shell and GitHub access, run through Hermes on my own Linux server. It wrote the code in `src/pedalpulse/`, the tests, the EDA notebook and the documents in `docs/`, ran the pipeline on the full data and committed under my account.

Prompt: a written brief (`AUFTRAG_person_A_daten.md`, in the appendix of the report) with a short start instruction: implement the data part completely, read the brief and the files it names, and work through it phase by phase. The design decisions in `docs/decisions.md` were set in the brief before the run; I took them from our proposal and the group chat (cohort chosen only from data up to 2024-09, forecast origin at midnight, day-2 weather forecast). Later instructions were a check before our meeting on 15 November, the fixes after the review below and the fix for the weather values.

How I checked the output:

- Unit tests with synthetic data for every cleaning rule, the DST rules, the nested 2023 archive, calendar samples, the forecast availability rule (including a day-1 case that has to fail) and the weather model fallback.
- Checks on the real outputs: per month, raw rows equal kept plus dropped rows, and the hourly pickups add up to the kept trips (all 39 months); unique keys; no grid rows before a station's first trip and no gaps after it; joins keep the row count; dev and holdout do not overlap and cover the whole period; release checksums.
- A fresh clone in an empty folder, once rebuilding July 2024 from the raw data and once the way Asena and Sandra start, only with the release files.
- Plausibility checks of the numbers: trips per year, filter counts, station ID formats, cohort thresholds, weather coverage per model and year.

Problems found and fixed:

- Some files write station IDs as floats (`6313.1` for `6313.10`) or with a trailing underscore, so one dock showed up as two stations. The IDs are now normalised and counted in the filter log.
- The first version of the station metadata needed about 7.7 GB of memory, more than the server allowed. It now uses per-month frequency tables and a weighted median.
- A first reason in the decision log ("e-bike GPS points scatter more") did not hold in the data; both bike types carry the dock coordinates. Corrected.
- Workshop and depot locations such as `SYS016` (Morgan Bike Mechanics) and `Shop Morgan` met the 80 % rule. I reviewed all non-numeric IDs, and they are now excluded.
- `daily_system` had no row for 2026-02-23, the only day without a single trip (snowstorm). Days without trips are now rows with 0, and a test checks that every day is there.
- The first fresh clone failed because the manifest could not be written while some months were missing. Fixed before the first push.
- A read-only review by a second model (GPT, OpenAI) found five problems: `lag_24h` as a plain 24-hour UTC shift leaks on the three fall-back days; `n_names`, `last_seen_utc` and `pickups_total` use the whole period and were not marked as non-features; the borough shares in the EDA came from summed per-day means instead of totals (Queens 6.8 % instead of 6.5 %); the 40+ bar was drawn as overlapping bars; the 2025 "summer" mean covered June only. All fixed in the contract, tests and notebook; the release data did not change.
- Asena noticed forecast values just outside the physical range (19 of 28,463 hours, e.g. cloud cover -1 or 101 %). The loader and the pipeline now clip them, and the contract describes it; the release file is unchanged.

What is mine and what is the tool's: Claude chose the code structure and wrote the code and the text drafts. The decisions and the responsibility for the results are mine.

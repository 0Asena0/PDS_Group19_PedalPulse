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
- On 15 November I asked Claude for a full audit of the data part against the course requirements, with a second read-only review by GPT-6.1 Sol (OpenAI, via Codex CLI). Both recomputed the numbers in the documents from the release files. Findings: the 1,123 trips dropped as "end before start" were valid trips across the repeated fall-back hour; the repeated hour itself was stored as observed zeros; one inactive run crossed the split date (24 development rows); an interrupted grid build counted as finished; the loader's clipping turned missing weather into zeros; target-hour `dropoffs` were not marked as a label; the licence note claimed a private repository; and several numbers in the texts were off (468 instead of 463 JMA hours, GFS from 12:00 instead of 11:00 UTC, two instead of six non-public locations passing the cohort rule, cohort complete from July instead of December 2024, winter zero share 58 instead of 56 %). The code, tests and documents are fixed; the release files are unchanged (decision E15).

What is mine and what is the tool's: Claude chose the code structure and wrote the code and the text drafts. The decisions and the responsibility for the results are mine.

## Modelling part (Asena Uenal)

I used Claude (Anthropic) as a coding agent in Claude Code on the web, in a cloud container with the repository and the release files. It wrote the code in `src/pedalpulse/` for features, folds, baselines, models, tuning, experiments, error analysis and the final test, the tests for it, and the drafts of the report sections 4 and 5. It ran the evaluations; I reviewed every commit and its diff before it was pushed under my account.

Prompt: I worked through the stations of our project roadmap one by one in German, in a running conversation. Decisions I set before the code was written: forecast origin as in the data contract (00:00, E1) with the roadmap's 23:00 rule as an experiment, four rolling-origin folds inside dev, the holdout locked in code until the team review, MAE as the main metric, a small tuning budget with a winner rule fixed in advance (better than the default in at least 3 of 4 folds), and the example stations drawn by a fixed rule before looking at any forecast.

How I checked the output:

- Tests for the leakage rules: a test changes all counts from the target day on and checks that the features of that day stay the same; lags on the fall-back and spring-forward days; missing history stays missing; folds without overlap or gap; the holdout refuses to open without an explicit unlock.
- Baselines and metrics checked against small examples computed by hand; the Poisson floor against a simulation.
- Three real station-hours looked up by hand in the release file and compared with the features (15 November).
- Building the features in chunks of stations, and building the holdout with a history window, both compared with one pass over the same data; the history test fails on purpose with a too short window.
- The 200-station sample against all 2,072 stations on fold f4 (WAPE 0.431 against 0.424).

Problems found and fixed:

- The first LightGBM setup (learning rate 0.05, at most 2,000 trees) hit the tree cap on the latest fold before early stopping. Changed to 0.1 with a cap of 4,000; on all stations it now stops at about 2,200 trees.
- Training on all stations ran out of memory as a pandas table. The data are now read month by month into one float32 matrix that is freed once LightGBM has binned it (peak about 9 GB).
- The tuning script used the current model default as its first trial. After the forest's default changed, a rerun would have drawn a different search than the saved one. The starting point is now fixed in the code and checked against the saved results.
- A few forecast weather values outside the physical range (cloud cover -1 or 101 %) showed up while building the features; Benjamin clipped them in the loader.
- `status` marks inactive days using later days, so it is never a model input; it is only used to drop inactive hours from training and scoring. A run of inactive days that crosses a fold cut-off is still classified with later days; this is described as a known residual in the report.

What is mine and what is the tool's: Claude chose the code structure and wrote the code, the tests and the text drafts. The modelling decisions, the review of every change and the responsibility for the results are mine.

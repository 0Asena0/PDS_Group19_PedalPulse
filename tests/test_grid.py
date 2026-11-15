"""Inactive-period rule on a tiny synthetic station."""

from datetime import date, timedelta

import duckdb
import pandas as pd

from pedalpulse import config
from pedalpulse.grid import build_inactive_days


def run_rule(active_dates):
    days = pd.date_range(config.HOLDOUT_START - timedelta(days=6), config.HOLDOUT_START + timedelta(days=5), freq="D")
    hours = pd.DataFrame({"hour_utc": days + pd.Timedelta(hours=12), "date_local": days.date, "hour_local": 8})
    pickups = pd.DataFrame({"station_id": "s1", "hour_utc": [pd.Timestamp(d) + pd.Timedelta(hours=12) for d in active_dates],
                            "pickups": 1})
    con = duckdb.connect()
    con.register("hours_df", hours)
    con.register("pickups_df", pickups)
    con.execute("CREATE TEMP TABLE hours AS SELECT * FROM hours_df")
    con.execute("CREATE TEMP TABLE members AS SELECT 's1' AS station_id, min(hour_utc) AS first_hour FROM hours")
    con.execute("CREATE TEMP TABLE pickups AS SELECT * FROM pickups_df")
    con.execute("CREATE TEMP TABLE dropoffs AS SELECT station_id, hour_utc, pickups AS dropoffs FROM pickups WHERE false")
    build_inactive_days(con)
    return sorted(r[0] for r in con.execute("SELECT date_local FROM inactive_days").fetchall())


def test_runs_shorter_than_three_days_stay_observed():
    first = config.HOLDOUT_START - timedelta(days=6)
    active = [first + timedelta(days=i) for i in range(12) if i not in (2, 3)]
    assert run_rule(active) == []


def test_three_day_run_is_inactive():
    first = config.HOLDOUT_START - timedelta(days=6)
    active = [first + timedelta(days=i) for i in range(12) if i not in (1, 2, 3)]
    assert run_rule(active) == [first + timedelta(days=i) for i in (1, 2, 3)]


def test_runs_do_not_cross_the_holdout_boundary():
    # last development day empty, first holdout days empty: the development day must stay observed
    dev_last = config.HOLDOUT_START - timedelta(days=1)
    first = config.HOLDOUT_START - timedelta(days=6)
    active = [first + timedelta(days=i) for i in range(12)
              if first + timedelta(days=i) not in (dev_last, config.HOLDOUT_START, config.HOLDOUT_START + timedelta(days=1),
                                                    config.HOLDOUT_START + timedelta(days=2))]
    assert run_rule(active) == [config.HOLDOUT_START + timedelta(days=i) for i in range(3)]

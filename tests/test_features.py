import numpy as np
import pandas as pd
import pytest

from pedalpulse.features import HISTORY_FEATURES, add_history_features

TZ = "America/New_York"


def station_hours(start="2024-10-01", end="2024-11-30", station="1.01"):
    """Synthetic station-hours shaped like the release files. pickups = running number, so every hour is unique."""
    # I use a running number instead of real counts: if a feature shows 1234, I know exactly which hour it came from.
    local_days = pd.date_range(start, end, freq="D", tz=TZ)
    hour_utc = pd.date_range(local_days[0].tz_convert("UTC"), (local_days[-1] + pd.Timedelta(days=1)).tz_convert("UTC"),
                             freq="h", inclusive="left")
    local = hour_utc.tz_convert(TZ)
    date_local = local.tz_localize(None).normalize()
    origin = pd.DatetimeIndex(date_local).tz_localize(TZ, ambiguous=True).tz_convert("UTC")
    return pd.DataFrame({
        "station_id": station,
        "hour_utc": hour_utc,
        "date_local": date_local,
        "hour_local": local.hour.astype("int8"),
        "pickups": np.arange(len(hour_utc), dtype="int32"),
        "origin_utc": origin,
    })


def row(df, date, hour, nth=0):
    return df[(df["date_local"] == pd.Timestamp(date)) & (df["hour_local"] == hour)].iloc[nth]


def test_lags_by_hand():
    # I pick one target hour (20 Oct, 8:00) and look up by hand what each feature should be.
    df = add_history_features(station_hours())
    target = row(df, "2024-10-20", 8)
    assert target["lag_1d"] == row(df, "2024-10-19", 8)["pickups"]
    assert target["lag_7d"] == row(df, "2024-10-13", 8)["pickups"]
    assert target["lag_14d"] == row(df, "2024-10-06", 8)["pickups"]
    assert np.isnan(target["lag_28d"])  # 2024-09-22 is before the data starts
    week = [row(df, f"2024-10-{d}", 8)["pickups"] for d in range(13, 20)]
    assert target["mean_same_hour_7d"] == pytest.approx(np.mean(week))
    day = df[df["date_local"] == pd.Timestamp("2024-10-19")]["pickups"].sum()
    assert target["day_total_1d"] == day


def test_missing_history_stays_missing():
    # On the first day there is no past at all. I check that the features are empty and not 0,
    # because "no data" is not the same as "no bikes rented".
    df = add_history_features(station_hours())
    first = df[df["date_local"] == pd.Timestamp("2024-10-01")]
    assert first[HISTORY_FEATURES].isna().all().all()


def test_no_future_information():
    # The cheating test: I change all counts from the target day on and check that the features of the
    # target day stay exactly the same. If they changed, the model would be looking into the future.
    base = station_hours()
    # I chose 4 Nov 2024 on purpose: the clocks went back the day before, which is where such bugs usually hide.
    target_day = pd.Timestamp("2024-11-04")
    changed = base.copy()
    changed.loc[changed["date_local"] >= target_day, "pickups"] += 1000
    a = add_history_features(base)
    b = add_history_features(changed)
    on_day = a["date_local"] == target_day
    pd.testing.assert_frame_equal(a.loc[on_day, HISTORY_FEATURES], b.loc[on_day, HISTORY_FEATURES])


def test_fall_back_uses_first_occurrence():
    # On 3 Nov 2024 the hour 1:00-2:00 happened twice. I check that the lag for the next day takes the first one.
    df = add_history_features(station_hours())
    first_one = row(df, "2024-11-03", 1, nth=0)
    second_one = row(df, "2024-11-03", 1, nth=1)
    assert second_one["hour_utc"] - first_one["hour_utc"] == pd.Timedelta(hours=1)
    assert row(df, "2024-11-04", 1)["lag_1d"] == first_one["pickups"]
    # both local hour 1 rows of the fall-back day see the same lag
    assert first_one["lag_1d"] == second_one["lag_1d"]


def test_spring_forward_hour_is_missing():
    # On 10 Mar 2024 the hour 2:00-3:00 did not exist. I check that "yesterday at 2:00" is empty on 11 Mar,
    # while 3:00 works normally.
    df = add_history_features(station_hours("2024-03-01", "2024-03-20"))
    assert np.isnan(row(df, "2024-03-11", 2)["lag_1d"])
    assert not np.isnan(row(df, "2024-03-11", 3)["lag_1d"])


def test_roadmap_origin_drops_late_hours():
    # With the roadmap rule (forecast at 23:00, data one hour late) yesterday 22:00 and 23:00 are not known yet.
    # I check that exactly those become empty, and that the day total of yesterday is empty too (the day is not over).
    df = add_history_features(station_hours(), origin_shift_h=-1, latency_h=1)
    day = df[df["date_local"] == pd.Timestamp("2024-10-20")].set_index("hour_local")
    assert not np.isnan(day.loc[21, "lag_1d"])
    assert day.loc[[22, 23], "lag_1d"].isna().all()
    assert day["day_total_1d"].isna().all()
    assert day["lag_2d"].notna().all()

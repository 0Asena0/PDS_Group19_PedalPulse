import numpy as np
import pandas as pd
import pytest

from pedalpulse.baselines import GlobalMean, HistoricalMean, PoissonBaseline, SeasonalNaive


def frame(rows):
    """Small station-hour table with the columns the baselines read."""
    df = pd.DataFrame(rows, columns=["station_id", "weekday", "hour_local", "pickups", "status"])
    df["month"] = 7
    df["is_weekend"] = df["weekday"] >= 5
    df["is_holiday"] = False
    df["lag_7d"] = np.nan
    return df


TRAIN = frame([
    ("A", 0, 8, 10, "observed"),
    ("A", 0, 8, 20, "observed"),
    ("A", 1, 8, 4, "observed"),
    ("A", 0, 9, 2, "observed"),
    ("B", 0, 8, 1, "observed"),
    # I add closed hours with huge counts: if they were used, every mean below would be far too high.
    ("A", 0, 8, 999, "inactive"),
    ("B", 0, 8, 999, "dst_repeat"),
])


def test_historical_mean_by_hand():
    model = HistoricalMean().fit(TRAIN)
    target = frame([("A", 0, 8, 0, "observed")])
    assert model.predict(target)[0] == pytest.approx(15)  # (10 + 20) / 2


def test_historical_mean_falls_back_step_by_step():
    # I ask for combinations the training data never saw and check each fallback level by hand.
    model = HistoricalMean().fit(TRAIN)
    target = frame([
        ("A", 3, 8, 0, "observed"),   # weekday 3 unseen -> station A at 8:00: (10 + 20 + 4) / 3
        ("A", 3, 17, 0, "observed"),  # hour 17 unseen too -> station A overall: (10 + 20 + 4 + 2) / 4
        ("C", 0, 8, 0, "observed"),   # station unseen -> mean of all observed training rows: 37 / 5
    ])
    assert model.predict(target) == pytest.approx([34 / 3, 9, 37 / 5])


def test_only_observed_hours_are_learned():
    assert GlobalMean().fit(TRAIN).predict(frame([("A", 0, 8, 0, "observed")]))[0] == pytest.approx(37 / 5)


def test_seasonal_naive_uses_last_week_and_falls_back():
    model = SeasonalNaive().fit(TRAIN)
    target = frame([("A", 0, 8, 0, "observed"), ("A", 0, 8, 0, "observed")])
    target["lag_7d"] = [6.0, np.nan]
    assert model.predict(target) == pytest.approx([6, 15])


def test_fit_ignores_validation_rows():
    # I change the rows we forecast and check the forecast does not move: the model only learned from TRAIN.
    model = HistoricalMean().fit(TRAIN)
    a = frame([("A", 0, 8, 0, "observed")])
    b = frame([("A", 0, 8, 500, "observed")])
    assert model.predict(a) == pytest.approx(model.predict(b))


def test_poisson_baseline_orders_stations_and_handles_new_ones():
    rng = np.random.default_rng(19)
    rows = [(s, d, h, rng.poisson(lam), "observed")
            for s, lam in (("busy", 8.0), ("quiet", 1.0)) for d in range(7) for h in range(24) for _ in range(3)]
    model = PoissonBaseline().fit(frame(rows))
    pred = model.predict(frame([("busy", 2, 8, 0, "observed"), ("quiet", 2, 8, 0, "observed"),
                                ("new", 2, 8, 0, "observed")]))
    assert (pred > 0).all()
    assert pred[0] > 4 * pred[1]
    # A station the model never saw gets no station effect, so it lands between the two.
    assert pred[1] < pred[2] < pred[0]

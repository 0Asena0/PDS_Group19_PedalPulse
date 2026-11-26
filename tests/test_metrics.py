import numpy as np
import pytest

from pedalpulse.metrics import poisson_floor, scores


def test_scores_by_hand():
    s = scores([0, 2, 4], [1, 2, 2])
    assert s["mae"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert s["wape"] == pytest.approx(3 / 6)
    assert s["bias"] == pytest.approx(-1 / 3)


def test_scores_refuse_missing_predictions():
    # I want every model scored on the same rows, so a gap in the predictions must be an error.
    with pytest.raises(ValueError):
        scores([1, 2], [1, np.nan])


def test_wape_undefined_without_demand():
    assert np.isnan(scores([0, 0], [1, 1])["wape"])


def test_poisson_floor_matches_simulation():
    # I check the formulas against brute force: draw many Poisson counts and measure the error directly.
    rng = np.random.default_rng(19)
    for lam in (0.3, 1.0, 2.5, 7.0):
        y = rng.poisson(lam, 400_000)
        floor = poisson_floor(np.full(10, lam))
        assert floor["mae_mean"] == pytest.approx(np.abs(y - lam).mean(), rel=0.01)
        assert floor["mae_median"] == pytest.approx(np.abs(y - np.median(y)).mean(), rel=0.01)

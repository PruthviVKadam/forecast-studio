"""Tests for the walk-forward engine — especially that it cannot leak the future."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import aggregate, make_splits, walk_forward
from core.models.baselines import NaiveForecaster


def test_make_splits_positions():
    # n=30, horizon=5, folds=3 -> first_split = 30-15 = 15, then 20, 25
    assert make_splits(30, 5, 3) == [15, 20, 25]


def test_make_splits_rejects_short_series():
    with pytest.raises(ValueError):
        make_splits(10, 5, 3)  # needs >= 8 + 15 = 23 points


def test_no_leakage_model_never_sees_future():
    # First 20 points are flat at 1; the future jumps to 1000.
    # A naive model trained only on the past must predict 1 — if it predicted ~1000
    # it would mean the future leaked into training.
    y = pd.Series([1.0] * 20 + [1000.0] * 5)
    results = walk_forward(y, lambda: NaiveForecaster(), horizon=5, n_folds=1)
    assert len(results) == 1
    # Reconstruct the forecast the same way to assert it ignored the spike.
    model = NaiveForecaster().fit(y.iloc[:20])
    assert np.allclose(model.predict(5), 1.0)
    # Naive vs the 1000-spike test gives a large, finite error (no peeking).
    assert results[0].rmse == pytest.approx(999.0)


def test_walk_forward_fold_count_and_aggregate():
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.normal(size=120)) + 100)
    results = walk_forward(y, lambda: NaiveForecaster(), horizon=5, n_folds=4)
    assert len(results) == 4
    agg = aggregate(results)
    assert agg["n_folds"] == 4
    assert agg["rmse"] >= 0
    assert {"rmse", "mape", "mase"} <= agg.keys()


def test_walk_forward_raises_on_too_short():
    with pytest.raises(ValueError):
        walk_forward(pd.Series(range(10)), lambda: NaiveForecaster(), horizon=5, n_folds=3)

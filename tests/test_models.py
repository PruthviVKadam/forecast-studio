"""Tests for the forecasters — interface, baselines, and XGBoost lag correctness."""

import numpy as np
import pandas as pd
import pytest

from core.models import default_models
from core.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from core.models.ml import XGBoostLagsForecaster
from core.models.statistical import ARIMAForecaster, ProphetForecaster


def _series(n=80):
    t = np.arange(n)
    return pd.Series(100 + 0.3 * t + 5 * np.sin(t / 5.0))


def test_naive_predicts_last_value():
    m = NaiveForecaster().fit(pd.Series([3.0, 7.0, 9.0]))
    assert np.allclose(m.predict(4), 9.0)


def test_seasonal_naive_repeats_season():
    m = SeasonalNaiveForecaster(season=3).fit(pd.Series([1, 2, 3, 4, 5, 6.0]))
    # last season is [4,5,6]; horizon 5 -> 4,5,6,4,5
    assert np.allclose(m.predict(5), [4, 5, 6, 4, 5])


def test_seasonal_naive_rejects_short_series():
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(season=10).fit(pd.Series([1, 2, 3.0]))


def test_xgboost_supervised_uses_only_past_lags():
    m = XGBoostLagsForecaster(n_lags=2)
    feats, targets = m._supervise(np.array([0, 1, 2, 3, 4, 5.0]))
    # row for target y[2]=2 must be [lag1=1, lag2=0]; for y[3]=3 -> [2,1]; ...
    assert feats[0].tolist() == [1.0, 0.0]
    assert feats[1].tolist() == [2.0, 1.0]
    assert targets.tolist() == [2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    "model",
    [
        NaiveForecaster(),
        SeasonalNaiveForecaster(5),
        ARIMAForecaster((1, 1, 1)),
        XGBoostLagsForecaster(8),
    ],
)
def test_models_predict_finite_horizon(model):
    model.fit(_series())
    out = model.predict(6)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))


def test_prophet_fits_and_predicts():
    out = ProphetForecaster().fit(_series(90)).predict(7)
    assert out.shape == (7,)
    assert np.all(np.isfinite(out))


def test_default_models_includes_baselines_and_factories_are_fresh():
    models = default_models(season=5)
    assert any(is_base for _, is_base in models.values())
    f1, _ = list(models.values())[0]
    assert f1() is not f1()  # each call builds a new instance

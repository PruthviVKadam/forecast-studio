"""Tests for core.metrics against hand-computed values."""

import numpy as np
import pytest

from core.metrics import mape, mase, rmse


def test_rmse_known_values():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    # errors [1, -1] -> sqrt(mean([1,1])) = 1
    assert rmse([0, 0], [1, -1]) == pytest.approx(1.0)
    # errors [3, 4] -> sqrt((9+16)/2) = sqrt(12.5)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_mape_known_values_and_skips_zeros():
    # |10/100| and |20/200| -> mean 0.1 -> 10%
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    # the zero actual is skipped; only the 100 vs 110 term counts
    assert mape([0, 100], [5, 110]) == pytest.approx(10.0)


def test_mase_scales_by_train_naive_error():
    # train naive (season 1) MAE = mean|diff([10,11,12,13])| = 1.0
    # forecast abs error = mean(|[14,15]-[14.5,15.5]|) = 0.5 -> MASE 0.5
    assert mase([14, 15], [14.5, 15.5], [10, 11, 12, 13], season=1) == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        mape([1, 2], [1])


def test_mase_rejects_short_training():
    with pytest.raises(ValueError):
        mase([1, 2], [1, 2], [5], season=1)

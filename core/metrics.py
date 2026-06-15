"""Forecast accuracy metrics. Pure functions, fully unit-tested.

- RMSE  : root mean squared error (same units as the series).
- MAPE  : mean absolute percentage error (%), skips zero actuals.
- MASE  : mean absolute scaled error — error relative to a seasonal-naive forecast
          computed ON THE TRAINING SET. MASE < 1 means "better than naive".
"""

from __future__ import annotations

import numpy as np


def _as_float(*arrays):
    return tuple(np.asarray(a, dtype=float).ravel() for a in arrays)


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_float(y_true, y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must be the same length.")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%). Observations with a zero actual are skipped."""
    yt, yp = _as_float(y_true, y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must be the same length.")
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def mase(y_true, y_pred, y_train, season: int = 1) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal-naive MAE.

    season=1 is the standard non-seasonal MASE (scale = mean |y_t - y_{t-1}| on train).
    """
    yt, yp, ytr = _as_float(y_true, y_pred, y_train)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must be the same length.")
    if season < 1 or len(ytr) <= season:
        raise ValueError("Training series too short for the requested season.")
    scale = np.mean(np.abs(ytr[season:] - ytr[:-season]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)

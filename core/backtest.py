"""Walk-forward (expanding-window) backtesting — the heart of the studio.

LEAKAGE IS THE FAILURE MODE. Each fold trains ONLY on data up to its split point and
predicts the next `horizon` steps; nothing downstream of the split is visible to the
model. A model is just a factory (a zero-arg callable returning a fresh forecaster),
so it is re-fit from scratch every fold — no state, scaler, or lag can leak across the
split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.metrics import mape, mase, rmse

MIN_TRAIN = 8  # minimum points in the first training window


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_end_idx: int
    rmse: float
    mape: float
    mase: float


def make_splits(n: int, horizon: int, n_folds: int) -> list[int]:
    """Split positions for an expanding-window backtest.

    Fold k trains on y[:split] and tests on y[split:split+horizon]. The last
    n_folds*horizon points form the test region, stepped by `horizon`.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    first_split = n - n_folds * horizon
    if first_split < MIN_TRAIN:
        raise ValueError(
            f"Series too short: need at least {MIN_TRAIN + n_folds * horizon} points "
            f"for {n_folds} folds × horizon {horizon}, got {n}."
        )
    return [first_split + k * horizon for k in range(n_folds)]


def walk_forward(
    y,
    model_factory: Callable[[], object],
    horizon: int = 5,
    n_folds: int = 4,
    season: int = 1,
) -> list[FoldResult]:
    """Run an expanding-window backtest and return per-fold metrics."""
    y = pd.Series(y).dropna()
    n = len(y)
    results: list[FoldResult] = []

    for k, split in enumerate(make_splits(n, horizon, n_folds)):
        train = y.iloc[:split]
        test = y.iloc[split:split + horizon]

        # Hard leakage guard: every training timestamp must precede every test timestamp.
        if not train.index[-1] < test.index[0]:
            raise RuntimeError("Leakage detected: train overlaps test.")

        model = model_factory()
        model.fit(train)
        forecast = np.asarray(model.predict(horizon), dtype=float)[:horizon]
        if forecast.shape[0] != len(test):
            raise RuntimeError(
                f"Model returned {forecast.shape[0]} predictions, expected {len(test)}."
            )

        results.append(
            FoldResult(
                fold=k,
                train_end_idx=split - 1,
                rmse=rmse(test, forecast),
                mape=mape(test, forecast),
                mase=mase(test, forecast, train, season),
            )
        )
    return results


def aggregate(results: list[FoldResult]) -> dict:
    """Mean of each metric across folds (NaN-safe)."""
    return {
        "rmse": float(np.nanmean([r.rmse for r in results])),
        "mape": float(np.nanmean([r.mape for r in results])),
        "mase": float(np.nanmean([r.mase for r in results])),
        "n_folds": len(results),
    }

"""Common forecaster interface. Every model is fit on a training Series and then
asked for `horizon` point forecasts — nothing more, so the backtester can treat them
all identically and re-fit them per fold."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    name: str = "base"

    def fit(self, y: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

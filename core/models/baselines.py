"""Baseline forecasters. These are mandatory in every leaderboard — a 'fancy' model
that cannot beat them is not worth shipping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class NaiveForecaster(Forecaster):
    """Predict the last observed value for every future step."""

    name = "Naive (last value)"

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self._last = float(pd.Series(y).iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last, dtype=float)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season (e.g. last 5 business days)."""

    def __init__(self, season: int = 5):
        if season < 1:
            raise ValueError("season must be >= 1.")
        self.season = season
        self.name = f"Seasonal naive (m={season})"

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        values = pd.Series(y).to_numpy(dtype=float)
        if len(values) < self.season:
            raise ValueError("Series is shorter than the seasonal period.")
        self._last_season = values[-self.season:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array([self._last_season[i % self.season] for i in range(horizon)], dtype=float)

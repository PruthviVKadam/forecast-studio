"""Gradient-boosted forecaster using lag features (recursive multi-step).

Leakage safety: the supervised frame for target y[t] uses ONLY past values
y[t-1] … y[t-k]. Multi-step forecasts feed the model its OWN previous predictions,
never future actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class XGBoostLagsForecaster(Forecaster):
    def __init__(self, n_lags: int = 10, n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.05):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.name = f"XGBoost (lags={n_lags})"

    def _supervise(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of [lag1 … lagk] -> target. Uses only values strictly before t."""
        features, targets = [], []
        for t in range(self.n_lags, len(values)):
            window = values[t - self.n_lags:t][::-1]  # [y[t-1], y[t-2], …, y[t-k]]
            features.append(window)
            targets.append(values[t])
        return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)

    def fit(self, y: pd.Series) -> "XGBoostLagsForecaster":
        import xgboost as xgb

        self._values = pd.Series(y).to_numpy(dtype=float)
        if len(self._values) <= self.n_lags + 1:
            raise ValueError("Series too short for the requested number of lags.")
        features, targets = self._supervise(self._values)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=2,
            random_state=42,
        )
        self._model.fit(features, targets)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        history = list(self._values[-self.n_lags:])
        preds = []
        for _ in range(horizon):
            window = np.asarray(history[-self.n_lags:][::-1], dtype=float).reshape(1, -1)
            yhat = float(self._model.predict(window)[0])
            preds.append(yhat)
            history.append(yhat)
        return np.asarray(preds, dtype=float)

"""Statistical forecasters: ARIMA (statsmodels) and Prophet."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .base import Forecaster


class ARIMAForecaster(Forecaster):
    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.name = f"ARIMA{order}"

    def fit(self, y: pd.Series) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        values = pd.Series(y).reset_index(drop=True).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ARIMA(values, order=self.order).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._res.forecast(steps=horizon), dtype=float)


class ProphetForecaster(Forecaster):
    """Prophet on a synthetic regular daily index (we compare positionally, so the
    actual calendar gaps in e.g. stock data don't matter)."""

    name = "Prophet"

    def fit(self, y: pd.Series) -> "ProphetForecaster":
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        values = pd.Series(y).to_numpy(dtype=float)
        frame = pd.DataFrame(
            {"ds": pd.date_range("2000-01-01", periods=len(values), freq="D"), "y": values}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False
            )
            model.fit(frame)
        self._model = model
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        forecast = self._model.predict(future)
        return forecast["yhat"].to_numpy(dtype=float)[-horizon:]

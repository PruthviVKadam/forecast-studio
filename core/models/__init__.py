"""Model registry. Each entry is (factory, is_baseline). Baselines are mandatory in
the leaderboard and cannot be removed in the UI."""

from __future__ import annotations

from typing import Callable

from .base import Forecaster
from .baselines import NaiveForecaster, SeasonalNaiveForecaster
from .ml import XGBoostLagsForecaster
from .statistical import ARIMAForecaster, ProphetForecaster

__all__ = [
    "Forecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ARIMAForecaster",
    "ProphetForecaster",
    "XGBoostLagsForecaster",
    "default_models",
    "BASELINE_NAMES",
]


def default_models(season: int = 5) -> dict[str, tuple[Callable[[], Forecaster], bool]]:
    return {
        NaiveForecaster().name: (lambda: NaiveForecaster(), True),
        f"Seasonal naive (m={season})": (lambda s=season: SeasonalNaiveForecaster(s), True),
        "ARIMA(1,1,1)": (lambda: ARIMAForecaster((1, 1, 1)), False),
        "Prophet": (lambda: ProphetForecaster(), False),
        "XGBoost (lags=10)": (lambda: XGBoostLagsForecaster(10), False),
    }


BASELINE_NAMES = [name for name, (_, is_base) in default_models().items() if is_base]

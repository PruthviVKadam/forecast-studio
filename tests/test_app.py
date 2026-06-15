"""Headless tests for the Dash app — callbacks are plain functions, so we call them
directly (no browser). Uses the committed SPY cache, so it runs offline."""

import plotly.graph_objects as go

import app as appmod
from app import on_run, run_study
from core.models import BASELINE_NAMES


def test_run_study_includes_baselines_and_is_ranked():
    series, rows, forecasts = run_study("SPY", ["ARIMA(1,1,1)"], horizon=5, n_folds=3)
    names = [r["Model"] for r in rows]
    for baseline in BASELINE_NAMES:
        assert baseline in names
    assert "ARIMA(1,1,1)" in names
    mases = [r["MASE"] for r in rows]
    assert mases == sorted(mases)  # ranked ascending by MASE
    assert all(len(fc) == 5 for fc in forecasts.values())
    assert len(series) > 100


def test_baselines_present_even_with_no_optional_models():
    _, rows, _ = run_study("SPY", [], horizon=5, n_folds=2)
    assert {r["Model"] for r in rows} >= set(BASELINE_NAMES)


def test_on_run_returns_figure_table_takeaway():
    fig, data, note = on_run(1, "SPY", None, ["ARIMA(1,1,1)"], 5, 3)
    assert isinstance(fig, go.Figure)
    assert data and isinstance(note, str) and len(note) > 0


def test_on_run_surfaces_errors_gracefully(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated data outage")

    monkeypatch.setattr(appmod, "get_series", boom)
    fig, data, note = on_run(1, "SPY", None, [], 5, 3)
    assert data == []
    assert "simulated data outage" in note

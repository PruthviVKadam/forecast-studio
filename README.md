# 📈 Forecasting Studio

An interactive **time-series model bake-off**. Pick a series, choose models, and the app runs
**walk-forward backtesting**, ranks the models by error, and overlays each forecast against a
held-out tail. Naive baselines are always included — and on a near-random-walk like a stock
index, beating them is *hard*. That honesty is the point.

**Live demo:** _deploy to Render / Hugging Face (Docker) — URL goes here._

---

## Problem → Approach → Result

- **Problem:** Forecasting is full of traps — data leakage, invalid cross-validation, and fancy
  models that quietly lose to a naive baseline.
- **Approach:** A leakage-free **walk-forward** engine re-fits every model from scratch on each
  expanding-window fold, scores it with **MASE / MAPE / RMSE**, and compares ARIMA, Prophet, and
  gradient-boosted lag models against naive and seasonal-naive baselines.
- **Result:** An honest leaderboard. The interview-worthy takeaway is usually *"the baseline is
  hard to beat"* — which is exactly what a good forecaster should know.

## Example leaderboard (SPY, copied from app output)

Cached daily SPY, 2024-06-13 → 2026-06-12 (501 points), horizon = 5, 4 folds:

| Model | MASE | MAPE % | RMSE |
| --- | --- | --- | --- |
| **ARIMA(1,1,1)** | **0.685** | 0.89 | 8.415 |
| Naive (last value) | 0.699 | 0.91 | 8.412 |
| XGBoost (lags=10) | 0.957 | 1.24 | 10.066 |
| Seasonal naive (m=5) | 1.122 | 1.46 | 11.965 |
| Prophet | 3.42 | 4.42 | 33.701 |

ARIMA only *just* edges out the naive last-value forecast; XGBoost and Prophet do worse. On a
near-efficient price series this is expected — and a model that can't clear the naive bar
(MASE ≥ 1, like seasonal-naive and Prophet here) shouldn't ship. *(Numbers reproduce from the
committed cache; they shift as data updates.)*

## Why there's no leakage

`core/backtest.py` treats each model as a **factory** (a zero-arg callable). On fold *k* it builds
a fresh model, fits it only on `y[:split]`, and predicts `y[split:split+horizon]`. Lags, scalers,
and Prophet fits are all rebuilt per fold, so nothing downstream of the split can leak in — and
the engine raises if a train window ever overlaps its test window.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
python app.py                     # http://localhost:8050
```

Works offline out of the box (the 3 preset series are cached in `data/`). Typing a custom ticker
fetches it via yfinance and caches it.

## Test

```bash
python -m pytest
```

24 tests: metrics vs hand-computed values, a **no-leakage** test (a model trained before a future
spike must not predict the spike), the **XGBoost supervised frame uses only past lags**, every
model's fit/predict contract, and the Dash callback (called directly, no browser).

## Project layout

```text
core/metrics.py     # MAPE / RMSE / MASE
core/backtest.py    # walk-forward engine + leakage guard
core/models/        # base + baselines + statistical (ARIMA/Prophet) + ml (XGBoost-lags) + registry
core/data.py        # yfinance fetch + parquet cache + offline fallback
app.py              # Dash UI + callbacks (server = WSGI entrypoint)
data/*.parquet      # committed offline cache (SPY, AAPL, MSFT)
tests/              # pytest suite
Dockerfile          # gunicorn serving image
```

## Deploy

Containerized (`Dockerfile`, serves on `$PORT`/7860 via gunicorn). Free options: **Render** (Web
Service → Docker) or **Hugging Face Spaces** (Docker SDK). See the repo owner's `ManualSteps.md`.

## Stack

Python · Dash · Plotly · statsmodels (ARIMA) · Prophet · XGBoost · scikit-learn · yfinance · Docker.

---
_Educational demo — not investment advice._

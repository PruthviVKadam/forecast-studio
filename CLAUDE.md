# Forecasting Studio — CLAUDE.md
Plotly Dash app benchmarking ARIMA / Prophet / XGBoost-lags vs naive baselines with walk-forward backtesting. Data: yfinance (cached). Python 3.11+.

## Commands
- Setup: pip install -r requirements-dev.txt
- Run: python app.py  (Dash on :8050; offline-capable via committed data/ caches)
- Test: python -m pytest  (backtest leakage check is the critical unit)
- Container: docker build -t forecast . && docker run -p 7860:7860 forecast

## Structure
- core/data.py (fetch + parquet cache + offline fallback) · core/backtest.py (walk-forward engine, model-agnostic)
- core/models/ (base + baselines + statistical + ml; one common fit/predict interface; registry in __init__)
- core/metrics.py (MAPE/RMSE/MASE) · app.py (Dash UI + callbacks; `server` is the WSGI entrypoint)
- data/SPY|AAPL|MSFT.parquet committed for offline demo; other tickers are gitignored

## Rules
- LEAKAGE IS THE FAILURE MODE: models are factories re-fit per fold; no scaler/lag/Prophet fit may see data past the split. backtest.py raises if train overlaps test. Any new feature needs a leakage check in tests (see test_backtest, test_models XGBoost supervised).
- Naive + seasonal-naive baselines are ALWAYS in the leaderboard and cannot be removed from the UI (BASELINE_NAMES).
- Leaderboard shows MASE, MAPE, RMSE per model (mean across folds), ranked by MASE; README copies the table verbatim from app output.
- Cache yfinance pulls (parquet, 24h TTL); the app must still demo offline from the committed cache.
- Prophet fits on a synthetic regular daily index (we compare positionally). It is optional in the UI and the slowest model.
- UI footer: educational demo, not investment advice.

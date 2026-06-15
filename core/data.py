"""Series loading with a Parquet cache and an offline fallback.

get_series() returns a daily price Series (DatetimeIndex). It serves a fresh cache if
one exists (< 24h old), otherwise tries yfinance and refreshes the cache, and if the
network is unavailable it falls back to whatever cache is on disk. A few preset series
are committed to data/ so the app and tests work fully offline.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
TTL_SECONDS = 24 * 3600

PRESETS = {
    "SPY": "S&P 500 ETF (SPY)",
    "AAPL": "Apple (AAPL)",
    "MSFT": "Microsoft (MSFT)",
}


def _cache_path(ticker: str) -> Path:
    return CACHE / f"{ticker.upper()}.parquet"


def _fetch_yfinance(ticker: str, period: str = "2y") -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    close = df["Close"]
    if hasattr(close, "columns"):  # yfinance MultiIndex columns -> single series
        close = close.iloc[:, 0]
    series = pd.Series(close.to_numpy(dtype=float).ravel(), index=pd.to_datetime(df.index))
    series.name = ticker.upper()
    return series.dropna()


def _save(series: pd.Series, path: Path) -> None:
    CACHE.mkdir(exist_ok=True)
    pd.DataFrame({"date": series.index, "value": series.to_numpy(dtype=float)}).to_parquet(
        path, index=False
    )


def _load(path: Path, name: str) -> pd.Series:
    df = pd.read_parquet(path)
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]), name=name)


def get_series(ticker: str, period: str = "2y", force_refresh: bool = False) -> pd.Series:
    ticker = ticker.upper()
    path = _cache_path(ticker)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < TTL_SECONDS

    if fresh and not force_refresh:
        return _load(path, ticker)

    try:
        series = _fetch_yfinance(ticker, period)
        _save(series, path)
        return series
    except Exception:
        if path.exists():  # offline: serve stale cache rather than fail
            return _load(path, ticker)
        raise

"""Price data acquisition: live download, CSV files, or synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Common exchange-traded oil benchmarks (Yahoo Finance tickers).
KNOWN_TICKERS = {
    "BZ=F": "Brent Crude Futures (ICE)",
    "CL=F": "WTI Crude Futures (NYMEX)",
    "HO=F": "Heating Oil Futures (NYMEX)",
    "RB=F": "RBOB Gasoline Futures (NYMEX)",
    "USO": "United States Oil Fund (ETF)",
    "BNO": "United States Brent Oil Fund (ETF)",
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and validate OHLC presence."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns: {missing}")
    if "Volume" not in df.columns:
        df = df.assign(Volume=np.nan)
    df = df[REQUIRED_COLUMNS + ["Volume"]].dropna(subset=["Close"])
    if df.empty:
        raise ValueError("price data contains no usable rows")
    return df.sort_index()


def fetch_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history for ``ticker`` from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(
        ticker, period=period, interval=interval, progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        raise ValueError(
            f"no data returned for {ticker!r}; check the ticker symbol "
            f"(e.g. {', '.join(KNOWN_TICKERS)}) and network access"
        )
    return _normalize(df)


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV history from a CSV with a date index column.

    Expects columns Date, Open, High, Low, Close and optionally Volume.
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return _normalize(df)


def synthetic_history(
    days: int = 500, seed: int | None = 42, start_price: float = 80.0
) -> pd.DataFrame:
    """Generate a plausible synthetic oil price series for offline use/tests.

    Geometric Brownian motion with a slow sinusoidal drift so the series has
    distinct trending and ranging regimes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    drift = 0.0008 * np.sin(2 * np.pi * t / 250)
    shocks = rng.normal(0.0, 0.02, size=days)
    log_returns = drift + shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    intraday = np.abs(rng.normal(0.0, 0.01, size=days))
    open_ = np.empty(days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0.0, 0.003, size=days - 1))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.integers(20_000, 80_000, size=days).astype(float)

    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )

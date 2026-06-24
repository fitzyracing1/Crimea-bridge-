"""Technical indicators computed from OHLCV data."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100 - 100 / (1 + rs)
    # When there were no losses at all in the window, RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = float("nan")
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands plus %B (position of price within the bands)."""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    percent_b = (close - lower) / width.replace(0.0, float("nan"))
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "percent_b": percent_b}
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (volatility), from High/Low/Close columns."""
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with every indicator the signal engine uses."""
    out = df.copy()
    close = out["Close"]
    out["sma_fast"] = sma(close, 20)
    out["sma_slow"] = sma(close, 50)
    out["rsi"] = rsi(close)
    macd_df = macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["histogram"]
    boll = bollinger(close)
    out["bb_upper"] = boll["upper"]
    out["bb_lower"] = boll["lower"]
    out["bb_percent_b"] = boll["percent_b"]
    out["atr"] = atr(out)
    return out

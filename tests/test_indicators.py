import numpy as np
import pandas as pd
import pytest

from oil_analyst.data import synthetic_history
from oil_analyst.indicators import atr, bollinger, compute_all, ema, macd, rsi, sma


@pytest.fixture()
def df():
    return synthetic_history(days=300, seed=7)


def test_sma_matches_manual_mean(df):
    result = sma(df["Close"], 20)
    expected = df["Close"].iloc[0:20].mean()
    assert result.iloc[19] == pytest.approx(expected)
    assert result.iloc[:19].isna().all()


def test_ema_converges_to_constant():
    constant = pd.Series([50.0] * 100)
    result = ema(constant, 10)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_rsi_bounds_and_extremes(df):
    values = rsi(df["Close"]).dropna()
    assert ((values >= 0) & (values <= 100)).all()

    rising = pd.Series(np.linspace(10, 100, 50))
    assert rsi(rising).iloc[-1] == pytest.approx(100.0)

    falling = pd.Series(np.linspace(100, 10, 50))
    assert rsi(falling).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_macd_histogram_is_difference(df):
    result = macd(df["Close"]).dropna()
    assert np.allclose(result["histogram"], result["macd"] - result["signal"])


def test_bollinger_band_ordering(df):
    bands = bollinger(df["Close"]).dropna()
    assert (bands["upper"] >= bands["mid"]).all()
    assert (bands["mid"] >= bands["lower"]).all()


def test_atr_positive(df):
    values = atr(df).dropna()
    assert (values > 0).all()


def test_compute_all_adds_expected_columns(df):
    out = compute_all(df)
    for col in [
        "sma_fast",
        "sma_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_percent_b",
        "atr",
    ]:
        assert col in out.columns
    # Original frame must not be mutated.
    assert "rsi" not in df.columns

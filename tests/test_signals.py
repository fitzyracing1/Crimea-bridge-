import numpy as np
import pandas as pd
import pytest

from oil_analyst.data import synthetic_history
from oil_analyst.signals import BUY, HOLD, SELL, Recommendation, recommend, signal_series


def make_trend(
    days: int, start: float, end: float, seed: int = 0, noise: float = 0.1
) -> pd.DataFrame:
    """OHLCV frame following a noisy linear trend."""
    rng = np.random.default_rng(seed)
    close = np.linspace(start, end, days) + rng.normal(0, noise, days)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    index = pd.bdate_range(end="2026-01-01", periods=days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000.0},
        index=index,
    )


def test_recommend_returns_valid_recommendation():
    rec = recommend(synthetic_history(days=300, seed=1))
    assert isinstance(rec, Recommendation)
    assert rec.action in {BUY, SELL, HOLD}
    assert -1.0 <= rec.score <= 1.0
    assert len(rec.votes) == 4
    assert rec.price > 0


def test_strong_downtrend_mostly_sell():
    # Realistic noise keeps RSI from pinning at 0 (which would trigger the
    # contrarian oversold vote); the bearish trend should dominate. Assert on
    # the distribution across the trend rather than a single bar, since noise
    # can produce a stray crossover on any given day.
    df = make_trend(200, start=120.0, end=90.0, noise=1.0)
    labels = signal_series(df).iloc[60:]  # skip the indicator warm-up
    counts = labels.value_counts()
    assert counts.get(SELL, 0) > counts.get(BUY, 0)
    assert counts.get(SELL, 0) > len(labels) * 0.4


def test_recent_upturn_scores_higher_than_downturn():
    # V-shape: long decline then sharp recovery should score above the
    # mirrored inverted-V.
    decline = make_trend(150, 100.0, 70.0)
    recovery = make_trend(60, 70.0, 95.0)
    v_shape = pd.concat([decline, recovery])
    v_shape.index = pd.bdate_range(end="2026-01-01", periods=len(v_shape))

    rally = make_trend(150, 70.0, 100.0)
    drop = make_trend(60, 100.0, 75.0)
    inverted = pd.concat([rally, drop])
    inverted.index = pd.bdate_range(end="2026-01-01", periods=len(inverted))

    assert recommend(v_shape).score > recommend(inverted).score


def test_buy_recommendation_includes_risk_levels():
    decline = make_trend(150, 100.0, 70.0)
    recovery = make_trend(60, 70.0, 95.0)
    df = pd.concat([decline, recovery])
    df.index = pd.bdate_range(end="2026-01-01", periods=len(df))
    rec = recommend(df)
    if rec.action == BUY:
        assert rec.stop_loss < rec.price < rec.take_profit


def test_explain_mentions_every_vote():
    rec = recommend(synthetic_history(days=300, seed=3))
    text = rec.explain()
    for name in ["trend", "momentum", "macd", "mean_reversion"]:
        assert name in text


def test_signal_series_matches_recommend_actions():
    """The vectorised path must agree with the per-row path."""
    df = synthetic_history(days=400, seed=11)
    series = signal_series(df)
    for end in range(60, len(df), 37):
        window = df.iloc[:end]
        assert series.iloc[end - 1] == recommend(window).action


def test_short_history_yields_hold():
    rec = recommend(synthetic_history(days=10, seed=2))
    assert rec.action == HOLD

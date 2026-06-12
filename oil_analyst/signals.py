"""Signal engine: turns indicator values into BUY / SELL / HOLD recommendations.

Each indicator family casts a vote in [-1, +1]; the weighted sum is the
composite score. Scores above ``buy_threshold`` mean BUY, below
``sell_threshold`` mean SELL, anything between is HOLD. Every vote carries a
plain-language reason so a recommendation can always be explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import compute_all

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

DEFAULT_WEIGHTS = {
    "trend": 0.35,
    "momentum": 0.25,
    "macd": 0.25,
    "mean_reversion": 0.15,
}


@dataclass
class Vote:
    name: str
    score: float  # in [-1, +1]
    weight: float
    reason: str


@dataclass
class Recommendation:
    action: str
    score: float
    price: float
    as_of: pd.Timestamp
    votes: list[Vote] = field(default_factory=list)
    stop_loss: float | None = None
    take_profit: float | None = None

    def explain(self) -> str:
        lines = [
            f"{self.action} (composite score {self.score:+.2f}) "
            f"at {self.price:.2f} as of {self.as_of.date()}"
        ]
        for v in self.votes:
            lines.append(
                f"  - {v.name} [{v.score:+.2f} x {v.weight:.2f}]: {v.reason}"
            )
        if self.stop_loss is not None and self.take_profit is not None:
            lines.append(
                f"  Suggested risk levels: stop-loss {self.stop_loss:.2f}, "
                f"take-profit {self.take_profit:.2f} (ATR-based)"
            )
        return "\n".join(lines)


def _trend_vote(row: pd.Series, weight: float) -> Vote:
    fast, slow, close = row["sma_fast"], row["sma_slow"], row["Close"]
    if pd.isna(fast) or pd.isna(slow):
        return Vote("trend", 0.0, weight, "not enough history for SMA 20/50")
    spread = (fast - slow) / slow
    score = max(-1.0, min(1.0, spread * 50))  # +/-2% spread saturates the vote
    direction = "above" if fast > slow else "below"
    return Vote(
        "trend",
        score,
        weight,
        f"SMA20 {fast:.2f} is {direction} SMA50 {slow:.2f} "
        f"({spread:+.2%}); close {close:.2f}",
    )


def _momentum_vote(row: pd.Series, weight: float) -> Vote:
    value = row["rsi"]
    if pd.isna(value):
        return Vote("momentum", 0.0, weight, "not enough history for RSI")
    if value <= 30:
        score, label = 1.0, "oversold"
    elif value >= 70:
        score, label = -1.0, "overbought"
    else:
        # Linear between the bands: RSI 30 -> +1, RSI 50 -> 0, RSI 70 -> -1.
        score = (50 - value) / 20
        label = "neutral zone"
    return Vote("momentum", score, weight, f"RSI(14) = {value:.1f} ({label})")


def _macd_vote(row: pd.Series, prev: pd.Series | None, weight: float) -> Vote:
    line, sig, hist = row["macd"], row["macd_signal"], row["macd_hist"]
    if pd.isna(line) or pd.isna(sig):
        return Vote("macd", 0.0, weight, "not enough history for MACD")
    crossed = ""
    score = 0.6 if hist > 0 else -0.6
    if prev is not None and not pd.isna(prev["macd_hist"]):
        if prev["macd_hist"] <= 0 < hist:
            score, crossed = 1.0, "; bullish crossover just occurred"
        elif prev["macd_hist"] >= 0 > hist:
            score, crossed = -1.0, "; bearish crossover just occurred"
    side = "above" if hist > 0 else "below"
    return Vote(
        "macd",
        score,
        weight,
        f"MACD {line:.3f} is {side} its signal {sig:.3f}{crossed}",
    )


def _mean_reversion_vote(row: pd.Series, weight: float) -> Vote:
    pb = row["bb_percent_b"]
    if pd.isna(pb):
        return Vote("mean_reversion", 0.0, weight, "not enough history for Bollinger")
    if pb <= 0.0:
        score, label = 1.0, "below the lower Bollinger band"
    elif pb >= 1.0:
        score, label = -1.0, "above the upper Bollinger band"
    else:
        score = (0.5 - pb) * 2 * 0.5  # mild pull toward the middle of the bands
        label = "inside the bands"
    return Vote("mean_reversion", score, weight, f"%B = {pb:.2f}, price {label}")


def recommend(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
    atr_stop_mult: float = 2.0,
    atr_target_mult: float = 3.0,
    precomputed: bool = False,
) -> Recommendation:
    """Produce a recommendation from the most recent row of ``df``.

    ``df`` must be OHLCV history (oldest first). Set ``precomputed`` if the
    indicator columns are already present.
    """
    weights = weights or DEFAULT_WEIGHTS
    data = df if precomputed else compute_all(df)
    row = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else None

    votes = [
        _trend_vote(row, weights["trend"]),
        _momentum_vote(row, weights["momentum"]),
        _macd_vote(row, prev, weights["macd"]),
        _mean_reversion_vote(row, weights["mean_reversion"]),
    ]
    total_weight = sum(v.weight for v in votes)
    score = sum(v.score * v.weight for v in votes) / total_weight

    if score >= buy_threshold:
        action = BUY
    elif score <= sell_threshold:
        action = SELL
    else:
        action = HOLD

    price = float(row["Close"])
    stop_loss = take_profit = None
    if action == BUY and not pd.isna(row["atr"]):
        stop_loss = price - atr_stop_mult * float(row["atr"])
        take_profit = price + atr_target_mult * float(row["atr"])

    return Recommendation(
        action=action,
        score=round(float(score), 4),
        price=price,
        as_of=data.index[-1],
        votes=votes,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def signal_series(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
) -> pd.Series:
    """Vectorised BUY/SELL/HOLD label for every bar (used by the backtester)."""
    weights = weights or DEFAULT_WEIGHTS
    data = compute_all(df)
    total_weight = sum(weights.values())

    spread = (data["sma_fast"] - data["sma_slow"]) / data["sma_slow"]
    trend = (spread * 50).clip(-1, 1).fillna(0.0)

    rsi_v = data["rsi"]
    momentum = ((50 - rsi_v) / 20).clip(-1, 1)
    momentum = momentum.where(rsi_v > 30, 1.0).where(rsi_v < 70, -1.0).fillna(0.0)

    hist = data["macd_hist"]
    macd_score = pd.Series(0.0, index=data.index)
    macd_score[hist > 0] = 0.6
    macd_score[hist < 0] = -0.6
    prev_hist = hist.shift(1)
    macd_score[(prev_hist <= 0) & (hist > 0)] = 1.0
    macd_score[(prev_hist >= 0) & (hist < 0)] = -1.0
    macd_score[hist.isna()] = 0.0

    pb = data["bb_percent_b"]
    mr = ((0.5 - pb) * 2 * 0.5).clip(-1, 1)
    mr = mr.where(pb > 0.0, 1.0).where(pb < 1.0, -1.0).fillna(0.0)

    score = (
        trend * weights["trend"]
        + momentum * weights["momentum"]
        + macd_score * weights["macd"]
        + mr * weights["mean_reversion"]
    ) / total_weight

    labels = pd.Series(HOLD, index=data.index)
    labels[score >= buy_threshold] = BUY
    labels[score <= sell_threshold] = SELL
    return labels

"""Event-driven backtest of the signal engine on historical data.

Long-only: BUY opens a position with the full cash balance, SELL closes it.
Fills happen at the next bar's open (no look-ahead), with configurable
commission and slippage. Reports the usual performance metrics plus a
buy-and-hold benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import BUY, SELL, signal_series

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None

    @property
    def return_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    initial_cash: float = 10_000.0

    @property
    def total_return_pct(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.initial_cash - 1)

    @property
    def max_drawdown_pct(self) -> float:
        peak = self.equity_curve.cummax()
        drawdown = self.equity_curve / peak - 1
        return float(drawdown.min())

    @property
    def sharpe_ratio(self) -> float:
        rets = self.equity_curve.pct_change().dropna()
        if rets.std(ddof=0) == 0 or rets.empty:
            return 0.0
        return float(rets.mean() / rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    @property
    def win_rate_pct(self) -> float | None:
        closed = self.closed_trades
        if not closed:
            return None
        wins = sum(1 for t in closed if t.return_pct > 0)
        return wins / len(closed)

    def summary(self, benchmark_return_pct: float | None = None) -> str:
        lines = [
            f"Final equity:    {self.equity_curve.iloc[-1]:,.2f} "
            f"(started with {self.initial_cash:,.2f})",
            f"Total return:    {self.total_return_pct:+.2%}",
            f"Max drawdown:    {self.max_drawdown_pct:.2%}",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}",
            f"Closed trades:   {len(self.closed_trades)}",
        ]
        if self.win_rate_pct is not None:
            lines.append(f"Win rate:        {self.win_rate_pct:.1%}")
        if benchmark_return_pct is not None:
            lines.append(f"Buy & hold:      {benchmark_return_pct:+.2%}")
        return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.0005,
    weights: dict[str, float] | None = None,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
) -> BacktestResult:
    """Replay the signal engine over ``df`` and simulate the resulting trades."""
    signals = signal_series(
        df, weights=weights, buy_threshold=buy_threshold, sell_threshold=sell_threshold
    )

    cash = initial_cash
    units = 0.0
    trades: list[Trade] = []
    equity = pd.Series(np.nan, index=df.index, dtype=float)

    opens = df["Open"]
    closes = df["Close"]

    for i in range(len(df)):
        # Execute yesterday's signal at today's open to avoid look-ahead bias.
        if i > 0:
            prior_signal = signals.iloc[i - 1]
            fill = float(opens.iloc[i])
            if prior_signal == BUY and units == 0.0:
                fill_price = fill * (1 + slippage_pct)
                cost = cash * (1 - commission_pct)
                units = cost / fill_price
                cash = 0.0
                trades.append(Trade(entry_date=df.index[i], entry_price=fill_price))
            elif prior_signal == SELL and units > 0.0:
                fill_price = fill * (1 - slippage_pct)
                cash = units * fill_price * (1 - commission_pct)
                trades[-1].exit_date = df.index[i]
                trades[-1].exit_price = fill_price
                units = 0.0
        equity.iloc[i] = cash + units * float(closes.iloc[i])

    return BacktestResult(equity_curve=equity, trades=trades, initial_cash=initial_cash)


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Benchmark: buy at the first open, hold to the last close."""
    return float(df["Close"].iloc[-1] / df["Open"].iloc[0] - 1)

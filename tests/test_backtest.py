import pandas as pd
import pytest

from oil_analyst.backtest import buy_and_hold_return, run_backtest
from oil_analyst.data import synthetic_history


@pytest.fixture()
def df():
    return synthetic_history(days=400, seed=5)


def test_backtest_runs_and_reports(df):
    result = run_backtest(df, initial_cash=10_000.0)
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.notna().all()
    assert (result.equity_curve > 0).all()
    assert result.max_drawdown_pct <= 0
    summary = result.summary(benchmark_return_pct=buy_and_hold_return(df))
    assert "Total return" in summary and "Buy & hold" in summary


def test_trades_are_consistent(df):
    result = run_backtest(df)
    for trade in result.closed_trades:
        assert trade.exit_date > trade.entry_date
        assert trade.entry_price > 0 and trade.exit_price > 0
    # At most one trade can be left open at the end.
    open_trades = [t for t in result.trades if t.exit_price is None]
    assert len(open_trades) <= 1


def test_no_signals_means_flat_equity():
    # With thresholds no score can reach, the bot never trades.
    df = synthetic_history(days=300, seed=9)
    result = run_backtest(df, buy_threshold=2.0, sell_threshold=-2.0)
    assert result.trades == []
    assert (result.equity_curve == result.initial_cash).all()


def test_costs_reduce_returns(df):
    cheap = run_backtest(df, commission_pct=0.0, slippage_pct=0.0)
    costly = run_backtest(df, commission_pct=0.01, slippage_pct=0.01)
    if cheap.trades:
        assert costly.total_return_pct < cheap.total_return_pct

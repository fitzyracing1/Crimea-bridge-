import json

import pytest

from oil_analyst.broker import PaperBroker
from oil_analyst.data import synthetic_history
from oil_analyst.signals import BUY, HOLD, SELL, Recommendation
from oil_analyst.trader import TradingEngine


def fixed_strategy(action, price=80.0):
    def strategy(df):
        return Recommendation(
            action=action, score=0.5, price=price, as_of=df.index[-1]
        )

    return strategy


@pytest.fixture()
def df():
    return synthetic_history(days=120, seed=4)


def make_engine(tmp_path, df, action, **kwargs):
    broker = PaperBroker(
        state_path=tmp_path / "portfolio.json", starting_cash=10_000.0
    )
    engine = TradingEngine(
        broker=broker,
        symbol="BNO",
        data_fn=lambda: df,
        strategy=fixed_strategy(action),
        journal_path=tmp_path / "journal.jsonl",
        **kwargs,
    )
    return engine, broker


def test_buy_signal_opens_position(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, BUY)
    entry = engine.run_once()
    assert entry["executed"] is True
    assert broker.get_position("BNO") > 0
    assert broker.get_cash() == pytest.approx(10_000.0 * 0.05)  # 95% deployed


def test_buy_while_holding_does_nothing(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, BUY)
    engine.run_once()
    position = broker.get_position("BNO")
    entry = engine.run_once()
    assert entry["executed"] is False
    assert "already holding" in entry["note"]
    assert broker.get_position("BNO") == position


def test_sell_closes_position(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, BUY)
    engine.run_once()
    engine.strategy = fixed_strategy(SELL)
    entry = engine.run_once()
    assert entry["executed"] is True
    assert broker.get_position("BNO") == 0.0


def test_sell_when_flat_does_nothing(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, SELL)
    entry = engine.run_once()
    assert entry["executed"] is False
    assert broker.get_cash() == 10_000.0


def test_hold_does_nothing(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, HOLD)
    entry = engine.run_once()
    assert entry["executed"] is False
    assert broker.get_cash() == 10_000.0


def test_dry_run_never_trades(tmp_path, df):
    engine, broker = make_engine(tmp_path, df, BUY, dry_run=True)
    entry = engine.run_once()
    assert entry["executed"] is False
    assert broker.get_cash() == 10_000.0
    assert "dry run" in entry["note"]


def test_journal_records_every_decision(tmp_path, df):
    engine, _ = make_engine(tmp_path, df, BUY)
    engine.run_once()
    engine.strategy = fixed_strategy(HOLD)
    engine.run_once()
    lines = (tmp_path / "journal.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["action"] == BUY and first["executed"] is True
    assert "fill" in first
    assert second["action"] == HOLD and second["executed"] is False


def test_loop_runs_n_times_and_sleeps_between(tmp_path, df):
    engine, _ = make_engine(tmp_path, df, HOLD)
    sleeps = []
    entries = []
    engine.run_loop(
        interval_seconds=7.0,
        max_iterations=3,
        sleep_fn=sleeps.append,
        on_entry=entries.append,
    )
    assert len(entries) == 3
    assert sleeps == [7.0, 7.0]


def test_loop_survives_data_errors(tmp_path, df):
    engine, _ = make_engine(tmp_path, df, HOLD)

    calls = {"n": 0}

    def flaky_data():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("network down")
        return df

    engine.data_fn = flaky_data
    entries = []
    engine.run_loop(
        interval_seconds=0.0,
        max_iterations=2,
        sleep_fn=lambda _s: None,
        on_entry=entries.append,
    )
    assert "error" in entries[0]
    assert "ConnectionError" in entries[0]["error"]
    assert entries[1]["action"] == HOLD


def test_invalid_cash_fraction_rejected(tmp_path, df):
    with pytest.raises(ValueError, match="cash_fraction"):
        make_engine(tmp_path, df, HOLD, cash_fraction=1.5)

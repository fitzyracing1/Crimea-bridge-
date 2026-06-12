import pandas as pd
import pytest

from oil_analyst.cli import main
from oil_analyst.data import load_csv, synthetic_history


def test_synthetic_history_shape_and_columns():
    df = synthetic_history(days=100, seed=1)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 100
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()


def test_synthetic_history_is_deterministic():
    a = synthetic_history(days=50, seed=42)
    b = synthetic_history(days=50, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_load_csv_roundtrip(tmp_path):
    df = synthetic_history(days=60, seed=3)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index_label="Date")
    loaded = load_csv(str(path))
    pd.testing.assert_frame_equal(loaded, df, check_freq=False, check_names=False)


def test_load_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Close\n2026-01-01,80.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_csv(str(path))


def test_cli_analyze_synthetic(capsys):
    assert main(["analyze", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert any(action in out for action in ["BUY", "SELL", "HOLD"])
    assert "Disclaimer" in out


def test_cli_backtest_synthetic(capsys):
    assert main(["backtest", "--synthetic", "--cash", "5000"]) == 0
    out = capsys.readouterr().out
    assert "Total return" in out
    assert "5,000.00" in out


def test_cli_reports_bad_csv(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Close\n2026-01-01,80.0\n")
    assert main(["analyze", "--csv", str(path)]) == 1
    assert "error:" in capsys.readouterr().err

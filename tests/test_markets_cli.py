import json
import io
import contextlib

from difference_engine.markets_cli import main


def _run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


def test_odds_command_mock(tmp_path):
    code, out = _run([
        "--mock", "--log-dir", str(tmp_path), "--json",
        "odds", "GPT-5", "Claude", "--limit", "12",
    ])
    assert code == 0
    data = json.loads(out)
    assert "side_a" in data and "side_b" in data
    p = data["side_a"]["model_prob"] + data["side_b"]["model_prob"]
    assert abs(p - 1.0) < 1e-2
    assert "Simulation only" in data["disclaimer"]


def test_odds_with_market_odds_reports_value(tmp_path):
    code, out = _run([
        "--mock", "--log-dir", str(tmp_path), "--json",
        "odds", "Strong AI", "Weak AI", "--limit", "12",
        "--market-odds-a", "5.0", "--market-odds-b", "1.1",
    ])
    assert code == 0
    data = json.loads(out)
    # both sides priced against the market
    assert data["side_a"]["ev"] is not None
    assert data["side_b"]["ev"] is not None


def test_bet_then_wallet_then_settle(tmp_path):
    logdir = str(tmp_path)
    # reset bankroll
    code, _ = _run(["--log-dir", logdir, "reset", "--bankroll", "1000"])
    assert code == 0

    # place a paper bet
    code, out = _run([
        "--mock", "--log-dir", logdir, "--json",
        "bet", "GPT-5", "Claude", "--on", "a", "--stake", "100", "--limit", "12",
    ])
    assert code == 0
    placed = json.loads(out)
    bet_id = placed["bet"]["id"]
    assert placed["wallet"]["balance"] == 900.0

    # wallet shows the open bet
    code, out = _run(["--log-dir", logdir, "--json", "wallet"])
    data = json.loads(out)
    assert data["summary"]["open_bets"] == 1

    # settle it as a win for side 'a'
    code, out = _run(["--log-dir", logdir, "--json", "settle", bet_id, "--winner", "a"])
    assert code == 0
    settled = json.loads(out)
    assert settled["bet"]["status"] in ("won", "lost")
    # balance changed from the escrowed 900
    assert settled["wallet"]["balance"] != 900.0


def test_bet_insufficient_balance(tmp_path):
    logdir = str(tmp_path)
    _run(["--log-dir", logdir, "reset", "--bankroll", "10"])
    code, _ = _run([
        "--mock", "--log-dir", logdir,
        "bet", "A", "B", "--on", "a", "--stake", "100", "--limit", "12",
    ])
    assert code == 2

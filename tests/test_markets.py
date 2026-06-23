import math

import pytest

from difference_engine.config import Config
from difference_engine.markets import (
    Subject,
    PaperBook,
    assess_subject,
    make_market,
    win_probability,
    decimal_odds,
    apply_margin,
    implied_prob,
    expected_value,
    kelly_fraction,
)


# -- math ---------------------------------------------------------------
def test_win_probability_symmetry():
    p = win_probability(0.5, 0.5, 1.0)
    assert abs(p - 0.5) < 1e-9


def test_win_probability_favours_stronger_subject():
    p = win_probability(0.8, -0.2, 1.0)
    assert p > 0.5


def test_low_confidence_flattens_to_even():
    strong = win_probability(0.9, -0.9, 1.0)
    weak = win_probability(0.9, -0.9, 0.05)
    assert strong > weak
    assert abs(weak - 0.5) < abs(strong - 0.5)


def test_decimal_odds_inverse_of_prob():
    assert decimal_odds(0.5) == 2.0
    assert math.isclose(decimal_odds(0.25), 4.0, rel_tol=1e-6)


def test_apply_margin_shortens_odds():
    fair = decimal_odds(0.5)  # 2.0
    house = apply_margin(fair, 0.05)
    assert house < fair


def test_expected_value_and_kelly_positive_edge():
    # model says 0.6, market offers 2.0 (implies 0.5) -> +EV
    ev = expected_value(0.6, 2.0)
    assert ev > 0
    f = kelly_fraction(0.6, 2.0)
    assert 0 < f <= 1
    # exact Kelly: (b*p - q)/b with b=1, p=0.6, q=0.4 -> 0.2
    assert math.isclose(f, 0.2, rel_tol=1e-6)


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.4, 2.0) == 0.0  # implied 0.5 > model 0.4
    assert expected_value(0.4, 2.0) < 0


def test_implied_prob():
    assert implied_prob(4.0) == 0.25


# -- market construction ------------------------------------------------
def test_make_market_probabilities_sum_to_one():
    a = Subject("A", strength=0.6, confidence=0.8, record_count=10)
    b = Subject("B", strength=-0.4, confidence=0.8, record_count=10)
    market = make_market(a, b, market_odds_a=None, market_odds_b=None)
    assert math.isclose(market.side_a.model_prob + market.side_b.model_prob, 1.0, abs_tol=1e-3)
    assert market.side_a.model_prob > market.side_b.model_prob
    assert market.side_a.fair_odds < market.side_b.fair_odds  # favourite has shorter odds


def test_make_market_detects_value_pick():
    a = Subject("A", strength=0.6, confidence=0.9, record_count=12)
    b = Subject("B", strength=-0.6, confidence=0.9, record_count=12)
    # offer generous odds on the favourite A -> should be value
    market = make_market(a, b, market_odds_a=3.0, market_odds_b=1.3, bankroll=1000)
    assert market.value_pick == "A"
    assert market.side_a.ev is not None and market.side_a.ev > 0
    assert market.side_a.suggested_stake > 0


def test_make_market_no_value_when_odds_tight():
    a = Subject("A", strength=0.6, confidence=0.9, record_count=12)
    b = Subject("B", strength=-0.6, confidence=0.9, record_count=12)
    market = make_market(a, b, market_odds_a=1.01, market_odds_b=1.01, bankroll=1000)
    assert market.value_pick is None


def test_assess_subject_with_mock():
    cfg = Config(use_mock=True)
    subj = assess_subject("some ai model", limit=12, config=cfg)
    assert subj.name == "some ai model"
    assert subj.record_count == 12
    assert -1.0 <= subj.strength <= 1.0
    assert 0.0 <= subj.confidence <= 1.0


# -- paper bankroll -----------------------------------------------------
def test_paperbook_place_and_settle_win(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=1000.0)
    bet = book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)
    assert book.balance == 900.0  # stake escrowed
    settled = book.settle_bet(bet["id"], winner="A")
    assert settled["status"] == "won"
    assert book.balance == 1100.0  # +100 profit
    assert settled["pnl"] == 100.0


def test_paperbook_settle_loss(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=1000.0)
    bet = book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)
    settled = book.settle_bet(bet["id"], winner="B")
    assert settled["status"] == "lost"
    assert book.balance == 900.0
    assert settled["pnl"] == -100.0


def test_paperbook_rejects_overstake(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=50.0)
    with pytest.raises(ValueError):
        book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)


def test_paperbook_rejects_bad_odds(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=1000.0)
    with pytest.raises(ValueError):
        book.place_bet(pair="A vs B", side="A", stake=10, odds=1.0)


def test_paperbook_persists_across_instances(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=500.0)
    book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)
    reopened = PaperBook(log_dir=str(tmp_path))
    assert reopened.balance == 400.0
    assert len(reopened.open_bets()) == 1


def test_paperbook_double_settle_rejected(tmp_path):
    book = PaperBook(log_dir=str(tmp_path), starting_balance=1000.0)
    bet = book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)
    book.settle_bet(bet["id"], winner="A")
    with pytest.raises(ValueError):
        book.settle_bet(bet["id"], winner="A")


def test_ledger_file_written(tmp_path):
    import os
    book = PaperBook(log_dir=str(tmp_path), starting_balance=1000.0)
    bet = book.place_bet(pair="A vs B", side="A", stake=100, odds=2.0)
    book.settle_bet(bet["id"], winner="A")
    ledger = os.path.join(str(tmp_path), "bets.jsonl")
    assert os.path.exists(ledger)
    with open(ledger, encoding="utf-8") as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 2  # place + settle

"""Pairs betting layer for the difference engine.

This turns a head-to-head pair of AI subjects (e.g. two models, companies or
technologies) into a market:

  1. the difference engine pulls and scores data about each subject, producing
     a net support / sentiment strength and a confidence;
  2. the strength gap is mapped to a model win probability for each side;
  3. that probability is expressed as fair decimal odds and as "house" odds
     with a configurable margin;
  4. if you supply the odds a real market is offering, it computes your edge,
     expected value and the Kelly-optimal stake — i.e. whether there is value;
  5. a paper bankroll lets you place and settle simulated bets, and every bet
     is logged.

IMPORTANT — this is a simulation / decision-support tool. It does NOT connect
to real money or any betting venue, and a model probability is not a guarantee.
Wagering real money requires a licensed, regulated venue and carries real risk.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from .config import Config
from .client import EartheareconClient
from .engine import DifferenceEngine

DISCLAIMER = (
    "Simulation only — no real money is wagered. Model probabilities are "
    "estimates, not guarantees. Real betting is risky and regulated."
)


# --------------------------------------------------------------------------
# Subject assessment
# --------------------------------------------------------------------------
@dataclass
class Subject:
    name: str
    strength: float        # -1..1 net support / sentiment about the subject
    confidence: float      # 0..1 how much signal backs the strength
    record_count: int
    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def assess_subject(
    name: str,
    limit: int = 20,
    client: EartheareconClient | None = None,
    engine: DifferenceEngine | None = None,
    config: Config | None = None,
) -> Subject:
    """Pull and score data about ``name`` and reduce it to a strength signal."""
    config = config or Config.from_env()
    client = client or EartheareconClient(config)
    engine = engine or DifferenceEngine()

    records = client.search(name, limit=limit)
    result = engine.analyze(name, records)
    summary = engine.summarize(result)

    relevant = [i for i in result.items if i.relevance > 0] or result.items
    mean_conf = sum(i.confidence for i in relevant) / len(relevant) if relevant else 0.0
    data_factor = min(1.0, len(records) / 8.0)
    confidence = round(0.7 * mean_conf + 0.3 * data_factor, 4)

    return Subject(
        name=name,
        strength=round(float(summary.get("net_support", 0.0)), 4),
        confidence=confidence,
        record_count=len(records),
        verdict=summary.get("verdict", ""),
    )


# --------------------------------------------------------------------------
# Probability / odds math
# --------------------------------------------------------------------------
def win_probability(strength_a: float, strength_b: float, combined_confidence: float, k: float = 3.0) -> float:
    """Logistic map from the strength gap to P(A beats B).

    Low combined confidence flattens the curve toward 50/50, so thin evidence
    never produces an overconfident price.
    """
    z = k * max(0.0, min(1.0, combined_confidence)) * (strength_a - strength_b)
    z = max(-30.0, min(30.0, z))
    p = 1.0 / (1.0 + math.exp(-z))
    return min(0.99, max(0.01, p))


def decimal_odds(prob: float) -> float:
    prob = min(0.999, max(0.001, prob))
    return round(1.0 / prob, 4)


def apply_margin(fair_decimal_odds: float, margin: float) -> float:
    """Shorten fair odds by a bookmaker margin (the overround / vig)."""
    return round(fair_decimal_odds / (1.0 + max(0.0, margin)), 4)


def implied_prob(decimal_odds_value: float) -> float:
    if decimal_odds_value <= 0:
        return 0.0
    return round(1.0 / decimal_odds_value, 4)


def expected_value(prob: float, decimal_odds_value: float) -> float:
    """EV per 1 unit staked at the given odds (positive = +EV)."""
    return round(prob * decimal_odds_value - 1.0, 4)


def kelly_fraction(prob: float, decimal_odds_value: float) -> float:
    """Fraction of bankroll to stake (Kelly). 0 when there is no edge."""
    b = decimal_odds_value - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return round(max(0.0, f), 4)


# --------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------
@dataclass
class SideQuote:
    name: str
    model_prob: float
    fair_odds: float
    house_odds: float
    market_odds: float | None = None
    market_implied_prob: float | None = None
    edge: float | None = None          # model_prob - market_implied_prob
    ev: float | None = None            # EV per unit at market odds
    kelly: float | None = None         # Kelly fraction at market odds
    suggested_stake: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairMarket:
    subject_a: Subject
    subject_b: Subject
    side_a: SideQuote
    side_b: SideQuote
    margin: float
    combined_confidence: float
    value_pick: str | None = None      # side name with the best +EV, if any
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "subject_a": self.subject_a.to_dict(),
            "subject_b": self.subject_b.to_dict(),
            "side_a": self.side_a.to_dict(),
            "side_b": self.side_b.to_dict(),
            "margin": self.margin,
            "combined_confidence": self.combined_confidence,
            "value_pick": self.value_pick,
            "disclaimer": self.disclaimer,
        }


def make_market(
    subject_a: Subject,
    subject_b: Subject,
    k: float = 3.0,
    margin: float = 0.05,
    market_odds_a: float | None = None,
    market_odds_b: float | None = None,
    bankroll: float = 0.0,
    kelly_cap: float = 0.25,
) -> PairMarket:
    """Build a head-to-head market from two assessed subjects."""
    combined_conf = round((subject_a.confidence + subject_b.confidence) / 2.0, 4)
    prob_a = win_probability(subject_a.strength, subject_b.strength, combined_conf, k=k)
    prob_b = round(1.0 - prob_a, 4)
    prob_a = round(prob_a, 4)

    side_a = _build_side(subject_a.name, prob_a, margin, market_odds_a, bankroll, kelly_cap)
    side_b = _build_side(subject_b.name, prob_b, margin, market_odds_b, bankroll, kelly_cap)

    value_pick = None
    candidates = [s for s in (side_a, side_b) if s.ev is not None and s.ev > 0]
    if candidates:
        value_pick = max(candidates, key=lambda s: s.ev).name

    return PairMarket(
        subject_a=subject_a,
        subject_b=subject_b,
        side_a=side_a,
        side_b=side_b,
        margin=margin,
        combined_confidence=combined_conf,
        value_pick=value_pick,
    )


def _build_side(name, prob, margin, market_odds, bankroll, kelly_cap) -> SideQuote:
    fair = decimal_odds(prob)
    house = apply_margin(fair, margin)
    quote = SideQuote(name=name, model_prob=prob, fair_odds=fair, house_odds=house)
    if market_odds and market_odds > 1.0:
        quote.market_odds = round(market_odds, 4)
        quote.market_implied_prob = implied_prob(market_odds)
        quote.edge = round(prob - quote.market_implied_prob, 4)
        quote.ev = expected_value(prob, market_odds)
        kelly = kelly_fraction(prob, market_odds)
        quote.kelly = kelly
        capped = min(kelly, kelly_cap)
        quote.suggested_stake = round(bankroll * capped, 2) if bankroll else round(capped, 4)
    return quote


# --------------------------------------------------------------------------
# Paper bankroll + bet ledger
# --------------------------------------------------------------------------
class PaperBook:
    """A simulated bankroll persisted to disk, with an append-only bet ledger."""

    def __init__(self, log_dir: str = "logs", starting_balance: float = 1000.0, currency: str = "UNITS"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.wallet_path = os.path.join(self.log_dir, "wallet.json")
        self.ledger_path = os.path.join(self.log_dir, "bets.jsonl")
        self.starting_balance = starting_balance
        self.currency = currency
        self._state = self._load()
        self.currency = self._state.get("currency", currency)

    # -- persistence ----------------------------------------------------
    def _load(self) -> dict:
        if os.path.exists(self.wallet_path):
            with open(self.wallet_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "balance": self.starting_balance,
            "currency": self.currency,
            "starting_balance": self.starting_balance,
            "bets": {},
        }

    def _save(self) -> None:
        with open(self.wallet_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, ensure_ascii=False)

    def _append_ledger(self, entry: dict) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # -- accessors ------------------------------------------------------
    @property
    def balance(self) -> float:
        return round(self._state["balance"], 2)

    def reset(self, starting_balance: float | None = None) -> dict:
        bal = starting_balance if starting_balance is not None else self.starting_balance
        self._state = {
            "balance": bal,
            "currency": self.currency,
            "starting_balance": bal,
            "bets": {},
        }
        self._save()
        return self._state

    def open_bets(self) -> list[dict]:
        return [b for b in self._state["bets"].values() if b["status"] == "open"]

    def all_bets(self) -> list[dict]:
        return list(self._state["bets"].values())

    # -- actions --------------------------------------------------------
    def place_bet(self, pair: str, side: str, stake: float, odds: float) -> dict:
        if stake <= 0:
            raise ValueError("stake must be positive")
        if odds <= 1.0:
            raise ValueError("decimal odds must be > 1.0")
        if stake > self._state["balance"] + 1e-9:
            raise ValueError(f"insufficient balance: stake {stake} > balance {self.balance}")

        bet_id = uuid.uuid4().hex[:12]
        bet = {
            "id": bet_id,
            "pair": pair,
            "side": side,
            "stake": round(stake, 2),
            "odds": round(odds, 4),
            "potential_return": round(stake * odds, 2),
            "potential_profit": round(stake * (odds - 1.0), 2),
            "status": "open",
            "placed_at": datetime.now(timezone.utc).isoformat(),
            "settled_at": None,
            "winner": None,
            "pnl": None,
        }
        self._state["balance"] -= bet["stake"]   # stake is escrowed
        self._state["bets"][bet_id] = bet
        self._save()
        self._append_ledger({"event": "place", **bet})
        return bet

    def settle_bet(self, bet_id: str, winner: str) -> dict:
        bet = self._state["bets"].get(bet_id)
        if bet is None:
            raise KeyError(f"unknown bet id: {bet_id}")
        if bet["status"] != "open":
            raise ValueError(f"bet {bet_id} already settled ({bet['status']})")

        won = (winner == bet["side"])
        if won:
            self._state["balance"] += bet["potential_return"]
            bet["pnl"] = bet["potential_profit"]
            bet["status"] = "won"
        else:
            bet["pnl"] = -bet["stake"]
            bet["status"] = "lost"
        bet["winner"] = winner
        bet["settled_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        self._append_ledger({"event": "settle", **bet})
        return bet

    def summary(self) -> dict:
        bets = self.all_bets()
        settled = [b for b in bets if b["status"] in ("won", "lost")]
        realized = round(sum(b["pnl"] for b in settled), 2) if settled else 0.0
        staked_open = round(sum(b["stake"] for b in self.open_bets()), 2)
        return {
            "balance": self.balance,
            "currency": self.currency,
            "starting_balance": self._state.get("starting_balance"),
            "open_bets": len(self.open_bets()),
            "staked_in_open_bets": staked_open,
            "settled_bets": len(settled),
            "realized_pnl": realized,
            "wins": sum(1 for b in settled if b["status"] == "won"),
            "losses": sum(1 for b in settled if b["status"] == "lost"),
        }


if __name__ == "__main__":
    from .markets_cli import main

    raise SystemExit(main())

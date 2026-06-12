"""Command-line interface for the pairs betting layer.

    # price a head-to-head (offline mock data)
    python -m difference_engine.markets odds "GPT-5" "Claude" --mock

    # find value against the odds a real market is offering
    python -m difference_engine.markets odds "GPT-5" "Claude" --mock \
        --market-odds-a 1.8 --market-odds-b 2.1

    # place / settle simulated bets against a paper bankroll
    python -m difference_engine.markets bet "GPT-5" "Claude" --on a --stake 50 --mock
    python -m difference_engine.markets settle <bet_id> --winner a
    python -m difference_engine.markets wallet
    python -m difference_engine.markets reset --bankroll 1000

This is a simulation. No real money, no betting venue. See DISCLAIMER.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .client import EartheareconClient
from .engine import DifferenceEngine
from .markets import (
    DISCLAIMER,
    PaperBook,
    assess_subject,
    make_market,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="difference_engine.markets",
        description="Price head-to-head AI 'pairs', find value vs market odds, and paper-bet. SIMULATION ONLY.",
    )
    p.add_argument("--mock", action="store_true", help="use the offline mock data source")
    p.add_argument("--log-dir", default=None, help="directory for the paper wallet + bet ledger")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    sub = p.add_subparsers(dest="command", required=True)

    common_odds = argparse.ArgumentParser(add_help=False)
    common_odds.add_argument("a", help="first subject (e.g. an AI model/company)")
    common_odds.add_argument("b", help="second subject")
    common_odds.add_argument("--limit", type=int, default=20, help="records to pull per subject")
    common_odds.add_argument("--margin", type=float, default=0.05, help="house margin / vig (default 0.05)")
    common_odds.add_argument("--k", type=float, default=3.0, help="probability curve steepness")
    common_odds.add_argument("--market-odds-a", type=float, default=None, help="decimal odds a market offers on A")
    common_odds.add_argument("--market-odds-b", type=float, default=None, help="decimal odds a market offers on B")
    common_odds.add_argument("--kelly-cap", type=float, default=0.25, help="cap on Kelly stake fraction")

    odds = sub.add_parser("odds", parents=[common_odds], help="price a pair and show any value")
    odds.add_argument("--bankroll", type=float, default=None, help="bankroll for stake sizing (default: wallet balance)")

    bet = sub.add_parser("bet", parents=[common_odds], help="place a simulated bet on a pair")
    bet.add_argument("--on", required=True, help="side to back: subject name, or 'a' / 'b' / 'value'")
    g = bet.add_mutually_exclusive_group(required=True)
    g.add_argument("--stake", type=float, help="stake amount in paper units")
    g.add_argument("--kelly", action="store_true", help="stake the Kelly-suggested amount (needs market odds)")
    bet.add_argument("--odds", type=float, default=None, help="override decimal odds to bet at")

    settle = sub.add_parser("settle", help="settle an open bet")
    settle.add_argument("bet_id")
    settle.add_argument("--winner", required=True, help="winning side: subject name, or 'a' / 'b'")

    sub.add_parser("wallet", help="show the paper bankroll and open bets")

    reset = sub.add_parser("reset", help="reset the paper bankroll")
    reset.add_argument("--bankroll", type=float, default=1000.0, help="starting balance (default 1000)")

    sub.add_parser("history", help="show settled bets")
    return p


def _book(args) -> PaperBook:
    config = Config.from_env(log_dir=args.log_dir, use_mock=True if args.mock else None)
    return PaperBook(log_dir=config.log_dir)


def _engine_ctx(args):
    config = Config.from_env(log_dir=args.log_dir, use_mock=True if args.mock else None)
    return EartheareconClient(config), DifferenceEngine(), config


def _price(args, bankroll: float):
    client, engine, _ = _engine_ctx(args)
    subj_a = assess_subject(args.a, limit=args.limit, client=client, engine=engine)
    subj_b = assess_subject(args.b, limit=args.limit, client=client, engine=engine)
    market = make_market(
        subj_a, subj_b,
        k=args.k, margin=args.margin,
        market_odds_a=args.market_odds_a, market_odds_b=args.market_odds_b,
        bankroll=bankroll, kelly_cap=args.kelly_cap,
    )
    return market


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "wallet":
        return _cmd_wallet(args)
    if args.command == "reset":
        return _cmd_reset(args)
    if args.command == "history":
        return _cmd_history(args)
    if args.command == "odds":
        return _cmd_odds(args)
    if args.command == "bet":
        return _cmd_bet(args)
    if args.command == "settle":
        return _cmd_settle(args)
    return 1


# -- command handlers ---------------------------------------------------
def _cmd_odds(args) -> int:
    book = _book(args)
    bankroll = args.bankroll if args.bankroll is not None else book.balance
    market = _price(args, bankroll)
    if args.json:
        _dump(market.to_dict())
    else:
        _print_market(market, bankroll, book.currency)
    return 0


def _cmd_bet(args) -> int:
    book = _book(args)
    market = _price(args, book.balance)
    side = _resolve_side(args.on, market)

    odds = args.odds if args.odds is not None else (side.market_odds or side.house_odds)
    if args.kelly:
        if side.suggested_stake is None:
            print("Cannot use --kelly without --market-odds for the chosen side.", file=sys.stderr)
            return 2
        stake = float(side.suggested_stake)
        if stake <= 0:
            print(f"Kelly suggests no bet on {side.name} (no positive edge). Nothing placed.")
            return 0
    else:
        stake = float(args.stake)

    pair_label = f"{market.subject_a.name} vs {market.subject_b.name}"
    try:
        bet = book.place_bet(pair=pair_label, side=side.name, stake=stake, odds=odds)
    except ValueError as exc:
        print(f"Could not place bet: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _dump({"bet": bet, "wallet": book.summary()})
    else:
        print(f"\n{DISCLAIMER}\n")
        print(f"Placed paper bet {bet['id']}: {stake:.2f} {book.currency} on '{side.name}' "
              f"@ {odds} (pair: {pair_label})")
        print(f"  potential return : {bet['potential_return']:.2f} {book.currency} "
              f"(profit {bet['potential_profit']:.2f})")
        print(f"  model probability: {side.model_prob:.3f}   "
              f"{'EV ' + format(side.ev, '+.3f') if side.ev is not None else ''}")
        print(f"  balance now      : {book.balance:.2f} {book.currency}")
        print(f"  settle with      : python -m difference_engine.markets settle {bet['id']} --winner a|b")
    return 0


def _cmd_settle(args) -> int:
    book = _book(args)
    winner = args.winner.strip()
    bet = book._state["bets"].get(args.bet_id)
    if bet is None:
        print(f"Unknown bet id: {args.bet_id}", file=sys.stderr)
        return 2
    # Allow 'a'/'b' shorthand resolved from the bet's pair label "A vs B".
    if winner.lower() in ("a", "b"):
        parts = bet["pair"].split(" vs ")
        if len(parts) == 2:
            winner = parts[0] if winner.lower() == "a" else parts[1]
    try:
        settled = book.settle_bet(args.bet_id, winner)
    except (KeyError, ValueError) as exc:
        print(f"Could not settle: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _dump({"bet": settled, "wallet": book.summary()})
    else:
        outcome = settled["status"].upper()
        print(f"Bet {settled['id']} settled: {outcome} (winner '{winner}'), "
              f"P&L {settled['pnl']:+.2f} {book.currency}")
        print(f"  balance now: {book.balance:.2f} {book.currency}")
    return 0


def _cmd_wallet(args) -> int:
    book = _book(args)
    summary = book.summary()
    if args.json:
        _dump({"summary": summary, "open_bets": book.open_bets()})
        return 0
    print(f"\nPAPER WALLET  ({DISCLAIMER})")
    print(f"  balance        : {summary['balance']:.2f} {summary['currency']}")
    print(f"  starting       : {summary['starting_balance']}")
    print(f"  realized P&L   : {summary['realized_pnl']:+.2f} "
          f"({summary['wins']}W / {summary['losses']}L)")
    print(f"  open bets      : {summary['open_bets']} "
          f"(staked {summary['staked_in_open_bets']:.2f})")
    for b in book.open_bets():
        print(f"    - {b['id']}: {b['stake']:.2f} on '{b['side']}' @ {b['odds']} "
              f"[{b['pair']}] -> returns {b['potential_return']:.2f}")
    print()
    return 0


def _cmd_reset(args) -> int:
    book = _book(args)
    book.reset(starting_balance=args.bankroll)
    print(f"Paper wallet reset to {args.bankroll:.2f} {book.currency}.")
    return 0


def _cmd_history(args) -> int:
    book = _book(args)
    settled = [b for b in book.all_bets() if b["status"] in ("won", "lost")]
    if args.json:
        _dump(settled)
        return 0
    if not settled:
        print("No settled bets yet.")
        return 0
    print("\nSETTLED BETS")
    for b in settled:
        print(f"  {b['id']}: {b['stake']:.2f} on '{b['side']}' @ {b['odds']} "
              f"[{b['pair']}] -> {b['status'].upper()} (P&L {b['pnl']:+.2f}, winner {b['winner']})")
    print()
    return 0


# -- helpers ------------------------------------------------------------
def _resolve_side(token: str, market):
    t = token.strip().lower()
    if t == "a":
        return market.side_a
    if t == "b":
        return market.side_b
    if t == "value":
        if market.value_pick is None:
            raise SystemExit("No +EV value side available (supply --market-odds to find value).")
        return market.side_a if market.side_a.name == market.value_pick else market.side_b
    if token == market.side_a.name:
        return market.side_a
    if token == market.side_b.name:
        return market.side_b
    raise SystemExit(f"Unknown side '{token}'. Use the subject name, or 'a' / 'b' / 'value'.")


def _print_market(market, bankroll, currency) -> None:
    a, b = market.side_a, market.side_b
    bar = "=" * 72
    print(bar)
    print(f"PAIRS MARKET  |  {market.subject_a.name}  vs  {market.subject_b.name}")
    print(bar)
    print(f"\n{DISCLAIMER}\n")

    for subj, side in ((market.subject_a, a), (market.subject_b, b)):
        print(f"{side.name}")
        print(f"  strength {subj.strength:+.3f} | confidence {subj.confidence:.2f} "
              f"| records {subj.record_count} | {subj.verdict}")
        print(f"  model win prob : {side.model_prob:.3f}")
        print(f"  fair odds      : {side.fair_odds:.2f}   house odds (margin {market.margin:.0%}): {side.house_odds:.2f}")
        if side.market_odds is not None:
            tag = "VALUE (+EV)" if (side.ev or 0) > 0 else "no edge"
            print(f"  market odds    : {side.market_odds:.2f} (implied {side.market_implied_prob:.3f})")
            print(f"  edge           : {side.edge:+.3f}   EV/unit: {side.ev:+.3f}   [{tag}]")
            print(f"  Kelly fraction : {side.kelly:.3f}   suggested stake: "
                  f"{side.suggested_stake:.2f} {currency}")
        print()

    print(f"combined confidence: {market.combined_confidence:.2f}")
    if market.value_pick:
        print(f"VALUE PICK         : {market.value_pick}  (positive expected value vs the market odds you supplied)")
    elif market.side_a.market_odds is not None:
        print("VALUE PICK         : none — no positive expected value at those market odds")
    else:
        print("Tip: pass --market-odds-a / --market-odds-b to check for value against a real market.")
    print()


def _dump(obj) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

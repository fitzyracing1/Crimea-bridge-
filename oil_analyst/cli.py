"""Command-line interface for the oil analyst bot.

Examples:
    python -m oil_analyst analyze --ticker BZ=F --period 1y
    python -m oil_analyst analyze --csv prices.csv
    python -m oil_analyst backtest --ticker CL=F --period 2y --cash 25000
    python -m oil_analyst analyze --synthetic        # offline demo data
    python -m oil_analyst trade --ticker BNO                  # paper portfolio
    python -m oil_analyst trade --ticker BNO --loop --interval 3600
    python -m oil_analyst trade --ticker USO --broker alpaca  # alpaca paper acct
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtest import buy_and_hold_return, run_backtest
from .data import KNOWN_TICKERS, fetch_history, load_csv, synthetic_history
from .signals import recommend
from .trader import TradingEngine

DISCLAIMER = (
    "Disclaimer: output is for research/education only and is not financial "
    "advice. Orders are simulated (paper) unless you explicitly configure a "
    "live brokerage endpoint; live trading is at your own risk."
)


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--ticker",
        default="BZ=F",
        help="Yahoo Finance ticker (default: BZ=F Brent futures). "
        f"Known oil benchmarks: {', '.join(KNOWN_TICKERS)}",
    )
    source.add_argument("--csv", help="load OHLCV history from a CSV file instead")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated demo data (no network needed)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="history length for live downloads, e.g. 6mo, 1y, 2y (default: 1y)",
    )


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.synthetic:
        return synthetic_history(), "synthetic demo data"
    if args.csv:
        return load_csv(args.csv), f"CSV file {args.csv}"
    label = KNOWN_TICKERS.get(args.ticker, args.ticker)
    return fetch_history(args.ticker, period=args.period), f"{args.ticker} ({label})"


def cmd_analyze(args: argparse.Namespace) -> int:
    df, source = _load(args)
    rec = recommend(df)
    print(f"Source: {source}, {len(df)} bars through {df.index[-1].date()}")
    print()
    print(rec.explain())
    print()
    print(DISCLAIMER)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    df, source = _load(args)
    result = run_backtest(df, initial_cash=args.cash)
    print(f"Source: {source}, {len(df)} bars through {df.index[-1].date()}")
    print()
    print(result.summary(benchmark_return_pct=buy_and_hold_return(df)))
    print()
    print(DISCLAIMER)
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    if args.broker == "alpaca":
        from .broker import AlpacaBroker

        broker = AlpacaBroker(allow_live=args.allow_live)
        endpoint = broker.base_url
    else:
        from .broker import PaperBroker

        broker = PaperBroker(
            state_path=args.portfolio, starting_cash=args.cash
        )
        endpoint = f"local paper portfolio ({args.portfolio})"

    if args.synthetic:
        data_fn = synthetic_history
    elif args.csv:
        data_fn = lambda: load_csv(args.csv)  # noqa: E731
    else:
        data_fn = lambda: fetch_history(args.ticker, period=args.period)  # noqa: E731

    engine = TradingEngine(
        broker=broker,
        symbol=args.ticker,
        data_fn=data_fn,
        cash_fraction=args.cash_fraction,
        journal_path=args.journal,
        dry_run=args.dry_run,
    )

    def report(entry: dict) -> None:
        if "error" in entry:
            print(f"[{entry['timestamp']}] ERROR {entry['error']}")
        else:
            print(
                f"[{entry['timestamp']}] {entry['symbol']} {entry['action']} "
                f"(score {entry['score']:+.2f}) at {entry['price']:.2f} -- "
                f"{entry['note']}"
            )

    print(f"Broker: {endpoint}")
    print(f"Journal: {args.journal}")
    print(DISCLAIMER)
    print()
    if args.loop:
        print(f"Running every {args.interval}s; Ctrl-C to stop.")
        try:
            engine.run_loop(interval_seconds=args.interval, on_entry=report)
        except KeyboardInterrupt:
            print("stopped")
    else:
        report(engine.run_once())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oil_analyst",
        description="Buy/sell/hold analyst bot for exchange-traded oil benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="print today's recommendation")
    _add_source_args(analyze)
    analyze.set_defaults(func=cmd_analyze)

    backtest = sub.add_parser("backtest", help="backtest the strategy on history")
    _add_source_args(backtest)
    backtest.add_argument(
        "--cash", type=float, default=10_000.0, help="starting cash (default: 10000)"
    )
    backtest.set_defaults(func=cmd_backtest)

    trade = sub.add_parser(
        "trade", help="run the bot and place orders through a broker"
    )
    _add_source_args(trade)
    trade.add_argument(
        "--broker",
        choices=["paper", "alpaca"],
        default="paper",
        help="paper = built-in simulator (default); alpaca = Alpaca API "
        "(paper-trading endpoint unless --allow-live)",
    )
    trade.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="starting cash for a new paper portfolio (default: 10000)",
    )
    trade.add_argument(
        "--cash-fraction",
        type=float,
        default=0.95,
        help="fraction of cash to deploy on a BUY (default: 0.95)",
    )
    trade.add_argument(
        "--portfolio",
        default="paper_portfolio.json",
        help="paper portfolio state file (default: paper_portfolio.json)",
    )
    trade.add_argument(
        "--journal",
        default="trade_journal.jsonl",
        help="decision journal file (default: trade_journal.jsonl)",
    )
    trade.add_argument(
        "--loop", action="store_true", help="keep running instead of a single pass"
    )
    trade.add_argument(
        "--interval",
        type=float,
        default=3600.0,
        help="seconds between runs with --loop (default: 3600)",
    )
    trade.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and journal but never place orders",
    )
    trade.add_argument(
        "--allow-live",
        action="store_true",
        help="permit a non-paper ALPACA_BASE_URL (real money; off by default)",
    )
    trade.set_defaults(func=cmd_trade)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

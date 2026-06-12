"""Command-line interface for the oil analyst bot.

Examples:
    python -m oil_analyst analyze --ticker BZ=F --period 1y
    python -m oil_analyst analyze --csv prices.csv
    python -m oil_analyst backtest --ticker CL=F --period 2y --cash 25000
    python -m oil_analyst analyze --synthetic        # offline demo data
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtest import buy_and_hold_return, run_backtest
from .data import KNOWN_TICKERS, fetch_history, load_csv, synthetic_history
from .signals import recommend

DISCLAIMER = (
    "Disclaimer: output is for research/education only and is not financial "
    "advice. This tool analyzes public exchange data and never places orders."
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

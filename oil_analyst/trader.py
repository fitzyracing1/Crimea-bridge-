"""Trading engine: turns recommendations into broker orders.

Long-only position management on a single symbol:
  BUY  -> invest ``cash_fraction`` of available cash, if currently flat
  SELL -> close the whole position, if currently holding
  HOLD -> do nothing

Every decision (executed or not) is appended to a JSONL journal so there is
an audit trail of what the bot saw and did.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .signals import BUY, SELL, Recommendation, recommend


class TradingEngine:
    def __init__(
        self,
        broker,
        symbol: str,
        data_fn: Callable[[], pd.DataFrame],
        strategy: Callable[[pd.DataFrame], Recommendation] = recommend,
        cash_fraction: float = 0.95,
        journal_path: str | Path = "trade_journal.jsonl",
        dry_run: bool = False,
    ) -> None:
        if not 0 < cash_fraction <= 1:
            raise ValueError("cash_fraction must be in (0, 1]")
        self.broker = broker
        self.symbol = symbol
        self.data_fn = data_fn
        self.strategy = strategy
        self.cash_fraction = cash_fraction
        self.journal_path = Path(journal_path)
        self.dry_run = dry_run

    def run_once(self) -> dict:
        """Fetch data, decide, execute. Returns the journal entry."""
        df = self.data_fn()
        rec = self.strategy(df)
        position = self.broker.get_position(self.symbol)
        cash = self.broker.get_cash()

        executed = False
        fill = None
        note = ""
        if self.dry_run:
            note = "dry run: no order placed"
        elif rec.action == BUY:
            if position > 0:
                note = "already holding; no action"
            else:
                notional = cash * self.cash_fraction
                if notional < 1.0:
                    note = f"cash {cash:.2f} too low to open a position"
                else:
                    fill = self.broker.market_buy(self.symbol, notional, rec.price)
                    executed = True
                    note = f"bought {fill.units:.4f} units at ~{fill.price:.2f}"
        elif rec.action == SELL:
            if position <= 0:
                note = "no position to sell"
            else:
                fill = self.broker.market_sell_all(self.symbol, rec.price)
                executed = True
                note = f"sold {fill.units:.4f} units at ~{fill.price:.2f}"
        else:
            note = "holding"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "action": rec.action,
            "score": rec.score,
            "price": rec.price,
            "data_as_of": str(rec.as_of),
            "position_before": position,
            "cash_before": cash,
            "executed": executed,
            "note": note,
            "votes": [
                {"name": v.name, "score": v.score, "reason": v.reason}
                for v in rec.votes
            ],
        }
        if fill is not None:
            entry["fill"] = fill.__dict__
        self._journal(entry)
        return entry

    def run_loop(
        self,
        interval_seconds: float,
        max_iterations: int | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_entry: Callable[[dict], None] | None = None,
    ) -> None:
        """Run forever (or ``max_iterations`` times), pausing between runs.

        Errors from a single iteration (e.g. a transient network failure) are
        journaled and the loop continues.
        """
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                entry = self.run_once()
            except Exception as exc:  # noqa: BLE001 - keep the bot alive
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": self.symbol,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self._journal(entry)
            if on_entry is not None:
                on_entry(entry)
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                sleep_fn(interval_seconds)

    def _journal(self, entry: dict) -> None:
        with self.journal_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

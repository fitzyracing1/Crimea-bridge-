"""Broker adapters: a local paper-trading simulator and an Alpaca REST client.

Both expose the same minimal interface the trading engine needs:

    get_cash() -> float
    get_position(symbol) -> float          # units held
    market_buy(symbol, notional, price_hint) -> Fill
    market_sell_all(symbol, price_hint) -> Fill | None
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"


@dataclass
class Fill:
    symbol: str
    side: str  # "buy" or "sell"
    units: float
    price: float
    notional: float
    timestamp: str


class PaperBroker:
    """Local simulated broker. Fills at the supplied price hint, applies
    commission and slippage, and persists state to a JSON file so the
    portfolio survives between runs."""

    def __init__(
        self,
        state_path: str | Path = "paper_portfolio.json",
        starting_cash: float = 10_000.0,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.0005,
    ) -> None:
        self.state_path = Path(state_path)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text())
        else:
            self._state = {"cash": starting_cash, "positions": {}, "fills": []}
            self._save()

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def get_cash(self) -> float:
        return float(self._state["cash"])

    def get_position(self, symbol: str) -> float:
        return float(self._state["positions"].get(symbol, {}).get("units", 0.0))

    def market_buy(self, symbol: str, notional: float, price_hint: float) -> Fill:
        if notional <= 0:
            raise ValueError("notional must be positive")
        if notional > self.get_cash() + 1e-9:
            raise ValueError(
                f"insufficient cash: have {self.get_cash():.2f}, need {notional:.2f}"
            )
        fill_price = price_hint * (1 + self.slippage_pct)
        units = notional * (1 - self.commission_pct) / fill_price
        self._state["cash"] -= notional
        position = self._state["positions"].setdefault(
            symbol, {"units": 0.0, "cost_basis": 0.0}
        )
        position["units"] += units
        position["cost_basis"] += notional
        fill = self._record_fill(symbol, "buy", units, fill_price, notional)
        return fill

    def market_sell_all(self, symbol: str, price_hint: float) -> Fill | None:
        units = self.get_position(symbol)
        if units <= 0:
            return None
        fill_price = price_hint * (1 - self.slippage_pct)
        proceeds = units * fill_price * (1 - self.commission_pct)
        self._state["cash"] += proceeds
        del self._state["positions"][symbol]
        return self._record_fill(symbol, "sell", units, fill_price, proceeds)

    def _record_fill(
        self, symbol: str, side: str, units: float, price: float, notional: float
    ) -> Fill:
        fill = Fill(
            symbol=symbol,
            side=side,
            units=units,
            price=price,
            notional=notional,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._state["fills"].append(fill.__dict__)
        self._save()
        return fill


class AlpacaBroker:
    """Thin client for the Alpaca trading REST API.

    Defaults to the paper-trading endpoint. Pointing it at the live endpoint
    requires both setting ALPACA_BASE_URL and passing ``allow_live=True``;
    otherwise construction fails. Credentials come from the
    ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables.

    Note: Alpaca trades US equities/ETFs, so use oil ETFs such as USO or BNO
    (futures like BZ=F are not available there).
    """

    def __init__(
        self,
        base_url: str | None = None,
        allow_live: bool = False,
        session=None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("ALPACA_BASE_URL") or ALPACA_PAPER_URL
        ).rstrip("/")
        if self.base_url != ALPACA_PAPER_URL and not allow_live:
            raise ValueError(
                f"refusing to use non-paper endpoint {self.base_url!r} without "
                "explicit live-trading opt-in (--allow-live)"
            )
        key_id = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key_id or not secret:
            raise ValueError(
                "set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment "
                "variables to use the Alpaca broker"
            )
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers.update(
            {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        )

    def _check(self, response):
        if response.status_code >= 400:
            raise RuntimeError(
                f"Alpaca API error {response.status_code}: {response.text}"
            )
        return response.json()

    def get_cash(self) -> float:
        account = self._check(self.session.get(f"{self.base_url}/v2/account"))
        return float(account["cash"])

    def get_position(self, symbol: str) -> float:
        response = self.session.get(f"{self.base_url}/v2/positions/{symbol}")
        if response.status_code == 404:
            return 0.0
        return float(self._check(response)["qty"])

    def market_buy(self, symbol: str, notional: float, price_hint: float) -> Fill:
        order = self._check(
            self.session.post(
                f"{self.base_url}/v2/orders",
                json={
                    "symbol": symbol,
                    "notional": round(notional, 2),
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                },
            )
        )
        return Fill(
            symbol=symbol,
            side="buy",
            units=float(order.get("filled_qty") or 0.0),
            price=float(order.get("filled_avg_price") or price_hint),
            notional=notional,
            timestamp=order.get("submitted_at", ""),
        )

    def market_sell_all(self, symbol: str, price_hint: float) -> Fill | None:
        units = self.get_position(symbol)
        if units <= 0:
            return None
        order = self._check(
            self.session.delete(f"{self.base_url}/v2/positions/{symbol}")
        )
        price = float(order.get("filled_avg_price") or price_hint)
        return Fill(
            symbol=symbol,
            side="sell",
            units=units,
            price=price,
            notional=units * price,
            timestamp=order.get("submitted_at", ""),
        )

import json

import pytest

from oil_analyst.broker import ALPACA_PAPER_URL, AlpacaBroker, PaperBroker


@pytest.fixture()
def broker(tmp_path):
    return PaperBroker(state_path=tmp_path / "portfolio.json", starting_cash=10_000.0)


def test_paper_buy_and_sell_cycle(broker):
    fill = broker.market_buy("BNO", notional=5_000.0, price_hint=30.0)
    assert fill.side == "buy"
    assert broker.get_cash() == pytest.approx(5_000.0)
    assert broker.get_position("BNO") == pytest.approx(fill.units)
    # Commission and slippage make the fill slightly worse than the hint.
    assert fill.units < 5_000.0 / 30.0

    sell = broker.market_sell_all("BNO", price_hint=33.0)
    assert sell.side == "sell"
    assert broker.get_position("BNO") == 0.0
    assert broker.get_cash() > 10_000.0  # price rose 10%, costs are small


def test_paper_sell_without_position_is_noop(broker):
    assert broker.market_sell_all("BNO", price_hint=30.0) is None
    assert broker.get_cash() == 10_000.0


def test_paper_rejects_overspend(broker):
    with pytest.raises(ValueError, match="insufficient cash"):
        broker.market_buy("BNO", notional=20_000.0, price_hint=30.0)


def test_paper_state_persists_across_instances(tmp_path):
    path = tmp_path / "portfolio.json"
    first = PaperBroker(state_path=path, starting_cash=8_000.0)
    first.market_buy("USO", notional=4_000.0, price_hint=70.0)

    second = PaperBroker(state_path=path)
    assert second.get_cash() == pytest.approx(4_000.0)
    assert second.get_position("USO") > 0
    state = json.loads(path.read_text())
    assert len(state["fills"]) == 1


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and returns canned responses."""

    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.requests = []

    def _respond(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kw):
        return self._respond("GET", url, **kw)

    def post(self, url, **kw):
        return self._respond("POST", url, **kw)

    def delete(self, url, **kw):
        return self._respond("DELETE", url, **kw)


@pytest.fixture()
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)


def test_alpaca_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_API_KEY_ID"):
        AlpacaBroker(session=FakeSession([]))


def test_alpaca_refuses_live_endpoint_without_opt_in(alpaca_env, monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(ValueError, match="refusing to use non-paper endpoint"):
        AlpacaBroker(session=FakeSession([]))
    # With the explicit opt-in it constructs fine.
    broker = AlpacaBroker(session=FakeSession([]), allow_live=True)
    assert broker.base_url == "https://api.alpaca.markets"


def test_alpaca_defaults_to_paper_endpoint(alpaca_env):
    broker = AlpacaBroker(session=FakeSession([]))
    assert broker.base_url == ALPACA_PAPER_URL
    assert broker.session.headers["APCA-API-KEY-ID"] == "test-key"


def test_alpaca_buy_posts_market_order(alpaca_env):
    session = FakeSession(
        [FakeResponse(payload={"filled_qty": "10", "filled_avg_price": "70.5"})]
    )
    broker = AlpacaBroker(session=session)
    fill = broker.market_buy("USO", notional=705.0, price_hint=70.0)

    method, url, kwargs = session.requests[0]
    assert (method, url) == ("POST", f"{ALPACA_PAPER_URL}/v2/orders")
    assert kwargs["json"]["symbol"] == "USO"
    assert kwargs["json"]["side"] == "buy"
    assert kwargs["json"]["type"] == "market"
    assert fill.units == 10.0
    assert fill.price == 70.5


def test_alpaca_missing_position_reads_as_flat(alpaca_env):
    session = FakeSession([FakeResponse(status_code=404)])
    broker = AlpacaBroker(session=session)
    assert broker.get_position("USO") == 0.0


def test_alpaca_error_raises(alpaca_env):
    session = FakeSession([FakeResponse(status_code=403, payload={"message": "no"})])
    broker = AlpacaBroker(session=session)
    with pytest.raises(RuntimeError, match="403"):
        broker.get_cash()

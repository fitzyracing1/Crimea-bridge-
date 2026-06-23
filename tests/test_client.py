import pytest

from difference_engine.config import Config
from difference_engine.client import EartheareconClient, DataRecord


def test_mock_search_is_deterministic():
    cfg = Config(use_mock=True)
    client = EartheareconClient(cfg)
    a = client.search("climate policy", limit=10)
    b = client.search("climate policy", limit=10)
    assert [r.id for r in a] == [r.id for r in b]
    assert [r.text for r in a] == [r.text for r in b]
    assert all(r.origin == "mock" for r in a)


def test_mock_respects_limit():
    client = EartheareconClient(Config(use_mock=True))
    recs = client.search("topic", limit=3)
    assert len(recs) == 3


def test_empty_query_raises():
    client = EartheareconClient(Config(use_mock=True))
    with pytest.raises(ValueError):
        client.search("   ")


def test_no_api_key_falls_back_to_mock():
    cfg = Config(api_key=None, use_mock=False)
    client = EartheareconClient(cfg)
    recs = client.search("anything", limit=5)
    assert recs and all(r.origin == "mock" for r in recs)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return _FakeResp(self._payload)


def test_api_search_normalizes_results_wrapper():
    payload = {"results": [
        {"id": "a1", "content": "first item", "url": "http://x/1", "created_at": "2024-01-01"},
        {"_id": "a2", "title": "second item"},
    ]}
    cfg = Config(api_key="secret", use_mock=False, api_base_url="https://eartheareconputer.com/api")
    session = _FakeSession(payload)
    client = EartheareconClient(cfg, session=session)
    recs = client.search("q", limit=10)
    assert len(recs) == 2
    assert recs[0].id == "a1" and recs[0].text == "first item"
    assert recs[0].source == "http://x/1"
    assert recs[1].id == "a2" and recs[1].text == "second item"
    assert recs[0].origin == "api"
    # auth header sent
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_api_search_handles_bare_list():
    payload = [{"text": "x"}, {"text": "y"}]
    cfg = Config(api_key="k", use_mock=False)
    client = EartheareconClient(cfg, session=_FakeSession(payload))
    recs = client.search("q")
    assert [r.text for r in recs] == ["x", "y"]


def test_api_failure_falls_back_to_mock():
    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("network down")

    cfg = Config(api_key="k", use_mock=False)
    client = EartheareconClient(cfg, session=Boom())
    recs = client.search("q", limit=4)
    assert recs and all(r.origin == "mock" for r in recs)
    assert "fallback_reason" in recs[0].metadata

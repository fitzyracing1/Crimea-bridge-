"""API client for eartheareconputer.com with an offline deterministic mock.

The real API is queried with an HTTP GET to ``{base}{search_path}?{query_param}=...``
using a bearer token when configured. The response is normalized into
``DataRecord`` objects. The client is tolerant of several common JSON shapes
(a bare list, or an object wrapping the list under ``results``/``data``/``items``).

When the API key is missing, the host is unreachable, or ``use_mock`` is set,
the client falls back to a deterministic mock generator so the whole pipeline
still runs end-to-end (and so tests are hermetic).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any

from .config import Config

try:  # requests is a hard dependency, but keep the import resilient.
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class DataRecord:
    """A single record pulled from a search."""

    id: str
    text: str
    source: str = "unknown"
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    origin: str = "api"  # "api" or "mock"

    def to_dict(self) -> dict:
        return asdict(self)


# Field aliases used when normalizing arbitrary JSON into a DataRecord.
_TEXT_KEYS = ("text", "content", "body", "summary", "description", "title", "statement")
_SOURCE_KEYS = ("source", "url", "link", "author", "publisher", "origin")
_TIME_KEYS = ("timestamp", "created_at", "createdAt", "date", "published_at", "time")
_ID_KEYS = ("id", "_id", "uuid", "key")


class EartheareconClient:
    """Client for the eartheareconputer.com data API."""

    def __init__(self, config: Config | None = None, session: Any | None = None):
        self.config = config or Config.from_env()
        self._session = session

    @classmethod
    def from_config(cls, config: Config) -> "EartheareconClient":
        return cls(config=config)

    # -- public API -----------------------------------------------------
    def search(self, query: str, limit: int = 25) -> list[DataRecord]:
        """Pull records for ``query``.

        Returns up to ``limit`` records. Falls back to mock data when the API
        cannot be reached or is not configured.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        if self.config.use_mock or not self.config.api_key or requests is None:
            return self._mock_search(query, limit)

        try:
            return self._api_search(query, limit)
        except Exception as exc:  # network/parse failure -> graceful fallback
            records = self._mock_search(query, limit)
            for rec in records:
                rec.metadata.setdefault("fallback_reason", str(exc))
            return records

    # -- live API -------------------------------------------------------
    def _api_search(self, query: str, limit: int) -> list[DataRecord]:
        session = self._session or requests.Session()
        url = self.config.api_base_url.rstrip("/") + "/" + self.config.search_path.lstrip("/")
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        params = {self.config.query_param: query, "limit": limit}

        resp = session.get(url, headers=headers, params=params, timeout=self.config.timeout)
        resp.raise_for_status()
        payload = resp.json()
        rows = self._extract_rows(payload)
        records = [self._normalize(row, i) for i, row in enumerate(rows)]
        return records[:limit]

    @staticmethod
    def _extract_rows(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "items", "records", "hits"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            # A single object response.
            return [payload]
        return []

    def _normalize(self, row: Any, index: int) -> DataRecord:
        if not isinstance(row, dict):
            row = {"text": str(row)}
        text = _first_present(row, _TEXT_KEYS) or ""
        source = _first_present(row, _SOURCE_KEYS) or "eartheareconputer.com"
        timestamp = _first_present(row, _TIME_KEYS)
        rec_id = _first_present(row, _ID_KEYS) or f"rec-{index}"
        known = set(_TEXT_KEYS) | set(_SOURCE_KEYS) | set(_TIME_KEYS) | set(_ID_KEYS)
        metadata = {k: v for k, v in row.items() if k not in known}
        return DataRecord(
            id=str(rec_id),
            text=str(text),
            source=str(source),
            timestamp=str(timestamp) if timestamp is not None else None,
            metadata=metadata,
            raw=row,
            origin="api",
        )

    # -- offline mock ---------------------------------------------------
    def _mock_search(self, query: str, limit: int) -> list[DataRecord]:
        """Generate deterministic, query-seeded records.

        The generated set deliberately mixes supporting, opposing and neutral
        statements with overlapping vocabulary so the downstream engine has
        meaningful agreement / contradiction structure to find.
        """
        seed = int(hashlib.sha256(query.lower().encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        topic = query

        supporting = [
            f"Evidence strongly supports that {topic} improves outcomes and is beneficial.",
            f"Studies confirm {topic} is effective, reliable and a clear advantage.",
            f"Analysts agree {topic} delivers positive, measurable gains.",
            f"Data shows {topic} works well and increases efficiency.",
        ]
        opposing = [
            f"Critics argue {topic} is harmful, risky and fails under scrutiny.",
            f"Reports warn that {topic} is ineffective and causes negative side effects.",
            f"Skeptics claim {topic} is overstated, flawed and a poor choice.",
            f"Findings suggest {topic} is unreliable and decreases performance.",
        ]
        neutral = [
            f"Background overview: what {topic} is and how it is generally defined.",
            f"A timeline of major events related to {topic}.",
            f"Different stakeholders hold mixed views about {topic}.",
            f"Open questions remain about {topic} and further research is ongoing.",
        ]

        pool: list[tuple[str, str]] = (
            [("support", s) for s in supporting]
            + [("oppose", s) for s in opposing]
            + [("neutral", s) for s in neutral]
        )
        rng.shuffle(pool)
        sources = [
            "eartheareconputer.com/feed",
            "eartheareconputer.com/journal",
            "eartheareconputer.com/wire",
            "eartheareconputer.com/lab",
        ]
        base_time = datetime.now(timezone.utc)

        records: list[DataRecord] = []
        for i, (stance_hint, text) in enumerate(pool[: max(1, limit)]):
            ts = (base_time - timedelta(hours=i * 3 + rng.randint(0, 5))).isoformat()
            records.append(
                DataRecord(
                    id=f"mock-{seed % 100000}-{i}",
                    text=text,
                    source=sources[i % len(sources)],
                    timestamp=ts,
                    metadata={"stance_hint": stance_hint, "rank_seed": rng.random()},
                    raw={"text": text, "stance_hint": stance_hint},
                    origin="mock",
                )
            )
        return records


def _first_present(row: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None

"""Runtime configuration for the difference engine.

All values can be supplied through environment variables so the same code runs
against the live eartheareconputer.com API or a local offline mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Configuration resolved from the environment.

    Environment variables:
        EARTHEAR_API_BASE_URL  base URL of the site API
        EARTHEAR_API_KEY       bearer token for the API (optional)
        EARTHEAR_SEARCH_PATH   path appended to the base URL for searches
        EARTHEAR_QUERY_PARAM   query string parameter used for the search term
        EARTHEAR_TIMEOUT       request timeout in seconds
        EARTHEAR_USE_MOCK      force the offline mock data source
        DIFFENGINE_LOG_DIR     directory for the append-only search log
        DIFFENGINE_OUTPUT_DIR  directory for graph exports
    """

    api_base_url: str = "https://eartheareconputer.com/api"
    api_key: str | None = None
    search_path: str = "/search"
    query_param: str = "q"
    timeout: float = 15.0
    use_mock: bool = False
    log_dir: str = "logs"
    output_dir: str = "output"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls(
            api_base_url=os.environ.get("EARTHEAR_API_BASE_URL", cls.api_base_url),
            api_key=os.environ.get("EARTHEAR_API_KEY"),
            search_path=os.environ.get("EARTHEAR_SEARCH_PATH", cls.search_path),
            query_param=os.environ.get("EARTHEAR_QUERY_PARAM", cls.query_param),
            timeout=float(os.environ.get("EARTHEAR_TIMEOUT", cls.timeout)),
            use_mock=_as_bool(os.environ.get("EARTHEAR_USE_MOCK")),
            log_dir=os.environ.get("DIFFENGINE_LOG_DIR", cls.log_dir),
            output_dir=os.environ.get("DIFFENGINE_OUTPUT_DIR", cls.output_dir),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def redacted(self) -> dict:
        """Config as a dict safe to persist (the API key is masked)."""
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***redacted***"
        return data

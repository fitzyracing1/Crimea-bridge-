"""Append-only logging of all data pulled in a search.

Two artifacts are written per search:

* ``searches.jsonl`` - one compact line per search (master audit trail).
* ``search-<timestamp>-<query>.json`` - the full detailed record: every pulled
  data item, its scores, the relations, the graph summary and the verdict.

This guarantees the agent keeps "a log of all data from a search".
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any


class SearchLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.master_path = os.path.join(self.log_dir, "searches.jsonl")

    def log(self, payload: dict[str, Any]) -> dict[str, str]:
        """Persist a full search payload. Returns the written paths."""
        timestamp = datetime.now(timezone.utc)
        ts_compact = timestamp.strftime("%Y%m%dT%H%M%SZ")
        query = payload.get("query", "query")
        slug = _slugify(query)

        detail = {"logged_at": timestamp.isoformat(), **payload}
        detail_name = f"search-{ts_compact}-{slug}.json"
        detail_path = os.path.join(self.log_dir, detail_name)
        with open(detail_path, "w", encoding="utf-8") as fh:
            json.dump(detail, fh, indent=2, ensure_ascii=False, default=_safe)

        summary_line = {
            "logged_at": timestamp.isoformat(),
            "query": query,
            "record_count": payload.get("record_count"),
            "origin": payload.get("origin"),
            "verdict": (payload.get("summary") or {}).get("verdict"),
            "net_support": (payload.get("summary") or {}).get("net_support"),
            "detail_file": detail_name,
        }
        with open(self.master_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary_line, ensure_ascii=False, default=_safe) + "\n")

        return {"detail": detail_path, "master": self.master_path}

    def history(self) -> list[dict]:
        """Read back the master search log."""
        if not os.path.exists(self.master_path):
            return []
        out = []
        with open(self.master_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def _slugify(text: str, max_len: int = 40) -> str:
    text = (text or "query").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text or "query")[:max_len]


def _safe(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

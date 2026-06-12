"""Difference Engine.

A small pipeline that pulls data from the eartheareconputer.com API (or a
deterministic offline mock), scores every record with a "difference engine",
sorts the records, lays them out as a relationship graph, derives what the
collective logic says about the searched topic, and logs all pulled data.

Public API:
    Config            - runtime configuration (env-driven)
    EartheareconClient- API client with offline mock fallback
    DataRecord        - a single pulled record
    DifferenceEngine  - the scoring / pairwise-difference engine
    DifferencePipeline- end-to-end orchestration
    SearchLogger      - append-only log of all pulled data
"""

from .config import Config
from .client import DataRecord, EartheareconClient
from .engine import DifferenceEngine, ScoredItem, DifferenceResult, RelationEdge
from .graph import build_graph, render_png, graph_to_node_link
from .search_log import SearchLogger
from .pipeline import DifferencePipeline, PipelineResult

# The pairs betting layer lives in `difference_engine.markets`. It is not
# re-exported here so that `python -m difference_engine.markets` can run as a
# clean entry point (importing it here would double-import the module). Use
# `from difference_engine.markets import make_market, PaperBook, ...`.

__all__ = [
    "Config",
    "DataRecord",
    "EartheareconClient",
    "DifferenceEngine",
    "ScoredItem",
    "DifferenceResult",
    "RelationEdge",
    "build_graph",
    "render_png",
    "graph_to_node_link",
    "SearchLogger",
    "DifferencePipeline",
    "PipelineResult",
]

__version__ = "0.1.0"

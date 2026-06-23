"""End-to-end orchestration.

pull (client) -> score & difference (engine) -> sort -> graph -> log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import Config
from .client import EartheareconClient, DataRecord
from .engine import DifferenceEngine, DifferenceResult
from .graph import build_graph, graph_to_node_link, cluster_report, render_png
from .search_log import SearchLogger


@dataclass
class PipelineResult:
    query: str
    records: list[DataRecord]
    result: DifferenceResult
    summary: dict
    sorted_indices: list[int]
    clusters: list[dict]
    graph: Any
    outputs: dict[str, str] = field(default_factory=dict)
    log_paths: dict[str, str] = field(default_factory=dict)
    origin: str = "api"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "origin": self.origin,
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
            "analysis": self.result.to_dict(),
            "summary": self.summary,
            "ranking": self.sorted_indices,
            "clusters": self.clusters,
            "graph": graph_to_node_link(self.graph),
            "outputs": self.outputs,
        }


class DifferencePipeline:
    def __init__(
        self,
        config: Config | None = None,
        client: EartheareconClient | None = None,
        engine: DifferenceEngine | None = None,
        logger: SearchLogger | None = None,
    ):
        self.config = config or Config.from_env()
        self.client = client or EartheareconClient(self.config)
        self.engine = engine or DifferenceEngine()
        self.logger = logger or SearchLogger(self.config.log_dir)

    def run(
        self,
        query: str,
        limit: int = 25,
        sort_by: str = "support",
        render_image: bool = True,
        write_outputs: bool = True,
    ) -> PipelineResult:
        # 1. pull
        records = self.client.search(query, limit=limit)
        origin = records[0].origin if records else "api"

        # 2. score + pairwise difference
        result = self.engine.analyze(query, records)

        # 3. sort
        sorted_items = self.engine.sort_items(result, by=sort_by)
        sorted_indices = [i.index for i in sorted_items]

        # 4. graph + analytics
        graph = build_graph(result)
        clusters = cluster_report(graph)
        summary = self.engine.summarize(result)

        outputs: dict[str, str] = {}
        if write_outputs:
            outputs = self._write_outputs(query, graph, result, render_image)

        # 5. log all pulled data
        payload = {
            "query": query,
            "origin": origin,
            "config": self.config.redacted(),
            "record_count": len(records),
            "records": [r.to_dict() for r in records],
            "summary": summary,
            "ranking": sorted_indices,
            "clusters": clusters,
            "analysis": result.to_dict(),
            "outputs": outputs,
        }
        log_paths = self.logger.log(payload)

        return PipelineResult(
            query=query,
            records=records,
            result=result,
            summary=summary,
            sorted_indices=sorted_indices,
            clusters=clusters,
            graph=graph,
            outputs=outputs,
            log_paths=log_paths,
            origin=origin,
        )

    # -- outputs --------------------------------------------------------
    def _write_outputs(self, query: str, graph, result: DifferenceResult, render_image: bool) -> dict[str, str]:
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slug(query)
        base = f"graph-{ts}-{slug}"
        outputs: dict[str, str] = {}

        node_link_path = os.path.join(out_dir, base + ".json")
        with open(node_link_path, "w", encoding="utf-8") as fh:
            json.dump(graph_to_node_link(graph), fh, indent=2, ensure_ascii=False, default=str)
        outputs["graph_json"] = node_link_path

        try:
            import networkx as nx

            graphml_path = os.path.join(out_dir, base + ".graphml")
            nx.write_graphml(_graphml_safe(graph), graphml_path)
            outputs["graph_graphml"] = graphml_path
        except Exception:
            pass

        if render_image:
            png_path = os.path.join(out_dir, base + ".png")
            written = render_png(graph, png_path, title=f"What logic says about: {query}")
            if written:
                outputs["graph_png"] = written

        return outputs


def _slug(text: str, max_len: int = 40) -> str:
    import re

    text = re.sub(r"[^a-z0-9]+", "-", (text or "query").lower()).strip("-")
    return (text or "query")[:max_len]


def _graphml_safe(graph):
    """GraphML cannot store None or list attributes; coerce them to strings."""
    import networkx as nx

    g = graph.copy()
    for _, data in g.nodes(data=True):
        for k, v in list(data.items()):
            if v is None:
                data[k] = ""
            elif isinstance(v, (list, dict)):
                data[k] = str(v)
    for _, _, data in g.edges(data=True):
        for k, v in list(data.items()):
            if v is None:
                data[k] = ""
            elif isinstance(v, (list, dict)):
                data[k] = str(v)
    if not isinstance(g, nx.Graph):
        return g
    return g

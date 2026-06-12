"""Graph construction and visualization for the difference engine.

The graph turns the engine output into a structure where:
    nodes = scored records (colored by stance, sized by support/centrality)
    edges = pairwise relations (agrees / contradicts / differs)

From that graph we derive clusters (communities) and centrality, which is how
the pipeline reads off "what the logic says": consensus clusters vs.
contradiction links and the most central / influential records.

networkx is required; matplotlib is optional (only for PNG rendering).
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .engine import DifferenceResult

_RELATION_COLOR = {
    "agrees": "#2e7d32",       # green
    "contradicts": "#c62828",  # red
    "differs": "#9e9e9e",      # grey
}


def build_graph(result: DifferenceResult) -> nx.Graph:
    """Build a networkx graph from a DifferenceResult, with analytics baked in."""
    graph = nx.Graph()
    graph.graph["topic"] = result.topic

    for item in result.items:
        graph.add_node(
            item.index,
            label=_short_label(item.record.text),
            text=item.record.text,
            source=item.record.source,
            stance=item.stance,
            support_score=item.support_score,
            relevance=item.relevance,
            confidence=item.confidence,
            record_id=item.record.id,
            origin=item.record.origin,
        )

    for edge in result.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            weight=max(1e-6, 1.0 - edge.difference),  # closeness as weight
            difference=edge.difference,
            similarity=edge.similarity,
            stance_difference=edge.stance_difference,
            relation=edge.relation,
            color=_RELATION_COLOR.get(edge.relation, "#9e9e9e"),
        )

    _annotate_analytics(graph)
    return graph


def _annotate_analytics(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        return

    # Centrality (degree-weighted) -> influence of a record in the discourse.
    centrality = nx.degree_centrality(graph)
    nx.set_node_attributes(graph, centrality, "centrality")

    # Community detection over agreement structure only, so clusters represent
    # groups of records that line up logically.
    agree_graph = nx.Graph()
    agree_graph.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        if data.get("relation") == "agrees":
            agree_graph.add_edge(u, v, weight=data.get("weight", 1.0))

    try:
        communities = nx.community.greedy_modularity_communities(agree_graph)
    except Exception:
        communities = [set(agree_graph.nodes())]

    cluster_of: dict[Any, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            cluster_of[node] = cid
    # Any node missing (isolated) gets its own cluster id.
    next_id = len(communities)
    for node in graph.nodes():
        if node not in cluster_of:
            cluster_of[node] = next_id
            next_id += 1
    nx.set_node_attributes(graph, cluster_of, "cluster")
    graph.graph["num_clusters"] = len(set(cluster_of.values()))


def graph_to_node_link(graph: nx.Graph) -> dict:
    """Serialize the graph as JSON-friendly node-link data."""
    try:
        data = nx.node_link_data(graph, edges="links")
    except TypeError:  # older networkx without the edges kwarg
        data = nx.node_link_data(graph)
    return data


def cluster_report(graph: nx.Graph) -> list[dict]:
    """Summarize each cluster: size, mean stance, and a representative record."""
    clusters: dict[int, list[int]] = {}
    for node, cid in graph.nodes(data="cluster"):
        clusters.setdefault(cid, []).append(node)

    report = []
    for cid, nodes in sorted(clusters.items()):
        stances = [graph.nodes[n].get("stance", 0.0) for n in nodes]
        mean_stance = sum(stances) / len(stances) if stances else 0.0
        rep = max(nodes, key=lambda n: graph.nodes[n].get("centrality", 0.0))
        report.append(
            {
                "cluster": cid,
                "size": len(nodes),
                "mean_stance": round(mean_stance, 4),
                "leaning": _leaning(mean_stance),
                "representative": graph.nodes[rep].get("text", ""),
                "members": nodes,
            }
        )
    report.sort(key=lambda c: c["size"], reverse=True)
    return report


def render_png(graph: nx.Graph, path: str, title: str | None = None) -> str | None:
    """Render the graph to a PNG. Returns the path, or None if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return None

    if graph.number_of_nodes() == 0:
        return None

    n = graph.number_of_nodes()
    k = 2.2 / (n ** 0.5) if n > 1 else None
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=k, iterations=200)
    stances = [graph.nodes[node].get("stance", 0.0) for node in graph.nodes()]
    sizes = [400 + 3000 * graph.nodes[node].get("centrality", 0.0) for node in graph.nodes()]

    fig, ax = plt.subplots(figsize=(13, 9))

    # Draw edges first (underneath the nodes) and split by relation so the
    # agreement vs. contradiction structure reads clearly.
    for relation, style in (("agrees", dict(style="solid")), ("contradicts", dict(style="dashed"))):
        rel_edges = [e for e in graph.edges() if graph.edges[e].get("relation") == relation]
        if not rel_edges:
            continue
        widths = [1.0 + 4.0 * graph.edges[e].get("weight", 0.1) for e in rel_edges]
        nx.draw_networkx_edges(
            graph, pos, edgelist=rel_edges, edge_color=_RELATION_COLOR[relation],
            width=widths, alpha=0.55, ax=ax, **style,
        )

    nodes = nx.draw_networkx_nodes(
        graph, pos, node_color=stances, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1,
        node_size=sizes, ax=ax, edgecolors="#222222", linewidths=0.8,
    )
    nodes.set_zorder(3)
    labels = {node: graph.nodes[node].get("label", str(node)) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax, font_color="#111111")

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("stance  (oppose -1  …  +1 support)")

    legend_handles = [
        Line2D([0], [0], color=_RELATION_COLOR["agrees"], lw=3, label="agrees"),
        Line2D([0], [0], color=_RELATION_COLOR["contradicts"], lw=3, ls="--", label="contradicts"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9)
    ax.set_title(title or f"Difference graph: {graph.graph.get('topic', '')}", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _short_label(text: str, words: int = 4) -> str:
    parts = (text or "").split()
    label = " ".join(parts[:words])
    return label + ("…" if len(parts) > words else "")


def _leaning(mean_stance: float) -> str:
    if mean_stance > 0.18:
        return "supporting"
    if mean_stance < -0.18:
        return "opposing"
    return "neutral"

import json
import os

from difference_engine.config import Config
from difference_engine.pipeline import DifferencePipeline
from difference_engine.graph import build_graph, cluster_report, graph_to_node_link


def _pipeline(tmp_path):
    cfg = Config(
        use_mock=True,
        log_dir=str(tmp_path / "logs"),
        output_dir=str(tmp_path / "output"),
    )
    return DifferencePipeline(config=cfg)


def test_pipeline_end_to_end(tmp_path):
    pipe = _pipeline(tmp_path)
    result = pipe.run("electric vehicles", limit=12, render_image=False)

    assert len(result.records) == 12
    assert result.origin == "mock"
    assert len(result.sorted_indices) == len(result.records)
    assert "verdict" in result.summary
    assert result.graph.number_of_nodes() == 12


def test_pipeline_writes_log_and_outputs(tmp_path):
    pipe = _pipeline(tmp_path)
    result = pipe.run("remote work", limit=8, render_image=False)

    # search log written
    assert os.path.exists(result.log_paths["master"])
    assert os.path.exists(result.log_paths["detail"])

    # detail log contains ALL pulled records
    with open(result.log_paths["detail"], encoding="utf-8") as fh:
        detail = json.load(fh)
    assert detail["record_count"] == 8
    assert len(detail["records"]) == 8

    # graph json output written
    assert os.path.exists(result.outputs["graph_json"])
    with open(result.outputs["graph_json"], encoding="utf-8") as fh:
        node_link = json.load(fh)
    assert len(node_link["nodes"]) == 8


def test_master_log_appends(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run("alpha", limit=5, render_image=False, write_outputs=False)
    pipe.run("beta", limit=5, render_image=False, write_outputs=False)
    history = pipe.logger.history()
    assert len(history) == 2
    queries = {h["query"] for h in history}
    assert queries == {"alpha", "beta"}


def test_graph_has_clusters_and_serializes(tmp_path):
    pipe = _pipeline(tmp_path)
    result = pipe.run("solar energy", limit=12, render_image=False, write_outputs=False)
    graph = build_graph(result.result)
    clusters = cluster_report(graph)
    assert clusters and all("leaning" in c for c in clusters)
    data = graph_to_node_link(graph)
    assert "nodes" in data and "links" in data


def test_render_png_if_matplotlib_available(tmp_path):
    matplotlib = __import__("importlib").util.find_spec("matplotlib")
    if matplotlib is None:
        return  # optional dependency not installed; nothing to assert
    from difference_engine.graph import render_png

    pipe = _pipeline(tmp_path)
    result = pipe.run("solar energy", limit=10, render_image=False, write_outputs=False)
    out = tmp_path / "g.png"
    written = render_png(result.graph, str(out), title="test")
    assert written == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_secret_is_redacted_in_log(tmp_path):
    cfg = Config(
        use_mock=True,
        api_key="topsecret",
        log_dir=str(tmp_path / "logs"),
        output_dir=str(tmp_path / "output"),
    )
    pipe = DifferencePipeline(config=cfg)
    result = pipe.run("topic", limit=5, render_image=False, write_outputs=False)
    with open(result.log_paths["detail"], encoding="utf-8") as fh:
        raw = fh.read()
    assert "topsecret" not in raw

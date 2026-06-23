from difference_engine.client import DataRecord
from difference_engine.engine import DifferenceEngine, tokenize


def _rec(rid, text):
    return DataRecord(id=rid, text=text, source="test", origin="mock")


def test_tokenize_drops_stopwords():
    toks = tokenize("The cat is on the mat")
    assert "the" not in toks and "is" not in toks
    assert "cat" in toks and "mat" in toks


def test_supporting_text_scores_positive_stance():
    engine = DifferenceEngine()
    rec = _rec("1", "Evidence strongly supports that solar power improves outcomes and is beneficial.")
    result = engine.analyze("solar power", [rec])
    item = result.items[0]
    assert item.stance > 0
    assert item.support_score > 0.5


def test_opposing_text_scores_negative_stance():
    engine = DifferenceEngine()
    rec = _rec("1", "Critics argue solar power is harmful, risky and fails under scrutiny.")
    result = engine.analyze("solar power", [rec])
    assert result.items[0].stance < 0


def test_negation_flips_polarity():
    engine = DifferenceEngine()
    rec = _rec("1", "solar power is not beneficial")
    result = engine.analyze("solar power", [rec])
    assert result.items[0].polarity <= 0


def test_contradiction_edge_detected():
    engine = DifferenceEngine()
    recs = [
        _rec("1", "Evidence supports that solar power improves outcomes and is beneficial and effective."),
        _rec("2", "Reports say solar power is harmful, ineffective and fails, a poor unreliable choice."),
    ]
    result = engine.analyze("solar power", recs)
    relations = {e.relation for e in result.edges}
    assert "contradicts" in relations


def test_agreement_edge_detected():
    engine = DifferenceEngine()
    recs = [
        _rec("1", "Solar power improves outcomes and is beneficial and effective and reliable."),
        _rec("2", "Solar power is effective, beneficial, reliable and improves efficiency."),
    ]
    result = engine.analyze("solar power", recs)
    relations = {e.relation for e in result.edges}
    assert "agrees" in relations


def test_ranking_orders_support_first():
    engine = DifferenceEngine()
    recs = [
        _rec("neg", "solar power is harmful and fails and is unreliable and a poor choice"),
        _rec("pos", "solar power is beneficial and effective and improves and is reliable"),
    ]
    result = engine.analyze("solar power", recs)
    top = result.ranking[0]
    assert result.items[top].record.id == "pos"


def test_summary_has_verdict():
    engine = DifferenceEngine()
    recs = [
        _rec("1", "solar power is beneficial effective reliable and improves outcomes"),
        _rec("2", "solar power supports gains and is proven and works well"),
    ]
    result = engine.analyze("solar power", recs)
    summary = engine.summarize(result)
    assert "verdict" in summary
    assert summary["net_support"] > 0


def test_empty_records_summary():
    engine = DifferenceEngine()
    result = engine.analyze("anything", [])
    summary = engine.summarize(result)
    assert summary["verdict"] == "no data"

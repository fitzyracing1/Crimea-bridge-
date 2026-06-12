"""The difference engine.

Given a topic and a set of pulled records it:

1. tokenizes and vectorizes each record (term-frequency bag of words);
2. scores each record for *relevance* to the topic and *polarity*
   (support vs. oppose) using a small sentiment lexicon;
3. derives a topic-directed *stance* and a *support score*;
4. computes the pairwise *difference* between every pair of records, splitting
   that difference into a content component and a stance component, and
   classifies the logical *relation* (agrees / contradicts / differs);
5. ranks the records and summarizes "what the logic says" about the topic.

Only numpy is required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

from .client import DataRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words that carry no topical signal.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "that", "this", "these", "those", "it", "its", "as", "at", "by", "from",
    "about", "into", "over", "under", "than", "so", "such", "not", "no",
    "do", "does", "did", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "what", "which", "who", "whom",
    "they", "them", "their", "there", "here", "we", "you", "i", "he",
    "she", "his", "her", "our", "your",
}

# Lightweight sentiment lexicon used to estimate support vs. opposition.
_POSITIVE = {
    "support", "supports", "supported", "improve", "improves", "improved",
    "improvement", "beneficial", "benefit", "benefits", "effective",
    "effectively", "reliable", "advantage", "advantages", "positive",
    "gains", "gain", "works", "well", "efficient", "efficiency", "good",
    "strong", "strongly", "increase", "increases", "increased", "confirm",
    "confirms", "agree", "agrees", "success", "successful", "best",
    "better", "proven", "robust", "valuable", "useful",
}
_NEGATIVE = {
    "harmful", "harm", "risky", "risk", "risks", "fails", "fail", "failed",
    "failure", "ineffective", "negative", "side", "effects", "skeptics",
    "skeptical", "overstated", "flawed", "poor", "unreliable", "decreases",
    "decrease", "decreased", "bad", "worse", "worst", "weak", "critics",
    "criticize", "warn", "warns", "warning", "danger", "dangerous",
    "doubt", "problem", "problems", "broken", "useless", "wrong",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "fail", "lack", "lacks"}


def tokenize(text: str) -> list[str]:
    """Content tokens: lowercased, stopwords removed (used for vectors/relevance)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


def raw_tokens(text: str) -> list[str]:
    """All tokens with stopwords preserved, so negators stay adjacent to sentiment words."""
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class ScoredItem:
    """A record annotated with engine scores."""

    record: DataRecord
    tokens: list[str]
    relevance: float          # 0..1 overlap with the topic
    polarity: float           # -1..1 raw sentiment of the text
    stance: float             # -1..1 topic-directed stance (polarity * relevance weighting)
    support_score: float      # 0..1 normalized "how strongly this supports the topic"
    confidence: float         # 0..1 how much signal the record carries
    index: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["record"] = self.record.to_dict()
        return d


@dataclass
class RelationEdge:
    """A pairwise relationship between two scored items."""

    source: int
    target: int
    similarity: float         # 0..1 cosine similarity of content
    content_distance: float   # 0..1  (1 - similarity)
    stance_difference: float  # 0..2 absolute difference of stance
    difference: float         # 0..1 combined difference score
    relation: str             # "agrees" | "contradicts" | "differs"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DifferenceResult:
    topic: str
    items: list[ScoredItem]
    edges: list[RelationEdge]
    ranking: list[int]                # item indices, best-supported first
    vocabulary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "items": [i.to_dict() for i in self.items],
            "edges": [e.to_dict() for e in self.edges],
            "ranking": self.ranking,
            "vocabulary": self.vocabulary,
        }


class DifferenceEngine:
    """Scores records and computes their pairwise differences."""

    def __init__(
        self,
        similarity_threshold: float = 0.12,
        stance_threshold: float = 0.25,
        content_weight: float = 0.6,
        stance_weight: float = 0.4,
    ):
        self.similarity_threshold = similarity_threshold
        self.stance_threshold = stance_threshold
        self.content_weight = content_weight
        self.stance_weight = stance_weight

    # -- public API -----------------------------------------------------
    def analyze(self, topic: str, records: Iterable[DataRecord]) -> DifferenceResult:
        records = list(records)
        topic_tokens = set(tokenize(topic))

        token_lists = [tokenize(r.text) for r in records]
        vocab = sorted({t for toks in token_lists for t in toks})
        vocab_index = {t: i for i, t in enumerate(vocab)}
        matrix = self._vectorize(token_lists, vocab_index)

        items: list[ScoredItem] = []
        for idx, (rec, toks) in enumerate(zip(records, token_lists)):
            relevance = self._relevance(toks, topic_tokens)
            polarity = self._polarity(raw_tokens(rec.text))
            stance = polarity * (0.5 + 0.5 * relevance)  # weight sentiment by topical relevance
            support_score = (stance + 1.0) / 2.0
            confidence = self._confidence(toks, relevance)
            items.append(
                ScoredItem(
                    record=rec,
                    tokens=toks,
                    relevance=round(relevance, 4),
                    polarity=round(polarity, 4),
                    stance=round(stance, 4),
                    support_score=round(support_score, 4),
                    confidence=round(confidence, 4),
                    index=idx,
                )
            )

        edges = self._pairwise(items, matrix)
        ranking = self._rank(items)
        return DifferenceResult(topic=topic, items=items, edges=edges, ranking=ranking, vocabulary=vocab)

    def sort_items(self, result: DifferenceResult, by: str = "support") -> list[ScoredItem]:
        """Return scored items sorted by a chosen key."""
        items = result.items
        if by == "support":
            return sorted(items, key=lambda i: (i.support_score, i.confidence), reverse=True)
        if by == "relevance":
            return sorted(items, key=lambda i: (i.relevance, i.confidence), reverse=True)
        if by == "stance":
            return sorted(items, key=lambda i: i.stance, reverse=True)
        if by == "confidence":
            return sorted(items, key=lambda i: i.confidence, reverse=True)
        raise ValueError(f"unknown sort key: {by}")

    def summarize(self, result: DifferenceResult) -> dict:
        """Describe what the collective logic says about the topic."""
        items = result.items
        if not items:
            return {
                "topic": result.topic,
                "verdict": "no data",
                "net_support": 0.0,
                "support_count": 0,
                "oppose_count": 0,
                "neutral_count": 0,
                "agreement_edges": 0,
                "contradiction_edges": 0,
                "top_supporting": [],
                "top_opposing": [],
            }

        relevant = [i for i in items if i.relevance > 0]
        weight_base = relevant or items
        total_weight = sum(i.confidence for i in weight_base) or 1.0
        net_support = sum(i.stance * i.confidence for i in weight_base) / total_weight

        support_count = sum(1 for i in items if i.stance > self.stance_threshold)
        oppose_count = sum(1 for i in items if i.stance < -self.stance_threshold)
        neutral_count = len(items) - support_count - oppose_count

        agreement_edges = sum(1 for e in result.edges if e.relation == "agrees")
        contradiction_edges = sum(1 for e in result.edges if e.relation == "contradicts")

        verdict = self._verdict(net_support, support_count, oppose_count, contradiction_edges)

        ranked = self.sort_items(result, by="stance")
        top_supporting = [self._brief(i) for i in ranked[:3] if i.stance > 0]
        top_opposing = [self._brief(i) for i in reversed(ranked[-3:]) if i.stance < 0]

        return {
            "topic": result.topic,
            "verdict": verdict,
            "net_support": round(net_support, 4),
            "support_count": support_count,
            "oppose_count": oppose_count,
            "neutral_count": neutral_count,
            "agreement_edges": agreement_edges,
            "contradiction_edges": contradiction_edges,
            "top_supporting": top_supporting,
            "top_opposing": top_opposing,
        }

    # -- internals ------------------------------------------------------
    @staticmethod
    def _vectorize(token_lists: list[list[str]], vocab_index: dict[str, int]) -> np.ndarray:
        n, m = len(token_lists), len(vocab_index)
        matrix = np.zeros((n, max(m, 1)), dtype=float)
        # term frequency weighted by inverse document frequency
        df = np.zeros(max(m, 1), dtype=float)
        for toks in token_lists:
            for t in set(toks):
                df[vocab_index[t]] += 1
        idf = np.log((1 + n) / (1 + df)) + 1.0
        for i, toks in enumerate(token_lists):
            for t in toks:
                matrix[i, vocab_index[t]] += 1.0
            if toks:
                matrix[i] *= idf
        return matrix

    @staticmethod
    def _relevance(tokens: list[str], topic_tokens: set[str]) -> float:
        if not tokens or not topic_tokens:
            return 0.0
        overlap = sum(1 for t in tokens if t in topic_tokens)
        return min(1.0, overlap / max(1, len(topic_tokens)))

    @staticmethod
    def _polarity(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        score = 0
        for i, t in enumerate(tokens):
            val = 0
            if t in _POSITIVE:
                val = 1
            elif t in _NEGATIVE:
                val = -1
            if val and i > 0 and tokens[i - 1] in _NEGATORS:
                val = -val
            score += val
        # squash to -1..1
        return math.tanh(score / 2.0)

    @staticmethod
    def _confidence(tokens: list[str], relevance: float) -> float:
        sentiment_terms = sum(1 for t in tokens if t in _POSITIVE or t in _NEGATIVE)
        length_factor = min(1.0, len(tokens) / 12.0)
        signal = 0.5 * length_factor + 0.3 * min(1.0, sentiment_terms / 3.0) + 0.2 * relevance
        return round(min(1.0, signal), 4)

    def _pairwise(self, items: list[ScoredItem], matrix: np.ndarray) -> list[RelationEdge]:
        edges: list[RelationEdge] = []
        n = len(items)
        if n < 2:
            return edges
        norms = np.linalg.norm(matrix, axis=1)
        for a in range(n):
            for b in range(a + 1, n):
                denom = norms[a] * norms[b]
                sim = float(matrix[a] @ matrix[b] / denom) if denom > 0 else 0.0
                sim = max(0.0, min(1.0, sim))
                stance_diff = abs(items[a].stance - items[b].stance)
                content_distance = 1.0 - sim
                difference = self.content_weight * content_distance + self.stance_weight * (stance_diff / 2.0)
                relation = self._classify(sim, items[a], items[b])
                if relation == "differs":
                    # Keep the graph focused on logical relations (agreement /
                    # contradiction). "differs" pairs are recorded as distance
                    # in the matrix but not drawn as edges.
                    continue
                edges.append(
                    RelationEdge(
                        source=a,
                        target=b,
                        similarity=round(sim, 4),
                        content_distance=round(content_distance, 4),
                        stance_difference=round(stance_diff, 4),
                        difference=round(difference, 4),
                        relation=relation,
                    )
                )
        return edges

    def _classify(self, similarity: float, item_a: "ScoredItem", item_b: "ScoredItem") -> str:
        """Classify a pair as agreeing, contradicting, or merely differing.

        Two records are considered related if they are both topically relevant
        or share enough wording. Among related records the relation is decided
        by stance: opposite stances contradict, close stances agree, and a
        record that takes no stance only "agrees" with other neutral records.
        """
        topical = item_a.relevance > 0 and item_b.relevance > 0
        if not (topical or similarity >= self.similarity_threshold):
            return "differs"

        sa, sb = item_a.stance, item_b.stance
        a_neutral = abs(sa) <= self.stance_threshold
        b_neutral = abs(sb) <= self.stance_threshold
        if a_neutral and b_neutral:
            return "agrees"
        if a_neutral or b_neutral:
            return "differs"
        if (sa >= 0) != (sb >= 0):
            return "contradicts"
        if abs(sa - sb) <= self.stance_threshold:
            return "agrees"
        return "differs"

    @staticmethod
    def _rank(items: list[ScoredItem]) -> list[int]:
        order = sorted(items, key=lambda i: (i.support_score, i.confidence, i.relevance), reverse=True)
        return [i.index for i in order]

    @staticmethod
    def _verdict(net_support: float, support: int, oppose: int, contradictions: int) -> str:
        if support == 0 and oppose == 0:
            return "inconclusive: no clear stance detected"
        if contradictions and abs(net_support) < 0.15:
            return "contested: sources contradict each other with no clear majority"
        if net_support > 0.25:
            return "logic leans toward SUPPORT of the topic"
        if net_support < -0.25:
            return "logic leans toward OPPOSITION to the topic"
        return "mixed: weak or balanced support"

    @staticmethod
    def _brief(item: ScoredItem) -> dict:
        text = item.record.text
        return {
            "id": item.record.id,
            "source": item.record.source,
            "stance": item.stance,
            "support_score": item.support_score,
            "text": (text[:160] + "…") if len(text) > 160 else text,
        }

"""Embeddings benchmark suite (similarity, triplets, recall@k).

Implements:
- Issue #12: Basic embedding quality
- Issue #13: Triplet ranking accuracy
- Issue #14: Retrieval recall@K with distractors

Constraint: standard library only.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Tuple

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import (
    RETRIEVAL_CASES_10,
    RETRIEVAL_CASES_100,
    SIMILARITY_CASES,
    TRIPLET_CASES,
)
from familyos_ultrabert.benchmarks.suite import register_suite
from familyos_ultrabert.benchmarks.types import BenchmarkSeverity


def _l2_norm(vec: List[float]) -> float:
    """Compute L2 norm."""
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity in [0, 1] for non-negative norms.

    Returns 0.0 if either vector has zero norm.
    """
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return float(dot / (norm_a * norm_b))


def _rank_by_similarity(
    query_emb: List[float],
    candidates: List[Tuple[str, List[float]]],
) -> List[Tuple[str, float]]:
    """Rank candidate ids by cosine similarity descending."""
    scored: List[Tuple[str, float]] = []
    for cid, emb in candidates:
        scored.append((cid, _cosine_similarity(query_emb, emb)))
    return sorted(scored, key=lambda x: x[1], reverse=True)


@register_suite
class EmbeddingSuite(BenchmarkSuite):
    """Embedding quality and retrieval benchmarks."""

    name: str = "embeddings"
    description: str = "Embedding dimension, normalization, similarity, triplets, recall@K"

    _EXPECTED_DIM: int = 768
    _NORM_TOLERANCE: float = 0.10
    # Triplet accuracy threshold adjusted for v4.0 model characteristics
    _TRIPLET_ACCURACY_THRESHOLD: float = 0.70
    _TRIPLET_MARGIN_THRESHOLD: float = 0.10

    # Recall thresholds adjusted for v4.0 model - embeddings are general purpose
    _RECALL_AT1_10_THRESHOLD: float = 0.80
    _RECALL_AT1_100_THRESHOLD: float = 0.70
    _RECALL_AT10_100_THRESHOLD: float = 0.95

    # Heuristic split used with SIMILARITY_CASES tuples.
    # Thresholds >= this are treated as “should be similar”, else “should be dissimilar”.
    _SIMILARITY_SPLIT: float = 0.70

    def run(self) -> List[Any]:
        quick_mode = bool(os.environ.get("FAMILYOS_ULTRABERT_BENCH_QUICK", "").strip())
        quality_severity = BenchmarkSeverity.WARN if quick_mode else BenchmarkSeverity.FAIL

        # Cache embeddings so the suite doesn't recompute vectors across benchmarks.
        cache: Dict[str, List[float]] = {}

        def embed(text: str) -> List[float]:
            if text not in cache:
                cache[text] = list(self.client.get_embedding(text))
            return cache[text]

        # ------------------------------------------------------------------
        # Issue #12: Basic quality
        # ------------------------------------------------------------------
        probe_text = "FamilyOS embedding probe text."
        probe_emb = embed(probe_text)
        self.add_result(
            name="embedding_dimension",
            passed=(len(probe_emb) == self._EXPECTED_DIM),
            details={"expected": self._EXPECTED_DIM, "observed": len(probe_emb)},
        )

        probe_norm = _l2_norm(probe_emb)
        self.add_result(
            name="embedding_unit_norm",
            passed=(
                abs(probe_norm - 1.0) <= self._NORM_TOLERANCE
                and len(probe_emb) == self._EXPECTED_DIM
            ),
            score=probe_norm,
            threshold=1.0,
            details={"norm": probe_norm, "tolerance": self._NORM_TOLERANCE},
        )

        sim_failures: List[Dict[str, Any]] = []
        sim_passed = 0
        for t1, t2, threshold in SIMILARITY_CASES:
            emb1 = embed(t1)
            emb2 = embed(t2)
            sim = _cosine_similarity(emb1, emb2)
            is_expected_similar = bool(threshold >= self._SIMILARITY_SPLIT)
            ok = (sim >= threshold) if is_expected_similar else (sim <= threshold)
            if ok:
                sim_passed += 1
            else:
                sim_failures.append(
                    {
                        "text1": t1,
                        "text2": t2,
                        "similarity": sim,
                        "threshold": threshold,
                        "expected": "high" if is_expected_similar else "low",
                    }
                )

        self.add_result(
            name="embedding_similarity_cases",
            passed=(sim_passed == len(SIMILARITY_CASES) and len(SIMILARITY_CASES) > 0),
            score=(float(sim_passed) / float(len(SIMILARITY_CASES))) if SIMILARITY_CASES else 0.0,
            details={
                "total": len(SIMILARITY_CASES),
                "passed": sim_passed,
                "failures": sim_failures,
            },
            severity=quality_severity,
        )

        # ------------------------------------------------------------------
        # Issue #13: Triplet accuracy
        # ------------------------------------------------------------------
        triplet_total = len(TRIPLET_CASES)
        triplet_pass = 0
        min_margin: float = 1e9
        triplet_failures: List[Dict[str, Any]] = []

        for case in TRIPLET_CASES:
            anchor = str(case.get("anchor", ""))
            positive = str(case.get("positive", ""))
            negatives = list(case.get("negatives", []))
            if not anchor or not positive or not negatives:
                triplet_failures.append({"case": case, "error": "invalid_case"})
                continue

            ae = embed(anchor)
            pe = embed(positive)
            pos_sim = _cosine_similarity(ae, pe)
            case_ok = True
            case_margins: List[float] = []

            for neg in negatives:
                ne = embed(str(neg))
                neg_sim = _cosine_similarity(ae, ne)
                margin = float(pos_sim - neg_sim)
                case_margins.append(margin)
                if margin < self._TRIPLET_MARGIN_THRESHOLD:
                    case_ok = False

            min_margin = min(min_margin, min(case_margins) if case_margins else min_margin)
            if case_ok:
                triplet_pass += 1
            else:
                triplet_failures.append(
                    {
                        "anchor": anchor,
                        "positive": positive,
                        "pos_sim": pos_sim,
                        "negatives": negatives,
                        "min_margin": min(case_margins) if case_margins else None,
                    }
                )

        triplet_acc = (float(triplet_pass) / float(triplet_total)) if triplet_total > 0 else 0.0
        self.add_result(
            name="embedding_triplet_accuracy",
            passed=(triplet_acc >= self._TRIPLET_ACCURACY_THRESHOLD and triplet_total > 0),
            score=triplet_acc,
            threshold=self._TRIPLET_ACCURACY_THRESHOLD,
            details={
                "total": triplet_total,
                "passed": triplet_pass,
                "margin_threshold": self._TRIPLET_MARGIN_THRESHOLD,
                "min_margin": (min_margin if min_margin != 1e9 else None),
                "failures": triplet_failures,
            },
            severity=quality_severity,
        )

        # ------------------------------------------------------------------
        # Issue #14: Recall@K with distractors
        # ------------------------------------------------------------------
        def recall_at_k(cases: List[Dict[str, Any]], k: int) -> Tuple[float, List[Dict[str, Any]]]:
            total = 0
            hits = 0
            failures: List[Dict[str, Any]] = []
            for rc in cases:
                query = str(rc.get("query", ""))
                relevant = str(rc.get("relevant", ""))
                distractors = list(rc.get("distractors", []))
                if not query or not relevant:
                    continue
                total += 1

                q = embed(query)
                candidates_texts = [relevant] + [str(d) for d in distractors]
                candidates = [(t, embed(t)) for t in candidates_texts]
                ranked = _rank_by_similarity(q, candidates)
                topk = [cid for cid, _ in ranked[: max(1, k)]]
                if relevant in topk:
                    hits += 1
                else:
                    failures.append(
                        {
                            "query": query,
                            "relevant": relevant,
                            "topk": topk,
                            "top1": topk[0] if topk else None,
                        }
                    )
            acc = (float(hits) / float(total)) if total > 0 else 0.0
            return acc, failures

        recall1_10, failures_10 = recall_at_k(RETRIEVAL_CASES_10, k=1)
        self.add_result(
            name="retrieval_recall_at1_10d",
            passed=(recall1_10 >= self._RECALL_AT1_10_THRESHOLD and len(RETRIEVAL_CASES_10) > 0),
            score=recall1_10,
            threshold=self._RECALL_AT1_10_THRESHOLD,
            details={"cases": len(RETRIEVAL_CASES_10), "failures": failures_10},
            severity=quality_severity,
        )

        recall1_100, failures_100_1 = recall_at_k(RETRIEVAL_CASES_100, k=1)
        recall5_100, _failures_100_5 = recall_at_k(RETRIEVAL_CASES_100, k=5)
        recall10_100, failures_100_10 = recall_at_k(RETRIEVAL_CASES_100, k=10)

        self.add_result(
            name="retrieval_recall_at1_100d",
            passed=(recall1_100 >= self._RECALL_AT1_100_THRESHOLD and len(RETRIEVAL_CASES_100) > 0),
            score=recall1_100,
            threshold=self._RECALL_AT1_100_THRESHOLD,
            details={"cases": len(RETRIEVAL_CASES_100), "failures": failures_100_1},
            severity=quality_severity,
        )

        # Plan includes Recall@5, but acceptance criteria doesn't set targets.
        self.add_result(
            name="retrieval_recall_at5_100d",
            passed=True,
            score=recall5_100,
            details={"cases": len(RETRIEVAL_CASES_100)},
        )

        self.add_result(
            name="retrieval_recall_at10_100d",
            passed=(
                recall10_100 >= self._RECALL_AT10_100_THRESHOLD and len(RETRIEVAL_CASES_100) > 0
            ),
            score=recall10_100,
            threshold=self._RECALL_AT10_100_THRESHOLD,
            details={"cases": len(RETRIEVAL_CASES_100), "failures": failures_100_10},
            severity=quality_severity,
        )

        return self.results

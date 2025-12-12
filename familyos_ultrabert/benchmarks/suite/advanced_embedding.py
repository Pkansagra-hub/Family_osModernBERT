"""Advanced embedding metrics benchmark suite.

Implements:
- Issue #30: AdvancedEmbeddingSuite

This suite evaluates graded relevance ranking quality using:
- Precision@K
- MRR (Mean Reciprocal Rank)
- nDCG@K (Normalized Discounted Cumulative Gain)

Constraint: standard library only.

Note:
- This suite is primarily for regression tracking and relative comparisons.
- It does not enforce aggressive thresholds by default to avoid environment
  brittleness across backends/hardware.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import ADVANCED_RANKING_CASES
from familyos_ultrabert.benchmarks.suite import register_suite


def _l2_norm(vec: List[float]) -> float:
	return math.sqrt(sum(float(x) * float(x) for x in vec))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
	norm_a = _l2_norm(a)
	norm_b = _l2_norm(b)
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	dot = sum(float(x) * float(y) for x, y in zip(a, b))
	return float(dot / (norm_a * norm_b))


def _dcg(relevances: List[int], k: int) -> float:
	"""Discounted cumulative gain with exponential gain."""
	k = max(0, int(k))
	total = 0.0
	for i, rel in enumerate(relevances[:k]):
		gain = (2.0 ** float(rel)) - 1.0
		denom = math.log2(float(i) + 2.0)
		total += float(gain / denom)
	return float(total)


def _ndcg(relevances: List[int], k: int) -> float:
	ideal = sorted(relevances, reverse=True)
	idcg = _dcg(ideal, k)
	if idcg == 0.0:
		return 0.0
	return float(_dcg(relevances, k) / idcg)


def _precision_at_k(binary_relevances: List[int], k: int) -> float:
	k = max(1, int(k))
	cut = binary_relevances[:k]
	return float(sum(int(x) for x in cut) / float(k))


def _reciprocal_rank(binary_relevances: List[int]) -> float:
	for idx, rel in enumerate(binary_relevances):
		if int(rel) > 0:
			return float(1.0 / float(idx + 1))
	return 0.0


@register_suite
class AdvancedEmbeddingSuite(BenchmarkSuite):
	"""Suite for graded relevance ranking metrics."""

	name: str = "advanced_embedding"
	description: str = "Graded relevance ranking metrics (MRR, nDCG, Precision@K)"

	_AT_K: Tuple[int, ...] = (1, 3, 5)
	_NDCG_K: int = 5

	def run(self) -> List[Any]:
		cache: Dict[str, List[float]] = {}

		def embed(text: str) -> List[float]:
			if text not in cache:
				cache[text] = list(self.client.get_embedding(text))
			return cache[text]

		per_query: List[Dict[str, Any]] = []
		mrrs: List[float] = []
		ndcgs: List[float] = []
		precisions: Dict[int, List[float]] = {k: [] for k in self._AT_K}
		errors: List[Dict[str, Any]] = []

		for case in ADVANCED_RANKING_CASES:
			query = str(case.get("query", ""))
			docs = list(case.get("documents", []))
			if not query or not docs:
				continue

			try:
				q_emb = embed(query)
				scored: List[Tuple[str, str, int, float]] = []
				for d in docs:
					did = str(d.get("id", ""))
					dtext = str(d.get("text", ""))
					rel = int(d.get("relevance", 0))
					if not did or not dtext:
						continue
					sim = _cosine_similarity(q_emb, embed(dtext))
					scored.append((did, dtext, rel, sim))

				scored_sorted = sorted(scored, key=lambda x: x[3], reverse=True)
				rels = [int(rel) for _did, _text, rel, _sim in scored_sorted]
				binary = [1 if int(r) > 0 else 0 for r in rels]

				query_mrr = _reciprocal_rank(binary)
				query_ndcg = _ndcg(rels, self._NDCG_K)
				mrrs.append(query_mrr)
				ndcgs.append(query_ndcg)

				p_at: Dict[str, float] = {}
				for k in self._AT_K:
					p = _precision_at_k(binary, k)
					precisions[k].append(p)
					p_at[str(k)] = p

				per_query.append(
					{
						"query": query,
						"mrr": query_mrr,
						"ndcg_at": {str(self._NDCG_K): query_ndcg},
						"precision_at": p_at,
						"ranked": [
							{"id": did, "relevance": rel, "similarity": sim}
							for did, _text, rel, sim in scored_sorted
						],
					}
				)
			except Exception as exc:  # noqa: BLE001
				errors.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})

		def avg(xs: List[float]) -> float:
			return float(sum(xs) / float(len(xs))) if xs else 0.0

		mrr = avg(mrrs)
		ndcg = avg(ndcgs)
		p_scores: Dict[int, float] = {k: avg(vs) for k, vs in precisions.items()}

		self.add_result(
			name="advanced_embedding_metrics_computed",
			passed=(len(errors) == 0 and len(per_query) > 0),
			details={"total_queries": len(per_query), "errors": errors},
		)

		# Informational: record metrics for trending.
		self.add_result(
			name="advanced_embedding_mrr",
			passed=True,
			score=mrr,
			details={"per_query": per_query},
		)
		self.add_result(
			name=f"advanced_embedding_ndcg_at{self._NDCG_K}",
			passed=True,
			score=ndcg,
			details={"k": self._NDCG_K},
		)
		for k in self._AT_K:
			self.add_result(
				name=f"advanced_embedding_precision_at{k}",
				passed=True,
				score=float(p_scores.get(k, 0.0)),
				details={"k": k},
			)

		# Gating: metrics should be in [0, 1].
		all_metrics: List[float] = [mrr, ndcg] + [float(p_scores.get(k, 0.0)) for k in self._AT_K]
		bad = [x for x in all_metrics if (x < 0.0 or x > 1.0)]
		self.add_result(
			name="advanced_embedding_metric_ranges_valid",
			passed=(len(bad) == 0 and len(per_query) > 0),
			details={"bad": bad, "metrics": {"mrr": mrr, "ndcg": ndcg, "precision": p_scores}},
		)

		return self.results

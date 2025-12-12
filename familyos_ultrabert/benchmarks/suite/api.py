"""API benchmark suite (Client methods, backends, convenience).

Implements:
- Issue #18: Client method surface tests
- Issue #19: Backend consistency (PyTorch vs ONNX)

Constraint: standard library only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.types import BenchmarkSeverity
from familyos_ultrabert.benchmarks.data.test_cases import CLIENT_METHODS
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


def _sentiment_valence(label: str) -> int:
	"""Map a sentiment label to a coarse valence.

	This is used for cross-backend consistency checks. Exact label parity can
	be brittle across backends/quantization, but contradictory valence is still
	a meaningful regression signal.

	Args:
		label: Sentiment label.

	Returns:
		-1 for negative, 0 for neutral/unknown, 1 for positive.
	"""
	key = str(label).strip().lower()
	if key in ("very_positive", "positive"):
		return 1
	if key in ("very_negative", "negative"):
		return -1
	return 0


@register_suite
class APISuite(BenchmarkSuite):
	"""API correctness and backend consistency suite."""

	name: str = "api"
	description: str = "Client methods, return types, and backend consistency"

	# Cross-backend embeddings will not be bit-identical. Keep this threshold
	# conservative to reduce false failures caused by quantization/runtime drift.
	_EMBEDDING_SIMILARITY_THRESHOLD: float = 0.95

	def run(self) -> List["BenchmarkResult"]:
		text = "Mom picked up the kids from school today."
		text2 = "Mother collected the children after classes."

		# ------------------------------------------------------------------
		# Issue #18: Client methods callable and return expected types
		# ------------------------------------------------------------------
		missing = [name for name in CLIENT_METHODS if not hasattr(self.client, name)]
		self.add_result(
			name="client_methods_present",
			passed=(len(missing) == 0),
			details={"missing": missing},
		)

		method_failures: List[Dict[str, Any]] = []
		try:
			result = self.client.analyze(text, capabilities=["sentiment", "emotions", "safety_familyos"])
			for attr_name in ("sentiment", "emotions", "safety"):
				_ = getattr(result, attr_name)
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "analyze", "error": f"{type(exc).__name__}: {exc}"})

		try:
			pred = self.client.get_sentiment(text)
			if not isinstance(pred, str):
				method_failures.append({"method": "get_sentiment", "error": f"expected str, got {type(pred).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_sentiment", "error": f"{type(exc).__name__}: {exc}"})

		try:
			emos = self.client.get_emotions(text)
			if not isinstance(emos, list):
				method_failures.append({"method": "get_emotions", "error": f"expected list, got {type(emos).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_emotions", "error": f"{type(exc).__name__}: {exc}"})

		try:
			band = self.client.get_safety(text)
			if not isinstance(band, str):
				method_failures.append({"method": "get_safety", "error": f"expected str, got {type(band).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_safety", "error": f"{type(exc).__name__}: {exc}"})

		try:
			intent = self.client.get_intent(text)
			if not isinstance(intent, str):
				method_failures.append({"method": "get_intent", "error": f"expected str, got {type(intent).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_intent", "error": f"{type(exc).__name__}: {exc}"})

		try:
			ingress = self.client.get_ingress(text)
			if not isinstance(ingress, str):
				method_failures.append({"method": "get_ingress", "error": f"expected str, got {type(ingress).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_ingress", "error": f"{type(exc).__name__}: {exc}"})

		try:
			entities = self.client.get_entities(text)
			if not isinstance(entities, list):
				method_failures.append({"method": "get_entities", "error": f"expected list, got {type(entities).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_entities", "error": f"{type(exc).__name__}: {exc}"})

		try:
			temporal = self.client.get_temporal(text)
			if not isinstance(temporal, list):
				method_failures.append({"method": "get_temporal", "error": f"expected list, got {type(temporal).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_temporal", "error": f"{type(exc).__name__}: {exc}"})

		try:
			emb = self.client.get_embedding(text)
			if not isinstance(emb, list):
				method_failures.append({"method": "get_embedding", "error": f"expected list, got {type(emb).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_embedding", "error": f"{type(exc).__name__}: {exc}"})

		# Boolean convenience methods
		for method_name in ("is_safe", "is_crisis", "needs_attention", "is_positive", "is_negative"):
			try:
				val = getattr(self.client, method_name)(text)
				if not isinstance(val, bool):
					method_failures.append({"method": method_name, "error": f"expected bool, got {type(val).__name__}"})
			except Exception as exc:  # noqa: BLE001
				method_failures.append({"method": method_name, "error": f"{type(exc).__name__}: {exc}"})

		# Utilities requiring multiple inputs
		try:
			sim = self.client.similarity(text, text2)
			if not isinstance(sim, (int, float)):
				method_failures.append({"method": "similarity", "error": f"expected float, got {type(sim).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "similarity", "error": f"{type(exc).__name__}: {exc}"})

		try:
			res = self.client.find_similar(text, [text2, "Unrelated topic"], top_k=2)
			if not isinstance(res, list):
				method_failures.append({"method": "find_similar", "error": f"expected list, got {type(res).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "find_similar", "error": f"{type(exc).__name__}: {exc}"})

		try:
			batch_embs = self.client.embed_batch([text, text2])
			if not isinstance(batch_embs, list) or (batch_embs and not isinstance(batch_embs[0], list)):
				method_failures.append({"method": "embed_batch", "error": "expected List[List[float]]"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "embed_batch", "error": f"{type(exc).__name__}: {exc}"})

		try:
			batch_preds = self.client.classify_batch([text, text2], capability="sentiment")
			if not isinstance(batch_preds, list):
				method_failures.append({"method": "classify_batch", "error": f"expected list, got {type(batch_preds).__name__}"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "classify_batch", "error": f"{type(exc).__name__}: {exc}"})

		try:
			hc = self.client.health_check()
			if not isinstance(hc, dict) or "status" not in hc:
				method_failures.append({"method": "health_check", "error": "expected dict with status"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "health_check", "error": f"{type(exc).__name__}: {exc}"})

		try:
			stats = self.client.get_stats()
			if not isinstance(stats, dict) or "total_calls" not in stats:
				method_failures.append({"method": "get_stats", "error": "expected dict with total_calls"})
		except Exception as exc:  # noqa: BLE001
			method_failures.append({"method": "get_stats", "error": f"{type(exc).__name__}: {exc}"})

		self.add_result(
			name="client_methods_callable",
			passed=(len(method_failures) == 0),
			details={"failures": method_failures},
		)

		# ------------------------------------------------------------------
		# Issue #19: Backend consistency
		# ------------------------------------------------------------------
		try:
			from familyos_ultrabert import Client
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="backend_consistency_setup", error=f"{type(exc).__name__}: {exc}")
			return self.results

		pytorch_client: Optional[Any] = None
		onnx_client: Optional[Any] = None
		try:
			pytorch_client = Client(backend="pytorch", warmup=False, verbose=False)
			self.add_result(name="backend_pytorch_available", passed=True)
		except Exception as exc:  # noqa: BLE001
			self.add_skipped(name="backend_pytorch_available", reason=f"{type(exc).__name__}: {exc}")
		try:
			onnx_client = Client(backend="onnx", warmup=False, verbose=False)
			self.add_result(name="backend_onnx_available", passed=True)
		except Exception as exc:  # noqa: BLE001
			self.add_skipped(name="backend_onnx_available", reason=f"{type(exc).__name__}: {exc}")

		if pytorch_client is None or onnx_client is None:
			self.add_skipped(
				name="backend_consistency",
				reason="both backends must be available",
				details={"pytorch": pytorch_client is not None, "onnx": onnx_client is not None},
			)
			return self.results

		self.add_result(name="backend_consistency_ready", passed=True)

		consistency_failures: List[Dict[str, Any]] = []
		try:
			sent_pt = str(pytorch_client.get_sentiment(text))
			sent_ox = str(onnx_client.get_sentiment(text))
			val_pt = _sentiment_valence(sent_pt)
			val_ox = _sentiment_valence(sent_ox)
			# Fail only on contradictory valence (positive vs negative).
			if abs(val_pt - val_ox) >= 2:
				consistency_failures.append(
					{
						"field": "sentiment",
						"pytorch": sent_pt,
						"onnx": sent_ox,
						"pytorch_valence": val_pt,
						"onnx_valence": val_ox,
					}
				)
		except Exception as exc:  # noqa: BLE001
			consistency_failures.append({"field": "sentiment", "error": f"{type(exc).__name__}: {exc}"})

		try:
			saf_pt = str(pytorch_client.get_safety(text))
			saf_ox = str(onnx_client.get_safety(text))
			if saf_pt != saf_ox:
				consistency_failures.append({"field": "safety", "pytorch": saf_pt, "onnx": saf_ox})
		except Exception as exc:  # noqa: BLE001
			consistency_failures.append({"field": "safety", "error": f"{type(exc).__name__}: {exc}"})

		try:
			emb_pt = list(pytorch_client.get_embedding(text))
			emb_ox = list(onnx_client.get_embedding(text))
			sim = _cosine_similarity(emb_pt, emb_ox)
			self.add_result(
				name="backend_embedding_similarity",
				passed=(sim >= self._EMBEDDING_SIMILARITY_THRESHOLD),
				severity=BenchmarkSeverity.WARN,
				score=sim,
				threshold=self._EMBEDDING_SIMILARITY_THRESHOLD,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="backend_embedding_similarity", error=f"{type(exc).__name__}: {exc}")

		self.add_result(
			name="backend_consistency_labels",
			passed=(len(consistency_failures) == 0),
			severity=BenchmarkSeverity.WARN,
			details={"failures": consistency_failures},
		)

		return self.results

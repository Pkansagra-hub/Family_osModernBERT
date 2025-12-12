"""Semantic complexity benchmark suite.

Implements:
- Issue #27: SemanticComplexitySuite

Goal: cover hard-to-parse language patterns that commonly trigger brittleness.
This suite is intentionally *no hard accuracy gating*; it validates:
- no crashes
- output types are valid and stable

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import SEMANTIC_COMPLEXITY_CASES
from familyos_ultrabert.benchmarks.suite import register_suite


@register_suite
class SemanticComplexitySuite(BenchmarkSuite):
	"""Suite for semantic complexity and tricky language."""

	name: str = "semantic_complexity"
	description: str = "Sarcasm, negation chains, hypotheticals, code-switching, garden-path sentences"

	_VALID_SAFETY_BANDS = {"GREEN", "AMBER", "RED", "CRISIS"}

	def run(self) -> List[Any]:
		capabilities = ["sentiment", "safety_familyos", "emotions", "intent"]

		failures: List[Dict[str, Any]] = []
		invalid_types: List[Dict[str, Any]] = []
		invalid_safety: List[Dict[str, Any]] = []

		for label, text in SEMANTIC_COMPLEXITY_CASES:
			try:
				res = self.client.analyze(text, capabilities=capabilities)

				sentiment = getattr(res, "sentiment", None)
				safety = getattr(res, "safety", None)
				emotions = getattr(res, "emotions", None)
				intent = getattr(res, "intent", None)

				if not isinstance(sentiment, str):
					invalid_types.append({"case": label, "field": "sentiment", "type": type(sentiment).__name__})
				if not isinstance(safety, str):
					invalid_types.append({"case": label, "field": "safety", "type": type(safety).__name__})
				if emotions is not None and not isinstance(emotions, list):
					invalid_types.append({"case": label, "field": "emotions", "type": type(emotions).__name__})
				if not isinstance(intent, str):
					invalid_types.append({"case": label, "field": "intent", "type": type(intent).__name__})

				if isinstance(safety, str) and safety and safety not in self._VALID_SAFETY_BANDS:
					invalid_safety.append({"case": label, "safety": safety})
			except Exception as exc:  # noqa: BLE001
				failures.append({"case": label, "text": text, "error": f"{type(exc).__name__}: {exc}"})

		self.add_result(
			name="semantic_complexity_no_crash",
			passed=(len(failures) == 0 and len(SEMANTIC_COMPLEXITY_CASES) > 0),
			details={"total": len(SEMANTIC_COMPLEXITY_CASES), "failures": failures},
		)
		self.add_result(
			name="semantic_complexity_output_types_valid",
			passed=(len(invalid_types) == 0 and len(SEMANTIC_COMPLEXITY_CASES) > 0),
			details={"invalid": invalid_types},
		)
		self.add_result(
			name="semantic_complexity_safety_band_valid",
			passed=(len(invalid_safety) == 0 and len(SEMANTIC_COMPLEXITY_CASES) > 0),
			details={"invalid": invalid_safety},
		)

		return self.results

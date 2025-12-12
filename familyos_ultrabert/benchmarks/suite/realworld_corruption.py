"""Real-world corruption benchmark suite.

Implements:
- Issue #29: RealWorldCorruptionSuite

Targets typical ingestion corruption patterns:
- encoding artifacts (mojibake)
- control characters
- broken markup
- mixed directionality
- truncated strings

Primary acceptance: no crashes; output fields should remain well-typed.

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import REALWORLD_CORRUPTION_CASES
from familyos_ultrabert.benchmarks.suite import register_suite


@register_suite
class RealWorldCorruptionSuite(BenchmarkSuite):
	"""Suite for corrupted/dirty input robustness."""

	name: str = "realworld_corruption"
	description: str = "Mojibake, control chars, truncation, broken markup, bidi"

	_VALID_SAFETY_BANDS = {"GREEN", "AMBER", "RED", "CRISIS"}

	def run(self) -> List[Any]:
		capabilities = ["safety_familyos", "sentiment", "emotions", "intent"]

		failures: List[Dict[str, Any]] = []
		invalid_safety: List[Dict[str, Any]] = []
		invalid_types: List[Dict[str, Any]] = []

		for label, text in REALWORLD_CORRUPTION_CASES:
			try:
				res = self.client.analyze(text, capabilities=capabilities)
				safety = getattr(res, "safety", None)
				sentiment = getattr(res, "sentiment", None)
				emotions = getattr(res, "emotions", None)
				intent = getattr(res, "intent", None)

				if not isinstance(safety, str):
					invalid_types.append({"case": label, "field": "safety", "type": type(safety).__name__})
				elif safety and safety not in self._VALID_SAFETY_BANDS:
					invalid_safety.append({"case": label, "safety": safety})

				if not isinstance(sentiment, str):
					invalid_types.append({"case": label, "field": "sentiment", "type": type(sentiment).__name__})
				if emotions is not None and not isinstance(emotions, list):
					invalid_types.append({"case": label, "field": "emotions", "type": type(emotions).__name__})
				if not isinstance(intent, str):
					invalid_types.append({"case": label, "field": "intent", "type": type(intent).__name__})
			except Exception as exc:  # noqa: BLE001
				failures.append({"case": label, "text": text, "error": f"{type(exc).__name__}: {exc}"})

		self.add_result(
			name="realworld_corruption_no_crash",
			passed=(len(failures) == 0 and len(REALWORLD_CORRUPTION_CASES) > 0),
			details={"total": len(REALWORLD_CORRUPTION_CASES), "failures": failures},
		)
		self.add_result(
			name="realworld_corruption_safety_band_valid",
			passed=(len(invalid_safety) == 0 and len(REALWORLD_CORRUPTION_CASES) > 0),
			details={"invalid": invalid_safety},
		)
		self.add_result(
			name="realworld_corruption_output_types_valid",
			passed=(len(invalid_types) == 0 and len(REALWORLD_CORRUPTION_CASES) > 0),
			details={"invalid": invalid_types},
		)

		return self.results

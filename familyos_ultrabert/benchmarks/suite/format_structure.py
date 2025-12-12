"""Format/structure robustness benchmark suite.

Implements:
- Issue #28: FormatStructureSuite

This suite targets structured inputs commonly seen in production telemetry:
JSON/XML/YAML, code blocks, markdown, email headers, HTML.

Primary acceptance: no crashes and safety band remains valid.

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import FORMAT_STRUCTURE_CASES
from familyos_ultrabert.benchmarks.suite import register_suite


@register_suite
class FormatStructureSuite(BenchmarkSuite):
	"""Suite for format-heavy, structured text robustness."""

	name: str = "format_structure"
	description: str = "Embedded JSON/XML/YAML, Markdown, code blocks, email headers, HTML"

	_VALID_SAFETY_BANDS = {"GREEN", "AMBER", "RED", "CRISIS"}

	def run(self) -> List[Any]:
		# Focus on safety availability + general stability.
		capabilities = ["safety_familyos", "sentiment", "emotions", "intent"]

		failures: List[Dict[str, Any]] = []
		invalid_safety: List[Dict[str, Any]] = []
		invalid_types: List[Dict[str, Any]] = []

		for label, text in FORMAT_STRUCTURE_CASES:
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
			name="format_structure_no_crash",
			passed=(len(failures) == 0 and len(FORMAT_STRUCTURE_CASES) > 0),
			details={"total": len(FORMAT_STRUCTURE_CASES), "failures": failures},
		)
		self.add_result(
			name="format_structure_safety_band_valid",
			passed=(len(invalid_safety) == 0 and len(FORMAT_STRUCTURE_CASES) > 0),
			details={"invalid": invalid_safety},
		)
		self.add_result(
			name="format_structure_output_types_valid",
			passed=(len(invalid_types) == 0 and len(FORMAT_STRUCTURE_CASES) > 0),
			details={"invalid": invalid_types},
		)

		return self.results

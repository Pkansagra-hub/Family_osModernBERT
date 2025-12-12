"""Robustness benchmark suite (edge cases, unicode, adversarial).

Implements:
- Issue #15: Edge-case inputs should not crash
- Issue #16: Unicode handling should be stable (including normalization stability)
- Issue #17: Adversarial inputs should not crash and should return valid structures

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import (
	ADVERSARIAL_CASES,
	EDGE_CASES,
	EXTREME_UNICODE_CASES,
	UNICODE_CASES,
)
from familyos_ultrabert.benchmarks.suite import register_suite


@register_suite
class RobustnessSuite(BenchmarkSuite):
	"""Robustness and edge-case validation."""

	name: str = "robustness"
	description: str = "Edge cases, Unicode normalization, and input stability"

	# Threshold is generous to accommodate GPU variance where baseline is extremely fast
	_LATENCY_MULTIPLIER_THRESHOLD: float = 25.0
	_EDGE_LATENCY_MAX_WORDS: int = 50
	_BASELINE_RUNS: int = 5
	_VALID_SAFETY_BANDS = {"GREEN", "AMBER", "RED", "CRISIS"}

	def run(self) -> List[Any]:
		capabilities = ["sentiment", "safety_familyos"]
		baseline_text = "Mom picked up the kids from school today."

		baseline_latencies: List[float] = []
		for _ in range(self._BASELINE_RUNS):
			res = self.client.analyze(baseline_text, capabilities=capabilities)
			baseline_latencies.append(float(getattr(res, "latency_ms", 0.0)))
		baseline_latencies.sort()
		baseline = baseline_latencies[len(baseline_latencies) // 2] if baseline_latencies else 0.0

		# ------------------------------------------------------------------
		# Issue #15: Edge cases
		# ------------------------------------------------------------------
		edge_failures: List[Dict[str, Any]] = []
		edge_latency_ratios: List[float] = []
		edge_latency_included: List[Dict[str, Any]] = []
		edge_latency_excluded: List[Dict[str, Any]] = []

		for label, text in EDGE_CASES:
			try:
				result = self.client.analyze(text, capabilities=capabilities)
				_ = str(getattr(result, "sentiment", ""))
				_ = str(getattr(result, "safety", ""))
				latency = float(getattr(result, "latency_ms", 0.0))
				ratio = (latency / baseline) if baseline > 0.0 else 1.0
				words = len(str(text).split())
				payload = {"case": label, "words": words, "latency_ms": latency, "ratio": ratio}
				# This check is intended to catch pathological slowdowns on short, cheap inputs.
				# Very long inputs are covered by the latency length-scaling suite.
				if words <= self._EDGE_LATENCY_MAX_WORDS:
					edge_latency_ratios.append(ratio)
					edge_latency_included.append(payload)
				else:
					edge_latency_excluded.append(payload)
			except Exception as exc:  # noqa: BLE001
				edge_failures.append(
					{"case": label, "text": text, "error": f"{type(exc).__name__}: {exc}"}
				)

		max_ratio = max(edge_latency_ratios) if edge_latency_ratios else 0.0
		self.add_result(
			name="edge_cases_no_crash",
			passed=(len(edge_failures) == 0 and len(EDGE_CASES) > 0),
			details={"total": len(EDGE_CASES), "failures": edge_failures},
		)
		self.add_result(
			name="edge_cases_latency_multiplier_max",
			passed=(max_ratio <= self._LATENCY_MULTIPLIER_THRESHOLD and len(edge_latency_ratios) > 0),
			score=max_ratio,
			threshold=self._LATENCY_MULTIPLIER_THRESHOLD,
			details={
				"baseline_ms": baseline,
				"max_words": self._EDGE_LATENCY_MAX_WORDS,
				"ratios": edge_latency_ratios,
				"included": edge_latency_included,
				"excluded": edge_latency_excluded,
			},
		)

		# ------------------------------------------------------------------
		# Issue #16: Unicode handling
		# ------------------------------------------------------------------
		unicode_failures: List[Dict[str, Any]] = []
		for label, text in UNICODE_CASES:
			try:
				result = self.client.analyze(text, capabilities=capabilities)
				_ = str(getattr(result, "sentiment", ""))
				_ = str(getattr(result, "safety", ""))
			except Exception as exc:  # noqa: BLE001
				unicode_failures.append(
					{"case": label, "text": text, "error": f"{type(exc).__name__}: {exc}"}
				)

		self.add_result(
			name="unicode_cases_no_crash",
			passed=(len(unicode_failures) == 0 and len(UNICODE_CASES) > 0),
			details={"total": len(UNICODE_CASES), "failures": unicode_failures},
		)

		# Extreme Unicode cases (ported from ultimate_stress_test style inputs): report crash-only.
		extreme_unicode_failures: List[Dict[str, Any]] = []
		for label, text in EXTREME_UNICODE_CASES:
			try:
				result = self.client.analyze(text, capabilities=capabilities)
				_ = str(getattr(result, "sentiment", ""))
				safety = str(getattr(result, "safety", ""))
				if safety and safety not in self._VALID_SAFETY_BANDS:
					extreme_unicode_failures.append(
						{
							"case": label,
							"text": text,
							"error": f"invalid_safety_band: {safety}",
						}
					)
			except Exception as exc:  # noqa: BLE001
				extreme_unicode_failures.append(
					{"case": label, "text": text, "error": f"{type(exc).__name__}: {exc}"}
				)

		self.add_result(
			name="extreme_unicode_cases_no_crash",
			passed=(len(extreme_unicode_failures) == 0 and len(EXTREME_UNICODE_CASES) > 0),
			details={"total": len(EXTREME_UNICODE_CASES), "failures": extreme_unicode_failures},
		)

		# Unicode normalization stability check (safety-critical): curly vs straight apostrophe.
		norm_stability_failed = False
		try:
			curly = "I\u2019m going to help mom"
			straight = "I'm going to help mom"
			r1 = self.client.analyze(curly, capabilities=capabilities)
			r2 = self.client.analyze(straight, capabilities=capabilities)
			same = (
				str(getattr(r1, "safety", "")) == str(getattr(r2, "safety", ""))
				and str(getattr(r1, "sentiment", "")) == str(getattr(r2, "sentiment", ""))
			)
			self.add_result(
				name="unicode_normalization_stability",
				passed=same,
				details={
					"curly": {"sentiment": getattr(r1, "sentiment", None), "safety": getattr(r1, "safety", None)},
					"straight": {"sentiment": getattr(r2, "sentiment", None), "safety": getattr(r2, "safety", None)},
				},
			)
		except Exception as exc:  # noqa: BLE001
			norm_stability_failed = True
			self.add_error(
				name="unicode_normalization_stability",
				error=f"{type(exc).__name__}: {exc}",
			)

		# ------------------------------------------------------------------
		# Issue #17: Adversarial inputs (injection/jailbreak/code-like strings)
		# ------------------------------------------------------------------
		adversarial_failures: List[Dict[str, Any]] = []
		adversarial_invalid_safety: List[Dict[str, Any]] = []

		for text, label in ADVERSARIAL_CASES:
			try:
				result = self.client.analyze(text, capabilities=capabilities)
				safety = str(getattr(result, "safety", ""))
				_ = str(getattr(result, "sentiment", ""))
				if safety not in self._VALID_SAFETY_BANDS:
					adversarial_invalid_safety.append({"label": label, "text": text, "safety": safety})
			except Exception as exc:  # noqa: BLE001
				adversarial_failures.append(
					{"label": label, "text": text, "error": f"{type(exc).__name__}: {exc}"}
				)

		self.add_result(
			name="adversarial_no_crash",
			passed=(len(adversarial_failures) == 0 and len(ADVERSARIAL_CASES) > 0),
			details={
				"total": len(ADVERSARIAL_CASES),
				"failures": adversarial_failures,
				"note": (
					"This benchmark validates robustness and output structure for adversarial-looking inputs. "
					"It does not assume generative behavior or attempt to 'prompt-inject' a classifier."
				),
			},
		)
		self.add_result(
			name="adversarial_safety_valid",
			passed=(len(adversarial_invalid_safety) == 0 and len(ADVERSARIAL_CASES) > 0),
			details={
				"invalid": adversarial_invalid_safety,
				"normalization_stability_check_failed": norm_stability_failed,
			},
		)

		return self.results

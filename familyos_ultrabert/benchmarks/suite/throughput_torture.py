"""Throughput torture benchmark suite.

Implements:
- Issue #31: Throughput torture (1000+ calls)

Goal:
- Detect memory leaks, perf cliffs, and stability issues under repeated load.

Notes:
- This suite supports a quick/CI-safe mode via environment variable:
  FAMILYOS_ULTRABERT_BENCH_QUICK=1

Constraint: standard library only.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import DEFAULT_LATENCY_TEXT
from familyos_ultrabert.benchmarks.suite import register_suite


def _truthy_env(value: str) -> bool:
	v = value.strip().lower()
	return v in {"1", "true", "yes", "y", "on"}


@register_suite
class ThroughputTortureSuite(BenchmarkSuite):
	"""Suite for repeated full inference throughput and stability."""

	name: str = "throughput_torture"
	description: str = "Repeated full inference load (quick mode supported)"

	_DEFAULT_WARMUP: int = 10
	_DEFAULT_RUNS: int = 1200
	_QUICK_RUNS: int = 250

	def _runs(self) -> int:
		val = os.getenv("FAMILYOS_ULTRABERT_BENCH_THROUGHPUT_RUNS")
		if val is not None:
			try:
				n = int(val)
				return max(1, n)
			except Exception:  # noqa: BLE001
				return self._DEFAULT_RUNS

		quick_flag = os.getenv("FAMILYOS_ULTRABERT_BENCH_QUICK")
		if quick_flag is not None and _truthy_env(quick_flag):
			return self._QUICK_RUNS
		return self._DEFAULT_RUNS

	def run(self) -> List[Any]:
		text = DEFAULT_LATENCY_TEXT
		runs = int(self._runs())
		warmup = int(self._DEFAULT_WARMUP)

		failures: List[Dict[str, Any]] = []

		def do_call() -> Any:
			return self.client.analyze(text)

		try:
			for _ in range(max(0, warmup)):
				do_call()

			start = time.perf_counter()
			for _ in range(runs):
				do_call()
			total_sec = float(time.perf_counter() - start)
			calls_per_sec = (float(runs) / total_sec) if total_sec > 0 else 0.0

			self.add_result(
				name="throughput_torture_no_crash",
				passed=True,
				details={"runs": runs, "warmup": warmup},
			)
			self.add_result(
				name="throughput_torture_calls_per_sec",
				passed=True,
				score=calls_per_sec,
				details={"runs": runs, "total_sec": total_sec},
			)
			self.add_result(
				name="throughput_torture_total_sec",
				passed=True,
				score=total_sec,
				details={"runs": runs, "calls_per_sec": calls_per_sec},
			)

			self.add_result(
				name="throughput_torture_sanity_nonzero",
				passed=(total_sec > 0.0 and calls_per_sec > 0.0),
				details={"runs": runs, "total_sec": total_sec, "calls_per_sec": calls_per_sec},
			)
		except Exception as exc:  # noqa: BLE001
			failures.append({"error": f"{type(exc).__name__}: {exc}"})
			self.add_result(
				name="throughput_torture_no_crash",
				passed=False,
				details={"runs": runs, "warmup": warmup, "failures": failures},
			)

		return self.results

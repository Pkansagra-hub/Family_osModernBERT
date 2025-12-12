"""Latency & performance benchmark suite.

Implements:
- Issue #5: Per-capability latency
- Issue #6: Text length scaling

Constraint: standard library only.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import (
	CAPABILITIES,
	DEFAULT_LATENCY_TEXT,
	LATENCY_THRESHOLDS,
	LENGTH_TESTS,
	MIXED_LENGTH_WORKLOAD,
	THROUGHPUT_SEQUENTIAL_RUNS,
	THROUGHPUT_WARMUP_RUNS,
)
from familyos_ultrabert.benchmarks.suite import register_suite


def _percentile(sorted_samples: List[float], p: float) -> float:
	"""Compute percentile using linear interpolation.

	Args:
		sorted_samples: Samples sorted ascending.
		p: Percentile in [0, 1].

	Returns:
		Percentile value.
	"""
	if not sorted_samples:
		return 0.0
	if p <= 0.0:
		return float(sorted_samples[0])
	if p >= 1.0:
		return float(sorted_samples[-1])

	n = len(sorted_samples)
	k = (n - 1) * p
	f = int(k)
	c = f + 1 if f + 1 < n else f
	if c == f:
		return float(sorted_samples[f])
	return float(sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f]))


def _measure_latency_samples(
	func: Callable[[], Any],
	*,
	warmup: int,
	runs: int,
) -> List[float]:
	"""Measure latency samples in milliseconds."""
	for _ in range(max(0, warmup)):
		func()

	samples: List[float] = []
	for _ in range(runs):
		start = time.perf_counter()
		func()
		samples.append((time.perf_counter() - start) * 1000.0)
	return samples


def _device_kind(client: Any) -> str:
	"""Infer a coarse device kind for threshold selection."""
	backend = getattr(client, "backend", "unknown")
	if backend == "onnx":
		return "cpu"

	# For pytorch, prefer a CUDA check if torch is available.
	try:
		import torch  # type: ignore

		return "gpu" if bool(torch.cuda.is_available()) else "cpu"
	except Exception:  # noqa: BLE001
		# Conservative default.
		return "gpu" if backend == "pytorch" else "cpu"


def _make_length_text(word_count: int) -> str:
	"""Create a text with approximately the requested number of words."""
	if word_count <= 0:
		return ""
	# Simple, deterministic content.
	words = ["family"] * max(0, word_count - 1)
	return "Mom " + " ".join(words) + "."


def _throughput(
	func: Callable[[], Any],
	*,
	warmup: int,
	runs: int,
) -> Dict[str, float]:
	"""Measure throughput for repeated calls.

	Args:
		func: Callable to execute.
		warmup: Number of warmup calls.
		runs: Number of timed calls.

	Returns:
		Dict containing total_sec and calls_per_sec.
	"""
	for _ in range(max(0, warmup)):
		func()

	start = time.perf_counter()
	for _ in range(runs):
		func()
	total_sec = time.perf_counter() - start
	cps = (float(runs) / total_sec) if total_sec > 0 else 0.0
	return {"total_sec": float(total_sec), "calls_per_sec": float(cps)}


@register_suite
class LatencySuite(BenchmarkSuite):
	"""Latency suite: per-capability and length scaling benchmarks."""

	name: str = "latency"
	description: str = "Latency and scaling benchmarks"

	_DEFAULT_WARMUP: int = 2
	_DEFAULT_RUNS_SINGLE: int = 10
	_DEFAULT_RUNS_FULL: int = 10
	_DEFAULT_RUNS_LENGTH: int = 7

	def _measure(self, func: Callable[[], Any], *, warmup: int, runs: int) -> Dict[str, float]:
		samples = _measure_latency_samples(func, warmup=warmup, runs=runs)
		samples_sorted = sorted(samples)
		return {
			"mean": float(sum(samples) / len(samples)) if samples else 0.0,
			"p50": _percentile(samples_sorted, 0.50),
			"p95": _percentile(samples_sorted, 0.95),
			"p99": _percentile(samples_sorted, 0.99),
			"min": float(samples_sorted[0]) if samples_sorted else 0.0,
			"max": float(samples_sorted[-1]) if samples_sorted else 0.0,
		}

	def _thresholds(self) -> Dict[str, float]:
		kind = _device_kind(self.client)
		thresholds = LATENCY_THRESHOLDS.get(kind, LATENCY_THRESHOLDS["cpu"])
		return {"device_kind": kind, **thresholds}

	def run(self) -> List[Any]:
		"""Run latency benchmarks."""
		text = DEFAULT_LATENCY_TEXT
		thresholds = self._thresholds()
		single_threshold_ms = float(thresholds["single"])
		full_threshold_ms = float(thresholds["full"])

		# Issue #5.1: Single capability latency (all 12 capabilities)
		for cap in CAPABILITIES:
			try:
				stats = self._measure(
					lambda c=cap: self.client.analyze(text, capabilities=[c]),
					warmup=self._DEFAULT_WARMUP,
					runs=self._DEFAULT_RUNS_SINGLE,
				)
				passed = stats["p95"] <= single_threshold_ms
				self.add_result(
					name=f"single_{cap}_p95",
					passed=passed,
					threshold=single_threshold_ms,
					latency_ms=stats["p95"],
					details={
						"device_kind": thresholds["device_kind"],
						"p50_ms": stats["p50"],
						"p95_ms": stats["p95"],
						"p99_ms": stats["p99"],
						"mean_ms": stats["mean"],
						"min_ms": stats["min"],
						"max_ms": stats["max"],
						"runs": self._DEFAULT_RUNS_SINGLE,
						"capability": cap,
					},
				)
			except Exception as exc:  # noqa: BLE001
				self.add_error(name=f"single_{cap}", error=f"{type(exc).__name__}: {exc}")

		# Issue #5.2: Full multi-task latency
		try:
			full_stats = self._measure(
				lambda: self.client.analyze(text),
				warmup=self._DEFAULT_WARMUP,
				runs=self._DEFAULT_RUNS_FULL,
			)
			passed = full_stats["p95"] <= full_threshold_ms
			self.add_result(
				name="full_inference_p95",
				passed=passed,
				threshold=full_threshold_ms,
				latency_ms=full_stats["p95"],
				details={
					"device_kind": thresholds["device_kind"],
					"p50_ms": full_stats["p50"],
					"p95_ms": full_stats["p95"],
					"p99_ms": full_stats["p99"],
					"mean_ms": full_stats["mean"],
					"min_ms": full_stats["min"],
					"max_ms": full_stats["max"],
					"runs": self._DEFAULT_RUNS_FULL,
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="full_inference", error=f"{type(exc).__name__}: {exc}")

		# Issue #5.3: Warmup vs cold start difference (best-effort; informational)
		try:
			from familyos_ultrabert import Client

			cold_client = Client(
				backend=getattr(self.client, "_backend_preference", "auto"),
				warmup=False,
				warmup_rounds=0,
				lazy_load=False,
				verbose=False,
			)
			cold_start_ms = float(cold_client.analyze(text).latency_ms)

			warm_stats = self._measure(lambda: self.client.analyze(text), warmup=0, runs=5)
			warm_p50_ms = float(warm_stats["p50"])
			ratio = (cold_start_ms / warm_p50_ms) if warm_p50_ms > 0 else 0.0

			# Keep this non-gating to avoid false failures across environments.
			self.add_result(
				name="cold_vs_warm_ratio",
				passed=True,
				score=ratio,
				threshold=None,
				latency_ms=None,
				details={
					"cold_start_ms": cold_start_ms,
					"warm_p50_ms": warm_p50_ms,
					"ratio": ratio,
					"note": "Informational only; cold start varies by hardware and cache state.",
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="cold_vs_warm", error=f"{type(exc).__name__}: {exc}")

		# Issue #6: Text length scaling
		length_latencies_p50: Dict[str, float] = {}
		for label, words in LENGTH_TESTS:
			try:
				length_text = _make_length_text(words)
				stats = self._measure(
					lambda t=length_text: self.client.analyze(t),
					warmup=1,
					runs=self._DEFAULT_RUNS_LENGTH,
				)
				length_latencies_p50[label] = float(stats["p50"])

				# Non-gating: record p95 for visibility and ensure no crashes.
				self.add_result(
					name=f"length_{label}_p95",
					passed=True,
					threshold=None,
					latency_ms=stats["p95"],
					details={
						"words": words,
						"p50_ms": stats["p50"],
						"p95_ms": stats["p95"],
						"p99_ms": stats["p99"],
						"mean_ms": stats["mean"],
						"runs": self._DEFAULT_RUNS_LENGTH,
					},
				)
			except Exception as exc:  # noqa: BLE001
				self.add_error(name=f"length_{label}", error=f"{type(exc).__name__}: {exc}")

		# Scaling factor (very_long/short) as requested in plan.
		short_p50 = length_latencies_p50.get("short")
		very_long_p50 = length_latencies_p50.get("very_long")
		if short_p50 and very_long_p50 and short_p50 > 0:
			factor = float(very_long_p50 / short_p50)
			self.add_result(
				name="length_scaling_factor_very_long_over_short",
				passed=True,
				score=factor,
				details={
					"short_p50_ms": short_p50,
					"very_long_p50_ms": very_long_p50,
					"factor": factor,
				},
			)
		else:
			self.add_skipped(
				name="length_scaling_factor_very_long_over_short",
				reason="Missing measurements for short and/or very_long.",
			)

		# Truncation behavior at ~512 tokens: validate extreme input does not crash.
		if "extreme" in length_latencies_p50:
			self.add_result(
				name="long_input_handling",
				passed=True,
				details={
					"note": "Validated that very long input completes without error. Token truncation is model/tokenizer-dependent and not introspected here.",
				},
			)
		else:
			self.add_skipped(
				name="long_input_handling",
				reason="Extreme input measurement failed or was not recorded.",
			)

		# Issue #7: Throughput
		try:
			tp = _throughput(
				lambda: self.client.analyze(text),
				warmup=THROUGHPUT_WARMUP_RUNS,
				runs=THROUGHPUT_SEQUENTIAL_RUNS,
			)
			self.add_result(
				name="throughput_full_sequential",
				passed=True,
				score=tp["calls_per_sec"],
				details={
					"calls": THROUGHPUT_SEQUENTIAL_RUNS,
					"total_sec": tp["total_sec"],
					"inferences_per_sec": tp["calls_per_sec"],
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="throughput_full_sequential", error=f"{type(exc).__name__}: {exc}")

		# Mixed-length workload throughput
		try:
			mixed_texts: List[str] = []
			for label, words, count in MIXED_LENGTH_WORKLOAD:
				mixed_texts.extend([_make_length_text(words)] * int(count))

			# Deterministic order: short->medium->long->very_long
			idx = 0

			def _mixed_call() -> None:
				nonlocal idx
				t = mixed_texts[idx % len(mixed_texts)]
				idx += 1
				self.client.analyze(t)

			tp = _throughput(
				_mixed_call,
				warmup=min(THROUGHPUT_WARMUP_RUNS, max(1, len(mixed_texts) // 20)),
				runs=len(mixed_texts),
			)
			self.add_result(
				name="throughput_mixed_length",
				passed=True,
				score=tp["calls_per_sec"],
				details={
					"calls": len(mixed_texts),
					"total_sec": tp["total_sec"],
					"inferences_per_sec": tp["calls_per_sec"],
					"workload": [
						{"label": label, "words": words, "count": count}
						for (label, words, count) in MIXED_LENGTH_WORKLOAD
					],
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="throughput_mixed_length", error=f"{type(exc).__name__}: {exc}")

		# Embedding-only throughput
		try:
			tp = _throughput(
				lambda: self.client.analyze(text, capabilities=["embedding"]),
				warmup=THROUGHPUT_WARMUP_RUNS,
				runs=THROUGHPUT_SEQUENTIAL_RUNS,
			)
			self.add_result(
				name="throughput_embedding_only",
				passed=True,
				score=tp["calls_per_sec"],
				details={
					"calls": THROUGHPUT_SEQUENTIAL_RUNS,
					"total_sec": tp["total_sec"],
					"embeddings_per_sec": tp["calls_per_sec"],
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="throughput_embedding_only", error=f"{type(exc).__name__}: {exc}")

		return self.results

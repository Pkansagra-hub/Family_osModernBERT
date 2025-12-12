"""Base classes for benchmark suites."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import statistics

from familyos_ultrabert.benchmarks.types import BenchmarkResult, BenchmarkSeverity, BenchmarkStatus


class BenchmarkSuite(ABC):
    """Base class for all benchmark suites."""

    name: str = "base"
    description: str = ""

    def __init__(self, client: Any):
        self._client = client
        self.results: List[BenchmarkResult] = []

    @property
    def client(self) -> Any:
        """The UltraBERT client used by this suite."""
        return self._client

    @abstractmethod
    def run(self) -> List[BenchmarkResult]:
        """Run the suite and return benchmark results."""

    def add_result(
        self,
        *,
        name: str,
        passed: bool,
        score: Optional[float] = None,
        threshold: Optional[float] = None,
        latency_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        severity: BenchmarkSeverity = BenchmarkSeverity.FAIL,
    ) -> BenchmarkResult:
        """Build a BenchmarkResult with standard fields.

        Args:
            name: Benchmark name.
            passed: Whether the check passed.
            score: Optional score.
            threshold: Optional threshold.
            latency_ms: Optional latency in milliseconds.
            details: Optional free-form metadata.
            error: Optional error string.

        Returns:
            The created BenchmarkResult.
        """
        status = BenchmarkStatus.PASS if passed else BenchmarkStatus.FAIL
        result = BenchmarkResult(
            name=name,
            category=self.name,
            status=status,
            severity=severity,
            score=score,
            threshold=threshold,
            latency_ms=latency_ms,
            details=details or {},
            error=error,
        )
        self.results.append(result)
        return result

    def add_skipped(
        self,
        *,
        name: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Create a skipped benchmark result."""
        result = BenchmarkResult(
            name=name,
            category=self.name,
            status=BenchmarkStatus.SKIP,
            severity=BenchmarkSeverity.INFO,
            details={"reason": reason, **(details or {})},
        )
        self.results.append(result)
        return result

    def add_error(
        self,
        *,
        name: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Create an errored benchmark result."""
        result = BenchmarkResult(
            name=name,
            category=self.name,
            status=BenchmarkStatus.ERROR,
            severity=BenchmarkSeverity.FAIL,
            error=error,
            details=details or {},
        )
        self.results.append(result)
        return result

    def add_info(
        self,
        *,
        name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Record an informational (non-gating) benchmark result."""
        result = BenchmarkResult(
            name=name,
            category=self.name,
            status=BenchmarkStatus.PASS,
            severity=BenchmarkSeverity.INFO,
            details=details or {},
        )
        self.results.append(result)
        return result

    def measure_latency(
        self,
        func: Callable[[], Any],
        *,
        warmup: int = 2,
        runs: int = 10,
    ) -> Dict[str, float]:
        """Measure latency statistics for a callable.

        Args:
            func: Callable to measure.
            warmup: Warmup invocations before measurement.
            runs: Number of timed runs.

        Returns:
            A dict containing mean/median/stdev/min/max/p95 (all in milliseconds).

        Raises:
            ValueError: If runs is less than 1.
        """
        if runs < 1:
            raise ValueError("runs must be >= 1")

        for _ in range(max(0, warmup)):
            func()

        samples_ms: List[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            func()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

        stdev = float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
        samples_sorted = sorted(samples_ms)
        # Keep behavior simple and deterministic without external deps.
        # Use the max as p95 for small sample sizes (mirrors plan intent).
        if len(samples_sorted) >= 20:
            p95 = float(samples_sorted[int(len(samples_sorted) * 0.95)])
        else:
            p95 = float(max(samples_sorted))

        return {
            "mean": float(statistics.mean(samples_ms)),
            "median": float(statistics.median(samples_ms)),
            "stdev": stdev,
            "min": float(min(samples_ms)),
            "max": float(max(samples_ms)),
            "p95": p95,
        }

    def measure_latency_extended(
        self,
        func: Callable[[], Any],
        *,
        warmup: int = 2,
        runs: int = 10,
        percentiles: Optional[List[float]] = None,
        synchronize: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, float]:
        """Measure latency statistics for a callable with configurable percentiles.

        This exists to keep percentile math consistent across suites.

        Args:
            func: Callable to measure.
            warmup: Warmup invocations before measurement.
            runs: Number of timed runs.
            percentiles: Optional percentiles in [0, 1]. If not provided,
                defaults to [0.50, 0.95, 0.99].

        Returns:
            A dict containing mean/median/stdev/min/max and percentile keys
            like p50/p95/p99 (all in milliseconds).

        Raises:
            ValueError: If runs is less than 1.
        """
        if runs < 1:
            raise ValueError("runs must be >= 1")

        for _ in range(max(0, warmup)):
            func()

        samples_ms: List[float] = []
        for _ in range(runs):
            if synchronize is not None:
                synchronize()
            start = time.perf_counter()
            func()
            if synchronize is not None:
                synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

        stdev = float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
        samples_sorted = sorted(samples_ms)

        def _percentile(sorted_samples: List[float], p: float) -> float:
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

        ps = percentiles if percentiles is not None else [0.50, 0.95, 0.99]
        out: Dict[str, float] = {
            "mean": float(statistics.mean(samples_ms)),
            "median": float(statistics.median(samples_ms)),
            "stdev": stdev,
            "min": float(min(samples_ms)),
            "max": float(max(samples_ms)),
        }
        for p in ps:
            key = f"p{int(round(float(p) * 100.0))}"
            out[key] = _percentile(samples_sorted, float(p))

        # Keep the legacy p95 key present even if percentiles were customized.
        if "p95" not in out:
            out["p95"] = _percentile(samples_sorted, 0.95)

        return out

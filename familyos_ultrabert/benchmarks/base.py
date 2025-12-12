"""Base classes for benchmark suites."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import statistics

from familyos_ultrabert.benchmarks.types import BenchmarkResult, BenchmarkStatus


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
            error=error,
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

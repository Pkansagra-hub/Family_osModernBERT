"""Milestone 1: Core benchmark infrastructure tests.

Covers Issues:
- #2 BenchmarkResult + SuiteResult structures
- #3 BenchmarkSuite base class helpers
- #4 BenchmarkRunner behavior when no suites are registered
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

import pytest


# Ensure the repository root is importable so the local `familyos_ultrabert/` package
# is used during tests (pytest import modes and plugins can affect sys.path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestBenchmarkTypes:
    """Tests for benchmark result data structures."""

    def test_suite_result_aggregates_counts(self) -> None:
        """Verify SuiteResult computes passed/failed/skipped/errored."""
        from familyos_ultrabert.benchmarks.types import BenchmarkResult, BenchmarkStatus, SuiteResult

        results = [
            BenchmarkResult(name="a", category="x", status=BenchmarkStatus.PASS),
            BenchmarkResult(name="b", category="x", status=BenchmarkStatus.FAIL),
            BenchmarkResult(name="c", category="x", status=BenchmarkStatus.SKIP),
            BenchmarkResult(name="d", category="x", status=BenchmarkStatus.ERROR),
        ]
        suite = SuiteResult(suite_name="latency", results=results, total_time_sec=0.01)

        assert suite.passed == 1
        assert suite.failed == 1
        assert suite.skipped == 1
        assert suite.errored == 1


class TestBenchmarkSuiteBase:
    """Tests for BenchmarkSuite base helpers."""

    def test_add_result_appends(self) -> None:
        """Verify add_result/add_skipped/add_error append into suite.results."""
        from familyos_ultrabert.benchmarks.base import BenchmarkSuite

        class DummySuite(BenchmarkSuite):
            name = "dummy"

            def run(self) -> List[Any]:
                self.add_result(name="ok", passed=True)
                self.add_result(name="bad", passed=False)
                self.add_skipped(name="skip", reason="not implemented")
                self.add_error(name="err", error="boom")
                return self.results

        suite = DummySuite(client=None)
        out = suite.run()

        assert out is suite.results
        assert len(out) == 4
        assert {r.status.value for r in out} == {"pass", "fail", "skip", "error"}
        assert all(r.category == "dummy" for r in out)

    def test_measure_latency_returns_stats_dict(self) -> None:
        """Verify measure_latency returns the stdlib stats dict shape."""
        from familyos_ultrabert.benchmarks.base import BenchmarkSuite

        class DummySuite(BenchmarkSuite):
            name = "dummy"

            def run(self) -> List[Any]:
                return []

        suite = DummySuite(client=None)
        stats = suite.measure_latency(lambda: None, warmup=0, runs=5)

        assert set(stats.keys()) == {"mean", "median", "stdev", "min", "max", "p95"}
        assert all(isinstance(v, float) for v in stats.values())
        assert stats["min"] <= stats["median"] <= stats["max"]


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner behavior."""

    def test_runner_returns_empty_when_no_suites_registered(self) -> None:
        """Runner should short-circuit when no suites exist (Milestone 1)."""
        from familyos_ultrabert.benchmarks.runner import BenchmarkRunner
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        # Ensure we don't accidentally force model load in this test.
        if get_suite_classes():
            pytest.skip("Suites are registered; Milestone 1 empty-suite behavior not applicable.")

        result = BenchmarkRunner(verbose=False).run()

        assert result.summary.total == 0
        assert result.summary.passed == 0
        assert result.summary.failed == 0
        assert "note" in result.metadata

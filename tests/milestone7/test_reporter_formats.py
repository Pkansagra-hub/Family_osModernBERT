"""Milestone 7: reporter output formats.

These tests validate that Reporter can serialize a synthetic BenchmarkRunResult
without any model/runtime dependencies.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from familyos_ultrabert.benchmarks.reporter import Reporter
from familyos_ultrabert.benchmarks.types import BenchmarkResult, BenchmarkRunResult, BenchmarkStatus, BenchmarkSummary, SuiteResult


def _fake_result() -> BenchmarkRunResult:
    summary = BenchmarkSummary(total=2, passed=1, failed=1, skipped=0, errored=0, duration_sec=1.25)
    suite = SuiteResult(
        suite_name="api",
        results=[
            BenchmarkResult(name="client_methods_present", category="api", status=BenchmarkStatus.PASS, score=1.0),
            BenchmarkResult(name="client_methods_callable", category="api", status=BenchmarkStatus.FAIL, error="boom"),
        ],
        total_time_sec=0.5,
    )
    return BenchmarkRunResult(
        version="2.2.0",
        backend="pytorch",
        suites=[suite],
        summary=summary,
        metadata={"timestamp": "2025-12-11T14:30:00Z", "device": "cpu"},
    )


class TestReporterFormats:
    """Validate Milestone 7 reporter outputs."""

    def test_json_schema_has_required_keys(self):
        reporter = Reporter(_fake_result())
        payload = reporter.to_json()

        assert '"version"' in payload
        assert '"backend"' in payload
        assert '"device"' in payload
        assert '"timestamp"' in payload
        assert '"summary"' in payload
        assert '"suites"' in payload

    def test_markdown_contains_tables(self):
        reporter = Reporter(_fake_result())
        md = reporter.to_markdown()

        assert "# FamilyOS UltraBERT Benchmark Report" in md
        assert "## Summary" in md
        assert "| Total | Passed | Failed |" in md
        assert "## Results by suite" in md
        assert "### api" in md
        assert "| Status | Name |" in md

    def test_text_contains_summary_and_suite(self):
        reporter = Reporter(_fake_result())
        txt = reporter.to_text()

        assert "FamilyOS UltraBERT Benchmark Report" in txt
        assert "SUMMARY" in txt
        assert "RESULTS BY SUITE" in txt
        assert "[api]" in txt

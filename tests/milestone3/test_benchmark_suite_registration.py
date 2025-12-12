"""Milestone 3: Benchmark suite registration tests.

These tests validate that suite classes are registered and discoverable.
They intentionally do not execute benchmarks (which would load the model).
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestBenchmarkSuiteRegistration:
    """Registration tests for benchmark suites."""

    def test_suites_are_registered(self) -> None:
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        names = {getattr(cls, "name", cls.__name__) for cls in get_suite_classes()}
        # Milestone 2
        assert "latency" in names
        # Milestone 3
        assert "safety" in names
        assert "classification" in names

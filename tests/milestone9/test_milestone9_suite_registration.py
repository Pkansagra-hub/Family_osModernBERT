"""Milestone 9: Benchmark suite registration tests.

These tests validate that Milestone 9 suite classes are registered.
They intentionally do not execute benchmarks (which would load the model).
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestMilestone9SuiteRegistration:
    """Registration tests for Milestone 9 benchmark suites."""

    def test_milestone9_suites_are_registered(self) -> None:
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        names = {getattr(cls, "name", cls.__name__) for cls in get_suite_classes()}

        assert "semantic_complexity" in names
        assert "format_structure" in names
        assert "realworld_corruption" in names
        assert "advanced_embedding" in names
        assert "throughput_torture" in names

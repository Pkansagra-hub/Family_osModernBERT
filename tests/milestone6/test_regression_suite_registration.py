"""Milestone 6: Benchmark suite registration tests.

These tests validate that the regression suite class is registered.
They intentionally do not execute benchmarks (which would load the model).
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestRegressionSuiteRegistration:
    """Registration tests for regression benchmark suite."""

    def test_regression_suite_is_registered(self) -> None:
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        names = {getattr(cls, "name", cls.__name__) for cls in get_suite_classes()}
        assert "regression" in names

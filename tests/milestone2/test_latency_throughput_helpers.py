"""Milestone 2: Throughput helper tests.

These tests validate pure helper behavior without loading the model.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestLatencyThroughputHelpers:
    """Helper-level tests for throughput/scaling."""

    def test_make_length_text_is_deterministic(self) -> None:
        from familyos_ultrabert.benchmarks.suite.latency import _make_length_text

        a = _make_length_text(50)
        b = _make_length_text(50)
        assert a == b
        assert isinstance(a, str)
        assert len(a) > 0

    def test_percentile(self) -> None:
        from familyos_ultrabert.benchmarks.suite.latency import _percentile

        samples = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(samples, 0.0) == 1.0
        assert _percentile(samples, 1.0) == 4.0
        p50 = _percentile(samples, 0.5)
        assert 1.0 <= p50 <= 4.0

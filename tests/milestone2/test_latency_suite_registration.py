"""Milestone 2: Latency suite wiring tests.

These tests validate suite registration and basic metadata without loading the model.
"""

from __future__ import annotations

import os
import sys


# Ensure repository root importability for local `familyos_ultrabert/`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestLatencySuiteRegistration:
    """Suite registration tests."""

    def test_latency_suite_is_registered(self) -> None:
        """LatencySuite should be discoverable via suite registry."""
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        suite_names = {getattr(cls, "name", cls.__name__) for cls in get_suite_classes()}
        assert "latency" in suite_names

    def test_latency_suite_metadata(self) -> None:
        """LatencySuite should have stable name/description fields."""
        from familyos_ultrabert.benchmarks.suite.latency import LatencySuite

        assert LatencySuite.name == "latency"
        assert isinstance(LatencySuite.description, str)
        assert len(LatencySuite.description) > 0

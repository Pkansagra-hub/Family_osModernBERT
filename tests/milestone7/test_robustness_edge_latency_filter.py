"""Milestone 7: Robustness edge-case latency gating.

These tests run the robustness suite with a fake client (no model load) and
validate that the latency multiplier gate only applies to short edge cases.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class _FakeAnalyzeResult:
    latency_ms: float
    sentiment: str = "neutral"
    safety: str = "GREEN"


class _FakeClient:
    """Fake client returning deterministic latencies.

    The goal is to simulate a single very-long edge case being much slower than
    the short baseline and short edge cases.
    """

    def analyze(self, text: str, *, capabilities: Optional[List[str]] = None) -> Any:  # noqa: ARG002
        words = len(str(text).split())
        # Keep baseline and short cases stable.
        if words <= 50:
            latency = 20.0
        else:
            # Simulate long inputs being much slower.
            latency = 300.0
        return _FakeAnalyzeResult(latency_ms=latency)


class TestRobustnessEdgeLatencyFilter:
    """Unit tests for robustness latency multiplier filtering."""

    def test_long_edge_case_is_excluded_from_latency_multiplier_gate(self) -> None:
        from familyos_ultrabert.benchmarks.suite.robustness import RobustnessSuite
        from familyos_ultrabert.benchmarks.types import BenchmarkStatus

        suite = RobustnessSuite(_FakeClient())
        results = suite.run()

        gate = next(r for r in results if r.name == "edge_cases_latency_multiplier_max")
        assert gate.status == BenchmarkStatus.PASS

        excluded = gate.details.get("excluded", [])
        excluded_cases = {d.get("case") for d in excluded}
        assert "very_long" in excluded_cases

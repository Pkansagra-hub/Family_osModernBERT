"""Milestone 7: regression suite crisis recall semantics.

These tests are structured to avoid loading model weights.
"""

from __future__ import annotations

from typing import Any, List


class _DummyClient:
    """Deterministic dummy client for benchmark suites."""

    backend: str = "dummy"
    device: str = "cpu"

    def analyze(self, text: str, capabilities: List[str] | None = None) -> Any:  # noqa: ANN401
        class _Res:
            pass

        res = _Res()
        caps = capabilities or []

        # Sentiment direction tests want stable outputs.
        if "sentiment" in caps:
            res.sentiment = "positive"

        # Safety tests: deliberately return CRISIS even for some GREEN examples
        # to simulate a false positive. The regression *crisis recall* check
        # should filter to CRISIS-only cases and still pass.
        if "safety_familyos" in caps:
            res.safety = "CRISIS"
            res.safety_confidence = 0.9

        # Emotions may be requested by determinism checks.
        if "emotions" in caps:
            res.emotions = ["joy"]

        return res

    def get_embedding(self, text: str) -> List[float]:
        # Stable, unit-norm vector (dim=768) so determinism and cosine checks pass.
        vec = [0.0] * 768
        vec[0] = 1.0
        return vec


def test_regression_crisis_recall_filters_to_crisis_only(monkeypatch):
    """The crisis recall check must exclude GREEN examples.

    If GREEN examples are included, a single false positive would incorrectly
    fail the recall check.
    """
    from familyos_ultrabert.benchmarks.suite import regression as reg
    from familyos_ultrabert.benchmarks.types import BenchmarkStatus

    monkeypatch.setenv("FAMILYOS_ULTRABERT_BENCH_QUICK", "1")

    # Reduce golden sets to a tiny controlled subset.
    monkeypatch.setattr(reg, "GOLDEN_SENTIMENT_DIRECTION_CASES", [("hello", "positive")])
    monkeypatch.setattr(reg, "GOLDEN_SENTIMENT_CASES", [])
    monkeypatch.setattr(reg, "GOLDEN_SAFETY_BAND_CASES", [("safe text", "CRISIS")])

    # Include one CRISIS and one GREEN example. The dummy client predicts CRISIS
    # for both, so recall should still be 1.0 after filtering.
    monkeypatch.setattr(
        reg,
        "GOLDEN_SAFETY_CRISIS_CASES",
        [("I want to kill myself", "CRISIS"), ("I'm dying of laughter", "GREEN")],
    )

    monkeypatch.setattr(reg, "GOLDEN_EMOTION_CASES", [])
    monkeypatch.setattr(reg, "GOLDEN_OUTPUTS", {})
    monkeypatch.setattr(reg, "GOLDEN_EMBEDDING_METRICS", {})

    suite = reg.RegressionSuite(_DummyClient())
    results = suite.run()

    target = [r for r in results if r.name == "golden_safety_crisis_recall"]
    assert len(target) == 1
    r = target[0]

    assert r.status == BenchmarkStatus.PASS
    assert r.score == 1.0
    assert r.details.get("total") == 1

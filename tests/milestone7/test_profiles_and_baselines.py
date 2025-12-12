"""Milestone 7+: profiles + baseline drift tracking.

These tests are intentionally lightweight (no model/runtime deps).
"""

from __future__ import annotations

import json
from pathlib import Path

from familyos_ultrabert.benchmarks.baselines import compare_and_update_baseline, environment_key
from familyos_ultrabert.benchmarks.types import (
    BenchmarkResult,
    BenchmarkRunResult,
    BenchmarkSeverity,
    BenchmarkStatus,
    BenchmarkSummary,
    SuiteResult,
)


def _fake_run(*, latency_ms: float) -> BenchmarkRunResult:
    summary = BenchmarkSummary(total=1, passed=1, failed=0, skipped=0, errored=0, duration_sec=0.1, warned=0, info=0)
    suite = SuiteResult(
        suite_name="latency",
        results=[
            BenchmarkResult(
                name="single_sentiment_p95",
                category="latency",
                status=BenchmarkStatus.PASS,
                severity=BenchmarkSeverity.FAIL,
                latency_ms=latency_ms,
                details={"p95_ms": latency_ms, "p50_ms": latency_ms * 0.8, "p99_ms": latency_ms * 1.2},
            )
        ],
        total_time_sec=0.1,
    )
    return BenchmarkRunResult(
        version="0.0.0",
        backend="onnx",
        suites=[suite],
        summary=summary,
        metadata={"device": "cpu", "model": {"sha256": "abc"}},
    )


def test_environment_key_stable():
    key = environment_key(_fake_run(latency_ms=10.0))
    assert "onnx" in key
    assert "cpu" in key
    assert "abc" in key


def test_baseline_compare_and_update(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ULTRABERT_BENCH_BASELINE_GATE_LATENCY", "1")
    # First run: baseline is created, no compare.
    r1 = _fake_run(latency_ms=10.0)
    rep1 = compare_and_update_baseline(r1, baseline_dir=tmp_path)
    assert rep1["enabled"] is True
    assert rep1["compared"] is False

    # Second run: slower latency beyond default 10% tolerance => regression, FAIL gate.
    r2 = _fake_run(latency_ms=12.0)
    rep2 = compare_and_update_baseline(r2, baseline_dir=tmp_path)
    assert rep2["compared"] is True
    assert int(rep2.get("failed", 0)) >= 1

    # Ensure baseline file exists and is valid JSON.
    p = Path(rep2["baseline_path"])
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["backend"] == "onnx"

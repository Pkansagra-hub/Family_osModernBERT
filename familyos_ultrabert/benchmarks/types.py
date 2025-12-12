"""Shared types for the benchmark suite.

These types are part of the public API for results serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BenchmarkStatus(str, Enum):
    """Status for a single benchmark result."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class BenchmarkSeverity(str, Enum):
    """Severity of a benchmark result.

    Use this to separate hard release gates (FAIL) from soft signals that
    should be tracked over time (WARN/INFO).
    """

    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single benchmark check."""

    name: str
    category: str
    status: BenchmarkStatus
    severity: BenchmarkSeverity = BenchmarkSeverity.FAIL
    score: Optional[float] = None
    threshold: Optional[float] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SuiteResult:
    """Result of executing a benchmark suite."""

    suite_name: str
    results: List[BenchmarkResult]
    total_time_sec: float

    passed: int = 0
    failed: int = 0
    warned: int = 0
    info: int = 0
    skipped: int = 0
    errored: int = 0

    def __post_init__(self) -> None:
        self.passed = sum(1 for r in self.results if r.status == BenchmarkStatus.PASS)
        self.failed = sum(
            1
            for r in self.results
            if r.status == BenchmarkStatus.FAIL and r.severity == BenchmarkSeverity.FAIL
        )
        self.warned = sum(
            1
            for r in self.results
            if r.status == BenchmarkStatus.FAIL and r.severity == BenchmarkSeverity.WARN
        )
        self.info = sum(1 for r in self.results if r.severity == BenchmarkSeverity.INFO)
        self.skipped = sum(1 for r in self.results if r.status == BenchmarkStatus.SKIP)
        self.errored = sum(1 for r in self.results if r.status == BenchmarkStatus.ERROR)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate pass/fail counts across the entire run."""

    total: int
    passed: int
    failed: int
    skipped: int
    errored: int
    duration_sec: float
    warned: int = 0
    info: int = 0


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Top-level benchmark run result."""

    version: str
    backend: str
    suites: List[SuiteResult]
    summary: BenchmarkSummary
    metadata: Dict[str, Any] = field(default_factory=dict)

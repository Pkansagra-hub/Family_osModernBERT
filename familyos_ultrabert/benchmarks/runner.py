"""Benchmark runner.

Milestone 1 runner is intentionally lightweight:
- loads `familyos_ultrabert.Client`
- discovers suites from `familyos_ultrabert.benchmarks.suite`
- runs suites and aggregates results

Per-capability suites are introduced in later milestones.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from familyos_ultrabert import __version__
from familyos_ultrabert.benchmarks.types import (
    BenchmarkResult,
    BenchmarkRunResult,
    BenchmarkStatus,
    BenchmarkSummary,
    SuiteResult,
)


class BenchmarkRunner:
    """Runs benchmark suites and returns structured results."""

    def __init__(
        self,
        *,
        suites: Optional[List[str]] = None,
        backend: str = "auto",
        warmup_rounds: int = 3,
        verbose: bool = True,
    ):
        self._suite_filter = suites
        self._backend = backend
        self._warmup_rounds = warmup_rounds
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _create_client(self) -> Any:
        """Create a configured client instance."""
        from familyos_ultrabert import Client

        return Client(
            backend=self._backend,
            warmup=True,
            warmup_rounds=self._warmup_rounds,
            verbose=self._verbose,
        )

    def _discover_suites(self) -> List[Any]:
        """Discover suite classes."""
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        suite_classes = get_suite_classes()
        if self._suite_filter:
            suite_classes = [c for c in suite_classes if getattr(c, "name", "") in self._suite_filter]
        return suite_classes

    def run(self) -> BenchmarkRunResult:
        """Run all discovered suites.

        Returns:
            BenchmarkRunResult containing all suite results.
        """
        overall_start = time.time()

        suite_results: List[SuiteResult] = []
        suite_classes = self._discover_suites()

        # Milestone 1: no suites are registered yet. Avoid loading the model.
        if not suite_classes:
            duration = time.time() - overall_start
            summary = BenchmarkSummary(
                total=0,
                passed=0,
                failed=0,
                skipped=0,
                errored=0,
                duration_sec=duration,
            )
            return BenchmarkRunResult(
                version=__version__,
                backend="unknown",
                suites=[],
                summary=summary,
                metadata={
                    "note": "No benchmark suites are registered yet."
                    " This is expected in Milestone 1.",
                    "runner": {
                        "backend": self._backend,
                        "warmup_rounds": self._warmup_rounds,
                        "suite_filter": self._suite_filter,
                    },
                },
            )

        client = self._create_client()
        active_backend = getattr(client, "backend", "unknown")

        self._log(f"UltraBERT benchmark runner (package={__version__}, backend={active_backend})")

        for suite_cls in suite_classes:
            suite_name = getattr(suite_cls, "name", suite_cls.__name__)
            self._log("")
            self._log(f"Running suite: {suite_name}")

            suite_start = time.time()
            suite = suite_cls(client)
            try:
                results = suite.run()
            except Exception as exc:  # noqa: BLE001
                # Keep runner resilient: record error as a single result.
                results = [
                    BenchmarkResult(
                        name="suite_error",
                        category=suite_name,
                        status=BenchmarkStatus.ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                ]

            elapsed = time.time() - suite_start
            suite_results.append(SuiteResult(suite_name=suite_name, results=list(results), total_time_sec=elapsed))

        # Aggregate counts
        all_results: List[BenchmarkResult] = []
        for s in suite_results:
            all_results.extend(s.results)

        passed = sum(1 for r in all_results if r.status == BenchmarkStatus.PASS)
        failed = sum(1 for r in all_results if r.status == BenchmarkStatus.FAIL)
        skipped = sum(1 for r in all_results if r.status == BenchmarkStatus.SKIP)
        errored = sum(1 for r in all_results if r.status == BenchmarkStatus.ERROR)

        duration = time.time() - overall_start
        summary = BenchmarkSummary(
            total=len(all_results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errored=errored,
            duration_sec=duration,
        )

        metadata: Dict[str, Any] = {
            "client_backend": active_backend,
            "client_stats": getattr(client, "stats", None),
            "runner": {
                "backend": self._backend,
                "warmup_rounds": self._warmup_rounds,
                "suite_filter": self._suite_filter,
            },
        }

        # Convert dataclass stats if necessary
        if metadata.get("client_stats") is not None and hasattr(metadata["client_stats"], "__dict__"):
            try:
                metadata["client_stats"] = asdict(metadata["client_stats"])  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                pass

        return BenchmarkRunResult(
            version=__version__,
            backend=active_backend,
            suites=suite_results,
            summary=summary,
            metadata=metadata,
        )

"""FamilyOS UltraBERT benchmark suite.

This package is intended to ship inside the `familyos-ultrabert` wheel.

Design goals:
- Self-contained: no dependencies beyond `familyos_ultrabert` and the Python standard library.
- Runnable by end users after installation.
- Produces structured results that can be exported as JSON/Markdown later.

Usage (after full implementation):
    python -m familyos_ultrabert.benchmarks

Milestone 1 provides the core framework (types, base class, runner) without
implementing the per-capability benchmark suites yet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from familyos_ultrabert.benchmarks.runner import BenchmarkRunner
from familyos_ultrabert.benchmarks.reporter import Reporter
from familyos_ultrabert.benchmarks.types import BenchmarkRunResult


_BENCHMARK_PROFILES: dict[str, List[str]] = {
    # Fast CI gate (always run). Keep this correctness-focused.
    "smoke": [
        "api",
        "regression",
        "safety",
        "embeddings",
        "latency",
        "format_structure",
        "realworld_corruption",
    ],
    # Nightly / pre-release (superset). Includes heavier suites.
    "full": [
        "api",
        "regression",
        "safety",
        "embeddings",
        "latency",
        "format_structure",
        "realworld_corruption",
        "robustness",
        "classification",
        "semantic_complexity",
        "throughput_torture",
        "advanced_embedding",
    ],
}


def run_all(
    suites: Optional[List[str]] = None,
    backend: str = "auto",
    warmup_rounds: int = 3,
    verbose: bool = True,
) -> BenchmarkRunResult:
    """Run all benchmark suites and return a structured result.

    Args:
        suites: Optional list of suite names to run. When None, run all available.
        backend: "auto", "pytorch", or "onnx".
        warmup_rounds: Number of warmup rounds to run before measuring.
        verbose: If True, print progress.

    Returns:
        Aggregated benchmark results.
    """
    runner = BenchmarkRunner(
        suites=suites,
        backend=backend,
        warmup_rounds=warmup_rounds,
        verbose=verbose,
    )
    return runner.run()


def _available_suite_names() -> List[str]:
    """Return all registered suite names."""
    from familyos_ultrabert.benchmarks.suite import get_suite_classes

    names: List[str] = []
    for cls in get_suite_classes():
        name = str(getattr(cls, "name", "")).strip()
        if name:
            names.append(name)
    return sorted(set(names))


def _parse_suite_list(raw: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated suite list."""
    if raw is None:
        return None
    suites = [s.strip() for s in str(raw).split(",") if s.strip()]
    return suites or None


def _profile_suites(profile: str, available: List[str]) -> List[str]:
    """Return suites for a known profile.

    Args:
        profile: Profile name.
        available: Available suite names.

    Returns:
        List of suites to run (in profile order), filtered to those available.

    Raises:
        ValueError: If profile is unknown.
    """
    key = str(profile).strip().lower()
    if key not in _BENCHMARK_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    want = list(_BENCHMARK_PROFILES[key])
    avail = set(available)
    return [s for s in want if s in avail]


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for `python -m familyos_ultrabert.benchmarks`.

    Args:
        argv: Optional argv override (excluding program name). When None,
            argparse reads from sys.argv.

    Returns:
        Process exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="FamilyOS UltraBERT Benchmark Suite")
    parser.add_argument(
        "--profile",
        choices=sorted(_BENCHMARK_PROFILES.keys()),
        help="Run a standard benchmark profile (takes priority over --suite)",
    )
    parser.add_argument("--suite", type=str, help="Comma-separated list of suites to run")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--output", type=str, help="Save report to file")
    parser.add_argument(
        "--baseline-dir",
        type=str,
        help="Directory to store/compare last-known-good benchmark baselines",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Enable baseline drift compare+save (CI-grade regression tracking)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(list(argv) if argv is not None else None)

    suites = _parse_suite_list(args.suite)
    profile_name = str(args.profile).strip().lower() if args.profile else None

    available = _available_suite_names()
    if profile_name:
        if suites is not None:
            print("Note: --profile takes priority over --suite")
        try:
            suites = _profile_suites(profile_name, available)
        except ValueError as exc:
            print(str(exc))
            print(f"Available profiles: {', '.join(sorted(_BENCHMARK_PROFILES.keys()))}")
            return 2
        missing = [s for s in _BENCHMARK_PROFILES[profile_name] if s not in set(available)]
        if missing:
            # Keep this non-fatal so installed environments without optional deps can still run.
            print(f"Note: profile '{profile_name}' missing suites: {', '.join(missing)}")
    if args.quick:
        # Suites may consult this to reduce runtime and avoid large loops.
        os.environ["FAMILYOS_ULTRABERT_BENCH_QUICK"] = "1"
        # Back-compat: if user did not pick suites or profile, default to smoke.
        if suites is None and profile_name is None:
            suites = _BENCHMARK_PROFILES["smoke"]

    # Validate suite names early to avoid unnecessary model load.
    if suites is not None:
        unknown = [s for s in suites if s not in set(available)]
        if unknown:
            valid = ", ".join(available) if available else "(none)"
            print(f"Unknown suite(s): {', '.join(unknown)}")
            print(f"Available suites: {valid}")
            return 2

    runner = BenchmarkRunner(suites=suites, verbose=bool(args.verbose))
    results = runner.run()

    baseline_failed = 0
    baseline_warned = 0
    if bool(args.baseline):
        try:
            from familyos_ultrabert.benchmarks.baselines import compare_and_update_baseline

            base_dir = Path(args.baseline_dir) if args.baseline_dir else None
            baseline_report = compare_and_update_baseline(results, baseline_dir=base_dir)
            # Even though BenchmarkRunResult is frozen, metadata is a mutable dict.
            results.metadata["baseline"] = baseline_report
            baseline_failed = int(baseline_report.get("failed", 0))
            baseline_warned = int(baseline_report.get("warned", 0))
        except Exception as exc:  # noqa: BLE001
            results.metadata["baseline"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    reporter = Reporter(results)
    if args.format == "json":
        output = reporter.to_json()
    elif args.format == "markdown":
        output = reporter.to_markdown()
    else:
        output = reporter.to_text()

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
        except OSError as exc:
            print(f"Failed to write output file: {args.output} ({type(exc).__name__}: {exc})")
            return 2
    else:
        print(output, end="")

    # Non-zero if there are hard failures or errors (including baseline drift gates).
    hard_failed = (results.summary.failed + results.summary.errored + baseline_failed) > 0
    if hard_failed:
        return 1
    # Baseline warnings should not fail the run.
    _ = baseline_warned
    return 0


def main() -> int:
    """Script entry point.

    This function is intentionally a thin wrapper around :func:`cli` so it can
    be used as a console script entrypoint.

    Returns:
        Process exit code.
    """
    return cli(None)


__all__ = [
    "BenchmarkRunner",
    "BenchmarkRunResult",
    "Reporter",
    "cli",
    "main",
    "run_all",
]

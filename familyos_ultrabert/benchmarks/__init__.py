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
from typing import List, Optional, Sequence

from familyos_ultrabert.benchmarks.runner import BenchmarkRunner
from familyos_ultrabert.benchmarks.reporter import Reporter
from familyos_ultrabert.benchmarks.types import BenchmarkRunResult


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


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for `python -m familyos_ultrabert.benchmarks`.

    Args:
        argv: Optional argv override (excluding program name). When None,
            argparse reads from sys.argv.

    Returns:
        Process exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="FamilyOS UltraBERT Benchmark Suite")
    parser.add_argument("--suite", type=str, help="Comma-separated list of suites to run")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--output", type=str, help="Save report to file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(list(argv) if argv is not None else None)

    suites = _parse_suite_list(args.suite)
    if args.quick:
        # Suites may consult this to reduce runtime and avoid large loops.
        os.environ["FAMILYOS_ULTRABERT_BENCH_QUICK"] = "1"
        # If user did not pick suites, default to a fast, correctness-focused subset.
        if suites is None:
            suites = ["api", "regression"]

    # Validate suite names early to avoid unnecessary model load.
    available = _available_suite_names()
    if suites is not None:
        unknown = [s for s in suites if s not in set(available)]
        if unknown:
            valid = ", ".join(available) if available else "(none)"
            print(f"Unknown suite(s): {', '.join(unknown)}")
            print(f"Available suites: {valid}")
            return 2

    runner = BenchmarkRunner(suites=suites, verbose=bool(args.verbose))
    results = runner.run()

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

    # Non-zero if there are failures or errors.
    return 0 if (results.summary.failed == 0 and results.summary.errored == 0) else 1


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

"""Benchmark suite registry.

Suites are registered here so the runner can discover them without filesystem
scanning or third-party plugins.

Imports at the bottom are intentionally defensive: if an optional suite fails to
import (due to missing runtime deps), other suites can still register.
"""

from __future__ import annotations

from typing import List, Type

from familyos_ultrabert.benchmarks.base import BenchmarkSuite


_SUITE_CLASSES: List[Type[BenchmarkSuite]] = []


def register_suite(suite_cls: Type[BenchmarkSuite]) -> Type[BenchmarkSuite]:
    """Register a benchmark suite.

    Args:
        suite_cls: The BenchmarkSuite subclass to register.

    Returns:
        The same class, to allow decorator use.
    """
    _SUITE_CLASSES.append(suite_cls)
    return suite_cls


def get_suite_classes() -> List[Type[BenchmarkSuite]]:
    """Return registered suite classes."""
    return list(_SUITE_CLASSES)


# Import suites for registration side-effects.
# Keep this at the bottom to avoid circular imports during module initialization.
#
# Important: each import is isolated so a failure in one suite does not prevent
# other suites from registering.
for _mod in (
    "api",
    "advanced_embedding",
    "classification",
    "embeddings",
    "format_structure",
    "latency",
    "realworld_corruption",
    "regression",
    "robustness",
    "semantic_complexity",
    "safety",
    "throughput_torture",
):
    try:
        __import__(f"{__name__}.{_mod}")
    except Exception:
        # Benchmarks must remain importable even if optional runtime dependencies
        # for some suites are not available in the environment.
        pass

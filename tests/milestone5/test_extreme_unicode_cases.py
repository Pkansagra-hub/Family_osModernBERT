"""Milestone 5: Extreme Unicode test-case validations.

These tests validate EXTREME_UNICODE_CASES is present and well-formed.
They intentionally do not load the model or execute inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestExtremeUnicodeCases:
    """Validates extreme unicode test case shapes."""

    def test_extreme_unicode_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import EXTREME_UNICODE_CASES

        assert isinstance(EXTREME_UNICODE_CASES, list)
        assert len(EXTREME_UNICODE_CASES) >= 5

        labels = set()
        for label, text in EXTREME_UNICODE_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str)
            labels.add(label)

        assert len(labels) == len(EXTREME_UNICODE_CASES)

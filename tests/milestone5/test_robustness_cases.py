"""Milestone 5: Robustness test-case validations.

These tests validate the inline test cases for the robustness suite.
They intentionally do not load the model or run inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestRobustnessCases:
    """Validates robustness suite test case shapes."""

    def test_edge_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import EDGE_CASES

        assert isinstance(EDGE_CASES, list)
        assert len(EDGE_CASES) >= 9
        labels = set()
        for label, text in EDGE_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str)
            labels.add(label)
        assert len(labels) == len(EDGE_CASES)

    def test_unicode_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import UNICODE_CASES

        assert isinstance(UNICODE_CASES, list)
        assert len(UNICODE_CASES) >= 10
        labels = set()
        for label, text in UNICODE_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str) and text.strip()
            labels.add(label)
        assert len(labels) == len(UNICODE_CASES)

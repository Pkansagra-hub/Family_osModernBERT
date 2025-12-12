"""Milestone 5 / Issue #17: Adversarial test-case validations.

These tests validate the inline adversarial test cases.
They intentionally do not load the model or run inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestAdversarialCases:
    """Validates adversarial suite test case shapes."""

    def test_adversarial_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import ADVERSARIAL_CASES

        assert isinstance(ADVERSARIAL_CASES, list)
        assert len(ADVERSARIAL_CASES) >= 6

        allowed = {"injection", "jailbreak", "sql", "xss", "format"}
        for text, label in ADVERSARIAL_CASES:
            assert isinstance(text, str) and text.strip()
            assert isinstance(label, str) and label.strip()
            assert label in allowed

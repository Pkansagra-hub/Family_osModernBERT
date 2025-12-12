"""Milestone 3: Intent test-case validations.

These tests validate the inline INTENT_CASES data.
They intentionally do not load the model or execute inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestIntentCases:
    """Validates that intent benchmark inputs are present and well-formed."""

    def test_intent_cases_well_formed_and_labels_valid(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import INTENT_CASES
        from familyos_ultrabert.labels import INTENT_LABELS

        assert isinstance(INTENT_CASES, list)
        assert len(INTENT_CASES) >= 7

        valid = set(INTENT_LABELS.label2id.keys())
        labels = set()

        for text, expected in INTENT_CASES:
            assert isinstance(text, str) and text.strip()
            assert isinstance(expected, str) and expected.strip()
            assert expected in valid
            labels.add(expected)

        # Ensure we're covering multiple distinct intents.
        assert len(labels) >= 4

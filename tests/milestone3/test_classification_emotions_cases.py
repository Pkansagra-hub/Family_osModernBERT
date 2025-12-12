"""Milestone 3 / Issue #11: Emotions test-case and helper validations.

These tests validate the inline EMOTION_CASES data and normalization helper.
They intentionally do not load the model or execute inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestClassificationEmotionsCases:
    """Validates that emotions benchmark inputs are present and well-formed."""

    def test_emotion_cases_exist_and_include_fine_grained(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import EMOTION_CASES

        assert isinstance(EMOTION_CASES, list)
        assert len(EMOTION_CASES) >= 7

        all_expected = set()
        for text, expected in EMOTION_CASES:
            assert isinstance(text, str)
            assert text.strip()
            assert isinstance(expected, list)
            assert all(isinstance(e, str) and e.strip() for e in expected)
            all_expected.update(e.strip().lower() for e in expected)

        # Plan requires fine-grained emotions to be covered.
        assert "nostalgia" in all_expected
        assert "protectiveness" in all_expected

    def test_normalize_emotion_labels(self) -> None:
        from familyos_ultrabert.benchmarks.suite.classification import _normalize_emotion_labels

        assert _normalize_emotion_labels(["Joy", " love "]) == ["joy", "love"]
        assert _normalize_emotion_labels({"Gratitude": 0.9, "": 0.1}) == ["gratitude"]
        assert _normalize_emotion_labels("  Embarrassment  ") == ["embarrassment"]
        assert _normalize_emotion_labels(None) == []

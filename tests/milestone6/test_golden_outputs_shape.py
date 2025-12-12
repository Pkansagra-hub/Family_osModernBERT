"""Milestone 6 / Issue #20: Golden outputs shape validations.

These tests validate the golden outputs structure.
They intentionally do not load the model or run inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestGoldenOutputs:
    """Validates regression golden outputs are well-formed."""

    def test_golden_outputs_dict_shape(self) -> None:
        from familyos_ultrabert.benchmarks.data.golden_outputs import GOLDEN_OUTPUTS

        assert isinstance(GOLDEN_OUTPUTS, dict)
        assert len(GOLDEN_OUTPUTS) >= 2

        for text, expected in GOLDEN_OUTPUTS.items():
            assert isinstance(text, str) and text.strip()
            assert isinstance(expected, dict)

            if "sentiment" in expected:
                assert expected["sentiment"] in {
                    "very_negative",
                    "negative",
                    "neutral",
                    "positive",
                    "very_positive",
                }
            if "safety" in expected:
                assert expected["safety"] in {"GREEN", "AMBER", "RED", "CRISIS"}
            if "emotions_contain" in expected:
                assert isinstance(expected["emotions_contain"], list)
            if "entities_contain" in expected:
                assert isinstance(expected["entities_contain"], list)

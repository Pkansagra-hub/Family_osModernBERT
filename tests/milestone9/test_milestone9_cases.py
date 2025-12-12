"""Milestone 9: Dataset shape validations.

These tests validate Milestone 9 datasets are present and well-formed.
They intentionally do not load the model or execute inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestMilestone9Cases:
    """Validates Milestone 9 test case shapes."""

    def test_semantic_complexity_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import SEMANTIC_COMPLEXITY_CASES

        assert isinstance(SEMANTIC_COMPLEXITY_CASES, list)
        assert len(SEMANTIC_COMPLEXITY_CASES) >= 5

        labels = set()
        for label, text in SEMANTIC_COMPLEXITY_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str)
            labels.add(label)

        assert len(labels) == len(SEMANTIC_COMPLEXITY_CASES)

    def test_format_structure_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import FORMAT_STRUCTURE_CASES

        assert isinstance(FORMAT_STRUCTURE_CASES, list)
        assert len(FORMAT_STRUCTURE_CASES) >= 5

        labels = set()
        for label, text in FORMAT_STRUCTURE_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str)
            labels.add(label)

        assert len(labels) == len(FORMAT_STRUCTURE_CASES)

    def test_realworld_corruption_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import REALWORLD_CORRUPTION_CASES

        assert isinstance(REALWORLD_CORRUPTION_CASES, list)
        assert len(REALWORLD_CORRUPTION_CASES) >= 5

        labels = set()
        for label, text in REALWORLD_CORRUPTION_CASES:
            assert isinstance(label, str) and label.strip()
            assert isinstance(text, str)
            labels.add(label)

        assert len(labels) == len(REALWORLD_CORRUPTION_CASES)

    def test_advanced_ranking_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import ADVANCED_RANKING_CASES

        assert isinstance(ADVANCED_RANKING_CASES, list)
        assert len(ADVANCED_RANKING_CASES) >= 3

        for case in ADVANCED_RANKING_CASES:
            assert isinstance(case, dict)
            assert isinstance(case.get("query"), str) and case["query"].strip()

            docs = case.get("documents")
            assert isinstance(docs, list)
            assert len(docs) >= 3

            ids = set()
            for d in docs:
                assert isinstance(d, dict)
                did = d.get("id")
                text = d.get("text")
                rel = d.get("relevance")

                assert isinstance(did, str) and did.strip()
                assert isinstance(text, str) and text.strip()
                assert isinstance(rel, int)
                assert rel in {0, 1, 2}

                ids.add(did)

            assert len(ids) == len(docs)

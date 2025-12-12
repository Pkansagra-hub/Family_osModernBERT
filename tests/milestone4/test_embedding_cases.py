"""Milestone 4: Embeddings test-case validations.

These tests validate the inline test cases for the embedding suite.
They intentionally do not load the model or run inference.
"""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestEmbeddingCases:
    """Validates embedding suite test case shapes."""

    def test_similarity_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import SIMILARITY_CASES

        assert isinstance(SIMILARITY_CASES, list)
        assert len(SIMILARITY_CASES) >= 5
        for t1, t2, thr in SIMILARITY_CASES:
            assert isinstance(t1, str) and t1.strip()
            assert isinstance(t2, str) and t2.strip()
            assert isinstance(thr, float)
            assert 0.0 < thr <= 1.0

    def test_triplet_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import TRIPLET_CASES

        assert isinstance(TRIPLET_CASES, list)
        assert len(TRIPLET_CASES) >= 2
        for case in TRIPLET_CASES:
            assert "anchor" in case
            assert "positive" in case
            assert "negatives" in case
            assert isinstance(case["negatives"], list)
            assert len(case["negatives"]) >= 1

    def test_retrieval_cases_well_formed(self) -> None:
        from familyos_ultrabert.benchmarks.data.test_cases import RETRIEVAL_CASES_10, RETRIEVAL_CASES_100

        assert len(RETRIEVAL_CASES_10) >= 1
        assert len(RETRIEVAL_CASES_100) >= 1

        for cases in (RETRIEVAL_CASES_10, RETRIEVAL_CASES_100):
            for rc in cases:
                assert isinstance(rc.get("query"), str) and rc["query"].strip()
                assert isinstance(rc.get("relevant"), str) and rc["relevant"].strip()
                distractors = rc.get("distractors")
                assert isinstance(distractors, list)
                assert len(distractors) > 0

"""
Tests for head type mapping (Issue 14.1.2).

Tests that CounterfactualDecoderHead is properly registered.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Issue 14.1.2: Register Head Type Mapping
# =============================================================================


class TestCounterfactualHeadMapped:
    """Tests for COUNTERFACTUAL head mapping (14.1.2-T1)."""

    def test_counterfactual_head_mapped(self):
        """14.1.2-T1: COUNTERFACTUAL maps to CounterfactualDecoderHead."""
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.decoder_moe import CounterfactualDecoderHead
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert Capability.COUNTERFACTUAL in CAPABILITY_TO_HEAD_TYPE
        assert CAPABILITY_TO_HEAD_TYPE[Capability.COUNTERFACTUAL] is CounterfactualDecoderHead

    def test_mapping_has_13_entries(self):
        """Mapping has 13 entries (all capabilities)."""
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE

        assert len(CAPABILITY_TO_HEAD_TYPE) == 13


class TestImportNoErrors:
    """Tests for clean imports (14.1.2-T2)."""

    def test_import_no_errors(self):
        """14.1.2-T2: Import succeeds without errors."""
        # This test passes if imports work
        from modeling_studio.data.labels import Capability
        from modeling_studio.models.modernbert_multitask import (
            CAPABILITY_TO_HEAD_TYPE,
            ModernBertMultiTaskModel,
        )

        assert ModernBertMultiTaskModel is not None
        assert CAPABILITY_TO_HEAD_TYPE is not None
        assert Capability.COUNTERFACTUAL is not None

    def test_import_decoder_head_directly(self):
        """CounterfactualDecoderHead importable directly."""
        from modeling_studio.models.decoder_moe import CounterfactualDecoderHead

        assert CounterfactualDecoderHead is not None

    def test_import_via_modernbert_multitask(self):
        """Import chain through modernbert_multitask works."""
        from modeling_studio.models.modernbert_multitask import CAPABILITY_TO_HEAD_TYPE
        from modeling_studio.data.labels import Capability

        head_cls = CAPABILITY_TO_HEAD_TYPE[Capability.COUNTERFACTUAL]
        assert head_cls.__name__ == "CounterfactualDecoderHead"

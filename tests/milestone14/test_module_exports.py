"""
Tests for module exports (Issue 14.2.1, 14.2.2).

Tests that all new classes are properly exported.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Issue 14.2.1: Update models/__init__.py
# =============================================================================


class TestDecoderClassesExported:
    """Tests for decoder classes export (14.2.1-T1)."""

    def test_decoder_classes_exported(self):
        """14.2.1-T1: Decoder classes importable from models."""
        from modeling_studio.models import (
            CounterfactualDecoderHead,
            DecoderBlock,
            DecoderMoEConfig,
            EncoderProjection,
        )

        assert CounterfactualDecoderHead is not None
        assert DecoderBlock is not None
        assert DecoderMoEConfig is not None
        assert EncoderProjection is not None

    def test_decoder_config_importable(self):
        """DecoderMoEConfig importable from models."""
        from modeling_studio.models import DecoderMoEConfig

        # Should be able to instantiate with defaults
        config = DecoderMoEConfig()
        assert config.hidden_size == 1280
        assert config.num_layers == 8


class TestMoEClassesExported:
    """Tests for MoE classes export (14.2.1-T2)."""

    def test_moe_classes_exported(self):
        """14.2.1-T2: MoE classes importable from models."""
        from modeling_studio.models import (
            MoELayer,
            SwiGLUExpert,
            TopKRouter,
        )

        assert MoELayer is not None
        assert SwiGLUExpert is not None
        assert TopKRouter is not None


class TestAttentionClassesExported:
    """Tests for attention classes export."""

    def test_attention_classes_exported(self):
        """Attention classes importable from models."""
        from modeling_studio.models import (
            CrossAttention,
            GroupedQueryAttention,
            RotaryEmbedding,
        )

        assert CrossAttention is not None
        assert GroupedQueryAttention is not None
        assert RotaryEmbedding is not None


class TestModelsAllExported:
    """Tests for __all__ list in models."""

    def test_all_new_classes_in_all(self):
        """All new classes are in __all__."""
        from modeling_studio import models

        expected_exports = [
            "DecoderMoEConfig",
            "CounterfactualDecoderHead",
            "DecoderBlock",
            "EncoderProjection",
            "MoELayer",
            "TopKRouter",
            "SwiGLUExpert",
            "GroupedQueryAttention",
            "CrossAttention",
            "RotaryEmbedding",
        ]

        for name in expected_exports:
            assert name in models.__all__, f"{name} not in models.__all__"


# =============================================================================
# Issue 14.2.2: Add Collator Export
# =============================================================================


class TestCollatorExported:
    """Tests for collator export (14.2.2-T1)."""

    def test_collator_exported(self):
        """14.2.2-T1: CounterfactualCollator importable from trainers.collators."""
        from modeling_studio.trainers.collators import CounterfactualCollator

        assert CounterfactualCollator is not None

    def test_collator_in_all(self):
        """CounterfactualCollator is in __all__."""
        from modeling_studio.trainers import collators

        assert "CounterfactualCollator" in collators.__all__

    def test_no_circular_imports(self):
        """No circular imports when importing collator."""
        # This test passes if imports work without error
        from modeling_studio.trainers.collators import (
            CounterfactualCollator,
            MultiTaskCollator,
            SequenceClassificationCollator,
        )

        assert CounterfactualCollator is not None
        assert MultiTaskCollator is not None
        assert SequenceClassificationCollator is not None


class TestCollatorInstantiation:
    """Tests for CounterfactualCollator instantiation."""

    def test_collator_instantiation(self):
        """CounterfactualCollator can be instantiated."""
        from transformers import AutoTokenizer

        from modeling_studio.trainers.collators import CounterfactualCollator

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        collator = CounterfactualCollator(tokenizer=tokenizer)

        assert collator is not None
        assert collator.tokenizer is tokenizer

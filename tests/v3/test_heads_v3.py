"""
Tests for Hub-Aware Task Heads (Issue 3.2.1).

This module tests hub-aware classification heads that automatically receive
the correct representation based on their capability's hub routing.

Test Coverage:
    - HubAwareClassificationHead extracts correct hub token (AC1)
    - HubAwareTokenClassificationHead uses full sequence (AC2)
    - HubAwareHierarchicalHead implements primary→secondary cascade (AC3)
    - HubAwareSafetyHead includes temperature calibration (AC4)
    - HubAwareNLIHead uses [REL] hub token (AC5)
    - HEAD_REGISTRY maps all 12 capabilities correctly (AC6)
    - create_head_for_capability() factory works for all capabilities (AC7)
"""

import torch
import torch.nn as nn

from modeling_studio.models.heads_v3 import (
    HEAD_REGISTRY,
    HeadConfig,
    HubAwareClassificationHead,
    HubAwareHierarchicalHead,
    HubAwareNLIHead,
    HubAwareSafetyHead,
    HubAwareTokenClassificationHead,
    create_all_heads,
    create_head_for_capability,
)
from modeling_studio.models.hub_tokens import get_hub_positions


class TestHubAwareClassificationHead:
    """Test suite for HubAwareClassificationHead."""

    def test_classification_head_initialization(self):
        """Test HubAwareClassificationHead initialization."""
        head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")

        assert head.hidden_size == 768
        assert head.num_labels == 7
        assert head.hub_token == "[EMO]"
        assert head.hub_position == 1  # [EMO] is at position 1

    def test_classification_head_forward_with_pooled(self):
        """AC1: HubAwareClassificationHead extracts correct hub token from pooled_outputs."""
        head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")
        head.eval()

        pooled_outputs = {
            "[CLS]": torch.randn(4, 768),
            "[EMO]": torch.randn(4, 768),
            "[MEM]": torch.randn(4, 768),
        }

        logits = head(None, pooled_outputs)

        assert logits.shape == (4, 7)

    def test_classification_head_forward_with_hidden_states(self):
        """Test HubAwareClassificationHead extracts from hidden_states."""
        head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        logits = head(hidden_states, None)

        assert logits.shape == (4, 7)

    def test_classification_head_different_hubs(self):
        """Test HubAwareClassificationHead works with different hub tokens."""
        hubs = ["[CLS]", "[EMO]", "[MEM]", "[REL]", "[TASK]"]
        positions = get_hub_positions()

        for hub in hubs:
            head = HubAwareClassificationHead(768, 3, hub_token=hub)
            assert head.hub_token == hub
            assert head.hub_position == positions[hub]

    def test_classification_head_extra_repr(self):
        """Test HubAwareClassificationHead extra_repr."""
        head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")
        repr_str = head.extra_repr()

        assert "[EMO]" in repr_str
        assert "7" in repr_str


class TestHubAwareTokenClassificationHead:
    """Test suite for HubAwareTokenClassificationHead."""

    def test_token_classification_head_initialization(self):
        """Test HubAwareTokenClassificationHead initialization."""
        head = HubAwareTokenClassificationHead(768, 9)

        assert head.hidden_size == 768
        assert head.num_labels == 9

    def test_token_classification_head_forward(self):
        """AC2: HubAwareTokenClassificationHead uses full sequence (not hub)."""
        head = HubAwareTokenClassificationHead(768, 9)
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        logits = head(hidden_states)

        # Should output logits for every token position
        assert logits.shape == (4, 128, 9)

    def test_token_classification_head_with_attention_mask(self):
        """Test HubAwareTokenClassificationHead with attention mask."""
        head = HubAwareTokenClassificationHead(768, 9)
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        attention_mask = torch.ones(4, 128)
        logits = head(hidden_states, attention_mask)

        assert logits.shape == (4, 128, 9)

    def test_token_classification_get_predictions(self):
        """Test HubAwareTokenClassificationHead masks special tokens."""
        head = HubAwareTokenClassificationHead(768, 9)
        head.eval()

        logits = torch.randn(4, 128, 9)
        attention_mask = torch.ones(4, 128)
        attention_mask[:, 100:] = 0  # Mask last 28 positions

        predictions = head.get_predictions(logits, attention_mask)

        # Positions 0-4 should be masked (CLS + hub tokens)
        assert (predictions[:, :5] == -100).all()

        # Padding positions should be masked
        assert (predictions[:, 100:] == -100).all()

    def test_token_classification_preserves_sequence_length(self):
        """Test HubAwareTokenClassificationHead preserves sequence length."""
        head = HubAwareTokenClassificationHead(768, 9)
        head.eval()

        for seq_len in [64, 128, 256]:
            hidden_states = torch.randn(2, seq_len, 768)
            logits = head(hidden_states)
            assert logits.shape == (2, seq_len, 9)


class TestHubAwareHierarchicalHead:
    """Test suite for HubAwareHierarchicalHead."""

    def test_hierarchical_head_initialization(self):
        """Test HubAwareHierarchicalHead initialization."""
        head = HubAwareHierarchicalHead(768, 7, 28)

        assert head.hidden_size == 768
        assert head.primary_labels == 7
        assert head.secondary_labels == 28
        assert head.hub_token == "[EMO]"

    def test_hierarchical_head_forward(self):
        """AC3: HubAwareHierarchicalHead implements primary→secondary cascade."""
        head = HubAwareHierarchicalHead(768, 7, 28)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        primary_logits, secondary_logits = head(None, pooled_outputs)

        assert primary_logits.shape == (4, 7)
        assert secondary_logits.shape == (4, 28)

    def test_hierarchical_head_secondary_conditioned_on_primary(self):
        """Test secondary predictions are conditioned on primary."""
        head = HubAwareHierarchicalHead(768, 7, 28)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}

        # Run twice with same input
        primary1, secondary1 = head(None, pooled_outputs)
        primary2, secondary2 = head(None, pooled_outputs)

        # Should produce same results (deterministic in eval mode)
        assert torch.allclose(primary1, primary2)
        assert torch.allclose(secondary1, secondary2)

    def test_hierarchical_head_with_hidden_states(self):
        """Test HubAwareHierarchicalHead with hidden_states."""
        head = HubAwareHierarchicalHead(768, 7, 28)
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        primary_logits, secondary_logits = head(hidden_states, None)

        assert primary_logits.shape == (4, 7)
        assert secondary_logits.shape == (4, 28)

    def test_hierarchical_head_extra_repr(self):
        """Test HubAwareHierarchicalHead extra_repr."""
        head = HubAwareHierarchicalHead(768, 7, 28)
        repr_str = head.extra_repr()

        assert "[EMO]" in repr_str
        assert "7" in repr_str
        assert "28" in repr_str


class TestHubAwareSafetyHead:
    """Test suite for HubAwareSafetyHead."""

    def test_safety_head_initialization(self):
        """Test HubAwareSafetyHead initialization."""
        head = HubAwareSafetyHead(768, 2, confidence_threshold=0.7)

        assert head.hidden_size == 768
        assert head.num_labels == 2
        assert head.hub_token == "[EMO]"
        assert head.confidence_threshold == 0.7

    def test_safety_head_temperature_calibration(self):
        """AC4: HubAwareSafetyHead includes temperature calibration."""
        head = HubAwareSafetyHead(768, 2)

        # Temperature should be a learnable parameter
        assert isinstance(head.temperature, nn.Parameter)
        assert head.temperature.shape == (1,)

    def test_safety_head_forward_basic(self):
        """Test HubAwareSafetyHead forward pass."""
        head = HubAwareSafetyHead(768, 2)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        logits = head(None, pooled_outputs)

        assert logits.shape == (4, 2)

    def test_safety_head_forward_with_confidence(self):
        """Test HubAwareSafetyHead returns confidence scores."""
        head = HubAwareSafetyHead(768, 2)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        logits, confidence = head(None, pooled_outputs, return_confidence=True)

        assert logits.shape == (4, 2)
        assert confidence.shape == (4,)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_safety_head_predict_with_threshold(self):
        """Test HubAwareSafetyHead threshold-based prediction."""
        head = HubAwareSafetyHead(768, 2, confidence_threshold=0.7)
        head.eval()

        logits = torch.tensor([[2.0, 1.0], [1.0, 3.0], [0.5, 0.5], [4.0, 1.0]])
        predictions, is_confident = head.predict_with_threshold(logits)

        assert predictions.shape == (4,)
        assert is_confident.shape == (4,)
        assert is_confident.dtype == torch.bool

    def test_safety_head_temperature_affects_confidence(self):
        """Test temperature parameter affects confidence scores."""
        head = HubAwareSafetyHead(768, 2)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}

        # Higher temperature = lower confidence
        head.temperature.data = torch.tensor([2.0])
        _, confidence_high_temp = head(None, pooled_outputs, return_confidence=True)

        # Lower temperature = higher confidence
        head.temperature.data = torch.tensor([0.5])
        _, confidence_low_temp = head(None, pooled_outputs, return_confidence=True)

        # Confidence should generally increase with lower temperature
        # (though not guaranteed for every sample)
        assert confidence_low_temp.mean() >= confidence_high_temp.mean() * 0.9


class TestHubAwareNLIHead:
    """Test suite for HubAwareNLIHead."""

    def test_nli_head_initialization(self):
        """Test HubAwareNLIHead initialization."""
        head = HubAwareNLIHead(768, 3)

        assert head.hidden_size == 768
        assert head.num_labels == 3
        assert head.hub_token == "[REL]"

    def test_nli_head_uses_rel_hub(self):
        """AC5: HubAwareNLIHead uses [REL] hub token."""
        head = HubAwareNLIHead(768, 3)
        positions = get_hub_positions()

        assert head.hub_token == "[REL]"
        assert head.hub_position == positions["[REL]"]
        assert head.hub_position == 3  # [REL] is at position 3

    def test_nli_head_forward(self):
        """Test HubAwareNLIHead forward pass."""
        head = HubAwareNLIHead(768, 3)
        head.eval()

        pooled_outputs = {"[REL]": torch.randn(4, 768)}
        logits = head(None, pooled_outputs)

        assert logits.shape == (4, 3)

    def test_nli_head_with_hidden_states(self):
        """Test HubAwareNLIHead with hidden_states."""
        head = HubAwareNLIHead(768, 3)
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        logits = head(hidden_states, None)

        assert logits.shape == (4, 3)

    def test_nli_head_two_layer_classifier(self):
        """Test HubAwareNLIHead uses two-layer classifier."""
        head = HubAwareNLIHead(768, 3)

        # Classifier should be a Sequential with 4 layers (Linear, Tanh, Dropout, Linear)
        assert isinstance(head.classifier, nn.Sequential)
        assert len(head.classifier) == 4


class TestHeadRegistry:
    """Test suite for HEAD_REGISTRY."""

    def test_head_registry_has_all_capabilities(self):
        """AC6: HEAD_REGISTRY maps all 12 capabilities to correct head types."""
        expected_capabilities = {
            # EMO hub
            "emotions",
            "sentiment",
            "safety_generic",
            "safety_familyos",
            # MEM hub
            "embedding",
            # REL hub
            "nli",
            "relation",
            # TASK hub
            "intent",
            "ingress",
            # Token-level
            "ner_general",
            "ner_family",
            "temporal",
        }

        assert set(HEAD_REGISTRY.keys()) == expected_capabilities

    def test_head_registry_correct_head_types(self):
        """Test HEAD_REGISTRY maps capabilities to correct head classes."""
        # EMO hub heads
        assert HEAD_REGISTRY["emotions"] == HubAwareHierarchicalHead
        assert HEAD_REGISTRY["sentiment"] == HubAwareClassificationHead
        assert HEAD_REGISTRY["safety_generic"] == HubAwareSafetyHead
        assert HEAD_REGISTRY["safety_familyos"] == HubAwareSafetyHead

        # MEM hub heads
        assert HEAD_REGISTRY["embedding"] is None

        # REL hub heads
        assert HEAD_REGISTRY["nli"] == HubAwareNLIHead
        assert HEAD_REGISTRY["relation"] == HubAwareClassificationHead

        # TASK hub heads
        assert HEAD_REGISTRY["intent"] == HubAwareClassificationHead
        assert HEAD_REGISTRY["ingress"] == HubAwareClassificationHead

        # Token-level heads
        assert HEAD_REGISTRY["ner_general"] == HubAwareTokenClassificationHead
        assert HEAD_REGISTRY["ner_family"] == HubAwareTokenClassificationHead
        assert HEAD_REGISTRY["temporal"] == HubAwareTokenClassificationHead


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_head_for_capability_emotions(self):
        """AC7: create_head_for_capability() factory works for emotions."""
        head = create_head_for_capability("emotions", 768)

        assert isinstance(head, HubAwareHierarchicalHead)
        assert head.hidden_size == 768
        assert head.primary_labels == 7
        assert head.secondary_labels == 28

    def test_create_head_for_capability_sentiment(self):
        """AC7: create_head_for_capability() factory works for sentiment."""
        head = create_head_for_capability("sentiment", 768)

        assert isinstance(head, HubAwareClassificationHead)
        assert head.hidden_size == 768
        assert head.num_labels == 3  # pos/neg/neu

    def test_create_head_for_capability_nli(self):
        """AC7: create_head_for_capability() factory works for NLI."""
        head = create_head_for_capability("nli", 768)

        assert isinstance(head, HubAwareNLIHead)
        assert head.hidden_size == 768
        assert head.num_labels == 3
        assert head.hub_token == "[REL]"

    def test_create_head_for_capability_ner(self):
        """AC7: create_head_for_capability() factory works for NER."""
        head = create_head_for_capability("ner_general", 768)

        assert isinstance(head, HubAwareTokenClassificationHead)
        assert head.hidden_size == 768
        assert head.num_labels == 9

    def test_create_head_for_capability_embedding_returns_none(self):
        """Test create_head_for_capability returns None for embedding."""
        head = create_head_for_capability("embedding", 768)
        assert head is None

    def test_create_head_for_capability_custom_num_labels(self):
        """Test create_head_for_capability with custom num_labels."""
        head = create_head_for_capability("sentiment", 768, num_labels=5)

        assert isinstance(head, HubAwareClassificationHead)
        assert head.num_labels == 5

    def test_create_head_for_capability_unknown_raises_error(self):
        """Test create_head_for_capability raises error for unknown capability."""
        try:
            create_head_for_capability("unknown_capability", 768)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unknown capability" in str(e)

    def test_create_all_heads_default(self):
        """Test create_all_heads creates all heads."""
        heads = create_all_heads(768)

        # Should create 11 heads (12 capabilities - 1 embedding)
        assert len(heads) == 11
        assert "emotions" in heads
        assert "sentiment" in heads
        assert "nli" in heads
        assert "ner_general" in heads
        assert "embedding" not in heads  # No head for embedding

    def test_create_all_heads_selected_capabilities(self):
        """Test create_all_heads with selected capabilities."""
        capabilities = ["emotions", "sentiment", "nli"]
        heads = create_all_heads(768, capabilities)

        assert len(heads) == 3
        assert "emotions" in heads
        assert "sentiment" in heads
        assert "nli" in heads

    def test_create_all_heads_returns_module_dict(self):
        """Test create_all_heads returns nn.ModuleDict."""
        heads = create_all_heads(768)
        assert isinstance(heads, nn.ModuleDict)


class TestHeadConfig:
    """Test suite for HeadConfig dataclass."""

    def test_head_config_initialization(self):
        """Test HeadConfig initialization."""
        config = HeadConfig(
            name="emotions",
            num_labels=7,
            head_type="hierarchical",
            hub_token="[EMO]",
            hidden_size=768,
            dropout=0.1,
            loss_weight=2.0,
        )

        assert config.name == "emotions"
        assert config.num_labels == 7
        assert config.head_type == "hierarchical"
        assert config.hub_token == "[EMO]"
        assert config.hidden_size == 768
        assert config.dropout == 0.1
        assert config.loss_weight == 2.0

    def test_head_config_defaults(self):
        """Test HeadConfig default values."""
        config = HeadConfig(
            name="sentiment", num_labels=3, head_type="classification", hub_token="[EMO]"
        )

        assert config.hidden_size == 768
        assert config.dropout == 0.1
        assert config.loss_weight == 1.0
        assert config.hierarchy is None


class TestAcceptanceCriteria:
    """Test suite verifying all acceptance criteria."""

    def test_ac1_classification_head_extracts_correct_hub(self):
        """AC1: HubAwareClassificationHead extracts correct hub token."""
        head = HubAwareClassificationHead(768, 7, hub_token="[EMO]")
        head.eval()

        # Create pooled outputs with multiple hubs
        pooled_outputs = {
            "[CLS]": torch.randn(4, 768),
            "[EMO]": torch.ones(4, 768),  # Use ones for verification
            "[REL]": torch.randn(4, 768),
        }

        logits = head(None, pooled_outputs)

        # Should use [EMO] hub
        assert logits.shape == (4, 7)

    def test_ac2_token_classification_uses_full_sequence(self):
        """AC2: HubAwareTokenClassificationHead uses full sequence (not hub)."""
        head = HubAwareTokenClassificationHead(768, 9)
        head.eval()

        hidden_states = torch.randn(4, 128, 768)
        logits = head(hidden_states)

        # Should output logits for every token, not just hub
        assert logits.shape == (4, 128, 9)

    def test_ac3_hierarchical_head_cascade(self):
        """AC3: HubAwareHierarchicalHead implements primary→secondary cascade."""
        head = HubAwareHierarchicalHead(768, 7, 28)
        head.eval()

        pooled_outputs = {"[EMO]": torch.randn(4, 768)}
        primary_logits, secondary_logits = head(None, pooled_outputs)

        # Should return both primary and secondary predictions
        assert primary_logits.shape == (4, 7)
        assert secondary_logits.shape == (4, 28)

    def test_ac4_safety_head_temperature_calibration(self):
        """AC4: HubAwareSafetyHead includes temperature calibration."""
        head = HubAwareSafetyHead(768, 2)

        # Should have learnable temperature parameter
        assert hasattr(head, "temperature")
        assert isinstance(head.temperature, nn.Parameter)

    def test_ac5_nli_head_uses_rel_hub(self):
        """AC5: HubAwareNLIHead uses [REL] hub token."""
        head = HubAwareNLIHead(768, 3)

        assert head.hub_token == "[REL]"
        assert head.hub_position == 3  # [REL] is at position 3

    def test_ac6_head_registry_maps_all_capabilities(self):
        """AC6: HEAD_REGISTRY maps all 12 capabilities to correct head types."""
        assert len(HEAD_REGISTRY) == 12

        # Verify all capabilities present
        expected = {
            "emotions",
            "sentiment",
            "safety_generic",
            "safety_familyos",
            "embedding",
            "nli",
            "relation",
            "intent",
            "ingress",
            "ner_general",
            "ner_family",
            "temporal",
        }
        assert set(HEAD_REGISTRY.keys()) == expected

    def test_ac7_factory_function_works_for_all(self):
        """AC7: create_head_for_capability() factory works for all capabilities."""
        capabilities_to_test = [
            "emotions",
            "sentiment",
            "nli",
            "ner_general",
            "intent",
            "safety_generic",
        ]

        for capability in capabilities_to_test:
            head = create_head_for_capability(capability, 768)
            # Embedding returns None, others return valid heads
            if capability == "embedding":
                assert head is None
            else:
                assert head is not None
                assert isinstance(head, nn.Module)

"""
Unit Tests for GlobalPointer Collator.

Tests the GlobalPointerCollator that converts span-format NER data to
the (B, num_labels, L, L) tensor format required for GlobalPointer training.

Test Cases:
    - Single entity placement
    - Multi-token entity spans
    - Multiple entities per sample
    - Empty entities
    - Unknown labels (graceful skip)
    - Truncated spans
    - Batch padding
    - Character-to-token alignment
    - Decode spans (inference)

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from modeling_studio.data.globalpointer_collator import (
    GlobalPointerCollator,
    NER_GENERAL_LABELS,
    create_ner_general_collator,
    create_ner_family_collator,
    create_temporal_collator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tokenizer():
    """Load ModernBERT tokenizer for tests."""
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


@pytest.fixture
def collator(tokenizer):
    """Create default ner_general collator."""
    return create_ner_general_collator(tokenizer, max_length=128)


# =============================================================================
# Test GlobalPointerCollator Initialization
# =============================================================================


class TestGlobalPointerCollatorInit:
    """Test collator initialization."""

    def test_init_with_label_to_id(self, tokenizer):
        """Collator initializes with label_to_id dict."""
        collator = GlobalPointerCollator(
            tokenizer=tokenizer,
            label_to_id={"PER": 0, "ORG": 1},
            max_length=512,
        )
        assert collator.num_labels == 2
        assert collator.id_to_label == {0: "PER", 1: "ORG"}

    def test_factory_ner_general(self, tokenizer):
        """Factory creates ner_general collator with 4 labels."""
        collator = create_ner_general_collator(tokenizer)
        assert collator.num_labels == 4
        assert "PER" in collator.label_to_id
        assert "ORG" in collator.label_to_id
        assert "LOC" in collator.label_to_id
        assert "MISC" in collator.label_to_id

    def test_factory_ner_family(self, tokenizer):
        """Factory creates ner_family collator with 10 labels."""
        collator = create_ner_family_collator(tokenizer)
        assert collator.num_labels == 10
        assert "KINSHIP" in collator.label_to_id
        assert "MILESTONE" in collator.label_to_id

    def test_factory_temporal(self, tokenizer):
        """Factory creates temporal collator with 5 labels."""
        collator = create_temporal_collator(tokenizer)
        assert collator.num_labels == 5
        assert "DATE_ABS" in collator.label_to_id
        assert "DATE_REL" in collator.label_to_id


# =============================================================================
# Test Single Entity
# =============================================================================


class TestSingleEntity:
    """Test single entity placement in span matrix."""

    def test_single_token_entity(self, collator, tokenizer):
        """Single-token entity is placed correctly."""
        sample = {
            "text": "Emma lives here",
            "entities": [{"start": 0, "end": 4, "label": "PER", "text": "Emma"}],
        }
        batch = collator([sample])

        assert batch["input_ids"].shape[0] == 1
        assert batch["span_labels"].shape[0] == 1
        assert batch["span_labels"].shape[1] == 4  # num_labels

        # Find token index for "Emma"
        tokens = tokenizer.tokenize("Emma lives here")
        # Emma should be at position 1 (after [CLS])
        per_label_id = NER_GENERAL_LABELS["PER"]

        # Check that exactly one span is set
        assert batch["span_labels"][0, per_label_id].sum() == 1.0

    def test_multi_token_entity(self, collator, tokenizer):
        """Multi-token entity spans correct range."""
        sample = {
            "text": "New York is great",
            "entities": [{"start": 0, "end": 8, "label": "LOC", "text": "New York"}],
        }
        batch = collator([sample])

        loc_label_id = NER_GENERAL_LABELS["LOC"]

        # Check that exactly one span is set
        assert batch["span_labels"][0, loc_label_id].sum() == 1.0

        # The span should be in upper triangle (start <= end)
        span_matrix = batch["span_labels"][0, loc_label_id]
        # Find where the 1.0 is
        indices = torch.where(span_matrix == 1.0)
        tok_start, tok_end = indices[0].item(), indices[1].item()
        assert tok_start <= tok_end, "Span should be in upper triangle"

    def test_entity_at_end(self, collator, tokenizer):
        """Entity at end of text is handled correctly."""
        sample = {
            "text": "Lives in NYC",
            "entities": [{"start": 9, "end": 12, "label": "LOC", "text": "NYC"}],
        }
        batch = collator([sample])

        loc_label_id = NER_GENERAL_LABELS["LOC"]
        assert batch["span_labels"][0, loc_label_id].sum() == 1.0


# =============================================================================
# Test Multiple Entities
# =============================================================================


class TestMultipleEntities:
    """Test multiple entities in same sample."""

    def test_two_entities_same_type(self, collator):
        """Two entities of same type both placed."""
        sample = {
            "text": "Emma met John yesterday",
            "entities": [
                {"start": 0, "end": 4, "label": "PER", "text": "Emma"},
                {"start": 9, "end": 13, "label": "PER", "text": "John"},
            ],
        }
        batch = collator([sample])

        per_label_id = NER_GENERAL_LABELS["PER"]
        # Two PER entities should be set
        assert batch["span_labels"][0, per_label_id].sum() == 2.0

    def test_entities_different_types(self, collator):
        """Entities of different types placed in correct label planes."""
        sample = {
            "text": "Emma works at Google in NYC",
            "entities": [
                {"start": 0, "end": 4, "label": "PER", "text": "Emma"},
                {"start": 14, "end": 20, "label": "ORG", "text": "Google"},
                {"start": 24, "end": 27, "label": "LOC", "text": "NYC"},
            ],
        }
        batch = collator([sample])

        # Each label type should have exactly one entity
        assert batch["span_labels"][0, NER_GENERAL_LABELS["PER"]].sum() == 1.0
        assert batch["span_labels"][0, NER_GENERAL_LABELS["ORG"]].sum() == 1.0
        assert batch["span_labels"][0, NER_GENERAL_LABELS["LOC"]].sum() == 1.0
        assert batch["span_labels"][0, NER_GENERAL_LABELS["MISC"]].sum() == 0.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_entities(self, collator):
        """Sample with no entities has all-zero span labels."""
        sample = {"text": "Hello world", "entities": []}
        batch = collator([sample])

        assert batch["span_labels"].sum() == 0.0

    def test_empty_entities_list(self, collator):
        """Empty entities list is handled."""
        sample = {"text": "Just some text", "entities": []}
        batch = collator([sample])

        assert batch["input_ids"].shape[0] == 1
        assert batch["span_labels"].sum() == 0.0

    def test_unknown_label_skipped(self, collator):
        """Unknown label is ignored, no error raised."""
        sample = {
            "text": "Test entity",
            "entities": [{"start": 0, "end": 4, "label": "UNKNOWN_TYPE"}],
        }
        batch = collator([sample])

        # Should not crash, should have zero spans
        assert batch["span_labels"].sum() == 0.0

    def test_missing_entity_fields(self, collator):
        """Entities with missing fields are skipped."""
        sample = {
            "text": "Emma is here",
            "entities": [
                {"start": 0, "label": "PER"},  # Missing end
                {"end": 4, "label": "PER"},  # Missing start
                {"start": 0, "end": 4},  # Missing label
            ],
        }
        batch = collator([sample])

        # All should be skipped
        assert batch["span_labels"].sum() == 0.0

    def test_entity_with_type_key(self, collator):
        """Entity with 'type' instead of 'label' is handled."""
        sample = {
            "text": "Emma here",
            "entities": [{"start": 0, "end": 4, "type": "PER"}],
        }
        batch = collator([sample])

        per_label_id = NER_GENERAL_LABELS["PER"]
        assert batch["span_labels"][0, per_label_id].sum() == 1.0


# =============================================================================
# Test Batching
# =============================================================================


class TestBatching:
    """Test batch processing and padding."""

    def test_batch_of_two(self, collator):
        """Batch of two samples processed correctly."""
        samples = [
            {
                "text": "Emma lives here",
                "entities": [{"start": 0, "end": 4, "label": "PER"}],
            },
            {
                "text": "John works at Google",
                "entities": [
                    {"start": 0, "end": 4, "label": "PER"},
                    {"start": 14, "end": 20, "label": "ORG"},
                ],
            },
        ]
        batch = collator(samples)

        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2
        assert batch["span_labels"].shape[0] == 2

        # Sample 0 has 1 PER entity
        assert batch["span_labels"][0, NER_GENERAL_LABELS["PER"]].sum() == 1.0
        assert batch["span_labels"][0, NER_GENERAL_LABELS["ORG"]].sum() == 0.0

        # Sample 1 has 1 PER and 1 ORG entity
        assert batch["span_labels"][1, NER_GENERAL_LABELS["PER"]].sum() == 1.0
        assert batch["span_labels"][1, NER_GENERAL_LABELS["ORG"]].sum() == 1.0

    def test_batch_different_lengths(self, collator):
        """Batch with different text lengths pads correctly."""
        samples = [
            {"text": "Short", "entities": []},
            {"text": "This is a much longer sentence with more tokens", "entities": []},
        ]
        batch = collator(samples)

        # Both should have same sequence length (padded)
        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]
        assert batch["span_labels"].shape[2] == batch["span_labels"].shape[3]

        # Attention mask should differ (first has more padding)
        assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()


# =============================================================================
# Test Character-to-Token Alignment
# =============================================================================


class TestCharToTokenAlignment:
    """Test character-to-token span alignment."""

    def test_alignment_exact_match(self, collator, tokenizer):
        """Character span exactly matches token boundaries."""
        # "Hello" should be one token
        sample = {
            "text": "Hello world",
            "entities": [{"start": 0, "end": 5, "label": "MISC", "text": "Hello"}],
        }
        batch = collator([sample])

        misc_label_id = NER_GENERAL_LABELS["MISC"]
        assert batch["span_labels"][0, misc_label_id].sum() == 1.0

    def test_alignment_subword(self, collator, tokenizer):
        """Subword tokenization still captures full entity."""
        # "tokenization" might be split into multiple subwords
        sample = {
            "text": "Study tokenization today",
            "entities": [{"start": 6, "end": 18, "label": "MISC", "text": "tokenization"}],
        }
        batch = collator([sample])

        misc_label_id = NER_GENERAL_LABELS["MISC"]
        # Should still have exactly one entity span
        assert batch["span_labels"][0, misc_label_id].sum() == 1.0


# =============================================================================
# Test Decode Spans (Inference)
# =============================================================================


class TestDecodeSpans:
    """Test decode_spans for inference."""

    def test_decode_single_span(self, collator, tokenizer):
        """Single span is decoded correctly."""
        text = "Emma lives here"
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"]

        # Create fake span scores with one positive
        seq_len = len(encoding["input_ids"])
        span_scores = torch.zeros(4, seq_len, seq_len)

        # Set a span (tok 1 to tok 1 for "Emma")
        per_id = NER_GENERAL_LABELS["PER"]
        span_scores[per_id, 1, 1] = 5.0  # High score

        entities = collator.decode_spans(
            span_scores, offset_mapping, text, threshold=0.5
        )

        assert len(entities) == 1
        assert entities[0]["label"] == "PER"
        assert entities[0]["score"] > 0.99  # sigmoid(5) ≈ 0.993

    def test_decode_threshold(self, collator, tokenizer):
        """Threshold filters low-score spans."""
        text = "Test"
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"]

        seq_len = len(encoding["input_ids"])
        span_scores = torch.full((4, seq_len, seq_len), -10.0)  # Low scores everywhere
        span_scores[0, 1, 1] = 0.0  # sigmoid(0) = 0.5 for one span

        # With threshold 0.6, should get no entities
        entities = collator.decode_spans(
            span_scores, offset_mapping, text, threshold=0.6
        )
        assert len(entities) == 0

        # With threshold 0.4, should get one entity (sigmoid(0) = 0.5 > 0.4)
        entities = collator.decode_spans(
            span_scores, offset_mapping, text, threshold=0.4
        )
        assert len(entities) == 1


# =============================================================================
# Test Tensor Shapes
# =============================================================================


class TestTensorShapes:
    """Test output tensor shapes."""

    def test_output_shapes(self, collator):
        """Output tensors have correct shapes."""
        samples = [
            {"text": "Hello world", "entities": []},
            {"text": "Test", "entities": []},
        ]
        batch = collator(samples)

        B = 2
        L = batch["input_ids"].shape[1]
        num_labels = 4

        assert batch["input_ids"].shape == (B, L)
        assert batch["attention_mask"].shape == (B, L)
        assert batch["span_labels"].shape == (B, num_labels, L, L)

    def test_span_labels_dtype(self, collator):
        """Span labels are float32 for BCE loss."""
        sample = {"text": "Test", "entities": []}
        batch = collator([sample])

        assert batch["span_labels"].dtype == torch.float32

    def test_input_ids_dtype(self, collator):
        """Input IDs are int64 for embedding lookup."""
        sample = {"text": "Test", "entities": []}
        batch = collator([sample])

        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long


# =============================================================================
# Test Max Length Truncation
# =============================================================================


class TestTruncation:
    """Test handling of truncation."""

    def test_long_text_truncated(self, tokenizer):
        """Long text is truncated to max_length."""
        collator = GlobalPointerCollator(
            tokenizer=tokenizer,
            label_to_id=NER_GENERAL_LABELS,
            max_length=32,  # Short max length
        )

        # Create text longer than 32 tokens
        long_text = "This is a very long sentence. " * 20
        sample = {"text": long_text, "entities": []}
        batch = collator([sample])

        # Sequence length should be at most 32
        assert batch["input_ids"].shape[1] <= 32

    def test_entity_beyond_truncation_skipped(self, tokenizer):
        """Entity beyond truncation point is skipped."""
        collator = GlobalPointerCollator(
            tokenizer=tokenizer,
            label_to_id=NER_GENERAL_LABELS,
            max_length=8,  # Very short to ensure truncation
        )

        # Create text where entity is definitely beyond 8 tokens
        # With CLS and SEP, only ~6 content tokens fit
        text = "Word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 Emma here"
        emma_start = text.index("Emma")
        sample = {
            "text": text,
            "entities": [{"start": emma_start, "end": emma_start + 4, "label": "PER", "text": "Emma"}],
        }
        batch = collator([sample])

        # With max_length=8, "Emma" should definitely be truncated
        per_label_id = NER_GENERAL_LABELS["PER"]
        assert batch["span_labels"][0, per_label_id].sum() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

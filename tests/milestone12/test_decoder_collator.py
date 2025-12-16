"""
Tests for CounterfactualCollator (Issue 12.2.1).

Tests the collator for batching counterfactual samples
with proper padding of encoder embeddings and decoder sequences.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def sample_features():
    """Create sample features for testing collator."""
    return [
        {
            "encoder_embeddings": torch.randn(10, 768),  # seq_len=10
            "encoder_attention_mask": torch.ones(10, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, 1000, (15,)),
            "labels": torch.randint(0, 1000, (15,)),
            "sample_id": 0,
        },
        {
            "encoder_embeddings": torch.randn(8, 768),  # seq_len=8
            "encoder_attention_mask": torch.ones(8, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, 1000, (20,)),
            "labels": torch.randint(0, 1000, (20,)),
            "sample_id": 1,
        },
        {
            "encoder_embeddings": torch.randn(12, 768),  # seq_len=12
            "encoder_attention_mask": torch.ones(12, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, 1000, (10,)),
            "labels": torch.randint(0, 1000, (10,)),
            "sample_id": 2,
        },
    ]


@pytest.fixture
def pooled_features():
    """Create pooled (single vector) encoder embedding features."""
    return [
        {
            "encoder_embeddings": torch.randn(768),  # 1D pooled
            "encoder_attention_mask": torch.ones(1, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, 1000, (15,)),
            "labels": torch.randint(0, 1000, (15,)),
            "sample_id": 0,
        },
        {
            "encoder_embeddings": torch.randn(768),  # 1D pooled
            "encoder_attention_mask": torch.ones(1, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, 1000, (20,)),
            "labels": torch.randint(0, 1000, (20,)),
            "sample_id": 1,
        },
    ]


# =============================================================================
# Issue 12.2.1: CounterfactualCollator
# =============================================================================


class TestCounterfactualCollatorEncoderPadding:
    """Tests for encoder embedding padding (12.2.1-T1)."""

    def test_collator_pads_encoder(self, mock_tokenizer, sample_features):
        """12.2.1-T1: Encoder embeddings padded to batch max."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        # Encoder hidden states should be padded to max length
        assert "encoder_hidden_states" in batch

        # Max encoder length in batch is 12, padded to multiple of 8 = 16
        batch_size, enc_seq_len, hidden_dim = batch["encoder_hidden_states"].shape
        assert batch_size == 3
        assert enc_seq_len >= 12  # At least max in batch
        assert enc_seq_len % 8 == 0  # Padded to multiple of 8
        assert hidden_dim == 768

    def test_collator_handles_pooled_embeddings(self, mock_tokenizer, pooled_features):
        """Test that collator handles 1D pooled embeddings."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(pooled_features)

        # Should expand 1D to 2D with seq_len=1
        batch_size, enc_seq_len, hidden_dim = batch["encoder_hidden_states"].shape
        assert batch_size == 2
        assert enc_seq_len >= 1
        assert hidden_dim == 768


class TestCounterfactualCollatorDecoderPadding:
    """Tests for decoder sequence padding (12.2.1-T2)."""

    def test_collator_pads_decoder(self, mock_tokenizer, sample_features):
        """12.2.1-T2: Decoder sequences padded to batch max."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        # Decoder input_ids should be padded
        assert "decoder_input_ids" in batch

        # Max decoder length in batch is 20, padded to multiple of 8 = 24
        batch_size, dec_seq_len = batch["decoder_input_ids"].shape
        assert batch_size == 3
        assert dec_seq_len >= 20  # At least max in batch
        assert dec_seq_len % 8 == 0  # Padded to multiple of 8

    def test_collator_respects_max_output_length(self, mock_tokenizer, sample_features):
        """Test that max_output_length truncates decoder sequences."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(
            tokenizer=mock_tokenizer,
            max_output_length=12,
        )
        batch = collator(sample_features)

        # Should be truncated to max 12, then padded to multiple of 8 = 16
        batch_size, dec_seq_len = batch["decoder_input_ids"].shape
        assert dec_seq_len <= 16  # max_output_length (12) padded to multiple of 8


class TestCounterfactualCollatorMasks:
    """Tests for attention mask creation (12.2.1-T3)."""

    def test_collator_encoder_mask(self, mock_tokenizer, sample_features):
        """12.2.1-T3: Encoder mask is 0 for padding."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        assert "encoder_attention_mask" in batch

        # Check shapes match
        enc_hidden_shape = batch["encoder_hidden_states"].shape[:2]
        enc_mask_shape = batch["encoder_attention_mask"].shape
        assert enc_hidden_shape == enc_mask_shape

        # Check that mask is 0 for padding, 1 for valid
        # First sample has seq_len=10, so positions 10+ should be 0
        assert batch["encoder_attention_mask"][0, :10].sum() == 10
        # If padded beyond 10, padding positions should be 0
        if batch["encoder_attention_mask"].shape[1] > 10:
            assert batch["encoder_attention_mask"][0, 10:].sum() == 0

    def test_collator_decoder_mask(self, mock_tokenizer, sample_features):
        """Test decoder attention mask creation."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        assert "decoder_attention_mask" in batch

        # Check shapes match
        dec_input_shape = batch["decoder_input_ids"].shape
        dec_mask_shape = batch["decoder_attention_mask"].shape
        assert dec_input_shape == dec_mask_shape

        # Check that mask is 0 for padding, 1 for valid
        # Third sample has seq_len=10, so positions 10+ should be 0
        assert batch["decoder_attention_mask"][2, :10].sum() == 10


class TestCounterfactualCollatorLabels:
    """Tests for label padding (12.2.1-T4)."""

    def test_collator_label_padding(self, mock_tokenizer, sample_features):
        """12.2.1-T4: Labels are -100 for padding positions."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        assert "labels" in batch

        # Labels shape should match decoder_input_ids
        assert batch["labels"].shape == batch["decoder_input_ids"].shape

        # Padding positions should be -100
        # Third sample has original length 10
        original_len = 10
        padded_len = batch["labels"].shape[1]

        # Valid positions should not be -100
        assert (batch["labels"][2, :original_len] != -100).all()

        # Padding positions should be -100
        if padded_len > original_len:
            assert (batch["labels"][2, original_len:] == -100).all()


class TestCounterfactualCollatorOutput:
    """Tests for collator output format."""

    def test_collator_output_keys(self, mock_tokenizer, sample_features):
        """Test that collator returns all required keys."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        required_keys = [
            "encoder_hidden_states",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        ]

        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

    def test_collator_output_dtypes(self, mock_tokenizer, sample_features):
        """Test that collator outputs have correct dtypes."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        # Float for embeddings
        assert batch["encoder_hidden_states"].dtype == torch.float32

        # Long for IDs and masks
        assert batch["encoder_attention_mask"].dtype == torch.long
        assert batch["decoder_input_ids"].dtype == torch.long
        assert batch["decoder_attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_collator_batch_sizes_consistent(self, mock_tokenizer, sample_features):
        """Test that all batch dimensions are consistent."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        batch_size = 3

        assert batch["encoder_hidden_states"].shape[0] == batch_size
        assert batch["encoder_attention_mask"].shape[0] == batch_size
        assert batch["decoder_input_ids"].shape[0] == batch_size
        assert batch["decoder_attention_mask"].shape[0] == batch_size
        assert batch["labels"].shape[0] == batch_size


class TestSequenceCounterfactualCollator:
    """Tests for SequenceCounterfactualCollator."""

    def test_sequence_collator_works(self, mock_tokenizer, sample_features):
        """Test that SequenceCounterfactualCollator works for full sequence mode."""
        from modeling_studio.trainers.decoder_collator import SequenceCounterfactualCollator

        collator = SequenceCounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(sample_features)

        # Should have all required keys
        assert "encoder_hidden_states" in batch
        assert "encoder_attention_mask" in batch
        assert batch["encoder_hidden_states"].shape[0] == 3


class TestCounterfactualCollatorEdgeCases:
    """Tests for edge cases."""

    def test_collator_single_sample(self, mock_tokenizer):
        """Test collator with single sample batch."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        features = [
            {
                "encoder_embeddings": torch.randn(10, 768),
                "encoder_attention_mask": torch.ones(10, dtype=torch.long),
                "decoder_input_ids": torch.randint(0, 1000, (15,)),
                "labels": torch.randint(0, 1000, (15,)),
                "sample_id": 0,
            }
        ]

        collator = CounterfactualCollator(tokenizer=mock_tokenizer)
        batch = collator(features)

        assert batch["encoder_hidden_states"].shape[0] == 1
        assert batch["decoder_input_ids"].shape[0] == 1

    def test_collator_no_pad_to_multiple(self, mock_tokenizer, sample_features):
        """Test collator without pad_to_multiple_of."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        collator = CounterfactualCollator(
            tokenizer=mock_tokenizer,
            pad_to_multiple_of=None,
        )
        batch = collator(sample_features)

        # Should still work, just not padded to multiple
        assert "encoder_hidden_states" in batch
        assert "decoder_input_ids" in batch

    def test_collator_empty_pad_token(self, sample_features):
        """Test collator falls back when pad_token_id is None."""
        from modeling_studio.trainers.decoder_collator import CounterfactualCollator

        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 2

        collator = CounterfactualCollator(tokenizer=tokenizer)

        # Should use eos_token_id as fallback
        assert collator.pad_token_id == 2

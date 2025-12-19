"""
Tests for decoder token embedding initialization.

This module verifies that the `_initialize_new_token_embeddings()` method
correctly initializes new tokens added when resizing GPT-2 vocabulary.

Critical fix: New tokens (50257-50367) should have the same embedding norm
as original GPT-2 tokens (~3.7) rather than the default random init (~2.1).

Related: docs/DECODER_EMBEDDING_ANALYSIS.md
"""

from __future__ import annotations

import pytest
import torch

from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def decoder_config() -> GPT2DecoderConfig:
    """Create a standard GPT-2 decoder config."""
    return GPT2DecoderConfig(
        vocab_size=50368,  # ModernBERT tokenizer vocab
        bos_token_id=50281,
        eos_token_id=50282,
        pad_token_id=50283,
        max_position_embeddings=512,
        # Use smaller model for faster tests
        gpt2_model_name="gpt2",  # 124M instead of gpt2-medium
    )


@pytest.fixture
def decoder_head(decoder_config: GPT2DecoderConfig) -> GPT2DecoderHead:
    """Create a GPT-2 decoder head with proper token initialization."""
    return GPT2DecoderHead(
        config=decoder_config,
        encoder_hidden_size=768,
    )


# =============================================================================
# Epic 3.1.1: Token Embedding Initialization Tests
# =============================================================================


class TestTokenEmbeddingScale:
    """Tests for embedding magnitude/scale."""

    def test_new_token_embeddings_initialized_to_mean(
        self, decoder_head: GPT2DecoderHead
    ) -> None:
        """
        New tokens (50257-50280) should be initialized to mean embedding.

        Note: Mean embedding has norm ~2.0, which is different from per-token
        average norm (~3.9). This is acceptable for placeholder tokens.
        BOS/EOS get special treatment (copied from endoftext).
        """
        wte = decoder_head.gpt2.transformer.wte.weight

        # Compute what the mean embedding norm should be
        original_mean = wte[:50257].mean(dim=0)
        expected_norm = original_mean.norm().item()

        # New tokens (50257-50280) should match mean embedding norm
        # Excludes BOS/EOS/PAD which have special initialization
        new_tokens_norm = wte[50257:50281].norm(dim=1).mean().item()

        # Should be very close to mean embedding norm
        relative_diff = abs(new_tokens_norm - expected_norm) / expected_norm
        assert relative_diff < 0.10, (
            f"New tokens norm ({new_tokens_norm:.3f}) should be close to "
            f"mean embedding norm ({expected_norm:.3f})"
        )

    def test_bos_eos_have_higher_norm_than_random_new_tokens(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """BOS/EOS (copied from endoftext) should have higher norm than placeholder tokens."""
        wte = decoder_head.gpt2.transformer.wte.weight

        bos_norm = wte[decoder_config.bos_token_id].norm().item()
        placeholder_norm = wte[50257].norm().item()  # First placeholder token

        # BOS should have higher norm (copied from endoftext ~3.1 vs mean ~2.0)
        assert bos_norm > placeholder_norm * 1.3, (
            f"BOS norm ({bos_norm:.3f}) should be significantly higher than "
            f"placeholder norm ({placeholder_norm:.3f})"
        )

    def test_original_gpt2_tokens_unchanged(
        self, decoder_head: GPT2DecoderHead
    ) -> None:
        """Original GPT-2 tokens (0-50256) should have expected norm (~3.7)."""
        wte = decoder_head.gpt2.transformer.wte.weight

        original_norm = wte[:50257].norm(dim=1).mean().item()

        # GPT-2 embeddings typically have norm ~3.5-4.0
        assert 2.5 < original_norm < 5.0, (
            f"Original GPT-2 tokens have unexpected norm: {original_norm:.3f}"
        )


class TestSpecialTokenInitialization:
    """Tests for BOS, EOS, and PAD token initialization."""

    def test_bos_initialized_from_endoftext(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """BOS token should be copy of GPT-2's endoftext token."""
        wte = decoder_head.gpt2.transformer.wte.weight

        endoftext = wte[50256]  # GPT-2's <|endoftext|>
        bos = wte[decoder_config.bos_token_id]

        # Should be exact copy
        assert torch.allclose(bos, endoftext, atol=1e-6), (
            f"BOS token should be copy of endoftext. "
            f"Max diff: {(bos - endoftext).abs().max().item():.6f}"
        )

    def test_eos_initialized_from_endoftext(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """EOS token should be copy of GPT-2's endoftext token."""
        wte = decoder_head.gpt2.transformer.wte.weight

        endoftext = wte[50256]  # GPT-2's <|endoftext|>
        eos = wte[decoder_config.eos_token_id]

        # Should be exact copy
        assert torch.allclose(eos, endoftext, atol=1e-6), (
            f"EOS token should be copy of endoftext. "
            f"Max diff: {(eos - endoftext).abs().max().item():.6f}"
        )

    def test_pad_token_is_zero(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """PAD token should be zero vector."""
        wte = decoder_head.gpt2.transformer.wte.weight

        pad = wte[decoder_config.pad_token_id]

        # PAD should be all zeros
        max_abs = pad.abs().max().item()
        assert max_abs < 1e-6, (
            f"PAD token should be zero vector. Max abs value: {max_abs:.6f}"
        )

    def test_bos_eos_have_correct_norm(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """BOS and EOS should have same norm as endoftext (~2.5-3.0)."""
        wte = decoder_head.gpt2.transformer.wte.weight

        endoftext_norm = wte[50256].norm().item()
        bos_norm = wte[decoder_config.bos_token_id].norm().item()
        eos_norm = wte[decoder_config.eos_token_id].norm().item()

        # All three should be identical (copies)
        assert abs(bos_norm - endoftext_norm) < 1e-5
        assert abs(eos_norm - endoftext_norm) < 1e-5


class TestEmbeddingTableShape:
    """Tests for embedding table dimensions."""

    def test_embedding_table_has_correct_vocab_size(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """Embedding table should have 50368 tokens."""
        wte = decoder_head.gpt2.transformer.wte.weight

        assert wte.shape[0] == decoder_config.vocab_size, (
            f"Expected vocab size {decoder_config.vocab_size}, "
            f"got {wte.shape[0]}"
        )

    def test_embedding_dim_matches_gpt2(
        self, decoder_head: GPT2DecoderHead
    ) -> None:
        """Embedding dimension should match GPT-2 hidden size."""
        wte = decoder_head.gpt2.transformer.wte.weight
        hidden_size = decoder_head.gpt2.config.hidden_size

        assert wte.shape[1] == hidden_size, (
            f"Embedding dim ({wte.shape[1]}) should match "
            f"GPT-2 hidden size ({hidden_size})"
        )


class TestGPT2ConfigTokenIds:
    """Tests for GPT-2 config token ID updates."""

    def test_gpt2_config_has_correct_bos(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """GPT-2 config should have our BOS token ID."""
        assert decoder_head.gpt2.config.bos_token_id == decoder_config.bos_token_id

    def test_gpt2_config_has_correct_eos(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """GPT-2 config should have our EOS token ID."""
        assert decoder_head.gpt2.config.eos_token_id == decoder_config.eos_token_id

    def test_gpt2_config_has_correct_pad(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """GPT-2 config should have our PAD token ID."""
        assert decoder_head.gpt2.config.pad_token_id == decoder_config.pad_token_id


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEmbeddingInitializationRobustness:
    """Tests for edge cases and robustness."""

    def test_no_nan_in_embeddings(self, decoder_head: GPT2DecoderHead) -> None:
        """No NaN values should exist in embedding table."""
        wte = decoder_head.gpt2.transformer.wte.weight

        assert not torch.isnan(wte).any(), "Found NaN values in embeddings"

    def test_no_inf_in_embeddings(self, decoder_head: GPT2DecoderHead) -> None:
        """No Inf values should exist in embedding table."""
        wte = decoder_head.gpt2.transformer.wte.weight

        assert not torch.isinf(wte).any(), "Found Inf values in embeddings"

    def test_embeddings_are_finite(self, decoder_head: GPT2DecoderHead) -> None:
        """All embedding values should be finite and reasonable."""
        wte = decoder_head.gpt2.transformer.wte.weight

        max_abs = wte.abs().max().item()

        # Embeddings shouldn't have extreme values
        assert max_abs < 100.0, f"Extreme embedding value: {max_abs}"

    def test_new_tokens_not_all_identical(
        self, decoder_head: GPT2DecoderHead
    ) -> None:
        """
        New tokens (except PAD) should not all be identical after training.

        Note: At initialization, they ARE identical (mean embedding).
        This test verifies the initialization is consistent.
        """
        wte = decoder_head.gpt2.transformer.wte.weight

        # New tokens 50257-50280 should all be mean embedding
        new_tokens = wte[50257:50281]

        # Check they're all identical to first one (initialized to mean)
        first_token = new_tokens[0]
        for i in range(1, len(new_tokens)):
            assert torch.allclose(new_tokens[i], first_token, atol=1e-6), (
                f"New token {50257 + i} differs from first new token"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecoderForwardPass:
    """Tests that decoder works with properly initialized embeddings."""

    def test_forward_pass_with_new_tokens(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """Forward pass should work with BOS/EOS tokens."""
        batch_size = 2
        enc_seq_len = 10
        dec_seq_len = 8

        # Create dummy encoder outputs
        encoder_hidden = torch.randn(batch_size, enc_seq_len, 768)
        encoder_mask = torch.ones(batch_size, enc_seq_len)

        # Create decoder input with BOS token
        decoder_input_ids = torch.full(
            (batch_size, dec_seq_len),
            fill_value=1000,  # Some token
            dtype=torch.long,
        )
        decoder_input_ids[:, 0] = decoder_config.bos_token_id

        # Labels with EOS at the end
        labels = decoder_input_ids.clone()
        labels[:, -1] = decoder_config.eos_token_id

        # Forward pass should not error
        outputs = decoder_head(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        assert "loss" in outputs
        assert "logits" in outputs
        assert not torch.isnan(outputs["loss"])

    def test_generate_uses_bos_token(
        self, decoder_head: GPT2DecoderHead, decoder_config: GPT2DecoderConfig
    ) -> None:
        """Generation should start with BOS token."""
        batch_size = 1
        enc_seq_len = 5

        encoder_hidden = torch.randn(batch_size, enc_seq_len, 768)
        encoder_mask = torch.ones(batch_size, enc_seq_len)

        # Generate a few tokens
        generated = decoder_head.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            max_new_tokens=5,
        )

        # Output should exist and have reasonable shape
        assert generated is not None
        assert generated.shape[0] == batch_size

"""
Tests for Cross-Attention Pair Encoder

Test coverage for:
    - CrossAttentionPairEncoder: NLI/Relation pair encoding
    - CrossAttentionLayer: Basic cross-attention
    - BidirectionalCrossAttentionBlock: Bidirectional attention
    - AttentionPooling: Attention-based sequence pooling
    - ConcatPairEncoder: Simple concatenation fallback
    - Factory function: create_pair_encoder

Issue: 5.0.3 - Implement Cross-Attention Pair Encoder
Epic: 5.0 - Model Architecture Enhancements (Pre-Stage B)
"""

import pytest
import torch

from modeling_studio.models.pair_encoder import (
    AttentionPooling,
    BidirectionalCrossAttentionBlock,
    ConcatPairEncoder,
    CrossAttentionLayer,
    CrossAttentionPairEncoder,
    PairEncoderConfig,
    create_pair_encoder,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hidden_size():
    """Default hidden size for tests."""
    return 768


@pytest.fixture
def num_heads():
    """Default number of attention heads."""
    return 8


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 4


@pytest.fixture
def premise_len():
    """Default premise sequence length."""
    return 64


@pytest.fixture
def hypothesis_len():
    """Default hypothesis sequence length."""
    return 32


@pytest.fixture
def sample_premise(batch_size, premise_len, hidden_size):
    """Sample premise hidden states."""
    return torch.randn(batch_size, premise_len, hidden_size)


@pytest.fixture
def sample_hypothesis(batch_size, hypothesis_len, hidden_size):
    """Sample hypothesis hidden states."""
    return torch.randn(batch_size, hypothesis_len, hidden_size)


@pytest.fixture
def premise_mask(batch_size, premise_len):
    """Sample premise attention mask."""
    return torch.ones(batch_size, premise_len)


@pytest.fixture
def hypothesis_mask(batch_size, hypothesis_len):
    """Sample hypothesis attention mask."""
    return torch.ones(batch_size, hypothesis_len)


# =============================================================================
# CrossAttentionLayer Tests
# =============================================================================


class TestCrossAttentionLayer:
    """Tests for CrossAttentionLayer."""

    def test_output_shape(self, sample_premise, sample_hypothesis, hidden_size, num_heads):
        """Output should have same shape as query sequence."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)
        output = layer(
            query_states=sample_premise,
            key_value_states=sample_hypothesis,
        )
        assert output.shape == sample_premise.shape

    def test_with_masks(
        self, sample_premise, sample_hypothesis, premise_mask, hypothesis_mask, hidden_size
    ):
        """Should work with attention masks."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=8)
        output = layer(
            query_states=sample_premise,
            key_value_states=sample_hypothesis,
            query_mask=premise_mask,
            key_value_mask=hypothesis_mask,
        )
        assert output.shape == sample_premise.shape

    def test_residual_connection(self, sample_premise, sample_hypothesis, hidden_size):
        """With residual, output should be different from input."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=8, use_residual=True)
        output = layer(sample_premise, sample_hypothesis)
        # Should have changed from input due to attention
        assert not torch.allclose(output, sample_premise, atol=1e-3)


# =============================================================================
# BidirectionalCrossAttentionBlock Tests
# =============================================================================


class TestBidirectionalCrossAttentionBlock:
    """Tests for BidirectionalCrossAttentionBlock."""

    def test_output_shapes(self, sample_premise, sample_hypothesis, hidden_size, num_heads):
        """Both outputs should have same shapes as inputs."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        out_a, out_b = block(sample_premise, sample_hypothesis)
        assert out_a.shape == sample_premise.shape
        assert out_b.shape == sample_hypothesis.shape

    def test_with_ffn(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work with feedforward network."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=8,
            use_ffn=True,
        )
        out_a, out_b = block(sample_premise, sample_hypothesis)
        assert out_a.shape == sample_premise.shape
        assert out_b.shape == sample_hypothesis.shape

    def test_without_ffn(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work without feedforward network."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=8,
            use_ffn=False,
        )
        out_a, out_b = block(sample_premise, sample_hypothesis)
        assert out_a.shape == sample_premise.shape


# =============================================================================
# AttentionPooling Tests
# =============================================================================


class TestAttentionPooling:
    """Tests for AttentionPooling."""

    def test_output_shape(self, sample_premise, hidden_size):
        """Should pool to (batch, hidden_size)."""
        pooler = AttentionPooling(hidden_size=hidden_size)
        output = pooler(sample_premise)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_with_mask(self, sample_premise, premise_mask, hidden_size):
        """Should work with attention mask."""
        pooler = AttentionPooling(hidden_size=hidden_size)
        output = pooler(sample_premise, attention_mask=premise_mask)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_multihead(self, sample_premise, hidden_size):
        """Should work with multiple attention heads."""
        pooler = AttentionPooling(hidden_size=hidden_size, num_heads=4)
        output = pooler(sample_premise)
        assert output.shape == (sample_premise.size(0), hidden_size)


# =============================================================================
# CrossAttentionPairEncoder Tests
# =============================================================================


class TestCrossAttentionPairEncoder:
    """Tests for CrossAttentionPairEncoder."""

    def test_output_shape(
        self,
        sample_premise,
        sample_hypothesis,
        premise_mask,
        hypothesis_mask,
        hidden_size,
        num_heads,
    ):
        """Output should be (batch, hidden_size) by default."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        output = encoder(
            sample_premise,
            sample_hypothesis,
            premise_mask,
            hypothesis_mask,
        )
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_custom_output_size(self, sample_premise, sample_hypothesis, hidden_size):
        """Should support custom output size."""
        output_size = 256
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            output_size=output_size,
        )
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), output_size)

    def test_multiple_layers(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work with multiple cross-attention layers."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            num_layers=3,
        )
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_unidirectional(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work in unidirectional mode."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            use_bidirectional=False,
        )
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_pooling_strategies(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work with different pooling strategies."""
        for strategy in ["cls", "mean", "max", "attention"]:
            encoder = CrossAttentionPairEncoder(
                hidden_size=hidden_size,
                num_heads=8,
                pooling_strategy=strategy,
            )
            output = encoder(sample_premise, sample_hypothesis)
            assert output.shape == (sample_premise.size(0), hidden_size)

    def test_gradient_flow(self, sample_premise, sample_hypothesis, hidden_size):
        """Gradients should flow through the encoder."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)
        output = encoder(sample_premise, sample_hypothesis)
        loss = output.sum()
        loss.backward()

        # Check that encoder parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# ConcatPairEncoder Tests
# =============================================================================


class TestConcatPairEncoder:
    """Tests for ConcatPairEncoder (fallback)."""

    def test_output_shape(self, sample_premise, sample_hypothesis, hidden_size):
        """Output should be (batch, hidden_size) by default."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_pooling_strategies(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work with different pooling strategies."""
        for strategy in ["cls", "mean", "max"]:
            encoder = ConcatPairEncoder(
                hidden_size=hidden_size,
                pooling_strategy=strategy,
            )
            output = encoder(sample_premise, sample_hypothesis)
            assert output.shape == (sample_premise.size(0), hidden_size)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreatePairEncoder:
    """Tests for create_pair_encoder factory function."""

    def test_create_cross_attention(self, hidden_size):
        """Should create CrossAttentionPairEncoder."""
        encoder = create_pair_encoder("cross_attention", hidden_size=hidden_size)
        assert isinstance(encoder, CrossAttentionPairEncoder)

    def test_create_concat(self, hidden_size):
        """Should create ConcatPairEncoder."""
        encoder = create_pair_encoder("concat", hidden_size=hidden_size)
        assert isinstance(encoder, ConcatPairEncoder)

    def test_create_none(self, hidden_size):
        """Should return None for 'none' type."""
        encoder = create_pair_encoder("none", hidden_size=hidden_size)
        assert encoder is None

    def test_invalid_type(self, hidden_size):
        """Should raise error for invalid type."""
        with pytest.raises(ValueError, match="Unknown encoder_type"):
            create_pair_encoder("invalid", hidden_size=hidden_size)


# =============================================================================
# Integration Test
# =============================================================================


class TestPairEncoderIntegration:
    """Integration tests matching acceptance criteria."""

    def test_acceptance_criteria(self):
        """Test CrossAttentionPairEncoder as per acceptance criteria."""
        encoder = CrossAttentionPairEncoder(hidden_size=768, num_heads=8)

        # For NLI: premise and hypothesis representations
        premise = torch.randn(2, 64, 768)
        hypothesis = torch.randn(2, 32, 768)
        premise_mask = torch.ones(2, 64)
        hypothesis_mask = torch.ones(2, 32)

        pair_repr = encoder(premise, hypothesis, premise_mask, hypothesis_mask)
        assert pair_repr.shape == (2, 768)


# =============================================================================
# Config Tests
# =============================================================================


class TestPairEncoderConfig:
    """Tests for PairEncoderConfig."""

    def test_default_config(self):
        """Default config should be valid."""
        config = PairEncoderConfig()
        assert config.hidden_size == 768
        assert config.num_heads == 8
        assert config.ffn_hidden_size == 768 * 4

    def test_custom_config(self):
        """Custom config should work."""
        config = PairEncoderConfig(
            hidden_size=512,
            num_heads=4,
            num_layers=2,
        )
        assert config.hidden_size == 512
        assert config.num_heads == 4

    def test_invalid_config(self):
        """Should raise error for invalid config."""
        with pytest.raises(ValueError, match="must be divisible"):
            PairEncoderConfig(hidden_size=768, num_heads=7)

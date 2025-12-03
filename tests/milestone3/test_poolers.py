"""
Tests for Issue 3.2.1: models/poolers.py

This module tests the pooling strategies for converting token-level encoder
outputs into fixed-size representations.

Pooling Concepts Tested:
    - BasePooler: Abstract base class with mask expansion
    - CLSPooler: [CLS] token extraction with optional dense layer
    - MeanPooler: Masked mean pooling (sum * mask / sum_mask)
    - MaxPooler: Masked max pooling with -inf for padding
    - WeightedMeanPooler: Learnable attention-weighted mean
    - LastTokenPooler: Last non-padding token extraction
    - CLSMeanPooler: Concatenation of CLS + mean representations
    - AttentionPooler: Multi-head cross-attention with learnable query
    - get_pooler: Factory function for pooler instantiation

Mathematical Formulas Tested:
    - Mean: pooled = sum(hidden * mask) / sum(mask)
    - Max: pooled = max(hidden where mask == 1)
    - Weighted: weights = softmax(W @ hidden), pooled = sum(weights * hidden)
    - Attention: pool_query -> MHA(q, k=hidden, v=hidden)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from abc import ABC

from modeling_studio.models.poolers import (
    BasePooler,
    CLSPooler,
    MeanPooler,
    MaxPooler,
    WeightedMeanPooler,
    LastTokenPooler,
    CLSMeanPooler,
    AttentionPooler,
    get_pooler,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hidden_size():
    """Standard hidden size for tests."""
    return 768


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def hidden_states(batch_size, seq_len, hidden_size):
    """Sample hidden states tensor."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, hidden_size)


@pytest.fixture
def attention_mask(batch_size, seq_len):
    """Sample attention mask with variable lengths."""
    # Create masks with different sequence lengths
    mask = torch.zeros(batch_size, seq_len)
    lengths = [seq_len, seq_len - 5, seq_len - 10, seq_len // 2]
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
    return mask


@pytest.fixture
def full_attention_mask(batch_size, seq_len):
    """Attention mask with no padding."""
    return torch.ones(batch_size, seq_len)


# =============================================================================
# Test: BasePooler Abstract Class
# =============================================================================


class TestBasePooler:
    """Tests for BasePooler abstract base class."""

    def test_base_pooler_abstract(self):
        """BasePooler is abstract and cannot be instantiated directly.

        Concept: BasePooler defines the interface that all poolers must implement.
        It inherits from ABC and has an abstract forward method.
        """
        with pytest.raises(TypeError, match="abstract"):
            BasePooler(hidden_size=768)

    def test_base_pooler_inheritance(self):
        """BasePooler inherits from ABC and nn.Module."""
        assert issubclass(BasePooler, ABC)
        assert issubclass(BasePooler, nn.Module)

    def test_base_pooler_expand_mask(self, hidden_states, hidden_size):
        """Test mask expansion for broadcasting.

        Concept: _expand_mask converts (batch, seq) -> (batch, seq, 1)
        to enable element-wise multiplication with (batch, seq, hidden).
        """
        # Create a concrete pooler to test _expand_mask
        pooler = CLSPooler(hidden_size=hidden_size)
        batch_size, seq_len = 4, 32

        # Test with mask provided
        attention_mask = torch.ones(batch_size, seq_len)
        expanded = pooler._expand_mask(attention_mask, hidden_states)

        assert expanded.shape == (batch_size, seq_len, 1)
        assert expanded.dtype == hidden_states.dtype

    def test_base_pooler_expand_mask_none(self, hidden_states, hidden_size):
        """When attention_mask is None, create all-ones mask.

        Concept: If no mask is provided, assume all tokens are valid.
        """
        pooler = CLSPooler(hidden_size=hidden_size)
        expanded = pooler._expand_mask(None, hidden_states)

        batch_size, seq_len = hidden_states.shape[:2]
        assert expanded.shape == (batch_size, seq_len, 1)
        assert torch.all(expanded == 1.0)

    def test_base_pooler_expand_mask_dtype_match(self, hidden_states, hidden_size):
        """Expanded mask dtype should match hidden_states dtype.

        Concept: For mixed precision training, mask must match the
        dtype of hidden_states to avoid dtype mismatches in operations.
        """
        pooler = CLSPooler(hidden_size=hidden_size)

        # Test with float16 hidden states
        hidden_float16 = hidden_states.half()
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1])

        expanded = pooler._expand_mask(attention_mask, hidden_float16)
        assert expanded.dtype == torch.float16


# =============================================================================
# Test: CLSPooler
# =============================================================================


class TestCLSPooler:
    """Tests for CLSPooler - [CLS] token extraction."""

    def test_cls_pooler_init(self, hidden_size):
        """CLSPooler initializes with correct parameters.

        Concept: CLSPooler extracts the [CLS] token and optionally
        applies a dense layer with tanh activation.
        """
        pooler = CLSPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.output_size == hidden_size
        assert pooler.cls_position == 0
        assert pooler.use_dense is True
        assert hasattr(pooler, "dense")
        assert hasattr(pooler, "activation")

    def test_cls_pooler_init_no_dense(self, hidden_size):
        """CLSPooler without dense layer."""
        pooler = CLSPooler(hidden_size=hidden_size, use_dense=False)

        assert pooler.use_dense is False
        assert pooler.dense is None
        assert pooler.activation is None

    def test_cls_pooler_forward(self, hidden_states, hidden_size):
        """CLSPooler extracts and transforms [CLS] token.

        Concept: The [CLS] token (position 0) aggregates sequence-level
        information during BERT pre-training.
        """
        pooler = CLSPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, hidden_size)
        assert pooled.dtype == hidden_states.dtype

    def test_cls_pooler_forward_no_dense(self, hidden_states, hidden_size):
        """Without dense layer, returns raw [CLS] token."""
        pooler = CLSPooler(hidden_size=hidden_size, use_dense=False)

        pooled = pooler(hidden_states)
        cls_token = hidden_states[:, 0, :]

        # Should be exactly the CLS token
        assert torch.allclose(pooled, cls_token, atol=1e-6)

    def test_cls_pooler_with_dense(self, hidden_states, hidden_size):
        """With dense layer, applies dense + tanh transformation.

        Concept: The dense layer projects the CLS token and tanh
        squashes the output to [-1, 1] for downstream tasks.
        """
        pooler = CLSPooler(hidden_size=hidden_size, use_dense=True)

        pooled = pooler(hidden_states)

        # Output should be in tanh range
        assert pooled.min() >= -1.0
        assert pooled.max() <= 1.0

    def test_cls_pooler_custom_position(self, hidden_states, hidden_size):
        """CLSPooler can extract from custom position."""
        pooler = CLSPooler(hidden_size=hidden_size, use_dense=False, cls_position=1)

        pooled = pooler(hidden_states)
        expected = hidden_states[:, 1, :]

        assert torch.allclose(pooled, expected, atol=1e-6)

    def test_cls_pooler_custom_output_size(self, hidden_states, hidden_size):
        """CLSPooler with different output size."""
        output_size = 256
        pooler = CLSPooler(hidden_size=hidden_size, output_size=output_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, output_size)


# =============================================================================
# Test: MeanPooler
# =============================================================================


class TestMeanPooler:
    """Tests for MeanPooler - masked mean pooling."""

    def test_mean_pooler_init(self, hidden_size):
        """MeanPooler initializes correctly."""
        pooler = MeanPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.output_size == hidden_size
        assert pooler.use_projection is False

    def test_mean_pooler_forward(self, hidden_states, full_attention_mask, hidden_size):
        """MeanPooler computes mean over all tokens.

        Concept: For full attention mask, mean pooling is simply
        the average of all token representations.
        """
        pooler = MeanPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states, full_attention_mask)

        assert pooled.shape == (batch_size, hidden_size)

        # Verify against manual mean
        expected = hidden_states.mean(dim=1)
        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_mean_pooler_ignores_padding(self, hidden_size):
        """MeanPooler correctly excludes padding tokens.

        Concept: The formula is sum(hidden * mask) / sum(mask)
        Padding tokens (mask=0) don't contribute to sum or count.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # First sequence: 6 valid tokens, second: 4 valid tokens
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :6] = 1.0
        attention_mask[1, :4] = 1.0

        pooler = MeanPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        # Manually compute expected means
        expected_0 = hidden_states[0, :6].mean(dim=0)
        expected_1 = hidden_states[1, :4].mean(dim=0)

        assert torch.allclose(pooled[0], expected_0, atol=1e-5)
        assert torch.allclose(pooled[1], expected_1, atol=1e-5)

    def test_mean_pooler_no_mask(self, hidden_states, hidden_size):
        """Without mask, mean pooler averages all tokens."""
        pooler = MeanPooler(hidden_size=hidden_size)

        pooled = pooler(hidden_states, attention_mask=None)
        expected = hidden_states.mean(dim=1)

        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_mean_pooler_with_projection(self, hidden_states, hidden_size):
        """MeanPooler with optional projection layer."""
        output_size = 256
        pooler = MeanPooler(hidden_size=hidden_size, output_size=output_size, use_projection=True)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, output_size)

    def test_mean_pooler_avoids_div_by_zero(self, hidden_size):
        """MeanPooler handles edge case of all-masked sequence.

        Concept: Uses clamp(min=1e-9) to avoid division by zero
        when a sequence has no valid tokens.
        """
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Second sequence is all padding
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :5] = 1.0  # First sequence has 5 valid tokens
        # Second sequence has no valid tokens

        pooler = MeanPooler(hidden_size=hidden_size)

        # Should not raise an error
        pooled = pooler(hidden_states, attention_mask)

        assert pooled.shape == (batch_size, hidden_size)
        assert not torch.isnan(pooled).any()


# =============================================================================
# Test: MaxPooler
# =============================================================================


class TestMaxPooler:
    """Tests for MaxPooler - masked max pooling."""

    def test_max_pooler_init(self, hidden_size):
        """MaxPooler initializes correctly."""
        pooler = MaxPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.output_size == hidden_size
        assert pooler.use_projection is False

    def test_max_pooler_forward(self, hidden_states, full_attention_mask, hidden_size):
        """MaxPooler computes element-wise max over tokens.

        Concept: Max pooling captures the most salient features
        across the sequence for each dimension.
        """
        pooler = MaxPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states, full_attention_mask)

        assert pooled.shape == (batch_size, hidden_size)

        # Verify against manual max
        expected, _ = hidden_states.max(dim=1)
        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_max_pooler_ignores_padding(self, hidden_size):
        """MaxPooler sets padding to -inf before max.

        Concept: Setting padding to -inf ensures they won't be
        selected as the maximum value for any dimension.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Set padding positions to very high values (should be ignored)
        hidden_states[0, 6:] = 100.0  # These should be masked out
        hidden_states[1, 4:] = 100.0  # These should be masked out

        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :6] = 1.0
        attention_mask[1, :4] = 1.0

        pooler = MaxPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        # Max should be computed only over valid tokens
        expected_0, _ = hidden_states[0, :6].max(dim=0)
        expected_1, _ = hidden_states[1, :4].max(dim=0)

        assert torch.allclose(pooled[0], expected_0, atol=1e-5)
        assert torch.allclose(pooled[1], expected_1, atol=1e-5)

        # Verify padding values (100.0) were NOT selected
        assert (pooled < 50.0).all()

    def test_max_pooler_handles_all_masked(self, hidden_size):
        """MaxPooler handles all-masked sequences gracefully.

        Concept: When all tokens are masked, -inf values are
        replaced with zeros to avoid downstream issues.
        """
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Second sequence is all padding
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :5] = 1.0
        # attention_mask[1] is all zeros

        pooler = MaxPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        # Should not have inf values
        assert not torch.isinf(pooled).any()

        # All-masked sequence should have zeros
        assert torch.allclose(pooled[1], torch.zeros(hidden_size), atol=1e-6)


# =============================================================================
# Test: WeightedMeanPooler
# =============================================================================


class TestWeightedMeanPooler:
    """Tests for WeightedMeanPooler - attention-weighted mean pooling."""

    def test_weighted_mean_pooler_init(self, hidden_size):
        """WeightedMeanPooler initializes with attention layer.

        Concept: Learns attention weights via a linear projection
        from hidden_size to num_attention_heads.
        """
        pooler = WeightedMeanPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.num_attention_heads == 1
        assert hasattr(pooler, "attention_dense")

        # attention_dense projects to num_attention_heads
        assert pooler.attention_dense.out_features == 1

    def test_weighted_mean_pooler_forward(self, hidden_states, full_attention_mask, hidden_size):
        """WeightedMeanPooler computes attention-weighted mean.

        Concept: weights = softmax(W @ hidden + b)
                 pooled = sum(weights * hidden)

        Reference: Lin et al. "A Structured Self-Attentive Sentence Embedding"
        """
        pooler = WeightedMeanPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states, full_attention_mask)

        assert pooled.shape == (batch_size, hidden_size)

    def test_weighted_mean_pooler_learned_weights(self, hidden_size):
        """Attention weights are learned and sum to 1 (after softmax)."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        pooler = WeightedMeanPooler(hidden_size=hidden_size)

        # Get attention scores before pooling
        attention_scores = pooler.attention_dense(hidden_states)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weights should sum to 1 across sequence dimension
        weight_sums = attention_weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(batch_size, 1), atol=1e-5)

    def test_weighted_mean_pooler_multi_head(self, hidden_states, hidden_size):
        """WeightedMeanPooler with multiple attention heads."""
        num_heads = 4
        pooler = WeightedMeanPooler(hidden_size=hidden_size, num_attention_heads=num_heads)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        # With multi-head, output is still hidden_size (or output_size if different)
        assert pooled.shape == (batch_size, hidden_size)

    def test_weighted_mean_pooler_masks_padding(self, hidden_size):
        """WeightedMeanPooler applies -inf to padding before softmax.

        Concept: Setting padding scores to -inf ensures they get
        zero attention weight after softmax.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :6] = 1.0  # 6 valid tokens
        attention_mask[1, :4] = 1.0  # 4 valid tokens

        pooler = WeightedMeanPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        assert pooled.shape == (batch_size, hidden_size)
        assert not torch.isnan(pooled).any()


# =============================================================================
# Test: LastTokenPooler
# =============================================================================


class TestLastTokenPooler:
    """Tests for LastTokenPooler - last non-padding token extraction."""

    def test_last_token_pooler_init(self, hidden_size):
        """LastTokenPooler initializes correctly."""
        pooler = LastTokenPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.use_dense is False

    def test_last_token_pooler_forward(self, hidden_states, hidden_size):
        """LastTokenPooler extracts last token when no mask.

        Concept: Useful for causal models (GPT-style) where the
        last token aggregates information from the entire sequence.
        """
        pooler = LastTokenPooler(hidden_size=hidden_size)
        batch_size, seq_len = hidden_states.shape[:2]

        pooled = pooler(hidden_states, attention_mask=None)

        # Should be the last token
        expected = hidden_states[:, seq_len - 1, :]

        assert pooled.shape == (batch_size, hidden_size)
        assert torch.allclose(pooled, expected, atol=1e-6)

    def test_last_token_pooler_with_mask(self, hidden_size):
        """LastTokenPooler extracts last non-padding token.

        Concept: Uses seq_length = mask.sum(dim=1) to find the
        index of the last valid token for each sequence.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 3, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Different sequence lengths
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :8] = 1.0  # Last valid: index 7
        attention_mask[1, :5] = 1.0  # Last valid: index 4
        attention_mask[2, :10] = 1.0  # Last valid: index 9

        pooler = LastTokenPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        assert torch.allclose(pooled[0], hidden_states[0, 7], atol=1e-6)
        assert torch.allclose(pooled[1], hidden_states[1, 4], atol=1e-6)
        assert torch.allclose(pooled[2], hidden_states[2, 9], atol=1e-6)

    def test_last_token_pooler_with_dense(self, hidden_states, hidden_size):
        """LastTokenPooler with optional dense transformation."""
        output_size = 256
        pooler = LastTokenPooler(hidden_size=hidden_size, output_size=output_size, use_dense=True)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, output_size)
        # With tanh activation, values should be bounded
        assert pooled.min() >= -1.0
        assert pooled.max() <= 1.0


# =============================================================================
# Test: CLSMeanPooler
# =============================================================================


class TestCLSMeanPooler:
    """Tests for CLSMeanPooler - combined CLS + mean pooling."""

    def test_cls_mean_pooler_init(self, hidden_size):
        """CLSMeanPooler initializes with projection layer.

        Concept: Concatenates CLS and mean representations, then
        projects from 2*hidden_size back to output_size.
        """
        pooler = CLSMeanPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert hasattr(pooler, "projection")
        # Projection from 2*hidden to hidden
        assert pooler.projection.in_features == hidden_size * 2
        assert pooler.projection.out_features == hidden_size

    def test_cls_mean_pooler_forward(self, hidden_states, full_attention_mask, hidden_size):
        """CLSMeanPooler combines CLS and mean.

        Concept: concat([CLS], mean_pool) -> Linear -> LayerNorm
        This combines: (1) trained [CLS] representation and
        (2) distributional information from all tokens.
        """
        pooler = CLSMeanPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states, full_attention_mask)

        assert pooled.shape == (batch_size, hidden_size)

    def test_cls_mean_pooler_custom_output(self, hidden_states, hidden_size):
        """CLSMeanPooler with custom output size."""
        output_size = 256
        pooler = CLSMeanPooler(hidden_size=hidden_size, output_size=output_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, output_size)

    def test_cls_mean_pooler_with_layer_norm(self, hidden_states, hidden_size):
        """CLSMeanPooler applies layer norm to output."""
        pooler = CLSMeanPooler(hidden_size=hidden_size, use_layer_norm=True)

        pooled = pooler(hidden_states)

        # Layer norm normalizes across feature dimension
        # Mean should be close to 0, std close to 1
        mean = pooled.mean(dim=-1)
        std = pooled.std(dim=-1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)


# =============================================================================
# Test: AttentionPooler
# =============================================================================


class TestAttentionPooler:
    """Tests for AttentionPooler - multi-head cross-attention pooling."""

    def test_attention_pooler_init(self, hidden_size):
        """AttentionPooler initializes with learnable query.

        Concept: Uses a learnable [POOL] query token that attends
        to the sequence via multi-head cross-attention.
        """
        pooler = AttentionPooler(hidden_size=hidden_size)

        assert pooler.hidden_size == hidden_size
        assert pooler.num_heads == 8
        assert hasattr(pooler, "pool_query")
        assert pooler.pool_query.shape == (1, 1, hidden_size)

    def test_attention_pooler_forward(self, hidden_states, full_attention_mask, hidden_size):
        """AttentionPooler computes cross-attention with learnable query.

        Concept: pool_query -> MultiHeadAttention(q, k=hidden, v=hidden)
        The query learns what information to extract from the sequence.
        """
        pooler = AttentionPooler(hidden_size=hidden_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states, full_attention_mask)

        assert pooled.shape == (batch_size, hidden_size)

    def test_attention_pooler_learnable_query(self, hidden_size):
        """Pool query is a learnable parameter."""
        pooler = AttentionPooler(hidden_size=hidden_size)

        # pool_query should be a Parameter (will be updated during training)
        assert isinstance(pooler.pool_query, nn.Parameter)
        assert pooler.pool_query.requires_grad is True

    def test_attention_pooler_custom_output(self, hidden_states, hidden_size):
        """AttentionPooler with custom output size."""
        output_size = 256
        pooler = AttentionPooler(hidden_size=hidden_size, output_size=output_size)
        batch_size = hidden_states.shape[0]

        pooled = pooler(hidden_states)

        assert pooled.shape == (batch_size, output_size)

    def test_attention_pooler_masks_padding(self, hidden_size):
        """AttentionPooler respects attention mask.

        Concept: Converts attention_mask to key_padding_mask format
        (True = ignore, False = attend) for nn.MultiheadAttention.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :6] = 1.0
        attention_mask[1, :4] = 1.0

        pooler = AttentionPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)

        assert pooled.shape == (batch_size, hidden_size)
        assert not torch.isnan(pooled).any()


# =============================================================================
# Test: get_pooler Factory Function
# =============================================================================


class TestGetPoolerFactory:
    """Tests for get_pooler factory function."""

    def test_get_pooler_factory_cls(self, hidden_size):
        """Factory creates CLSPooler for 'cls' strategy."""
        pooler = get_pooler("cls", hidden_size=hidden_size)
        assert isinstance(pooler, CLSPooler)

    def test_get_pooler_factory_mean(self, hidden_size):
        """Factory creates MeanPooler for 'mean' strategy."""
        pooler = get_pooler("mean", hidden_size=hidden_size)
        assert isinstance(pooler, MeanPooler)

    def test_get_pooler_factory_max(self, hidden_size):
        """Factory creates MaxPooler for 'max' strategy."""
        pooler = get_pooler("max", hidden_size=hidden_size)
        assert isinstance(pooler, MaxPooler)

    def test_get_pooler_factory_weighted_mean(self, hidden_size):
        """Factory creates WeightedMeanPooler for 'weighted_mean' strategy."""
        pooler = get_pooler("weighted_mean", hidden_size=hidden_size)
        assert isinstance(pooler, WeightedMeanPooler)

    def test_get_pooler_factory_weighted_alias(self, hidden_size):
        """Factory supports 'weighted' alias for WeightedMeanPooler."""
        pooler = get_pooler("weighted", hidden_size=hidden_size)
        assert isinstance(pooler, WeightedMeanPooler)

    def test_get_pooler_factory_last_token(self, hidden_size):
        """Factory creates LastTokenPooler for 'last_token' strategy."""
        pooler = get_pooler("last_token", hidden_size=hidden_size)
        assert isinstance(pooler, LastTokenPooler)

    def test_get_pooler_factory_last_alias(self, hidden_size):
        """Factory supports 'last' alias for LastTokenPooler."""
        pooler = get_pooler("last", hidden_size=hidden_size)
        assert isinstance(pooler, LastTokenPooler)

    def test_get_pooler_factory_cls_mean(self, hidden_size):
        """Factory creates CLSMeanPooler for 'cls_mean' strategy."""
        pooler = get_pooler("cls_mean", hidden_size=hidden_size)
        assert isinstance(pooler, CLSMeanPooler)

    def test_get_pooler_factory_attention(self, hidden_size):
        """Factory creates AttentionPooler for 'attention' strategy."""
        pooler = get_pooler("attention", hidden_size=hidden_size)
        assert isinstance(pooler, AttentionPooler)

    def test_get_pooler_factory_case_insensitive(self, hidden_size):
        """Factory is case-insensitive."""
        pooler1 = get_pooler("CLS", hidden_size=hidden_size)
        pooler2 = get_pooler("Cls", hidden_size=hidden_size)
        pooler3 = get_pooler("cls", hidden_size=hidden_size)

        assert isinstance(pooler1, CLSPooler)
        assert isinstance(pooler2, CLSPooler)
        assert isinstance(pooler3, CLSPooler)

    def test_get_pooler_factory_with_output_size(self, hidden_size):
        """Factory passes output_size to pooler."""
        output_size = 256
        pooler = get_pooler("mean", hidden_size=hidden_size, output_size=output_size)

        assert pooler.output_size == output_size

    def test_get_pooler_factory_unknown_strategy(self, hidden_size):
        """Factory raises error for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            get_pooler("unknown_strategy", hidden_size=hidden_size)

    def test_get_pooler_factory_with_kwargs(self, hidden_size):
        """Factory passes additional kwargs to pooler."""
        pooler = get_pooler("cls", hidden_size=hidden_size, use_dense=False)

        assert pooler.use_dense is False


# =============================================================================
# Test: Edge Cases and Mathematical Correctness
# =============================================================================


class TestPoolerEdgeCases:
    """Test edge cases and verify mathematical correctness."""

    def test_single_token_sequence(self, hidden_size):
        """All poolers handle single-token sequences."""
        batch_size, seq_len = 2, 1
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        poolers = [
            CLSPooler(hidden_size=hidden_size, use_dense=False),
            MeanPooler(hidden_size=hidden_size),
            MaxPooler(hidden_size=hidden_size),
            LastTokenPooler(hidden_size=hidden_size),
        ]

        for pooler in poolers:
            pooled = pooler(hidden_states, attention_mask)
            # For single token, all pooling strategies should return that token
            expected = hidden_states.squeeze(1)
            assert torch.allclose(
                pooled, expected, atol=1e-5
            ), f"Failed for {type(pooler).__name__}"

    def test_batch_independence(self, hidden_size):
        """Pooling is independent across batch elements."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        pooler = MeanPooler(hidden_size=hidden_size)

        # Pool full batch
        full_pooled = pooler(hidden_states, attention_mask)

        # Pool each element individually
        for i in range(batch_size):
            single_pooled = pooler(hidden_states[i : i + 1], attention_mask[i : i + 1])
            assert torch.allclose(full_pooled[i : i + 1], single_pooled, atol=1e-5)

    def test_gradient_flow(self, hidden_size):
        """Poolers allow gradient flow back to input."""
        hidden_states = torch.randn(2, 8, hidden_size, requires_grad=True)
        attention_mask = torch.ones(2, 8)

        pooler = MeanPooler(hidden_size=hidden_size)
        pooled = pooler(hidden_states, attention_mask)
        loss = pooled.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

    def test_deterministic_output(self, hidden_size):
        """Poolers produce deterministic output for same input."""
        torch.manual_seed(42)
        hidden_states = torch.randn(2, 8, hidden_size)
        attention_mask = torch.ones(2, 8)

        pooler = WeightedMeanPooler(hidden_size=hidden_size)
        pooler.eval()  # Ensure deterministic mode

        output1 = pooler(hidden_states, attention_mask)
        output2 = pooler(hidden_states, attention_mask)

        assert torch.allclose(output1, output2, atol=1e-6)


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    """Test that all expected classes are exported from the module."""

    def test_all_poolers_exported(self):
        """Verify __all__ contains all pooler classes."""
        from modeling_studio.models import poolers

        expected = [
            "BasePooler",
            "CLSPooler",
            "MeanPooler",
            "MaxPooler",
            "WeightedMeanPooler",
            "LastTokenPooler",
            "CLSMeanPooler",
            "AttentionPooler",
            "get_pooler",
        ]

        for name in expected:
            assert name in poolers.__all__, f"{name} not in __all__"
            assert hasattr(poolers, name), f"{name} not exported"

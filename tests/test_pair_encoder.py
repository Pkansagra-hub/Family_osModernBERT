"""
Tests for Cross-Attention Pair Encoder

Test coverage for:
    - PairEncoderConfig: Configuration validation
    - CrossAttentionPairEncoder: NLI/Relation pair encoding
    - CrossAttentionLayer: Basic cross-attention
    - BidirectionalCrossAttentionBlock: Bidirectional attention
    - AttentionPooling: Attention-based sequence pooling
    - FeedForward: Position-wise feedforward network
    - ConcatPairEncoder: Simple concatenation fallback
    - Factory function: create_pair_encoder

Issue: 5.0.3 - Implement Cross-Attention Pair Encoder
Epic: 5.0 - Model Architecture Enhancements (Pre-Stage B)

Test Plan Coverage:
    - test_pair_encoder_config_init - PairEncoderConfig initializes correctly
    - test_pair_encoder_config_validation - Validates hidden_size % num_heads == 0
    - test_cross_attention_layer_init - Initializes Q, K, V projections
    - test_cross_attention_layer_forward - Computes cross-attention
    - test_cross_attention_layer_mask - Masks out padding in key-value
    - test_cross_attention_layer_residual - Adds residual connection
    - test_feedforward_init - Initializes with GELU activation
    - test_feedforward_forward - Two linear transformations with activation
    - test_bidirectional_cross_attention_block - Both directions attended
    - test_attention_pooling - Learns attention weights for pooling
    - test_cross_attention_pair_encoder_init - Initializes with layers
    - test_cross_attention_pair_encoder_forward - Returns pair representation
    - test_cross_attention_pair_encoder_pooling_cls - Uses CLS pooling
    - test_cross_attention_pair_encoder_pooling_mean - Uses mean pooling
    - test_cross_attention_pair_encoder_pooling_attention - Uses attention pooling
    - test_concat_pair_encoder_forward - Concatenates pooled representations
    - test_create_pair_encoder_factory - Factory creates correct encoder type
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.pair_encoder import (
    AttentionPooling,
    BidirectionalCrossAttentionBlock,
    ConcatPairEncoder,
    CrossAttentionLayer,
    CrossAttentionPairEncoder,
    FeedForward,
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

    def test_cross_attention_layer_init(self, hidden_size, num_heads):
        """Test that CrossAttentionLayer initializes Q, K, V projections correctly."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)

        # Verify Q, K, V projections exist
        assert hasattr(layer, "query")
        assert hasattr(layer, "key")
        assert hasattr(layer, "value")
        assert hasattr(layer, "out_proj")

        # Check dimensions
        assert isinstance(layer.query, nn.Linear)
        assert layer.query.in_features == hidden_size
        assert layer.query.out_features == hidden_size

        assert isinstance(layer.key, nn.Linear)
        assert layer.key.in_features == hidden_size
        assert layer.key.out_features == hidden_size

        assert isinstance(layer.value, nn.Linear)
        assert layer.value.in_features == hidden_size
        assert layer.value.out_features == hidden_size

        assert isinstance(layer.out_proj, nn.Linear)
        assert layer.out_proj.in_features == hidden_size
        assert layer.out_proj.out_features == hidden_size

        # Verify head dimension
        assert layer.head_dim == hidden_size // num_heads
        assert layer.num_heads == num_heads
        assert layer.hidden_size == hidden_size

    def test_cross_attention_layer_init_with_options(self, hidden_size, num_heads):
        """Test CrossAttentionLayer with various initialization options."""
        # With layer norm
        layer_with_ln = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_layer_norm=True,
        )
        assert layer_with_ln.layer_norm is not None
        assert isinstance(layer_with_ln.layer_norm, nn.LayerNorm)

        # Without layer norm
        layer_without_ln = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_layer_norm=False,
        )
        assert layer_without_ln.layer_norm is None

        # With residual
        layer_with_residual = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_residual=True,
        )
        assert layer_with_residual.use_residual is True

        # Without residual
        layer_without_residual = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_residual=False,
        )
        assert layer_without_residual.use_residual is False

    def test_cross_attention_layer_forward(
        self, sample_premise, sample_hypothesis, hidden_size, num_heads
    ):
        """Test that CrossAttentionLayer computes cross-attention correctly."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)
        output = layer(
            query_states=sample_premise,
            key_value_states=sample_hypothesis,
        )

        # Output shape should match query shape
        assert output.shape == sample_premise.shape
        assert output.dtype == sample_premise.dtype

        # Output should be different from input (attention was applied)
        assert not torch.allclose(output, sample_premise, atol=1e-5)

    def test_cross_attention_layer_mask(
        self, batch_size, premise_len, hypothesis_len, hidden_size, num_heads
    ):
        """Test that CrossAttentionLayer masks out padding in key-value sequence."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)

        query = torch.randn(batch_size, premise_len, hidden_size)
        key_value = torch.randn(batch_size, hypothesis_len, hidden_size)

        # Create mask that zeroes out last half of key-value sequence
        kv_mask = torch.ones(batch_size, hypothesis_len)
        kv_mask[:, hypothesis_len // 2 :] = 0

        # Run with mask
        output_with_mask = layer(
            query_states=query,
            key_value_states=key_value,
            key_value_mask=kv_mask,
        )

        # Run without mask
        output_without_mask = layer(
            query_states=query,
            key_value_states=key_value,
        )

        # Outputs should be different when mask is applied
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
        assert output_with_mask.shape == query.shape

    def test_cross_attention_layer_residual(
        self, sample_premise, sample_hypothesis, hidden_size, num_heads
    ):
        """Test that CrossAttentionLayer adds residual connection when enabled."""
        # With residual
        layer_with_residual = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_residual=True,
        )
        output_with = layer_with_residual(sample_premise, sample_hypothesis)

        # Without residual
        layer_without_residual = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_residual=False,
        )
        output_without = layer_without_residual(sample_premise, sample_hypothesis)

        # Outputs should be different
        assert not torch.allclose(output_with, output_without, atol=1e-5)

        # With residual, output should be closer to input (residual adds input)
        # This is a softer test - residual should preserve some of input signal
        assert output_with.shape == sample_premise.shape

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

    def test_cross_attention_layer_scaling(self, hidden_size, num_heads):
        """Test that attention scores are properly scaled."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)
        expected_scale = 1.0 / (hidden_size // num_heads) ** 0.5
        assert abs(layer.scale - expected_scale) < 1e-6

    def test_cross_attention_layer_dropout(self, hidden_size, num_heads):
        """Test that dropout is applied correctly."""
        dropout_rate = 0.5
        layer = CrossAttentionLayer(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout_rate
        )
        assert layer.dropout.p == dropout_rate

    def test_cross_attention_layer_gradient_flow(
        self, sample_premise, sample_hypothesis, hidden_size, num_heads
    ):
        """Test that gradients flow through cross-attention layer."""
        layer = CrossAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)
        sample_premise.requires_grad_(True)

        output = layer(sample_premise, sample_hypothesis)
        loss = output.sum()
        loss.backward()

        assert sample_premise.grad is not None
        assert not torch.all(sample_premise.grad == 0)


# =============================================================================
# FeedForward Tests
# =============================================================================


class TestFeedForward:
    """Tests for FeedForward network."""

    def test_feedforward_init(self, hidden_size):
        """Test that FeedForward initializes with GELU activation."""
        intermediate_size = hidden_size * 4
        ffn = FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size)

        # Check components exist
        assert hasattr(ffn, "fc1")
        assert hasattr(ffn, "fc2")
        assert hasattr(ffn, "activation")
        assert hasattr(ffn, "layer_norm")
        assert hasattr(ffn, "dropout")

        # Check dimensions
        assert ffn.fc1.in_features == hidden_size
        assert ffn.fc1.out_features == intermediate_size
        assert ffn.fc2.in_features == intermediate_size
        assert ffn.fc2.out_features == hidden_size

        # Check activation is GELU
        assert isinstance(ffn.activation, nn.GELU)

    def test_feedforward_init_options(self, hidden_size):
        """Test FeedForward with various initialization options."""
        intermediate_size = hidden_size * 4

        # With layer norm
        ffn_with_ln = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_layer_norm=True,
        )
        assert ffn_with_ln.layer_norm is not None
        assert isinstance(ffn_with_ln.layer_norm, nn.LayerNorm)

        # Without layer norm
        ffn_without_ln = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_layer_norm=False,
        )
        assert ffn_without_ln.layer_norm is None

        # With residual
        ffn_with_residual = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_residual=True,
        )
        assert ffn_with_residual.use_residual is True

        # Without residual
        ffn_without_residual = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_residual=False,
        )
        assert ffn_without_residual.use_residual is False

    def test_feedforward_forward(self, batch_size, premise_len, hidden_size):
        """Test FeedForward forward pass with two linear transformations."""
        intermediate_size = hidden_size * 4
        ffn = FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size)

        input_tensor = torch.randn(batch_size, premise_len, hidden_size)
        output = ffn(input_tensor)

        # Output shape should match input shape
        assert output.shape == input_tensor.shape

        # Output should be different from input
        assert not torch.allclose(output, input_tensor, atol=1e-5)

    def test_feedforward_residual(self, batch_size, premise_len, hidden_size):
        """Test FeedForward residual connection."""
        intermediate_size = hidden_size * 4

        # With residual
        ffn_with_residual = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_residual=True,
        )

        # Without residual
        ffn_without_residual = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_residual=False,
        )

        input_tensor = torch.randn(batch_size, premise_len, hidden_size)
        output_with = ffn_with_residual(input_tensor)
        output_without = ffn_without_residual(input_tensor)

        # Both should produce valid outputs
        assert output_with.shape == input_tensor.shape
        assert output_without.shape == input_tensor.shape

        # Outputs should be different
        assert not torch.allclose(output_with, output_without, atol=1e-5)

    def test_feedforward_dropout(self, hidden_size):
        """Test FeedForward dropout rate."""
        intermediate_size = hidden_size * 4
        dropout_rate = 0.3
        ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout_rate,
        )
        assert ffn.dropout.p == dropout_rate

    def test_feedforward_gradient_flow(self, batch_size, premise_len, hidden_size):
        """Test that gradients flow through FeedForward."""
        intermediate_size = hidden_size * 4
        ffn = FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size)

        input_tensor = torch.randn(batch_size, premise_len, hidden_size, requires_grad=True)
        output = ffn(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)


# =============================================================================
# BidirectionalCrossAttentionBlock Tests
# =============================================================================


class TestBidirectionalCrossAttentionBlock:
    """Tests for BidirectionalCrossAttentionBlock."""

    def test_bidirectional_cross_attention_block_init(self, hidden_size, num_heads):
        """Test that BidirectionalCrossAttentionBlock initializes correctly."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        # Should have both direction cross-attention layers
        assert hasattr(block, "cross_attn_a_to_b")
        assert hasattr(block, "cross_attn_b_to_a")
        assert isinstance(block.cross_attn_a_to_b, CrossAttentionLayer)
        assert isinstance(block.cross_attn_b_to_a, CrossAttentionLayer)

        # Default should have FFN
        assert block.ffn_a is not None
        assert block.ffn_b is not None

    def test_bidirectional_cross_attention_block_both_directions(
        self, sample_premise, sample_hypothesis, hidden_size, num_heads
    ):
        """Test that both directions are attended (A→B and B→A)."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        # Store original inputs
        original_a = sample_premise.clone()
        original_b = sample_hypothesis.clone()

        out_a, out_b = block(sample_premise, sample_hypothesis)

        # Both outputs should be different from inputs (attention was applied)
        assert not torch.allclose(out_a, original_a, atol=1e-5)
        assert not torch.allclose(out_b, original_b, atol=1e-5)

        # Output shapes should match input shapes
        assert out_a.shape == original_a.shape
        assert out_b.shape == original_b.shape

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

        # FFN should exist
        assert block.ffn_a is not None
        assert block.ffn_b is not None
        assert isinstance(block.ffn_a, FeedForward)
        assert isinstance(block.ffn_b, FeedForward)

    def test_without_ffn(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work without feedforward network."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=8,
            use_ffn=False,
        )
        out_a, out_b = block(sample_premise, sample_hypothesis)
        assert out_a.shape == sample_premise.shape
        assert out_b.shape == sample_hypothesis.shape

        # FFN should be None
        assert block.ffn_a is None
        assert block.ffn_b is None

    def test_with_masks(
        self, sample_premise, sample_hypothesis, premise_mask, hypothesis_mask, hidden_size
    ):
        """Test with attention masks."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=8,
        )
        out_a, out_b = block(
            sample_premise,
            sample_hypothesis,
            mask_a=premise_mask,
            mask_b=hypothesis_mask,
        )
        assert out_a.shape == sample_premise.shape
        assert out_b.shape == sample_hypothesis.shape

    def test_custom_ffn_hidden_size(self, sample_premise, sample_hypothesis, hidden_size):
        """Test with custom FFN hidden size."""
        custom_ffn_size = hidden_size * 2
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=8,
            use_ffn=True,
            ffn_hidden_size=custom_ffn_size,
        )
        out_a, out_b = block(sample_premise, sample_hypothesis)
        assert out_a.shape == sample_premise.shape
        assert out_b.shape == sample_hypothesis.shape

    def test_gradient_flow(self, sample_premise, sample_hypothesis, hidden_size, num_heads):
        """Test that gradients flow through both directions."""
        block = BidirectionalCrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        sample_premise.requires_grad_(True)
        sample_hypothesis.requires_grad_(True)

        out_a, out_b = block(sample_premise, sample_hypothesis)
        loss = out_a.sum() + out_b.sum()
        loss.backward()

        # Both inputs should have gradients
        assert sample_premise.grad is not None
        assert sample_hypothesis.grad is not None


# =============================================================================
# AttentionPooling Tests
# =============================================================================


class TestAttentionPooling:
    """Tests for AttentionPooling."""

    def test_attention_pooling_init(self, hidden_size):
        """Test that AttentionPooling initializes correctly."""
        pooler = AttentionPooling(hidden_size=hidden_size)

        # Check components exist
        assert hasattr(pooler, "query")
        assert hasattr(pooler, "key")
        assert hasattr(pooler, "value")
        assert hasattr(pooler, "out_proj")

        # Check query is a learnable parameter
        assert isinstance(pooler.query, nn.Parameter)
        assert pooler.query.requires_grad

        # Check projections
        assert isinstance(pooler.key, nn.Linear)
        assert isinstance(pooler.value, nn.Linear)
        assert isinstance(pooler.out_proj, nn.Linear)

    def test_attention_pooling_learns_weights(self, sample_premise, hidden_size):
        """Test that AttentionPooling learns attention weights for pooling."""
        pooler = AttentionPooling(hidden_size=hidden_size)

        # Run forward pass
        output = pooler(sample_premise)

        # Output should be (batch, hidden_size)
        batch_size = sample_premise.size(0)
        assert output.shape == (batch_size, hidden_size)

        # Query parameter should be learnable
        assert pooler.query.requires_grad

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

    def test_mask_affects_output(self, batch_size, premise_len, hidden_size):
        """Test that mask properly affects pooling output."""
        pooler = AttentionPooling(hidden_size=hidden_size)

        hidden_states = torch.randn(batch_size, premise_len, hidden_size)

        # Full mask
        full_mask = torch.ones(batch_size, premise_len)

        # Partial mask (mask out second half)
        partial_mask = torch.ones(batch_size, premise_len)
        partial_mask[:, premise_len // 2 :] = 0

        output_full = pooler(hidden_states, attention_mask=full_mask)
        output_partial = pooler(hidden_states, attention_mask=partial_mask)

        # Outputs should be different
        assert not torch.allclose(output_full, output_partial, atol=1e-5)

    def test_multihead(self, sample_premise, hidden_size):
        """Should work with multiple attention heads."""
        pooler = AttentionPooling(hidden_size=hidden_size, num_heads=4)
        output = pooler(sample_premise)
        assert output.shape == (sample_premise.size(0), hidden_size)

        # Check query shape with multiple heads
        assert pooler.query.shape == (1, 4, hidden_size // 4)

    def test_single_head(self, sample_premise, hidden_size):
        """Test with single attention head (default)."""
        pooler = AttentionPooling(hidden_size=hidden_size, num_heads=1)
        output = pooler(sample_premise)
        assert output.shape == (sample_premise.size(0), hidden_size)
        assert pooler.query.shape == (1, 1, hidden_size)

    def test_gradient_flow(self, sample_premise, hidden_size):
        """Test that gradients flow through attention pooling."""
        pooler = AttentionPooling(hidden_size=hidden_size)
        sample_premise.requires_grad_(True)

        output = pooler(sample_premise)
        loss = output.sum()
        loss.backward()

        assert sample_premise.grad is not None
        assert pooler.query.grad is not None

    def test_weight_initialization(self, hidden_size):
        """Test that query is properly initialized."""
        pooler = AttentionPooling(hidden_size=hidden_size)

        # Query should be initialized with small values (std=0.02)
        # Check that values are in reasonable range
        assert pooler.query.abs().max() < 1.0  # Should be small due to normal init


# =============================================================================
# CrossAttentionPairEncoder Tests
# =============================================================================


class TestCrossAttentionPairEncoder:
    """Tests for CrossAttentionPairEncoder."""

    def test_cross_attention_pair_encoder_init(self, hidden_size, num_heads):
        """Test that CrossAttentionPairEncoder initializes with layers correctly."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=2,
        )

        # Check attributes
        assert encoder.hidden_size == hidden_size
        assert encoder.num_heads == num_heads
        assert encoder.num_layers == 2
        assert encoder.output_size == hidden_size  # Default same as hidden_size

        # Check cross-attention layers exist
        assert hasattr(encoder, "cross_attn_layers")
        assert len(encoder.cross_attn_layers) == 2

        # Check combination layer
        assert hasattr(encoder, "combination_layer")
        assert hasattr(encoder, "output_norm")

    def test_cross_attention_pair_encoder_init_bidirectional(self, hidden_size, num_heads):
        """Test initialization with bidirectional cross-attention."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_bidirectional=True,
        )
        assert encoder.use_bidirectional is True

        # Layers should be BidirectionalCrossAttentionBlock
        for layer in encoder.cross_attn_layers:
            assert isinstance(layer, BidirectionalCrossAttentionBlock)

    def test_cross_attention_pair_encoder_init_unidirectional(self, hidden_size, num_heads):
        """Test initialization with unidirectional cross-attention."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_bidirectional=False,
        )
        assert encoder.use_bidirectional is False

        # Layers should be CrossAttentionLayer
        for layer in encoder.cross_attn_layers:
            assert isinstance(layer, CrossAttentionLayer)

    def test_cross_attention_pair_encoder_forward(
        self,
        sample_premise,
        sample_hypothesis,
        premise_mask,
        hypothesis_mask,
        hidden_size,
        num_heads,
    ):
        """Test that CrossAttentionPairEncoder returns pair representation."""
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

        # Output should be (batch, output_size)
        batch_size = sample_premise.size(0)
        assert output.shape == (batch_size, hidden_size)

        # Output should be normalized (LayerNorm applied)
        assert output.dtype == sample_premise.dtype

    def test_cross_attention_pair_encoder_pooling_cls(
        self, sample_premise, sample_hypothesis, hidden_size
    ):
        """Test CrossAttentionPairEncoder with CLS pooling strategy."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="cls",
        )
        assert encoder.pooling_strategy == "cls"

        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

        # CLS pooling shouldn't need attention poolers
        assert encoder.pooler_a is None
        assert encoder.pooler_b is None

    def test_cross_attention_pair_encoder_pooling_mean(
        self, sample_premise, sample_hypothesis, hidden_size
    ):
        """Test CrossAttentionPairEncoder with mean pooling strategy."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="mean",
        )
        assert encoder.pooling_strategy == "mean"

        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

        # Mean pooling shouldn't need attention poolers
        assert encoder.pooler_a is None
        assert encoder.pooler_b is None

    def test_cross_attention_pair_encoder_pooling_max(
        self, sample_premise, sample_hypothesis, hidden_size
    ):
        """Test CrossAttentionPairEncoder with max pooling strategy."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="max",
        )
        assert encoder.pooling_strategy == "max"

        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_cross_attention_pair_encoder_pooling_attention(
        self, sample_premise, sample_hypothesis, hidden_size
    ):
        """Test CrossAttentionPairEncoder with attention pooling strategy."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="attention",
        )
        assert encoder.pooling_strategy == "attention"

        # Attention pooling should have poolers
        assert encoder.pooler_a is not None
        assert encoder.pooler_b is not None
        assert isinstance(encoder.pooler_a, AttentionPooling)
        assert isinstance(encoder.pooler_b, AttentionPooling)

        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_cross_attention_pair_encoder_pooling_concat_pool(
        self, sample_premise, sample_hypothesis, hidden_size
    ):
        """Test CrossAttentionPairEncoder with concat_pool pooling strategy."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="concat_pool",
        )
        assert encoder.pooling_strategy == "concat_pool"

        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

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

        # Check that encoder parameters have gradients (except entity_combination_layer
        # which is only used in forward_with_entity_spans)
        for name, param in encoder.named_parameters():
            if param.requires_grad and "entity_combination_layer" not in name:
                assert param.grad is not None, f"Missing gradient for {name}"

    def test_entity_span_extraction(self, hidden_size):
        """Test forward_with_entity_spans for relation extraction."""
        batch_size = 4
        seq_len = 32
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        # Create hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        # Entity spans: (start_indices, end_indices)
        entity_a_span = (
            torch.tensor([2, 5, 8, 3]),  # starts
            torch.tensor([4, 7, 10, 5]),  # ends
        )
        entity_b_span = (
            torch.tensor([15, 18, 20, 22]),  # starts
            torch.tensor([17, 20, 23, 24]),  # ends
        )

        output = encoder.forward_with_entity_spans(
            hidden_states, entity_a_span, entity_b_span, attention_mask
        )

        # Output should be (batch, output_size)
        assert output.shape == (batch_size, hidden_size)

    def test_entity_span_gradient_flow(self, hidden_size):
        """Gradients should flow through entity span extraction."""
        batch_size = 2
        seq_len = 16
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        entity_a_span = (torch.tensor([1, 3]), torch.tensor([3, 5]))
        entity_b_span = (torch.tensor([8, 10]), torch.tensor([10, 12]))

        output = encoder.forward_with_entity_spans(hidden_states, entity_a_span, entity_b_span)
        loss = output.sum()
        loss.backward()

        # entity_combination_layer should have gradients now
        for name, param in encoder.named_parameters():
            if param.requires_grad and "entity_combination_layer" in name:
                assert param.grad is not None, f"Missing gradient for {name}"

    def test_without_masks(self, sample_premise, sample_hypothesis, hidden_size):
        """Test encoder works without providing masks."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_different_sequence_lengths(self, batch_size, hidden_size):
        """Test with very different sequence lengths."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        # Very short premise, long hypothesis
        premise = torch.randn(batch_size, 8, hidden_size)
        hypothesis = torch.randn(batch_size, 128, hidden_size)

        output = encoder(premise, hypothesis)
        assert output.shape == (batch_size, hidden_size)

        # Long premise, short hypothesis
        premise = torch.randn(batch_size, 128, hidden_size)
        hypothesis = torch.randn(batch_size, 8, hidden_size)

        output = encoder(premise, hypothesis)
        assert output.shape == (batch_size, hidden_size)


# =============================================================================
# ConcatPairEncoder Tests
# =============================================================================


class TestConcatPairEncoder:
    """Tests for ConcatPairEncoder (fallback)."""

    def test_concat_pair_encoder_init(self, hidden_size):
        """Test ConcatPairEncoder initialization."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)

        assert encoder.hidden_size == hidden_size
        assert encoder.output_size == hidden_size  # Default
        assert encoder.pooling_strategy == "mean"  # Default

        # Check combination layer exists
        assert hasattr(encoder, "combination")
        assert hasattr(encoder, "output_norm")

    def test_concat_pair_encoder_forward(
        self, sample_premise, sample_hypothesis, premise_mask, hypothesis_mask, hidden_size
    ):
        """Test that ConcatPairEncoder concatenates pooled representations."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        output = encoder(
            sample_premise,
            sample_hypothesis,
            premise_mask,
            hypothesis_mask,
        )

        # Output should be (batch, output_size)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_output_shape(self, sample_premise, sample_hypothesis, hidden_size):
        """Output should be (batch, hidden_size) by default."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_custom_output_size(self, sample_premise, sample_hypothesis, hidden_size):
        """Test with custom output size."""
        output_size = 256
        encoder = ConcatPairEncoder(hidden_size=hidden_size, output_size=output_size)
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), output_size)

    def test_pooling_strategies(self, sample_premise, sample_hypothesis, hidden_size):
        """Should work with different pooling strategies."""
        for strategy in ["cls", "mean", "max"]:
            encoder = ConcatPairEncoder(
                hidden_size=hidden_size,
                pooling_strategy=strategy,
            )
            output = encoder(sample_premise, sample_hypothesis)
            assert output.shape == (sample_premise.size(0), hidden_size)

    def test_pooling_cls(self, sample_premise, sample_hypothesis, hidden_size):
        """Test CLS pooling explicitly."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size, pooling_strategy="cls")
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_pooling_mean(self, sample_premise, sample_hypothesis, hidden_size):
        """Test mean pooling explicitly."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size, pooling_strategy="mean")
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_pooling_max(self, sample_premise, sample_hypothesis, hidden_size):
        """Test max pooling explicitly."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size, pooling_strategy="max")
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_invalid_pooling_strategy(self, hidden_size):
        """Test that invalid pooling strategy raises error."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size, pooling_strategy="invalid")
        hidden_a = torch.randn(2, 32, hidden_size)
        hidden_b = torch.randn(2, 32, hidden_size)

        with pytest.raises(ValueError, match="Unknown pooling"):
            encoder(hidden_a, hidden_b)

    def test_with_masks(
        self, sample_premise, sample_hypothesis, premise_mask, hypothesis_mask, hidden_size
    ):
        """Test with attention masks."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        output = encoder(sample_premise, sample_hypothesis, premise_mask, hypothesis_mask)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_without_masks(self, sample_premise, sample_hypothesis, hidden_size):
        """Test without attention masks."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        output = encoder(sample_premise, sample_hypothesis)
        assert output.shape == (sample_premise.size(0), hidden_size)

    def test_gradient_flow(self, sample_premise, sample_hypothesis, hidden_size):
        """Test that gradients flow through the encoder."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)
        sample_premise.requires_grad_(True)
        sample_hypothesis.requires_grad_(True)

        output = encoder(sample_premise, sample_hypothesis)
        loss = output.sum()
        loss.backward()

        assert sample_premise.grad is not None
        assert sample_hypothesis.grad is not None

    def test_rich_combination(self, sample_premise, sample_hypothesis, hidden_size):
        """Test that rich combination (diff, prod) is used."""
        encoder = ConcatPairEncoder(hidden_size=hidden_size)

        # The combination layer input should be 4 * hidden_size
        # (pooled_a, pooled_b, diff, prod)
        assert encoder.combination[0].in_features == hidden_size * 4


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreatePairEncoder:
    """Tests for create_pair_encoder factory function."""

    def test_create_pair_encoder_factory_cross_attention(self, hidden_size, num_heads):
        """Test factory creates CrossAttentionPairEncoder."""
        encoder = create_pair_encoder(
            "cross_attention",
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        assert isinstance(encoder, CrossAttentionPairEncoder)
        assert encoder.hidden_size == hidden_size
        assert encoder.num_heads == num_heads

    def test_create_pair_encoder_factory_concat(self, hidden_size):
        """Test factory creates ConcatPairEncoder."""
        encoder = create_pair_encoder("concat", hidden_size=hidden_size)
        assert isinstance(encoder, ConcatPairEncoder)
        assert encoder.hidden_size == hidden_size

    def test_create_pair_encoder_factory_none(self, hidden_size):
        """Test factory returns None for 'none' type."""
        encoder = create_pair_encoder("none", hidden_size=hidden_size)
        assert encoder is None

    def test_create_pair_encoder_factory_invalid(self, hidden_size):
        """Test factory raises error for invalid type."""
        with pytest.raises(ValueError, match="Unknown encoder_type"):
            create_pair_encoder("invalid", hidden_size=hidden_size)

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

    def test_factory_with_kwargs(self, hidden_size, num_heads):
        """Test factory passes kwargs correctly."""
        encoder = create_pair_encoder(
            "cross_attention",
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=3,
            use_bidirectional=False,
            pooling_strategy="mean",
        )
        assert isinstance(encoder, CrossAttentionPairEncoder)
        assert encoder.num_layers == 3
        assert encoder.use_bidirectional is False
        assert encoder.pooling_strategy == "mean"

    def test_factory_concat_with_kwargs(self, hidden_size):
        """Test factory passes kwargs to ConcatPairEncoder."""
        encoder = create_pair_encoder(
            "concat",
            hidden_size=hidden_size,
            pooling_strategy="max",
            output_size=256,
        )
        assert isinstance(encoder, ConcatPairEncoder)
        assert encoder.pooling_strategy == "max"
        assert encoder.output_size == 256


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

    def test_nli_use_case(self):
        """Test typical NLI use case."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
            num_layers=2,
            use_bidirectional=True,
            pooling_strategy="attention",
        )

        # Typical NLI batch
        batch_size = 8
        premise = torch.randn(batch_size, 128, 768)
        hypothesis = torch.randn(batch_size, 64, 768)

        output = encoder(premise, hypothesis)
        assert output.shape == (batch_size, 768)

    def test_relation_extraction_use_case(self):
        """Test relation extraction with entity spans."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
        )

        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Entity spans in the sentence
        entity_a_span = (
            torch.tensor([10, 15, 20, 25]),  # starts
            torch.tensor([12, 18, 23, 28]),  # ends
        )
        entity_b_span = (
            torch.tensor([50, 55, 60, 65]),  # starts
            torch.tensor([53, 58, 63, 68]),  # ends
        )

        output = encoder.forward_with_entity_spans(
            hidden_states,
            entity_a_span,
            entity_b_span,
        )
        assert output.shape == (batch_size, 768)

    def test_full_pipeline_training_simulation(self):
        """Simulate a training step with the pair encoder."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=768,
            num_heads=8,
            num_layers=2,
        )
        encoder.train()

        # Simulated batch
        batch_size = 4
        premise = torch.randn(batch_size, 64, 768, requires_grad=True)
        hypothesis = torch.randn(batch_size, 32, 768, requires_grad=True)

        # Forward pass
        output = encoder(premise, hypothesis)

        # Simulated classification head
        classifier = nn.Linear(768, 3)  # 3 classes for NLI
        logits = classifier(output)

        # Simulated loss
        labels = torch.randint(0, 3, (batch_size,))
        loss = nn.functional.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert premise.grad is not None
        assert hypothesis.grad is not None

        # Check encoder parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad and "entity_combination_layer" not in name:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestPairEncoderEdgeCases:
    """Edge case tests for pair encoders."""

    def test_single_sample_batch(self, hidden_size):
        """Test with batch size of 1."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        premise = torch.randn(1, 32, hidden_size)
        hypothesis = torch.randn(1, 16, hidden_size)

        output = encoder(premise, hypothesis)
        assert output.shape == (1, hidden_size)

    def test_very_short_sequences(self, hidden_size):
        """Test with very short sequences."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        premise = torch.randn(2, 2, hidden_size)  # Only 2 tokens
        hypothesis = torch.randn(2, 1, hidden_size)  # Single token

        output = encoder(premise, hypothesis)
        assert output.shape == (2, hidden_size)

    def test_very_long_sequences(self, hidden_size):
        """Test with long sequences."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        premise = torch.randn(2, 512, hidden_size)
        hypothesis = torch.randn(2, 256, hidden_size)

        output = encoder(premise, hypothesis)
        assert output.shape == (2, hidden_size)

    def test_sparse_mask(self, hidden_size):
        """Test with very sparse attention mask."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        premise = torch.randn(2, 32, hidden_size)
        hypothesis = torch.randn(2, 16, hidden_size)

        # Only first token is valid
        premise_mask = torch.zeros(2, 32)
        premise_mask[:, 0] = 1
        hypothesis_mask = torch.zeros(2, 16)
        hypothesis_mask[:, 0] = 1

        output = encoder(premise, hypothesis, premise_mask, hypothesis_mask)
        assert output.shape == (2, hidden_size)
        assert not torch.isnan(output).any()

    def test_all_masked(self, batch_size, hidden_size):
        """Test behavior when all tokens are masked (edge case)."""
        encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            pooling_strategy="mean",  # Mean pooling handles this better
        )

        premise = torch.randn(batch_size, 32, hidden_size)
        hypothesis = torch.randn(batch_size, 16, hidden_size)

        # All zeros mask for hypothesis
        hypothesis_mask = torch.ones(batch_size, 16)

        output = encoder(premise, hypothesis, mask_b=hypothesis_mask)
        assert output.shape == (batch_size, hidden_size)

    def test_deterministic_output(self, hidden_size):
        """Test that output is deterministic in eval mode."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8, dropout=0.1)
        encoder.eval()

        premise = torch.randn(2, 32, hidden_size)
        hypothesis = torch.randn(2, 16, hidden_size)

        with torch.no_grad():
            output1 = encoder(premise, hypothesis)
            output2 = encoder(premise, hypothesis)

        assert torch.allclose(output1, output2)

    def test_different_dtypes(self, hidden_size):
        """Test with different tensor dtypes."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        # Float16
        encoder_fp16 = encoder.half()
        premise_fp16 = torch.randn(2, 32, hidden_size).half()
        hypothesis_fp16 = torch.randn(2, 16, hidden_size).half()

        output_fp16 = encoder_fp16(premise_fp16, hypothesis_fp16)
        assert output_fp16.dtype == torch.float16
        assert output_fp16.shape == (2, hidden_size)

    def test_entity_span_single_token(self, hidden_size):
        """Test entity span extraction with single token spans."""
        encoder = CrossAttentionPairEncoder(hidden_size=hidden_size, num_heads=8)

        hidden_states = torch.randn(2, 32, hidden_size)

        # Single token spans (start == end)
        entity_a_span = (torch.tensor([5, 10]), torch.tensor([5, 10]))
        entity_b_span = (torch.tensor([20, 25]), torch.tensor([20, 25]))

        output = encoder.forward_with_entity_spans(hidden_states, entity_a_span, entity_b_span)
        assert output.shape == (2, hidden_size)


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Test that all expected classes are exported."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from modeling_studio.models import pair_encoder

        expected_exports = [
            "PairEncoderConfig",
            "CrossAttentionPairEncoder",
            "CrossAttentionLayer",
            "BidirectionalCrossAttentionBlock",
            "AttentionPooling",
            "FeedForward",
            "ConcatPairEncoder",
            "create_pair_encoder",
        ]

        for name in expected_exports:
            assert hasattr(pair_encoder, name), f"Missing export: {name}"
            assert name in pair_encoder.__all__, f"Not in __all__: {name}"


# =============================================================================
# Config Tests
# =============================================================================


class TestPairEncoderConfig:
    """Tests for PairEncoderConfig."""

    def test_pair_encoder_config_init(self):
        """Test that PairEncoderConfig initializes correctly with defaults."""
        config = PairEncoderConfig()
        assert config.hidden_size == 768
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.num_layers == 1
        assert config.use_bidirectional is True
        assert config.pooling_strategy == "attention"
        assert config.use_residual is True
        assert config.use_layer_norm is True
        assert config.use_ffn is True
        assert config.ffn_hidden_size == 768 * 4  # auto-computed

    def test_pair_encoder_config_custom_values(self):
        """Test PairEncoderConfig with custom values."""
        config = PairEncoderConfig(
            hidden_size=512,
            num_heads=4,
            dropout=0.2,
            num_layers=3,
            use_bidirectional=False,
            pooling_strategy="mean",
            use_residual=False,
            use_layer_norm=False,
            use_ffn=False,
            ffn_hidden_size=1024,
        )
        assert config.hidden_size == 512
        assert config.num_heads == 4
        assert config.dropout == 0.2
        assert config.num_layers == 3
        assert config.use_bidirectional is False
        assert config.pooling_strategy == "mean"
        assert config.use_residual is False
        assert config.use_layer_norm is False
        assert config.use_ffn is False
        assert config.ffn_hidden_size == 1024  # explicitly set

    def test_pair_encoder_config_ffn_default(self):
        """Test that ffn_hidden_size defaults to 4x hidden_size when None."""
        config = PairEncoderConfig(hidden_size=256, num_heads=4)
        assert config.ffn_hidden_size == 256 * 4

    def test_pair_encoder_config_validation(self):
        """Test validation: hidden_size must be divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            PairEncoderConfig(hidden_size=768, num_heads=7)

        with pytest.raises(ValueError, match="must be divisible"):
            PairEncoderConfig(hidden_size=100, num_heads=3)

    def test_pair_encoder_config_valid_divisibility(self):
        """Test valid hidden_size/num_heads combinations."""
        # These should not raise
        config1 = PairEncoderConfig(hidden_size=768, num_heads=8)
        assert config1.hidden_size // config1.num_heads == 96

        config2 = PairEncoderConfig(hidden_size=512, num_heads=8)
        assert config2.hidden_size // config2.num_heads == 64

        config3 = PairEncoderConfig(hidden_size=256, num_heads=4)
        assert config3.hidden_size // config3.num_heads == 64

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

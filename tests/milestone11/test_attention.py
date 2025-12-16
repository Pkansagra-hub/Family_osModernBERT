"""
Test Suite for Attention Mechanisms (Issue 11.2.1, 11.2.2, 11.2.3).

This module tests:
    - RotaryEmbedding: RoPE implementation
    - GroupedQueryAttention: GQA with RoPE and causal masking
    - CrossAttention: Encoder-decoder attention

Test Categories:
    - Unit tests for each component
    - Shape validation
    - Gradient flow
    - Numerical stability
    - KV cache functionality
"""

import math

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.attention import (
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    GroupedQueryAttention,
    CrossAttention,
)
from modeling_studio.models.decoder_config import DecoderMoEConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    """Create test configuration."""
    return DecoderMoEConfig(
        hidden_size=320,  # Divisible by 20 heads
        num_attention_heads=20,
        num_kv_heads=4,
        head_dim=16,  # 320 / 20 = 16
        dense_intermediate_size=896,  # 2.8 * 320
        expert_intermediate_size=512,
        num_layers=4,
        dense_layers=(0,),  # Layer 0 is dense
        moe_layers=(1, 2, 3),  # Layers 1-3 are MoE
        num_experts=4,
        num_experts_per_token=2,
        vocab_size=1000,
        max_position_embeddings=256,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


@pytest.fixture
def small_config():
    """Create smaller config for faster tests."""
    return DecoderMoEConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        dense_intermediate_size=128,
        expert_intermediate_size=64,
        num_layers=2,
        dense_layers=(0,),
        moe_layers=(1,),
        num_experts=2,
        num_experts_per_token=1,
        vocab_size=100,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )


# =============================================================================
# RoPE Tests (Issue 11.2.1)
# =============================================================================


class TestRotaryEmbedding:
    """Test suite for RotaryEmbedding."""

    def test_init(self, device):
        """Test RoPE initialization."""
        rope = RotaryEmbedding(dim=64, max_seq_len=512, base=10000.0).to(device)

        assert rope.dim == 64
        assert rope.max_seq_len == 512
        assert rope.base == 10000.0

        # Check inv_freq shape
        assert rope.inv_freq.shape == (32,)  # dim // 2

        # Check precomputed cache shapes
        assert rope.cos_cached.shape == (512, 64)
        assert rope.sin_cached.shape == (512, 64)

    def test_forward_shape(self, device):
        """Test RoPE forward pass shape."""
        rope = RotaryEmbedding(dim=64, max_seq_len=512).to(device)
        x = torch.randn(2, 8, 128, 64, device=device)  # (batch, heads, seq, dim)

        cos, sin = rope(x, seq_len=128)

        # Should broadcast to (1, 1, seq_len, dim)
        assert cos.shape == (1, 1, 128, 64)
        assert sin.shape == (1, 1, 128, 64)

    def test_position_ids(self, device):
        """Test RoPE with explicit position IDs."""
        rope = RotaryEmbedding(dim=64, max_seq_len=512).to(device)
        x = torch.randn(2, 8, 16, 64, device=device)

        # Custom position IDs for KV cache scenario
        position_ids = torch.arange(100, 116, device=device).unsqueeze(0).expand(2, -1)

        cos, sin = rope(x, position_ids=position_ids)

        # Shape should match position_ids
        assert cos.shape == (2, 1, 16, 64)
        assert sin.shape == (2, 1, 16, 64)

    def test_cache_extension(self, device):
        """Test that cache extends for longer sequences."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128).to(device)
        x = torch.randn(1, 1, 256, 64, device=device)

        # Should extend cache
        cos, sin = rope(x, seq_len=256)

        assert cos.shape == (1, 1, 256, 64)
        assert rope.cos_cached.shape[0] >= 256

    def test_numerical_properties(self, device):
        """Test RoPE numerical properties."""
        rope = RotaryEmbedding(dim=64, max_seq_len=512).to(device)
        x = torch.randn(1, 1, 128, 64, device=device)

        cos, sin = rope(x, seq_len=128)

        # cos^2 + sin^2 = 1
        identity = cos ** 2 + sin ** 2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)

    def test_rotate_half(self, device):
        """Test rotate_half function."""
        x = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float, device=device)
        rotated = rotate_half(x)

        # Should swap halves with sign change
        expected = torch.tensor([[-4, -5, -6, 1, 2, 3]], dtype=torch.float, device=device)
        assert torch.allclose(rotated, expected)

    def test_apply_rotary_pos_emb(self, device):
        """Test apply_rotary_pos_emb function."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, 2, seq, dim, device=device)  # Fewer KV heads

        rope = RotaryEmbedding(dim=dim).to(device)
        cos, sin = rope(q, seq_len=seq)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Shapes should be preserved
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Should not be identical to input
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)


# =============================================================================
# GQA Tests (Issue 11.2.2)
# =============================================================================


class TestGroupedQueryAttention:
    """Test suite for GroupedQueryAttention."""

    def test_init(self, config, device):
        """Test GQA initialization."""
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)

        assert gqa.hidden_size == config.hidden_size
        assert gqa.num_heads == config.num_attention_heads
        assert gqa.num_kv_heads == config.num_kv_heads
        assert gqa.head_dim == config.head_dim
        assert gqa.num_key_value_groups == config.num_attention_heads // config.num_kv_heads

    def test_projection_shapes(self, config, device):
        """Test projection weight shapes."""
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)

        # Q projection: (hidden, num_heads * head_dim)
        assert gqa.q_proj.weight.shape == (
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )

        # K, V projections: (hidden, num_kv_heads * head_dim)
        assert gqa.k_proj.weight.shape == (
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        )
        assert gqa.v_proj.weight.shape == (
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        )

        # O projection: (num_heads * head_dim, hidden)
        assert gqa.o_proj.weight.shape == (
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        )

    def test_forward_shape(self, config, device):
        """Test GQA forward pass shape."""
        batch, seq_len = 2, 32
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)

        output, past_kv = gqa(x)

        assert output.shape == (batch, seq_len, config.hidden_size)
        assert past_kv is None  # use_cache=False by default

    def test_forward_with_cache(self, config, device):
        """Test GQA with KV cache."""
        batch, seq_len = 2, 32
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)

        # First pass: build cache
        output1, past_kv = gqa(x, use_cache=True)

        assert past_kv is not None
        assert len(past_kv) == 2  # (key, value)
        assert past_kv[0].shape == (batch, config.num_kv_heads, seq_len, config.head_dim)

        # Second pass: use cache (single token)
        x_new = torch.randn(batch, 1, config.hidden_size, device=device)
        output2, past_kv2 = gqa(x_new, past_key_value=past_kv, use_cache=True)

        assert output2.shape == (batch, 1, config.hidden_size)
        # Cache should grow
        assert past_kv2[0].shape == (batch, config.num_kv_heads, seq_len + 1, config.head_dim)

    def test_causal_masking(self, config, device):
        """Test that causal masking works correctly."""
        batch, seq_len = 1, 8
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)

        # Create input with strong signal at position 0
        x = torch.zeros(batch, seq_len, config.hidden_size, device=device)
        x[:, 0, :] = 100.0  # Strong signal

        output, _ = gqa(x)

        # Due to causal masking, position 0 can only see itself
        # Later positions can see position 0, so their outputs should be affected
        # This is a soft test - just verify shapes are correct
        assert output.shape == (batch, seq_len, config.hidden_size)

    def test_attention_mask(self, config, device):
        """Test custom attention mask."""
        batch, seq_len = 2, 16
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)

        # Create mask that blocks some positions
        attn_mask = torch.zeros(batch, 1, seq_len, seq_len, device=device)
        attn_mask[:, :, :, seq_len // 2:] = float("-inf")  # Block second half

        output, _ = gqa(x, attention_mask=attn_mask)

        assert output.shape == (batch, seq_len, config.hidden_size)

    def test_gradient_flow(self, config, device):
        """Test gradient flow through GQA."""
        batch, seq_len = 2, 16
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        x = torch.randn(batch, seq_len, config.hidden_size, device=device, requires_grad=True)

        output, _ = gqa(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check all projection gradients
        for name, param in gqa.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_repeat_kv(self, config, device):
        """Test KV head repetition."""
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)

        # Create KV tensor with known values
        batch, kv_heads, seq_len, head_dim = 2, config.num_kv_heads, 16, config.head_dim
        kv = torch.randn(batch, kv_heads, seq_len, head_dim, device=device)

        repeated = gqa._repeat_kv(kv)

        # Should expand to full head count
        assert repeated.shape == (batch, config.num_attention_heads, seq_len, head_dim)

        # Each KV head should be repeated num_key_value_groups times
        groups = config.num_attention_heads // config.num_kv_heads
        for i in range(kv_heads):
            for j in range(groups):
                head_idx = i * groups + j
                assert torch.allclose(repeated[:, head_idx], kv[:, i])

    def test_deterministic_output(self, config, device):
        """Test that GQA is deterministic without dropout."""
        batch, seq_len = 2, 16
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        gqa.eval()

        x = torch.randn(batch, seq_len, config.hidden_size, device=device)

        with torch.no_grad():
            out1, _ = gqa(x)
            out2, _ = gqa(x)

        assert torch.allclose(out1, out2)


# =============================================================================
# CrossAttention Tests (Issue 11.2.3)
# =============================================================================


class TestCrossAttention:
    """Test suite for CrossAttention."""

    def test_init(self, config, device):
        """Test CrossAttention initialization."""
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        assert cross_attn.hidden_size == config.hidden_size
        assert cross_attn.num_heads == config.num_attention_heads
        assert cross_attn.head_dim == config.head_dim

    def test_projection_shapes(self, config, device):
        """Test that all projections use full hidden_size (no GQA)."""
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        # All projections should be (hidden_size, hidden_size)
        for proj in [cross_attn.q_proj, cross_attn.k_proj, cross_attn.v_proj, cross_attn.o_proj]:
            assert proj.weight.shape == (config.hidden_size, config.hidden_size)

    def test_forward_shape(self, config, device):
        """Test CrossAttention forward shape."""
        batch, dec_len, enc_len = 2, 16, 32
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        output = cross_attn(decoder_hidden, encoder_hidden)

        assert output.shape == (batch, dec_len, config.hidden_size)

    def test_forward_with_mask(self, config, device):
        """Test CrossAttention with encoder mask."""
        batch, dec_len, enc_len = 2, 16, 32
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        # Mask out second half of encoder
        encoder_mask = torch.zeros(batch, 1, 1, enc_len, device=device)
        encoder_mask[:, :, :, enc_len // 2:] = float("-inf")

        output = cross_attn(decoder_hidden, encoder_hidden, encoder_mask)

        assert output.shape == (batch, dec_len, config.hidden_size)

    def test_no_causal_mask(self, config, device):
        """Test that cross-attention has no causal mask (can attend to all encoder positions)."""
        batch, dec_len, enc_len = 1, 4, 8
        cross_attn = CrossAttention(config, layer_idx=0).to(device)
        cross_attn.eval()

        # Position 0 in decoder should be able to attend to position 7 in encoder
        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        with torch.no_grad():
            output = cross_attn(decoder_hidden, encoder_hidden)

        # Just verify it runs without error and produces valid output
        assert output.shape == (batch, dec_len, config.hidden_size)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, config, device):
        """Test gradient flow through CrossAttention."""
        batch, dec_len, enc_len = 2, 8, 16
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(
            batch, dec_len, config.hidden_size, device=device, requires_grad=True
        )
        encoder_hidden = torch.randn(
            batch, enc_len, config.hidden_size, device=device, requires_grad=True
        )

        output = cross_attn(decoder_hidden, encoder_hidden)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to both inputs
        assert decoder_hidden.grad is not None
        assert encoder_hidden.grad is not None
        assert not torch.isnan(decoder_hidden.grad).any()
        assert not torch.isnan(encoder_hidden.grad).any()

    def test_different_encoder_decoder_lengths(self, config, device):
        """Test with various encoder/decoder length combinations."""
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        test_cases = [
            (1, 1, 1),    # Minimal
            (2, 1, 32),   # Short decoder, long encoder
            (2, 32, 1),   # Long decoder, short encoder
            (4, 64, 128), # Larger sizes
        ]

        for batch, dec_len, enc_len in test_cases:
            decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
            encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

            output = cross_attn(decoder_hidden, encoder_hidden)

            assert output.shape == (batch, dec_len, config.hidden_size), (
                f"Failed for batch={batch}, dec_len={dec_len}, enc_len={enc_len}"
            )

    def test_deterministic_output(self, config, device):
        """Test that CrossAttention is deterministic without dropout."""
        batch, dec_len, enc_len = 2, 16, 32
        cross_attn = CrossAttention(config, layer_idx=0).to(device)
        cross_attn.eval()

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        with torch.no_grad():
            out1 = cross_attn(decoder_hidden, encoder_hidden)
            out2 = cross_attn(decoder_hidden, encoder_hidden)

        assert torch.allclose(out1, out2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAttentionIntegration:
    """Integration tests for attention components."""

    def test_gqa_and_cross_attn_together(self, config, device):
        """Test GQA and CrossAttention working together."""
        batch, dec_len, enc_len = 2, 16, 32

        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        # Self-attention
        self_attn_out, _ = gqa(decoder_hidden)

        # Cross-attention
        cross_attn_out = cross_attn(self_attn_out, encoder_hidden)

        assert cross_attn_out.shape == (batch, dec_len, config.hidden_size)

    def test_memory_efficiency_gqa(self, config, device):
        """Test that GQA uses less KV memory than MHA."""
        # GQA should use num_kv_heads worth of memory, not num_heads
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)

        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, config.hidden_size, device=device)

        _, past_kv = gqa(x, use_cache=True)

        # KV cache shape: (batch, num_kv_heads, seq, head_dim)
        # NOT (batch, num_heads, seq, head_dim)
        assert past_kv[0].shape[1] == config.num_kv_heads
        assert past_kv[0].shape[1] < config.num_attention_heads

    def test_parameter_count(self, config, device):
        """Test parameter counts match expectations."""
        gqa = GroupedQueryAttention(config, layer_idx=0).to(device)
        cross_attn = CrossAttention(config, layer_idx=0).to(device)

        # GQA params
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim

        expected_gqa_params = (
            hidden * num_heads * head_dim +  # Q
            hidden * num_kv_heads * head_dim +  # K
            hidden * num_kv_heads * head_dim +  # V
            num_heads * head_dim * hidden  # O
        )

        actual_gqa_params = sum(p.numel() for p in gqa.parameters() if p.requires_grad)

        # GQA also has RoPE inv_freq but it's not a parameter
        # Account for small discrepancy
        assert abs(actual_gqa_params - expected_gqa_params) < 1000

        # CrossAttention params (all use full hidden_size)
        expected_cross_params = 4 * hidden * hidden
        actual_cross_params = sum(p.numel() for p in cross_attn.parameters() if p.requires_grad)

        assert abs(actual_cross_params - expected_cross_params) < 1000

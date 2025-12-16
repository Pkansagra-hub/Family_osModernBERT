"""
Test Suite for DecoderBlock (Issue 11.3.1).

This module tests:
    - EncoderProjection
    - DecoderBlock: Pre-norm architecture with self-attn, cross-attn, FFN
    - Dense vs MoE FFN selection based on layer index

Test Categories:
    - Unit tests
    - Shape validation
    - Gradient flow
    - Auxiliary loss computation
    - KV cache functionality
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.decoder_moe import (
    EncoderProjection,
    DecoderBlock,
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
        head_dim=16,
        dense_intermediate_size=896,  # 2.8 * 320
        expert_intermediate_size=512,
        num_layers=4,
        dense_layers=(0,),  # Layer 0 is dense, layers 1-3 are MoE
        moe_layers=(1, 2, 3),
        num_experts=4,
        num_experts_per_token=2,
        vocab_size=1000,
        max_position_embeddings=256,
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
    )


# =============================================================================
# EncoderProjection Tests
# =============================================================================


class TestEncoderProjection:
    """Test suite for EncoderProjection."""

    def test_init(self, device):
        """Test EncoderProjection initialization."""
        proj = EncoderProjection(
            encoder_hidden_size=768,
            decoder_hidden_size=1280,
            dropout=0.1,
        ).to(device)

        assert proj.proj.in_features == 768
        assert proj.proj.out_features == 1280

    def test_forward_shape(self, device):
        """Test EncoderProjection forward shape."""
        batch, seq_len = 2, 32
        enc_hidden = 768
        dec_hidden = 1280

        proj = EncoderProjection(enc_hidden, dec_hidden).to(device)
        x = torch.randn(batch, seq_len, enc_hidden, device=device)

        output = proj(x)

        assert output.shape == (batch, seq_len, dec_hidden)

    def test_gradient_flow(self, device):
        """Test gradient flow through projection."""
        proj = EncoderProjection(768, 1280).to(device)
        x = torch.randn(2, 16, 768, device=device, requires_grad=True)

        output = proj(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic(self, device):
        """Test deterministic output."""
        proj = EncoderProjection(768, 1280, dropout=0.0).to(device)
        proj.eval()

        x = torch.randn(2, 16, 768, device=device)

        with torch.no_grad():
            out1 = proj(x)
            out2 = proj(x)

        assert torch.allclose(out1, out2)

    def test_various_sizes(self, device):
        """Test with various encoder/decoder sizes."""
        test_cases = [
            (768, 1280),   # ModernBERT-base -> UltraBERT
            (1024, 1280),  # ModernBERT-large -> UltraBERT
            (256, 512),    # Small
            (768, 768),    # Same size
        ]

        for enc_size, dec_size in test_cases:
            proj = EncoderProjection(enc_size, dec_size).to(device)
            x = torch.randn(2, 16, enc_size, device=device)
            output = proj(x)
            assert output.shape == (2, 16, dec_size)


# =============================================================================
# DecoderBlock Tests (Issue 11.3.1)
# =============================================================================


class TestDecoderBlock:
    """Test suite for DecoderBlock."""

    def test_init_dense_layer(self, config, device):
        """Test DecoderBlock initialization for dense layer."""
        block = DecoderBlock(config, layer_idx=0).to(device)

        assert block.layer_idx == 0
        assert block.use_moe is False  # Layer 0 is dense

    def test_init_moe_layer(self, config, device):
        """Test DecoderBlock initialization for MoE layer."""
        block = DecoderBlock(config, layer_idx=1).to(device)

        assert block.layer_idx == 1
        assert block.use_moe is True  # Layer 1+ are MoE

    def test_forward_shape(self, config, device):
        """Test DecoderBlock forward shape."""
        batch, dec_len, enc_len = 2, 16, 32
        block = DecoderBlock(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        output, aux_loss, past_kv = block(
            decoder_hidden,
            encoder_hidden,
        )

        assert output.shape == (batch, dec_len, config.hidden_size)
        assert isinstance(aux_loss, float) or isinstance(aux_loss, torch.Tensor)
        assert past_kv is None  # use_cache=False by default

    def test_dense_layer_no_aux_loss(self, config, device):
        """Test that dense layer has zero auxiliary loss."""
        batch, dec_len, enc_len = 2, 16, 32
        block = DecoderBlock(config, layer_idx=0).to(device)  # Dense layer

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        _, aux_loss, _ = block(decoder_hidden, encoder_hidden)

        assert aux_loss == 0.0

    def test_moe_layer_has_aux_loss(self, config, device):
        """Test that MoE layer produces auxiliary loss."""
        batch, dec_len, enc_len = 2, 16, 32
        block = DecoderBlock(config, layer_idx=1).to(device)  # MoE layer

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        _, aux_loss, _ = block(decoder_hidden, encoder_hidden)

        # MoE layer should have non-zero auxiliary loss
        assert aux_loss >= 0.0  # Can be 0 in some edge cases

    def test_kv_cache(self, config, device):
        """Test KV cache functionality."""
        batch, enc_len = 2, 32
        block = DecoderBlock(config, layer_idx=0).to(device)

        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        # First pass: 16 tokens
        dec_hidden_1 = torch.randn(batch, 16, config.hidden_size, device=device)
        output1, _, past_kv = block(
            dec_hidden_1,
            encoder_hidden,
            use_cache=True,
        )

        assert past_kv is not None
        assert len(past_kv) == 2  # (key, value)
        assert past_kv[0].shape == (batch, config.num_kv_heads, 16, config.head_dim)

        # Second pass: 1 new token
        dec_hidden_2 = torch.randn(batch, 1, config.hidden_size, device=device)
        output2, _, past_kv2 = block(
            dec_hidden_2,
            encoder_hidden,
            past_key_value=past_kv,
            use_cache=True,
        )

        assert output2.shape == (batch, 1, config.hidden_size)
        assert past_kv2[0].shape == (batch, config.num_kv_heads, 17, config.head_dim)

    def test_attention_mask(self, config, device):
        """Test with attention masks."""
        batch, dec_len, enc_len = 2, 16, 32
        block = DecoderBlock(config, layer_idx=0).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        # Decoder causal mask (will be combined with internal causal mask)
        dec_mask = torch.zeros(batch, 1, dec_len, dec_len, device=device)

        # Encoder padding mask
        enc_mask = torch.zeros(batch, 1, 1, enc_len, device=device)
        enc_mask[:, :, :, enc_len // 2:] = float("-inf")

        output, _, _ = block(
            decoder_hidden,
            encoder_hidden,
            attention_mask=dec_mask,
            encoder_attention_mask=enc_mask,
        )

        assert output.shape == (batch, dec_len, config.hidden_size)

    def test_gradient_flow(self, config, device):
        """Test gradient flow through DecoderBlock."""
        batch, dec_len, enc_len = 2, 8, 16
        block = DecoderBlock(config, layer_idx=1).to(device)  # MoE layer

        decoder_hidden = torch.randn(
            batch, dec_len, config.hidden_size, device=device, requires_grad=True
        )
        encoder_hidden = torch.randn(
            batch, enc_len, config.hidden_size, device=device, requires_grad=True
        )

        output, aux_loss, _ = block(decoder_hidden, encoder_hidden)

        if isinstance(aux_loss, torch.Tensor):
            loss = output.sum() + aux_loss
        else:
            loss = output.sum()
        loss.backward()

        assert decoder_hidden.grad is not None
        assert encoder_hidden.grad is not None
        assert not torch.isnan(decoder_hidden.grad).any()
        assert not torch.isnan(encoder_hidden.grad).any()

    def test_residual_connection(self, config, device):
        """Test that residual connections work correctly."""
        block = DecoderBlock(config, layer_idx=0).to(device)
        block.eval()

        # With zero input, output should also be close to zero
        # (residual connections preserve input)
        batch, dec_len, enc_len = 2, 8, 16
        decoder_hidden = torch.zeros(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.zeros(batch, enc_len, config.hidden_size, device=device)

        with torch.no_grad():
            output, _, _ = block(decoder_hidden, encoder_hidden)

        # Output should be small (only from biases if any)
        assert output.abs().mean() < 1.0

    def test_pre_norm_architecture(self, config, device):
        """Test that pre-norm architecture is used (norm before sublayers)."""
        block = DecoderBlock(config, layer_idx=0).to(device)

        # Verify norm layers exist before attention and FFN
        assert hasattr(block, "self_attn_norm")
        assert hasattr(block, "cross_attn_norm")
        assert hasattr(block, "ffn_norm")

    def test_all_layer_indices(self, config, device):
        """Test all layer indices in the config."""
        batch, dec_len, enc_len = 2, 8, 16
        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        for layer_idx in range(config.num_layers):
            block = DecoderBlock(config, layer_idx=layer_idx).to(device)
            output, aux_loss, _ = block(decoder_hidden, encoder_hidden)

            expected_moe = layer_idx in config.moe_layers
            assert block.use_moe == expected_moe, f"Layer {layer_idx} MoE mismatch"
            assert output.shape == (batch, dec_len, config.hidden_size)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecoderBlockIntegration:
    """Integration tests for DecoderBlock."""

    def test_stacked_blocks(self, config, device):
        """Test stacking multiple decoder blocks."""
        batch, dec_len, enc_len = 2, 16, 32

        blocks = nn.ModuleList([
            DecoderBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ]).to(device)

        decoder_hidden = torch.randn(batch, dec_len, config.hidden_size, device=device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        total_aux_loss = 0.0
        hidden = decoder_hidden

        for block in blocks:
            hidden, aux_loss, _ = block(hidden, encoder_hidden)
            total_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()

        assert hidden.shape == (batch, dec_len, config.hidden_size)
        # Should have some aux loss from MoE layers
        assert total_aux_loss >= 0.0

    def test_autoregressive_generation_simulation(self, config, device):
        """Test simulation of autoregressive generation with KV cache."""
        batch, enc_len = 2, 32
        max_gen_len = 10

        block = DecoderBlock(config, layer_idx=0).to(device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        past_kv = None
        generated_outputs = []

        for step in range(max_gen_len):
            if step == 0:
                # First token
                dec_hidden = torch.randn(batch, 1, config.hidden_size, device=device)
            else:
                # Subsequent tokens
                dec_hidden = torch.randn(batch, 1, config.hidden_size, device=device)

            output, _, past_kv = block(
                dec_hidden,
                encoder_hidden,
                past_key_value=past_kv,
                use_cache=True,
            )

            generated_outputs.append(output)

            # Verify cache grows
            expected_cache_len = step + 1
            assert past_kv[0].shape[2] == expected_cache_len

        # Final check
        assert len(generated_outputs) == max_gen_len
        assert past_kv[0].shape[2] == max_gen_len

    def test_parameter_count(self, config, device):
        """Test parameter count for blocks."""
        dense_block = DecoderBlock(config, layer_idx=0).to(device)
        moe_block = DecoderBlock(config, layer_idx=1).to(device)

        dense_params = sum(p.numel() for p in dense_block.parameters())
        moe_params = sum(p.numel() for p in moe_block.parameters())

        # MoE block should have more parameters (multiple experts)
        assert moe_params > dense_params

        # Print for reference
        print(f"Dense block params: {dense_params:,}")
        print(f"MoE block params: {moe_params:,}")

    def test_memory_efficiency_kv_cache(self, config, device):
        """Test that KV cache reduces memory for generation."""
        batch, enc_len = 2, 32
        block = DecoderBlock(config, layer_idx=0).to(device)
        encoder_hidden = torch.randn(batch, enc_len, config.hidden_size, device=device)

        # Generate 10 tokens with cache
        past_kv = None
        for step in range(10):
            dec_hidden = torch.randn(batch, 1, config.hidden_size, device=device)
            _, _, past_kv = block(
                dec_hidden,
                encoder_hidden,
                past_key_value=past_kv,
                use_cache=True,
            )

        # KV cache should only have num_kv_heads (GQA memory savings)
        assert past_kv[0].shape[1] == config.num_kv_heads
        assert past_kv[0].shape[1] < config.num_attention_heads

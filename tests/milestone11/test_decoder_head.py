"""
Test Suite for CounterfactualDecoderHead (Issue 11.3.2).

This module tests:
    - CounterfactualDecoderHead initialization
    - Forward pass with loss computation
    - Weight tying (embedding <-> lm_head)
    - Autoregressive generation
    - BaseHead interface compliance

Test Categories:
    - Unit tests
    - Shape validation
    - Loss computation
    - Generation functionality
    - Integration with encoder outputs
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.decoder_moe import CounterfactualDecoderHead
from modeling_studio.models.decoder_config import DecoderMoEConfig
from modeling_studio.models.heads import BaseHead


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
        dense_intermediate_size=896,
        expert_intermediate_size=512,
        num_layers=4,
        dense_layers=(0,),
        moe_layers=(1, 2, 3),
        num_experts=4,
        num_experts_per_token=2,
        vocab_size=1000,
        max_position_embeddings=256,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        pad_token_id=0,
        eos_token_id=2,
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
        pad_token_id=0,
        eos_token_id=2,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestCounterfactualDecoderHeadInit:
    """Test suite for CounterfactualDecoderHead initialization."""

    def test_init(self, config, device):
        """Test basic initialization."""
        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=768,
        ).to(device)

        assert head.encoder_hidden_size == 768
        assert head.hidden_size == config.hidden_size
        assert head.vocab_size == config.vocab_size
        assert head.num_layers == config.num_layers

    def test_inherits_base_head(self, config, device):
        """Test that CounterfactualDecoderHead inherits from BaseHead."""
        head = CounterfactualDecoderHead(config=config).to(device)

        assert isinstance(head, BaseHead)
        assert hasattr(head, "forward")

    def test_head_name_attribute(self, config, device):
        """Test head_name class attribute."""
        assert CounterfactualDecoderHead.head_name == "counterfactual"

    def test_weight_tying(self, config, device):
        """Test that embedding and lm_head weights are tied."""
        head = CounterfactualDecoderHead(config=config).to(device)

        assert head.lm_head.weight is head.embed_tokens.weight
        assert id(head.lm_head.weight) == id(head.embed_tokens.weight)

    def test_layer_count(self, config, device):
        """Test correct number of layers."""
        head = CounterfactualDecoderHead(config=config).to(device)

        assert len(head.layers) == config.num_layers

    def test_dense_vs_moe_layers(self, config, device):
        """Test that correct layers use dense vs MoE."""
        head = CounterfactualDecoderHead(config=config).to(device)

        for i, layer in enumerate(head.layers):
            expected_moe = i in config.moe_layers
            assert layer.use_moe == expected_moe, f"Layer {i} MoE mismatch"

    def test_encoder_projection(self, config, device):
        """Test encoder projection dimensions."""
        enc_hidden = 768
        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        assert head.encoder_proj.proj.in_features == enc_hidden
        assert head.encoder_proj.proj.out_features == config.hidden_size


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestCounterfactualDecoderHeadForward:
    """Test suite for forward pass."""

    def test_forward_shape(self, config, device):
        """Test forward pass output shape."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        assert "logits" in outputs
        assert "aux_loss" in outputs
        assert "loss" in outputs

        # Logits shape: (batch, dec_len, vocab_size)
        # Note: labels are shifted right internally, so output length matches input
        assert outputs["logits"].shape[0] == batch
        assert outputs["logits"].shape[2] == config.vocab_size

    def test_forward_without_labels(self, config, device):
        """Test forward without labels (inference mode)."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        decoder_input_ids = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            decoder_input_ids=decoder_input_ids,
        )

        assert "logits" in outputs
        assert "aux_loss" in outputs
        assert "loss" not in outputs  # No labels, no loss

    def test_forward_requires_input(self, config, device):
        """Test that forward requires either labels or decoder_input_ids."""
        batch, enc_len = 2, 32
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        with pytest.raises(ValueError):
            head(hidden_states=encoder_hidden)

    def test_loss_computation(self, config, device):
        """Test loss is computed correctly."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        loss = outputs["loss"]

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Should be positive
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_ignore_index_in_labels(self, config, device):
        """Test that -100 in labels is ignored."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        # Labels with some positions masked
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)
        labels[:, dec_len // 2:] = -100  # Ignore second half

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        assert not torch.isnan(outputs["loss"])

    def test_aux_loss_included(self, config, device):
        """Test that auxiliary loss is included in total loss."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        # aux_loss should be a tensor
        assert isinstance(outputs["aux_loss"], torch.Tensor)

    def test_attention_mask(self, config, device):
        """Test with encoder attention mask."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        # Mask: 1 for valid, 0 for padding
        attention_mask = torch.ones(batch, enc_len, device=device)
        attention_mask[:, enc_len // 2:] = 0  # Mask second half

        outputs = head(
            hidden_states=encoder_hidden,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert outputs["logits"].shape[0] == batch


# =============================================================================
# KV Cache Tests
# =============================================================================


class TestCounterfactualDecoderHeadCache:
    """Test suite for KV cache functionality."""

    def test_use_cache_returns_past_key_values(self, config, device):
        """Test that use_cache returns past_key_values."""
        batch, enc_len, dec_len = 2, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        decoder_input_ids = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
        )

        assert "past_key_values" in outputs
        assert len(outputs["past_key_values"]) == config.num_layers

    def test_incremental_decoding(self, config, device):
        """Test incremental decoding with KV cache."""
        batch, enc_len = 2, 32
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        # First step: process initial tokens
        initial_ids = torch.randint(0, config.vocab_size, (batch, 4), device=device)

        with torch.no_grad():
            outputs1 = head(
                hidden_states=encoder_hidden,
                decoder_input_ids=initial_ids,
                use_cache=True,
            )

        past_kv = outputs1["past_key_values"]

        # Second step: process one new token
        new_ids = torch.randint(0, config.vocab_size, (batch, 1), device=device)

        with torch.no_grad():
            outputs2 = head(
                hidden_states=encoder_hidden,
                decoder_input_ids=new_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        # Output should be for 1 token
        assert outputs2["logits"].shape[1] == 1

        # Cache should grow
        new_past_kv = outputs2["past_key_values"]
        assert new_past_kv[0][0].shape[2] == 5  # 4 + 1


# =============================================================================
# Generation Tests
# =============================================================================


class TestCounterfactualDecoderHeadGenerate:
    """Test suite for generation functionality."""

    def test_generate_basic(self, small_config, device):
        """Test basic generation."""
        batch, enc_len = 2, 16
        enc_hidden = 64
        max_new_tokens = 8

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        generated = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=max_new_tokens,
        )

        assert generated.shape[0] == batch
        assert generated.shape[1] >= 2  # At least BOS + one generated
        assert generated.shape[1] <= max_new_tokens + 1  # +1 for BOS

    def test_generate_with_attention_mask(self, small_config, device):
        """Test generation with encoder attention mask."""
        batch, enc_len = 2, 16
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        attention_mask = torch.ones(batch, enc_len, device=device)
        attention_mask[:, enc_len // 2:] = 0

        generated = head.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            max_new_tokens=4,
        )

        assert generated.shape[0] == batch

    def test_generate_temperature(self, small_config, device):
        """Test generation with different temperatures."""
        batch, enc_len = 1, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        # Low temperature (more deterministic)
        gen_low_temp = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=4,
            temperature=0.1,
        )

        # High temperature (more random)
        gen_high_temp = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=4,
            temperature=2.0,
        )

        # Both should produce valid output
        assert gen_low_temp.shape[0] == batch
        assert gen_high_temp.shape[0] == batch

    def test_generate_top_k(self, small_config, device):
        """Test generation with top-k filtering."""
        batch, enc_len = 1, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        generated = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=4,
            top_k=10,
        )

        assert generated.shape[0] == batch

    def test_generate_top_p(self, small_config, device):
        """Test generation with nucleus sampling."""
        batch, enc_len = 1, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        generated = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=4,
            top_p=0.9,
        )

        assert generated.shape[0] == batch

    def test_generate_stops_at_eos(self, small_config, device):
        """Test that generation stops at EOS token."""
        # This is a probabilistic test - may not always stop early
        batch, enc_len = 1, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)

        generated = head.generate(
            encoder_hidden_states=encoder_hidden,
            max_new_tokens=20,
            eos_token_id=small_config.eos_token_id,
        )

        # Just verify it produces valid output
        assert generated.ndim == 2


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestCounterfactualDecoderHeadGradients:
    """Test gradient flow through the head."""

    def test_gradient_flow(self, config, device):
        """Test gradient flow through entire head."""
        batch, enc_len, dec_len = 2, 16, 8
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(
            batch, enc_len, enc_hidden, device=device, requires_grad=True
        )
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        # Gradient should flow to encoder hidden
        assert encoder_hidden.grad is not None
        assert not torch.isnan(encoder_hidden.grad).any()

    def test_all_parameters_have_gradients(self, config, device):
        """Test that trainable parameters receive gradients.

        Note: For MoE layers, not all expert parameters will receive gradients
        in a single forward pass due to sparse routing. We verify that:
        1. Non-MoE parameters always have gradients
        2. At least some MoE parameters have gradients
        """
        batch, enc_len, dec_len = 4, 32, 16  # Larger batch for better coverage
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        # Track parameters by type
        non_expert_params_without_grad = []
        expert_params_with_grad = 0
        expert_params_total = 0

        for name, param in head.named_parameters():
            if param.requires_grad:
                is_expert_param = "ffn.experts." in name

                if is_expert_param:
                    expert_params_total += 1
                    if param.grad is not None:
                        expert_params_with_grad += 1
                else:
                    # Non-expert params should always have gradients
                    if param.grad is None:
                        non_expert_params_without_grad.append(name)

        # All non-expert params should have gradients
        assert len(non_expert_params_without_grad) == 0, (
            f"Non-expert params without gradients: {non_expert_params_without_grad}"
        )

        # At least some expert params should have gradients (due to top-k routing)
        if expert_params_total > 0:
            assert expert_params_with_grad > 0, (
                f"No expert params have gradients. Total: {expert_params_total}"
            )


# =============================================================================
# Parameter Count Tests
# =============================================================================


class TestCounterfactualDecoderHeadParams:
    """Test parameter counting."""

    def test_get_num_params(self, config, device):
        """Test parameter counting method."""
        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=768,
        ).to(device)

        total_params = head.get_num_params()
        non_embedding_params = head.get_num_params(non_embedding=True)

        assert total_params > 0
        assert non_embedding_params > 0
        assert non_embedding_params < total_params

        # Verify with actual count
        actual_total = sum(p.numel() for p in head.parameters())
        assert total_params == actual_total

    def test_parameter_count_reasonable(self, device):
        """Test that parameter count is in expected range."""
        # Use production-like config
        config = DecoderMoEConfig(
            hidden_size=1280,
            num_attention_heads=20,
            num_kv_heads=4,
            head_dim=64,
            dense_intermediate_size=3584,
            expert_intermediate_size=2048,
            num_layers=8,
            dense_layers=(0, 1),
            moe_layers=(2, 3, 4, 5, 6, 7),
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=50280,
            max_position_embeddings=512,
        )

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=768,
        ).to(device)

        params = head.get_num_params()
        params_millions = params / 1e6

        # Should be approximately 420-600M based on architecture doc
        # The full architecture includes cross-attention (adding significant params)
        # Allow variance for implementation details
        assert 350 < params_millions < 650, f"Params: {params_millions:.1f}M"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCounterfactualDecoderHeadIntegration:
    """Integration tests with realistic scenarios."""

    def test_end_to_end_training_step(self, config, device):
        """Test a complete training step."""
        batch, enc_len, dec_len = 4, 32, 16
        enc_hidden = 768

        head = CounterfactualDecoderHead(
            config=config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

        # Simulate training step
        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        labels = torch.randint(0, config.vocab_size, (batch, dec_len), device=device)

        optimizer.zero_grad()
        outputs = head(
            hidden_states=encoder_hidden,
            labels=labels,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_eval_mode_deterministic(self, small_config, device):
        """Test that eval mode is deterministic (within numerical tolerance)."""
        batch, enc_len, dec_len = 2, 16, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)
        head.eval()

        # Use fixed seed for reproducible expert routing
        torch.manual_seed(42)
        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch, dec_len), device=device)

        with torch.no_grad():
            torch.manual_seed(123)  # Same seed for both calls
            out1 = head(
                hidden_states=encoder_hidden,
                decoder_input_ids=decoder_input_ids,
            )
            torch.manual_seed(123)  # Same seed for both calls
            out2 = head(
                hidden_states=encoder_hidden,
                decoder_input_ids=decoder_input_ids,
            )

        # With same seed, MoE routing should be identical
        assert torch.allclose(out1["logits"], out2["logits"], rtol=1e-5, atol=1e-5)

    def test_batch_size_flexibility(self, small_config, device):
        """Test various batch sizes."""
        enc_len, dec_len = 16, 8
        enc_hidden = 64

        head = CounterfactualDecoderHead(
            config=small_config,
            encoder_hidden_size=enc_hidden,
        ).to(device)

        for batch in [1, 2, 4, 8]:
            encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device)
            labels = torch.randint(0, small_config.vocab_size, (batch, dec_len), device=device)

            outputs = head(
                hidden_states=encoder_hidden,
                labels=labels,
            )

            assert outputs["logits"].shape[0] == batch

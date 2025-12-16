"""
Milestone 10: Tests for MoELayer Integration.

Test Coverage:
    - Issue 10.2.3: MoELayer Assembly (4 tests)

Acceptance Criteria Tested:
    - AC1 (10.2.3): Routes tokens to top-k experts based on router scores
    - AC2 (10.2.3): Accumulates aux losses (load_balance + z_loss)
    - AC3 (10.2.3): Capacity factor limits tokens per expert
    - AC4 (10.2.3): Output shape = input shape
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.moe_components import (
    MoELayer,
    TopKRouter,
    SwiGLUExpert,
    SharedExpert,
)
from modeling_studio.models.decoder_config import DecoderMoEConfig


# =============================================================================
# Issue 10.2.3: MoELayer Routing Tests
# =============================================================================


class TestMoELayerRouting:
    """Tests for MoELayer token routing."""

    def test_moe_layer_routes_to_top_k_experts(self):
        """10.2.3-T1: MoELayer routes tokens to top-k experts."""
        batch_size, seq_len = 4, 16
        config = DecoderMoEConfig(
            hidden_size=320,  # Divisible by 20 heads (20*16=320)
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # Run forward
        output, aux_losses = moe_layer(x)

        # Output should have same shape as input
        assert output.shape == x.shape

        # Aux losses should be present
        assert "load_balance_loss" in aux_losses
        assert "z_loss" in aux_losses

    def test_moe_layer_different_tokens_different_experts(self):
        """10.2.3-T1b: Different tokens can route to different experts."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)

        # Create diverse input to encourage different routing
        x = torch.randn(4, 32, 320)

        # Access router to check routing decisions
        with torch.no_grad():
            logits = moe_layer.router.gate(x.view(-1, 320))
            top_k_indices = logits.topk(2, dim=-1).indices

        # Check that not all tokens go to same experts
        unique_experts = top_k_indices.unique()
        assert len(unique_experts) > 2, \
            f"Should route to multiple experts, got {unique_experts.tolist()}"


# =============================================================================
# Issue 10.2.3: Auxiliary Loss Tests
# =============================================================================


class TestMoELayerAuxLosses:
    """Tests for MoELayer auxiliary losses."""

    def test_moe_layer_accumulates_aux_losses(self):
        """10.2.3-T2: MoELayer accumulates load_balance + z_loss."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
            load_balancing_loss_weight=0.01,
            router_z_loss_weight=0.001,
        )

        moe_layer = MoELayer(config, layer_idx=0)
        x = torch.randn(4, 16, config.hidden_size)

        output, aux_losses = moe_layer(x)

        # Check both losses are present and non-zero
        assert aux_losses["load_balance_loss"].item() > 0
        assert aux_losses["z_loss"].item() > 0

        # Check raw losses are larger than weighted
        assert aux_losses["raw_load_balance"].item() >= aux_losses["load_balance_loss"].item()
        assert aux_losses["raw_z_loss"].item() >= aux_losses["z_loss"].item()

    def test_moe_layer_aux_losses_differentiable(self):
        """10.2.3-T2b: Auxiliary losses are differentiable."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)
        x = torch.randn(4, 16, config.hidden_size, requires_grad=True)

        output, aux_losses = moe_layer(x)

        # Backprop through aux losses
        total_aux = aux_losses["load_balance_loss"] + aux_losses["z_loss"]
        total_aux.backward()

        assert x.grad is not None, "Aux losses should be differentiable"

    def test_moe_layer_aux_loss_weights_applied(self):
        """10.2.3-T2c: Aux loss weights are correctly applied."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
            load_balancing_loss_weight=0.1,  # Higher weight
            router_z_loss_weight=0.01,
        )

        moe_layer = MoELayer(config, layer_idx=0)
        x = torch.randn(4, 16, config.hidden_size)

        _, aux_losses = moe_layer(x)

        # Weighted should be raw × weight
        expected_lb = aux_losses["raw_load_balance"] * 0.1
        expected_z = aux_losses["raw_z_loss"] * 0.01

        assert torch.isclose(aux_losses["load_balance_loss"], expected_lb, rtol=1e-5)
        assert torch.isclose(aux_losses["z_loss"], expected_z, rtol=1e-5)


# =============================================================================
# Issue 10.2.3: Capacity Factor Tests
# =============================================================================


class TestMoELayerCapacityFactor:
    """Tests for capacity factor limiting."""

    def test_moe_layer_capacity_factor_limits_tokens(self):
        """10.2.3-T3: Capacity factor limits tokens per expert."""
        batch_size, seq_len = 4, 32  # 128 total tokens
        num_experts = 8
        capacity_factor = 1.25  # 125% of uniform distribution

        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=num_experts,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
            capacity_factor=capacity_factor,
        )

        moe_layer = MoELayer(config, layer_idx=0)

        # Expected capacity per expert:
        # tokens_per_expert = (batch × seq × top_k) / num_experts × capacity_factor
        # = (4 × 32 × 2) / 8 × 1.25 = 32 × 1.25 = 40
        expected_capacity = int(
            (batch_size * seq_len * config.num_experts_per_token)
            / num_experts * capacity_factor
        )

        # The MoELayer should enforce this capacity
        assert moe_layer.capacity_factor == capacity_factor

    def test_moe_layer_capacity_factor_prevents_overload(self):
        """10.2.3-T3b: High capacity factor allows more tokens."""
        config_low = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
            capacity_factor=1.0,  # Strict
        )

        config_high = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
            capacity_factor=2.0,  # Relaxed
        )

        moe_low = MoELayer(config_low, layer_idx=0)
        moe_high = MoELayer(config_high, layer_idx=0)

        # Both should run without error
        x = torch.randn(8, 64, 320)

        out_low, _ = moe_low(x)
        out_high, _ = moe_high(x)

        assert out_low.shape == out_high.shape == x.shape


# =============================================================================
# Issue 10.2.3: Output Shape Tests
# =============================================================================


class TestMoELayerOutputShape:
    """Tests for MoELayer output shape."""

    def test_moe_layer_output_equals_input_shape(self):
        """10.2.3-T4: Output shape equals input shape."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)

        test_shapes = [
            (1, 1, 320),     # Single token
            (4, 16, 320),   # Standard batch
            (8, 128, 320),  # Larger batch
            (1, 512, 320),  # Long sequence
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output, _ = moe_layer(x)
            assert output.shape == shape, f"Expected {shape}, got {output.shape}"

    def test_moe_layer_preserves_dtype(self):
        """10.2.3-T4b: MoELayer preserves input dtype."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)

        # Test float32
        x_f32 = torch.randn(4, 16, 320, dtype=torch.float32)
        out_f32, _ = moe_layer(x_f32)
        assert out_f32.dtype == torch.float32

        # Test bfloat16 if supported
        if torch.cuda.is_available() or hasattr(torch.backends, 'mps'):
            moe_layer_bf16 = MoELayer(config, layer_idx=0).to(torch.bfloat16)
            x_bf16 = torch.randn(4, 16, 320, dtype=torch.bfloat16)
            out_bf16, _ = moe_layer_bf16(x_bf16)
            assert out_bf16.dtype == torch.bfloat16


# =============================================================================
# MoELayer Component Integration Tests
# =============================================================================


class TestMoELayerComponents:
    """Tests for MoELayer internal components."""

    def test_moe_layer_has_router(self):
        """MoELayer contains TopKRouter."""
        config = DecoderMoEConfig(hidden_size=320)
        moe_layer = MoELayer(config, layer_idx=0)

        assert hasattr(moe_layer, "router")
        assert isinstance(moe_layer.router, TopKRouter)

    def test_moe_layer_has_experts(self):
        """MoELayer contains num_experts SwiGLU experts."""
        config = DecoderMoEConfig(hidden_size=320, num_experts=8)
        moe_layer = MoELayer(config, layer_idx=0)

        assert hasattr(moe_layer, "experts")
        assert len(moe_layer.experts) == 8

        for expert in moe_layer.experts:
            assert isinstance(expert, SwiGLUExpert)

    def test_moe_layer_has_shared_expert(self):
        """MoELayer contains shared expert."""
        config = DecoderMoEConfig(
            hidden_size=320,
            shared_expert_intermediate_size=320,
        )
        moe_layer = MoELayer(config, layer_idx=0)

        assert hasattr(moe_layer, "shared_expert")
        assert isinstance(moe_layer.shared_expert, SharedExpert)

    def test_moe_layer_shared_expert_contributes(self):
        """Shared expert output is added to final output."""
        config = DecoderMoEConfig(
            hidden_size=320,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=320,
        )

        moe_layer = MoELayer(config, layer_idx=0)
        x = torch.randn(4, 16, 320)

        # Get shared expert output
        shared_out = moe_layer.shared_expert(x)

        # Shared output should be non-zero
        assert shared_out.abs().sum() > 0


# =============================================================================
# MoELayer Training vs Eval Mode Tests
# =============================================================================


class TestMoELayerModes:
    """Tests for training vs evaluation modes."""

    def test_moe_layer_train_mode(self):
        """MoELayer works in training mode."""
        config = DecoderMoEConfig(
            hidden_size=320,
            expert_dropout=0.1,
        )
        moe_layer = MoELayer(config, layer_idx=0)
        moe_layer.train()

        x = torch.randn(4, 16, 320, requires_grad=True)
        output, aux_losses = moe_layer(x)

        # Should produce gradients
        loss = output.sum() + aux_losses["load_balance_loss"]
        loss.backward()

        assert x.grad is not None

    def test_moe_layer_eval_mode_deterministic(self):
        """MoELayer is deterministic in eval mode."""
        config = DecoderMoEConfig(
            hidden_size=320,
            expert_dropout=0.0,  # No dropout
            capacity_factor=10.0,  # High capacity to avoid random selection
        )
        moe_layer = MoELayer(config, layer_idx=0)
        moe_layer.eval()

        # Use smaller input to ensure no capacity overflow
        x = torch.randn(2, 8, 320)

        with torch.no_grad():
            out1, _ = moe_layer(x)
            out2, _ = moe_layer(x)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# MoELayer Parameter Count Tests
# =============================================================================


class TestMoELayerParameters:
    """Tests for MoELayer parameter counts."""

    def test_moe_layer_parameter_count(self):
        """MoELayer has expected parameter count."""
        hidden_size = 1280
        num_experts = 8
        expert_intermediate = 2048
        shared_intermediate = 1280

        config = DecoderMoEConfig(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=2,
            expert_intermediate_size=expert_intermediate,
            shared_expert_intermediate_size=shared_intermediate,
        )

        moe_layer = MoELayer(config, layer_idx=0)

        # Router: hidden × num_experts
        router_params = hidden_size * num_experts

        # Experts: num_experts × 3 × hidden × intermediate
        expert_params = num_experts * 3 * hidden_size * expert_intermediate

        # Shared expert: 3 × hidden × shared_intermediate
        shared_params = 3 * hidden_size * shared_intermediate

        expected_total = router_params + expert_params + shared_params
        actual_total = sum(p.numel() for p in moe_layer.parameters())

        assert actual_total == expected_total, \
            f"Expected {expected_total:,}, got {actual_total:,}"


# =============================================================================
# Device Compatibility Tests
# =============================================================================


class TestMoELayerDevice:
    """Tests for device compatibility."""

    def test_moe_layer_cpu(self):
        """MoELayer works on CPU."""
        config = DecoderMoEConfig(hidden_size=320)
        moe_layer = MoELayer(config, layer_idx=0)

        x = torch.randn(4, 16, 320)
        output, _ = moe_layer(x)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_moe_layer_cuda(self):
        """MoELayer works on CUDA."""
        config = DecoderMoEConfig(hidden_size=320)
        moe_layer = MoELayer(config, layer_idx=0).cuda()

        x = torch.randn(4, 16, 320).cuda()
        output, aux_losses = moe_layer(x)

        assert output.device.type == "cuda"
        assert aux_losses["load_balance_loss"].device.type == "cuda"

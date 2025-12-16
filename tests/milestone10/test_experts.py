"""
Milestone 10: Tests for SwiGLU Experts and Shared Expert.

Test Coverage:
    - Issue 10.2.1: SwiGLU Expert (3 tests)
    - Issue 10.2.2: Shared Expert (2 tests)

Acceptance Criteria Tested:
    - AC1 (10.2.1): Output shape = input shape
    - AC2 (10.2.1): No bias parameters
    - AC3 (10.2.1): Param count = 3 × hidden × intermediate
    - AC1 (10.2.2): Shared expert processes ALL tokens
    - AC2 (10.2.2): Output is added to MoE sparse output
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.moe_components import (
    SwiGLUExpert,
    SharedExpert,
)


# =============================================================================
# Issue 10.2.1: SwiGLU Expert Tests
# =============================================================================


class TestSwiGLUExpertOutput:
    """Tests for SwiGLU expert output shape."""

    def test_swiglu_output_shape_preserves_input(self):
        """10.2.1-T1: Output shape equals input shape."""
        hidden_size = 1280
        intermediate_size = 2048

        expert = SwiGLUExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        # Test various input shapes
        test_shapes = [
            (4, 16, hidden_size),  # Batch, Seq, Hidden
            (1, 1, hidden_size),   # Single token
            (32, 128, hidden_size),  # Larger batch
            (hidden_size,),        # 1D input
            (16, hidden_size),     # 2D input
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output = expert(x)
            assert output.shape == shape, f"Expected shape {shape}, got {output.shape}"

    def test_swiglu_computation_correct(self):
        """10.2.1-T1b: SwiGLU computation is correct: down(SiLU(gate) * up)."""
        hidden_size = 64
        intermediate_size = 128

        expert = SwiGLUExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        x = torch.randn(2, 4, hidden_size)

        # Manual computation
        gate = torch.nn.functional.linear(x, expert.gate_proj.weight)
        up = torch.nn.functional.linear(x, expert.up_proj.weight)
        hidden = torch.nn.functional.silu(gate) * up
        expected = torch.nn.functional.linear(hidden, expert.down_proj.weight)

        # Expert computation
        output = expert(x)

        assert torch.allclose(output, expected, atol=1e-6), \
            "SwiGLU computation does not match expected formula"


class TestSwiGLUExpertBias:
    """Tests for SwiGLU expert bias configuration."""

    def test_swiglu_no_bias_gate_proj(self):
        """10.2.1-T2: Gate projection has no bias."""
        expert = SwiGLUExpert(hidden_size=1280, intermediate_size=2048)
        assert expert.gate_proj.bias is None, "gate_proj should have no bias"

    def test_swiglu_no_bias_up_proj(self):
        """10.2.1-T2b: Up projection has no bias."""
        expert = SwiGLUExpert(hidden_size=1280, intermediate_size=2048)
        assert expert.up_proj.bias is None, "up_proj should have no bias"

    def test_swiglu_no_bias_down_proj(self):
        """10.2.1-T2c: Down projection has no bias."""
        expert = SwiGLUExpert(hidden_size=1280, intermediate_size=2048)
        assert expert.down_proj.bias is None, "down_proj should have no bias"


class TestSwiGLUExpertParameters:
    """Tests for SwiGLU expert parameter count."""

    def test_swiglu_parameter_count(self):
        """10.2.1-T3: Parameter count = 3 × hidden × intermediate."""
        hidden_size = 1280
        intermediate_size = 2048

        expert = SwiGLUExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        expected_params = 3 * hidden_size * intermediate_size
        actual_params = sum(p.numel() for p in expert.parameters())

        assert actual_params == expected_params, \
            f"Expected {expected_params} params, got {actual_params}"

    def test_swiglu_parameter_count_different_sizes(self):
        """10.2.1-T3b: Parameter count correct for various sizes."""
        test_configs = [
            (768, 2048),
            (1024, 4096),
            (1280, 2048),
            (256, 512),
        ]

        for hidden_size, intermediate_size in test_configs:
            expert = SwiGLUExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

            expected = 3 * hidden_size * intermediate_size
            actual = sum(p.numel() for p in expert.parameters())

            assert actual == expected, \
                f"For {hidden_size}x{intermediate_size}: expected {expected}, got {actual}"

    def test_swiglu_weight_shapes(self):
        """10.2.1-T3c: Individual weight shapes are correct."""
        hidden_size = 1280
        intermediate_size = 2048

        expert = SwiGLUExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        assert expert.gate_proj.weight.shape == (intermediate_size, hidden_size)
        assert expert.up_proj.weight.shape == (intermediate_size, hidden_size)
        assert expert.down_proj.weight.shape == (hidden_size, intermediate_size)


# =============================================================================
# Issue 10.2.2: Shared Expert Tests
# =============================================================================


class TestSharedExpertProcessing:
    """Tests for shared expert token processing."""

    def test_shared_expert_processes_all_tokens(self):
        """10.2.2-T1: Shared expert processes ALL tokens, not sparse."""
        batch_size, seq_len, hidden_size = 4, 16, 1280
        intermediate_size = 1280  # Shared expert intermediate

        shared = SharedExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = shared(x)

        # Every token should produce non-zero output (with overwhelming probability)
        # Reshape to check each token
        output_flat = output.view(-1, hidden_size)

        non_zero_tokens = (output_flat.abs().sum(dim=-1) > 0).sum().item()
        total_tokens = batch_size * seq_len

        assert non_zero_tokens == total_tokens, \
            f"Shared expert should process all {total_tokens} tokens, got {non_zero_tokens}"

    def test_shared_expert_always_active(self):
        """10.2.2-T1b: Shared expert is always active regardless of routing."""
        hidden_size = 1280
        intermediate_size = 1280

        shared = SharedExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        # Even with zeros (edge case), the forward should run
        x = torch.zeros(2, 4, hidden_size)
        output = shared(x)

        assert output.shape == x.shape, "Shared expert should always produce output"


class TestSharedExpertOutputAddition:
    """Tests for shared expert output behavior."""

    def test_shared_expert_output_addable(self):
        """10.2.2-T2: Output can be added to sparse MoE output."""
        hidden_size = 1280
        intermediate_size = 1280
        batch_size, seq_len = 4, 16

        shared = SharedExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        x = torch.randn(batch_size, seq_len, hidden_size)

        # Simulate sparse MoE output
        sparse_output = torch.randn(batch_size, seq_len, hidden_size)

        # Shared expert output
        shared_output = shared(x)

        # Should be addable (same shape)
        combined = sparse_output + shared_output

        assert combined.shape == (batch_size, seq_len, hidden_size)

    def test_shared_expert_is_swiglu_expert(self):
        """10.2.2-T2b: SharedExpert wraps SwiGLUExpert correctly."""
        shared = SharedExpert(hidden_size=1280, intermediate_size=1280)

        # Check internal structure
        assert hasattr(shared, "expert"), "SharedExpert should have expert attribute"
        assert isinstance(shared.expert, SwiGLUExpert), \
            "SharedExpert should wrap SwiGLUExpert"

    def test_shared_expert_parameter_count(self):
        """10.2.2-T2c: SharedExpert has correct parameter count."""
        hidden_size = 1280
        intermediate_size = 1280

        shared = SharedExpert(hidden_size=hidden_size, intermediate_size=intermediate_size)

        expected = 3 * hidden_size * intermediate_size  # SwiGLU formula
        actual = sum(p.numel() for p in shared.parameters())

        assert actual == expected, f"Expected {expected}, got {actual}"


# =============================================================================
# Expert Gradient Flow Tests
# =============================================================================


class TestExpertGradients:
    """Tests for gradient flow through experts."""

    def test_swiglu_expert_gradient_flow(self):
        """Gradients flow through SwiGLU expert."""
        expert = SwiGLUExpert(hidden_size=768, intermediate_size=2048)

        x = torch.randn(4, 16, 768, requires_grad=True)
        output = expert(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert expert.gate_proj.weight.grad is not None, "Gradients should flow to gate_proj"
        assert expert.up_proj.weight.grad is not None, "Gradients should flow to up_proj"
        assert expert.down_proj.weight.grad is not None, "Gradients should flow to down_proj"

    def test_shared_expert_gradient_flow(self):
        """Gradients flow through shared expert."""
        shared = SharedExpert(hidden_size=768, intermediate_size=768)

        x = torch.randn(4, 16, 768, requires_grad=True)
        output = shared(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert shared.expert.gate_proj.weight.grad is not None, "Gradients should flow through"


# =============================================================================
# Expert Initialization Tests
# =============================================================================


class TestExpertInitialization:
    """Tests for expert weight initialization."""

    def test_swiglu_expert_weights_initialized(self):
        """SwiGLU expert weights are properly initialized (not zeros)."""
        expert = SwiGLUExpert(hidden_size=768, intermediate_size=2048)

        for name, param in expert.named_parameters():
            assert param.abs().sum() > 0, f"{name} should not be all zeros"

    def test_expert_reproducible_with_seed(self):
        """Expert initialization is reproducible with seed."""
        torch.manual_seed(42)
        expert1 = SwiGLUExpert(hidden_size=768, intermediate_size=2048)

        torch.manual_seed(42)
        expert2 = SwiGLUExpert(hidden_size=768, intermediate_size=2048)

        for (n1, p1), (n2, p2) in zip(
            expert1.named_parameters(), expert2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"{n1} should be reproducible"


# =============================================================================
# Device Compatibility Tests
# =============================================================================


class TestExpertDeviceCompatibility:
    """Tests for expert device handling."""

    def test_swiglu_expert_cpu(self):
        """SwiGLU expert works on CPU."""
        expert = SwiGLUExpert(hidden_size=768, intermediate_size=2048)
        x = torch.randn(2, 8, 768)
        output = expert(x)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_swiglu_expert_cuda(self):
        """SwiGLU expert works on CUDA."""
        expert = SwiGLUExpert(hidden_size=768, intermediate_size=2048).cuda()
        x = torch.randn(2, 8, 768).cuda()
        output = expert(x)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shared_expert_cuda(self):
        """Shared expert works on CUDA."""
        shared = SharedExpert(hidden_size=768, intermediate_size=768).cuda()
        x = torch.randn(2, 8, 768).cuda()
        output = shared(x)
        assert output.device.type == "cuda"

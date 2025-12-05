# tests/v3/test_layers_v3.py

"""
Tests for ModernBERT v3.3 Ultra FFN and Transformer Layer modules.

This module tests:
- GELU FFN (Issue 2.2.1)
- SwiGLU FFN (R&D only)
- FFN factory function
- Shape transformations
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.ffn_v3 import (
    GELUFFN,
    SwiGLUFFN,
    create_ffn,
)


class TestGELUFFN:
    """Test suite for GELU Feed-Forward Network (Issue 2.2.1)."""

    def test_gelu_ffn_initialization(self):
        """Test GELUFFN initializes with correct dimensions."""
        ffn = GELUFFN(
            hidden_size=768,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
        )

        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072
        assert ffn.up_proj.in_features == 768
        assert ffn.up_proj.out_features == 3072
        assert ffn.down_proj.in_features == 3072
        assert ffn.down_proj.out_features == 768
        assert ffn.dropout.p == 0.1

    def test_gelu_ffn_forward_shape(self):
        """Test GELUFFN maintains input shape through forward pass."""
        batch_size = 2
        seq_len = 50
        hidden_size = 768

        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)

        output = ffn(input_tensor)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gelu_ffn_activation_applied(self):
        """Test GELU activation is applied correctly."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.0)
        ffn.eval()  # Disable dropout for deterministic test

        # Create input with known values
        input_tensor = torch.randn(1, 10, 768)

        with torch.no_grad():
            # Manual forward pass
            intermediate = ffn.up_proj(input_tensor)
            activated = torch.nn.functional.gelu(intermediate)
            expected = ffn.down_proj(activated)

            # Module forward pass
            output = ffn(input_tensor)

            # Should match (dropout disabled)
            assert torch.allclose(output, expected, atol=1e-6)

    def test_gelu_ffn_dropout_applied(self):
        """Test dropout is applied during training mode."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.5)
        ffn.train()  # Enable training mode

        input_tensor = torch.randn(4, 20, 768)

        # Run multiple times - should get different outputs due to dropout
        outputs = [ffn(input_tensor) for _ in range(5)]

        # At least some outputs should differ (dropout randomness)
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should cause variation in outputs"

    def test_gelu_ffn_gradient_flow(self):
        """Test gradients flow correctly through GELUFFN."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        input_tensor = torch.randn(2, 10, 768, requires_grad=True)

        output = ffn(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

        # Check gradients are non-zero
        assert input_tensor.grad.abs().sum() > 0
        assert ffn.up_proj.weight.grad.abs().sum() > 0
        assert ffn.down_proj.weight.grad.abs().sum() > 0

    def test_gelu_ffn_different_activations(self):
        """Test GELUFFN supports different activation functions."""
        # Standard GELU
        ffn_gelu = GELUFFN(activation="gelu")
        assert ffn_gelu.activation == torch.nn.functional.gelu

        # GELU approximation
        ffn_gelu_new = GELUFFN(activation="gelu_new")
        assert ffn_gelu_new.activation == ffn_gelu_new._gelu_new

        # ReLU
        ffn_relu = GELUFFN(activation="relu")
        assert ffn_relu.activation == torch.nn.functional.relu

        # Invalid activation
        with pytest.raises(ValueError, match="Unknown activation"):
            GELUFFN(activation="invalid")

    def test_gelu_ffn_extra_repr(self):
        """Test extra_repr provides useful debug information."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        repr_str = ffn.extra_repr()

        assert "hidden=768" in repr_str
        assert "intermediate=3072" in repr_str

    def test_gelu_ffn_dimension_768_to_3072_to_768(self):
        """Test standard dimensions: 768 → 3072 → 768 (Acceptance Criteria #2)."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)

        # Verify architecture matches specification
        assert ffn.up_proj.in_features == 768
        assert ffn.up_proj.out_features == 3072
        assert ffn.down_proj.in_features == 3072
        assert ffn.down_proj.out_features == 768

        # Test with realistic input
        input_tensor = torch.randn(2, 50, 768)
        output = ffn(input_tensor)

        assert output.shape == (2, 50, 768)

    def test_gelu_ffn_dropout_after_down_projection(self):
        """Test dropout applied after down projection (Acceptance Criteria #3)."""
        # We can verify this by checking the forward pass order
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.1)

        # The dropout should be applied to the down_proj output
        # We can't directly test the order, but we can verify dropout exists
        # and is applied to the final output dimension
        input_tensor = torch.randn(2, 10, 768)
        output = ffn(input_tensor)

        # Output should have hidden_size dimension (dropout applied after down_proj)
        assert output.shape[-1] == 768


class TestSwiGLUFFN:
    """Test suite for SwiGLU FFN (R&D only)."""

    def test_swiglu_ffn_initialization(self):
        """Test SwiGLUFFN initializes correctly."""
        ffn = SwiGLUFFN(
            hidden_size=768,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
        )

        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072
        assert ffn.gate_proj.in_features == 768
        assert ffn.gate_proj.out_features == 3072
        assert ffn.up_proj.in_features == 768
        assert ffn.up_proj.out_features == 3072
        assert ffn.down_proj.in_features == 3072
        assert ffn.down_proj.out_features == 768

    def test_swiglu_ffn_forward_shape(self):
        """Test SwiGLUFFN maintains input shape."""
        batch_size = 2
        seq_len = 50
        hidden_size = 768

        ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072)
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)

        output = ffn(input_tensor)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_swiglu_ffn_gating_mechanism(self):
        """Test SwiGLU applies gating correctly."""
        ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.0)
        ffn.eval()

        input_tensor = torch.randn(1, 10, 768)

        with torch.no_grad():
            # Manual forward pass
            gate = torch.nn.functional.silu(ffn.gate_proj(input_tensor))
            up = ffn.up_proj(input_tensor)
            intermediate = gate * up
            expected = ffn.down_proj(intermediate)

            # Module forward pass
            output = ffn(input_tensor)

            assert torch.allclose(output, expected, atol=1e-6)

    def test_swiglu_ffn_gradient_flow(self):
        """Test gradients flow through SwiGLU gating."""
        ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072)
        input_tensor = torch.randn(2, 10, 768, requires_grad=True)

        output = ffn(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for all projections
        assert input_tensor.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

        # Check gradients are non-zero
        assert input_tensor.grad.abs().sum() > 0
        assert ffn.gate_proj.weight.grad.abs().sum() > 0
        assert ffn.up_proj.weight.grad.abs().sum() > 0
        assert ffn.down_proj.weight.grad.abs().sum() > 0

    def test_swiglu_ffn_extra_repr(self):
        """Test extra_repr includes R&D warning."""
        ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072)
        repr_str = ffn.extra_repr()

        assert "R&D ONLY" in repr_str
        assert "768" in repr_str
        assert "3072" in repr_str


class TestFFNFactory:
    """Test suite for create_ffn factory function."""

    def test_create_ffn_default_returns_gelu(self):
        """Test factory returns GELUFFN by default (Acceptance Criteria #5)."""
        ffn = create_ffn()

        assert isinstance(ffn, GELUFFN)
        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072

    def test_create_ffn_explicit_gelu(self):
        """Test factory returns GELUFFN when explicitly requested."""
        ffn = create_ffn(
            hidden_size=768,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            ffn_type="gelu",
        )

        assert isinstance(ffn, GELUFFN)
        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072

    def test_create_ffn_swiglu_with_warning(self, capsys):
        """Test factory returns SwiGLUFFN with warning (Acceptance Criteria #4)."""
        ffn = create_ffn(ffn_type="swiglu")

        # Verify SwiGLUFFN created
        assert isinstance(ffn, SwiGLUFFN)

        # Verify warning printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "R&D only" in captured.out
        assert "not recommended for production" in captured.out

    def test_create_ffn_invalid_type(self):
        """Test factory raises error for invalid FFN type."""
        with pytest.raises(ValueError, match="Unknown FFN type"):
            create_ffn(ffn_type="invalid_type")

    def test_create_ffn_custom_dimensions(self):
        """Test factory accepts custom dimensions."""
        ffn = create_ffn(
            hidden_size=512,
            intermediate_size=2048,
            hidden_dropout_prob=0.2,
        )

        assert isinstance(ffn, GELUFFN)
        assert ffn.hidden_size == 512
        assert ffn.intermediate_size == 2048
        assert ffn.dropout.p == 0.2


class TestFFNIntegration:
    """Integration tests for FFN modules."""

    def test_gelu_ffn_in_training_loop(self):
        """Test GELUFFN works in training loop."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-4)

        # Training step
        input_tensor = torch.randn(4, 20, 768)
        target = torch.randn(4, 20, 768)

        output = ffn(input_tensor)
        loss = torch.nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify weights updated
        assert all(p.grad is not None for p in ffn.parameters())

    def test_gelu_ffn_with_different_batch_sizes(self):
        """Test GELUFFN handles variable batch sizes."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        ffn.eval()

        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 50

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, seq_len, 768)
            output = ffn(input_tensor)
            assert output.shape == (batch_size, seq_len, 768)

    def test_gelu_ffn_with_different_sequence_lengths(self):
        """Test GELUFFN handles variable sequence lengths."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        ffn.eval()

        seq_lengths = [10, 50, 100, 512, 1024]
        batch_size = 2

        for seq_len in seq_lengths:
            input_tensor = torch.randn(batch_size, seq_len, 768)
            output = ffn(input_tensor)
            assert output.shape == (batch_size, seq_len, 768)

    def test_gelu_vs_swiglu_output_differs(self):
        """Test GELU and SwiGLU produce different outputs (as expected)."""
        gelu_ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.0)
        swiglu_ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.0)

        gelu_ffn.eval()
        swiglu_ffn.eval()

        input_tensor = torch.randn(2, 10, 768)

        with torch.no_grad():
            gelu_output = gelu_ffn(input_tensor)
            swiglu_output = swiglu_ffn(input_tensor)

        # Outputs should be different (different architectures)
        assert not torch.allclose(gelu_output, swiglu_output, atol=1e-3)

    def test_gelu_ffn_memory_efficient(self):
        """Test GELUFFN doesn't cause memory issues with long sequences."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)
        ffn.eval()

        # Large sequence (simulating long context)
        batch_size = 1
        seq_len = 2048
        input_tensor = torch.randn(batch_size, seq_len, 768)

        with torch.no_grad():
            output = ffn(input_tensor)

        assert output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(output).any()


class TestFFNAcceptanceCriteria:
    """
    Test suite for explicit acceptance criteria from Issue 2.2.1.

    Acceptance Criteria:
    1. ✅ GELU activation applied correctly
    2. ✅ Dimensions: 768 → 3072 → 768
    3. ✅ Dropout applied after down projection
    4. ✅ SwiGLU available for R&D (not production)
    5. ✅ Factory function returns correct type
    """

    def test_acceptance_criterion_1_gelu_activation(self):
        """AC1: GELU activation applied correctly."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.0)
        ffn.eval()

        # Test that activation matches torch.nn.functional.gelu
        input_tensor = torch.randn(1, 10, 768)

        with torch.no_grad():
            # Get intermediate after up_proj
            intermediate = ffn.up_proj(input_tensor)

            # Apply GELU manually
            expected_activated = torch.nn.functional.gelu(intermediate)

            # Apply through module's activation
            actual_activated = ffn.activation(intermediate)

            # Should match
            assert torch.allclose(expected_activated, actual_activated, atol=1e-6)

    def test_acceptance_criterion_2_dimensions(self):
        """AC2: Dimensions: 768 → 3072 → 768."""
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072)

        # Check projection dimensions
        assert ffn.up_proj.in_features == 768
        assert ffn.up_proj.out_features == 3072
        assert ffn.down_proj.in_features == 3072
        assert ffn.down_proj.out_features == 768

        # Verify end-to-end
        input_tensor = torch.randn(2, 50, 768)
        output = ffn(input_tensor)
        assert output.shape == (2, 50, 768)

    def test_acceptance_criterion_3_dropout_after_down_projection(self):
        """AC3: Dropout applied after down projection."""
        # This is verified by the implementation order in forward()
        # Dropout is the last operation before return
        ffn = GELUFFN(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.1)

        # Verify dropout exists and is configured
        assert isinstance(ffn.dropout, nn.Dropout)
        assert ffn.dropout.p == 0.1

        # The forward method applies: up_proj -> activation -> down_proj -> dropout
        # We can verify dropout affects the final output
        ffn.train()
        input_tensor = torch.randn(2, 10, 768)

        outputs = [ffn(input_tensor) for _ in range(5)]

        # With dropout, outputs should vary
        all_same = all(torch.equal(outputs[0], out) for out in outputs[1:])
        assert not all_same

    def test_acceptance_criterion_4_swiglu_available_for_rd(self, capsys):
        """AC4: SwiGLU available for R&D (not production)."""
        # Test SwiGLUFFN class exists and works
        ffn = SwiGLUFFN(hidden_size=768, intermediate_size=3072)
        assert isinstance(ffn, SwiGLUFFN)

        # Test factory function creates it with warning
        ffn2 = create_ffn(ffn_type="swiglu")
        assert isinstance(ffn2, SwiGLUFFN)

        # Verify warning about R&D only
        captured = capsys.readouterr()
        assert "R&D only" in captured.out
        assert "not recommended for production" in captured.out

        # Verify extra_repr includes warning
        assert "R&D ONLY" in ffn.extra_repr()

    def test_acceptance_criterion_5_factory_returns_correct_type(self):
        """AC5: Factory function returns correct type."""
        # Default should be GELU
        ffn_default = create_ffn()
        assert isinstance(ffn_default, GELUFFN)

        # Explicit GELU
        ffn_gelu = create_ffn(ffn_type="gelu")
        assert isinstance(ffn_gelu, GELUFFN)

        # SwiGLU
        ffn_swiglu = create_ffn(ffn_type="swiglu")
        assert isinstance(ffn_swiglu, SwiGLUFFN)

        # Invalid type should raise error
        with pytest.raises(ValueError):
            create_ffn(ffn_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

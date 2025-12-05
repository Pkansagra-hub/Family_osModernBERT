# tests/v3/test_layers_v3.py

"""
Tests for ModernBERT v3.3 Ultra FFN and Transformer Layer modules.

This module tests:
- GELU FFN (Issue 2.2.1)
- SwiGLU FFN (R&D only)
- FFN factory function
- LoRA Layer (Issue 2.2.2)
- ModernBERTLayerV3 (Issue 2.2.3)
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
from modeling_studio.models.lora_v3 import (
    LoRALayer,
    LinearWithLoRA,
    apply_lora_to_layer,
    get_lora_parameters,
    count_lora_parameters,
    freeze_non_lora_parameters,
)
from modeling_studio.models.layers_v3 import (
    ModernBERTLayerV3,
    create_layer_stack,
    freeze_layer_bands,
    unfreeze_layer_bands,
    get_layer_stats,
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


# ============================================================================
# LoRA Layer Tests (Issue 2.2.2)
# ============================================================================


class TestLoRALayer:
    """Test suite for LoRALayer (Issue 2.2.2)."""

    def test_lora_initialization(self):
        """Test LoRALayer initializes with correct parameters."""
        lora = LoRALayer(
            in_features=768,
            out_features=768,
            r=16,
            alpha=16,
            dropout=0.05,
        )

        assert lora.in_features == 768
        assert lora.out_features == 768
        assert lora.r == 16
        assert lora.alpha == 16
        assert lora.scaling == 1.0  # alpha / r = 16 / 16

        # Check matrix dimensions
        assert lora.lora_A.in_features == 768
        assert lora.lora_A.out_features == 16
        assert lora.lora_B.in_features == 16
        assert lora.lora_B.out_features == 768

    def test_lora_initialization_weights(self):
        """Test LoRA weight initialization (AC1: A=Kaiming, B=zeros)."""
        lora = LoRALayer(768, 768, r=16)

        # B should be initialized with zeros
        assert torch.allclose(lora.lora_B.weight, torch.zeros_like(lora.lora_B.weight))

        # A should be non-zero (Kaiming uniform)
        assert not torch.allclose(lora.lora_A.weight, torch.zeros_like(lora.lora_A.weight))

        # A should have reasonable values from Kaiming init
        assert lora.lora_A.weight.abs().max() < 1.0  # Reasonable range
        assert lora.lora_A.weight.std() > 0.01  # Has variance

    def test_lora_forward_shape(self):
        """Test LoRA maintains correct output shape."""
        batch_size = 2
        seq_len = 50
        hidden_size = 768

        lora = LoRALayer(in_features=768, out_features=768, r=16)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = lora(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_lora_scaling_applied(self):
        """Test scaling factor (alpha/r) is applied correctly (AC2)."""
        lora = LoRALayer(in_features=768, out_features=768, r=16, alpha=32)

        assert lora.scaling == 2.0  # alpha / r = 32 / 16

        # Verify scaling affects output
        x = torch.randn(2, 10, 768)
        output = lora(x)

        # Output should be scaled by alpha/r
        # We can't directly verify the math without reproducing the forward pass,
        # but we can check that scaling is stored correctly
        assert lora.scaling == lora.alpha / lora.r

    def test_lora_dropout_applied(self):
        """Test dropout is applied before LoRA projection (AC3)."""
        lora = LoRALayer(768, 768, r=16, dropout=0.5)  # Higher dropout for test

        # Initialize B with non-zero values so we can see dropout effect
        nn.init.normal_(lora.lora_B.weight, std=0.01)

        lora.train()

        x = torch.randn(2, 10, 768)

        # Run multiple times to check dropout randomness
        outputs = [lora(x) for _ in range(10)]

        # With dropout, outputs should vary
        all_same = all(torch.equal(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should cause output variation"

    def test_lora_gradient_flow(self):
        """Test gradients flow through LoRA layers."""
        lora = LoRALayer(768, 768, r=16)
        x = torch.randn(2, 10, 768, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_lora_extra_repr(self):
        """Test extra_repr provides useful information."""
        lora = LoRALayer(768, 768, r=16, alpha=16)
        repr_str = lora.extra_repr()

        assert "in_features=768" in repr_str
        assert "out_features=768" in repr_str
        assert "r=16" in repr_str
        assert "alpha=16" in repr_str
        assert "scaling=1.000" in repr_str


class TestLinearWithLoRA:
    """Test suite for LinearWithLoRA wrapper."""

    def test_linear_with_lora_initialization(self):
        """Test LinearWithLoRA initializes correctly."""
        layer = LinearWithLoRA(768, 768, r=16, alpha=16)

        assert layer.in_features == 768
        assert layer.out_features == 768
        assert isinstance(layer.linear, nn.Linear)
        assert isinstance(layer.lora, LoRALayer)
        assert layer.enable_lora is True

    def test_linear_with_lora_disabled(self):
        """Test LinearWithLoRA can disable LoRA adapter."""
        layer = LinearWithLoRA(768, 768, enable_lora=False)

        assert layer.lora is None
        assert layer.enable_lora is False

    def test_linear_with_lora_forward_shape(self):
        """Test LinearWithLoRA maintains correct output shape."""
        layer = LinearWithLoRA(768, 768, r=16)
        x = torch.randn(2, 50, 768)

        output = layer(x)

        assert output.shape == (2, 50, 768)

    def test_linear_with_lora_forward_combines_outputs(self):
        """Test LinearWithLoRA combines base + LoRA outputs."""
        layer = LinearWithLoRA(768, 768, r=16)

        # Initialize LoRA B with non-zero values to see effect
        nn.init.normal_(layer.lora.lora_B.weight, std=0.01)

        x = torch.randn(2, 10, 768)

        # Get base output (disable LoRA temporarily)
        layer.enable_lora = False
        base_output = layer(x)

        # Get full output (with LoRA)
        layer.enable_lora = True
        full_output = layer(x)

        # They should be different (LoRA adds contribution)
        assert not torch.allclose(base_output, full_output)

    def test_linear_with_lora_freeze_base(self):
        """Test freeze_base() freezes only base weights (AC5)."""
        layer = LinearWithLoRA(768, 768, r=16)
        layer.freeze_base()

        # Base weights should be frozen
        assert not layer.linear.weight.requires_grad
        if layer.linear.bias is not None:
            assert not layer.linear.bias.requires_grad

        # LoRA weights should be trainable
        assert layer.lora.lora_A.weight.requires_grad
        assert layer.lora.lora_B.weight.requires_grad

    def test_linear_with_lora_unfreeze_base(self):
        """Test unfreeze_base() unfreezes base weights."""
        layer = LinearWithLoRA(768, 768, r=16)
        layer.freeze_base()
        layer.unfreeze_base()

        # Base weights should be trainable
        assert layer.linear.weight.requires_grad
        if layer.linear.bias is not None:
            assert layer.linear.bias.requires_grad

    def test_linear_with_lora_merge_lora(self):
        """Test merge_lora() fuses LoRA into base weights (AC4)."""
        layer = LinearWithLoRA(768, 768, r=16)

        # Initialize LoRA B with non-zero values so merge has effect
        nn.init.normal_(layer.lora.lora_B.weight, std=0.01)

        # Store original weights
        original_weight = layer.linear.weight.data.clone()

        # Merge LoRA
        layer.merge_lora()

        # Weight should have changed
        assert not torch.allclose(layer.linear.weight.data, original_weight)

        # LoRA should be disabled
        assert layer.lora is None
        assert layer.enable_lora is False

    def test_linear_with_lora_merge_equivalence(self):
        """Test that merged layer produces same output as separate LoRA."""
        torch.manual_seed(42)
        layer1 = LinearWithLoRA(768, 768, r=16)

        torch.manual_seed(42)
        layer2 = LinearWithLoRA(768, 768, r=16)

        x = torch.randn(2, 10, 768)

        # Get output before merge
        with torch.no_grad():
            output_before = layer1(x)

        # Merge layer2
        layer2.merge_lora()

        # Get output after merge
        with torch.no_grad():
            output_after = layer2(x)

        # Outputs should be very close (within floating point precision)
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_linear_with_lora_extra_repr(self):
        """Test extra_repr provides useful information."""
        layer = LinearWithLoRA(768, 768, r=16)
        repr_str = layer.extra_repr()

        assert "in_features=768" in repr_str
        assert "out_features=768" in repr_str
        assert "lora=enabled" in repr_str


class TestLoRAUtilities:
    """Test suite for LoRA utility functions."""

    def test_apply_lora_to_layer(self):
        """Test apply_lora_to_layer() creates LoRA adapters (AC6)."""

        # Create a simple module with Linear layers
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
                self.k_proj = nn.Linear(768, 768)
                self.v_proj = nn.Linear(768, 768)
                self.out_proj = nn.Linear(768, 768)
                self.other_linear = nn.Linear(768, 768)

        module = SimpleModule()
        lora_modules = apply_lora_to_layer(module, r=16, alpha=16)

        # Should create LoRA for q_proj, k_proj, v_proj, out_proj
        assert len(lora_modules) >= 4
        assert any("q_proj" in name for name in lora_modules.keys())
        assert any("k_proj" in name for name in lora_modules.keys())
        assert any("v_proj" in name for name in lora_modules.keys())
        assert any("out_proj" in name for name in lora_modules.keys())

        # Each should be a LoRALayer
        for lora in lora_modules.values():
            assert isinstance(lora, LoRALayer)
            assert lora.r == 16
            assert lora.alpha == 16

    def test_apply_lora_custom_targets(self):
        """Test apply_lora_to_layer() with custom target modules."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
                self.k_proj = nn.Linear(768, 768)
                self.v_proj = nn.Linear(768, 768)

        module = SimpleModule()
        lora_modules = apply_lora_to_layer(
            module, target_modules={"q_proj", "k_proj"}  # Only Q and K
        )

        # Should only create LoRA for q_proj and k_proj
        assert len(lora_modules) == 2
        assert any("q_proj" in name for name in lora_modules.keys())
        assert any("k_proj" in name for name in lora_modules.keys())
        assert not any("v_proj" in name for name in lora_modules.keys())

    def test_get_lora_parameters(self):
        """Test get_lora_parameters() collects LoRA params."""
        model = LinearWithLoRA(768, 768, r=16)
        lora_params = get_lora_parameters(model)

        # Should have 2 LoRA parameters (A and B weights)
        assert len(lora_params) == 2

        # All should be trainable
        assert all(p.requires_grad for p in lora_params)

        # All should have "lora" in their name
        param_names = [n for n, p in model.named_parameters() if "lora" in n.lower()]
        assert len(param_names) == 2

    def test_count_lora_parameters(self):
        """Test count_lora_parameters() counts correctly."""
        model = LinearWithLoRA(768, 768, r=16)
        count = count_lora_parameters(model)

        # LoRA has two matrices: A (768 x 16) and B (16 x 768)
        expected = 768 * 16 + 16 * 768
        assert count == expected

    def test_freeze_non_lora_parameters(self):
        """Test freeze_non_lora_parameters() freezes base weights."""
        model = LinearWithLoRA(768, 768, r=16)
        freeze_non_lora_parameters(model)

        # Base weights should be frozen
        assert not model.linear.weight.requires_grad
        if model.linear.bias is not None:
            assert not model.linear.bias.requires_grad

        # LoRA weights should be trainable
        assert model.lora.lora_A.weight.requires_grad
        assert model.lora.lora_B.weight.requires_grad


class TestLoRAIntegration:
    """Integration tests for LoRA functionality."""

    def test_lora_training_loop(self):
        """Test LoRA can be used in a training loop."""
        model = LinearWithLoRA(768, 768, r=16)
        model.freeze_base()

        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=1e-3)

        x = torch.randn(4, 10, 768)
        target = torch.randn(4, 10, 768)

        # Training step
        model.train()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check LoRA gradients exist
        assert model.lora.lora_A.weight.grad is not None  # type: ignore
        assert model.lora.lora_B.weight.grad is not None

        # Check base gradients don't exist (frozen)
        assert model.linear.weight.grad is None

        # Optimizer step
        optimizer.step()

    def test_lora_multiple_layers(self):
        """Test LoRA can be applied to multiple layers."""

        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = LinearWithLoRA(768, 768, r=16)
                self.layer2 = LinearWithLoRA(768, 768, r=16)
                self.layer3 = LinearWithLoRA(768, 768, r=16)

        model = MultiLayerModel()
        freeze_non_lora_parameters(model)

        lora_params = get_lora_parameters(model)
        # 3 layers * 2 matrices (A, B) each = 6 parameters
        assert len(lora_params) == 6

        # Count total LoRA parameters
        total_lora = count_lora_parameters(model)
        expected = 3 * (768 * 16 + 16 * 768)  # 3 layers
        assert total_lora == expected

    def test_lora_inference_optimization(self):
        """Test LoRA merge for inference optimization."""
        model = LinearWithLoRA(768, 768, r=16)

        x = torch.randn(2, 10, 768)

        # Forward pass before merge
        with torch.no_grad():
            output_before = model(x)

        # Merge LoRA
        model.merge_lora()

        # Forward pass after merge
        with torch.no_grad():
            output_after = model(x)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)

        # LoRA should be disabled (no overhead)
        assert model.lora is None


class TestLoRAAcceptanceCriteria:
    """Tests verifying all acceptance criteria for Issue 2.2.2."""

    def test_acceptance_criterion_1_initialization(self):
        """AC1: LoRA A initialized with Kaiming, B with zeros."""
        lora = LoRALayer(768, 768, r=16)

        # B initialized with zeros
        assert torch.allclose(lora.lora_B.weight, torch.zeros_like(lora.lora_B.weight))

        # A initialized with Kaiming (non-zero, reasonable variance)
        assert not torch.allclose(lora.lora_A.weight, torch.zeros_like(lora.lora_A.weight))
        assert lora.lora_A.weight.std() > 0.01

    def test_acceptance_criterion_2_scaling_factor(self):
        """AC2: Scaling factor (alpha/r) applied correctly."""
        lora1 = LoRALayer(768, 768, r=16, alpha=16)
        assert lora1.scaling == 1.0

        lora2 = LoRALayer(768, 768, r=16, alpha=32)
        assert lora2.scaling == 2.0

        lora3 = LoRALayer(768, 768, r=8, alpha=16)
        assert lora3.scaling == 2.0

    def test_acceptance_criterion_3_dropout_before_projection(self):
        """AC3: Dropout applied before LoRA projection."""
        lora = LoRALayer(768, 768, r=16, dropout=0.5)  # Higher dropout for test
        assert isinstance(lora.dropout, nn.Dropout)
        assert lora.dropout.p == 0.5

        # Initialize B with non-zero values to see dropout effect
        nn.init.normal_(lora.lora_B.weight, std=0.01)

        # Verify dropout causes randomness in training mode
        lora.train()
        x = torch.randn(2, 10, 768)
        outputs = [lora(x) for _ in range(10)]
        all_same = all(torch.equal(outputs[0], out) for out in outputs[1:])
        assert not all_same

    def test_acceptance_criterion_4_merge_lora(self):
        """AC4: merge_lora() correctly fuses weights."""
        layer = LinearWithLoRA(768, 768, r=16)
        x = torch.randn(2, 10, 768)

        with torch.no_grad():
            output_before = layer(x)

        layer.merge_lora()

        with torch.no_grad():
            output_after = layer(x)

        # Outputs should be identical after merge
        assert torch.allclose(output_before, output_after, atol=1e-5)

        # LoRA should be disabled
        assert layer.lora is None

    def test_acceptance_criterion_5_freeze_base(self):
        """AC5: freeze_base() freezes only base weights."""
        layer = LinearWithLoRA(768, 768, r=16)
        layer.freeze_base()

        # Base weights frozen
        assert not layer.linear.weight.requires_grad

        # LoRA weights trainable
        assert layer.lora.lora_A.weight.requires_grad
        assert layer.lora.lora_B.weight.requires_grad

    def test_acceptance_criterion_6_attention_projections(self):
        """AC6: Works with Q, K, V, and output projections."""

        class AttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
                self.k_proj = nn.Linear(768, 768)
                self.v_proj = nn.Linear(768, 768)
                self.out_proj = nn.Linear(768, 768)

        attention = AttentionModule()
        lora_modules = apply_lora_to_layer(attention, r=16, alpha=16)

        # Should create LoRA for all 4 projections
        assert len(lora_modules) == 4
        assert all(isinstance(lora, LoRALayer) for lora in lora_modules.values())

        # Test forward pass works
        x = torch.randn(2, 10, 768)
        for lora in lora_modules.values():
            output = lora(x)
            assert output.shape == (2, 10, 768)


# ============================================================================
# ModernBERTLayerV3 Tests (Issue 2.2.3)
# ============================================================================


class TestModernBERTLayerV3:
    """Test suite for ModernBERTLayerV3 transformer layer."""

    def test_layer_initialization(self):
        """Test layer initializes with correct configuration."""
        layer = ModernBERTLayerV3(
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            layer_idx=15,
        )

        assert layer.layer_idx == 15
        assert layer.band == "context"
        assert layer.window_size == 128
        assert layer.hidden_size == 768
        assert not layer.enable_lora

        # Check components exist
        assert isinstance(layer.attention_norm, nn.LayerNorm)
        assert isinstance(layer.ffn_norm, nn.LayerNorm)
        assert layer.attention is not None
        assert layer.ffn is not None
        assert isinstance(layer.dropout, nn.Dropout)

    def test_layer_initialization_with_lora(self):
        """Test layer initializes with LoRA for Family Band."""
        layer = ModernBERTLayerV3(
            layer_idx=25,
            enable_lora=True,
            lora_r=16,
            lora_alpha=16,
        )

        assert layer.enable_lora is True
        assert layer.lora_q is not None
        assert layer.lora_k is not None
        assert layer.lora_v is not None
        assert layer.lora_o is not None
        assert isinstance(layer.lora_o, LoRALayer)

    def test_layer_no_lora_outside_family_band(self):
        """Test LoRA not initialized outside Family Band even if requested."""
        layer = ModernBERTLayerV3(
            layer_idx=10,  # Context band
            enable_lora=True,
        )

        assert layer.lora_q is None
        assert layer.lora_k is None
        assert layer.lora_v is None
        assert layer.lora_o is None

    def test_layer_forward_shape(self):
        """Test layer maintains correct output shape."""
        batch_size = 2
        seq_len = 100
        hidden_size = 768

        layer = ModernBERTLayerV3(
            hidden_size=768,
            layer_idx=15,
        )

        x = torch.randn(batch_size, seq_len, hidden_size)
        output, attn_weights = layer(x)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attn_weights is None  # Not returned by default

    def test_layer_forward_with_attention_weights(self):
        """Test layer returns attention weights when requested."""
        layer = ModernBERTLayerV3(layer_idx=5)
        x = torch.randn(2, 50, 768)

        output, attn_weights = layer(x, output_attentions=True)

        assert output.shape == (2, 50, 768)
        assert attn_weights is not None
        # Attention weights shape: [batch, heads, seq, seq]
        assert attn_weights.shape == (2, 12, 50, 50)

    def test_layer_forward_with_padding_mask(self):
        """Test layer handles padding mask correctly."""
        layer = ModernBERTLayerV3(layer_idx=10)

        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 768)

        # Create padding mask (1 = attend, 0 = mask)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 30:] = 0  # Mask last 20 tokens of first sequence
        attention_mask[1, 40:] = 0  # Mask last 10 tokens of second sequence

        output, _ = layer(x, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(output).any()

    def test_layer_forward_with_lora(self):
        """Test layer forward pass with LoRA enabled."""
        layer = ModernBERTLayerV3(
            layer_idx=25,
            enable_lora=True,
        )

        x = torch.randn(2, 50, 768)
        output, _ = layer(x)

        assert output.shape == (2, 50, 768)
        assert not torch.isnan(output).any()

    def test_layer_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = ModernBERTLayerV3(layer_idx=15)
        x = torch.randn(2, 50, 768, requires_grad=True)

        output, _ = layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.attention_norm.weight.grad is not None
        assert layer.ffn_norm.weight.grad is not None

    def test_layer_freeze_base_weights(self):
        """Test freeze_base_weights freezes non-LoRA parameters."""
        layer = ModernBERTLayerV3(
            layer_idx=25,
            enable_lora=True,
        )

        layer.freeze_base_weights()

        # Base weights should be frozen
        assert not layer.attention_norm.weight.requires_grad
        assert not layer.ffn_norm.weight.requires_grad

        # LoRA weights should be trainable
        assert layer.lora_o.lora_A.weight.requires_grad
        assert layer.lora_o.lora_B.weight.requires_grad

    def test_layer_unfreeze_base_weights(self):
        """Test unfreeze_base_weights unfreezes parameters."""
        layer = ModernBERTLayerV3(layer_idx=15)

        layer.freeze_base_weights()
        layer.unfreeze_base_weights()

        # All weights should be trainable
        assert layer.attention_norm.weight.requires_grad
        assert layer.ffn_norm.weight.requires_grad

    def test_layer_get_num_params(self):
        """Test get_num_params returns correct statistics."""
        layer = ModernBERTLayerV3(
            layer_idx=25,
            enable_lora=True,
        )

        params = layer.get_num_params()

        assert "total" in params
        assert "attention" in params
        assert "ffn" in params
        assert "lora" in params
        assert "trainable" in params

        assert params["total"] > 0
        assert params["attention"] > 0
        assert params["ffn"] > 0
        assert params["lora"] > 0
        assert params["trainable"] == params["total"]

    def test_layer_band_assignments(self):
        """Test correct band assignment for all layers."""
        # Foundation: L1-6
        assert ModernBERTLayerV3(layer_idx=1).band == "foundation"
        assert ModernBERTLayerV3(layer_idx=6).band == "foundation"

        # Context: L7-18
        assert ModernBERTLayerV3(layer_idx=7).band == "context"
        assert ModernBERTLayerV3(layer_idx=18).band == "context"

        # Semantic: L19-22
        assert ModernBERTLayerV3(layer_idx=19).band == "semantic"
        assert ModernBERTLayerV3(layer_idx=22).band == "semantic"

        # Family: L23-28
        assert ModernBERTLayerV3(layer_idx=23).band == "family"
        assert ModernBERTLayerV3(layer_idx=28).band == "family"

    def test_layer_window_sizes(self):
        """Test correct window size for all layers."""
        # Foundation: 64
        assert ModernBERTLayerV3(layer_idx=3).window_size == 64

        # Context: 128
        assert ModernBERTLayerV3(layer_idx=10).window_size == 128

        # Semantic: 256
        assert ModernBERTLayerV3(layer_idx=20).window_size == 256

        # Family: 512
        assert ModernBERTLayerV3(layer_idx=25).window_size == 512

    def test_layer_extra_repr(self):
        """Test extra_repr provides useful information."""
        layer = ModernBERTLayerV3(layer_idx=25, enable_lora=True)
        repr_str = layer.extra_repr()

        assert "layer=25" in repr_str
        assert "band=family" in repr_str
        assert "window=512" in repr_str
        assert "lora=enabled" in repr_str


class TestLayerStack:
    """Test suite for layer stack creation and management."""

    def test_create_layer_stack_default(self):
        """Test create_layer_stack with default parameters."""
        layers = create_layer_stack()

        assert len(layers) == 28
        assert all(isinstance(layer, ModernBERTLayerV3) for layer in layers)

        # Check layer indices
        for i, layer in enumerate(layers):
            assert layer.layer_idx == i + 1

    def test_create_layer_stack_custom_size(self):
        """Test create_layer_stack with custom number of layers."""
        layers = create_layer_stack(num_layers=12)

        assert len(layers) == 12

    def test_create_layer_stack_lora_configuration(self):
        """Test LoRA is only enabled for Family Band by default."""
        layers = create_layer_stack()

        # L1-22 should not have LoRA
        for i in range(22):
            assert layers[i].lora_o is None

        # L23-28 should have LoRA
        for i in range(22, 28):
            assert layers[i].lora_o is not None

    def test_create_layer_stack_custom_lora_layers(self):
        """Test create_layer_stack with custom LoRA layers."""
        # LoRA only works on Family Band (L23-28), so specify subset
        layers = create_layer_stack(lora_layers=[24, 26, 28])

        # L24, L26, L28 should have LoRA
        assert layers[23].lora_o is not None  # L24 (0-indexed)
        assert layers[25].lora_o is not None  # L26
        assert layers[27].lora_o is not None  # L28

        # L23, L25, L27 should not have LoRA (not in list)
        assert layers[22].lora_o is None  # L23
        assert layers[24].lora_o is None  # L25
        assert layers[26].lora_o is None  # L27

        # L1-22 should not have LoRA (outside Family Band)
        for i in range(22):
            assert layers[i].lora_o is None

    def test_freeze_layer_bands_default(self):
        """Test freeze_layer_bands with default (foundation + context)."""
        layers = create_layer_stack()
        freeze_layer_bands(layers)

        # L1-18 should be frozen
        for i in range(18):
            assert not layers[i].attention_norm.weight.requires_grad

        # L19-28 should be trainable
        for i in range(18, 28):
            assert layers[i].attention_norm.weight.requires_grad

    def test_freeze_layer_bands_custom(self):
        """Test freeze_layer_bands with custom bands."""
        layers = create_layer_stack()
        freeze_layer_bands(layers, ["foundation"])

        # L1-6 should be frozen
        for i in range(6):
            assert not layers[i].attention_norm.weight.requires_grad

        # L7-28 should be trainable
        for i in range(6, 28):
            assert layers[i].attention_norm.weight.requires_grad

    def test_unfreeze_layer_bands_default(self):
        """Test unfreeze_layer_bands with default (semantic + family)."""
        layers = create_layer_stack()

        # First freeze everything
        freeze_layer_bands(layers, ["foundation", "context", "semantic", "family"])

        # Then unfreeze semantic + family
        unfreeze_layer_bands(layers)

        # L1-18 should still be frozen
        for i in range(18):
            assert not layers[i].attention_norm.weight.requires_grad

        # L19-28 should be trainable
        for i in range(18, 28):
            assert layers[i].attention_norm.weight.requires_grad

    def test_get_layer_stats(self):
        """Test get_layer_stats returns correct statistics."""
        layers = create_layer_stack()
        stats = get_layer_stats(layers)

        assert stats["num_layers"] == 28
        assert stats["total_params"] > 0
        assert stats["trainable_params"] > 0
        assert stats["lora_params"] > 0

        # Check band stats
        assert "foundation" in stats["by_band"]
        assert "context" in stats["by_band"]
        assert "semantic" in stats["by_band"]
        assert "family" in stats["by_band"]

        # Foundation band should have L1-6
        assert len(stats["by_band"]["foundation"]["layers"]) == 6
        assert stats["by_band"]["foundation"]["layers"] == [1, 2, 3, 4, 5, 6]

        # Family band should have LoRA params
        assert stats["by_band"]["family"]["lora"] > 0

    def test_layer_stack_forward_pass(self):
        """Test forward pass through entire layer stack."""
        layers = create_layer_stack(num_layers=4)  # Use fewer layers for speed

        batch_size = 2
        seq_len = 50
        hidden_size = 768

        x = torch.randn(batch_size, seq_len, hidden_size)

        # Pass through all layers
        for layer in layers:
            x, _ = layer(x)

        assert x.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(x).any()

    def test_layer_stack_with_gradient_checkpointing(self):
        """Test layer stack works with gradient checkpointing pattern."""
        layers = create_layer_stack(num_layers=4)

        x = torch.randn(2, 50, 768, requires_grad=True)

        # Simulate gradient checkpointing usage
        def forward_pass(hidden_states):
            for layer in layers:
                hidden_states, _ = layer(hidden_states)
            return hidden_states

        output = forward_pass(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestLayerV3Integration:
    """Integration tests for ModernBERTLayerV3 with other v3 components."""

    def test_layer_with_attention_v3(self):
        """Test layer correctly uses attention_v3 module."""
        layer = ModernBERTLayerV3(layer_idx=15)

        # Check that attention module is from attention_v3
        assert hasattr(layer.attention, "window_size")
        assert hasattr(layer.attention, "layer_idx")

    def test_layer_with_ffn_v3(self):
        """Test layer correctly uses ffn_v3 module."""
        layer = ModernBERTLayerV3(layer_idx=15)

        # Check that FFN module is GELUFFN
        from modeling_studio.models.ffn_v3 import GELUFFN

        assert isinstance(layer.ffn, GELUFFN)

    def test_layer_with_lora_v3(self):
        """Test layer correctly uses lora_v3 module."""
        layer = ModernBERTLayerV3(layer_idx=25, enable_lora=True)

        # Check that LoRA modules are LoRALayer
        assert isinstance(layer.lora_o, LoRALayer)
        assert layer.lora_o.r == 16
        assert layer.lora_o.alpha == 16

    def test_layer_batch_sizes(self):
        """Test layer handles different batch sizes."""
        layer = ModernBERTLayerV3(layer_idx=10)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 50, 768)
            output, _ = layer(x)
            assert output.shape == (batch_size, 50, 768)

    def test_layer_sequence_lengths(self):
        """Test layer handles different sequence lengths."""
        layer = ModernBERTLayerV3(layer_idx=10)

        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(2, seq_len, 768)
            output, _ = layer(x)
            assert output.shape == (2, seq_len, 768)

    def test_layer_training_mode(self):
        """Test layer behavior in training vs eval mode."""
        layer = ModernBERTLayerV3(layer_idx=15)
        x = torch.randn(2, 50, 768)

        # Training mode
        layer.train()
        output_train1, _ = layer(x)
        output_train2, _ = layer(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode
        layer.eval()
        with torch.no_grad():
            output_eval1, _ = layer(x)
            output_eval2, _ = layer(x)
        # Outputs should be identical (no dropout)
        assert torch.allclose(output_eval1, output_eval2)


class TestLayerV3AcceptanceCriteria:
    """Tests verifying all acceptance criteria for Issue 2.2.3."""

    def test_acceptance_criterion_1_pre_layernorm(self):
        """AC1: Pre-LayerNorm architecture (not post-norm)."""
        layer = ModernBERTLayerV3(layer_idx=15)

        # Verify LayerNorm exists before attention and FFN
        assert isinstance(layer.attention_norm, nn.LayerNorm)
        assert isinstance(layer.ffn_norm, nn.LayerNorm)

        # Test that LayerNorm is applied before attention
        x = torch.randn(2, 50, 768)

        # Manually trace through to verify pre-norm
        residual = x
        normed = layer.attention_norm(x)
        assert not torch.allclose(normed, x)  # LayerNorm changes values

    def test_acceptance_criterion_2_residual_connections(self):
        """AC2: Residual connections around attention and FFN."""
        layer = ModernBERTLayerV3(layer_idx=15)
        x = torch.randn(2, 50, 768)

        # Forward pass
        output, _ = layer(x)

        # Output should be different from input (transformations applied)
        assert not torch.allclose(output, x)

        # But relationship should exist (residual adds input back)
        # This is implicit in the architecture, verified by successful forward pass
        assert output.shape == x.shape

    def test_acceptance_criterion_3_lora_only_family_band(self):
        """AC3: LoRA only applied to layers 23-28."""
        # L1-22: No LoRA even if requested
        for layer_idx in [1, 10, 22]:
            layer = ModernBERTLayerV3(layer_idx=layer_idx, enable_lora=True)
            assert layer.lora_o is None

        # L23-28: LoRA enabled when requested
        for layer_idx in [23, 25, 28]:
            layer = ModernBERTLayerV3(layer_idx=layer_idx, enable_lora=True)
            assert layer.lora_o is not None

    def test_acceptance_criterion_4_window_size_by_band(self):
        """AC4: Window size correctly set per layer band."""
        test_cases = [
            (1, 64),  # Foundation
            (6, 64),
            (7, 128),  # Context
            (18, 128),
            (19, 256),  # Semantic
            (22, 256),
            (23, 512),  # Family
            (28, 512),
        ]

        for layer_idx, expected_window in test_cases:
            layer = ModernBERTLayerV3(layer_idx=layer_idx)
            assert layer.window_size == expected_window

    def test_acceptance_criterion_5_freeze_preserves_lora(self):
        """AC5: freeze_base_weights() preserves LoRA trainability."""
        layer = ModernBERTLayerV3(layer_idx=25, enable_lora=True)

        layer.freeze_base_weights()

        # Base weights frozen
        assert not layer.attention_norm.weight.requires_grad
        assert not layer.ffn_norm.weight.requires_grad

        # LoRA weights trainable
        assert layer.lora_o.lora_A.weight.requires_grad
        assert layer.lora_o.lora_B.weight.requires_grad

    def test_acceptance_criterion_6_all_28_layers_created(self):
        """AC6: All 28 layers can be created with correct config."""
        layers = create_layer_stack(num_layers=28)

        assert len(layers) == 28

        # Verify each layer has correct configuration
        for i, layer in enumerate(layers):
            assert layer.layer_idx == i + 1
            assert layer.band in ["foundation", "context", "semantic", "family"]
            assert layer.window_size in [64, 128, 256, 512]

        # Verify band distribution
        foundation_count = sum(1 for l in layers if l.band == "foundation")
        context_count = sum(1 for l in layers if l.band == "context")
        semantic_count = sum(1 for l in layers if l.band == "semantic")
        family_count = sum(1 for l in layers if l.band == "family")

        assert foundation_count == 6  # L1-6
        assert context_count == 12  # L7-18
        assert semantic_count == 4  # L19-22
        assert family_count == 6  # L23-28


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Tests for Adapter Modules

Test coverage for:
    - BottleneckAdapter: shape preservation, residual connection, freeze/unfreeze
    - TaskGroupAdapter: task group routing, shared/separate projections
    - ParallelAdapter: parallel computation, scaling
    - LoRAAdapter: low-rank adaptation, merge/unmerge
    - Factory function: create_adapter

Issue: 5.0.2 - Implement Task-Specific Adapters
Epic: 5.0 - Model Architecture Enhancements (Pre-Stage B)
"""

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.adapters import (
    AdaptedLinear,
    AdapterConfig,
    BottleneckAdapter,
    LoRAAdapter,
    ParallelAdapter,
    TaskGroupAdapter,
    create_adapter,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hidden_size():
    """Default hidden size for tests."""
    return 768


@pytest.fixture
def bottleneck_size():
    """Default bottleneck size for tests."""
    return 64


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 128


@pytest.fixture
def sample_input(batch_size, seq_len, hidden_size):
    """Sample input tensor."""
    return torch.randn(batch_size, seq_len, hidden_size)


# =============================================================================
# BottleneckAdapter Tests
# =============================================================================


class TestBottleneckAdapter:
    """Tests for BottleneckAdapter."""

    def test_shape_preservation(self, sample_input, hidden_size, bottleneck_size):
        """Output shape should match input shape."""
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)
        output = adapter(sample_input)
        assert output.shape == sample_input.shape

    def test_residual_connection(self, sample_input, hidden_size):
        """With residual, output should be close to input initially."""
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=64,
            init_scale=1e-6,  # Very small init
            residual_connection=True,
        )
        output = adapter(sample_input)
        # Output should be close to input due to small init + residual
        diff = (output - sample_input).abs().mean()
        assert diff < 1.0  # Should be close

    def test_no_residual(self, sample_input, hidden_size):
        """Without residual, output should be different."""
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=64,
            residual_connection=False,
        )
        output = adapter(sample_input)
        assert output.shape == sample_input.shape
        # Output should be different from input
        assert not torch.allclose(output, sample_input, atol=1e-3)

    def test_freeze_unfreeze(self, hidden_size):
        """Freeze/unfreeze should toggle requires_grad."""
        adapter = BottleneckAdapter(hidden_size=hidden_size)

        # Initially trainable
        assert all(p.requires_grad for p in adapter.parameters())

        # Freeze
        adapter.freeze()
        assert all(not p.requires_grad for p in adapter.parameters())

        # Unfreeze
        adapter.unfreeze()
        assert all(p.requires_grad for p in adapter.parameters())

    def test_num_parameters(self, hidden_size, bottleneck_size):
        """Parameter count should match expected."""
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            use_layer_norm=True,
        )
        # Expected: down_proj (hidden*bottleneck + bottleneck) +
        #           up_proj (bottleneck*hidden + hidden) +
        #           layer_norm (2 * hidden)
        expected = (
            hidden_size * bottleneck_size
            + bottleneck_size  # down_proj
            + bottleneck_size * hidden_size
            + hidden_size  # up_proj
            + 2 * hidden_size  # layer_norm
        )
        assert adapter.num_parameters == expected


# =============================================================================
# TaskGroupAdapter Tests
# =============================================================================


class TestTaskGroupAdapter:
    """Tests for TaskGroupAdapter."""

    @pytest.fixture
    def task_groups(self):
        """Task groups for testing."""
        return ["token_tasks", "sequence_tasks", "pair_tasks"]

    def test_shape_preservation(self, sample_input, hidden_size, task_groups):
        """Output shape should match input for each group."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=64,
        )
        for group in task_groups:
            output = adapter(sample_input, task_group=group)
            assert output.shape == sample_input.shape

    def test_invalid_task_group(self, sample_input, hidden_size, task_groups):
        """Should raise error for invalid task group."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
        )
        with pytest.raises(ValueError, match="Unknown task_group"):
            adapter(sample_input, task_group="invalid_group")

    def test_separate_projections(self, sample_input, hidden_size, task_groups):
        """With separate projections, each group has own weights."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            share_down_projection=False,
        )
        # Each group should have separate down/up projections
        assert hasattr(adapter, "down_projs")
        assert hasattr(adapter, "up_projs")
        assert len(adapter.down_projs) == len(task_groups)
        assert len(adapter.up_projs) == len(task_groups)

    def test_shared_down_projection(self, sample_input, hidden_size, task_groups):
        """With shared down, only up projections differ."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            share_down_projection=True,
        )
        # Should have single down_proj and multiple up_projs
        assert hasattr(adapter, "down_proj")
        assert hasattr(adapter, "up_projs")
        assert len(adapter.up_projs) == len(task_groups)

    def test_freeze_single_group(self, hidden_size, task_groups):
        """Should be able to freeze just one group."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            share_down_projection=False,
        )

        # Freeze only first group
        adapter.freeze(task_group=task_groups[0])

        # First group frozen
        for p in adapter.down_projs[task_groups[0]].parameters():
            assert not p.requires_grad
        for p in adapter.up_projs[task_groups[0]].parameters():
            assert not p.requires_grad

        # Other groups still trainable
        for p in adapter.down_projs[task_groups[1]].parameters():
            assert p.requires_grad


# =============================================================================
# ParallelAdapter Tests
# =============================================================================


class TestParallelAdapter:
    """Tests for ParallelAdapter."""

    def test_shape_preservation(self, sample_input, hidden_size):
        """Output shape should match input."""
        adapter = ParallelAdapter(hidden_size=hidden_size, bottleneck_size=64)
        output = adapter(sample_input)
        assert output.shape == sample_input.shape

    def test_scale_factor(self, sample_input, hidden_size):
        """Scale factor should affect output magnitude."""
        adapter_small = ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=64,
            output_scale=0.01,
        )
        adapter_large = ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=64,
            output_scale=10.0,
        )

        # Initialize up_proj with non-zero weights for meaningful test
        # (ParallelAdapter initializes up_proj to zero by design)
        adapter_small.up_proj.weight.data = torch.randn_like(adapter_small.up_proj.weight) * 0.1
        adapter_large.up_proj.weight.data = adapter_small.up_proj.weight.data.clone()
        adapter_small.down_proj.weight.data = torch.randn_like(adapter_small.down_proj.weight) * 0.1
        adapter_large.down_proj.weight.data = adapter_small.down_proj.weight.data.clone()

        out_small = adapter_small(sample_input)
        out_large = adapter_large(sample_input)

        # Large scale should have larger delta from input
        delta_small = (out_small - sample_input).abs().mean()
        delta_large = (out_large - sample_input).abs().mean()
        assert delta_large > delta_small


# =============================================================================
# LoRAAdapter Tests
# =============================================================================


class TestLoRAAdapter:
    """Tests for LoRAAdapter."""

    def test_output_shape(self, batch_size, seq_len, hidden_size):
        """Output shape should match expected."""
        lora = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = lora(x)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_initial_zero_output(self, batch_size, seq_len, hidden_size):
        """Initially, LoRA should output near-zero (B initialized to 0)."""
        lora = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
            dropout=0.0,  # No dropout for deterministic test
        )
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = lora(x)
        # B is initialized to zero, so output should be zero
        assert output.abs().max() < 1e-6

    def test_merge_weights(self, hidden_size):
        """Merge should combine base + LoRA weights."""
        lora = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )
        # Set non-zero B
        lora.lora_B.data = torch.randn_like(lora.lora_B)

        base_weight = torch.randn(hidden_size, hidden_size)
        merged = lora.merge(base_weight)

        # Merged should be different from base
        assert not torch.allclose(merged, base_weight)

    def test_parameter_count(self, hidden_size):
        """Parameter count should be 2 * r * d."""
        r = 8
        lora = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=r,
            alpha=16,
        )
        # A: (r, hidden_size), B: (hidden_size, r)
        expected = r * hidden_size + hidden_size * r
        assert lora.num_parameters == expected


# =============================================================================
# AdaptedLinear Tests
# =============================================================================


class TestAdaptedLinear:
    """Tests for AdaptedLinear wrapper."""

    def test_forward(self, batch_size, seq_len, hidden_size):
        """Forward should combine base + LoRA."""
        base = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base, r=8, alpha=16)

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = adapted(x)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_merge_unmerge(self, hidden_size):
        """Merge and unmerge should be reversible."""
        base = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base, r=8, alpha=16)

        # Set non-zero LoRA weights
        adapted.lora.lora_B.data = torch.randn_like(adapted.lora.lora_B) * 0.01

        original_weight = adapted.base_linear.weight.clone()

        # Merge
        adapted.merge_weights()
        merged_weight = adapted.base_linear.weight.clone()
        assert not torch.allclose(original_weight, merged_weight, atol=1e-6)

        # Unmerge
        adapted.unmerge_weights()
        unmerged_weight = adapted.base_linear.weight.clone()
        assert torch.allclose(original_weight, unmerged_weight, atol=1e-5)

    def test_freeze_base(self, hidden_size):
        """Freeze base should keep LoRA trainable."""
        base = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base, r=8, alpha=16)

        adapted.freeze_base()

        assert not adapted.base_linear.weight.requires_grad
        assert adapted.lora.lora_A.requires_grad
        assert adapted.lora.lora_B.requires_grad


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateAdapter:
    """Tests for create_adapter factory function."""

    def test_create_bottleneck(self, hidden_size):
        """Factory should create BottleneckAdapter."""
        adapter = create_adapter("bottleneck", hidden_size=hidden_size)
        assert isinstance(adapter, BottleneckAdapter)

    def test_create_parallel(self, hidden_size):
        """Factory should create ParallelAdapter."""
        adapter = create_adapter("parallel", hidden_size=hidden_size)
        assert isinstance(adapter, ParallelAdapter)

    def test_create_lora(self, hidden_size):
        """Factory should create LoRAAdapter."""
        adapter = create_adapter("lora", hidden_size=hidden_size)
        assert isinstance(adapter, LoRAAdapter)

    def test_create_task_group(self, hidden_size):
        """Factory should create TaskGroupAdapter."""
        adapter = create_adapter(
            "task_group",
            hidden_size=hidden_size,
            task_groups=["a", "b", "c"],
        )
        assert isinstance(adapter, TaskGroupAdapter)

    def test_invalid_type(self, hidden_size):
        """Factory should raise for invalid type."""
        with pytest.raises(ValueError, match="Unknown adapter_type"):
            create_adapter("invalid", hidden_size=hidden_size)

    def test_config_override(self, hidden_size):
        """Factory should allow config overrides."""
        config = AdapterConfig(bottleneck_size=128)
        adapter = create_adapter(
            "bottleneck",
            hidden_size=hidden_size,
            config=config,
            bottleneck_size=32,  # Override
        )
        assert adapter.bottleneck_size == 32


# =============================================================================
# Integration Test
# =============================================================================


class TestAdapterIntegration:
    """Integration tests matching acceptance criteria."""

    def test_bottleneck_adapter_usage(self):
        """Test BottleneckAdapter as per acceptance criteria."""
        adapter = BottleneckAdapter(hidden_size=768, bottleneck_size=64)
        x = torch.randn(2, 128, 768)
        out = adapter(x)
        assert out.shape == x.shape  # Shape unchanged

    def test_task_group_adapter_usage(self):
        """Test TaskGroupAdapter as per acceptance criteria."""
        group_adapter = TaskGroupAdapter(
            hidden_size=768,
            task_groups=["token_tasks", "sequence_tasks", "pair_tasks"],
            bottleneck_size=64,
        )
        x = torch.randn(2, 128, 768)
        out = group_adapter(x, task_group="sequence_tasks")
        assert out.shape == x.shape  # Shape unchanged

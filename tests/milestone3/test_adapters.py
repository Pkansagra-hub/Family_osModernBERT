"""
Tests for Issue 3.2.2: models/adapters.py

This module tests task-specific adapter layers for parameter-efficient
fine-tuning and multi-task learning.

Adapter Concepts Tested:
    - AdapterConfig: Configuration with validation
    - BottleneckAdapter: down → activation → up with residual (Houlsby et al. 2019)
    - TaskGroupAdapter: Task-specific adapters with optional shared down-projection
    - ParallelAdapter: output = x + scale * adapter(x)
    - LoRAAdapter: Low-rank adaptation W' = W + (α/r) * B @ A (Hu et al. 2021)
    - AdaptedLinear: Wraps Linear with LoRA, supports merge/unmerge
    - create_adapter: Factory function

Key Design Principles:
    - Minimal parameter overhead (<5% of base model)
    - Near-identity initialization for stable training start
    - Residual connections for gradient flow
    - Support for freezing/unfreezing independently

Mathematical Formulas Tested:
    - LoRA: W' = W + (α/r) * B @ A where A ∈ R^{r×d}, B ∈ R^{d×r}
    - Bottleneck: x → LayerNorm → down → activation → up → dropout → + x
    - Parallel: output = x + scale * adapter(x)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from modeling_studio.models.adapters import (
    AdapterConfig,
    TaskGroupConfig,
    get_activation,
    BottleneckAdapter,
    TaskGroupAdapter,
    ParallelAdapter,
    LoRAAdapter,
    AdaptedLinear,
    create_adapter,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hidden_size():
    """Standard hidden size for tests."""
    return 768


@pytest.fixture
def bottleneck_size():
    """Standard bottleneck size for tests."""
    return 64


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
def task_groups():
    """Standard task groups for testing."""
    return ["token_tasks", "sequence_tasks", "pair_tasks"]


# =============================================================================
# Test: AdapterConfig
# =============================================================================


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_adapter_config_init(self):
        """AdapterConfig initializes with correct defaults.

        Concept: Configuration dataclass holds all adapter hyperparameters
        including bottleneck size, activation type, dropout, and init scale.
        """
        config = AdapterConfig()

        assert config.bottleneck_size == 64
        assert config.adapter_type == "bottleneck"
        assert config.activation == "gelu"
        assert config.dropout == 0.1
        assert config.init_scale == 1e-3
        assert config.use_layer_norm is True
        assert config.residual_connection is True
        assert config.trainable is True

    def test_adapter_config_custom_values(self):
        """AdapterConfig accepts custom values."""
        config = AdapterConfig(
            bottleneck_size=128,
            adapter_type="parallel",
            activation="relu",
            dropout=0.2,
            init_scale=1e-4,
        )

        assert config.bottleneck_size == 128
        assert config.adapter_type == "parallel"
        assert config.activation == "relu"
        assert config.dropout == 0.2
        assert config.init_scale == 1e-4

    def test_adapter_config_validation_bottleneck_size(self):
        """AdapterConfig validates bottleneck_size > 0.

        Concept: The bottleneck dimension must be positive to create
        valid projection matrices.
        """
        with pytest.raises(ValueError, match="bottleneck_size must be positive"):
            AdapterConfig(bottleneck_size=0)

        with pytest.raises(ValueError, match="bottleneck_size must be positive"):
            AdapterConfig(bottleneck_size=-1)

    def test_adapter_config_validation_dropout(self):
        """AdapterConfig validates dropout in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            AdapterConfig(dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be in"):
            AdapterConfig(dropout=-0.1)

    def test_adapter_config_lora_params(self):
        """AdapterConfig includes LoRA-specific parameters."""
        config = AdapterConfig()

        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05


class TestTaskGroupConfig:
    """Tests for TaskGroupConfig dataclass."""

    def test_task_group_config_init(self):
        """TaskGroupConfig initializes correctly."""
        config = TaskGroupConfig()

        assert config.task_groups == ["default"]
        assert config.hidden_size == 768
        assert isinstance(config.adapter_config, AdapterConfig)
        assert config.share_down_projection is False

    def test_task_group_config_custom(self):
        """TaskGroupConfig with custom values."""
        config = TaskGroupConfig(
            task_groups=["token", "sequence"],
            hidden_size=1024,
            share_down_projection=True,
        )

        assert config.task_groups == ["token", "sequence"]
        assert config.hidden_size == 1024
        assert config.share_down_projection is True


# =============================================================================
# Test: get_activation
# =============================================================================


class TestGetActivation:
    """Tests for get_activation factory function."""

    def test_get_activation_gelu(self):
        """get_activation returns GELU for 'gelu'."""
        activation = get_activation("gelu")
        assert isinstance(activation, nn.GELU)

    def test_get_activation_relu(self):
        """get_activation returns ReLU for 'relu'."""
        activation = get_activation("relu")
        assert isinstance(activation, nn.ReLU)

    def test_get_activation_swish(self):
        """get_activation returns SiLU for 'swish'.

        Concept: Swish activation f(x) = x * sigmoid(x) is also known as SiLU.
        """
        activation = get_activation("swish")
        assert isinstance(activation, nn.SiLU)

    def test_get_activation_silu(self):
        """get_activation returns SiLU for 'silu' (alias for swish)."""
        activation = get_activation("silu")
        assert isinstance(activation, nn.SiLU)

    def test_get_activation_tanh(self):
        """get_activation returns Tanh for 'tanh'."""
        activation = get_activation("tanh")
        assert isinstance(activation, nn.Tanh)

    def test_get_activation_sigmoid(self):
        """get_activation returns Sigmoid for 'sigmoid'."""
        activation = get_activation("sigmoid")
        assert isinstance(activation, nn.Sigmoid)

    def test_get_activation_leaky_relu(self):
        """get_activation returns LeakyReLU for 'leaky_relu'."""
        activation = get_activation("leaky_relu")
        assert isinstance(activation, nn.LeakyReLU)

    def test_get_activation_case_insensitive(self):
        """get_activation is case-insensitive."""
        activation1 = get_activation("GELU")
        activation2 = get_activation("Gelu")
        activation3 = get_activation("gelu")

        assert isinstance(activation1, nn.GELU)
        assert isinstance(activation2, nn.GELU)
        assert isinstance(activation3, nn.GELU)

    def test_get_activation_unknown(self):
        """get_activation raises error for unknown activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("unknown")


# =============================================================================
# Test: BottleneckAdapter
# =============================================================================


class TestBottleneckAdapter:
    """Tests for BottleneckAdapter - classic bottleneck architecture."""

    def test_bottleneck_adapter_init(self, hidden_size, bottleneck_size):
        """BottleneckAdapter initializes with down/up projections.

        Concept: Bottleneck reduces dimensionality then projects back:
        down_proj: hidden_size → bottleneck_size
        up_proj: bottleneck_size → hidden_size
        """
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        assert adapter.hidden_size == hidden_size
        assert adapter.bottleneck_size == bottleneck_size
        assert hasattr(adapter, "down_proj")
        assert hasattr(adapter, "up_proj")
        assert hasattr(adapter, "activation")
        assert hasattr(adapter, "layer_norm")

        # Check projection dimensions
        assert adapter.down_proj.in_features == hidden_size
        assert adapter.down_proj.out_features == bottleneck_size
        assert adapter.up_proj.in_features == bottleneck_size
        assert adapter.up_proj.out_features == hidden_size

    def test_bottleneck_adapter_forward(self, hidden_states, hidden_size, bottleneck_size):
        """BottleneckAdapter preserves input shape.

        Concept: The adapter transforms hidden states through bottleneck
        and returns output with same shape as input.
        """
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        output = adapter(hidden_states)

        assert output.shape == hidden_states.shape

    def test_bottleneck_adapter_residual(self, hidden_states, hidden_size, bottleneck_size):
        """BottleneckAdapter adds residual connection.

        Concept: With residual_connection=True:
        output = x + adapter_transform(x)

        This allows the adapter to learn a delta from identity.
        """
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            residual_connection=True,
        )

        output = adapter(hidden_states)

        # Output should not be exactly equal to input (adapter does something)
        assert not torch.allclose(output, hidden_states, atol=1e-6)

    def test_bottleneck_adapter_no_residual(self, hidden_states, hidden_size, bottleneck_size):
        """BottleneckAdapter without residual connection."""
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            residual_connection=False,
        )

        output = adapter(hidden_states)

        assert output.shape == hidden_states.shape

    def test_bottleneck_adapter_near_identity_init(self, hidden_size, bottleneck_size):
        """BottleneckAdapter starts near identity with small init.

        Concept: Small initialization (init_scale=1e-3) ensures the adapter
        starts close to identity, allowing gradual learning of task-specific
        transformations. This is crucial for stable fine-tuning.

        Formula: With small weights, adapter_transform(x) ≈ 0, so output ≈ x
        """
        torch.manual_seed(42)
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            init_scale=1e-6,  # Very small init
            residual_connection=True,
        )
        adapter.eval()  # Disable dropout for determinism

        hidden_states = torch.randn(2, 8, hidden_size)
        output = adapter(hidden_states)

        # With very small init and residual, output should be close to input
        relative_diff = (output - hidden_states).abs() / (hidden_states.abs() + 1e-9)
        assert relative_diff.mean() < 0.1, "Near-identity initialization failed"

    def test_bottleneck_adapter_freeze_unfreeze(self, hidden_size, bottleneck_size):
        """BottleneckAdapter freeze/unfreeze controls requires_grad."""
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        # Initially trainable
        assert all(p.requires_grad for p in adapter.parameters())

        # Freeze
        adapter.freeze()
        assert all(not p.requires_grad for p in adapter.parameters())

        # Unfreeze
        adapter.unfreeze()
        assert all(p.requires_grad for p in adapter.parameters())

    def test_bottleneck_adapter_num_parameters(self, hidden_size, bottleneck_size):
        """BottleneckAdapter reports correct parameter count.

        Concept: Parameter-efficient adapters should add minimal overhead.
        Bottleneck parameters ≈ 2 × hidden_size × bottleneck_size
        """
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            use_layer_norm=False,  # Exclude LayerNorm for cleaner calculation
        )

        num_params = adapter.num_parameters

        # down_proj: hidden × bottleneck + bottleneck (bias)
        # up_proj: bottleneck × hidden + hidden (bias)
        expected_down = hidden_size * bottleneck_size + bottleneck_size
        expected_up = bottleneck_size * hidden_size + hidden_size
        expected = expected_down + expected_up

        assert num_params == expected

    def test_bottleneck_adapter_no_layer_norm(self, hidden_states, hidden_size, bottleneck_size):
        """BottleneckAdapter works without layer norm."""
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            use_layer_norm=False,
        )

        assert adapter.layer_norm is None

        output = adapter(hidden_states)
        assert output.shape == hidden_states.shape

    def test_bottleneck_adapter_different_activations(
        self, hidden_states, hidden_size, bottleneck_size
    ):
        """BottleneckAdapter works with different activations."""
        for activation in ["relu", "gelu", "tanh", "swish"]:
            adapter = BottleneckAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                activation=activation,
            )

            output = adapter(hidden_states)
            assert output.shape == hidden_states.shape


# =============================================================================
# Test: TaskGroupAdapter
# =============================================================================


class TestTaskGroupAdapter:
    """Tests for TaskGroupAdapter - task-group-specific adapters."""

    def test_task_group_adapter_init(self, hidden_size, bottleneck_size, task_groups):
        """TaskGroupAdapter initializes adapters for each group.

        Concept: Maintains separate adapters for different task groups
        (e.g., token_tasks, sequence_tasks, pair_tasks), allowing
        task-group-specific transformations.
        """
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )

        assert adapter.hidden_size == hidden_size
        assert adapter.task_groups == task_groups
        assert hasattr(adapter, "down_projs")  # Separate down projections
        assert hasattr(adapter, "up_projs")  # Separate up projections

        # One adapter per task group
        assert len(adapter.up_projs) == len(task_groups)

    def test_task_group_adapter_forward(
        self, hidden_states, hidden_size, bottleneck_size, task_groups
    ):
        """TaskGroupAdapter routes to correct adapter by task_group.

        Concept: During forward pass, the task_group argument determines
        which adapter is used. Different task groups get different transformations.
        """
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )

        for task_group in task_groups:
            output = adapter(hidden_states, task_group=task_group)
            assert output.shape == hidden_states.shape

    def test_task_group_adapter_different_outputs(
        self, hidden_states, hidden_size, bottleneck_size, task_groups
    ):
        """Different task groups produce different outputs.

        Concept: Each task group has its own adapter parameters,
        so the same input produces different outputs per group.
        """
        torch.manual_seed(42)
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )
        adapter.eval()  # Disable dropout

        outputs = {}
        for task_group in task_groups:
            outputs[task_group] = adapter(hidden_states, task_group=task_group)

        # Outputs should be different for different task groups
        assert not torch.allclose(outputs["token_tasks"], outputs["sequence_tasks"], atol=1e-4)
        assert not torch.allclose(outputs["token_tasks"], outputs["pair_tasks"], atol=1e-4)

    def test_task_group_adapter_shared_down(
        self, hidden_states, hidden_size, bottleneck_size, task_groups
    ):
        """TaskGroupAdapter with shared down-projection.

        Concept: share_down_projection=True uses a single down_proj
        for all groups, but separate up_projs. This reduces parameters
        while maintaining task-specific transformations.
        """
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
            share_down_projection=True,
        )

        assert hasattr(adapter, "down_proj")  # Single shared down projection
        assert adapter.share_down_projection is True

        # Still has separate up projections
        assert len(adapter.up_projs) == len(task_groups)

        for task_group in task_groups:
            output = adapter(hidden_states, task_group=task_group)
            assert output.shape == hidden_states.shape

    def test_task_group_adapter_unknown_group(
        self, hidden_states, hidden_size, bottleneck_size, task_groups
    ):
        """TaskGroupAdapter raises error for unknown task_group."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )

        with pytest.raises(ValueError, match="Unknown task_group"):
            adapter(hidden_states, task_group="unknown")

    def test_task_group_adapter_freeze_single_group(
        self, hidden_size, bottleneck_size, task_groups
    ):
        """TaskGroupAdapter can freeze individual task groups."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )

        # Freeze only one group
        adapter.freeze(task_group="token_tasks")

        # Check that token_tasks up_proj is frozen
        for param in adapter.up_projs["token_tasks"].parameters():
            assert not param.requires_grad

    def test_task_group_adapter_num_parameters(self, hidden_size, bottleneck_size, task_groups):
        """TaskGroupAdapter reports trainable parameter count."""
        adapter = TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
        )

        num_params = adapter.num_parameters
        assert num_params > 0


# =============================================================================
# Test: ParallelAdapter
# =============================================================================


class TestParallelAdapter:
    """Tests for ParallelAdapter - parallel addition to residual stream."""

    def test_parallel_adapter_init(self, hidden_size, bottleneck_size):
        """ParallelAdapter initializes with learnable scale.

        Concept: Unlike bottleneck adapters that transform sequentially,
        parallel adapters compute a delta added to the input:
        output = x + scale * adapter(x)
        """
        adapter = ParallelAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        assert adapter.hidden_size == hidden_size
        assert adapter.bottleneck_size == bottleneck_size
        assert hasattr(adapter, "scale")
        assert isinstance(adapter.scale, nn.Parameter)

    def test_parallel_adapter_forward(self, hidden_states, hidden_size, bottleneck_size):
        """ParallelAdapter adds scaled delta to input.

        Concept: output = x + scale * adapter_transform(x)
        This allows easy tuning of adapter strength via scale.
        """
        adapter = ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            output_scale=1.0,
        )

        output = adapter(hidden_states)

        assert output.shape == hidden_states.shape

    def test_parallel_adapter_scale(self, hidden_size, bottleneck_size):
        """ParallelAdapter output_scale controls contribution.

        Concept: With scale=0, output = x (adapter has no effect).
        With larger scale, adapter has more influence.

        Note: Because up_proj is zero-initialized, we need to set
        non-zero weights to see the scale effect.
        """
        torch.manual_seed(42)
        hidden_states = torch.randn(2, 8, hidden_size)

        adapter_small = ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            output_scale=0.01,
        )
        adapter_large = ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            output_scale=1.0,
        )

        # Initialize up_proj with non-zero weights to see scale effect
        # (by default it's zero for stability)
        with torch.no_grad():
            adapter_small.up_proj.weight.fill_(0.1)
            adapter_large.up_proj.weight.fill_(0.1)

        output_small = adapter_small(hidden_states)
        output_large = adapter_large(hidden_states)

        # Smaller scale should be closer to input
        diff_small = (output_small - hidden_states).abs().mean()
        diff_large = (output_large - hidden_states).abs().mean()

        assert diff_small < diff_large

    def test_parallel_adapter_learnable_scale(self, hidden_size, bottleneck_size):
        """ParallelAdapter scale is learnable parameter."""
        adapter = ParallelAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        assert adapter.scale.requires_grad is True

    def test_parallel_adapter_zero_init_up_proj(self, hidden_size, bottleneck_size):
        """ParallelAdapter up_proj initialized to zero for stability.

        Concept: Zero-initializing up_proj ensures the adapter starts
        as identity (output = x + 0), then gradually learns.
        """
        adapter = ParallelAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)

        # up_proj should be initialized to zeros
        assert torch.allclose(
            adapter.up_proj.weight, torch.zeros_like(adapter.up_proj.weight), atol=1e-8
        )


# =============================================================================
# Test: LoRAAdapter
# =============================================================================


class TestLoRAAdapter:
    """Tests for LoRAAdapter - Low-Rank Adaptation."""

    def test_lora_adapter_init(self, hidden_size):
        """LoRAAdapter initializes with low-rank matrices A and B.

        Concept: LoRA learns W' = W + (α/r) * B @ A where:
        - A ∈ R^{r×d} initialized with Kaiming
        - B ∈ R^{d×r} initialized to zeros
        - r is the rank (typically 4, 8, 16)
        - α is the scaling factor

        Reference: Hu et al. "LoRA: Low-Rank Adaptation" (2021)
        """
        r = 8
        alpha = 16
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=r,
            alpha=alpha,
        )

        assert adapter.in_features == hidden_size
        assert adapter.out_features == hidden_size
        assert adapter.r == r
        assert adapter.alpha == alpha
        assert adapter.scaling == alpha / r  # α/r scaling

        # Check matrix dimensions
        assert adapter.lora_A.shape == (r, hidden_size)
        assert adapter.lora_B.shape == (hidden_size, r)

    def test_lora_adapter_forward(self, hidden_states, hidden_size):
        """LoRAAdapter computes low-rank update.

        Concept: The forward pass computes:
        output = x @ A.T @ B.T * (α/r)

        This is the delta to add to the base layer output.
        """
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )
        batch_size, seq_len = hidden_states.shape[:2]

        output = adapter(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_lora_adapter_b_zero_init(self, hidden_size):
        """LoRA B matrix initialized to zeros.

        Concept: B starts at zero so initially LoRA has no effect.
        This ensures stable training start.
        """
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )

        assert torch.allclose(adapter.lora_B, torch.zeros_like(adapter.lora_B), atol=1e-8)

    def test_lora_adapter_a_kaiming_init(self, hidden_size):
        """LoRA A matrix initialized with Kaiming.

        Concept: A uses Kaiming initialization for proper gradient
        scaling when combined with the zero-initialized B.
        """
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )

        # A should NOT be all zeros (Kaiming initialized)
        assert not torch.allclose(adapter.lora_A, torch.zeros_like(adapter.lora_A), atol=1e-8)

    def test_lora_adapter_merge(self, hidden_size):
        """LoRAAdapter merge combines LoRA into base weights.

        Concept: For inference efficiency, LoRA can be merged:
        W_merged = W + (α/r) * B @ A

        This eliminates the extra computation at inference time.
        """
        r = 8
        alpha = 16
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=r,
            alpha=alpha,
        )

        # Create fake base weight
        base_weight = torch.randn(hidden_size, hidden_size)

        # Merge
        merged = adapter.merge(base_weight)

        # Verify merge formula: W + (α/r) * B @ A
        expected_delta = (adapter.lora_B @ adapter.lora_A) * (alpha / r)
        expected = base_weight + expected_delta

        assert torch.allclose(merged, expected, atol=1e-5)

    def test_lora_adapter_freeze_unfreeze(self, hidden_size):
        """LoRAAdapter freeze/unfreeze controls requires_grad."""
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=8,
            alpha=16,
        )

        # Initially trainable
        assert adapter.lora_A.requires_grad is True
        assert adapter.lora_B.requires_grad is True

        # Freeze
        adapter.freeze()
        assert adapter.lora_A.requires_grad is False
        assert adapter.lora_B.requires_grad is False

        # Unfreeze
        adapter.unfreeze()
        assert adapter.lora_A.requires_grad is True
        assert adapter.lora_B.requires_grad is True

    def test_lora_adapter_num_parameters(self, hidden_size):
        """LoRAAdapter has minimal parameters.

        Concept: LoRA adds only r×d + d×r = 2×r×d parameters,
        which is much smaller than d² for the full weight matrix.
        """
        r = 8
        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=r,
            alpha=16,
        )

        expected = r * hidden_size + hidden_size * r  # A + B
        assert adapter.num_parameters == expected

        # Should be much smaller than full weight matrix
        full_weight_params = hidden_size * hidden_size
        assert adapter.num_parameters < 0.1 * full_weight_params  # < 10% of full


# =============================================================================
# Test: AdaptedLinear
# =============================================================================


class TestAdaptedLinear:
    """Tests for AdaptedLinear - Linear layer with LoRA."""

    def test_adapted_linear_init(self, hidden_size):
        """AdaptedLinear wraps base linear with LoRA.

        Concept: Combines a frozen base Linear layer with trainable
        LoRA parameters for parameter-efficient fine-tuning.
        """
        base_linear = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base_linear, r=8, alpha=16)

        assert adapted.base_linear is base_linear
        assert hasattr(adapted, "lora")
        assert isinstance(adapted.lora, LoRAAdapter)
        assert adapted.merged is False

    def test_adapted_linear_forward(self, hidden_states, hidden_size):
        """AdaptedLinear forward = base + lora.

        Concept: During training, output = base_linear(x) + lora(x)
        """
        base_linear = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base_linear, r=8, alpha=16)

        output = adapted(hidden_states)

        assert output.shape == hidden_states.shape

    def test_adapted_linear_merge_unmerge(self, hidden_states, hidden_size):
        """AdaptedLinear can merge/unmerge LoRA weights.

        Concept: For inference, merge LoRA into base weights to
        eliminate extra computation. Unmerge to continue training.
        """
        base_linear = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base_linear, r=8, alpha=16)

        # Get output before merge
        output_before = adapted(hidden_states)

        # Merge
        adapted.merge_weights()
        assert adapted.merged is True
        output_merged = adapted(hidden_states)

        # Outputs should be approximately equal
        assert torch.allclose(output_before, output_merged, atol=1e-5)

        # Unmerge
        adapted.unmerge_weights()
        assert adapted.merged is False
        output_unmerged = adapted(hidden_states)

        # Should still match
        assert torch.allclose(output_before, output_unmerged, atol=1e-5)

    def test_adapted_linear_freeze_base(self, hidden_size):
        """AdaptedLinear can freeze base, keep LoRA trainable.

        Concept: Typical fine-tuning freezes base weights and only
        trains the LoRA parameters for efficiency.
        """
        base_linear = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base_linear, r=8, alpha=16)

        adapted.freeze_base()

        # Base should be frozen
        assert adapted.base_linear.weight.requires_grad is False

        # LoRA should be trainable
        assert adapted.lora.lora_A.requires_grad is True
        assert adapted.lora.lora_B.requires_grad is True

    def test_adapted_linear_freeze_lora(self, hidden_size):
        """AdaptedLinear can freeze LoRA, keep base trainable."""
        base_linear = nn.Linear(hidden_size, hidden_size)
        adapted = AdaptedLinear(base_linear, r=8, alpha=16)

        adapted.freeze_lora()

        # LoRA should be frozen
        assert adapted.lora.lora_A.requires_grad is False
        assert adapted.lora.lora_B.requires_grad is False

        # Base should be trainable
        assert adapted.base_linear.weight.requires_grad is True


# =============================================================================
# Test: create_adapter Factory Function
# =============================================================================


class TestCreateAdapterFactory:
    """Tests for create_adapter factory function."""

    def test_create_adapter_bottleneck(self, hidden_size):
        """Factory creates BottleneckAdapter for 'bottleneck' type."""
        adapter = create_adapter("bottleneck", hidden_size=hidden_size)
        assert isinstance(adapter, BottleneckAdapter)

    def test_create_adapter_parallel(self, hidden_size):
        """Factory creates ParallelAdapter for 'parallel' type."""
        adapter = create_adapter("parallel", hidden_size=hidden_size)
        assert isinstance(adapter, ParallelAdapter)

    def test_create_adapter_lora(self, hidden_size):
        """Factory creates LoRAAdapter for 'lora' type."""
        adapter = create_adapter("lora", hidden_size=hidden_size)
        assert isinstance(adapter, LoRAAdapter)

    def test_create_adapter_task_group(self, hidden_size, task_groups):
        """Factory creates TaskGroupAdapter for 'task_group' type."""
        adapter = create_adapter(
            "task_group",
            hidden_size=hidden_size,
            task_groups=task_groups,
        )
        assert isinstance(adapter, TaskGroupAdapter)

    def test_create_adapter_with_config(self, hidden_size):
        """Factory uses provided AdapterConfig."""
        config = AdapterConfig(bottleneck_size=128, activation="relu")
        adapter = create_adapter("bottleneck", hidden_size=hidden_size, config=config)

        assert adapter.bottleneck_size == 128

    def test_create_adapter_kwargs_override(self, hidden_size):
        """Factory kwargs override config values."""
        config = AdapterConfig(bottleneck_size=64)
        adapter = create_adapter(
            "bottleneck",
            hidden_size=hidden_size,
            config=config,
            bottleneck_size=256,  # Override
        )

        assert adapter.bottleneck_size == 256

    def test_create_adapter_unknown_type(self, hidden_size):
        """Factory raises error for unknown adapter type."""
        with pytest.raises(ValueError, match="Unknown adapter_type"):
            create_adapter("unknown", hidden_size=hidden_size)

    def test_create_adapter_case_insensitive(self, hidden_size):
        """Factory is case-insensitive."""
        adapter1 = create_adapter("BOTTLENECK", hidden_size=hidden_size)
        adapter2 = create_adapter("Bottleneck", hidden_size=hidden_size)
        adapter3 = create_adapter("bottleneck", hidden_size=hidden_size)

        assert isinstance(adapter1, BottleneckAdapter)
        assert isinstance(adapter2, BottleneckAdapter)
        assert isinstance(adapter3, BottleneckAdapter)


# =============================================================================
# Test: Edge Cases and Mathematical Correctness
# =============================================================================


class TestAdapterEdgeCases:
    """Test edge cases and verify mathematical correctness."""

    def test_adapter_gradient_flow(self, hidden_size, bottleneck_size):
        """Adapters allow gradient flow back to parameters."""
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)
        hidden_states = torch.randn(2, 8, hidden_size, requires_grad=True)

        output = adapter(hidden_states)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None

        # Adapter parameters should have gradients
        assert adapter.down_proj.weight.grad is not None
        assert adapter.up_proj.weight.grad is not None

    def test_adapter_deterministic(self, hidden_size, bottleneck_size):
        """Adapters produce deterministic output in eval mode."""
        torch.manual_seed(42)
        adapter = BottleneckAdapter(hidden_size=hidden_size, bottleneck_size=bottleneck_size)
        adapter.eval()

        hidden_states = torch.randn(2, 8, hidden_size)

        output1 = adapter(hidden_states)
        output2 = adapter(hidden_states)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_lora_scaling_correctness(self, hidden_size):
        """LoRA scaling factor (α/r) is applied correctly.

        Mathematical verification:
        output = x @ A.T @ B.T * (α/r)
        """
        r = 8
        alpha = 16
        scaling = alpha / r  # = 2.0

        adapter = LoRAAdapter(
            in_features=hidden_size,
            out_features=hidden_size,
            r=r,
            alpha=alpha,
        )
        adapter.eval()

        # Set known values for verification
        adapter.lora_A.data = torch.ones(r, hidden_size)
        adapter.lora_B.data = torch.ones(hidden_size, r)

        x = torch.ones(1, 1, hidden_size)
        output = adapter(x)

        # Manual calculation:
        # x @ A.T = 1s @ 1s.T = hidden_size (each element)
        # result @ B.T = hidden_size * r (each element)
        # scaled = hidden_size * r * scaling
        expected_value = hidden_size * r * scaling

        assert torch.allclose(output, torch.full_like(output, expected_value), atol=1e-4)

    def test_bottleneck_compression_ratio(self, hidden_size, bottleneck_size):
        """Bottleneck adapter achieves expected compression.

        Concept: Bottleneck reduces from hidden_size to bottleneck_size,
        achieving compression ratio of hidden_size / bottleneck_size.
        """
        adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            use_layer_norm=False,
        )

        compression_ratio = hidden_size / bottleneck_size

        # Verify dimensions
        assert adapter.down_proj.out_features == bottleneck_size
        assert adapter.up_proj.in_features == bottleneck_size

        # With hidden=768, bottleneck=64: ratio = 12x compression
        assert compression_ratio == hidden_size / bottleneck_size


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    """Test that all expected classes are exported from the module."""

    def test_all_adapters_exported(self):
        """Verify __all__ contains all adapter classes."""
        from modeling_studio.models import adapters

        expected = [
            "AdapterConfig",
            "TaskGroupConfig",
            "BottleneckAdapter",
            "TaskGroupAdapter",
            "ParallelAdapter",
            "LoRAAdapter",
            "AdaptedLinear",
            "create_adapter",
            "get_activation",
        ]

        for name in expected:
            assert name in adapters.__all__, f"{name} not in __all__"
            assert hasattr(adapters, name), f"{name} not exported"

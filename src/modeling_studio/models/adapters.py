"""
Task-Specific Adapter Modules for Multi-Task Learning

This module provides lightweight adapter layers that can be inserted into
the encoder to enable task-specific transformations without full fine-tuning.

Adapters Implemented:
    - BottleneckAdapter: Classic bottleneck architecture (down-project → activation → up-project)
    - TaskGroupAdapter: Multiple adapters for different task groups
    - ParallelAdapter: Parallel adapter that adds to residual stream
    - LoRAAdapter: Low-Rank Adaptation compatible with PEFT

Design Principles:
    - Minimal parameter overhead (<5% of base model)
    - Residual connections for gradient flow
    - Support for freezing/unfreezing independently
    - Compatible with PEFT library patterns

References:
    - "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)
    - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    - "AdapterHub: A Framework for Adapting Transformers" (Pfeiffer et al., 2020)

Issue: 5.0.2 - Implement Task-Specific Adapters
Epic: 5.0 - Model Architecture Enhancements (Pre-Stage B)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


# =============================================================================
# Adapter Configuration
# =============================================================================


@dataclass
class AdapterConfig:
    """
    Configuration for adapter layers.

    Args:
        bottleneck_size: Size of the bottleneck dimension (typical: 64, 128, 256)
        adapter_type: Type of adapter ("bottleneck", "parallel", "lora")
        activation: Activation function ("relu", "gelu", "swish", "tanh")
        dropout: Dropout probability in adapter
        init_scale: Scale for weight initialization (smaller = closer to identity)
        use_layer_norm: Apply layer norm before adapter
        residual_connection: Whether to add residual connection
        trainable: Whether adapter parameters are trainable by default
    """

    bottleneck_size: int = 64
    adapter_type: Literal["bottleneck", "parallel", "lora"] = "bottleneck"
    activation: str = "gelu"
    dropout: float = 0.1
    init_scale: float = 1e-3
    use_layer_norm: bool = True
    residual_connection: bool = True
    trainable: bool = True

    # LoRA-specific parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self):
        """Validate configuration."""
        if self.bottleneck_size <= 0:
            raise ValueError(f"bottleneck_size must be positive, got {self.bottleneck_size}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class TaskGroupConfig:
    """
    Configuration for task group adapters.

    Args:
        task_groups: List of task group names
        hidden_size: Size of encoder hidden states
        adapter_config: Base adapter configuration
        share_down_projection: Share down-projection across groups
    """

    task_groups: list[str] = field(default_factory=lambda: ["default"])
    hidden_size: int = 768
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    share_down_projection: bool = False


# =============================================================================
# Activation Functions
# =============================================================================


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


# =============================================================================
# Bottleneck Adapter
# =============================================================================


class BottleneckAdapter(nn.Module):
    """
    Classic bottleneck adapter layer.

    Architecture:
        x → LayerNorm → down_proj → activation → up_proj → dropout → + x (residual)

    The bottleneck reduces dimensionality, applies non-linearity, then projects
    back up. This allows task-specific transformations with minimal parameters.

    Args:
        hidden_size: Size of input/output hidden states
        bottleneck_size: Size of bottleneck dimension
        activation: Activation function name
        dropout: Dropout probability
        init_scale: Scale for weight initialization
        use_layer_norm: Apply layer norm before adapter
        residual_connection: Whether to add residual connection

    Example:
        >>> adapter = BottleneckAdapter(hidden_size=768, bottleneck_size=64)
        >>> x = torch.randn(2, 128, 768)
        >>> out = adapter(x)
        >>> assert out.shape == x.shape
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        use_layer_norm: bool = True,
        residual_connection: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.residual_connection = residual_connection

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None

        # Down projection: hidden_size → bottleneck_size
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)

        # Activation
        self.activation = get_activation(activation)

        # Up projection: bottleneck_size → hidden_size
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        """
        Initialize weights for near-identity at start.

        Small initialization ensures the adapter starts close to identity,
        allowing gradual learning of task-specific transformations.
        """
        # Initialize down_proj with small weights
        nn.init.normal_(self.down_proj.weight, std=scale)
        nn.init.zeros_(self.down_proj.bias)

        # Initialize up_proj to be near-zero (so residual dominates initially)
        nn.init.normal_(self.up_proj.weight, std=scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply bottleneck adapter.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            Adapted hidden states (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states

        # Optional layer norm
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # Bottleneck transformation
        hidden_states = self.down_proj(hidden_states)  # (batch, seq, bottleneck)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_proj(hidden_states)  # (batch, seq, hidden)
        hidden_states = self.dropout(hidden_states)

        # Residual connection
        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states

    def freeze(self) -> None:
        """Freeze all adapter parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all adapter parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Task Group Adapter
# =============================================================================


class TaskGroupAdapter(nn.Module):
    """
    Adapter with separate bottleneck for each task group.

    Maintains multiple adapters, one per task group (e.g., token_tasks,
    sequence_tasks, pair_tasks). During forward pass, the appropriate
    adapter is selected based on the task_group argument.

    Args:
        hidden_size: Size of input/output hidden states
        task_groups: List of task group names
        bottleneck_size: Size of bottleneck dimension
        activation: Activation function name
        dropout: Dropout probability
        init_scale: Scale for weight initialization
        share_down_projection: Whether to share down-projection across groups

    Example:
        >>> adapter = TaskGroupAdapter(
        ...     hidden_size=768,
        ...     task_groups=["token_tasks", "sequence_tasks", "pair_tasks"],
        ...     bottleneck_size=64,
        ... )
        >>> x = torch.randn(2, 128, 768)
        >>> out = adapter(x, task_group="sequence_tasks")
        >>> assert out.shape == x.shape
    """

    def __init__(
        self,
        hidden_size: int,
        task_groups: list[str],
        bottleneck_size: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        share_down_projection: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.task_groups = task_groups
        self.bottleneck_size = bottleneck_size
        self.share_down_projection = share_down_projection
        self._activation_name = activation
        self._dropout_rate = dropout
        self._init_scale = init_scale

        # Layer norm (shared across all groups)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Activation
        self.activation = get_activation(activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Create adapters for each task group
        if share_down_projection:
            # Shared down projection, separate up projections
            self.down_proj = nn.Linear(hidden_size, bottleneck_size)
            self.up_projs = nn.ModuleDict(
                {group: nn.Linear(bottleneck_size, hidden_size) for group in task_groups}
            )
            self._init_shared_weights(init_scale)
        else:
            # Separate down/up projections for each group
            self.down_projs = nn.ModuleDict(
                {group: nn.Linear(hidden_size, bottleneck_size) for group in task_groups}
            )
            self.up_projs = nn.ModuleDict(
                {group: nn.Linear(bottleneck_size, hidden_size) for group in task_groups}
            )
            self._init_separate_weights(init_scale)

    def _init_shared_weights(self, scale: float) -> None:
        """Initialize weights for shared down-projection mode."""
        nn.init.normal_(self.down_proj.weight, std=scale)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
        for group in self.task_groups:
            up_proj = self.up_projs[group]
            if isinstance(up_proj, nn.Linear):
                nn.init.normal_(up_proj.weight, std=scale)
                if up_proj.bias is not None:
                    nn.init.zeros_(up_proj.bias)

    def _init_separate_weights(self, scale: float) -> None:
        """Initialize weights for separate adapters mode."""
        for group in self.task_groups:
            down_proj = self.down_projs[group]
            up_proj = self.up_projs[group]
            if isinstance(down_proj, nn.Linear):
                nn.init.normal_(down_proj.weight, std=scale)
                if down_proj.bias is not None:
                    nn.init.zeros_(down_proj.bias)
            if isinstance(up_proj, nn.Linear):
                nn.init.normal_(up_proj.weight, std=scale)
                if up_proj.bias is not None:
                    nn.init.zeros_(up_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        task_group: str,
    ) -> torch.Tensor:
        """
        Apply task-group-specific adapter.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            task_group: Name of the task group to use

        Returns:
            Adapted hidden states (batch_size, seq_len, hidden_size)
        """
        if task_group not in self.task_groups:
            raise ValueError(f"Unknown task_group: {task_group}. Available: {self.task_groups}")

        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        if self.share_down_projection:
            # Shared down, separate up
            hidden_states = self.down_proj(hidden_states)
            hidden_states = self.activation(hidden_states)
            up_proj = self.up_projs[task_group]
            hidden_states = up_proj(hidden_states)
        else:
            # Separate adapters
            down_proj = self.down_projs[task_group]
            up_proj = self.up_projs[task_group]
            hidden_states = down_proj(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = up_proj(hidden_states)

        hidden_states = self.dropout(hidden_states)

        # Residual connection
        return hidden_states + residual

    def freeze(self, task_group: str | None = None) -> None:
        """
        Freeze adapter parameters.

        Args:
            task_group: If provided, freeze only this group. Otherwise freeze all.
        """
        if task_group is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            if self.share_down_projection:
                for param in self.up_projs[task_group].parameters():
                    param.requires_grad = False
            else:
                for param in self.down_projs[task_group].parameters():
                    param.requires_grad = False
                for param in self.up_projs[task_group].parameters():
                    param.requires_grad = False

    def unfreeze(self, task_group: str | None = None) -> None:
        """
        Unfreeze adapter parameters.

        Args:
            task_group: If provided, unfreeze only this group. Otherwise unfreeze all.
        """
        if task_group is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            if self.share_down_projection:
                for param in self.up_projs[task_group].parameters():
                    param.requires_grad = True
            else:
                for param in self.down_projs[task_group].parameters():
                    param.requires_grad = True
                for param in self.up_projs[task_group].parameters():
                    param.requires_grad = True

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Parallel Adapter
# =============================================================================


class ParallelAdapter(nn.Module):
    """
    Parallel adapter that runs alongside the main computation.

    Unlike bottleneck adapters that transform the hidden states sequentially,
    parallel adapters compute a delta that is added to the original:

        output = x + scale * adapter(x)

    This design allows for better gradient flow and easier tuning of
    adapter strength via the scale parameter.

    Args:
        hidden_size: Size of input/output hidden states
        bottleneck_size: Size of bottleneck dimension
        activation: Activation function name
        dropout: Dropout probability
        init_scale: Scale for weight initialization
        output_scale: Scale factor for adapter output (default: 1.0)

    Example:
        >>> adapter = ParallelAdapter(hidden_size=768, bottleneck_size=64, output_scale=0.1)
        >>> x = torch.randn(2, 128, 768)
        >>> out = adapter(x)
        >>> assert out.shape == x.shape
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.output_scale = output_scale

        # Bottleneck layers
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = get_activation(activation)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Learnable scale (optional)
        self.scale = nn.Parameter(torch.tensor(output_scale))

        # Initialize
        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        """Initialize weights."""
        nn.init.normal_(self.down_proj.weight, std=scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)  # Start at zero for parallel
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply parallel adapter.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            Adapted hidden states (batch_size, seq_len, hidden_size)
        """
        # Compute adapter delta
        delta = self.down_proj(hidden_states)
        delta = self.activation(delta)
        delta = self.up_proj(delta)
        delta = self.dropout(delta)

        # Add scaled delta to input
        return hidden_states + self.scale * delta

    def freeze(self) -> None:
        """Freeze adapter parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze adapter parameters."""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# LoRA Adapter
# =============================================================================


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.

    Implements LoRA for parameter-efficient fine-tuning. Instead of updating
    all weights W, LoRA learns low-rank matrices A and B such that:

        W' = W + (alpha/r) * B @ A

    Where A ∈ R^{r×d}, B ∈ R^{d×r}, and r << d.

    This module wraps a linear layer and adds LoRA adaptations.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        r: LoRA rank (smaller = fewer parameters)
        alpha: LoRA scaling factor
        dropout: Dropout on LoRA path
        merge_weights: Whether to merge LoRA into base weights for inference

    Example:
        >>> # Wrap an existing linear layer
        >>> base_linear = nn.Linear(768, 768)
        >>> lora = LoRAAdapter(768, 768, r=8, alpha=16)
        >>> x = torch.randn(2, 128, 768)
        >>> out = base_linear(x) + lora(x)  # Or use merged mode
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merge_weights = merge_weights
        self.merged = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LoRA weights."""
        # A uses Kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero (so initially LoRA has no effect)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta.

        Args:
            x: Input tensor (batch_size, seq_len, in_features)

        Returns:
            LoRA output to add to base layer output
        """
        # x @ A.T @ B.T * scaling
        # x: (batch, seq, in_features)
        # A: (r, in_features) -> A.T: (in_features, r)
        # B: (out_features, r) -> B.T: (r, out_features)

        result = self.dropout(x)
        result = result @ self.lora_A.T  # (batch, seq, r)
        result = result @ self.lora_B.T  # (batch, seq, out_features)
        result = result * self.scaling

        return result

    def merge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights into base weights.

        Args:
            base_weight: Base layer weight (out_features, in_features)

        Returns:
            Merged weight tensor
        """
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return base_weight + delta

    def freeze(self) -> None:
        """Freeze LoRA parameters."""
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze LoRA parameters."""
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    @property
    def num_parameters(self) -> int:
        """Return number of LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


# =============================================================================
# Adapter Wrapper for Linear Layers
# =============================================================================


class AdaptedLinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps a base linear layer and adds LoRA parameters. During forward pass:
        output = base_linear(x) + lora(x)

    Can optionally merge LoRA weights into base weights for inference.

    Args:
        base_linear: Base nn.Linear layer to adapt
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout on LoRA path

    Example:
        >>> base = nn.Linear(768, 768)
        >>> adapted = AdaptedLinear(base, r=8, alpha=16)
        >>> x = torch.randn(2, 128, 768)
        >>> out = adapted(x)  # Automatically combines base + LoRA
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.lora = LoRAAdapter(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with base + LoRA."""
        if self.merged:
            return F.linear(x, self.base_linear.weight, self.base_linear.bias)
        return self.base_linear(x) + self.lora(x)

    def merge_weights(self) -> None:
        """Merge LoRA into base weights for faster inference."""
        if not self.merged:
            self.base_linear.weight.data = self.lora.merge(self.base_linear.weight)
            self.merged = True

    def unmerge_weights(self) -> None:
        """Unmerge LoRA from base weights."""
        if self.merged:
            delta = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_linear.weight.data = self.base_linear.weight - delta
            self.merged = False

    def freeze_base(self) -> None:
        """Freeze base linear layer, keep LoRA trainable."""
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
        self.lora.unfreeze()

    def freeze_lora(self) -> None:
        """Freeze LoRA, keep base trainable."""
        self.lora.freeze()
        self.base_linear.weight.requires_grad = True
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = True


# =============================================================================
# Factory Function
# =============================================================================


def create_adapter(
    adapter_type: str,
    hidden_size: int,
    config: AdapterConfig | None = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create adapters.

    Args:
        adapter_type: Type of adapter ("bottleneck", "parallel", "lora", "task_group")
        hidden_size: Size of hidden states
        config: Optional AdapterConfig (uses defaults if None)
        **kwargs: Override config parameters

    Returns:
        Adapter module

    Example:
        >>> adapter = create_adapter("bottleneck", hidden_size=768, bottleneck_size=64)
        >>> adapter = create_adapter("task_group", hidden_size=768, task_groups=["a", "b"])
    """
    if config is None:
        config = AdapterConfig()

    # Override config with kwargs
    bottleneck_size = kwargs.get("bottleneck_size", config.bottleneck_size)
    activation = kwargs.get("activation", config.activation)
    dropout = kwargs.get("dropout", config.dropout)
    init_scale = kwargs.get("init_scale", config.init_scale)

    adapter_type = adapter_type.lower()

    if adapter_type == "bottleneck":
        return BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            activation=activation,
            dropout=dropout,
            init_scale=init_scale,
            use_layer_norm=config.use_layer_norm,
            residual_connection=config.residual_connection,
        )
    elif adapter_type == "parallel":
        output_scale = kwargs.get("output_scale", 1.0)
        return ParallelAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            activation=activation,
            dropout=dropout,
            init_scale=init_scale,
            output_scale=output_scale,
        )
    elif adapter_type == "lora":
        in_features = kwargs.get("in_features", hidden_size)
        out_features = kwargs.get("out_features", hidden_size)
        r = kwargs.get("r", config.lora_r)
        alpha = kwargs.get("alpha", config.lora_alpha)
        return LoRAAdapter(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=config.lora_dropout,
        )
    elif adapter_type == "task_group":
        task_groups = kwargs.get("task_groups", ["default"])
        share_down = kwargs.get("share_down_projection", False)
        return TaskGroupAdapter(
            hidden_size=hidden_size,
            task_groups=task_groups,
            bottleneck_size=bottleneck_size,
            activation=activation,
            dropout=dropout,
            init_scale=init_scale,
            share_down_projection=share_down,
        )
    else:
        raise ValueError(
            f"Unknown adapter_type: {adapter_type}. "
            f"Available: bottleneck, parallel, lora, task_group"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "AdapterConfig",
    "TaskGroupConfig",
    # Adapters
    "BottleneckAdapter",
    "TaskGroupAdapter",
    "ParallelAdapter",
    "LoRAAdapter",
    "AdaptedLinear",
    # Factory
    "create_adapter",
    # Utils
    "get_activation",
]

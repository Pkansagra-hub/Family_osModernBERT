"""
LoRA (Low-Rank Adaptation) for ModernBERT v3 Trainers.

This module provides LoRA management utilities for the training pipeline,
specifically designed for fine-tuning layers 23-28 (Family Band) with
efficient parameter updates.

Key Components:
    - LoRAConfig: Configuration for LoRA adapters
    - LoRALinear: Linear layer with LoRA adapter attached
    - LoRAManager: Manages LoRA application across model layers
    - apply_lora_to_family_band: Convenience function for Family Band

Features:
    - Merge/unmerge LoRA weights for inference/training
    - Save/load only LoRA weights (small checkpoints)
    - Enable/disable LoRA at runtime
    - Per-layer control of LoRA application

Reference: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"

Author: FamilyOS Team
Date: December 2025
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA Configuration
# =============================================================================


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapters.

    Attributes:
        rank: LoRA rank (r) - number of low-rank dimensions
        alpha: Scaling factor (alpha) - controls update magnitude
        dropout: Dropout probability on LoRA path
        target_modules: Which modules to apply LoRA to (e.g., q_proj, v_proj)
        layers: Which layer indices to apply LoRA to (0-indexed)

    Properties:
        scaling: Computed as alpha / rank for stable training

    Example:
        >>> config = LoRAConfig(rank=16, alpha=32, dropout=0.1)
        >>> config.scaling
        2.0
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=list)
    layers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set defaults for target_modules and layers if not provided."""
        if not self.target_modules:
            # Default: apply to attention Q, K, V and output projection
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if not self.layers:
            # Default: apply to Family Band (L23-28, 0-indexed as 22-27)
            self.layers = list(range(22, 28))

    @property
    def scaling(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# LoRA Linear Layer
# =============================================================================


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation) adapter.

    LoRA decomposes the weight update as a low-rank product:
        W' = W + ΔW = W + BA

    Where:
        - W: Original frozen weights [out_features, in_features]
        - B: Low-rank down projection [out_features, r]
        - A: Low-rank up projection [r, in_features]
        - r: LoRA rank (much smaller than in_features/out_features)

    Forward computation:
        y = Wx + (α/r) * BAx

    The scaling factor (α/r) maintains stable gradients across different ranks.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (default: 16)
        alpha: LoRA scaling factor (default: 32.0)
        dropout: Dropout on LoRA path (default: 0.1)
        bias: Whether base layer has bias (default: True)

    Attributes:
        linear: The base (frozen) linear layer
        lora_A: Down projection [in_features -> r]
        lora_B: Up projection [r -> out_features]
        lora_dropout: Dropout layer for regularization
        merged: Whether LoRA is merged into base weights
        enabled: Whether LoRA is active during forward

    Example:
        >>> lora_linear = LoRALinear(768, 768, rank=16)
        >>> x = torch.randn(2, 50, 768)
        >>> y = lora_linear(x)  # [2, 50, 768]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters: A projects down, B projects up
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        self._init_lora_weights()

        # State tracking
        self.merged = False
        self.enabled = True

    def _init_lora_weights(self) -> None:
        """
        Initialize LoRA weights for stable training start.

        A: Kaiming uniform (matches PyTorch default for Linear)
        B: Zero initialization (LoRA starts as identity, no initial contribution)
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            Output tensor [batch, seq, out_features]
        """
        # Base linear transformation
        result = self.linear(x)

        # Add LoRA contribution if enabled and not merged
        if self.enabled and not self.merged:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
            result = result + self.scaling * lora_out

        return result

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base linear layer.

        After merging: W' = W + (α/r) * B @ A

        This is useful for inference where you want the benefits of
        LoRA training without the runtime overhead of separate adapters.

        Warning: This is a one-way operation unless you save A/B first.
        """
        if self.merged:
            return

        with torch.no_grad():
            # Compute: W' = W + scaling * (B @ A)
            # B.weight: [out_features, rank]
            # A.weight: [rank, in_features]
            delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            self.linear.weight.add_(delta_w)

        self.merged = True
        logger.debug(f"Merged LoRA weights: rank={self.rank}, " f"scaling={self.scaling:.2f}")

    def unmerge_weights(self) -> None:
        """
        Unmerge LoRA weights from base linear layer.

        Reverses the merge operation: W = W' - (α/r) * B @ A

        This allows continuing LoRA training after inference.
        """
        if not self.merged:
            return

        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            self.linear.weight.sub_(delta_w)

        self.merged = False
        logger.debug("Unmerged LoRA weights")

    def get_lora_params(self) -> int:
        """Get number of LoRA parameters."""
        return self.lora_A.weight.numel() + self.lora_B.weight.numel()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ) -> "LoRALinear":
        """
        Create LoRALinear from an existing nn.Linear layer.

        This copies the weights from the original layer and freezes them,
        then adds fresh LoRA adapters for training.

        Args:
            linear: Source nn.Linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability

        Returns:
            New LoRALinear with copied (frozen) base weights

        Example:
            >>> orig = nn.Linear(768, 768)
            >>> lora = LoRALinear.from_linear(orig, rank=16)
            >>> assert not lora.linear.weight.requires_grad
        """
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )

        # Copy original weights
        lora_linear.linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.linear.bias.data = linear.bias.data.clone()

        # Freeze base weights (only LoRA trainable)
        lora_linear.linear.weight.requires_grad = False
        if lora_linear.linear.bias is not None:
            lora_linear.linear.bias.requires_grad = False

        return lora_linear

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.2f}, "
            f"merged={self.merged}, enabled={self.enabled}"
        )


# =============================================================================
# LoRA Manager
# =============================================================================


class LoRAManager:
    """
    Manages LoRA application to a model.

    This class handles:
        - Applying LoRA to specific layers and modules
        - Tracking all LoRA modules for parameter access
        - Merging/unmerging all LoRA weights
        - Saving/loading LoRA-only checkpoints
        - Enabling/disabling LoRA at runtime

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration

    Attributes:
        model: Reference to the model
        config: LoRA configuration
        lora_modules: Dict mapping full_name -> LoRALinear
        _original_modules: Dict mapping full_name -> original nn.Linear

    Example:
        >>> config = LoRAConfig(rank=16, layers=[22, 23, 24, 25, 26, 27])
        >>> manager = LoRAManager(model, config)
        >>> num_params = manager.apply_lora()
        >>> print(f"Added {num_params:,} LoRA parameters")
    """

    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.lora_modules: dict[str, LoRALinear] = {}
        self._original_modules: dict[str, nn.Linear] = {}
        self._applied = False

    def apply_lora(self) -> int:
        """
        Apply LoRA to configured layers and modules.

        Searches for matching linear layers in the target layers and
        replaces them with LoRALinear modules.

        Returns:
            Total number of LoRA parameters added

        Raises:
            RuntimeError: If LoRA has already been applied
        """
        if self._applied:
            raise RuntimeError("LoRA has already been applied to this model")

        lora_params = 0

        # Get encoder (may be model.encoder or model itself)
        encoder = getattr(self.model, "encoder", self.model)
        layers = getattr(encoder, "layers", None)

        if layers is None:
            logger.warning("No layers found in model - LoRA not applied")
            return 0

        for layer_idx in self.config.layers:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
                continue

            layer = layers[layer_idx]

            for module_name in self.config.target_modules:
                full_name = f"layer_{layer_idx}.{module_name}"
                module = self._get_module(layer, module_name)

                if module is None:
                    logger.debug(f"Module {module_name} not found in layer {layer_idx}")
                    continue

                if not isinstance(module, nn.Linear):
                    logger.debug(f"Module {full_name} is not nn.Linear, skipping")
                    continue

                # Store original for potential restoration
                self._original_modules[full_name] = module

                # Create LoRA version
                lora_module = LoRALinear.from_linear(
                    module,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )

                # Replace in model
                self._set_module(layer, module_name, lora_module)
                self.lora_modules[full_name] = lora_module

                # Count params (A + B weights)
                lora_params += lora_module.get_lora_params()

                logger.debug(f"Applied LoRA to {full_name}")

        self._applied = True

        logger.info(f"Applied LoRA to {len(self.lora_modules)} modules")
        logger.info(f"  LoRA rank: {self.config.rank}")
        logger.info(f"  LoRA alpha: {self.config.alpha}")
        logger.info(f"  LoRA params: {lora_params:,}")

        return lora_params

    def _get_module(self, parent: nn.Module, name: str) -> nn.Module | None:
        """
        Get a nested module by dot-separated name.

        Args:
            parent: Parent module to search from
            name: Dot-separated path (e.g., "attn.q_proj")

        Returns:
            The module if found, None otherwise
        """
        parts = name.split(".")
        module = parent

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def _set_module(self, parent: nn.Module, name: str, new_module: nn.Module) -> None:
        """
        Set a nested module by dot-separated name.

        Args:
            parent: Parent module
            name: Dot-separated path (e.g., "attn.q_proj")
            new_module: Module to set
        """
        parts = name.split(".")

        # Navigate to parent of target
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Set the final attribute
        setattr(parent, parts[-1], new_module)

    def get_lora_parameters(self) -> list[nn.Parameter]:
        """
        Get all LoRA parameters for optimizer.

        Returns only the trainable LoRA A and B matrices,
        not the frozen base weights.

        Returns:
            List of LoRA parameters

        Example:
            >>> params = manager.get_lora_parameters()
            >>> optimizer = torch.optim.AdamW(params, lr=1e-4)
        """
        params = []
        for lora_module in self.lora_modules.values():
            params.extend(lora_module.lora_A.parameters())
            params.extend(lora_module.lora_B.parameters())
        return params

    def get_lora_state_dict(self) -> dict[str, Any]:
        """
        Get state dict containing only LoRA weights.

        Returns:
            Dict with LoRA A and B weights for each module
        """
        state_dict = {}
        for name, lora_module in self.lora_modules.items():
            state_dict[f"{name}.lora_A.weight"] = lora_module.lora_A.weight
            state_dict[f"{name}.lora_B.weight"] = lora_module.lora_B.weight
        return state_dict

    def merge_all(self) -> None:
        """
        Merge all LoRA weights into base model.

        After merging, the model behaves as a standard model with
        LoRA contributions baked into the weights.
        """
        for name, lora_module in self.lora_modules.items():
            lora_module.merge_weights()
            logger.debug(f"Merged {name}")

        logger.info(f"Merged {len(self.lora_modules)} LoRA modules")

    def unmerge_all(self) -> None:
        """
        Unmerge all LoRA weights from base model.

        Reverses merge_all(), allowing continued LoRA training.
        """
        for name, lora_module in self.lora_modules.items():
            lora_module.unmerge_weights()
            logger.debug(f"Unmerged {name}")

        logger.info(f"Unmerged {len(self.lora_modules)} LoRA modules")

    def enable_lora(self, enable: bool = True) -> None:
        """
        Enable or disable LoRA contributions.

        When disabled, only the base layer is used (LoRA is bypassed).

        Args:
            enable: Whether to enable LoRA
        """
        for lora_module in self.lora_modules.values():
            lora_module.enabled = enable

        status = "enabled" if enable else "disabled"
        logger.info(f"LoRA {status} for {len(self.lora_modules)} modules")

    def save_lora_weights(self, path: str | Path) -> None:
        """
        Save only LoRA weights to a file.

        This creates a small checkpoint containing only the LoRA
        adapter weights, not the full model.

        Args:
            path: Path to save weights

        Example:
            >>> manager.save_lora_weights("lora_weights.pt")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lora_state = {
            "config": self.config.to_dict(),
            "weights": {},
        }

        for name, lora_module in self.lora_modules.items():
            lora_state["weights"][f"{name}.lora_A"] = lora_module.lora_A.state_dict()
            lora_state["weights"][f"{name}.lora_B"] = lora_module.lora_B.state_dict()

        torch.save(lora_state, path)
        logger.info(f"Saved LoRA weights to {path}")

    def load_lora_weights(self, path: str | Path) -> None:
        """
        Load LoRA weights from a file.

        The LoRA configuration must match the saved weights.

        Args:
            path: Path to load weights from

        Raises:
            RuntimeError: If LoRA has not been applied yet
        """
        if not self._applied:
            raise RuntimeError("Must apply_lora() before loading weights")

        path = Path(path)
        lora_state = torch.load(path, weights_only=False)

        weights = lora_state.get("weights", lora_state)  # Handle old format

        for name, lora_module in self.lora_modules.items():
            if f"{name}.lora_A" in weights:
                lora_module.lora_A.load_state_dict(weights[f"{name}.lora_A"])
            if f"{name}.lora_B" in weights:
                lora_module.lora_B.load_state_dict(weights[f"{name}.lora_B"])

        logger.info(f"Loaded LoRA weights from {path}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about LoRA configuration.

        Returns:
            Dict with LoRA statistics
        """
        total_lora_params = sum(m.get_lora_params() for m in self.lora_modules.values())

        return {
            "num_modules": len(self.lora_modules),
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "scaling": self.config.scaling,
            "total_lora_params": total_lora_params,
            "target_modules": self.config.target_modules,
            "layers": self.config.layers,
        }

    def print_summary(self) -> None:
        """Print a summary of LoRA configuration."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("LoRA Configuration Summary")
        print("=" * 60)
        print(f"  Modules with LoRA:  {stats['num_modules']}")
        print(f"  LoRA rank (r):      {stats['rank']}")
        print(f"  LoRA alpha:         {stats['alpha']}")
        print(f"  Scaling (α/r):      {stats['scaling']:.2f}")
        print(f"  Total LoRA params:  {stats['total_lora_params']:,}")
        print(f"  Target modules:     {stats['target_modules']}")
        print(f"  Target layers:      {stats['layers']}")
        print("=" * 60 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================


def apply_lora_to_family_band(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> LoRAManager:
    """
    Apply LoRA to Family Band (L23-28).

    This is a convenience function that creates a LoRAConfig targeting
    the Family Band layers with sensible defaults.

    Args:
        model: ModernBERTv3 model
        rank: LoRA rank (default: 16)
        alpha: LoRA scaling factor (default: 32.0)
        dropout: LoRA dropout (default: 0.1)
        target_modules: Modules to target (default: q_proj, k_proj, v_proj, o_proj)

    Returns:
        LoRAManager for controlling LoRA

    Example:
        >>> manager = apply_lora_to_family_band(model, rank=16)
        >>> manager.print_summary()
        >>> optimizer = torch.optim.AdamW(manager.get_lora_parameters(), lr=1e-4)
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        layers=list(range(22, 28)),  # L23-28 (0-indexed: 22-27)
    )

    manager = LoRAManager(model, config)
    manager.apply_lora()

    return manager


def get_lora_param_count(
    hidden_size: int = 768,
    rank: int = 16,
    num_layers: int = 6,
    num_modules_per_layer: int = 4,
) -> int:
    """
    Calculate expected LoRA parameter count.

    Args:
        hidden_size: Model hidden dimension
        rank: LoRA rank
        num_layers: Number of layers with LoRA
        num_modules_per_layer: Number of modules per layer (e.g., 4 for QKVO)

    Returns:
        Total LoRA parameters

    Example:
        >>> # Family Band: 6 layers, 4 modules each
        >>> params = get_lora_param_count(768, 16, 6, 4)
        >>> print(f"Expected: {params:,}")  # ~1.2M
    """
    # Each LoRA module has: A [hidden, rank] + B [rank, hidden]
    params_per_module = rank * hidden_size + rank * hidden_size
    return params_per_module * num_layers * num_modules_per_layer


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAManager",
    "apply_lora_to_family_band",
    "get_lora_param_count",
]

"""
Zipper Learning Rate Strategy for ModernBERT v3.

This module implements the Zipper Learning Rate strategy that provides smooth
LR transitions across the v2-v3 interface boundary. This prevents the "cliff
effect" at L22-L23 transition.

Layer Band Architecture:
    L1-18:  Foundation + Core (frozen, lr=0)
    L19-22: Feeder band (low lr, interface preparation)
    L23:    Interface layer (highest lr, maximum plasticity)
    L24-28: Family band (moderate lr, learning new tasks)

The Zipper strategy ensures:
    1. Smooth LR transition at v2-v3 interface
    2. Maximum plasticity at L23 (interface layer)
    3. Graduated LR decay in Family band
    4. Preserved v2 knowledge via frozen Foundation/Core
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Layer band boundaries (0-indexed)
FOUNDATION_END = 6  # L1-6 (indices 0-5)
CORE_END = 18  # L7-18 (indices 6-17)
FEEDER_END = 22  # L19-22 (indices 18-21)
INTERFACE_LAYER = 22  # L23 (index 22)
FAMILY_END = 28  # L24-28 (indices 23-27)

# Total layers in v3 architecture
V3_LAYER_COUNT = 28


# ============================================================================
# ZipperLRConfig
# ============================================================================


@dataclass
class ZipperLRConfig:
    """
    Configuration for Zipper Learning Rate strategy.

    The Zipper strategy creates a smooth LR transition across the
    v2-v3 interface boundary to prevent gradient discontinuities.

    Layer Layout:
        L1-18:  Foundation + Core (frozen, lr=0)
        L19-22: Feeder band (low lr, interface preparation)
        L23:    Interface layer (highest lr, maximum plasticity)
        L24-28: Family band (moderate lr, learning new tasks)

    LR Profile (Phase 0.5):
        L19: 1e-5 --+
        L20: 1e-5   | Feeder: gentle adaptation
        L21: 1e-5   |
        L22: 1e-5 --+
        L23: 5e-5 <-- Interface: highest plasticity
        L24: 4e-5 --+
        L25: 3.5e-5 | Family: decreasing toward output
        L26: 3e-5   |
        L27: 3e-5   |
        L28: 3e-5 --+

    Attributes:
        base_lr: Base learning rate (reference point)
        feeder_lr: Learning rate for Feeder band (L19-22)
        interface_lr: Learning rate for Interface layer (L23)
        family_lr: Learning rate for Family band (L24-28)
        family_graduated: Whether to decrease LR toward output
        family_decay: Decay factor for graduated family LR
        frozen_lr: Learning rate for frozen layers (L1-18)
        embeddings_lr: Learning rate for embeddings
        task_heads_lr: Learning rate for task heads
    """

    # Base learning rate (reference point)
    base_lr: float = 3e-5

    # Feeder band (L19-22) - uniform low LR
    feeder_lr: float = 1e-5

    # Interface layer (L23) - maximum plasticity
    interface_lr: float = 5e-5

    # Family band (L24-28) - can be uniform or graduated
    family_lr: float = 3e-5
    family_graduated: bool = True  # Decrease LR toward output
    family_decay: float = 0.9  # Each layer = prev * decay

    # Frozen layers (L1-18)
    frozen_lr: float = 0.0

    # Additional components
    embeddings_lr: float = 0.0  # Usually frozen
    task_heads_lr: float = 3e-5  # Same as family

    def get_layer_lr(self, layer_idx: int) -> float:
        """
        Get learning rate for a specific layer (0-indexed).

        Args:
            layer_idx: 0-indexed layer index

        Returns:
            Learning rate for the layer
        """
        if layer_idx < 18:
            # Foundation + Core: frozen
            return self.frozen_lr
        elif layer_idx < 22:
            # Feeder (L19-22, indices 18-21)
            return self.feeder_lr
        elif layer_idx == 22:
            # Interface (L23, index 22)
            return self.interface_lr
        else:
            # Family (L24-28, indices 23-27)
            if self.family_graduated:
                # Decay from interface
                steps_from_interface = layer_idx - 22
                return self.interface_lr * (self.family_decay**steps_from_interface)
            else:
                return self.family_lr

    def get_all_layer_lrs(self) -> dict[int, float]:
        """
        Get learning rates for all layers.

        Returns:
            Dictionary mapping layer index to learning rate
        """
        return {idx: self.get_layer_lr(idx) for idx in range(V3_LAYER_COUNT)}

    def get_trainable_layer_lrs(self) -> dict[int, float]:
        """
        Get learning rates for trainable layers only.

        Returns:
            Dictionary mapping layer index to learning rate (lr > 0)
        """
        return {idx: lr for idx, lr in self.get_all_layer_lrs().items() if lr > 0}

    def get_band_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get summary of LR configuration by band.

        Returns:
            Dictionary with band information
        """
        return {
            "foundation": {
                "layers": "L1-6",
                "indices": list(range(0, 6)),
                "lr": self.frozen_lr,
                "status": "frozen" if self.frozen_lr == 0 else "trainable",
            },
            "core": {
                "layers": "L7-18",
                "indices": list(range(6, 18)),
                "lr": self.frozen_lr,
                "status": "frozen" if self.frozen_lr == 0 else "trainable",
            },
            "feeder": {
                "layers": "L19-22",
                "indices": list(range(18, 22)),
                "lr": self.feeder_lr,
                "status": "trainable",
            },
            "interface": {
                "layers": "L23",
                "indices": [22],
                "lr": self.interface_lr,
                "status": "trainable (highest)",
            },
            "family": {
                "layers": "L24-28",
                "indices": list(range(23, 28)),
                "lr_start": self.get_layer_lr(23),
                "lr_end": self.get_layer_lr(27),
                "graduated": self.family_graduated,
                "status": "trainable",
            },
        }


# ============================================================================
# Preset Configurations
# ============================================================================

ZIPPER_PRESETS: dict[str, ZipperLRConfig] = {
    "phase_0.5_healing": ZipperLRConfig(
        base_lr=3e-5,
        feeder_lr=1e-5,
        interface_lr=5e-5,
        family_lr=3e-5,
        family_graduated=True,
        family_decay=0.85,
    ),
    "phase_1_multitask": ZipperLRConfig(
        base_lr=2e-5,
        feeder_lr=1e-5,
        interface_lr=4e-5,
        family_lr=2e-5,
        family_graduated=True,
        family_decay=0.9,
    ),
    "phase_2_polish": ZipperLRConfig(
        base_lr=1e-5,
        feeder_lr=5e-6,
        interface_lr=2e-5,
        family_lr=1e-5,
        family_graduated=False,
    ),
    "conservative": ZipperLRConfig(
        base_lr=1e-5,
        feeder_lr=5e-6,
        interface_lr=3e-5,
        family_lr=1e-5,
        family_graduated=False,
    ),
    "aggressive": ZipperLRConfig(
        base_lr=5e-5,
        feeder_lr=2e-5,
        interface_lr=1e-4,
        family_lr=5e-5,
        family_graduated=True,
        family_decay=0.8,
    ),
}


def get_zipper_preset(preset_name: str) -> ZipperLRConfig:
    """
    Get a Zipper LR preset configuration.

    Args:
        preset_name: Name of the preset

    Returns:
        ZipperLRConfig for the preset

    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in ZIPPER_PRESETS:
        available = ", ".join(ZIPPER_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    return copy.deepcopy(ZIPPER_PRESETS[preset_name])


def list_zipper_presets() -> list[str]:
    """
    List available Zipper LR presets.

    Returns:
        List of preset names
    """
    return list(ZIPPER_PRESETS.keys())


# ============================================================================
# ZipperLROptimizer
# ============================================================================


class ZipperLROptimizer:
    """
    Creates optimizer using Zipper Learning Rate strategy.

    The Zipper method ensures:
        1. Smooth LR transition at v2-v3 interface
        2. Maximum plasticity at L23 (interface layer)
        3. Graduated LR decay in Family band
        4. Preserved v2 knowledge via frozen Foundation/Core

    Attributes:
        model: The model to optimize
        config: Zipper LR configuration
        weight_decay: Weight decay for AdamW
        betas: Beta parameters for Adam
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        model: nn.Module,
        config: ZipperLRConfig,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Initialize ZipperLROptimizer.

        Args:
            model: ModernBERTv3 model
            config: Zipper LR configuration
            weight_decay: Weight decay for AdamW
            betas: Beta parameters for Adam
            eps: Epsilon for numerical stability
        """
        self.model = model
        self.config = config
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

        # Get encoder reference
        self.encoder = self._get_encoder()

    def _get_encoder(self) -> nn.Module:
        """Get the encoder module from the model."""
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        return self.model

    def _get_layers(self) -> nn.ModuleList | list[nn.Module]:
        """Get the layer list from the encoder."""
        if hasattr(self.encoder, "layers"):
            return self.encoder.layers
        elif hasattr(self.encoder, "layer"):
            return self.encoder.layer
        else:
            raise AttributeError(
                "Encoder has no 'layers' or 'layer' attribute. " "Cannot apply Zipper LR strategy."
            )

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with Zipper LR strategy.

        Returns:
            Configured AdamW optimizer
        """
        param_groups = self._build_zipper_param_groups()

        if not param_groups:
            raise ValueError(
                "No trainable parameters found. " "Ensure model has requires_grad=True parameters."
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )

        self._print_zipper_summary()
        return optimizer

    def _build_zipper_param_groups(self) -> list[dict[str, Any]]:
        """
        Build parameter groups with Zipper LR pattern.

        Returns:
            List of parameter group dictionaries
        """
        param_groups: list[dict[str, Any]] = []
        assigned_params: set[int] = set()

        layers = self._get_layers()

        # Per-layer groups for L19-28 (trainable layers)
        for layer_idx in range(18, min(28, len(layers))):
            layer = layers[layer_idx]
            lr = self.config.get_layer_lr(layer_idx)

            if lr <= 0:
                continue

            params = [
                p for p in layer.parameters() if p.requires_grad and id(p) not in assigned_params
            ]

            if params:
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "name": f"layer_{layer_idx + 1}",  # 1-indexed for display
                    }
                )
                for p in params:
                    assigned_params.add(id(p))

        # Embeddings (usually frozen)
        if hasattr(self.model, "embeddings"):
            emb_lr = self.config.embeddings_lr
            if emb_lr > 0:
                emb_params = [
                    p
                    for p in self.model.embeddings.parameters()
                    if p.requires_grad and id(p) not in assigned_params
                ]
                if emb_params:
                    param_groups.append(
                        {
                            "params": emb_params,
                            "lr": emb_lr,
                            "name": "embeddings",
                        }
                    )
                    for p in emb_params:
                        assigned_params.add(id(p))

        # Task heads
        if hasattr(self.model, "task_heads"):
            head_lr = self.config.task_heads_lr
            head_params = [
                p
                for p in self.model.task_heads.parameters()
                if p.requires_grad and id(p) not in assigned_params
            ]
            if head_params:
                param_groups.append(
                    {
                        "params": head_params,
                        "lr": head_lr,
                        "name": "task_heads",
                    }
                )
                for p in head_params:
                    assigned_params.add(id(p))

        # Any remaining trainable parameters
        remaining = [
            p for p in self.model.parameters() if p.requires_grad and id(p) not in assigned_params
        ]
        if remaining:
            param_groups.append(
                {
                    "params": remaining,
                    "lr": self.config.base_lr,
                    "name": "other",
                }
            )

        return param_groups

    def _print_zipper_summary(self) -> None:
        """Print Zipper LR visualization."""
        print("\n" + "=" * 60)
        print("Zipper Learning Rate Strategy")
        print("=" * 60)

        # ASCII visualization
        print("\nLR Profile:")
        print("  Layer | LR        | Band")
        print("  ------+-----------+---------")

        for layer_idx in range(V3_LAYER_COUNT):
            lr = self.config.get_layer_lr(layer_idx)
            layer_num = layer_idx + 1

            # Band name
            if layer_idx < 6:
                band = "Foundation"
            elif layer_idx < 18:
                band = "Core"
            elif layer_idx < 22:
                band = "Feeder"
            elif layer_idx == 22:
                band = "Interface *"
            else:
                band = "Family"

            # LR bar
            if lr > 0:
                bar_len = min(20, int(lr * 400000))
                bar = "#" * bar_len
                print(f"  L{layer_num:02d}   | {lr:.1e} | {band:12} {bar}")
            else:
                print(f"  L{layer_num:02d}   | frozen    | {band}")

        print("=" * 60 + "\n")

    def get_lr_dict(self) -> dict[str, float]:
        """
        Get dictionary of layer->LR mappings.

        Returns:
            Dictionary with layer names and learning rates
        """
        lr_dict: dict[str, float] = {}
        for layer_idx in range(V3_LAYER_COUNT):
            lr_dict[f"layer_{layer_idx + 1}"] = self.config.get_layer_lr(layer_idx)
        lr_dict["embeddings"] = self.config.embeddings_lr
        lr_dict["task_heads"] = self.config.task_heads_lr
        return lr_dict

    def get_param_group_count(self) -> int:
        """
        Get the number of parameter groups that would be created.

        Returns:
            Number of parameter groups
        """
        param_groups = self._build_zipper_param_groups()
        return len(param_groups)

    def get_trainable_param_count(self) -> int:
        """
        Get total count of trainable parameters.

        Returns:
            Total trainable parameter count
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================================
# Factory Function
# ============================================================================


def create_zipper_optimizer(
    model: nn.Module,
    preset: str = "phase_0.5_healing",
    weight_decay: float = 0.01,
    **overrides: Any,
) -> torch.optim.Optimizer:
    """
    Create optimizer with Zipper LR strategy.

    Args:
        model: ModernBERTv3 model
        preset: Preset name from ZIPPER_PRESETS
        weight_decay: Weight decay for AdamW
        **overrides: Override specific config values

    Returns:
        Configured AdamW optimizer

    Example:
        >>> optimizer = create_zipper_optimizer(
        ...     model,
        ...     preset="phase_0.5_healing",
        ...     interface_lr=6e-5,  # Override interface LR
        ... )
    """
    # Get preset config (or default)
    if preset in ZIPPER_PRESETS:
        config = copy.deepcopy(ZIPPER_PRESETS[preset])
    else:
        logger.warning(f"Unknown preset '{preset}', using 'phase_0.5_healing'")
        config = copy.deepcopy(ZIPPER_PRESETS["phase_0.5_healing"])

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key '{key}', ignoring")

    zipper = ZipperLROptimizer(model, config, weight_decay)
    return zipper.create_optimizer()


# ============================================================================
# Utility Functions
# ============================================================================


def print_zipper_lr_profile(config: ZipperLRConfig) -> None:
    """
    Print the Zipper LR profile for a configuration.

    Args:
        config: Zipper LR configuration
    """
    print("\n" + "=" * 60)
    print("Zipper Learning Rate Profile")
    print("=" * 60)

    for layer_idx in range(V3_LAYER_COUNT):
        lr = config.get_layer_lr(layer_idx)
        layer_num = layer_idx + 1

        if layer_idx < 6:
            band = "Foundation"
        elif layer_idx < 18:
            band = "Core"
        elif layer_idx < 22:
            band = "Feeder"
        elif layer_idx == 22:
            band = "Interface *"
        else:
            band = "Family"

        if lr > 0:
            print(f"  L{layer_num:02d}: {lr:.2e} ({band})")
        else:
            print(f"  L{layer_num:02d}: frozen ({band})")

    print("=" * 60 + "\n")


def compare_zipper_presets() -> dict[str, dict[str, float]]:
    """
    Compare learning rates across all Zipper presets.

    Returns:
        Dictionary mapping preset names to layer LR dictionaries
    """
    comparison: dict[str, dict[str, float]] = {}
    for name, config in ZIPPER_PRESETS.items():
        comparison[name] = {
            "feeder_lr": config.feeder_lr,
            "interface_lr": config.interface_lr,
            "family_lr_start": config.get_layer_lr(23),
            "family_lr_end": config.get_layer_lr(27),
            "family_graduated": config.family_graduated,
        }
    return comparison


def validate_zipper_config(config: ZipperLRConfig) -> list[str]:
    """
    Validate a Zipper LR configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings: list[str] = []

    # Check that interface LR is highest
    interface_lr = config.get_layer_lr(22)
    feeder_lr = config.feeder_lr
    family_start_lr = config.get_layer_lr(23)

    if interface_lr <= feeder_lr:
        warnings.append(
            f"Interface LR ({interface_lr:.2e}) should be higher than "
            f"Feeder LR ({feeder_lr:.2e})"
        )

    if interface_lr < family_start_lr:
        warnings.append(
            f"Interface LR ({interface_lr:.2e}) should be >= "
            f"Family start LR ({family_start_lr:.2e})"
        )

    # Check decay factor
    if config.family_graduated:
        if config.family_decay <= 0 or config.family_decay > 1:
            warnings.append(f"family_decay ({config.family_decay}) should be in (0, 1]")

    # Check non-negative LRs
    for idx in range(V3_LAYER_COUNT):
        lr = config.get_layer_lr(idx)
        if lr < 0:
            warnings.append(f"Layer {idx + 1} has negative LR: {lr}")

    return warnings


# ============================================================================
# Quick Reference
# ============================================================================

ZIPPER_LR_QUICK_REF = """
+----------------------------------------------------------+
|            Zipper Learning Rate Quick Reference          |
+----------------------------------------------------------+
| Layer    | Phase 0.5  | Phase 1    | Phase 2             |
|----------+------------+------------+---------------------|
| L1-18    | 0 (frozen) | 0 (frozen) | 1e-6 (low)          |
| L19-22   | 1e-5       | 1e-5       | 5e-6                |
| L23 *    | 5e-5       | 4e-5       | 2e-5                |
| L24-28   | 3e-5->     | 2e-5->     | 1e-5                |
+----------------------------------------------------------+

* = Interface layer (maximum plasticity)
-> = Graduated decay toward output layer
"""

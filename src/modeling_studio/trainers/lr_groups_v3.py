"""
Layer-Group Learning Rates for ModernBERT v3.

This module implements layer-group specific learning rates for optimal training.
Different layer bands require different learning rates based on their role and
whether they're transferred vs cloned.

Layer Band Architecture:
    - Foundation (L1-6): Very low or frozen - preserve v2 knowledge
    - Core (L7-18): Very low or frozen - preserve v2 knowledge
    - Semantic (L19-22): Low LR - gentle refinement of interface
    - Interface (L23): Highest LR - needs most adaptation
    - Family (L24-28): Moderate LR - learning new capabilities

Training Phases:
    Phase 0.5 (Healing): Foundation/Core frozen, high interface LR
    Phase 1 (Multi-task): Foundation/Core frozen, moderate LRs
    Phase 2 (Fine-tune): All layers trainable with low LRs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerGroupLRConfig:
    """
    Configuration for layer-group learning rates.

    Rationale:
        - Foundation/Core (L1-18): Very low or frozen - preserve v2 knowledge
        - Semantic (L19-22): Low LR - gentle refinement of interface
        - Interface (L23): Highest LR - needs most adaptation
        - Family (L24-28): Moderate LR - learning new capabilities

    Attributes:
        base_lr: Base learning rate (reference point for multipliers)
        foundation_mult: Multiplier for Foundation band (L1-6)
        core_mult: Multiplier for Core band (L7-18)
        semantic_mult: Multiplier for Semantic band (L19-22)
        interface_mult: Multiplier for Interface layer (L23)
        family_mult: Multiplier for Family band (L24-28)
        embeddings_mult: Multiplier for embedding layer
        task_heads_mult: Multiplier for task heads
        hub_tokens_mult: Multiplier for hub token embeddings
        warmup_ratio: Ratio of total steps for warmup (0.1 = 10%)
        min_lr_ratio: Minimum LR as ratio of peak (0.01 = 1%)
    """

    # Base learning rate
    base_lr: float = 3e-5

    # Layer band multipliers (relative to base_lr)
    foundation_mult: float = 0.0  # L1-6: Frozen or no training
    core_mult: float = 0.0  # L7-18: Frozen or no training
    semantic_mult: float = 0.33  # L19-22: 1/3 of base LR
    interface_mult: float = 1.67  # L23: 5/3 of base LR (highest)
    family_mult: float = 1.0  # L24-28: Base LR

    # Component-specific multipliers
    embeddings_mult: float = 0.1  # Usually frozen or very low
    task_heads_mult: float = 1.0  # Same as family band
    hub_tokens_mult: float = 0.5  # Careful with hub token gradients

    # Warmup settings
    warmup_ratio: float = 0.1  # 10% warmup
    min_lr_ratio: float = 0.01  # End at 1% of peak

    def get_layer_lr(self, layer_idx: int) -> float:
        """
        Get learning rate for a specific layer.

        Args:
            layer_idx: 0-indexed layer index

        Returns:
            Learning rate for the layer
        """
        if layer_idx < 6:  # Foundation (L1-6)
            return self.base_lr * self.foundation_mult
        elif layer_idx < 18:  # Core (L7-18)
            return self.base_lr * self.core_mult
        elif layer_idx < 22:  # Semantic (L19-22)
            return self.base_lr * self.semantic_mult
        elif layer_idx == 22:  # Interface (L23, 0-indexed)
            return self.base_lr * self.interface_mult
        else:  # Family (L24-28)
            return self.base_lr * self.family_mult

    def get_component_lr(self, component: str) -> float:
        """
        Get learning rate for a specific component.

        Args:
            component: Component name (embeddings, task_heads, hub_tokens)

        Returns:
            Learning rate for the component
        """
        if component == "embeddings":
            return self.base_lr * self.embeddings_mult
        elif component == "task_heads":
            return self.base_lr * self.task_heads_mult
        elif component == "hub_tokens":
            return self.base_lr * self.hub_tokens_mult
        else:
            return self.base_lr

    def get_band_lr(self, band: str) -> float:
        """
        Get learning rate for a layer band.

        Args:
            band: Band name (foundation, core, semantic, interface, family)

        Returns:
            Learning rate for the band
        """
        band_mults = {
            "foundation": self.foundation_mult,
            "core": self.core_mult,
            "semantic": self.semantic_mult,
            "interface": self.interface_mult,
            "family": self.family_mult,
        }
        mult = band_mults.get(band.lower(), 1.0)
        return self.base_lr * mult

    def get_warmup_steps(self, total_steps: int) -> int:
        """
        Get number of warmup steps.

        Args:
            total_steps: Total training steps

        Returns:
            Number of warmup steps
        """
        return int(total_steps * self.warmup_ratio)

    def get_min_lr(self) -> float:
        """
        Get minimum learning rate.

        Returns:
            Minimum learning rate
        """
        return self.base_lr * self.min_lr_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "base_lr": self.base_lr,
            "foundation_mult": self.foundation_mult,
            "core_mult": self.core_mult,
            "semantic_mult": self.semantic_mult,
            "interface_mult": self.interface_mult,
            "family_mult": self.family_mult,
            "embeddings_mult": self.embeddings_mult,
            "task_heads_mult": self.task_heads_mult,
            "hub_tokens_mult": self.hub_tokens_mult,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayerGroupLRConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Preset configurations for different phases
PHASE_LR_CONFIGS: dict[str, LayerGroupLRConfig] = {
    "phase_0.5": LayerGroupLRConfig(
        base_lr=3e-5,
        foundation_mult=0.0,
        core_mult=0.0,
        semantic_mult=0.33,
        interface_mult=1.67,
        family_mult=1.0,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
    ),
    "phase_1": LayerGroupLRConfig(
        base_lr=2e-5,
        foundation_mult=0.0,
        core_mult=0.0,
        semantic_mult=0.5,
        interface_mult=1.5,
        family_mult=1.0,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
    ),
    "phase_2": LayerGroupLRConfig(
        base_lr=1e-5,
        foundation_mult=0.1,
        core_mult=0.2,
        semantic_mult=0.5,
        interface_mult=1.0,
        family_mult=1.0,
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
    ),
}


# Layer band definitions (for reference and validation)
LAYER_BAND_RANGES: dict[str, range] = {
    "foundation": range(0, 6),  # L1-6
    "core": range(6, 18),  # L7-18
    "semantic": range(18, 22),  # L19-22
    "interface": range(22, 23),  # L23 only
    "family": range(23, 28),  # L24-28
}


def get_band_for_layer(layer_idx: int) -> str:
    """
    Get the band name for a layer index.

    Args:
        layer_idx: 0-indexed layer index

    Returns:
        Band name
    """
    for band_name, layer_range in LAYER_BAND_RANGES.items():
        if layer_idx in layer_range:
            return band_name
    return "unknown"


class LayerGroupOptimizer:
    """
    Creates optimizer with layer-group specific learning rates.

    This class builds parameter groups for each layer band with appropriate
    learning rates, enabling fine-grained control over training dynamics.

    Usage:
        config = LayerGroupLRConfig(base_lr=3e-5)
        group_optimizer = LayerGroupOptimizer(model, config)
        optimizer = group_optimizer.create_optimizer()

    Attributes:
        model: The model to create optimizer for
        config: LayerGroupLRConfig with LR settings
        weight_decay: Weight decay for AdamW optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        config: LayerGroupLRConfig,
        weight_decay: float = 0.01,
    ):
        """
        Initialize LayerGroupOptimizer.

        Args:
            model: ModernBERTv3 model (or any model with encoder.layers)
            config: Learning rate configuration
            weight_decay: Weight decay for AdamW
        """
        self.model = model
        self.config = config
        self.weight_decay = weight_decay

        # Get encoder reference
        self.encoder = self._get_encoder()
        self._assigned_params: set[int] = set()

    def _get_encoder(self) -> nn.Module:
        """Get encoder module from model."""
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            return self.model.model.encoder
        else:
            return self.model

    def _has_layers(self) -> bool:
        """Check if encoder has layers attribute."""
        return hasattr(self.encoder, "layers") and len(self.encoder.layers) > 0

    def _get_num_layers(self) -> int:
        """Get number of layers in encoder."""
        if self._has_layers():
            return len(self.encoder.layers)
        return 0

    def create_optimizer(
        self,
        optimizer_class: type = None,
        **optimizer_kwargs: Any,
    ) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with layer-group LRs.

        Args:
            optimizer_class: Optimizer class to use (default: AdamW)
            **optimizer_kwargs: Additional kwargs for optimizer

        Returns:
            Configured optimizer
        """
        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW

        param_groups = self._build_param_groups()

        # Merge default kwargs
        kwargs = {"weight_decay": self.weight_decay}
        kwargs.update(optimizer_kwargs)

        optimizer = optimizer_class(param_groups, **kwargs)

        self._log_param_groups(param_groups)
        return optimizer

    def _build_param_groups(self) -> list[dict[str, Any]]:
        """
        Build parameter groups with appropriate LRs.

        Returns:
            List of parameter group dictionaries
        """
        param_groups: list[dict[str, Any]] = []
        self._assigned_params = set()

        # Layer groups
        layer_groups = {
            "foundation": (range(0, 6), self.config.foundation_mult),
            "core": (range(6, 18), self.config.core_mult),
            "semantic": (range(18, 22), self.config.semantic_mult),
            "interface": ([22], self.config.interface_mult),
            "family": (range(23, 28), self.config.family_mult),
        }

        # Add layer groups
        if self._has_layers():
            num_layers = self._get_num_layers()
            for group_name, (layer_indices, mult) in layer_groups.items():
                lr = self.config.base_lr * mult

                if lr == 0:
                    # Skip frozen groups (they're not trainable anyway)
                    continue

                params = []
                for layer_idx in layer_indices:
                    if layer_idx >= num_layers:
                        continue
                    layer = self.encoder.layers[layer_idx]
                    for p in layer.parameters():
                        if p.requires_grad and id(p) not in self._assigned_params:
                            params.append(p)
                            self._assigned_params.add(id(p))

                if params:
                    param_groups.append(
                        {
                            "params": params,
                            "lr": lr,
                            "name": group_name,
                        }
                    )

        # Embeddings
        param_groups.extend(self._get_embedding_groups())

        # Task heads
        param_groups.extend(self._get_task_head_groups())

        # Any remaining parameters (e.g., poolers, classifiers)
        param_groups.extend(self._get_remaining_groups())

        return param_groups

    def _get_embedding_groups(self) -> list[dict[str, Any]]:
        """Get parameter groups for embeddings."""
        groups = []

        # Check various embedding attribute names
        embedding_attrs = ["embeddings", "embed_tokens", "word_embeddings"]
        for attr in embedding_attrs:
            if hasattr(self.model, attr):
                emb_module = getattr(self.model, attr)
                emb_params = [
                    p
                    for p in emb_module.parameters()
                    if p.requires_grad and id(p) not in self._assigned_params
                ]
                if emb_params:
                    groups.append(
                        {
                            "params": emb_params,
                            "lr": self.config.base_lr * self.config.embeddings_mult,
                            "name": "embeddings",
                        }
                    )
                    for p in emb_params:
                        self._assigned_params.add(id(p))
                break

        return groups

    def _get_task_head_groups(self) -> list[dict[str, Any]]:
        """Get parameter groups for task heads."""
        groups = []

        # Check various head attribute names
        head_attrs = ["task_heads", "heads", "classifier", "classifiers"]
        for attr in head_attrs:
            if hasattr(self.model, attr):
                head_module = getattr(self.model, attr)
                head_params = [
                    p
                    for p in head_module.parameters()
                    if p.requires_grad and id(p) not in self._assigned_params
                ]
                if head_params:
                    groups.append(
                        {
                            "params": head_params,
                            "lr": self.config.base_lr * self.config.task_heads_mult,
                            "name": "task_heads",
                        }
                    )
                    for p in head_params:
                        self._assigned_params.add(id(p))
                break

        return groups

    def _get_remaining_groups(self) -> list[dict[str, Any]]:
        """Get parameter groups for remaining parameters."""
        groups = []

        remaining_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in self._assigned_params
        ]
        if remaining_params:
            groups.append(
                {
                    "params": remaining_params,
                    "lr": self.config.base_lr,
                    "name": "other",
                }
            )
            for p in remaining_params:
                self._assigned_params.add(id(p))

        return groups

    def get_param_groups(self) -> list[dict[str, Any]]:
        """
        Get parameter groups without creating optimizer.

        Returns:
            List of parameter group dictionaries
        """
        return self._build_param_groups()

    def _log_param_groups(self, param_groups: list[dict[str, Any]]) -> None:
        """Log parameter group configuration."""
        print("\n" + "=" * 60)
        print("Layer Group Learning Rates")
        print("=" * 60)

        total_params = 0
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            total_params += n_params
            print(f"  {group['name']:15} | lr={group['lr']:.2e} | params={n_params:,}")

        print("-" * 60)
        print(f"  {'TOTAL':15} | base_lr={self.config.base_lr:.2e} | params={total_params:,}")
        print("=" * 60 + "\n")

        logger.info(f"Created {len(param_groups)} parameter groups with {total_params:,} params")


def create_layer_group_optimizer(
    model: nn.Module,
    phase: str = "phase_0.5",
    base_lr: float | None = None,
    weight_decay: float = 0.01,
    optimizer_class: type | None = None,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create optimizer with phase-appropriate layer-group LRs.

    This is a convenience function that creates a LayerGroupOptimizer
    with phase-specific configuration and returns the configured optimizer.

    Args:
        model: ModernBERTv3 model
        phase: Training phase name (phase_0.5, phase_1, phase_2)
        base_lr: Override base learning rate (optional)
        weight_decay: Weight decay for AdamW
        optimizer_class: Optimizer class to use (default: AdamW)
        **optimizer_kwargs: Additional kwargs for optimizer

    Returns:
        Configured optimizer

    Example:
        optimizer = create_layer_group_optimizer(
            model,
            phase="phase_0.5",
            base_lr=3e-5,
            weight_decay=0.01,
        )
    """
    # Get phase config (copy to avoid modifying preset)
    if phase in PHASE_LR_CONFIGS:
        config = LayerGroupLRConfig(**PHASE_LR_CONFIGS[phase].to_dict())
    else:
        logger.warning(f"Unknown phase '{phase}', using phase_0.5 config")
        config = LayerGroupLRConfig(**PHASE_LR_CONFIGS["phase_0.5"].to_dict())

    if base_lr is not None:
        config.base_lr = base_lr

    group_optimizer = LayerGroupOptimizer(model, config, weight_decay)
    return group_optimizer.create_optimizer(optimizer_class, **optimizer_kwargs)


def get_phase_config(phase: str) -> LayerGroupLRConfig:
    """
    Get the LR config for a training phase.

    Args:
        phase: Training phase name

    Returns:
        LayerGroupLRConfig for the phase
    """
    if phase in PHASE_LR_CONFIGS:
        return LayerGroupLRConfig(**PHASE_LR_CONFIGS[phase].to_dict())
    else:
        logger.warning(f"Unknown phase '{phase}', using phase_0.5 config")
        return LayerGroupLRConfig(**PHASE_LR_CONFIGS["phase_0.5"].to_dict())


def print_lr_summary(config: LayerGroupLRConfig) -> None:
    """
    Print a summary of learning rates for all components.

    Args:
        config: LayerGroupLRConfig to summarize
    """
    print("\n" + "=" * 60)
    print("Learning Rate Summary")
    print("=" * 60)
    print(f"Base LR: {config.base_lr:.2e}")
    print("-" * 60)
    print("Layer Bands:")
    print(
        f"  Foundation (L1-6):   {config.get_band_lr('foundation'):.2e} ({config.foundation_mult:.2f}x)"
    )
    print(f"  Core (L7-18):        {config.get_band_lr('core'):.2e} ({config.core_mult:.2f}x)")
    print(
        f"  Semantic (L19-22):   {config.get_band_lr('semantic'):.2e} ({config.semantic_mult:.2f}x)"
    )
    print(
        f"  Interface (L23):     {config.get_band_lr('interface'):.2e} ({config.interface_mult:.2f}x)"
    )
    print(f"  Family (L24-28):     {config.get_band_lr('family'):.2e} ({config.family_mult:.2f}x)")
    print("-" * 60)
    print("Components:")
    print(
        f"  Embeddings:          {config.get_component_lr('embeddings'):.2e} ({config.embeddings_mult:.2f}x)"
    )
    print(
        f"  Task Heads:          {config.get_component_lr('task_heads'):.2e} ({config.task_heads_mult:.2f}x)"
    )
    print(
        f"  Hub Tokens:          {config.get_component_lr('hub_tokens'):.2e} ({config.hub_tokens_mult:.2f}x)"
    )
    print("-" * 60)
    print("Scheduler:")
    print(f"  Warmup Ratio:        {config.warmup_ratio:.0%}")
    print(f"  Min LR Ratio:        {config.min_lr_ratio:.0%}")
    print(f"  Min LR:              {config.get_min_lr():.2e}")
    print("=" * 60 + "\n")

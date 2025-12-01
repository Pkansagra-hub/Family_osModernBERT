"""
Optimizer Utilities

This module provides optimizer creation utilities with support for:
    - Head-wise learning rates (encoder vs heads)
    - Layer-wise learning rate decay
    - Parameter group creation

Head-wise Learning Rates:
    - Encoder (pretrained): Lower LR (2e-5) - careful adaptation
    - Classification heads: Higher LR (1e-4) - faster learning
    - Token classification heads: Medium LR (5e-5) - finer updates

Expected Gains:
    - +1-3 pt improvement from proper LR separation
    - Prevents heads from overfitting while encoder adapts

Usage:
    from modeling_studio.trainers.optimizer import (
        create_optimizer_with_head_lr,
        create_param_groups,
    )

    optimizer = create_optimizer_with_head_lr(
        model,
        encoder_lr=2e-5,
        head_lr=1e-4,
        token_head_lr=5e-5,
    )
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_param_groups(
    model: nn.Module,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    token_head_lr: float = 5e-5,
    weight_decay: float = 0.01,
    no_decay_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Create parameter groups with different learning rates.

    Groups:
        1. Encoder parameters (pretrained backbone) - lowest LR
        2. Token classification heads (NER, temporal) - medium LR
        3. Other heads (classification, embedding) - highest LR

    Args:
        model: The model to create parameter groups for.
        encoder_lr: Learning rate for encoder/backbone. Default: 2e-5
        head_lr: Learning rate for classification heads. Default: 1e-4
        token_head_lr: Learning rate for token classification heads. Default: 5e-5
        weight_decay: Weight decay for parameters. Default: 0.01
        no_decay_patterns: Patterns for parameters that should not have weight decay.
            Default: ["bias", "LayerNorm", "layer_norm"]

    Returns:
        List of parameter group dicts for optimizer.

    Example:
        >>> param_groups = create_param_groups(model)
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    if no_decay_patterns is None:
        no_decay_patterns = ["bias", "LayerNorm", "layer_norm", "layernorm"]

    # Patterns to identify different parameter types
    encoder_patterns = [
        r"^encoder\.",
        r"^model\.",
        r"^backbone\.",
        r"^embeddings\.",
        r"^bert\.",
        r"^roberta\.",
        r"^modernbert\.",
    ]

    token_head_patterns = [
        r"ner_",
        r"token_",
        r"temporal_",
        r"_ner",
        r"_token",
        r"_temporal",
    ]

    def is_encoder_param(name: str) -> bool:
        return any(re.match(pattern, name) for pattern in encoder_patterns)

    def is_token_head_param(name: str) -> bool:
        return any(re.search(pattern, name) for pattern in token_head_patterns)

    def has_no_decay(name: str) -> bool:
        return any(pattern in name for pattern in no_decay_patterns)

    # Collect parameters into groups
    encoder_decay = []
    encoder_no_decay = []
    token_head_decay = []
    token_head_no_decay = []
    other_head_decay = []
    other_head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_encoder_param(name):
            if has_no_decay(name):
                encoder_no_decay.append(param)
            else:
                encoder_decay.append(param)
        elif is_token_head_param(name):
            if has_no_decay(name):
                token_head_no_decay.append(param)
            else:
                token_head_decay.append(param)
        else:
            if has_no_decay(name):
                other_head_no_decay.append(param)
            else:
                other_head_decay.append(param)

    param_groups = [
        {
            "params": encoder_decay,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder_decay",
        },
        {
            "params": encoder_no_decay,
            "lr": encoder_lr,
            "weight_decay": 0.0,
            "name": "encoder_no_decay",
        },
        {
            "params": token_head_decay,
            "lr": token_head_lr,
            "weight_decay": weight_decay,
            "name": "token_head_decay",
        },
        {
            "params": token_head_no_decay,
            "lr": token_head_lr,
            "weight_decay": 0.0,
            "name": "token_head_no_decay",
        },
        {
            "params": other_head_decay,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "other_head_decay",
        },
        {
            "params": other_head_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
            "name": "other_head_no_decay",
        },
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    # Log summary
    for group in param_groups:
        logger.info(
            f"Param group '{group['name']}': "
            f"{len(group['params'])} params, "
            f"lr={group['lr']}, wd={group['weight_decay']}"
        )

    return param_groups


def create_optimizer_with_head_lr(
    model: nn.Module,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    token_head_lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with different learning rates for encoder and heads.

    This prevents heads from overfitting while allowing the encoder to adapt carefully.

    Args:
        model: The model to optimize.
        encoder_lr: Learning rate for encoder. Default: 2e-5
        head_lr: Learning rate for classification heads. Default: 1e-4
        token_head_lr: Learning rate for token heads. Default: 5e-5
        weight_decay: Weight decay. Default: 0.01
        betas: AdamW betas. Default: (0.9, 0.999)
        eps: AdamW epsilon. Default: 1e-8

    Returns:
        Configured AdamW optimizer.

    Example:
        >>> optimizer = create_optimizer_with_head_lr(
        ...     model,
        ...     encoder_lr=2e-5,
        ...     head_lr=1e-4,
        ... )
    """
    param_groups = create_param_groups(
        model,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        token_head_lr=token_head_lr,
        weight_decay=weight_decay,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=betas,
        eps=eps,
    )

    logger.info(
        f"Created AdamW optimizer: "
        f"encoder_lr={encoder_lr}, head_lr={head_lr}, token_head_lr={token_head_lr}"
    )

    return optimizer


def create_optimizer_with_layer_decay(
    model: nn.Module,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    token_head_lr: float = 5e-5,
    layer_decay: float = 0.95,
    num_layers: int = 22,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with head-wise LR AND layer-wise decay.

    Combines:
    - Head-wise LR: Different LRs for encoder vs classification/token heads
    - Layer-wise decay: Lower layers get lower LRs (layer_decay^(num_layers - layer_idx))

    Args:
        model: The model to optimize.
        encoder_lr: Base learning rate for encoder (top layer). Default: 2e-5
        head_lr: Learning rate for classification heads. Default: 1e-4
        token_head_lr: Learning rate for token heads. Default: 5e-5
        layer_decay: Decay factor per layer. Default: 0.95
        num_layers: Number of transformer layers. Default: 22 (ModernBERT-base)
        weight_decay: Weight decay. Default: 0.01
        betas: AdamW betas. Default: (0.9, 0.999)
        eps: AdamW epsilon. Default: 1e-8

    Returns:
        Configured AdamW optimizer with layer-wise decay.

    Example:
        >>> # Layer 0 (bottom): lr = 2e-5 * 0.95^22 ≈ 6.5e-6
        >>> # Layer 22 (top): lr = 2e-5
        >>> # Heads: lr = 1e-4 (no decay applied)
        >>> optimizer = create_optimizer_with_layer_decay(model, layer_decay=0.95)
    """
    no_decay_patterns = ["bias", "LayerNorm", "layer_norm", "layernorm"]

    # Patterns to identify heads (get head_lr, no layer decay)
    token_head_patterns = [r"ner_", r"token_", r"temporal_", r"_ner", r"_token", r"_temporal"]
    head_patterns = [r"head", r"classifier", r"pooler", r"embedding_head"]

    # Pattern to extract layer number from encoder
    layer_pattern = re.compile(r"layer\.(\d+)\.")

    def is_token_head(name: str) -> bool:
        return any(re.search(p, name) for p in token_head_patterns)

    def is_other_head(name: str) -> bool:
        return any(re.search(p, name) for p in head_patterns) and not is_token_head(name)

    def has_no_decay(name: str) -> bool:
        return any(pattern in name for pattern in no_decay_patterns)

    def get_layer_lr(name: str, base_lr: float) -> float:
        """Get LR with layer-wise decay applied."""
        match = layer_pattern.search(name)
        if match:
            layer_num = int(match.group(1))
            return base_lr * (layer_decay ** (num_layers - layer_num))
        # Non-layer params (embeddings) get lowest LR
        return base_lr * (layer_decay**num_layers)

    # Group parameters
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine LR and weight decay
        if is_token_head(name):
            lr = token_head_lr  # No layer decay for heads
            group_prefix = "token_head"
        elif is_other_head(name):
            lr = head_lr  # No layer decay for heads
            group_prefix = "other_head"
        else:
            # Encoder params - apply layer-wise decay
            lr = get_layer_lr(name, encoder_lr)
            match = layer_pattern.search(name)
            if match:
                group_prefix = f"encoder_layer_{match.group(1)}"
            else:
                group_prefix = "encoder_embedding"

        wd = 0.0 if has_no_decay(name) else weight_decay
        decay_suffix = "no_decay" if wd == 0.0 else "decay"

        # Create unique group key
        group_key = (group_prefix, decay_suffix, lr)

        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
                "name": f"{group_prefix}_{decay_suffix}",
            }

        param_groups[group_key]["params"].append(param)

    groups = list(param_groups.values())

    # Log summary (condensed)
    encoder_groups = [g for g in groups if "encoder" in g["name"]]
    head_groups = [g for g in groups if "head" in g["name"]]

    if encoder_groups:
        min_lr = min(g["lr"] for g in encoder_groups)
        max_lr = max(g["lr"] for g in encoder_groups)
        logger.info(
            f"Layer-wise LR decay: {len(encoder_groups)} encoder groups, "
            f"lr range: {min_lr:.2e} - {max_lr:.2e}"
        )

    for g in head_groups:
        logger.info(f"Param group '{g['name']}': {len(g['params'])} params, lr={g['lr']}")

    optimizer = torch.optim.AdamW(groups, betas=betas, eps=eps)

    logger.info(
        f"Created AdamW optimizer with layer decay: "
        f"encoder_lr={encoder_lr}, layer_decay={layer_decay}, "
        f"head_lr={head_lr}, token_head_lr={token_head_lr}"
    )

    return optimizer


def create_layer_wise_lr_groups(
    model: nn.Module,
    base_lr: float = 2e-5,
    layer_decay: float = 0.95,
    num_layers: int = 22,
    weight_decay: float = 0.01,
) -> list[dict]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Lower layers (closer to input) get lower learning rates.
    This is useful for fine-tuning where lower layers learn more general features.

    Args:
        model: The model to create parameter groups for.
        base_lr: Learning rate for top layer. Default: 2e-5
        layer_decay: Decay factor per layer. Default: 0.95
        num_layers: Number of transformer layers. Default: 22 (ModernBERT-base)
        weight_decay: Weight decay. Default: 0.01

    Returns:
        List of parameter group dicts for optimizer.

    Example:
        >>> # Layer 0 (bottom): lr = 2e-5 * 0.95^22 ≈ 6.5e-6
        >>> # Layer 22 (top): lr = 2e-5
        >>> param_groups = create_layer_wise_lr_groups(model)
    """
    # Pattern to extract layer number
    layer_pattern = re.compile(r"layer\.(\d+)\.")

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Try to extract layer number
        match = layer_pattern.search(name)
        if match:
            layer_num = int(match.group(1))
            # Calculate LR for this layer (top layer = base_lr)
            lr = base_lr * (layer_decay ** (num_layers - layer_num))
        else:
            # Non-layer params (embeddings, heads) - use base LR
            layer_num = -1
            lr = base_lr

        # Create group key
        is_no_decay = any(nd in name for nd in ["bias", "LayerNorm", "layer_norm"])
        group_key = (layer_num, is_no_decay)

        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": lr,
                "weight_decay": 0.0 if is_no_decay else weight_decay,
                "name": f"layer_{layer_num}_{'no_decay' if is_no_decay else 'decay'}",
            }

        param_groups[group_key]["params"].append(param)

    groups = list(param_groups.values())

    # Log summary
    logger.info(f"Created {len(groups)} layer-wise LR groups with decay={layer_decay}")

    return groups


# Export public API
__all__ = [
    "create_param_groups",
    "create_optimizer_with_head_lr",
    "create_optimizer_with_layer_decay",
    "create_layer_wise_lr_groups",
]

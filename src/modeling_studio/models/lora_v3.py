# src/modeling_studio/models/lora_v3.py

"""
LoRA (Low-Rank Adaptation) implementation for ModernBERT v3.3 Ultra.

This module implements efficient fine-tuning via low-rank adaptation matrices,
specifically designed for the Family Band (layers 23-28). LoRA adds trainable
low-rank matrices to frozen transformer weights, enabling parameter-efficient
fine-tuning.

Reference: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for efficient fine-tuning.

    Adds trainable low-rank matrices A and B to a frozen weight matrix W:
        output = (W + BA) @ x = W @ x + B @ (A @ x)

    Where:
        - W: Original frozen weights [out_features, in_features]
        - A: Down projection [r, in_features] - initialized with Kaiming uniform
        - B: Up projection [out_features, r] - initialized with zeros
        - r: LoRA rank (default: 16)
        - alpha: Scaling hyperparameter (default: 16)

    The scaling factor (alpha / r) is applied to maintain stable learning rates
    across different ranks.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank (number of low-rank dimensions)
        alpha: Scaling hyperparameter
        dropout: Dropout probability applied before LoRA projection

    Shape:
        - Input: [batch, seq_len, in_features]
        - Output: [batch, seq_len, out_features] (the LoRA delta)

    Example:
        >>> lora = LoRALayer(in_features=768, out_features=768, r=16, alpha=16)
        >>> x = torch.randn(2, 50, 768)
        >>> delta = lora(x)  # LoRA contribution to add to base layer output
        >>> assert delta.shape == (2, 50, 768)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA matrices (no bias)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize: A with Kaiming uniform, B with zeros
        # This ensures gradients flow through A initially while B starts neutral
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA forward pass.

        Returns the LoRA delta (to be added to base layer output).

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            LoRA contribution [batch, seq, out_features]
        """
        # Apply dropout → A projection → B projection → scaling
        # x: [batch, seq, in_features] → [batch, seq, r] → [batch, seq, out_features]
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return lora_output * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adapter.

    This is a drop-in replacement for nn.Linear that optionally adds a LoRA
    adapter. During training, the base linear weights can be frozen while
    only the LoRA parameters are updated.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether base layer has bias
        r: LoRA rank
        alpha: LoRA scaling parameter
        dropout: Dropout for LoRA path
        enable_lora: Whether to enable LoRA adapter

    Shape:
        - Input: [batch, seq_len, in_features]
        - Output: [batch, seq_len, out_features]

    Example:
        >>> layer = LinearWithLoRA(768, 768, r=16, alpha=16)
        >>> layer.freeze_base()  # Freeze base weights for LoRA training
        >>> x = torch.randn(2, 50, 768)
        >>> y = layer(x)
        >>> assert y.shape == (2, 50, 768)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        enable_lora: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Base linear layer (will be frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapter
        self.enable_lora = enable_lora
        if enable_lora:
            self.lora = LoRALayer(
                in_features=in_features,
                out_features=out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )
        else:
            self.lora = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with base + LoRA.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            Output tensor [batch, seq, out_features]
        """
        # Base linear transformation
        output = self.linear(x)

        # Add LoRA contribution if enabled
        if self.lora is not None and self.enable_lora:
            output = output + self.lora(x)

        return output

    def merge_lora(self) -> None:
        """
        Merge LoRA weights into base weights (for inference).

        After merging:
            W' = W + (B @ A) * scaling

        The model then behaves as a standard linear layer with no LoRA overhead.
        This is useful for deployment where you want the benefits of LoRA training
        without the runtime cost.

        Note: This operation is irreversible - LoRA is disabled after merging.
        """
        if self.lora is None:
            return

        with torch.no_grad():
            # Compute low-rank update: B @ A * scaling
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight
            # Add to base weights
            self.linear.weight.add_(lora_weight * self.lora.scaling)

        # Disable LoRA after merging
        self.lora = None
        self.enable_lora = False

    def freeze_base(self) -> None:
        """
        Freeze base weights, keep LoRA trainable.

        This is the standard LoRA training setup where the pretrained weights
        are frozen and only the low-rank adaptation matrices are trained.
        """
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def unfreeze_base(self) -> None:
        """
        Unfreeze base weights.

        Useful for full fine-tuning after LoRA training or for debugging.
        """
        self.linear.weight.requires_grad_(True)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(True)

    def extra_repr(self) -> str:
        lora_status = "enabled" if self.lora is not None and self.enable_lora else "disabled"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, lora={lora_status}"
        )


def apply_lora_to_layer(
    layer: nn.Module,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: set[str] | None = None,
) -> dict[str, LoRALayer]:
    """
    Apply LoRA adapters to specific modules in a transformer layer.

    This function searches for Linear modules in the layer that match the
    target module names (typically attention projections: q_proj, k_proj,
    v_proj, out_proj) and creates LoRA adapters for them.

    Args:
        layer: Transformer layer to add LoRA to
        r: LoRA rank
        alpha: LoRA scaling parameter
        dropout: Dropout for LoRA path
        target_modules: Set of module names to target (e.g., {"q_proj", "k_proj"})
                       Default: {"q_proj", "k_proj", "v_proj", "out_proj"}

    Returns:
        Dict mapping module names to their LoRA adapters

    Example:
        >>> from modeling_studio.models.attention_v3 import MultiScaleAttentionWithGlobals
        >>> attention = MultiScaleAttentionWithGlobals(768, 12, layer_idx=23)
        >>> lora_modules = apply_lora_to_layer(attention, r=16, alpha=16)
        >>> print(f"Added LoRA to: {list(lora_modules.keys())}")
    """
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "out_proj"}

    lora_modules = {}

    for name, module in layer.named_modules():
        # Check if this module name matches any target
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA adapter
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                lora_modules[name] = lora

    return lora_modules


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Get all LoRA parameters for optimizer.

    This function collects all parameters with "lora" in their name,
    which should be the only trainable parameters during LoRA training.

    Args:
        model: Model containing LoRA layers

    Returns:
        List of LoRA parameters

    Example:
        >>> optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-4)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count trainable LoRA parameters.

    Args:
        model: Model containing LoRA layers

    Returns:
        Number of trainable LoRA parameters

    Example:
        >>> print(f"LoRA parameters: {count_lora_parameters(model):,}")
    """
    return sum(
        p.numel() for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad
    )


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """
    Freeze all non-LoRA parameters in the model.

    This is a convenience function for setting up LoRA training where
    only LoRA adapters should be trainable.

    Args:
        model: Model containing LoRA layers
    """
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad_(False)


def print_lora_info(model: nn.Module) -> None:
    """
    Print summary of LoRA configuration in the model.

    Args:
        model: Model containing LoRA layers
    """
    lora_params = count_lora_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("LoRA Configuration Summary")
    print("=" * 70)
    print(f"Total parameters:     {total_params:>15,}")
    print(f"Trainable parameters: {trainable_params:>15,}")
    print(f"LoRA parameters:      {lora_params:>15,}")
    print(f"LoRA percentage:      {100 * lora_params / total_params:>14.2f}%")
    print(f"Trainable percentage: {100 * trainable_params / total_params:>14.2f}%")
    print("=" * 70 + "\n")


# Export public API
__all__ = [
    "LoRALayer",
    "LinearWithLoRA",
    "apply_lora_to_layer",
    "get_lora_parameters",
    "count_lora_parameters",
    "freeze_non_lora_parameters",
    "print_lora_info",
]

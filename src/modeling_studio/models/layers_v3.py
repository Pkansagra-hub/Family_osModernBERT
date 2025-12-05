# src/modeling_studio/models/layers_v3.py

"""
ModernBERT v3.3 Ultra - Transformer Layer Implementation

This module implements the complete transformer layer combining:
- Multi-scale sliding window attention with global hub tokens (Issue 2.1.3)
- GELU FFN (Issue 2.2.1)
- LoRA adaptation for Family Band layers 23-28 (Issue 2.2.2)

Architecture: Pre-LayerNorm (like GPT-2, not BERT)
    x → LN → Attention → + → LN → FFN → + → output
        └──────────────────┘   └─────────┘
            (residual)          (residual)

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention_v3 import (
    create_attention_layer,
    get_window_size_for_layer,
    get_layer_band_name,
)
from .ffn_v3 import create_ffn
from .lora_v3 import LoRALayer


class ModernBERTLayerV3(nn.Module):
    """
    Single transformer layer for ModernBERT v3.3 Ultra.

    This layer implements the core transformer block with multi-scale attention,
    feed-forward network, and optional LoRA adaptation for the Family Band.

    Components:
        1. Multi-Scale Attention with Global Hub Tokens
           - Sliding window sizes: 64→128→256→512 by layer band
           - Global attention for hub tokens (positions 0-4)
           - Bidirectional: hubs see all, all see hubs
        2. GELU Feed-Forward Network
           - 768 → 3072 → 768 (4x expansion)
           - GELU activation
        3. Pre-LayerNorm Architecture
           - LayerNorm before attention and FFN (not after)
           - Residual connections around both sub-layers
        4. Optional LoRA Adapters (Family Band only, L23-28)
           - Low-rank adaptation for efficient fine-tuning
           - Applied to attention output

    Architecture Flow:
        Input [batch, seq, 768]
            ↓
        LayerNorm → Attention → Dropout → + (residual)
            ↓                              ↑
        LayerNorm → FFN → + (residual) ────┘
            ↓
        Output [batch, seq, 768]

    Args:
        hidden_size: Hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        intermediate_size: FFN intermediate size (default: 3072)
        hidden_dropout_prob: Dropout probability for residuals
        attention_probs_dropout_prob: Dropout in attention
        layer_idx: 1-indexed layer number (1-28)
        use_flash_attention: Whether to use Flash Attention
        enable_lora: Whether to enable LoRA adapters
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling parameter (default: 16)
        lora_dropout: Dropout for LoRA path (default: 0.05)

    Example:
        >>> layer = ModernBERTLayerV3(layer_idx=23, enable_lora=True)
        >>> x = torch.randn(2, 512, 768)
        >>> output, attn_weights = layer(x)
        >>> assert output.shape == (2, 512, 768)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_idx: int = 1,
        use_flash_attention: bool = True,
        enable_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # Layer metadata
        self.layer_idx = layer_idx
        self.band = get_layer_band_name(layer_idx)
        self.window_size = get_window_size_for_layer(layer_idx)
        self.hidden_size = hidden_size
        self.enable_lora = enable_lora

        # Pre-LayerNorm architecture (like GPT-2, not BERT)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Multi-scale attention with global hub tokens
        self.attention = create_attention_layer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_probs_dropout_prob,
            layer_idx=layer_idx,
            use_flash_attention=use_flash_attention,
        )

        # GELU Feed-Forward Network
        self.ffn = create_ffn(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_type="gelu",
        )

        # Dropout for residual connections
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # LoRA adapters (only for Family Band: layers 23-28)
        self.lora_q: Optional[LoRALayer] = None
        self.lora_k: Optional[LoRALayer] = None
        self.lora_v: Optional[LoRALayer] = None
        self.lora_o: Optional[LoRALayer] = None

        if enable_lora and 23 <= layer_idx <= 28:
            self._init_lora(hidden_size, lora_r, lora_alpha, lora_dropout)

    def _init_lora(
        self,
        hidden_size: int,
        r: int,
        alpha: int,
        dropout: float,
    ) -> None:
        """
        Initialize LoRA adapters for attention projections.

        LoRA is only applied to the Family Band (layers 23-28) for efficient
        fine-tuning of family-specific capabilities while keeping the
        foundation layers frozen.

        Args:
            hidden_size: Hidden dimension
            r: LoRA rank (number of low-rank dimensions)
            alpha: Scaling hyperparameter
            dropout: Dropout probability before LoRA projection
        """
        self.lora_q = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_k = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_v = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        self.lora_o = LoRALayer(hidden_size, hidden_size, r, alpha, dropout)
        print(f"  ✓ LoRA initialized for layer {self.layer_idx} (r={r}, alpha={alpha})")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer layer.

        Pre-LayerNorm architecture:
        1. Apply LayerNorm before attention
        2. Attention with residual connection
        3. Apply LayerNorm before FFN
        4. FFN with residual connection

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Padding mask [batch, seq_len]
                           1 = attend, 0 = mask out
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of:
                - output_hidden_states: [batch, seq_len, hidden_size]
                - attention_weights: [batch, heads, seq_len, seq_len] or None
        """
        # === Attention Block (Pre-LayerNorm) ===
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)

        # Multi-scale attention with global hub tokens
        attn_output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # Add LoRA contribution if enabled (Family Band only)
        if self.lora_o is not None:
            # LoRA is applied to the pre-attention hidden states
            # and added to the attention output
            # Note: Full LoRA integration would modify Q/K/V directly,
            # but this simplified approach adds LoRA to output projection
            lora_contrib = self.lora_o(hidden_states)
            attn_output = attn_output + lora_contrib

        # Dropout and residual connection
        attn_output = self.dropout(attn_output)
        hidden_states = residual + attn_output

        # === FFN Block (Pre-LayerNorm) ===
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)

        # GELU Feed-Forward Network
        ffn_output = self.ffn(hidden_states)

        # Residual connection (FFN has internal dropout)
        hidden_states = residual + ffn_output

        return hidden_states, attn_weights

    def freeze_base_weights(self) -> None:
        """
        Freeze all weights except LoRA adapters.

        This is used during Phase 1 training where only LoRA parameters
        in the Family Band are trained while the foundation layers remain frozen.
        """
        for name, param in self.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad_(False)
        print(f"  ❄️ Froze base weights for layer {self.layer_idx}")

    def unfreeze_base_weights(self) -> None:
        """
        Unfreeze all base weights.

        Used for full fine-tuning or debugging.
        """
        for name, param in self.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad_(True)
        print(f"  🔥 Unfroze base weights for layer {self.layer_idx}")

    def merge_lora_weights(self) -> None:
        """
        Merge LoRA weights into base weights for inference.

        After merging, the model behaves as a standard transformer
        without LoRA overhead. This is useful for deployment.

        Note: This is a placeholder. Full implementation would require
        modifying the attention module's projection layers.
        """
        if self.lora_o is None:
            return

        print(f"  ⚠️ LoRA merging not yet implemented for layer {self.layer_idx}")
        # TODO: Implement LoRA merging into attention projections
        # This requires modifying q_proj, k_proj, v_proj, out_proj in attention module

    def get_num_params(self) -> dict[str, int]:
        """
        Get parameter counts for the layer.

        Returns:
            Dict with:
                - total: Total parameters
                - attention: Attention parameters
                - ffn: FFN parameters
                - lora: LoRA parameters (if enabled)
                - trainable: Trainable parameters
        """
        total = sum(p.numel() for p in self.parameters())
        attention = sum(p.numel() for p in self.attention.parameters())
        ffn = sum(p.numel() for p in self.ffn.parameters())
        lora = sum(p.numel() for n, p in self.named_parameters() if "lora" in n.lower())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "attention": attention,
            "ffn": ffn,
            "lora": lora,
            "trainable": trainable,
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        lora_status = "enabled" if self.enable_lora and self.lora_q else "disabled"
        return (
            f"layer={self.layer_idx}, band={self.band}, "
            f"window={self.window_size}, lora={lora_status}"
        )


# ==============================================================================
# Layer Stack Creation
# ==============================================================================


def create_layer_stack(
    num_layers: int = 28,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    use_flash_attention: bool = True,
    lora_layers: list[int] | None = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
) -> nn.ModuleList:
    """
    Create the full 28-layer transformer stack for ModernBERT v3.3 Ultra.

    Layer Band Configuration:
        - Foundation Band (L1-6): Window=64, No LoRA, Frozen in Phase 1
        - Context Band (L7-18): Window=128, No LoRA, Frozen in Phase 1
        - Semantic Band (L19-22): Window=256, No LoRA, Trainable in Phase 1
        - Family Band (L23-28): Window=512, LoRA enabled, Trainable in Phase 1

    Args:
        num_layers: Number of layers (default: 28)
        hidden_size: Hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        intermediate_size: FFN intermediate size (default: 3072)
        hidden_dropout_prob: Dropout for residuals
        attention_probs_dropout_prob: Dropout in attention
        use_flash_attention: Whether to use Flash Attention
        lora_layers: Layer indices to apply LoRA (default: [23-28])
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling parameter (default: 16)

    Returns:
        nn.ModuleList of ModernBERTLayerV3 layers

    Example:
        >>> layers = create_layer_stack(num_layers=28)
        >>> print(f"Created {len(layers)} layers")
        >>> print(f"Layer 1 window: {layers[0].window_size}")  # 64
        >>> print(f"Layer 25 has LoRA: {layers[24].lora_o is not None}")  # True
    """
    if lora_layers is None:
        lora_layers = [23, 24, 25, 26, 27, 28]

    layers = nn.ModuleList()

    print("\n🏗️  Building v3 transformer stack...")
    print(f"   Layers: {num_layers}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Attention heads: {num_attention_heads}")
    print(f"   FFN intermediate: {intermediate_size}")
    print(f"   LoRA layers: {lora_layers}")
    print()

    for i in range(1, num_layers + 1):
        enable_lora = i in lora_layers
        layer = ModernBERTLayerV3(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_idx=i,
            use_flash_attention=use_flash_attention,
            enable_lora=enable_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        layers.append(layer)

    print(f"\n✓ Created {num_layers} layers")
    print(f"  - Foundation (L1-6): window=64, no LoRA")
    print(f"  - Context (L7-18): window=128, no LoRA")
    print(f"  - Semantic (L19-22): window=256, no LoRA")
    print(f"  - Family (L23-28): window=512, LoRA enabled")
    print()

    return layers


def freeze_layer_bands(
    layers: nn.ModuleList,
    freeze_bands: list[str] | None = None,
) -> None:
    """
    Freeze specific layer bands.

    Args:
        layers: ModuleList of layers
        freeze_bands: Bands to freeze (default: ["foundation", "context"])
                     Options: "foundation", "context", "semantic", "family"

    Example:
        >>> layers = create_layer_stack()
        >>> freeze_layer_bands(layers, ["foundation", "context"])
        ❄️ Froze 18 layers (Foundation + Context)
    """
    if freeze_bands is None:
        freeze_bands = ["foundation", "context"]

    band_ranges = {
        "foundation": range(0, 6),  # L1-6 (0-indexed)
        "context": range(6, 18),  # L7-18
        "semantic": range(18, 22),  # L19-22
        "family": range(22, 28),  # L23-28
    }

    frozen_count = 0
    for band_name in freeze_bands:
        if band_name not in band_ranges:
            print(f"⚠️ Unknown band: {band_name}")
            continue

        for i in band_ranges[band_name]:
            layers[i].freeze_base_weights()
            frozen_count += 1

    print(f"\n❄️ Froze {frozen_count} layers ({', '.join(freeze_bands)})")


def unfreeze_layer_bands(
    layers: nn.ModuleList,
    unfreeze_bands: list[str] | None = None,
) -> None:
    """
    Unfreeze specific layer bands.

    Args:
        layers: ModuleList of layers
        unfreeze_bands: Bands to unfreeze (default: ["semantic", "family"])

    Example:
        >>> unfreeze_layer_bands(layers, ["semantic", "family"])
        🔥 Unfroze 10 layers (Semantic + Family)
    """
    if unfreeze_bands is None:
        unfreeze_bands = ["semantic", "family"]

    band_ranges = {
        "foundation": range(0, 6),
        "context": range(6, 18),
        "semantic": range(18, 22),
        "family": range(22, 28),
    }

    unfrozen_count = 0
    for band_name in unfreeze_bands:
        if band_name not in band_ranges:
            print(f"⚠️ Unknown band: {band_name}")
            continue

        for i in band_ranges[band_name]:
            layers[i].unfreeze_base_weights()
            unfrozen_count += 1

    print(f"\n🔥 Unfroze {unfrozen_count} layers ({', '.join(unfreeze_bands)})")


def get_layer_stats(layers: nn.ModuleList) -> dict:
    """
    Get statistics about the layer stack.

    Returns:
        Dict with:
            - num_layers: Total layers
            - total_params: Total parameters
            - trainable_params: Trainable parameters
            - lora_params: LoRA parameters
            - by_band: Stats per band
    """
    stats = {
        "num_layers": len(layers),
        "total_params": 0,
        "trainable_params": 0,
        "lora_params": 0,
        "by_band": {
            "foundation": {"layers": [], "params": 0, "trainable": 0},
            "context": {"layers": [], "params": 0, "trainable": 0},
            "semantic": {"layers": [], "params": 0, "trainable": 0},
            "family": {"layers": [], "params": 0, "trainable": 0, "lora": 0},
        },
    }

    for layer in layers:
        layer_params = layer.get_num_params()
        band = layer.band

        stats["total_params"] += layer_params["total"]
        stats["trainable_params"] += layer_params["trainable"]
        stats["lora_params"] += layer_params["lora"]

        stats["by_band"][band]["layers"].append(layer.layer_idx)
        stats["by_band"][band]["params"] += layer_params["total"]
        stats["by_band"][band]["trainable"] += layer_params["trainable"]
        if band == "family":
            stats["by_band"][band]["lora"] += layer_params["lora"]

    return stats


def print_layer_stack_summary(layers: nn.ModuleList) -> None:
    """
    Print detailed summary of the layer stack.

    Args:
        layers: ModuleList of layers
    """
    stats = get_layer_stats(layers)

    print("\n" + "=" * 70)
    print("ModernBERT v3.3 Ultra - Layer Stack Summary")
    print("=" * 70)
    print(f"Total Layers: {stats['num_layers']}")
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"LoRA Parameters: {stats['lora_params']:,}")
    print(f"Trainable %: {100 * stats['trainable_params'] / stats['total_params']:.2f}%")
    print()

    print("Layer Bands:")
    for band_name, band_info in stats["by_band"].items():
        layers_str = f"L{min(band_info['layers'])}-L{max(band_info['layers'])}"
        print(
            f"  {band_name.capitalize():12} {layers_str:8} | {band_info['params']:>12,} params | {band_info['trainable']:>12,} trainable"
        )
        if band_name == "family":
            print(f"                                | {band_info['lora']:>12,} LoRA params")

    print("=" * 70 + "\n")


# Export public API
__all__ = [
    "ModernBERTLayerV3",
    "create_layer_stack",
    "freeze_layer_bands",
    "unfreeze_layer_bands",
    "get_layer_stats",
    "print_layer_stack_summary",
]

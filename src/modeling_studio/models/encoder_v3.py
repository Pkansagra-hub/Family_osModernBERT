"""
ModernBERT v3.3 Ultra - 28-Layer Encoder Stack

This module implements the complete 28-layer encoder stack that chains
ModernBERTLayerV3 layers with proper gradient checkpointing support for
memory-efficient training on long sequences (up to 8192 tokens).

Key Features:
    - 28 transformer layers organized in 4 bands
    - Layer bands: Foundation (L1-6), Context (L7-18), Semantic (L19-22), Family (L23-28)
    - Gradient checkpointing for memory efficiency
    - Layer freezing/unfreezing utilities for staged training
    - Band-based layer access and management
    - Hub token preservation through all layers

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers_v3 import (
    create_layer_stack,
    get_layer_stats,
)


class ModernBERTEncoderV3(nn.Module):
    """
    28-layer encoder stack for ModernBERT v3.3 Ultra.

    Layer Structure:
        - Layers 1-6:   Foundation Band (window=64, frozen in Phase 1)
        - Layers 7-18:  Context Band (window=128, frozen in Phase 1)
        - Layers 19-22: Semantic Band (window=256, trainable in Phase 1)
        - Layers 23-28: Family Band (window=512, trainable + LoRA in Phase 1)

    Features:
        - Gradient checkpointing for memory efficiency on 8k sequences
        - Progressive window sizes: 64 → 128 → 256 → 512 tokens
        - Hub token preservation through all layers
        - Layer band management for staged training

    Args:
        num_layers: Number of transformer layers (default: 28)
        hidden_size: Hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        intermediate_size: FFN intermediate size (default: 3072)
        hidden_dropout_prob: Dropout probability for residuals
        attention_probs_dropout_prob: Dropout probability for attention
        use_flash_attention: Whether to use Flash Attention
        gradient_checkpointing: Enable gradient checkpointing for memory
        lora_layers: Layer indices to apply LoRA (default: [23-28])
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling parameter (default: 16)

    Example:
        >>> encoder = ModernBERTEncoderV3(gradient_checkpointing=True)
        >>> x = torch.randn(2, 512, 768)
        >>> output, hidden_states, attentions = encoder(x, output_hidden_states=True)
        >>> print(f"Output shape: {output.shape}")
        >>> print(f"Hidden states: {len(hidden_states)} layers")
    """

    def __init__(
        self,
        num_layers: int = 28,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = False,
        lora_layers: list[int] | None = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.gradient_checkpointing = gradient_checkpointing

        # Create 28-layer transformer stack
        self.layers = create_layer_stack(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_flash_attention=use_flash_attention,
            use_fused_qkv=True,  # ModernBERT-compatible fused Wqkv
            lora_layers=lora_layers,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Layer band mapping for easy access
        self.layer_bands = {
            "foundation": list(range(0, 6)),  # L1-6 (0-indexed)
            "context": list(range(6, 18)),  # L7-18
            "semantic": list(range(18, 22)),  # L19-22
            "family": list(range(22, 28)),  # L23-28
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """
        Forward pass through all 28 layers.

        Args:
            hidden_states: Input embeddings [batch, seq_len, hidden_size]
            attention_mask: Padding mask [batch, seq_len]
            output_hidden_states: Return all layer outputs
            output_attentions: Return all attention weights

        Returns:
            Tuple of:
                - last_hidden_state: Final layer output [batch, seq_len, hidden_size]
                - all_hidden_states: List of all layer outputs (if output_hidden_states=True)
                - all_attentions: List of all attention weights (if output_attentions=True)

        Example:
            >>> encoder = ModernBERTEncoderV3()
            >>> x = torch.randn(2, 512, 768)
            >>> output, hidden_states, attentions = encoder(x, output_hidden_states=True)
            >>> len(hidden_states)  # 28 layers + input
            29
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        # Store input embeddings as first hidden state
        if output_hidden_states:
            all_hidden_states.append(hidden_states)  # type: ignore

        # Pass through all 28 layers
        for _i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                hidden_states, attn_weights = self._checkpoint_forward(
                    layer, hidden_states, attention_mask, output_attentions
                )
            else:
                # Standard forward pass
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            # Collect hidden states and attention weights if requested
            if output_hidden_states:
                all_hidden_states.append(hidden_states)  # type: ignore

            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)  # type: ignore

        return hidden_states, all_hidden_states, all_attentions

    def _checkpoint_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with gradient checkpointing.

        Note: Gradient checkpointing doesn't support returning attention weights,
        so we return None for attention if checkpointing is enabled.

        Args:
            layer: The layer to checkpoint
            hidden_states: Input to layer
            attention_mask: Padding mask
            output_attentions: Whether to return attention (ignored with checkpointing)

        Returns:
            (hidden_states, None) - Attention weights are None with checkpointing
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                output = module(*inputs, output_attentions=False)
                # Return only hidden_states (first element of tuple)
                return output[0] if isinstance(output, tuple) else output

            return custom_forward

        hidden_states = checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            use_reentrant=False,
        )  # type: ignore
        # Checkpoint returns only hidden_states
        # attn_weights will be None when checkpointing

        # Gradient checkpointing doesn't support attention output
        return hidden_states, None

    def freeze_layers(self, layer_indices: list[int]) -> None:
        """
        Freeze specific layers by index.

        Args:
            layer_indices: 1-indexed layer numbers to freeze (e.g., [1, 2, 3])

        Example:
            >>> encoder.freeze_layers([1, 2, 3, 4, 5, 6])  # Freeze foundation band
            ❄️ Froze 6 layers: [1, 2, 3, 4, 5, 6]
        """
        for idx in layer_indices:
            if 1 <= idx <= self.num_layers:
                self.layers[idx - 1].freeze_base_weights()  # type: ignore # Convert to 0-indexed
            else:
                print(f"[WARN] Invalid layer index: {idx} (valid: 1-{self.num_layers})")

        print(f"[FREEZE] Froze {len(layer_indices)} layers: {layer_indices}")

    def unfreeze_layers(self, layer_indices: list[int]) -> None:
        """
        Unfreeze specific layers by index.

        Args:
            layer_indices: 1-indexed layer numbers to unfreeze (e.g., [23, 24, 25, 26, 27, 28])

        Example:
            >>> encoder.unfreeze_layers([23, 24, 25, 26, 27, 28])  # Unfreeze family band
            🔥 Unfroze 6 layers: [23, 24, 25, 26, 27, 28]
        """
        for idx in layer_indices:
            if 1 <= idx <= self.num_layers:
                self.layers[idx - 1].unfreeze_base_weights()  # type: ignore # Convert to 0-indexed
            else:
                print(f"[WARN] Invalid layer index: {idx} (valid: 1-{self.num_layers})")

        print(f"[UNFREEZE] Unfroze {len(layer_indices)} layers: {layer_indices}")

    def freeze_by_band(self, bands: list[str]) -> None:
        """
        Freeze all layers in specified bands.

        Args:
            bands: Band names to freeze
                   Options: "foundation", "context", "semantic", "family"

        Example:
            >>> encoder.freeze_by_band(["foundation", "context"])
            ❄️ Froze bands: foundation, context (18 layers)
        """
        layer_indices = []
        for band_name in bands:
            if band_name in self.layer_bands:
                # Convert 0-indexed to 1-indexed
                layer_indices.extend([i + 1 for i in self.layer_bands[band_name]])
            else:
                print(f"[WARN] Unknown band: {band_name}")

        if layer_indices:
            self.freeze_layers(sorted(layer_indices))
            print(f"[FREEZE] Froze bands: {', '.join(bands)} ({len(layer_indices)} layers)")

    def unfreeze_by_band(self, bands: list[str]) -> None:
        """
        Unfreeze all layers in specified bands.

        Args:
            bands: Band names to unfreeze
                   Options: "foundation", "context", "semantic", "family"

        Example:
            >>> encoder.unfreeze_by_band(["semantic", "family"])
            🔥 Unfroze bands: semantic, family (10 layers)
        """
        layer_indices = []
        for band_name in bands:
            if band_name in self.layer_bands:
                # Convert 0-indexed to 1-indexed
                layer_indices.extend([i + 1 for i in self.layer_bands[band_name]])
            else:
                print(f"[WARN] Unknown band: {band_name}")

        if layer_indices:
            self.unfreeze_layers(sorted(layer_indices))
            print(f"[UNFREEZE] Unfroze bands: {', '.join(bands)} ({len(layer_indices)} layers)")

    def get_layers_by_band(self, band: str) -> list[nn.Module]:
        """
        Get all layers in a specific band.

        Args:
            band: Band name ("foundation", "context", "semantic", "family")

        Returns:
            List of layer modules in the band

        Example:
            >>> family_layers = encoder.get_layers_by_band("family")
            >>> len(family_layers)  # 6 layers (L23-28)
            6
        """
        if band not in self.layer_bands:
            raise ValueError(f"Unknown band: {band}. Valid: {list(self.layer_bands.keys())}")

        return [self.layers[i] for i in self.layer_bands[band]]

    def print_layer_summary(self) -> None:
        """
        Print detailed summary of encoder configuration.

        Shows layer bands, window sizes, LoRA status, and parameter counts.
        """
        print("\n" + "=" * 60)
        print("ModernBERT v3.3 Ultra - Encoder Stack Summary")
        print("=" * 60)
        print(f"Total Layers: {self.num_layers}")
        print(f"Gradient Checkpointing: {'Enabled' if self.gradient_checkpointing else 'Disabled'}")
        print()

        print("Layer Bands:")
        for band_name, layer_indices in self.layer_bands.items():
            first_layer = self.layers[layer_indices[0]]
            window = first_layer.window_size
            has_lora = first_layer.lora_o is not None
            lora_str = "LoRA enabled" if has_lora else "no LoRA"

            layer_range = f"L{layer_indices[0]+1}-L{layer_indices[-1]+1}"
            print(f"  {band_name.capitalize():12} {layer_range:8} | window={window:3} | {lora_str}")

        # Get parameter statistics
        stats = get_layer_stats(self.layers)
        print()
        print("Parameters:")
        print(f"  Total: {stats['total_params']:,}")
        print(f"  Trainable: {stats['trainable_params']:,}")
        print(f"  LoRA: {stats['lora_params']:,}")
        print(f"  Trainable %: {100 * stats['trainable_params'] / stats['total_params']:.2f}%")

        print("=" * 60 + "\n")

    def get_num_params(self) -> dict[str, int]:
        """
        Get parameter counts for the encoder.

        Returns:
            Dictionary with parameter counts by category

        Example:
            >>> params = encoder.get_num_params()
            >>> print(f"Total: {params['total']:,}")
        """
        stats = get_layer_stats(self.layers)
        return {
            "total": stats["total_params"],
            "trainable": stats["trainable_params"],
            "lora": stats["lora_params"],
            "by_band": {band: info["params"] for band, info in stats["by_band"].items()},
        }  # type: ignore

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"gradient_checkpointing={self.gradient_checkpointing}"
        )


# Export public API
__all__ = [
    "ModernBERTEncoderV3",
]

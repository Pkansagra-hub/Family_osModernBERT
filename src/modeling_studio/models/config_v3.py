"""
Configuration dataclass for ModernBERT v3.3 Ultra.

This module defines the complete architecture configuration for v3, including:
- Hub token system with 4 specialized tokens ([EMO], [MEM], [REL], [TASK])
- Multi-scale sliding window attention across 4 layer bands
- LoRA configuration for family-specific layers (L23-28)
- Layer source mapping for function-preserving growth from v2
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


@dataclass
class ModernBERTv3Config:
    """Configuration for ModernBERT v3.3 Ultra."""

    # Architecture
    hidden_size: int = 768  # Same as v2 (enables weight transfer)
    num_layers: int = 28  # 22 from v2 + 6 cloned
    num_attention_heads: int = 12  # MHA (no GQA)
    intermediate_size: int = 1152  # ModernBERT GLU FFN (not 4x hidden)
    max_position_embeddings: int = 8192
    vocab_size: int = 50432  # 256-aligned (50368 base + 4 hub + 60 padding = 256×197)

    # Hub Tokens
    hub_tokens: list[str] = field(default_factory=lambda: ["[EMO]", "[MEM]", "[REL]", "[TASK]"])
    hub_token_positions: dict[str, int] = field(
        default_factory=lambda: {"[CLS]": 0, "[EMO]": 1, "[MEM]": 2, "[REL]": 3, "[TASK]": 4}
    )
    global_attention_positions: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Sliding Window by Layer Band
    window_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "foundation": 64,  # Layers 1-6
            "context": 128,  # Layers 7-18
            "semantic": 256,  # Layers 19-22
            "family": 512,  # Layers 23-28
        }
    )

    # Layer Bands
    layer_bands: dict[str, list[int]] = field(
        default_factory=lambda: {
            "foundation": list(range(1, 7)),  # 1-6
            "context": list(range(7, 19)),  # 7-18
            "semantic": list(range(19, 23)),  # 19-22
            "family": list(range(23, 29)),  # 23-28
        }
    )

    # LoRA Configuration
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_layers: list[int] = field(default_factory=lambda: [23, 24, 25, 26, 27, 28])

    # Pair Encoder
    pair_encoder_enabled: bool = True
    pair_encoder_heads: int = 8
    pair_encoder_dropout: float = 0.1

    # Training
    frozen_layers_phase1: list[int] = field(default_factory=lambda: list(range(1, 19)))

    # FFN
    ffn_activation: str = "gelu"  # No SwiGLU (removed from roadmap)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate layer counts
        total_layers = sum(len(band) for band in self.layer_bands.values())
        if total_layers != self.num_layers:
            raise ValueError(
                f"Layer bands sum to {total_layers}, but num_layers is {self.num_layers}"
            )

        # Validate hub token positions
        expected_positions = {"[CLS]": 0, "[EMO]": 1, "[MEM]": 2, "[REL]": 3, "[TASK]": 4}
        if self.hub_token_positions != expected_positions:
            raise ValueError(
                f"Hub token positions mismatch. Expected {expected_positions}, "
                f"got {self.hub_token_positions}"
            )

        # Validate global attention positions
        if self.global_attention_positions != [0, 1, 2, 3, 4]:
            raise ValueError(
                f"Global attention positions must be [0, 1, 2, 3, 4], "
                f"got {self.global_attention_positions}"
            )

        # Validate window sizes
        for band, _ in self.layer_bands.items():
            if band not in self.window_sizes:
                raise ValueError(f"Window size not defined for band '{band}'")

        # Validate LoRA target layers
        if self.lora_enabled:
            family_layers = set(self.layer_bands["family"])
            lora_layers = set(self.lora_target_layers)
            if lora_layers != family_layers:
                raise ValueError(
                    f"LoRA target layers {lora_layers} must match family band layers {family_layers}"
                )

        # Validate frozen layers
        foundation_context = set(self.layer_bands["foundation"] + self.layer_bands["context"])
        frozen_set = set(self.frozen_layers_phase1)
        if frozen_set != foundation_context:
            raise ValueError(
                f"Frozen layers {frozen_set} must match foundation + context bands {foundation_context}"
            )

    def get_layer_band(self, layer_idx: int) -> str:
        """Get the band name for a given layer index (1-indexed)."""
        for band_name, layers in self.layer_bands.items():
            if layer_idx in layers:
                return band_name
        raise ValueError(f"Layer {layer_idx} not found in any band")

    def get_window_size(self, layer_idx: int) -> int:
        """Get sliding window size for a given layer."""
        band = self.get_layer_band(layer_idx)
        return self.window_sizes[band]

    def get_trainable_layers(self, phase: str = "phase1") -> list[int]:
        """Get list of trainable layers for a given training phase."""
        if phase == "phase0":
            # Phase 0 (healing): Only L23-28 trainable
            return self.layer_bands["family"]
        elif phase == "phase1":
            # Phase 1: L19-28 trainable (semantic + family)
            return self.layer_bands["semantic"] + self.layer_bands["family"]
        elif phase == "phase2":
            # Phase 2: All layers trainable
            return list(range(1, self.num_layers + 1))
        else:
            raise ValueError(f"Unknown training phase: {phase}")

    def get_lora_layers(self) -> list[int]:
        """Get list of layers with LoRA adapters."""
        return self.lora_target_layers if self.lora_enabled else []

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "hub_tokens": self.hub_tokens,
            "hub_token_positions": self.hub_token_positions,
            "global_attention_positions": self.global_attention_positions,
            "window_sizes": self.window_sizes,
            "layer_bands": self.layer_bands,
            "lora_enabled": self.lora_enabled,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_layers": self.lora_target_layers,
            "pair_encoder_enabled": self.pair_encoder_enabled,
            "pair_encoder_heads": self.pair_encoder_heads,
            "pair_encoder_dropout": self.pair_encoder_dropout,
            "frozen_layers_phase1": self.frozen_layers_phase1,
            "ffn_activation": self.ffn_activation,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModernBERTv3Config":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary containing config parameters

        Returns:
            ModernBERTv3Config instance
        """
        return cls(**config_dict)


# Layer Source Mapping for Function Preserving Growth


class LayerSource(Enum):
    """Source of layer weights during v3 initialization."""

    COPY = "copy"  # Direct copy from v2 same layer
    CLONE = "clone"  # Clone from different v2 layer
    RANDOM = "random"  # Random initialization


class LayerMapping(NamedTuple):
    """Mapping of v3 layer to its weight source."""

    v3_layer: int
    source: LayerSource
    v2_layer: int  # Source layer in v2 (ignored if RANDOM)


def get_layer_source_mapping() -> dict[int, LayerMapping]:
    """
    Get the complete layer source mapping for v3 initialization.

    Returns:
        Dict mapping v3 layer index (1-28) to LayerMapping

    Strategy:
        - Layers 1-22: Copy directly from v2 layers 1-22
        - Layer 23: Clone from v2 layer 15
        - Layer 24: Clone from v2 layer 16
        - Layer 25: Clone from v2 layer 17
        - Layer 26: Clone from v2 layer 18
        - Layer 27: Clone from v2 layer 19
        - Layer 28: Clone from v2 layer 20

    This mapping ensures function-preserving growth:
    - First 22 layers maintain identical behavior to v2
    - New layers 23-28 start with mature semantic processing weights
    """
    mapping = {}

    # Layers 1-22: Direct copy from v2
    for i in range(1, 23):
        mapping[i] = LayerMapping(v3_layer=i, source=LayerSource.COPY, v2_layer=i)

    # Layers 23-28: Clone from v2 layers 15-20
    v2_source_layers = [15, 16, 17, 18, 19, 20]
    for i, v2_layer in enumerate(v2_source_layers, start=23):
        mapping[i] = LayerMapping(v3_layer=i, source=LayerSource.CLONE, v2_layer=v2_layer)

    return mapping


def print_layer_source_mapping():
    """Print a human-readable view of the layer source mapping."""
    mapping = get_layer_source_mapping()

    print("=" * 80)
    print("ModernBERT v3 Layer Source Mapping (Function Preserving Growth)")
    print("=" * 80)

    # Group by band
    bands = {
        "Foundation (L1-6)": list(range(1, 7)),
        "Context (L7-18)": list(range(7, 19)),
        "Semantic (L19-22)": list(range(19, 23)),
        "Family (L23-28)": list(range(23, 29)),
    }

    for band_name, layers in bands.items():
        print(f"\n{band_name}:")
        for layer in layers:
            layer_map = mapping[layer]
            if layer_map.source == LayerSource.COPY:
                print(f"  Layer {layer:2} ← COPY from v2 Layer {layer_map.v2_layer:2}")
            elif layer_map.source == LayerSource.CLONE:
                print(f"  Layer {layer:2} ← CLONE from v2 Layer {layer_map.v2_layer:2}")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  • Layers 1-22: Direct copy (function-preserving)")
    print(f"  • Layers 23-28: Cloned from mature v2 layers 15-20")
    print(f"  • Total v3 layers: 28")
    print(f"  • Total v2 layers: 22")
    print("=" * 80)

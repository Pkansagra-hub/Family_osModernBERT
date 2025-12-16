"""
Configuration dataclass for UltraBERT-Gen MoE Decoder.

This module defines the DecoderMoEConfig dataclass containing all hyperparameters
for the 13th head counterfactual generation decoder with Mixture-of-Experts.

Architecture Specifications:
    - Hidden: 1280
    - Layers: 8 (2 dense + 6 MoE)
    - GQA: 20 heads / 4 KV heads (5:1 ratio)
    - 8 experts + 1 shared expert, top-2 routing
    - SwiGLU FFN, RoPE positions, RMSNorm

Usage:
    from modeling_studio.models.decoder_config import DecoderMoEConfig

    # Default configuration (matches architecture doc)
    config = DecoderMoEConfig()

    # Custom configuration
    config = DecoderMoEConfig(
        hidden_size=1024,
        num_experts=4,
    )

    # Validate configuration
    config.validate()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DecoderMoEConfig:
    """
    Configuration for the UltraBERT-Gen MoE Decoder.

    This configuration defines the architecture of the 13th head decoder
    for counterfactual generation. All defaults match the architecture
    specification in decoder_moe_architecture.md.

    Attributes:
        hidden_size: Decoder hidden dimension. Default: 1280
        num_layers: Total decoder layers. Default: 8
        vocab_size: Vocabulary size (matches ModernBERT tokenizer). Default: 50280
        max_position_embeddings: Maximum sequence length. Default: 512

        num_attention_heads: Number of query heads for GQA. Default: 20
        num_kv_heads: Number of key/value heads for GQA. Default: 4
        head_dim: Dimension per attention head. Default: 64
        attention_dropout: Dropout in attention (modern practice: 0). Default: 0.0

        dense_layers: Layer indices using dense FFN. Default: (0, 1)
        moe_layers: Layer indices using MoE FFN. Default: (2, 3, 4, 5, 6, 7)
        dense_intermediate_size: Dense FFN intermediate dim (2.8× hidden). Default: 3584

        num_experts: Number of sparse experts. Default: 8
        num_experts_per_token: Top-k routing. Default: 2
        expert_intermediate_size: Expert FFN intermediate (1.6× hidden). Default: 2048
        use_shared_expert: Whether to use always-active shared expert. Default: True
        shared_expert_intermediate_size: Shared expert intermediate (1× hidden). Default: 1280

        load_balancing_loss_weight: Weight for load balance auxiliary loss. Default: 0.01
        router_z_loss_weight: Weight for router z-loss. Default: 0.001
        capacity_factor: Expert capacity factor (tokens per expert limit). Default: 1.5
        expert_dropout: Expert dropout probability during training. Default: 0.05

        encoder_hidden_size: Encoder output dimension (ModernBERT). Default: 768

        rope_theta: RoPE base frequency. Default: 10000.0

        hidden_dropout: Dropout for hidden states. Default: 0.1
        tie_word_embeddings: Whether to tie embeddings and LM head. Default: True

        rms_norm_eps: Epsilon for RMSNorm. Default: 1e-6
        initializer_range: Std for weight initialization. Default: 0.02
        use_cache: Whether to use KV cache for generation. Default: True
    """

    # Core dimensions
    hidden_size: int = 1280
    num_layers: int = 8
    vocab_size: int = 50280
    max_position_embeddings: int = 512

    # Attention configuration
    num_attention_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 64
    attention_dropout: float = 0.0

    # FFN configuration - Dense layers
    dense_layers: tuple[int, ...] = (0, 1)
    dense_intermediate_size: int = 3584  # 2.8× hidden

    # FFN configuration - MoE layers
    moe_layers: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_intermediate_size: int = 2048  # 1.6× hidden
    use_shared_expert: bool = True
    shared_expert_intermediate_size: int = 1280  # 1× hidden

    # MoE robustness configuration
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    capacity_factor: float = 1.5
    expert_dropout: float = 0.05

    # Encoder interface
    encoder_hidden_size: int = 768

    # RoPE configuration
    rope_theta: float = 10000.0

    # Regularization
    hidden_dropout: float = 0.1
    tie_word_embeddings: bool = True

    # Normalization
    rms_norm_eps: float = 1e-6

    # Initialization
    initializer_range: float = 0.02

    # Generation
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 0  # Same as pad for this model (decoder-start token)
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration constraints.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Attention head divisibility
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        # head_dim consistency
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim != expected_head_dim:
            logger.warning(
                f"head_dim ({self.head_dim}) != hidden_size/num_attention_heads ({expected_head_dim}). "
                "This may cause dimension mismatches."
            )

        # Layer configuration
        all_layers = set(self.dense_layers) | set(self.moe_layers)
        expected_layers = set(range(self.num_layers))
        if all_layers != expected_layers:
            raise ValueError(
                f"dense_layers + moe_layers must cover all layers 0 to {self.num_layers - 1}. "
                f"Got: {sorted(all_layers)}, expected: {sorted(expected_layers)}"
            )

        overlap = set(self.dense_layers) & set(self.moe_layers)
        if overlap:
            raise ValueError(
                f"dense_layers and moe_layers must not overlap. Overlap: {overlap}"
            )

        # MoE configuration
        if self.num_experts_per_token > self.num_experts:
            raise ValueError(
                f"num_experts_per_token ({self.num_experts_per_token}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )

        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")

        # Capacity factor
        if self.capacity_factor < 1.0:
            raise ValueError(
                f"capacity_factor ({self.capacity_factor}) must be >= 1.0"
            )

        # Dropout ranges
        for name, value in [
            ("attention_dropout", self.attention_dropout),
            ("hidden_dropout", self.hidden_dropout),
            ("expert_dropout", self.expert_dropout),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        # Loss weights
        if self.load_balancing_loss_weight < 0:
            raise ValueError(
                f"load_balancing_loss_weight must be >= 0, got {self.load_balancing_loss_weight}"
            )
        if self.router_z_loss_weight < 0:
            raise ValueError(
                f"router_z_loss_weight must be >= 0, got {self.router_z_loss_weight}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        d = self.to_dict()
        # Convert tuples to lists for JSON compatibility
        d["dense_layers"] = list(d["dense_layers"])
        d["moe_layers"] = list(d["moe_layers"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderMoEConfig":
        """Create configuration from dictionary."""
        # Convert lists back to tuples
        if "dense_layers" in d:
            d["dense_layers"] = tuple(d["dense_layers"])
        if "moe_layers" in d:
            d["moe_layers"] = tuple(d["moe_layers"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "DecoderMoEConfig":
        """Create configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "DecoderMoEConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def get_num_params(self, include_embeddings: bool = True) -> dict[str, int]:
        """
        Calculate parameter counts for each component.

        Args:
            include_embeddings: Whether to include embedding parameters.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        params = {}

        # Token embeddings (tied with LM head, count once)
        if include_embeddings:
            params["embeddings"] = self.vocab_size * self.hidden_size

        # Encoder projection
        params["encoder_projection"] = (
            self.encoder_hidden_size * self.hidden_size  # Linear
            + self.hidden_size  # RMSNorm
        )

        # Per-layer attention params
        gqa_params_per_layer = (
            self.hidden_size * self.hidden_size  # Q proj
            + self.hidden_size * (self.num_kv_heads * self.head_dim)  # K proj
            + self.hidden_size * (self.num_kv_heads * self.head_dim)  # V proj
            + self.hidden_size * self.hidden_size  # O proj
        )
        params["gqa_attention"] = gqa_params_per_layer * self.num_layers

        # Cross attention params
        cross_attn_params_per_layer = 4 * self.hidden_size * self.hidden_size
        params["cross_attention"] = cross_attn_params_per_layer * self.num_layers

        # Dense FFN params
        dense_ffn_per_layer = 3 * self.hidden_size * self.dense_intermediate_size
        params["dense_ffn"] = dense_ffn_per_layer * len(self.dense_layers)

        # MoE FFN params
        router_per_layer = self.hidden_size * self.num_experts
        expert_per_layer = 3 * self.hidden_size * self.expert_intermediate_size
        experts_per_layer = self.num_experts * expert_per_layer
        shared_per_layer = 3 * self.hidden_size * self.shared_expert_intermediate_size if self.use_shared_expert else 0

        params["moe_routers"] = router_per_layer * len(self.moe_layers)
        params["moe_experts"] = experts_per_layer * len(self.moe_layers)
        params["moe_shared_experts"] = shared_per_layer * len(self.moe_layers)

        # RMSNorm params (3 per layer + 1 final)
        params["layer_norms"] = 3 * self.hidden_size * self.num_layers + self.hidden_size

        # Total
        params["total"] = sum(params.values())

        return params

    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (
            f"DecoderMoEConfig("
            f"hidden={self.hidden_size}, "
            f"layers={self.num_layers}, "
            f"heads={self.num_attention_heads}/{self.num_kv_heads}, "
            f"experts={self.num_experts}×{self.expert_intermediate_size}, "
            f"top_k={self.num_experts_per_token}"
            f")"
        )

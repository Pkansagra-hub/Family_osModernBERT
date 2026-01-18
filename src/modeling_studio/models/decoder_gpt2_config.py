"""
Configuration dataclass for GPT-2 Decoder Head.

This module defines the GPT2DecoderConfig dataclass containing all hyperparameters
for the counterfactual generation decoder using GPT-2 architecture.

Design Rationale:
    - GPT-2 Medium (355M params) chosen for edge deployment
    - Pre-trained on 40GB WebText = strong language prior
    - 1024 hidden / 16 heads / 24 layers = ~710MB VRAM
    - Prefix injection connects encoder to decoder (no cross-attention needed)
    - Total VRAM with encoder: ~1GB (fits RTX 5070 8GB easily)

Architecture:
    - GPT-2 Medium base (from Hugging Face)
    - Prefix injection: encoder outputs → projection → prepend to decoder input
    - Optional: freeze early layers to preserve language knowledge

Usage:
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

    # Default configuration (GPT-2 Medium)
    config = GPT2DecoderConfig()

    # Smaller model for memory-constrained environments
    config = GPT2DecoderConfig(
        gpt2_model_name="gpt2",  # GPT-2 Small (117M)
        freeze_layers=6,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class GPT2DecoderConfig:
    """
    Configuration for the GPT-2 based decoder head.

    This uses a pre-trained GPT-2 model with prefix injection to leverage
    encoder outputs for conditional generation. Designed for edge devices.

    Attributes:
        gpt2_model_name: Hugging Face model name or path.
            Options: "gpt2" (117M), "gpt2-medium" (355M), "gpt2-large" (774M)
            Default: "gpt2-medium" for balance of quality and edge efficiency.

        encoder_hidden_size: Size of encoder hidden states (ModernBERT). Default: 768

        projection_hidden_size: GPT-2 hidden size (determined by model). Default: 1024

        use_prefix_injection: Whether to prepend encoder outputs to decoder input.
            Default: True (the main mechanism for encoder-decoder connection).

        num_prefix_tokens: Number of virtual prefix tokens from encoder.
            If None, uses full encoder sequence length. Default: None

        freeze_layers: Number of GPT-2 layers to freeze from bottom.
            Freezing preserves pre-trained language knowledge.
            0 = train all layers, 12 = freeze half of medium. Default: 0

        prefix_projection_layers: Number of MLP layers for encoder projection.
            1 = linear, 2 = MLP with GELU. Default: 1

        dropout: Dropout probability for projection layers. Default: 0.1

        max_position_embeddings: Maximum generation length. Default: 512

        vocab_size: Vocabulary size. Should match tokenizer. Default: 50368

        pad_token_id: Padding token ID. Default: 50283

        bos_token_id: Beginning of sequence token ID. Default: 50281 (CLS)

        eos_token_id: End of sequence token ID. Default: 50282 (SEP)

        tie_word_embeddings: Whether to tie input/output embeddings.
            GPT-2 already ties by default. Default: True

        use_cache: Whether to use KV cache for generation. Default: True

        generation_max_length: Default max tokens for generation. Default: 128

        temperature: Default sampling temperature. Default: 1.0

        top_k: Default top-k for sampling. Default: 50

        top_p: Default nucleus sampling threshold. Default: 0.9

        repetition_penalty: Default repetition penalty. Default: 1.2
    """

    # Model selection
    gpt2_model_name: str = "gpt2-medium"

    # Encoder-decoder connection
    encoder_hidden_size: int = 768
    projection_hidden_size: int = 1024  # GPT-2 medium hidden size
    use_prefix_injection: bool = True
    num_prefix_tokens: int | None = None  # None = use full encoder sequence
    prefix_projection_layers: int = 1

    # Enhanced encoder-decoder coupling (for new training runs)
    use_projection_layer_norm: bool = True  # LayerNorm after projection
    scale_projection_to_gpt2_norm: bool = False  # Scale to match GPT-2 embedding norms
    use_cross_attention: bool = False  # Cross-attention bridge for stronger coupling
    cross_attention_heads: int = 8  # Number of heads for cross-attention
    use_gated_fusion: bool = False  # Gated fusion for dynamic balancing

    # Training configuration
    freeze_layers: int = 0
    dropout: float = 0.1

    # Sequence configuration
    max_position_embeddings: int = 512
    vocab_size: int = 50368

    # Special tokens (matching ModernBERT tokenizer)
    pad_token_id: int = 50283
    bos_token_id: int = 50281  # CLS token
    eos_token_id: int = 50282  # SEP token

    # Model behavior
    tie_word_embeddings: bool = True
    use_cache: bool = True

    # Generation defaults
    generation_max_length: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2

    def __post_init__(self):
        """Set projection_hidden_size based on GPT-2 variant if not explicitly set."""
        # Map model names to hidden sizes
        gpt2_hidden_sizes = {
            "gpt2": 768,
            "gpt2-medium": 1024,
            "gpt2-large": 1280,
            "gpt2-xl": 1600,
        }

        # Extract base model name (handle paths)
        base_name = self.gpt2_model_name.split("/")[-1].lower()

        # Auto-detect hidden size if using standard model
        if base_name in gpt2_hidden_sizes:
            expected_hidden = gpt2_hidden_sizes[base_name]
            if self.projection_hidden_size != expected_hidden:
                logger.debug(
                    f"Setting projection_hidden_size to {expected_hidden} "
                    f"to match {self.gpt2_model_name}"
                )
                # Note: Can't modify in __post_init__ with frozen=False by default
                # This is just for logging; actual hidden size comes from loaded model

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.encoder_hidden_size <= 0:
            errors.append(f"encoder_hidden_size must be positive, got {self.encoder_hidden_size}")

        if self.projection_hidden_size <= 0:
            errors.append(f"projection_hidden_size must be positive, got {self.projection_hidden_size}")

        if self.freeze_layers < 0:
            errors.append(f"freeze_layers must be non-negative, got {self.freeze_layers}")

        if self.prefix_projection_layers < 1:
            errors.append(f"prefix_projection_layers must be >= 1, got {self.prefix_projection_layers}")

        if self.dropout < 0 or self.dropout > 1:
            errors.append(f"dropout must be in [0, 1], got {self.dropout}")

        if self.max_position_embeddings <= 0:
            errors.append(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.debug("GPT2DecoderConfig validation passed")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved GPT2DecoderConfig to {path}")

    @classmethod
    def load(cls, path: str) -> "GPT2DecoderConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls(**data)
        logger.info(f"Loaded GPT2DecoderConfig from {path}")
        return config

    def get_estimated_params(self) -> dict[str, int]:
        """
        Estimate parameter counts for the decoder.

        Returns:
            Dictionary with parameter estimates for each component.
        """
        # GPT-2 parameter estimates (approximate)
        gpt2_params = {
            "gpt2": 117_000_000,
            "gpt2-medium": 345_000_000,
            "gpt2-large": 774_000_000,
            "gpt2-xl": 1_500_000_000,
        }

        base_name = self.gpt2_model_name.split("/")[-1].lower()
        gpt2_total = gpt2_params.get(base_name, 345_000_000)

        # Projection layer params
        if self.prefix_projection_layers == 1:
            projection_params = self.encoder_hidden_size * self.projection_hidden_size
        else:
            # MLP: enc -> hidden -> proj
            hidden = self.projection_hidden_size
            projection_params = (
                self.encoder_hidden_size * hidden +
                hidden * self.projection_hidden_size
            )

        return {
            "gpt2_backbone": gpt2_total,
            "projection": projection_params,
            "total": gpt2_total + projection_params,
        }

    def get_estimated_vram_gb(self) -> float:
        """
        Estimate VRAM usage in GB (fp16/bf16).

        Returns:
            Estimated VRAM in gigabytes for inference.
        """
        params = self.get_estimated_params()["total"]
        # 2 bytes per param in fp16/bf16
        bytes_needed = params * 2
        # Add ~20% overhead for activations and KV cache
        bytes_with_overhead = bytes_needed * 1.2
        return bytes_with_overhead / (1024 ** 3)


# =============================================================================
# Preset Configurations
# =============================================================================


def get_edge_config() -> GPT2DecoderConfig:
    """
    Get configuration optimized for edge deployment.

    Uses GPT-2 Medium with frozen early layers for efficient inference
    while maintaining generation quality.
    """
    return GPT2DecoderConfig(
        gpt2_model_name="gpt2-medium",
        freeze_layers=6,  # Freeze half the layers
        dropout=0.05,  # Lower dropout for inference
    )


def get_small_config() -> GPT2DecoderConfig:
    """
    Get minimal configuration for testing or very constrained devices.

    Uses GPT-2 Small (117M params) for lowest VRAM usage.
    """
    return GPT2DecoderConfig(
        gpt2_model_name="gpt2",
        projection_hidden_size=768,
        freeze_layers=0,
    )


def get_quality_config() -> GPT2DecoderConfig:
    """
    Get configuration optimized for generation quality.

    Uses GPT-2 Large with full training. Requires more VRAM.
    """
    return GPT2DecoderConfig(
        gpt2_model_name="gpt2-large",
        projection_hidden_size=1280,
        freeze_layers=0,
        generation_max_length=256,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "GPT2DecoderConfig",
    "get_edge_config",
    "get_small_config",
    "get_quality_config",
]

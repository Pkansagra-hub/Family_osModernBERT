"""
Hub Token Gradient Masking for ModernBERT v3.

This module implements gradient masking for hub tokens to enable selective training.
Hub token embeddings need special handling since they're part of the word embedding
matrix but may need different training dynamics.

Key Features:
    - Freeze original vocabulary (0-50367) gradients
    - Enable/disable specific hub token gradients
    - Scale hub token gradients for controlled training
    - Register hooks without memory leaks

Hub Token Positions:
    [EMO]:  50368  - Affective understanding
    [MEM]:  50369  - Memory retrieval & storage
    [REL]:  50370  - Relational reasoning
    [TASK]: 50371  - User action classification
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Hub token positions in vocabulary
HUB_TOKEN_POSITIONS: dict[str, int] = {
    "[EMO]": 50368,
    "[MEM]": 50369,
    "[REL]": 50370,
    "[TASK]": 50371,
}

# Vocabulary layout
V2_VOCAB_SIZE = 50368  # Original ModernBERT vocab
HUB_TOKEN_START = 50368
HUB_TOKEN_COUNT = 4
V3_VOCAB_SIZE = 50372  # V2 + hub tokens


@dataclass
class GradientMaskConfig:
    """
    Configuration for gradient masking.

    Attributes:
        train_hub_tokens: List of hub token names to train. None = all.
        freeze_original_vocab: Whether to freeze original vocab (0-50367).
        hub_token_grad_scale: Gradient scaling factor for hub tokens.
    """

    # Which hub tokens to train
    train_hub_tokens: list[str] | None = None
    # Freeze original vocabulary
    freeze_original_vocab: bool = True
    # Hub token gradient scaling
    hub_token_grad_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.train_hub_tokens is None:
            # Default: train all hub tokens
            self.train_hub_tokens = list(HUB_TOKEN_POSITIONS.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "train_hub_tokens": self.train_hub_tokens,
            "freeze_original_vocab": self.freeze_original_vocab,
            "hub_token_grad_scale": self.hub_token_grad_scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GradientMaskConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EmbeddingGradientHook:
    """
    Gradient hook for selective embedding training.

    Applies gradient masking to word embeddings to:
    1. Zero gradients for frozen token positions
    2. Scale gradients for hub tokens
    3. Enable per-token training control

    Attributes:
        embedding_weight: Reference to embedding weight tensor
        config: Gradient mask configuration
        grad_mask: Gradient mask tensor [vocab_size, 1]
        hook_handle: Hook handle for cleanup
    """

    def __init__(
        self,
        embedding_weight: nn.Parameter,
        config: GradientMaskConfig,
    ):
        """
        Initialize EmbeddingGradientHook.

        Args:
            embedding_weight: Word embedding weight [vocab_size, hidden_size]
            config: Gradient mask configuration
        """
        # Use weak reference to avoid keeping model in memory
        self._embedding_weight_ref = weakref.ref(embedding_weight)
        self.config = config
        self.hook_handle: torch.utils.hooks.RemovableHandle | None = None

        # Build gradient mask
        self.grad_mask = self._build_gradient_mask(embedding_weight)

    @property
    def embedding_weight(self) -> nn.Parameter | None:
        """Get embedding weight from weak reference."""
        return self._embedding_weight_ref()

    def _build_gradient_mask(self, embedding_weight: nn.Parameter) -> torch.Tensor:
        """
        Build gradient mask tensor.

        Args:
            embedding_weight: Embedding weight tensor for shape/device info

        Returns:
            Mask [vocab_size, 1] where 0=frozen, >0=trainable
        """
        vocab_size = embedding_weight.shape[0]
        device = embedding_weight.device
        dtype = embedding_weight.dtype

        # Start with all frozen or all trainable
        if self.config.freeze_original_vocab:
            mask = torch.zeros(vocab_size, 1, device=device, dtype=dtype)
        else:
            mask = torch.ones(vocab_size, 1, device=device, dtype=dtype)

        # Set hub token masks
        train_tokens = self.config.train_hub_tokens or []
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < vocab_size:
                if token_name in train_tokens:
                    # Trainable with scaling
                    mask[position] = self.config.hub_token_grad_scale
                else:
                    # Frozen
                    mask[position] = 0.0

        logger.debug(
            f"Built gradient mask: "
            f"original_vocab={'frozen' if self.config.freeze_original_vocab else 'trainable'}, "
            f"hub_tokens={train_tokens}, scale={self.config.hub_token_grad_scale}"
        )

        return mask

    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Hook function applied to gradients.

        Args:
            grad: Gradient tensor [vocab_size, hidden_size]

        Returns:
            Masked gradient tensor
        """
        # Move mask to same device as gradient if needed
        if self.grad_mask.device != grad.device:
            self.grad_mask = self.grad_mask.to(grad.device)

        # Apply mask
        masked_grad = grad * self.grad_mask
        return masked_grad

    def register(self) -> bool:
        """
        Register gradient hook on embedding weight.

        Returns:
            True if registration successful
        """
        embedding_weight = self.embedding_weight
        if embedding_weight is None:
            logger.warning("Embedding weight no longer exists, cannot register hook")
            return False

        if self.hook_handle is not None:
            self.hook_handle.remove()

        self.hook_handle = embedding_weight.register_hook(self._gradient_hook)
        logger.info("Registered embedding gradient hook")
        return True

    def remove(self) -> None:
        """Remove gradient hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("Removed embedding gradient hook")

    def is_registered(self) -> bool:
        """Check if hook is currently registered."""
        return self.hook_handle is not None

    def update_trainable_tokens(self, token_names: list[str]) -> None:
        """
        Update which hub tokens are trainable.

        Args:
            token_names: List of hub token names to train
        """
        self.config.train_hub_tokens = list(token_names)
        embedding_weight = self.embedding_weight
        if embedding_weight is not None:
            self.grad_mask = self._build_gradient_mask(embedding_weight)
        logger.debug(f"Updated trainable tokens: {token_names}")

    def update_grad_scale(self, scale: float) -> None:
        """
        Update gradient scaling for hub tokens.

        Args:
            scale: New gradient scaling factor
        """
        self.config.hub_token_grad_scale = scale
        embedding_weight = self.embedding_weight
        if embedding_weight is not None:
            self.grad_mask = self._build_gradient_mask(embedding_weight)
        logger.debug(f"Updated gradient scale: {scale}")

    def get_mask_stats(self) -> dict[str, Any]:
        """
        Get statistics about the gradient mask.

        Returns:
            Dictionary with mask statistics
        """
        frozen_count = (self.grad_mask == 0).sum().item()
        trainable_count = (self.grad_mask > 0).sum().item()
        total_count = self.grad_mask.shape[0]

        return {
            "total_tokens": total_count,
            "frozen_tokens": frozen_count,
            "trainable_tokens": trainable_count,
            "hub_tokens_trainable": self.config.train_hub_tokens,
            "grad_scale": self.config.hub_token_grad_scale,
        }


class HubTokenGradientManager:
    """
    Manages hub token gradient masking for a model.

    Provides high-level interface for controlling hub token training:
    - Setup/cleanup gradient hooks
    - Freeze/unfreeze specific hub tokens
    - Get hub token gradients and embeddings

    Attributes:
        model: The model to manage
        config: Gradient mask configuration
        hooks: List of registered hooks
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradientMaskConfig | None = None,
    ):
        """
        Initialize HubTokenGradientManager.

        Args:
            model: Model with embeddings.word_embeddings
            config: Gradient mask configuration (default creates one)
        """
        self.model = model
        self.config = config or GradientMaskConfig()
        self.hooks: list[EmbeddingGradientHook] = []
        self._setup_complete = False

    def get_embedding_weight(self) -> nn.Parameter | None:
        """
        Get word embedding weight from model.

        Searches common model architectures for embedding weights.

        Returns:
            Embedding weight parameter or None if not found
        """
        # Try model.embeddings.word_embeddings
        if hasattr(self.model, "embeddings"):
            embeddings = self.model.embeddings
            if hasattr(embeddings, "word_embeddings"):
                return embeddings.word_embeddings.weight
            if hasattr(embeddings, "weight"):
                return embeddings.weight

        # Try model.encoder.embeddings.word_embeddings
        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder
            if hasattr(encoder, "embeddings"):
                embeddings = encoder.embeddings
                if hasattr(embeddings, "word_embeddings"):
                    return embeddings.word_embeddings.weight

        # Try model.model.embeddings
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "embeddings"):
                embeddings = inner.embeddings
                if hasattr(embeddings, "word_embeddings"):
                    return embeddings.word_embeddings.weight

        return None

    def setup(self) -> bool:
        """
        Setup gradient masking for hub tokens.

        Creates and registers gradient hook on embedding weight.

        Returns:
            True if setup successful, False otherwise
        """
        if self._setup_complete:
            logger.warning("Gradient masking already setup, call cleanup() first")
            return True

        embedding_weight = self.get_embedding_weight()

        if embedding_weight is None:
            logger.warning("Could not find embedding weight in model")
            return False

        # Validate vocab size
        vocab_size = embedding_weight.shape[0]
        if vocab_size < V3_VOCAB_SIZE:
            logger.warning(
                f"Vocab size {vocab_size} is less than expected {V3_VOCAB_SIZE}. "
                f"Hub tokens may not exist."
            )

        # Create and register hook
        hook = EmbeddingGradientHook(embedding_weight, self.config)
        if hook.register():
            self.hooks.append(hook)
            self._setup_complete = True
            logger.info("Hub token gradient masking setup complete")
            return True

        return False

    def cleanup(self) -> None:
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._setup_complete = False
        logger.info("Cleaned up all gradient hooks")

    def is_setup(self) -> bool:
        """Check if gradient masking is setup."""
        return self._setup_complete and len(self.hooks) > 0

    def freeze_all_hub_tokens(self) -> None:
        """Freeze all hub token gradients."""
        for hook in self.hooks:
            hook.update_trainable_tokens([])
        logger.info("Froze all hub tokens")

    def unfreeze_all_hub_tokens(self) -> None:
        """Enable gradients for all hub tokens."""
        all_tokens = list(HUB_TOKEN_POSITIONS.keys())
        for hook in self.hooks:
            hook.update_trainable_tokens(all_tokens)
        logger.info("Unfroze all hub tokens")

    def train_specific_hub_tokens(self, token_names: list[str]) -> None:
        """
        Train only specific hub tokens.

        Args:
            token_names: List of hub token names to train (e.g., ["[EMO]", "[MEM]"])
        """
        valid_tokens = [t for t in token_names if t in HUB_TOKEN_POSITIONS]
        invalid_tokens = [t for t in token_names if t not in HUB_TOKEN_POSITIONS]

        if invalid_tokens:
            logger.warning(f"Unknown hub tokens ignored: {invalid_tokens}")

        for hook in self.hooks:
            hook.update_trainable_tokens(valid_tokens)

        logger.info(f"Training hub tokens: {valid_tokens}")

    def set_grad_scale(self, scale: float) -> None:
        """
        Set gradient scaling for hub tokens.

        Args:
            scale: Gradient scaling factor (1.0 = normal, 0.5 = half, etc.)
        """
        for hook in self.hooks:
            hook.update_grad_scale(scale)
        logger.info(f"Set hub token gradient scale: {scale}")

    def get_hub_token_gradients(self) -> dict[str, torch.Tensor | None]:
        """
        Get current gradients for hub tokens.

        Returns:
            Dict mapping token name to gradient tensor (or None if no gradient)
        """
        embedding_weight = self.get_embedding_weight()
        if embedding_weight is None or embedding_weight.grad is None:
            return dict.fromkeys(HUB_TOKEN_POSITIONS, None)

        gradients: dict[str, torch.Tensor | None] = {}
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < embedding_weight.grad.shape[0]:
                gradients[token_name] = embedding_weight.grad[position].clone()
            else:
                gradients[token_name] = None

        return gradients

    def get_hub_token_embeddings(self) -> dict[str, torch.Tensor]:
        """
        Get current hub token embeddings.

        Returns:
            Dict mapping token name to embedding tensor
        """
        embedding_weight = self.get_embedding_weight()
        if embedding_weight is None:
            return {}

        embeddings: dict[str, torch.Tensor] = {}
        for token_name, position in HUB_TOKEN_POSITIONS.items():
            if position < embedding_weight.shape[0]:
                embeddings[token_name] = embedding_weight[position].clone().detach()

        return embeddings

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about gradient masking.

        Returns:
            Dictionary with configuration and mask statistics
        """
        stats = {
            "is_setup": self._setup_complete,
            "num_hooks": len(self.hooks),
            "config": self.config.to_dict(),
        }

        if self.hooks:
            stats["mask_stats"] = self.hooks[0].get_mask_stats()

        embedding_weight = self.get_embedding_weight()
        if embedding_weight is not None:
            stats["vocab_size"] = embedding_weight.shape[0]
            stats["embedding_dim"] = embedding_weight.shape[1]

        return stats


def setup_hub_token_gradient_masking(
    model: nn.Module,
    train_hub_tokens: list[str] | None = None,
    freeze_original_vocab: bool = True,
    hub_token_grad_scale: float = 1.0,
) -> HubTokenGradientManager:
    """
    Setup hub token gradient masking for a model.

    This is a convenience function that creates a HubTokenGradientManager
    and sets up gradient masking in one call.

    Args:
        model: ModernBERTv3 model
        train_hub_tokens: Which hub tokens to train (None = all)
        freeze_original_vocab: Whether to freeze original vocab embeddings
        hub_token_grad_scale: Gradient scaling for hub tokens

    Returns:
        Configured and setup HubTokenGradientManager

    Example:
        manager = setup_hub_token_gradient_masking(
            model,
            train_hub_tokens=["[EMO]", "[TASK]"],
            freeze_original_vocab=True,
            hub_token_grad_scale=0.5,
        )

        # Later, cleanup when done
        manager.cleanup()
    """
    config = GradientMaskConfig(
        train_hub_tokens=train_hub_tokens,
        freeze_original_vocab=freeze_original_vocab,
        hub_token_grad_scale=hub_token_grad_scale,
    )

    manager = HubTokenGradientManager(model, config)
    manager.setup()

    return manager


def get_hub_token_positions() -> dict[str, int]:
    """
    Get hub token positions in vocabulary.

    Returns:
        Dict mapping token name to position
    """
    return HUB_TOKEN_POSITIONS.copy()


def get_vocab_layout() -> dict[str, int]:
    """
    Get vocabulary layout constants.

    Returns:
        Dict with V2_VOCAB_SIZE, HUB_TOKEN_START, HUB_TOKEN_COUNT, V3_VOCAB_SIZE
    """
    return {
        "V2_VOCAB_SIZE": V2_VOCAB_SIZE,
        "HUB_TOKEN_START": HUB_TOKEN_START,
        "HUB_TOKEN_COUNT": HUB_TOKEN_COUNT,
        "V3_VOCAB_SIZE": V3_VOCAB_SIZE,
    }

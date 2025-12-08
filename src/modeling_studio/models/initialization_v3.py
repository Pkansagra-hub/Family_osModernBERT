"""
ModernBERT v3 Initialization from v2 Checkpoints.

This module implements function-preserving growth by transferring weights from
ModernBERT v2 (22 layers) to v3 (28 layers). The initialization strategy:

1. **Direct Copy (L1-22)**: v2 layers 1-22 → v3 layers 1-22
   - Foundation Band (L1-6): window=64
   - Context Band (L7-18): window=128
   - Semantic Band (L19-22): window=256
   - Preserves exact v2 functionality

2. **Cloning (L23-28)**: v2 layers 15-20 → v3 layers 23-28
   - Family Band (L23-28): window=512
   - Uses mature mid/late-layer weights
   - Optional noise addition to break symmetry

3. **Embedding Transfer**: v2 vocab → v3 vocab + 4 hub tokens
   - Transfers 50,368 v2 token embeddings
   - Creates 4 hub token slots (positions 50368-50371)
   - Hub tokens initialized via semantic centroids

4. **Hub Token Initialization**: Semantic centroid from related tokens
   - [EMO]: emotion-related words
   - [MEM]: memory-related words
   - [REL]: relation-related words
   - [TASK]: intent/task-related words

Key Classes:
    - V2CheckpointInfo: Metadata about v2 checkpoint
    - WeightTransferStats: Statistics from weight transfer
    - V2CheckpointLoader: Robust checkpoint loading with validation
    - LayerCopier: Direct copy L1-22 (Issue 4.1.2)
    - LayerCloner: Clone L23-28 from L15-20 (Issue 4.1.3)
    - EmbeddingTransfer: Vocab + hub token slots (Issue 4.1.4)
    - HubTokenSemanticInitializer: Semantic init (Issue 4.1.5)

Functions:
    - load_v2_checkpoint: Factory for V2CheckpointLoader
    - initialize_from_v2: Complete v2→v3 initialization

Author: FamilyOS Team
Date: December 2025
Version: 3.3
Epic: 4.1 Weight Transfer
Issue: 4.1.1 - v2 Checkpoint Loader
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class V2CheckpointInfo:
    """
    Information about a v2 checkpoint.

    Attributes:
        path: Path to checkpoint file
        num_layers: Number of encoder layers (should be 22 for v2)
        hidden_size: Hidden dimension (should be 768)
        vocab_size: Vocabulary size (should be 50368 for ModernBERT-base)
        has_pooler: Whether checkpoint includes pooler weights
        has_task_heads: Whether checkpoint includes task-specific heads
        state_dict_keys: All keys in the state dict

    Example:
        >>> loader = V2CheckpointLoader("path/to/checkpoint.pt")
        >>> info = loader.get_info()
        >>> print(f"Layers: {info.num_layers}, Vocab: {info.vocab_size}")
        Layers: 22, Vocab: 50368
    """

    path: Path
    num_layers: int  # Should be 22
    hidden_size: int
    vocab_size: int
    has_pooler: bool
    has_task_heads: bool
    state_dict_keys: list[str]


@dataclass
class WeightTransferStats:
    """
    Statistics from weight transfer operation.

    Attributes:
        total_params: Total parameters in v3 model
        transferred_params: Parameters copied from v2
        initialized_params: New parameters (hub tokens, cloned layers)
        skipped_params: Parameters not transferred (e.g., pooler, heads)
        layer_mapping: Mapping v3_layer → v2_layer

    Example:
        >>> stats = initialize_from_v2(v3_model, "v2_checkpoint.pt")
        >>> print(f"Transferred: {stats.transferred_params:,}")
        >>> print(f"Initialized: {stats.initialized_params:,}")
    """

    total_params: int
    transferred_params: int
    initialized_params: int  # New params (hub tokens, new layers)
    skipped_params: int
    layer_mapping: dict[int, int]  # v3_layer -> v2_layer


# ══════════════════════════════════════════════════════════════════════════════
# V2 Checkpoint Loader
# ══════════════════════════════════════════════════════════════════════════════


class V2CheckpointLoader:
    """
    Loads and parses ModernBERT v2 checkpoints.

    v2 Architecture (22 layers):
        - Foundation Band: L1-6 (window=64)
        - Core Band: L7-18 (window=128)
        - Family Band: L19-22 (window=256)

    v3 Architecture (28 layers):
        - Foundation Band: L1-6 (window=64) ← COPY from v2 L1-6
        - Core Band: L7-18 (window=128) ← COPY from v2 L7-18
        - semantic Band: L19-22 (window=256) ← COPY from v2 L19-22
        - Family Band: L23-28 (window=512) ← CLONE from v2 L15-20

    Features:
        - Handles multiple checkpoint formats (state_dict, model, etc.)
        - Cleans DDP module. prefixes
        - Extracts metadata (layers, hidden_size, vocab_size)
        - Validates v2 compatibility
        - Per-layer and embedding weight extraction

    Args:
        checkpoint_path: Path to v2 checkpoint file

    Example:
        >>> loader = V2CheckpointLoader("checkpoints/modernbert-v2.pt")
        >>> is_valid, issues = loader.validate()
        >>> if is_valid:
        ...     state_dict = loader.load()
        ...     layer_0_weights = loader.get_layer_weights(0)
    """

    V2_NUM_LAYERS = 22
    V3_NUM_LAYERS = 28

    # Layer mapping: v3_layer -> v2_layer (0-indexed)
    LAYER_MAPPING = {
        # Foundation Band: Direct copy
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        # Core Band: Direct copy
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        # semantic Band: Direct copy from v2 Family Band
        18: 18,
        19: 19,
        20: 20,
        21: 21,
        # Family Band: Clone from v2 Core/Family layers 15-20
        22: 14,  # L23 ← L15 (0-indexed: 22 ← 14)
        23: 15,  # L24 ← L16
        24: 16,  # L25 ← L17
        25: 17,  # L26 ← L18
        26: 18,  # L27 ← L19
        27: 19,  # L28 ← L20
    }

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint loader.

        Args:
            checkpoint_path: Path to v2 checkpoint file (.pt, .pth, .bin, .safetensors)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self._state_dict: dict[str, torch.Tensor] | None = None
        self._info: V2CheckpointInfo | None = None

    def load(self) -> dict[str, torch.Tensor]:
        """
        Load checkpoint state dict.

        Handles multiple checkpoint formats:
        - {"state_dict": {...}} - Trainer/Lightning format
        - {"model_state_dict": {...}} - PyTorch save format
        - {"model": {...}} - Custom format
        - {...} - Direct state dict

        Also cleans 'module.' prefix from DDP checkpoints.

        Returns:
            State dict with tensor weights

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> state_dict = loader.load()
            >>> print(list(state_dict.keys())[:3])
            ['embeddings.word_embeddings.weight', 'encoder.layers.0.attention.q_proj.weight', ...]
        """
        if self._state_dict is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}\n"
                    f"Please ensure the v2 checkpoint file exists at this path."
                )

            try:
                # Handle safetensors format
                if self.checkpoint_path.suffix == ".safetensors":
                    checkpoint = {}
                    with safe_open(self.checkpoint_path, framework="pt") as f:
                        for key in f.keys():
                            checkpoint[key] = f.get_tensor(key)
                else:
                    # Standard PyTorch format
                    checkpoint = torch.load(
                        self.checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint {self.checkpoint_path}: {e}\n"
                    f"Ensure the file is a valid PyTorch or safetensors checkpoint."
                ) from e

            # Handle different checkpoint formats (for .pt/.pth files)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self._state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                self._state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                self._state_dict = checkpoint["model"]
            else:
                # Assume checkpoint is already a state dict
                self._state_dict = checkpoint

            # Clean up module. prefix if present
            if self._state_dict is not None:
                self._state_dict = self._clean_state_dict(self._state_dict)
                # Normalize key names to expected format
                self._state_dict = self._normalize_keys(self._state_dict)

                logger.info(
                    f"Loaded checkpoint with {len(self._state_dict)} keys from {self.checkpoint_path.name}"
                )

        if self._state_dict is None:
            raise RuntimeError("Failed to load state dict from checkpoint")

        return self._state_dict

    def _clean_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Remove 'module.' prefix from DDP checkpoints.

        PyTorch's DistributedDataParallel (DDP) wraps the model and adds
        'module.' prefix to all parameter names. This function removes
        that prefix for compatibility with non-DDP models.

        Args:
            state_dict: Original state dict

        Returns:
            Cleaned state dict without 'module.' prefix

        Example:
            >>> state_dict = {"module.embeddings.weight": tensor(...)}
            >>> cleaned = loader._clean_state_dict(state_dict)
            >>> print(list(cleaned.keys()))
            ['embeddings.weight']
        """
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]  # Remove 'module.' prefix
            cleaned[key] = value

        if len(cleaned) < len(state_dict):
            logger.debug(f"Cleaned 'module.' prefix from {len(state_dict) - len(cleaned)} keys")

        return cleaned

    def _normalize_keys(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Normalize key names from v2 checkpoint format to v3 model format.

        Handles the actual ModernBERT v2 checkpoint format:
        - encoder.embeddings.tok_embeddings -> embeddings.word_embeddings
        - encoder.embeddings.norm -> embeddings.LayerNorm
        - encoder.final_norm -> final_layer_norm
        - encoder.layers.X.* -> encoder.layers.X.* (keep as-is, layer internals match)

        The v2 checkpoint structure (from safetensors):
        - encoder.embeddings.tok_embeddings.weight [50368, 768]
        - encoder.embeddings.norm.weight [768]
        - encoder.layers.X.attn.Wqkv.weight [2304, 768]
        - encoder.layers.X.attn.Wo.weight [768, 768]
        - encoder.layers.X.attn_norm.weight [768] (except layer 0)
        - encoder.layers.X.mlp.Wi.weight [2304, 768]
        - encoder.layers.X.mlp.Wo.weight [768, 1152]
        - encoder.layers.X.mlp_norm.weight [768]
        - encoder.final_norm.weight [768]

        The v3 model structure:
        - embeddings.word_embeddings.weight [50432, 768]
        - embeddings.LayerNorm.weight [768]
        - embeddings.LayerNorm.bias [768]
        - encoder.layers.X.attn.Wqkv.weight [2304, 768]
        - encoder.layers.X.attn.Wo.weight [768, 768]
        - encoder.layers.X.attn_norm.weight [768]
        - encoder.layers.X.mlp.Wi.weight [2304, 768]
        - encoder.layers.X.mlp.Wo.weight [768, 1152]
        - encoder.layers.X.mlp_norm.weight [768]
        - final_layer_norm.weight [768]
        - final_layer_norm.bias [768]

        Args:
            state_dict: State dict with v2 key names

        Returns:
            State dict with normalized v3 key names

        Example:
            >>> state_dict = {"encoder.embeddings.tok_embeddings.weight": tensor(...)}
            >>> normalized = loader._normalize_keys(state_dict)
            >>> print(list(normalized.keys()))
            ['embeddings.word_embeddings.weight']
        """
        normalized = {}
        key_mappings = 0

        for key, value in state_dict.items():
            new_key = key

            # Embedding keys (encoder.embeddings.* -> embeddings.*)
            if key.startswith("encoder.embeddings.tok_embeddings"):
                new_key = key.replace(
                    "encoder.embeddings.tok_embeddings",
                    "embeddings.word_embeddings",
                )
            elif key.startswith("encoder.embeddings.norm"):
                new_key = key.replace(
                    "encoder.embeddings.norm",
                    "embeddings.LayerNorm",
                )

            # Final norm (encoder.final_norm -> final_layer_norm)
            elif key.startswith("encoder.final_norm"):
                new_key = key.replace("encoder.final_norm", "final_layer_norm")

            # Layer keys: encoder.layers.X.* stays the same
            # The internal structure (attn.Wqkv, mlp.Wi, etc.) already matches v3

            if new_key != key:
                key_mappings += 1
                logger.debug(f"Key mapping: {key} -> {new_key}")

            normalized[new_key] = value

        if key_mappings > 0:
            logger.info(f"Normalized {key_mappings} key names from v2 to v3 format")

        return normalized

    def get_info(self) -> V2CheckpointInfo:
        """
        Extract checkpoint metadata.

        Analyzes the state dict to determine:
        - Number of encoder layers
        - Hidden dimension
        - Vocabulary size
        - Presence of pooler/task heads

        Returns:
            V2CheckpointInfo with extracted metadata

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> info = loader.get_info()
            >>> print(f"{info.num_layers} layers, vocab={info.vocab_size}")
            22 layers, vocab=50368
        """
        if self._info is None:
            state_dict = self.load()

            # Detect layer count
            layer_pattern = re.compile(r"encoder\.layers\.(\d+)\.")
            layer_indices = set()
            for key in state_dict.keys():
                match = layer_pattern.search(key)
                if match:
                    layer_indices.add(int(match.group(1)))

            num_layers = max(layer_indices) + 1 if layer_indices else 0

            # Detect hidden size from first layer norm or attn_norm
            hidden_size = 768  # default
            for key, tensor in state_dict.items():
                if ("layernorm" in key.lower() or "attn_norm" in key.lower()) and tensor.dim() == 1:
                    hidden_size = tensor.shape[0]
                    break

            # Detect vocab size from embeddings
            vocab_size = 50368  # default v2 vocab
            for key, tensor in state_dict.items():
                if "word_embeddings" in key and tensor.dim() == 2:
                    vocab_size = tensor.shape[0]
                    break

            # Check for pooler and task heads
            has_pooler = any("pooler" in k for k in state_dict.keys())
            has_task_heads = any("head" in k.lower() for k in state_dict.keys())

            self._info = V2CheckpointInfo(
                path=self.checkpoint_path,
                num_layers=num_layers,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                has_pooler=has_pooler,
                has_task_heads=has_task_heads,
                state_dict_keys=list(state_dict.keys()),
            )

            logger.debug(
                f"Detected checkpoint: {num_layers} layers, "
                f"hidden_size={hidden_size}, vocab_size={vocab_size}"
            )

        return self._info

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate checkpoint is compatible with v2→v3 transfer.

        Checks:
        - Number of layers == 22
        - Hidden size == 768
        - Required keys present (embeddings, layers)

        Returns:
            (is_valid, list of issues)

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> is_valid, issues = loader.validate()
            >>> if not is_valid:
            ...     print("Validation issues:", issues)
        """
        info = self.get_info()
        issues = []

        if info.num_layers != self.V2_NUM_LAYERS:
            issues.append(f"Expected {self.V2_NUM_LAYERS} layers, found {info.num_layers}")

        if info.hidden_size != 768:
            issues.append(f"Expected hidden_size=768, found {info.hidden_size}")

        # Check for required keys
        required_patterns = [
            "embeddings.word_embeddings",
            "encoder.layers.0",
            f"encoder.layers.{self.V2_NUM_LAYERS - 1}",  # Last v2 layer
        ]

        state_dict = self.load()
        for pattern in required_patterns:
            if not any(pattern in k for k in state_dict.keys()):
                issues.append(f"Missing required pattern: {pattern}")

        if issues:
            logger.warning(f"Checkpoint validation failed with {len(issues)} issues")
        else:
            logger.info("✓ Checkpoint validation passed")

        return len(issues) == 0, issues

    def get_layer_weights(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """
        Get all weights for a specific layer.

        Args:
            layer_idx: v2 layer index (0-21, 0-indexed)

        Returns:
            Dict of weight name → tensor (with prefix removed)

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> layer_0 = loader.get_layer_weights(0)
            >>> print(layer_0.keys())
            dict_keys(['attention.q_proj.weight', 'attention.k_proj.weight', ...])
        """
        if layer_idx < 0 or layer_idx >= self.V2_NUM_LAYERS:
            raise ValueError(
                f"Invalid layer index {layer_idx}. "
                f"v2 has {self.V2_NUM_LAYERS} layers (0-{self.V2_NUM_LAYERS-1})"
            )

        state_dict = self.load()
        prefix = f"encoder.layers.{layer_idx}."

        layer_weights = {}
        for key, tensor in state_dict.items():
            if key.startswith(prefix):
                # Remove prefix for cleaner mapping
                short_key = key[len(prefix) :]
                layer_weights[short_key] = tensor

        if not layer_weights:
            logger.warning(f"No weights found for layer {layer_idx}")

        return layer_weights

    def get_embedding_weights(self) -> dict[str, torch.Tensor]:
        """
        Get embedding layer weights.

        Returns:
            Dict of embedding weight name → tensor (with prefix removed)

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> embeddings = loader.get_embedding_weights()
            >>> print(embeddings.keys())
            dict_keys(['norm.weight', 'tok_embeddings.weight', ...])
        """
        state_dict = self.load()
        embedding_weights = {}

        # Try both possible prefixes for embeddings
        prefixes = ["encoder.embeddings.", "embeddings."]

        for key, tensor in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    short_key = key[len(prefix) :]
                    embedding_weights[short_key] = tensor
                    break

        if not embedding_weights:
            logger.warning("No embedding weights found in checkpoint")

        return embedding_weights

    def get_encoder_weights(self) -> dict[str, torch.Tensor]:
        """
        Get encoder-level weights (embedding norm, final norm).

        Returns:
            Dict of encoder weight name → tensor

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> encoder_weights = loader.get_encoder_weights()
            >>> print(encoder_weights.keys())
            dict_keys(['embeddings.norm.weight', 'final_norm.weight'])
        """
        state_dict = self.load()
        encoder_weights = {}

        for key, tensor in state_dict.items():
            if key.startswith("encoder.") and not key.startswith("encoder.layers."):
                short_key = key[len("encoder.") :]
                encoder_weights[short_key] = tensor

        return encoder_weights

    def print_summary(self) -> None:
        """
        Print checkpoint summary.

        Displays:
        - Path
        - Number of layers
        - Hidden size
        - Vocabulary size
        - Pooler/task head presence
        - Total keys

        Example:
            >>> loader = V2CheckpointLoader("checkpoint.pt")
            >>> loader.print_summary()
            ════════════════════════════════════════════════════════════
            📦 v2 Checkpoint Summary
            ════════════════════════════════════════════════════════════
              Path: checkpoint.pt
              Layers: 22
              Hidden Size: 768
              Vocab Size: 50368
              Has Pooler: True
              Has Task Heads: True
              Total Keys: 198
            ════════════════════════════════════════════════════════════
        """
        info = self.get_info()
        print("\n" + "=" * 60)
        print("📦 v2 Checkpoint Summary")
        print("=" * 60)
        print(f"  Path: {info.path.name}")
        print(f"  Layers: {info.num_layers}")
        print(f"  Hidden Size: {info.hidden_size}")
        print(f"  Vocab Size: {info.vocab_size}")
        print(f"  Has Pooler: {info.has_pooler}")
        print(f"  Has Task Heads: {info.has_task_heads}")
        print(f"  Total Keys: {len(info.state_dict_keys)}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Factory Function
# ══════════════════════════════════════════════════════════════════════════════


def load_v2_checkpoint(path: str) -> V2CheckpointLoader:
    """
    Factory function to load v2 checkpoint.

    Convenience function that creates a loader, validates it,
    and prints a summary.

    Args:
        path: Path to checkpoint file

    Returns:
        Configured loader

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint validation fails critically

    Example:
        >>> loader = load_v2_checkpoint("checkpoints/modernbert-v2.pt")
        📦 v2 Checkpoint Summary
        ════════════════════════════════════════════════════════════
          Path: modernbert-v2.pt
          Layers: 22
          ...
        ════════════════════════════════════════════════════════════
        >>> state_dict = loader.load()
    """
    loader = V2CheckpointLoader(path)

    # Validate
    is_valid, issues = loader.validate()
    if not is_valid:
        logger.warning(f"Checkpoint validation issues: {issues}")
        # Don't fail - allow loading anyway for debugging

    loader.print_summary()
    return loader


# ══════════════════════════════════════════════════════════════════════════════
# Layer Copying (Issue 4.1.2)
# ══════════════════════════════════════════════════════════════════════════════


class LayerCopier:
    """
    Copies layer weights from v2 to v3 (direct 1:1 mapping).

    Implements Issue 4.1.2: Layer 1-22 Direct Copy

    Layer Mapping (v3 ← v2):
        - L1-6 (Foundation) ← L1-6: Direct copy (window 64)
        - L7-18 (Core) ← L7-18: Direct copy (window 128)
        - L19-22 (semantic) ← L19-22: Direct copy (window 256)

    This preserves all learned representations from v2 exactly,
    ensuring function-preserving growth.

    Attributes:
        v2_loader: Checkpoint loader for source weights
        strict: If True, warn about missing/mismatched weights
        copy_stats: Dict tracking matched/mismatched/missing weights

    Example:
        >>> loader = V2CheckpointLoader("v2_checkpoint.pt")
        >>> copier = LayerCopier(loader, strict=True)
        >>> copied = copier.copy_layers_1_to_22(v3_encoder)
        >>> print(f"Copied {copied:,} parameters")
        Copied 85,123,456 parameters
    """

    def __init__(
        self,
        v2_loader: V2CheckpointLoader,
        strict: bool = True,
    ):
        """
        Initialize layer copier.

        Args:
            v2_loader: Loader for v2 checkpoint
            strict: If True, warn about weight mismatches
        """
        self.v2_loader = v2_loader
        self.strict = strict
        self.copy_stats = {
            "matched": 0,
            "mismatched_shape": 0,
            "missing_in_v2": 0,
        }
        # Cache embedding norm for layer 0's attn_norm
        self._embedding_norm: Optional[torch.Tensor] = None

    def _get_embedding_norm(self) -> Optional[torch.Tensor]:
        """Get embedding norm weight for layer 0's attn_norm."""
        if self._embedding_norm is None:
            encoder_weights = self.v2_loader.get_encoder_weights()
            if "embeddings.norm.weight" in encoder_weights:
                self._embedding_norm = encoder_weights["embeddings.norm.weight"]
        return self._embedding_norm

    def copy_layer(
        self,
        v3_layer,
        v2_layer_idx: int,
        v3_layer_idx: int,
    ) -> int:
        """
        Copy weights from v2 layer to v3 layer.

        Performs direct state dict copy with shape validation.
        Preserves parameter values exactly (no modifications).

        Special case: Layer 0 has no attn_norm in ModernBERT checkpoint.
        We copy the embedding norm to layer 0's attn_norm instead.

        Args:
            v3_layer: Target v3 layer module (nn.Module)
            v2_layer_idx: Source layer index in v2 (0-indexed)
            v3_layer_idx: Target layer index in v3 (0-indexed, for logging)

        Returns:
            Number of parameters copied

        Example:
            >>> v3_layer = v3_encoder.layers[0]
            >>> copied = copier.copy_layer(v3_layer, v2_idx=0, v3_idx=0)
            >>> print(f"Layer 0: Copied {copied:,} params")
            Layer 0: Copied 3,890,304 params
        """
        v2_weights = self.v2_loader.get_layer_weights(v2_layer_idx)
        copied_params = 0

        v3_state = v3_layer.state_dict()

        for v3_key in v3_state.keys():
            # Map v3 key to v2 key (same structure expected)
            v2_key = v3_key

            if v2_key not in v2_weights:
                # Special case: Layer 0's attn_norm comes from embedding norm
                if v2_layer_idx == 0 and v3_key == "attn_norm.weight":
                    embed_norm = self._get_embedding_norm()
                    if embed_norm is not None:
                        v3_state[v3_key] = embed_norm.clone()
                        copied_params += embed_norm.numel()
                        self.copy_stats["matched"] += 1
                        logger.info(f"Layer 0: Copied embedding norm to attn_norm")
                        continue

                if self.strict:
                    logger.warning(f"Layer {v3_layer_idx}: Missing v2 weight for '{v3_key}'")
                self.copy_stats["missing_in_v2"] += 1
                continue

            v2_tensor = v2_weights[v2_key]
            v3_tensor = v3_state[v3_key]

            if v2_tensor.shape != v3_tensor.shape:
                logger.warning(
                    f"Layer {v3_layer_idx}: Shape mismatch for '{v3_key}': "
                    f"v2={v2_tensor.shape}, v3={v3_tensor.shape}"
                )
                self.copy_stats["mismatched_shape"] += 1
                continue

            # Copy weight (exact preservation)
            v3_state[v3_key] = v2_tensor.clone()
            copied_params += v2_tensor.numel()
            self.copy_stats["matched"] += 1

        # Load updated state into v3 layer
        v3_layer.load_state_dict(v3_state)

        return copied_params

    def copy_layers_1_to_22(
        self,
        v3_encoder,
    ) -> int:
        """
        Copy all v2 layers 0-21 to v3 layers 0-21 (direct 1:1 copy).

        This is the core of Issue 4.1.2, copying the first 22 layers
        from v2 to v3 to preserve learned representations.

        Args:
            v3_encoder: v3 encoder module with .layers attribute

        Returns:
            Total parameters copied

        Example:
            >>> copier = LayerCopier(loader)
            >>> total = copier.copy_layers_1_to_22(v3_model.encoder)
            >>> print(f"Total copied: {total:,}")
              Layer 0: Copied 3,890,304 params from v2 L0
              Layer 1: Copied 3,890,304 params from v2 L1
              ...
              Layer 21: Copied 3,890,304 params from v2 L21
            Total copied: 85,586,688
        """
        total_copied = 0

        for v3_idx in range(22):  # Layers 0-21 (v3 L1-22 in 1-indexed)
            v2_idx = v3_idx  # Direct 1:1 mapping

            v3_layer = v3_encoder.layers[v3_idx]
            copied = self.copy_layer(v3_layer, v2_idx, v3_idx)
            total_copied += copied

            logger.info(f"  Layer {v3_idx}: Copied {copied:,} params from v2 L{v2_idx}")

        return total_copied

    def get_stats(self) -> dict[str, int]:
        """
        Get copy statistics.

        Returns:
            Dict with 'matched', 'mismatched_shape', 'missing_in_v2' counts

        Example:
            >>> stats = copier.get_stats()
            >>> print(f"Matched: {stats['matched']}, Errors: {stats['mismatched_shape']}")
            Matched: 528, Errors: 0
        """
        return self.copy_stats.copy()


def copy_layers_direct(
    v3_model,
    v2_checkpoint_path: str,
) -> int:
    """
    Copy v2 layers 1-22 directly to v3 layers 1-22.

    Main entry point for Issue 4.1.2: Layer 1-22 Direct Copy.

    This function:
    1. Loads v2 checkpoint
    2. Creates LayerCopier
    3. Copies all 22 layers with 1:1 mapping
    4. Reports statistics

    Args:
        v3_model: Target v3 model (ModernBERTv3Ultra or similar)
        v2_checkpoint_path: Path to v2 checkpoint file

    Returns:
        Number of parameters copied

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> v3_model = ModernBERTv3Ultra(config)
        >>> copied = copy_layers_direct(v3_model, "checkpoints/modernbert-v2.pt")
        🔄 Copying v2 Layers 1-22 to v3 Layers 1-22...
          Layer 0: Copied 3,890,304 params from v2 L0
          ...
          Layer 21: Copied 3,890,304 params from v2 L21
        ✓ Direct copy complete:
          - Matched: 528
          - Shape mismatches: 0
          - Missing in v2: 0
          - Total params: 85,586,688
        >>> print(f"Success! Copied {copied:,} parameters")
        Success! Copied 85,586,688 parameters
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    copier = LayerCopier(loader)

    print("\n🔄 Copying v2 Layers 1-22 to v3 Layers 1-22...")

    # Get encoder from model
    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model

    total_copied = copier.copy_layers_1_to_22(encoder)

    stats = copier.get_stats()
    print(f"\n✓ Direct copy complete:")
    print(f"  - Matched: {stats['matched']}")
    print(f"  - Shape mismatches: {stats['mismatched_shape']}")
    print(f"  - Missing in v2: {stats['missing_in_v2']}")
    print(f"  - Total params: {total_copied:,}")

    return total_copied


# ══════════════════════════════════════════════════════════════════════════════
# Layer Cloning (Issue 4.1.3)
# ══════════════════════════════════════════════════════════════════════════════


class LayerCloner:
    """
    Clones layer weights from v2 to new v3 layers.

    Implements Issue 4.1.3: Layer 23-28 Cloning from L15-20

    Clone Mapping (v3 ← v2):
        L23 ← L15: First Family Band layer
        L24 ← L16: Second layer
        L25 ← L17: Third layer
        L26 ← L18: Fourth layer
        L27 ← L19: Fifth layer
        L28 ← L20: Sixth layer

    Why L15-20?
        - L15-18: Late Core Band - good general representations
        - L19-20: Early Family Band - task-relevant features
        - Together: balanced mix of general + specialized

    This provides strong initialization for the new Family Band layers
    with proven representations from v2, enabling "function preserving growth."

    Attributes:
        v2_loader: Checkpoint loader for source weights
        add_noise: If True, add small noise to break symmetry
        noise_std: Standard deviation of noise (default 0.01)
        clone_stats: Dict tracking cloned/noise_added counts

    Example:
        >>> loader = V2CheckpointLoader("v2_checkpoint.pt")
        >>> cloner = LayerCloner(loader, add_noise=True, noise_std=0.01)
        >>> cloned = cloner.clone_layers_23_to_28(v3_encoder)
        >>> print(f"Cloned {cloned:,} parameters with noise")
        Cloned 23,345,664 parameters with noise
    """

    # Clone mapping: v3_layer_idx -> v2_layer_idx (0-indexed)
    CLONE_MAPPING = {
        22: 14,  # L23 ← L15 (0-indexed: 22 ← 14)
        23: 15,  # L24 ← L16
        24: 16,  # L25 ← L17
        25: 17,  # L26 ← L18
        26: 18,  # L27 ← L19
        27: 19,  # L28 ← L20
    }

    def __init__(
        self,
        v2_loader: V2CheckpointLoader,
        add_noise: bool = False,
        noise_std: float = 0.01,
    ):
        """
        Initialize layer cloner.

        Args:
            v2_loader: Loader for v2 checkpoint
            add_noise: If True, add small noise to cloned weights to break symmetry
            noise_std: Standard deviation of Gaussian noise (default 0.01)
        """
        self.v2_loader = v2_loader
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.clone_stats = {
            "cloned": 0,
            "noise_added": 0,
            "missing_in_v2": 0,
            "shape_mismatch": 0,
        }

    def clone_layer(
        self,
        v3_layer,
        v2_layer_idx: int,
        v3_layer_idx: int,
    ) -> int:
        """
        Clone weights from v2 layer to v3 layer.

        Optionally adds small noise to break symmetry between
        cloned layers (helps them specialize during training).

        Noise is ONLY added to weight matrices (.weight suffix),
        NOT to biases or LayerNorm parameters.

        Args:
            v3_layer: Target v3 layer module (nn.Module)
            v2_layer_idx: Source layer index in v2 (0-indexed)
            v3_layer_idx: Target layer index in v3 (0-indexed, for logging)

        Returns:
            Number of parameters cloned

        Example:
            >>> v3_layer = v3_encoder.layers[22]
            >>> cloned = cloner.clone_layer(v3_layer, v2_idx=14, v3_idx=22)
            >>> print(f"Layer 22: Cloned {cloned:,} params from v2 L14")
            Layer 22: Cloned 3,890,304 params from v2 L14
        """
        v2_weights = self.v2_loader.get_layer_weights(v2_layer_idx)
        cloned_params = 0

        v3_state = v3_layer.state_dict()

        for v3_key in v3_state.keys():
            # Assume same key structure between v2 and v3 layers
            v2_key = v3_key

            if v2_key not in v2_weights:
                logger.warning(
                    f"Layer {v3_layer_idx}: No v2 weight for '{v3_key}', " "using random init"
                )
                self.clone_stats["missing_in_v2"] += 1
                continue

            v2_tensor = v2_weights[v2_key]
            v3_tensor = v3_state[v3_key]

            if v2_tensor.shape != v3_tensor.shape:
                logger.warning(
                    f"Layer {v3_layer_idx}: Shape mismatch for '{v3_key}': "
                    f"v2={v2_tensor.shape}, v3={v3_tensor.shape}"
                )
                self.clone_stats["shape_mismatch"] += 1
                continue

            # Clone with optional noise
            cloned = v2_tensor.clone()

            # Add noise only to weight matrices, not biases or LayerNorm
            if self.add_noise and v3_key.endswith(".weight"):
                # Check it's not a LayerNorm weight (1D tensor)
                if cloned.dim() > 1:
                    noise = torch.randn_like(cloned) * self.noise_std
                    cloned = cloned + noise
                    self.clone_stats["noise_added"] += 1

            v3_state[v3_key] = cloned
            cloned_params += cloned.numel()
            self.clone_stats["cloned"] += 1

        # Load updated state into v3 layer
        v3_layer.load_state_dict(v3_state)

        return cloned_params

    def clone_layers_23_to_28(
        self,
        v3_encoder,
    ) -> int:
        """
        Clone v2 layers 15-20 to v3 layers 23-28.

        This is the core of Issue 4.1.3, cloning 6 layers from v2
        to initialize the new Family Band in v3.

        Args:
            v3_encoder: v3 encoder module with .layers attribute

        Returns:
            Total parameters cloned

        Example:
            >>> cloner = LayerCloner(loader, add_noise=True)
            >>> total = cloner.clone_layers_23_to_28(v3_model.encoder)
            🧬 Cloning v2 Layers 15-20 to v3 Layers 23-28...
              Layer 22: Cloned 3,890,304 params from v2 L14 (+noise)
              Layer 23: Cloned 3,890,304 params from v2 L15 (+noise)
              ...
              Layer 27: Cloned 3,890,304 params from v2 L19 (+noise)
            >>> print(f"Total cloned: {total:,}")
            Total cloned: 23,341,824
        """
        total_cloned = 0

        print("\n🧬 Cloning v2 Layers 15-20 to v3 Layers 23-28...")

        for v3_idx, v2_idx in self.CLONE_MAPPING.items():
            v3_layer = v3_encoder.layers[v3_idx]
            cloned = self.clone_layer(v3_layer, v2_idx, v3_idx)
            total_cloned += cloned

            noise_str = " (+noise)" if self.add_noise else ""
            logger.info(f"  Layer {v3_idx}: Cloned {cloned:,} params from v2 L{v2_idx}{noise_str}")
            print(f"  Layer {v3_idx}: Cloned {cloned:,} params from v2 L{v2_idx}{noise_str}")

        return total_cloned

    def get_stats(self) -> dict[str, int]:
        """
        Get clone statistics.

        Returns:
            Dict with 'cloned', 'noise_added', 'missing_in_v2', 'shape_mismatch' counts

        Example:
            >>> stats = cloner.get_stats()
            >>> print(f"Cloned: {stats['cloned']}, Noise: {stats['noise_added']}")
            Cloned: 72, Noise: 24
        """
        return self.clone_stats.copy()


def clone_layers_for_growth(
    v3_model,
    v2_checkpoint_path: str,
    add_noise: bool = True,
    noise_std: float = 0.01,
) -> int:
    """
    Clone v2 layers 15-20 to v3 layers 23-28.

    Main entry point for Issue 4.1.3: Layer 23-28 Cloning from L15-20.

    This function:
    1. Loads v2 checkpoint
    2. Creates LayerCloner with noise settings
    3. Clones 6 layers (L15-20 → L23-28)
    4. Reports statistics

    Args:
        v3_model: Target v3 model (ModernBERTv3Ultra or similar)
        v2_checkpoint_path: Path to v2 checkpoint file
        add_noise: Add small noise to break symmetry (recommended True)
        noise_std: Standard deviation of noise (default 0.01)

    Returns:
        Number of parameters cloned

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> v3_model = ModernBERTv3Ultra(config)
        >>> cloned = clone_layers_for_growth(v3_model, "checkpoints/v2.pt")
        🧬 Cloning v2 Layers 15-20 to v3 Layers 23-28...
          Layer 22: Cloned 3,890,304 params from v2 L14 (+noise)
          ...
        ✓ Layer cloning complete:
          - Cloned weights: 72
          - Noise added to: 24 tensors
          - Total params: 23,341,824
        >>> print(f"Success! Cloned {cloned:,} parameters")
        Success! Cloned 23,341,824 parameters
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    cloner = LayerCloner(loader, add_noise=add_noise, noise_std=noise_std)

    # Get encoder from model
    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model

    total_cloned = cloner.clone_layers_23_to_28(encoder)

    stats = cloner.get_stats()
    print("\n✓ Layer cloning complete:")
    print(f"  - Cloned weights: {stats['cloned']}")
    print(f"  - Noise added to: {stats['noise_added']} tensors")
    print(f"  - Missing in v2: {stats['missing_in_v2']}")
    print(f"  - Shape mismatches: {stats['shape_mismatch']}")
    print(f"  - Total params: {total_cloned:,}")

    return total_cloned


# ══════════════════════════════════════════════════════════════════════════════
# Layer Band Configuration
# ══════════════════════════════════════════════════════════════════════════════


# Layer band configuration for v3
# Exported for use in training configuration
V3_LAYER_BANDS: dict[str, list[int]] = {
    "foundation": list(range(0, 6)),  # L1-6: window=64, frozen in Phase 1
    "core": list(range(6, 18)),  # L7-18: window=128, frozen in Phase 1
    "semantic": list(range(18, 22)),  # L19-22: window=256, trainable
    "family": list(range(22, 28)),  # L23-28: window=512, LoRA trainable
}


def get_clone_source_for_layer(v3_layer_idx: int) -> int | None:
    """
    Get the v2 layer that was cloned to create this v3 layer.

    Only layers 22-27 (v3 L23-28) are cloned from v2 L14-19.
    Layers 0-21 are direct copies.

    Args:
        v3_layer_idx: v3 layer index (0-indexed)

    Returns:
        v2 layer index that was cloned, or None if direct copy

    Example:
        >>> get_clone_source_for_layer(22)  # L23
        14
        >>> get_clone_source_for_layer(10)  # L11 (direct copy)
        None
    """
    return LayerCloner.CLONE_MAPPING.get(v3_layer_idx)


def get_band_for_layer(v3_layer_idx: int) -> str:
    """
    Get the band name for a v3 layer index.

    Args:
        v3_layer_idx: v3 layer index (0-indexed)

    Returns:
        Band name: 'foundation', 'core', 'semantic', or 'family'

    Raises:
        ValueError: If layer index is out of range

    Example:
        >>> get_band_for_layer(0)
        'foundation'
        >>> get_band_for_layer(10)
        'core'
        >>> get_band_for_layer(22)
        'family'
    """
    for band_name, layer_indices in V3_LAYER_BANDS.items():
        if v3_layer_idx in layer_indices:
            return band_name
    raise ValueError(f"Layer {v3_layer_idx} is not in any band (valid: 0-27)")


def get_layers_in_band(band_name: str) -> list[int]:
    """
    Get all layer indices in a band.

    Args:
        band_name: One of 'foundation', 'core', 'semantic', 'family'

    Returns:
        List of layer indices (0-indexed)

    Raises:
        ValueError: If band name is invalid

    Example:
        >>> get_layers_in_band('family')
        [22, 23, 24, 25, 26, 27]
    """
    if band_name not in V3_LAYER_BANDS:
        raise ValueError(f"Unknown band '{band_name}'. Valid: {list(V3_LAYER_BANDS.keys())}")
    return V3_LAYER_BANDS[band_name].copy()


def print_layer_band_summary() -> None:
    """
    Print summary of v3 layer bands.

    Example output:
        ══════════════════════════════════════════════════════════════
        📊 v3 Layer Band Configuration
        ══════════════════════════════════════════════════════════════
        Foundation (L1-6):   [0, 1, 2, 3, 4, 5]      window=64
        Core (L7-18):        [6, 7, ..., 17]         window=128
        semantic (L19-22):     [18, 19, 20, 21]        window=256
        Family (L23-28):     [22, 23, 24, 25, 26, 27] window=512
        ══════════════════════════════════════════════════════════════
    """
    window_sizes = {
        "foundation": 64,
        "core": 128,
        "semantic": 256,
        "family": 512,
    }

    print("\n" + "=" * 60)
    print("📊 v3 Layer Band Configuration")
    print("=" * 60)

    for band_name, layers in V3_LAYER_BANDS.items():
        window = window_sizes[band_name]
        layer_range = f"L{layers[0]+1}-{layers[-1]+1}"
        print(f"  {band_name.capitalize():12} ({layer_range:7}): {layers}  window={window}")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Transfer (Issue 4.1.4)
# ══════════════════════════════════════════════════════════════════════════════


class EmbeddingTransfer:
    """
    Transfers embeddings from v2 to v3 with hub token slot creation.

    Implements Issue 4.1.4: Embedding Transfer with Hub Token Slots

    v2 Vocabulary: 50,368 tokens (ModernBERT-base)
    v3 Vocabulary: 50,372 tokens (+4 hub tokens)

    Hub Token Positions (added at end of vocab):
        [EMO] = 50368
        [MEM] = 50369
        [REL] = 50370
        [TASK] = 50371

    Embedding Layout:
        v2: [vocab_embeddings: 50368]
        v3: [vocab_embeddings: 50368, hub_tokens: 4]

    The hub token embeddings are left uninitialized (Issue 4.1.5 handles
    semantic initialization using seed words).

    Attributes:
        v2_loader: Checkpoint loader for source embeddings
        transfer_stats: Dict tracking transfer statistics

    Example:
        >>> loader = V2CheckpointLoader("v2_checkpoint.pt")
        >>> transfer = EmbeddingTransfer(loader)
        >>> total = transfer.transfer_all(v3_model.embeddings)
        📝 Transferring Embeddings (with Hub Token Slots)...
          Transferred 50,368 vocab embeddings, created 4 hub token slots
          Transferred 8,192 position embeddings
          Transferred embedding LayerNorm: 1,536 params
        ✓ Embedding transfer complete: 39,168,000 params
    """

    V2_VOCAB_SIZE = 50368
    NUM_HUB_TOKENS = 4
    V3_VOCAB_SIZE = V2_VOCAB_SIZE + NUM_HUB_TOKENS

    def __init__(self, v2_loader: V2CheckpointLoader):
        """
        Initialize embedding transfer.

        Args:
            v2_loader: Loader for v2 checkpoint
        """
        self.v2_loader = v2_loader
        self.transfer_stats = {
            "vocab_transferred": 0,
            "hub_slots_created": 0,
            "position_embeddings_transferred": 0,
            "layer_norm_transferred": 0,
        }

    def transfer_word_embeddings(
        self,
        v3_embeddings,
    ) -> int:
        """
        Transfer word embeddings from v2, creating hub token slots.

        Copies all 50,368 v2 token embeddings to v3, leaving positions
        50368-50371 for the 4 hub tokens ([EMO], [MEM], [REL], [TASK]).

        Args:
            v3_embeddings: v3 embedding module with word_embeddings

        Returns:
            Number of parameters transferred

        Raises:
            ValueError: If v2 checkpoint missing embeddings or v3 has wrong size

        Example:
            >>> transfer.transfer_word_embeddings(v3_model.embeddings)
            38,682,624
        """
        v2_emb_weights = self.v2_loader.get_embedding_weights()

        if "word_embeddings.weight" not in v2_emb_weights:
            raise ValueError("v2 checkpoint missing word_embeddings.weight")

        v2_word_emb = v2_emb_weights["word_embeddings.weight"]
        v2_vocab_size, hidden_size = v2_word_emb.shape

        # Verify expected size
        if v2_vocab_size != self.V2_VOCAB_SIZE:
            logger.warning(
                f"Unexpected v2 vocab size: {v2_vocab_size} " f"(expected {self.V2_VOCAB_SIZE})"
            )

        # Get v3 word embeddings - handle different attribute structures
        v3_word_emb = self._get_word_embedding_weight(v3_embeddings)
        v3_vocab_size = v3_word_emb.shape[0]

        # Allow flexible v3 vocab size (could be larger if more tokens added)
        if v3_vocab_size < v2_vocab_size:
            raise ValueError(f"v3 vocab size ({v3_vocab_size}) smaller than v2 ({v2_vocab_size})")

        # Copy v2 vocab embeddings to v3 (first v2_vocab_size positions)
        with torch.no_grad():
            v3_word_emb[:v2_vocab_size] = v2_word_emb.clone()

        self.transfer_stats["vocab_transferred"] = v2_vocab_size * hidden_size
        self.transfer_stats["hub_slots_created"] = self.NUM_HUB_TOKENS

        logger.info(
            f"  Transferred {v2_vocab_size:,} vocab embeddings, "
            f"created {self.NUM_HUB_TOKENS} hub token slots"
        )
        print(
            f"  Transferred {v2_vocab_size:,} vocab embeddings, "
            f"created {self.NUM_HUB_TOKENS} hub token slots"
        )

        return self.transfer_stats["vocab_transferred"]

    def _get_word_embedding_weight(self, v3_embeddings) -> torch.Tensor:
        """
        Get word embedding weight tensor from various embedding module structures.

        Handles different embedding module structures:
        - v3_embeddings.word_embeddings.weight (standard)
        - v3_embeddings.tok_embeddings.weight (alternative)
        - v3_embeddings.weight (direct embedding layer)

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Word embedding weight tensor

        Raises:
            ValueError: If word embeddings not found
        """
        if hasattr(v3_embeddings, "word_embeddings"):
            return v3_embeddings.word_embeddings.weight
        elif hasattr(v3_embeddings, "tok_embeddings"):
            return v3_embeddings.tok_embeddings.weight
        elif hasattr(v3_embeddings, "weight"):
            return v3_embeddings.weight
        else:
            raise ValueError(
                "Cannot find word embeddings in v3_embeddings. "
                "Expected attributes: word_embeddings, tok_embeddings, or weight"
            )

    def transfer_position_embeddings(
        self,
        v3_embeddings,
    ) -> int:
        """
        Transfer position embeddings from v2 to v3.

        v3 supports up to 8192 positions. If v2 has fewer positions,
        the extra v3 positions are left randomly initialized.

        Note: If model uses RoPE (Rotary Position Embeddings), position
        embeddings may not exist and this is handled gracefully.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Number of parameters transferred (0 if using RoPE)

        Example:
            >>> transfer.transfer_position_embeddings(v3_model.embeddings)
            6,291,456  # 8192 * 768
        """
        v2_emb_weights = self.v2_loader.get_embedding_weights()

        # Check for position embeddings (may not exist if using RoPE)
        if "position_embeddings.weight" not in v2_emb_weights:
            logger.info("  No position embeddings in v2 (using RoPE)")
            print("  No position embeddings in v2 (using RoPE)")
            return 0

        v2_pos_emb = v2_emb_weights["position_embeddings.weight"]
        v2_max_pos, hidden_size = v2_pos_emb.shape

        # Get v3 position embeddings
        if not hasattr(v3_embeddings, "position_embeddings"):
            logger.info("  v3 uses RoPE, skipping position embedding transfer")
            print("  v3 uses RoPE, skipping position embedding transfer")
            return 0

        v3_pos_emb = v3_embeddings.position_embeddings.weight
        v3_max_pos = v3_pos_emb.shape[0]

        # Copy up to min of both sizes
        copy_length = min(v2_max_pos, v3_max_pos)

        with torch.no_grad():
            v3_embeddings.position_embeddings.weight[:copy_length] = v2_pos_emb[
                :copy_length
            ].clone()

        self.transfer_stats["position_embeddings_transferred"] = copy_length * hidden_size

        logger.info(f"  Transferred {copy_length:,} position embeddings")
        print(f"  Transferred {copy_length:,} position embeddings")

        return self.transfer_stats["position_embeddings_transferred"]

    def transfer_layer_norm(
        self,
        v3_embeddings,
    ) -> int:
        """
        Transfer embedding LayerNorm from v2.

        Transfers both weight and bias of the embedding LayerNorm.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Number of parameters transferred

        Example:
            >>> transfer.transfer_layer_norm(v3_model.embeddings)
            1536  # 768 * 2 (weight + bias)
        """
        v2_emb_weights = self.v2_loader.get_embedding_weights()
        transferred = 0

        # Try different LayerNorm attribute names
        v3_ln = self._get_layer_norm(v3_embeddings)
        if v3_ln is None:
            logger.info("  No embedding LayerNorm found in v3")
            return 0

        # LayerNorm weight
        if "LayerNorm.weight" in v2_emb_weights:
            with torch.no_grad():
                v3_ln.weight.copy_(v2_emb_weights["LayerNorm.weight"])
            transferred += v2_emb_weights["LayerNorm.weight"].numel()

        # LayerNorm bias
        if "LayerNorm.bias" in v2_emb_weights:
            with torch.no_grad():
                v3_ln.bias.copy_(v2_emb_weights["LayerNorm.bias"])
            transferred += v2_emb_weights["LayerNorm.bias"].numel()

        self.transfer_stats["layer_norm_transferred"] = transferred

        if transferred > 0:
            logger.info(f"  Transferred embedding LayerNorm: {transferred:,} params")
            print(f"  Transferred embedding LayerNorm: {transferred:,} params")

        return transferred

    def _get_layer_norm(self, v3_embeddings):
        """
        Get LayerNorm from various embedding module structures.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            LayerNorm module or None if not found
        """
        if hasattr(v3_embeddings, "LayerNorm"):
            return v3_embeddings.LayerNorm
        elif hasattr(v3_embeddings, "layer_norm"):
            return v3_embeddings.layer_norm
        elif hasattr(v3_embeddings, "norm"):
            return v3_embeddings.norm
        return None

    def transfer_all(
        self,
        v3_embeddings,
    ) -> int:
        """
        Transfer all embedding components from v2 to v3.

        Transfers:
        1. Word embeddings (50,368 tokens)
        2. Position embeddings (if not using RoPE)
        3. LayerNorm (weight + bias)

        Hub token slots (positions 50368-50371) are left uninitialized.
        Issue 4.1.5 handles semantic initialization using seed words.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Total parameters transferred

        Example:
            >>> transfer.transfer_all(v3_model.embeddings)
            📝 Transferring Embeddings (with Hub Token Slots)...
              Transferred 50,368 vocab embeddings, created 4 hub token slots
              Transferred 8,192 position embeddings
              Transferred embedding LayerNorm: 1,536 params
            ✓ Embedding transfer complete: 44,975,616 params
        """
        total = 0

        print("\n📝 Transferring Embeddings (with Hub Token Slots)...")

        total += self.transfer_word_embeddings(v3_embeddings)
        total += self.transfer_position_embeddings(v3_embeddings)
        total += self.transfer_layer_norm(v3_embeddings)

        print(f"\n✓ Embedding transfer complete: {total:,} params")

        return total

    def get_stats(self) -> dict[str, int]:
        """
        Get transfer statistics.

        Returns:
            Dict with 'vocab_transferred', 'hub_slots_created',
            'position_embeddings_transferred', 'layer_norm_transferred' counts

        Example:
            >>> stats = transfer.get_stats()
            >>> print(f"Vocab: {stats['vocab_transferred']:,}")
            Vocab: 38,682,624
        """
        return self.transfer_stats.copy()


def transfer_embeddings(
    v3_model,
    v2_checkpoint_path: str,
) -> int:
    """
    Transfer embeddings from v2 to v3 with hub token slots.

    Main entry point for Issue 4.1.4: Embedding Transfer with Hub Token Slots.

    This function:
    1. Loads v2 checkpoint
    2. Creates EmbeddingTransfer
    3. Transfers all embedding components
    4. Leaves hub token slots uninitialized (for Issue 4.1.5)

    Args:
        v3_model: Target v3 model (or its embeddings module)
        v2_checkpoint_path: Path to v2 checkpoint file

    Returns:
        Number of parameters transferred

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> v3_model = ModernBERTv3Ultra(config)
        >>> transferred = transfer_embeddings(v3_model, "checkpoints/v2.pt")
        📝 Transferring Embeddings (with Hub Token Slots)...
          Transferred 50,368 vocab embeddings, created 4 hub token slots
          Transferred 8,192 position embeddings
          Transferred embedding LayerNorm: 1,536 params
        ✓ Embedding transfer complete: 44,975,616 params
        >>> print(f"Success! Transferred {transferred:,} parameters")
        Success! Transferred 44,975,616 parameters
    """
    loader = V2CheckpointLoader(v2_checkpoint_path)
    transfer = EmbeddingTransfer(loader)

    # Get embeddings module from model
    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model

    return transfer.transfer_all(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# Hub Token Semantic Initialization (Issue 4.1.5)
# ══════════════════════════════════════════════════════════════════════════════


class HubTokenSemanticInitializer:
    """
    Initialize hub token embeddings with semantic meaning.

    Implements Issue 4.1.5: Hub Token Semantic Initialization

    Strategy: Average embeddings of semantically related tokens to create
    a meaningful starting point for each hub token.

    Hub Token Initialization:
        [EMO] ← avg("emotion", "feeling", "mood", "sentiment", "affect", ...)
        [MEM] ← avg("memory", "remember", "recall", "history", "context", ...)
        [REL] ← avg("relation", "relationship", "connection", "link", ...)
        [TASK] ← avg("task", "intent", "action", "goal", "purpose", ...)

    The semantic initialization provides each hub with a starting point
    that reflects its intended capability, rather than random initialization.

    Attributes:
        tokenizer: HuggingFace tokenizer for token ID lookup
        fallback_std: Standard deviation for random init (fallback)
        init_stats: Statistics about initialization per hub

    Example:
        >>> from transformers import AutoTokenizer
        >>> initializer = HubTokenSemanticInitializer()
        >>> initializer.initialize_all_hubs(v3_model.embeddings)
        🎯 Initializing Hub Token Embeddings (Semantic)...
        ✓ Hub Token Initialization Summary:
        --------------------------------------------------
          [EMO]: semantic_avg (8 seeds: emotion, feeling, mood...)
          [MEM]: semantic_avg (7 seeds: memory, remember, recall...)
          [REL]: semantic_avg (9 seeds: relation, relationship, connection...)
          [TASK]: semantic_avg (6 seeds: task, intent, action...)
        --------------------------------------------------
    """

    # Seed tokens for each hub - chosen to be semantically related
    # and likely to be single tokens in ModernBERT's vocabulary
    HUB_SEED_TOKENS = {
        "[EMO]": [
            "emotion",
            "feeling",
            "mood",
            "sentiment",
            "affect",
            "happy",
            "sad",
            "angry",
            "fear",
            "joy",
            "surprise",
        ],
        "[MEM]": [
            "memory",
            "remember",
            "recall",
            "history",
            "context",
            "past",
            "experience",
            "store",
            "retrieve",
            "knowledge",
        ],
        "[REL]": [
            "relation",
            "relationship",
            "connection",
            "link",
            "between",
            "entail",
            "contradict",
            "similar",
            "compare",
            "associate",
        ],
        "[TASK]": [
            "task",
            "intent",
            "action",
            "goal",
            "purpose",
            "do",
            "request",
            "question",
            "command",
            "want",
        ],
    }

    # Hub token vocab positions (appended after v2 vocab)
    HUB_POSITIONS = {
        "[EMO]": 50368,
        "[MEM]": 50369,
        "[REL]": 50370,
        "[TASK]": 50371,
    }

    def __init__(
        self,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        fallback_std: float = 0.02,
    ):
        """
        Initialize the hub token semantic initializer.

        Args:
            tokenizer_name: HuggingFace tokenizer to use for token ID lookup.
                Default is ModernBERT-base which has the same vocabulary.
            fallback_std: Standard deviation for random initialization if
                no valid seed tokens are found. Default 0.02 for small init.

        Note:
            Requires transformers library for AutoTokenizer.
        """
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            raise ImportError(
                "HubTokenSemanticInitializer requires the transformers library. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{tokenizer_name}': {e}")

        self.fallback_std = fallback_std
        self.init_stats: dict[str, dict] = {}

    def get_seed_token_ids(self, hub_name: str) -> list[int]:
        """
        Get token IDs for seed tokens.

        Only returns IDs for tokens that:
        1. Exist in the vocabulary
        2. Tokenize to exactly one token (not split into subwords)

        Args:
            hub_name: Hub token name (e.g., "[EMO]")

        Returns:
            List of valid token IDs

        Example:
            >>> initializer = HubTokenSemanticInitializer()
            >>> ids = initializer.get_seed_token_ids("[EMO]")
            >>> print(f"Found {len(ids)} valid seed tokens")
            Found 8 valid seed tokens
        """
        seed_tokens = self.HUB_SEED_TOKENS.get(hub_name, [])
        valid_ids = []

        for token in seed_tokens:
            # Tokenize without special tokens
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)

            # Only use single-token words (not split into subwords)
            if len(token_ids) == 1:
                valid_ids.append(token_ids[0])

        return valid_ids

    def initialize_hub_token(
        self,
        word_embeddings,
        hub_name: str,
    ) -> torch.Tensor:
        """
        Initialize a single hub token embedding.

        If valid seed tokens are found, averages their embeddings.
        Otherwise falls back to random initialization with small std.

        Args:
            word_embeddings: Embedding layer with .weight attribute
            hub_name: Hub token name (e.g., "[EMO]")

        Returns:
            Initialized embedding vector (1D tensor of size hidden_dim)

        Example:
            >>> init_emb = initializer.initialize_hub_token(
            ...     v3_model.embeddings.word_embeddings, "[EMO]"
            ... )
            >>> print(init_emb.shape)
            torch.Size([768])
        """
        seed_ids = self.get_seed_token_ids(hub_name)

        if len(seed_ids) == 0:
            # Fallback to random initialization with small std
            logger.warning(f"  {hub_name}: No valid seed tokens, using random init")
            hidden_size = word_embeddings.weight.shape[1]
            init_emb = torch.randn(hidden_size, device=word_embeddings.weight.device)
            init_emb = init_emb * self.fallback_std
            self.init_stats[hub_name] = {"method": "random", "seeds": 0}
            return init_emb

        # Average seed token embeddings
        with torch.no_grad():
            seed_embeddings = word_embeddings.weight[seed_ids]
            avg_embedding = seed_embeddings.mean(dim=0)

        # Store stats
        self.init_stats[hub_name] = {
            "method": "semantic_avg",
            "seeds": len(seed_ids),
            "seed_tokens": [self.tokenizer.decode([sid]) for sid in seed_ids[:5]],
        }

        logger.info(f"  {hub_name}: Initialized from {len(seed_ids)} seed tokens")

        return avg_embedding

    def initialize_all_hubs(
        self,
        v3_embeddings,
    ) -> int:
        """
        Initialize all 4 hub token embeddings.

        Updates the word embedding matrix in-place with semantically
        initialized vectors for positions 50368-50371.

        Args:
            v3_embeddings: v3 embedding module with word_embeddings attribute

        Returns:
            Number of hub tokens initialized

        Example:
            >>> num_initialized = initializer.initialize_all_hubs(v3_model.embeddings)
            🎯 Initializing Hub Token Embeddings (Semantic)...
            ✓ Hub Token Initialization Summary:
            --------------------------------------------------
              [EMO]: semantic_avg (8 seeds: emotion, feeling, mood...)
            ...
            >>> print(f"Initialized {num_initialized} hub tokens")
            Initialized 4 hub tokens
        """
        print("\n🎯 Initializing Hub Token Embeddings (Semantic)...")

        # Get word embeddings - handle different structures
        word_emb = self._get_word_embeddings(v3_embeddings)

        initialized_count = 0
        for hub_name, hub_position in self.HUB_POSITIONS.items():
            init_emb = self.initialize_hub_token(word_emb, hub_name)

            with torch.no_grad():
                word_emb.weight[hub_position] = init_emb

            initialized_count += 1

        self._print_summary()
        return initialized_count

    def _get_word_embeddings(self, v3_embeddings):
        """
        Get word embedding layer from various embedding module structures.

        Args:
            v3_embeddings: v3 embedding module

        Returns:
            Word embedding layer (nn.Embedding or similar)

        Raises:
            ValueError: If word embeddings not found
        """
        if hasattr(v3_embeddings, "word_embeddings"):
            return v3_embeddings.word_embeddings
        elif hasattr(v3_embeddings, "tok_embeddings"):
            return v3_embeddings.tok_embeddings
        elif hasattr(v3_embeddings, "weight"):
            # It's the embedding layer itself
            return v3_embeddings
        else:
            raise ValueError(
                "Cannot find word embeddings in v3_embeddings. "
                "Expected attributes: word_embeddings, tok_embeddings, or weight"
            )

    def _print_summary(self) -> None:
        """Print initialization summary."""
        print("\n✓ Hub Token Initialization Summary:")
        print("-" * 50)

        for hub, stats in self.init_stats.items():
            method = stats["method"]
            if method == "semantic_avg":
                seeds = stats["seeds"]
                examples = stats.get("seed_tokens", [])[:3]
                print(f"  {hub}: {method} ({seeds} seeds: {', '.join(examples)}...)")
            else:
                print(f"  {hub}: {method}")

        print("-" * 50)

    def get_stats(self) -> dict[str, dict]:
        """
        Get initialization statistics.

        Returns:
            Dict mapping hub name to stats dict with 'method', 'seeds',
            and optionally 'seed_tokens' keys.

        Example:
            >>> stats = initializer.get_stats()
            >>> print(f"[EMO] used {stats['[EMO]']['seeds']} seeds")
            [EMO] used 8 seeds
        """
        return self.init_stats.copy()


def initialize_hub_tokens_semantic(
    v3_model,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
) -> int:
    """
    Initialize hub token embeddings with semantic meaning.

    Convenience function for Issue 4.1.5.

    Creates a HubTokenSemanticInitializer and initializes all 4 hub tokens
    using semantic averaging of related seed words.

    Args:
        v3_model: Target v3 model (or its embeddings module)
        tokenizer_name: HuggingFace tokenizer for seed token lookup.
            Default is ModernBERT-base.

    Returns:
        Number of hub tokens initialized (should be 4)

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> v3_model = ModernBERTv3Ultra(config)
        >>> num = initialize_hub_tokens_semantic(v3_model)
        🎯 Initializing Hub Token Embeddings (Semantic)...
        ✓ Hub Token Initialization Summary:
        --------------------------------------------------
          [EMO]: semantic_avg (8 seeds: emotion, feeling, mood...)
          [MEM]: semantic_avg (7 seeds: memory, remember, recall...)
          [REL]: semantic_avg (9 seeds: relation, relationship, connection...)
          [TASK]: semantic_avg (6 seeds: task, intent, action...)
        --------------------------------------------------
        >>> print(f"Initialized {num} hub tokens")
        Initialized 4 hub tokens
    """
    initializer = HubTokenSemanticInitializer(tokenizer_name)

    # Get embeddings module from model
    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model

    return initializer.initialize_all_hubs(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# Main Initialization Function
# ══════════════════════════════════════════════════════════════════════════════


def initialize_from_v2(
    v3_model,
    v2_checkpoint_path: str,
    add_clone_noise: bool = True,
    clone_noise_std: float = 0.01,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
) -> WeightTransferStats:
    """
    Complete initialization of v3 model from v2 checkpoint.

    This is the main orchestration function that performs all steps of
    v2→v3 weight transfer as specified in Epic 4.1.

    Steps:
        1. Load and validate v2 checkpoint (Issue 4.1.1)
        2. Copy layers 1-22 directly (Issue 4.1.2)
        3. Clone layers 15-20 to layers 23-28 (Issue 4.1.3)
        4. Transfer embeddings with hub token slots (Issue 4.1.4)
        5. Initialize hub tokens semantically (Issue 4.1.5)

    Args:
        v3_model: Target v3 model to initialize
        v2_checkpoint_path: Path to v2 checkpoint file
        add_clone_noise: Add noise to cloned layers to break symmetry.
            Default True. Set False for exact reproduction.
        clone_noise_std: Standard deviation of clone noise. Default 0.01.
        tokenizer_name: Tokenizer for hub semantic initialization.
            Default is ModernBERT-base.

    Returns:
        WeightTransferStats with transfer details

    Raises:
        FileNotFoundError: If v2 checkpoint not found
        ValueError: If v2 checkpoint is invalid

    Example:
        >>> from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
        >>> from modeling_studio.models.config_v3 import ModernBERTv3Config
        >>>
        >>> config = ModernBERTv3Config(num_hidden_layers=28)
        >>> v3_model = ModernBERTv3Ultra(config)
        >>>
        >>> stats = initialize_from_v2(
        ...     v3_model,
        ...     "checkpoints/modernbert-v2/pytorch_model.bin",
        ...     add_clone_noise=True,
        ... )
        ═══════════════════════════════════════════════════════════════════════
        🚀 ModernBERT v2 → v3 Weight Transfer
        ═══════════════════════════════════════════════════════════════════════
        [checkpoint info]
        🔄 Copying v2 Layers 1-22 to v3...
        🧬 Cloning v2 Layers 15-20 to v3 Layers 23-28...
        📝 Transferring Embeddings (with Hub Token Slots)...
        🎯 Initializing Hub Token Embeddings (Semantic)...
        ═══════════════════════════════════════════════════════════════════════
        ✅ Weight Transfer Complete!
        ═══════════════════════════════════════════════════════════════════════
        >>>
        >>> print(f"Transferred: {stats.transferred_params:,} params")
        Transferred: 85,000,000 params
    """
    print("\n" + "=" * 70)
    print("🚀 ModernBERT v2 → v3 Weight Transfer")
    print("=" * 70)

    # Step 1: Load and validate v2 checkpoint
    loader = V2CheckpointLoader(v2_checkpoint_path)
    is_valid, issues = loader.validate()
    if not is_valid:
        print(f"⚠️  Checkpoint issues: {issues}")
    loader.print_summary()

    # Step 2: Copy layers 1-22 directly
    copier = LayerCopier(loader)
    encoder = v3_model.encoder if hasattr(v3_model, "encoder") else v3_model
    direct_copied = copier.copy_layers_1_to_22(encoder)

    # Step 3: Clone layers 15-20 to layers 23-28
    cloner = LayerCloner(loader, add_noise=add_clone_noise, noise_std=clone_noise_std)
    cloned = cloner.clone_layers_23_to_28(encoder)

    # Step 4: Transfer embeddings with hub token slots
    embeddings = v3_model.embeddings if hasattr(v3_model, "embeddings") else v3_model
    emb_transfer = EmbeddingTransfer(loader)
    emb_transferred = emb_transfer.transfer_all(embeddings)

    # Step 5: Initialize hub tokens semantically
    hub_init = HubTokenSemanticInitializer(tokenizer_name)
    hub_init.initialize_all_hubs(embeddings)

    # Create stats
    total_params = (
        v3_model.num_parameters
        if hasattr(v3_model, "num_parameters")
        else sum(p.numel() for p in v3_model.parameters())
    )

    stats = WeightTransferStats(
        total_params=total_params,
        transferred_params=direct_copied + emb_transferred,
        initialized_params=cloned,
        skipped_params=0,
        layer_mapping=LayerCloner.CLONE_MAPPING,
    )

    print("\n" + "=" * 70)
    print("✅ Weight Transfer Complete!")
    print("=" * 70)
    print(f"  Total v3 params: {stats.total_params:,}")
    print(f"  Direct transferred: {stats.transferred_params:,}")
    print(f"  Cloned (new layers): {stats.initialized_params:,}")
    print("=" * 70)

    return stats

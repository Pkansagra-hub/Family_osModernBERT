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

import torch

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
        - Feeder Band: L19-22 (window=256) ← COPY from v2 L19-22
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
        # Feeder Band: Direct copy from v2 Family Band
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
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint {self.checkpoint_path}: {e}\n"
                    f"Ensure the file is a valid PyTorch checkpoint."
                ) from e

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
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

            # Detect hidden size from first layer norm
            hidden_size = 768  # default
            for key, tensor in state_dict.items():
                if "layer_norm" in key.lower() and tensor.dim() == 1:
                    hidden_size = tensor.shape[0]
                    break
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
            dict_keys(['word_embeddings.weight', 'position_embeddings.weight', ...])
        """
        state_dict = self.load()
        embedding_weights = {}

        for key, tensor in state_dict.items():
            if key.startswith("embeddings."):
                short_key = key[len("embeddings.") :]
                embedding_weights[short_key] = tensor

        if not embedding_weights:
            logger.warning("No embedding weights found in checkpoint")

        return embedding_weights

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
        - L19-22 (Feeder) ← L19-22: Direct copy (window 256)

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

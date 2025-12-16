"""
Counterfactual Dataset for Stage C Decoder Training.

This module provides dataset classes for loading counterfactual training data
with precomputed encoder embeddings from HDF5 files or live encoding.

Classes:
    - CounterfactualDataset: Main dataset class supporting precomputed and live modes

Output Structure Expected (from prepare_decoder_training_data.py):
    data/counterfactual/training/
    ├── samples.jsonl              # Text data with sample_id
    ├── embeddings.h5              # HDF5 with (N, 768) float16 embeddings (pooled)
    ├── sequence_embeddings.h5     # HDF5 with (total_tokens, 768) + offsets (full seq)
    ├── manifest.json              # Metadata and statistics
    └── train_val_split.json       # 95/5 train/val split indices

Usage:
    from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

    # Precomputed mode (recommended for training)
    dataset = CounterfactualDataset(
        data_dir="data/counterfactual/training",
        tokenizer=tokenizer,
        mode="precomputed",
        split="train",
    )

    # Live encoder mode (for inference/debugging)
    dataset = CounterfactualDataset(
        data_dir="data/counterfactual/training",
        tokenizer=tokenizer,
        mode="live",
        encoder=encoder_model,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Optional HDF5 support
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)

# Label value to ignore in loss computation
IGNORE_INDEX = -100


class CounterfactualDataset(Dataset):
    """
    Dataset for counterfactual generation training with precomputed embeddings.

    Supports two modes:
        1. precomputed: Load embeddings from HDF5 (fast, recommended for training)
        2. live: Compute embeddings on-the-fly using encoder (flexible, slower)

    Args:
        data_dir: Directory containing samples.jsonl and embeddings.h5
        tokenizer: Tokenizer for encoding counterfactual text
        mode: "precomputed" or "live"
        split: "train", "val", or "all" (uses train_val_split.json)
        encoder: Encoder model for live mode (required if mode="live")
        max_input_length: Maximum encoder input length (for live mode)
        max_output_length: Maximum decoder output length
        full_sequence: Whether to use full sequence embeddings (for cross-attention)

    Returns dict per sample:
        - encoder_embeddings: (hidden_dim,) or (seq_len, hidden_dim) tensor
        - encoder_attention_mask: (seq_len,) tensor (for full sequence mode)
        - decoder_input_ids: (seq_len,) token IDs for decoder input
        - labels: (seq_len,) shifted labels with -100 for padding
        - sample_id: Original sample ID for debugging
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        mode: str = "precomputed",
        split: str = "train",
        encoder: Any | None = None,
        max_input_length: int = 256,
        max_output_length: int = 256,
        full_sequence: bool = False,
        load_to_ram: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.mode = mode
        self.split = split
        self.encoder = encoder
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.full_sequence = full_sequence
        self.load_to_ram = load_to_ram

        # Initialize file handles early to avoid __del__ issues
        self._embeddings_file = None
        self._embeddings = None
        self._sequence_offsets = None
        self._sequence_lengths = None
        self._embeddings_in_ram = None  # For load_to_ram mode

        # Validate mode
        if mode not in ("precomputed", "live"):
            raise ValueError(f"mode must be 'precomputed' or 'live', got {mode}")

        if mode == "live" and encoder is None:
            raise ValueError("encoder is required for live mode")

        # Load samples
        self.samples = self._load_samples()

        # Load split indices
        self.indices = self._load_split_indices()

        if mode == "precomputed":
            self._load_embeddings()

        logger.info(
            f"CounterfactualDataset initialized: mode={mode}, split={split}, "
            f"samples={len(self)}, full_sequence={full_sequence}, load_to_ram={load_to_ram}"
        )

    def _load_samples(self) -> list[dict]:
        """Load samples from JSONL file."""
        samples_path = self.data_dir / "samples.jsonl"

        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")

        samples = []
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        logger.info(f"Loaded {len(samples):,} samples from {samples_path}")
        return samples

    def _load_split_indices(self) -> list[int]:
        """Load train/val split indices."""
        split_path = self.data_dir / "train_val_split.json"

        if not split_path.exists():
            # No split file - use all samples
            logger.warning(f"No split file found at {split_path}, using all samples")
            return list(range(len(self.samples)))

        with open(split_path, encoding="utf-8") as f:
            split_data = json.load(f)

        if self.split == "train":
            indices = split_data["train_indices"]
        elif self.split == "val":
            indices = split_data["val_indices"]
        elif self.split == "all":
            indices = list(range(len(self.samples)))
        else:
            raise ValueError(f"split must be 'train', 'val', or 'all', got {self.split}")

        logger.info(f"Using {len(indices):,} samples for {self.split} split")
        return indices

    def _load_embeddings(self) -> None:
        """Load precomputed embeddings from HDF5.

        If load_to_ram=True, loads entire embedding array into system RAM
        for zero disk I/O during training (recommended for high-RAM systems).
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for precomputed mode. Install with: pip install h5py")

        if self.full_sequence:
            embeddings_path = self.data_dir / "sequence_embeddings.h5"
        else:
            embeddings_path = self.data_dir / "embeddings.h5"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        # Open HDF5 file
        self._embeddings_file = h5py.File(embeddings_path, "r")

        if self.load_to_ram:
            # Load entire embeddings array into RAM for zero disk I/O
            logger.info(f"Loading embeddings into RAM from {embeddings_path}...")
            self._embeddings_in_ram = self._embeddings_file["embeddings"][:]
            self._embeddings = self._embeddings_in_ram

            # Calculate RAM usage
            ram_mb = self._embeddings_in_ram.nbytes / (1024 * 1024)
            logger.info(f"Loaded {len(self._embeddings_in_ram):,} embeddings into RAM ({ram_mb:.1f} MB)")
        else:
            # Memory-mapped access (lazy loading from disk)
            self._embeddings = self._embeddings_file["embeddings"]

        if self.full_sequence:
            self._sequence_offsets = self._embeddings_file["offsets"][:]
            self._sequence_lengths = self._embeddings_file["sequence_lengths"][:]
            logger.info(
                f"Loaded full sequence embeddings: shape={self._embeddings.shape}, "
                f"total_tokens={self._embeddings_file.attrs.get('total_tokens', 'N/A')}"
            )
        else:
            logger.info(
                f"Loaded pooled embeddings: shape={self._embeddings.shape}, "
                f"mode={'RAM' if self.load_to_ram else 'memory-mapped'}"
            )

    def __len__(self) -> int:
        """Return number of samples in the split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample."""
        # Map to original sample index
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        # Get encoder embeddings
        if self.mode == "precomputed":
            encoder_out = self._get_precomputed_embeddings(sample_idx)
        else:
            encoder_out = self._get_live_embeddings(sample)

        # Tokenize counterfactual text for decoder
        decoder_out = self._tokenize_output(sample["counterfactual_full_text"])

        return {
            "encoder_embeddings": encoder_out["embeddings"],
            "encoder_attention_mask": encoder_out["attention_mask"],
            "decoder_input_ids": decoder_out["input_ids"],
            "labels": decoder_out["labels"],
            "sample_id": sample.get("sample_id", sample_idx),
        }

    def _get_precomputed_embeddings(self, sample_idx: int) -> dict[str, torch.Tensor]:
        """Get precomputed embeddings for a sample."""
        if self.full_sequence:
            # Get variable-length sequence from flattened storage
            start_offset = self._sequence_offsets[sample_idx]
            end_offset = self._sequence_offsets[sample_idx + 1]
            seq_len = self._sequence_lengths[sample_idx]

            embeddings = torch.tensor(
                self._embeddings[start_offset:end_offset],
                dtype=torch.float32,
            )
            attention_mask = torch.ones(seq_len, dtype=torch.long)
        else:
            # Pooled embedding - single vector
            embeddings = torch.tensor(
                self._embeddings[sample_idx],
                dtype=torch.float32,
            )
            # For pooled mode, attention mask is just 1 (single position)
            attention_mask = torch.ones(1, dtype=torch.long)

        return {
            "embeddings": embeddings,
            "attention_mask": attention_mask,
        }

    def _get_live_embeddings(self, sample: dict) -> dict[str, torch.Tensor]:
        """Compute embeddings on-the-fly using encoder."""
        input_text = sample["input_text"]

        # Tokenize input for encoder
        inputs = self.tokenizer(
            input_text,
            padding=False,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        # Get encoder output
        with torch.no_grad():
            if hasattr(self.encoder, "encode"):
                # Simple encode interface
                embeddings = self.encoder.encode(input_text)
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)
                attention_mask = torch.ones(1, dtype=torch.long)
            else:
                # Full encoder forward
                outputs = self.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                if self.full_sequence:
                    embeddings = outputs.last_hidden_state.squeeze(0)
                    attention_mask = inputs["attention_mask"].squeeze(0)
                else:
                    # CLS pooling
                    embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
                    attention_mask = torch.ones(1, dtype=torch.long)

        return {
            "embeddings": embeddings,
            "attention_mask": attention_mask,
        }

    def _tokenize_output(self, text: str) -> dict[str, torch.Tensor]:
        """
        Tokenize counterfactual text for decoder.

        Creates decoder_input_ids and shifted labels for causal LM training.
        Labels have -100 for positions that should be ignored in loss.

        For causal LM training:
        - decoder_input_ids: [CLS, tok1, tok2, ..., tokN, SEP]
        - labels:            [-100, tok1, tok2, ..., tokN, SEP]

        The loss is computed only on positions where label != -100.
        We mask the first position (CLS/BOS) since it's the input to start generation.
        """
        # Tokenize with special tokens (CLS at start, SEP at end)
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)

        # Create labels - same as input_ids but mask the first token (CLS/BOS)
        # Model learns to predict: tok1 from CLS, tok2 from tok1, ..., SEP from tokN
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX  # Don't compute loss on predicting the first token

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._embeddings_file is not None:
            try:
                self._embeddings_file.close()
            except (TypeError, ValueError):
                pass  # Ignore errors during shutdown
            self._embeddings_file = None

    def __del__(self):
        """Clean up on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown

    def get_sample_text(self, idx: int) -> dict[str, str]:
        """Get the original text for a sample (for debugging)."""
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        return {
            "input_text": sample.get("input_text", ""),
            "counterfactual_full_text": sample.get("counterfactual_full_text", ""),
            "domain": sample.get("domain", ""),
            "subdomain": sample.get("subdomain", ""),
        }


class CounterfactualSubset(Dataset):
    """
    Subset of CounterfactualDataset for custom splits.

    Useful for creating custom train/val splits or sampling.
    """

    def __init__(self, dataset: CounterfactualDataset, indices: list[int]):
        self.dataset = dataset
        self.subset_indices = indices

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.dataset[self.subset_indices[idx]]

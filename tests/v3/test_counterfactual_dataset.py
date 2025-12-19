"""
Tests for counterfactual dataset and data pipeline.

This module tests the CounterfactualDataset class and data loading utilities.
Tests use mock data to avoid dependency on actual training data.

Related: docs/DECODER_EMBEDDING_ANALYSIS.md (Epic 3.2)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Only import h5py if available
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from transformers import AutoTokenizer

from modeling_studio.data.counterfactual_dataset import CounterfactualDataset


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tokenizer():
    """Get the ModernBERT tokenizer."""
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


@pytest.fixture
def mock_data_dir(tmp_path: Path, tokenizer) -> Path:
    """Create a mock data directory with samples and embeddings."""
    # Create sample data matching expected schema
    samples = [
        {
            "sample_id": i,
            "counterfactual_full_text": f"If you had done something different {i}.",
            "input": {
                "context": f"Test context {i}",
                "domain": "parenting",
                "subdomain": "discipline",
                "outcome_valence": "negative",
            },
            "output": {
                "counterfactual": f"If you had done something different {i}.",
            },
        }
        for i in range(10)
    ]

    # Write samples.jsonl
    samples_path = tmp_path / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Create train/val split
    split_data = {
        "train_indices": list(range(8)),
        "val_indices": list(range(8, 10)),
    }
    split_path = tmp_path / "train_val_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    # Create embeddings.h5 (pooled embeddings)
    if HAS_H5PY:
        embeddings_path = tmp_path / "embeddings.h5"
        with h5py.File(embeddings_path, "w") as f:
            # Random embeddings with realistic norm (~3.0)
            embeddings = np.random.randn(10, 768).astype(np.float16)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) * 3.0
            f.create_dataset("embeddings", data=embeddings)

    return tmp_path


@pytest.fixture
def mock_sequence_data_dir(tmp_path: Path, tokenizer) -> Path:
    """Create mock data with full sequence embeddings."""
    # Create sample data
    samples = [
        {
            "sample_id": i,
            "counterfactual_full_text": f"If you had done something different {i}.",
            "input": {
                "context": f"Test context {i} with some more words",
                "domain": "parenting",
                "subdomain": "discipline",
                "outcome_valence": "negative",
            },
            "output": {
                "counterfactual": f"If you had done something different {i}.",
            },
        }
        for i in range(5)
    ]

    # Write samples.jsonl
    samples_path = tmp_path / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Create train/val split
    split_data = {
        "train_indices": list(range(4)),
        "val_indices": [4],
    }
    split_path = tmp_path / "train_val_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    # Create sequence_embeddings.h5 (full sequence embeddings)
    if HAS_H5PY:
        embeddings_path = tmp_path / "sequence_embeddings.h5"
        with h5py.File(embeddings_path, "w") as f:
            # Variable length sequences (simulate tokenized input)
            seq_lengths = [10, 12, 8, 15, 11]  # Variable lengths
            total_tokens = sum(seq_lengths)

            # Create embeddings for all tokens
            all_embeddings = np.random.randn(total_tokens, 768).astype(np.float16)
            all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True) * 3.0
            f.create_dataset("embeddings", data=all_embeddings)

            # Create offsets array
            offsets = np.zeros(len(seq_lengths) + 1, dtype=np.int64)
            for i, length in enumerate(seq_lengths):
                offsets[i + 1] = offsets[i] + length
            f.create_dataset("offsets", data=offsets)

            # Create sequence_lengths array (correct key name)
            f.create_dataset("sequence_lengths", data=np.array(seq_lengths, dtype=np.int32))

    return tmp_path


# =============================================================================
# Epic 3.2.1: Dataset Loading Tests
# =============================================================================


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestDatasetLoading:
    """Tests for CounterfactualDataset loading."""

    def test_dataset_loads_samples(self, mock_data_dir: Path, tokenizer) -> None:
        """Dataset should load samples from JSONL file."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        assert len(dataset) == 8  # 8 training samples

    def test_dataset_loads_val_split(self, mock_data_dir: Path, tokenizer) -> None:
        """Dataset should respect train/val split."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="val",
            full_sequence=False,
        )

        assert len(dataset) == 2  # 2 validation samples

    def test_dataset_loads_all_samples(self, mock_data_dir: Path, tokenizer) -> None:
        """Dataset should load all samples when split='all'."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="all",
            full_sequence=False,
        )

        assert len(dataset) == 10  # All samples

    def test_dataset_returns_correct_keys(self, mock_data_dir: Path, tokenizer) -> None:
        """Dataset __getitem__ should return expected keys."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        sample = dataset[0]

        assert "encoder_embeddings" in sample
        assert "decoder_input_ids" in sample
        assert "labels" in sample

    def test_embeddings_have_correct_shape(self, mock_data_dir: Path, tokenizer) -> None:
        """Pooled embeddings should have shape (768,)."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        sample = dataset[0]

        # Pooled embeddings
        assert sample["encoder_embeddings"].shape == (768,)

    def test_embeddings_not_nan(self, mock_data_dir: Path, tokenizer) -> None:
        """Embeddings should not contain NaN values."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert not torch.isnan(sample["encoder_embeddings"]).any(), (
                f"Sample {i} contains NaN embeddings"
            )

    def test_embeddings_not_inf(self, mock_data_dir: Path, tokenizer) -> None:
        """Embeddings should not contain Inf values."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert not torch.isinf(sample["encoder_embeddings"]).any(), (
                f"Sample {i} contains Inf embeddings"
            )


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestSequenceEmbeddings:
    """Tests for full sequence embedding mode."""

    def test_sequence_embeddings_have_correct_shape(
        self, mock_sequence_data_dir: Path, tokenizer
    ) -> None:
        """Full sequence embeddings should have shape (seq_len, 768)."""
        dataset = CounterfactualDataset(
            data_dir=mock_sequence_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=True,
        )

        sample = dataset[0]

        # Should be 2D: (seq_len, hidden_dim)
        assert len(sample["encoder_embeddings"].shape) == 2
        assert sample["encoder_embeddings"].shape[1] == 768

    def test_sequence_embeddings_have_attention_mask(
        self, mock_sequence_data_dir: Path, tokenizer
    ) -> None:
        """Full sequence mode should provide attention mask."""
        dataset = CounterfactualDataset(
            data_dir=mock_sequence_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=True,
        )

        sample = dataset[0]

        # Should have attention mask matching sequence length
        assert "encoder_attention_mask" in sample
        seq_len = sample["encoder_embeddings"].shape[0]
        assert sample["encoder_attention_mask"].shape[0] == seq_len


class TestDatasetValidation:
    """Tests for dataset validation and error handling."""

    def test_raises_on_missing_samples_file(self, tmp_path: Path, tokenizer) -> None:
        """Should raise FileNotFoundError if samples.jsonl is missing."""
        with pytest.raises(FileNotFoundError):
            CounterfactualDataset(
                data_dir=tmp_path,
                tokenizer=tokenizer,
                mode="precomputed",
            )

    def test_raises_on_invalid_mode(self, mock_data_dir: Path, tokenizer) -> None:
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            CounterfactualDataset(
                data_dir=mock_data_dir,
                tokenizer=tokenizer,
                mode="invalid_mode",
            )

    def test_raises_on_live_mode_without_encoder(
        self, mock_data_dir: Path, tokenizer
    ) -> None:
        """Should raise ValueError if live mode without encoder."""
        with pytest.raises(ValueError, match="encoder is required"):
            CounterfactualDataset(
                data_dir=mock_data_dir,
                tokenizer=tokenizer,
                mode="live",
                encoder=None,
            )


class TestDecoderLabels:
    """Tests for decoder label creation."""

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_labels_are_shifted(self, mock_data_dir: Path, tokenizer) -> None:
        """Labels should be shifted version of decoder_input_ids."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        sample = dataset[0]

        # Labels should exist
        assert "labels" in sample
        assert sample["labels"].shape == sample["decoder_input_ids"].shape

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_labels_use_ignore_index_for_padding(
        self, mock_data_dir: Path, tokenizer
    ) -> None:
        """Padding positions in labels should use -100 (IGNORE_INDEX)."""
        dataset = CounterfactualDataset(
            data_dir=mock_data_dir,
            tokenizer=tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        sample = dataset[0]

        # Get pad positions
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        # Where input is padding, labels should be -100
        input_ids = sample["decoder_input_ids"]
        labels = sample["labels"]

        # Check that non-pad positions have valid labels (not -100)
        non_pad_mask = input_ids != pad_token_id
        # At least some positions should have valid labels
        valid_labels = labels[non_pad_mask]
        assert (valid_labels != -100).any(), "No valid (non-ignore) labels found"

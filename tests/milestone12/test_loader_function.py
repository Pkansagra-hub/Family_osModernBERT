"""
Tests for load_counterfactual_dataset loader function (Issue 12.2.2).

Tests the convenience loader function in loaders.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip all tests if h5py not available
pytest.importorskip("h5py")
import h5py
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1

    def mock_call(text, **kwargs):
        import torch

        seq_len = min(10, kwargs.get("max_length", 256))
        input_ids = torch.randint(3, 1000, (seq_len,))

        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": torch.ones_like(input_ids.unsqueeze(0)),
            }
        return {"input_ids": input_ids.tolist()}

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    return tokenizer


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create sample data
        samples = [
            {
                "sample_id": i,
                "input_text": f"Input text {i}",
                "counterfactual_full_text": f"Counterfactual text {i}",
                "domain": "test",
                "subdomain": "test",
            }
            for i in range(10)
        ]

        # Write samples.jsonl
        samples_path = data_dir / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Create embeddings
        hidden_dim = 768
        embeddings = np.random.randn(10, hidden_dim).astype(np.float16)

        embeddings_path = data_dir / "embeddings.h5"
        with h5py.File(embeddings_path, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings, dtype="float16")
            hf.attrs["num_samples"] = 10
            hf.attrs["hidden_dim"] = hidden_dim

        # Create train/val split
        split_data = {
            "train_indices": list(range(8)),
            "val_indices": [8, 9],
            "train_size": 8,
            "val_size": 2,
            "seed": 42,
        }

        split_path = data_dir / "train_val_split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f)

        yield data_dir


# =============================================================================
# Issue 12.2.2: Loader Function
# =============================================================================


class TestLoaderFunctionImport:
    """Tests for loader function accessibility (12.2.2-T1)."""

    def test_load_counterfactual_exists(self):
        """12.2.2-T1: load_counterfactual_dataset is importable."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        assert callable(load_counterfactual_dataset)

    def test_load_counterfactual_in_all(self):
        """Test that load_counterfactual_dataset is in __all__."""
        from modeling_studio.data import loaders

        assert "load_counterfactual_dataset" in loaders.__all__


class TestLoaderFunctionReturns:
    """Tests for loader function return value (12.2.2-T2)."""

    def test_load_counterfactual_returns_dataset(self, temp_data_dir, mock_tokenizer):
        """12.2.2-T2: Returns CounterfactualDataset instance."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset
        from modeling_studio.data.loaders import load_counterfactual_dataset

        dataset = load_counterfactual_dataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            split="train",
        )

        assert isinstance(dataset, CounterfactualDataset)

    def test_load_counterfactual_train_split(self, temp_data_dir, mock_tokenizer):
        """Test loading train split."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        dataset = load_counterfactual_dataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            split="train",
        )

        assert len(dataset) == 8

    def test_load_counterfactual_val_split(self, temp_data_dir, mock_tokenizer):
        """Test loading val split."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        dataset = load_counterfactual_dataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            split="val",
        )

        assert len(dataset) == 2


class TestLoaderFunctionValidation:
    """Tests for loader function validation."""

    def test_load_counterfactual_validates_path(self, mock_tokenizer):
        """Test that loader validates path exists."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_counterfactual_dataset(
                data_dir="/nonexistent/path",
                tokenizer=mock_tokenizer,
                split="train",
            )

    def test_load_counterfactual_validates_samples(self, mock_tokenizer):
        """Test that loader validates samples.jsonl exists."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="samples.jsonl not found"):
                load_counterfactual_dataset(
                    data_dir=tmpdir,
                    tokenizer=mock_tokenizer,
                    split="train",
                )

    def test_load_counterfactual_validates_embeddings(self, mock_tokenizer):
        """Test that loader validates embeddings.h5 exists in precomputed mode."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create samples but no embeddings
            samples_path = data_dir / "samples.jsonl"
            with open(samples_path, "w") as f:
                f.write(json.dumps({"sample_id": 0, "counterfactual_full_text": "test"}) + "\n")

            with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
                load_counterfactual_dataset(
                    data_dir=data_dir,
                    tokenizer=mock_tokenizer,
                    mode="precomputed",
                    split="all",
                )


class TestLoaderFunctionParameters:
    """Tests for loader function parameters."""

    def test_load_counterfactual_with_max_lengths(self, temp_data_dir, mock_tokenizer):
        """Test loader with custom max lengths."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        dataset = load_counterfactual_dataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            split="train",
            max_input_length=128,
            max_output_length=128,
        )

        assert dataset.max_input_length == 128
        assert dataset.max_output_length == 128

    def test_load_counterfactual_live_mode(self, temp_data_dir, mock_tokenizer):
        """Test loader with live mode."""
        from modeling_studio.data.loaders import load_counterfactual_dataset

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.randn(768).astype(np.float32)

        dataset = load_counterfactual_dataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="live",
            encoder=mock_encoder,
            split="train",
        )

        assert dataset.mode == "live"
        assert dataset.encoder is mock_encoder

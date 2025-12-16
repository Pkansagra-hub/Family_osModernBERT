"""
Tests for CounterfactualDataset (Issue 12.1.1, 12.1.2).

Tests the dataset class for loading counterfactual training data
with precomputed embeddings from HDF5 or live encoding.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Skip all tests if h5py not available
pytest.importorskip("h5py")
import h5py


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
        # Simple mock tokenization: return random IDs
        if isinstance(text, str):
            seq_len = min(len(text.split()) + 2, kwargs.get("max_length", 256))
            input_ids = torch.randint(3, 1000, (seq_len,))
        else:
            input_ids = torch.randint(3, 1000, (10,))

        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
                "attention_mask": torch.ones_like(input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids),
            }
        return {"input_ids": input_ids.tolist(), "attention_mask": [1] * len(input_ids)}

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
                "sample_id": 0,
                "input_text": "The child ran into the street without looking.",
                "counterfactual_full_text": "If the child had looked both ways before crossing, they would have seen the car.",
                "domain": "safety",
                "subdomain": "traffic",
            },
            {
                "sample_id": 1,
                "input_text": "She forgot her umbrella and got soaked.",
                "counterfactual_full_text": "If she had checked the weather forecast, she would have brought an umbrella.",
                "domain": "daily_life",
                "subdomain": "weather",
            },
            {
                "sample_id": 2,
                "input_text": "The student failed the exam.",
                "counterfactual_full_text": "If the student had studied more, they would have passed the exam.",
                "domain": "education",
                "subdomain": "academics",
            },
            {
                "sample_id": 3,
                "input_text": "He missed the train by one minute.",
                "counterfactual_full_text": "If he had left home five minutes earlier, he would have caught the train.",
                "domain": "transportation",
                "subdomain": "commute",
            },
        ]

        # Write samples.jsonl
        samples_path = data_dir / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Create pooled embeddings (4 samples, 768 hidden)
        hidden_dim = 768
        embeddings = np.random.randn(4, hidden_dim).astype(np.float16)

        embeddings_path = data_dir / "embeddings.h5"
        with h5py.File(embeddings_path, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings, dtype="float16")
            hf.attrs["num_samples"] = 4
            hf.attrs["hidden_dim"] = hidden_dim

        # Create train/val split
        split_data = {
            "train_indices": [0, 1, 2],
            "val_indices": [3],
            "train_size": 3,
            "val_size": 1,
            "seed": 42,
        }

        split_path = data_dir / "train_val_split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f)

        yield data_dir


@pytest.fixture
def temp_data_dir_full_sequence():
    """Create a temporary data directory with full sequence embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create sample data
        samples = [
            {
                "sample_id": 0,
                "input_text": "Short text here.",
                "counterfactual_full_text": "Alternative outcome if action was different.",
                "domain": "test",
                "subdomain": "test",
            },
            {
                "sample_id": 1,
                "input_text": "Another sample with more words to test variable length.",
                "counterfactual_full_text": "Different counterfactual with different outcome.",
                "domain": "test",
                "subdomain": "test",
            },
        ]

        # Write samples.jsonl
        samples_path = data_dir / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Create full sequence embeddings (variable length)
        hidden_dim = 768
        seq_lengths = [5, 10]  # Different lengths per sample
        total_tokens = sum(seq_lengths)

        # Flattened embeddings
        all_embeddings = np.random.randn(total_tokens, hidden_dim).astype(np.float16)
        offsets = [0, 5, 15]  # Start offsets for each sample + end

        embeddings_path = data_dir / "sequence_embeddings.h5"
        with h5py.File(embeddings_path, "w") as hf:
            hf.create_dataset("embeddings", data=all_embeddings, dtype="float16")
            hf.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
            hf.create_dataset("sequence_lengths", data=np.array(seq_lengths, dtype=np.int32))
            hf.attrs["num_samples"] = 2
            hf.attrs["total_tokens"] = total_tokens
            hf.attrs["hidden_dim"] = hidden_dim

        # Create train/val split (all train for simplicity)
        split_data = {
            "train_indices": [0, 1],
            "val_indices": [],
            "train_size": 2,
            "val_size": 0,
            "seed": 42,
        }

        split_path = data_dir / "train_val_split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f)

        yield data_dir


# =============================================================================
# Issue 12.1.1: CounterfactualDataset (Precomputed Mode)
# =============================================================================


class TestCounterfactualDatasetLoading:
    """Tests for basic dataset loading (12.1.1-T1, T2)."""

    def test_dataset_loads_samples(self, temp_data_dir, mock_tokenizer):
        """12.1.1-T1: Dataset loads samples from JSONL."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="all",
        )

        # Should have loaded all 4 samples
        assert len(dataset.samples) == 4
        assert dataset.samples[0]["sample_id"] == 0
        assert "input_text" in dataset.samples[0]
        assert "counterfactual_full_text" in dataset.samples[0]

    def test_dataset_loads_embeddings(self, temp_data_dir, mock_tokenizer):
        """12.1.1-T2: Dataset loads embeddings from HDF5."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="all",
        )

        # Check embeddings loaded
        assert dataset._embeddings is not None
        assert dataset._embeddings.shape[0] == 4
        assert dataset._embeddings.shape[1] == 768

    def test_dataset_respects_split(self, temp_data_dir, mock_tokenizer):
        """Test that dataset respects train/val split."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        train_ds = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        val_ds = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="val",
        )

        assert len(train_ds) == 3
        assert len(val_ds) == 1


class TestCounterfactualDatasetTokenization:
    """Tests for tokenization (12.1.1-T3, T4)."""

    def test_dataset_tokenization(self, temp_data_dir, mock_tokenizer):
        """12.1.1-T3: Output text tokenized with BOS/EOS."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        sample = dataset[0]

        # Should have decoder_input_ids
        assert "decoder_input_ids" in sample
        assert isinstance(sample["decoder_input_ids"], torch.Tensor)
        assert sample["decoder_input_ids"].dim() == 1
        assert len(sample["decoder_input_ids"]) > 0

    def test_dataset_label_shifting(self, temp_data_dir, mock_tokenizer):
        """12.1.1-T4: Labels are present for causal LM."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        sample = dataset[0]

        # Should have labels
        assert "labels" in sample
        assert isinstance(sample["labels"], torch.Tensor)
        # Labels should have same length as decoder_input_ids
        assert sample["labels"].shape == sample["decoder_input_ids"].shape


class TestCounterfactualDatasetOutput:
    """Tests for dataset output format."""

    def test_dataset_output_keys(self, temp_data_dir, mock_tokenizer):
        """Test that dataset returns all required keys."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        sample = dataset[0]

        required_keys = [
            "encoder_embeddings",
            "encoder_attention_mask",
            "decoder_input_ids",
            "labels",
            "sample_id",
        ]

        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_dataset_encoder_embeddings_shape(self, temp_data_dir, mock_tokenizer):
        """Test encoder embeddings have correct shape."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=False,
        )

        sample = dataset[0]

        # Pooled mode: (hidden_dim,)
        assert sample["encoder_embeddings"].shape == (768,)
        assert sample["encoder_attention_mask"].shape == (1,)

    def test_dataset_full_sequence_shape(self, temp_data_dir_full_sequence, mock_tokenizer):
        """Test full sequence embeddings have correct shape."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir_full_sequence,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
            full_sequence=True,
        )

        sample0 = dataset[0]
        sample1 = dataset[1]

        # Full sequence mode: (seq_len, hidden_dim)
        assert sample0["encoder_embeddings"].shape == (5, 768)
        assert sample0["encoder_attention_mask"].shape == (5,)

        assert sample1["encoder_embeddings"].shape == (10, 768)
        assert sample1["encoder_attention_mask"].shape == (10,)


# =============================================================================
# Issue 12.1.2: CounterfactualDataset (Live Encoder Mode)
# =============================================================================


class TestCounterfactualDatasetLiveMode:
    """Tests for live encoder mode (12.1.2-T1, T2)."""

    def test_live_mode_requires_encoder(self, temp_data_dir, mock_tokenizer):
        """Test that live mode requires encoder."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        with pytest.raises(ValueError, match="encoder is required"):
            CounterfactualDataset(
                data_dir=temp_data_dir,
                tokenizer=mock_tokenizer,
                mode="live",
                split="train",
                encoder=None,
            )

    def test_live_mode_encodes_text(self, temp_data_dir, mock_tokenizer):
        """12.1.2-T1: Live mode calls encoder.encode()."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        # Create mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.randn(768).astype(np.float32)

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="live",
            split="train",
            encoder=mock_encoder,
        )

        sample = dataset[0]

        # Encoder should have been called
        mock_encoder.encode.assert_called_once()

        # Should still have embeddings
        assert "encoder_embeddings" in sample
        assert sample["encoder_embeddings"].shape == (768,)

    def test_live_mode_same_format(self, temp_data_dir, mock_tokenizer):
        """12.1.2-T2: Live mode output matches precomputed format."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        # Create mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.randn(768).astype(np.float32)

        live_dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="live",
            split="train",
            encoder=mock_encoder,
        )

        precomputed_dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        live_sample = live_dataset[0]
        precomputed_sample = precomputed_dataset[0]

        # Both should have same keys
        assert set(live_sample.keys()) == set(precomputed_sample.keys())


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestCounterfactualDatasetErrors:
    """Tests for error handling."""

    def test_invalid_mode_raises_error(self, temp_data_dir, mock_tokenizer):
        """Test that invalid mode raises ValueError."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        with pytest.raises(ValueError, match="mode must be"):
            CounterfactualDataset(
                data_dir=temp_data_dir,
                tokenizer=mock_tokenizer,
                mode="invalid",
                split="train",
            )

    def test_missing_samples_file_raises_error(self, mock_tokenizer):
        """Test that missing samples.jsonl raises FileNotFoundError."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Samples file not found"):
                CounterfactualDataset(
                    data_dir=tmpdir,
                    tokenizer=mock_tokenizer,
                    mode="precomputed",
                    split="train",
                )

    def test_missing_embeddings_file_raises_error(self, mock_tokenizer):
        """Test that missing embeddings.h5 raises FileNotFoundError."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create samples but no embeddings
            samples_path = data_dir / "samples.jsonl"
            with open(samples_path, "w") as f:
                f.write(json.dumps({"sample_id": 0, "counterfactual_full_text": "test"}) + "\n")

            with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
                CounterfactualDataset(
                    data_dir=data_dir,
                    tokenizer=mock_tokenizer,
                    mode="precomputed",
                    split="all",
                )


class TestCounterfactualDatasetUtilities:
    """Tests for utility methods."""

    def test_get_sample_text(self, temp_data_dir, mock_tokenizer):
        """Test get_sample_text utility method."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        text_info = dataset.get_sample_text(0)

        assert "input_text" in text_info
        assert "counterfactual_full_text" in text_info
        assert "domain" in text_info
        assert len(text_info["input_text"]) > 0

    def test_dataset_close(self, temp_data_dir, mock_tokenizer):
        """Test that close() properly closes HDF5 file."""
        from modeling_studio.data.counterfactual_dataset import CounterfactualDataset

        dataset = CounterfactualDataset(
            data_dir=temp_data_dir,
            tokenizer=mock_tokenizer,
            mode="precomputed",
            split="train",
        )

        # Should have open file
        assert dataset._embeddings_file is not None

        dataset.close()

        # Should be closed
        assert dataset._embeddings_file is None

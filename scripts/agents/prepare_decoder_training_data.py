#!/usr/bin/env python3
"""
Prepare Decoder Training Data with UltraBERT Encoder Embeddings.

This script:
1. Loads counterfactual samples from synthetic generation
2. Computes UltraBERT encoder embeddings for each input.text
3. Saves embeddings in efficient format (HDF5/numpy) alongside text data
4. Creates training-ready dataset for Stage C decoder training

Usage:
    # Pooled embeddings (N, 768) - simpler, smaller storage
    python scripts/agents/prepare_decoder_training_data.py \
        --input-dir data/counterfactual/synthetic \
        --output-dir data/counterfactual/training \
        --batch-size 32

    # Full sequence embeddings (total_tokens, 768) - for cross-attention
    python scripts/agents/prepare_decoder_training_data.py \
        --input-dir data/counterfactual/synthetic \
        --output-dir data/counterfactual/training \
        --full-sequence \
        --batch-size 32

Output Structure (Pooled Mode):
    data/counterfactual/training/
    ├── samples.jsonl          # Text data with sample_id
    ├── embeddings.h5          # HDF5 with (N, 768) float16 embeddings
    ├── manifest.json          # Metadata and statistics
    └── train_val_split.json   # 95/5 train/val split indices

Output Structure (Full Sequence Mode):
    data/counterfactual/training/
    ├── samples.jsonl              # Text data with sample_id
    ├── sequence_embeddings.h5     # HDF5 with (total_tokens, 768) + offsets
    ├── manifest.json              # Metadata and statistics
    └── train_val_split.json       # 95/5 train/val split indices

Decoder Architecture Considerations:
    - Pooled (CLS/mean): Decoder conditions on single 768-dim vector
      - Simpler cross-attention (or just concat to decoder input)
      - ~150 MB for 100K samples

    - Full Sequence: Decoder cross-attends to all encoder positions
      - Richer context, position-aware generation
      - ~2-5 GB for 100K samples (variable length, compressed)
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Optional HDF5 support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Constitution registry for 3-layer constitutional training
try:
    from modeling_studio.data.constitution_registry import (
        extract_constitution_from_sample,
        get_default_registry,
    )
    HAS_CONSTITUTION_REGISTRY = True
except ImportError:
    HAS_CONSTITUTION_REGISTRY = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_constitution_fields(sample: dict) -> dict:
    """
    Extract 3-layer constitution fields from a training sample.

    FamilyOS Constitutional Layers:
        Layer 1: Family Values - Core principles (privacy, respect, support)
        Layer 2: Individual Preferences - Per-member boundaries
        Layer 3: Situational Context - Context-adaptive rules

    Args:
        sample: Raw sample from JSONL

    Returns:
        Dict with constitution fields to merge into processed sample
    """
    if HAS_CONSTITUTION_REGISTRY:
        try:
            return extract_constitution_from_sample(sample)
        except Exception as e:
            logger.debug(f"Constitution extraction failed: {e}")

    # Fallback: basic extraction without registry
    metadata = sample.get("metadata", {})
    cultural_context = metadata.get("cultural_context", "universal")

    # Map cultural_context to family_value_id
    CULTURAL_TO_ID = {
        "universal": 0,
        "indian": 5,    # indian_joint_family
        "western": 6,   # western_nuclear
        "asian": 7,     # asian_collectivist
    }

    return {
        "family_value": cultural_context,
        "family_value_id": CULTURAL_TO_ID.get(cultural_context, 0),
        "individual_pref": "default",
        "individual_pref_id": 0,
        "situational_context": "normal",
        "situational_context_id": 0,
        "constitution_text": "",
    }


def load_counterfactual_samples(input_dir: Path) -> list[dict]:
    """Load all counterfactual samples from shards or samples.jsonl."""
    samples = []

    # Try shard files first
    shards = sorted(input_dir.glob("shard_*.jsonl"))

    # If no shards, try samples.jsonl
    if not shards:
        samples_file = input_dir / "samples.jsonl"
        if samples_file.exists():
            shards = [samples_file]
        else:
            # Also try any .jsonl file
            shards = sorted(input_dir.glob("*.jsonl"))

    if not shards:
        raise ValueError(f"No JSONL files found in {input_dir}")

    logger.info(f"Loading samples from {len(shards)} file(s)...")

    for shard in tqdm(shards, desc="Loading shards"):
        with open(shard, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    # Validate required fields
                    input_text = sample.get("input", {}).get("text", "")
                    cf_text = sample.get("counterfactual", {}).get("full_text", "")

                    if input_text and cf_text:
                        samples.append({
                            "sample_id": len(samples),
                            "shard": shard.stem,
                            "line": line_num,
                            "domain": sample.get("domain", ""),
                            "subdomain": sample.get("subdomain", ""),
                            "input_text": input_text,
                            "input_outcome_valence": sample.get("input", {}).get("outcome_valence", ""),
                            "input_severity": sample.get("input", {}).get("severity", ""),
                            "counterfactual_action": sample.get("counterfactual", {}).get("alternative_action", ""),
                            "counterfactual_outcome": sample.get("counterfactual", {}).get("predicted_outcome", ""),
                            "counterfactual_full_text": cf_text,
                            "metadata": sample.get("metadata", {}),
                            # 3-Layer Constitution extraction
                            **_extract_constitution_fields(sample),
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {shard.name}:{line_num}")
                    continue

    logger.info(f"Loaded {len(samples):,} valid samples")
    return samples


def load_ultrabert_encoder(model_path: Path | None, device: str = "cuda"):
    """Load UltraBERT encoder for embedding extraction using installed library.

    Args:
        model_path: Path to model weights. If None, uses bundled weights from library.
        device: Device to use ("cuda" or "cpu")

    Returns:
        PyTorchInferenceEngine with access to encoder
    """
    from familyos_ultrabert.pytorch_inference import PyTorchInferenceEngine


    # Use bundled weights if no path provided
    if model_path is None:
        from familyos_ultrabert.model import DEFAULT_PYTORCH_PATH
        model_path = DEFAULT_PYTORCH_PATH
        logger.info(f"Using bundled weights from familyos_ultrabert: {model_path}")
    else:
        logger.info(f"Loading UltraBERT from custom path: {model_path}")

    # Load using the library's PyTorch engine for direct encoder access
    engine = PyTorchInferenceEngine.load(
        model_path=str(model_path),
        device=device,
        enable_cache=False,  # Don't need cache for batch processing
    )

    logger.info(f"Loaded model on {device}")
    logger.info(f"Available capabilities: {engine.capabilities}")

    return engine

def extract_embeddings(
    samples: list[dict],
    engine,  # PyTorchInferenceEngine from familyos_ultrabert
    batch_size: int = 32,
    max_length: int = 512,
    pooling: str = "cls",  # "cls" or "mean"
) -> np.ndarray:
    """Extract encoder embeddings for all input texts using familyos_ultrabert.

    Args:
        samples: List of sample dicts with 'input_text' key
        engine: PyTorchInferenceEngine from familyos_ultrabert
        batch_size: Batch size for inference
        max_length: Max sequence length
        pooling: Pooling strategy - "cls" for [CLS] token, "mean" for mean pooling

    Returns:
        numpy array of shape (num_samples, hidden_dim) in float16
    """
    num_samples = len(samples)
    hidden_dim = engine.model.config.hidden_size  # Should be 768
    device = engine.device

    # Pre-allocate embeddings array
    embeddings = np.zeros((num_samples, hidden_dim), dtype=np.float16)

    logger.info(f"Extracting pooled embeddings for {num_samples:,} samples...")
    logger.info(f"Batch size: {batch_size}, Pooling: {pooling}, Device: {device}")

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Extracting embeddings"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = [s["input_text"] for s in samples[batch_start:batch_end]]

            # Tokenize batch
            inputs = engine.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass through encoder
            outputs = engine.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )

            # Get hidden states
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Pool embeddings
            if pooling == "cls":
                # Use [CLS] token embedding
                batch_embeddings = hidden_states[:, 0, :]  # (batch, hidden_dim)
            elif pooling == "mean":
                # Mean pooling over non-padded tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                masked_hidden = hidden_states * mask
                batch_embeddings = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            # Store as float16 to save space
            embeddings[batch_start:batch_end] = batch_embeddings.cpu().numpy().astype(np.float16)

    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    logger.info(f"Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    return embeddings


def extract_full_sequence_embeddings(
    samples: list[dict],
    engine,  # PyTorchInferenceEngine from familyos_ultrabert
    output_dir: Path,
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    """Extract FULL sequence embeddings for cross-attention in decoder.

    Saves variable-length sequences efficiently to avoid 75GB+ storage.
    Each sample's hidden states are stored with their actual sequence length.

    Args:
        samples: List of sample dicts with 'input_text' key
        engine: PyTorchInferenceEngine from familyos_ultrabert
        output_dir: Directory to save sequence embeddings
        batch_size: Batch size for inference
        max_length: Max sequence length

    Returns:
        dict with metadata about saved sequences
    """
    num_samples = len(samples)
    hidden_dim = engine.model.config.hidden_size  # 768
    device = engine.device

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting FULL SEQUENCE embeddings for {num_samples:,} samples...")
    logger.info(f"Batch size: {batch_size}, Max length: {max_length}, Device: {device}")
    logger.info("This enables decoder cross-attention to encoder token positions")

    # Track sequence lengths and offsets for efficient storage
    sequence_lengths = []
    total_tokens = 0

    # First pass: calculate total storage needed
    logger.info("Pass 1/2: Calculating sequence lengths...")
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Counting tokens"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = [s["input_text"] for s in samples[batch_start:batch_end]]

            inputs = engine.tokenizer(
                batch_texts,
                padding=False,  # Don't pad for counting
                truncation=True,
                max_length=max_length,
            )

            for input_ids in inputs["input_ids"]:
                seq_len = len(input_ids)
                sequence_lengths.append(seq_len)
                total_tokens += seq_len

    # Calculate storage
    storage_bytes = total_tokens * hidden_dim * 2  # float16
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Avg sequence length: {total_tokens / num_samples:.1f}")
    logger.info(f"Estimated storage: {storage_bytes / 1024 / 1024:.1f} MB")

    # Pre-allocate flattened storage
    # Store as (total_tokens, hidden_dim) with offsets for reconstruction
    all_embeddings = np.zeros((total_tokens, hidden_dim), dtype=np.float16)
    attention_masks = np.ones(total_tokens, dtype=np.uint8)  # All 1s for valid tokens

    # Second pass: extract embeddings
    logger.info("Pass 2/2: Extracting embeddings...")
    current_offset = 0
    offsets = [0]  # Start offsets for each sample

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Extracting sequences"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = [s["input_text"] for s in samples[batch_start:batch_end]]

            # Tokenize with padding for batch processing
            inputs = engine.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass through encoder
            outputs = engine.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )

            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            masks = inputs["attention_mask"]  # (batch, seq_len)

            # Store only non-padded tokens for each sample
            for i in range(hidden_states.shape[0]):
                sample_idx = batch_start + i
                seq_len = sequence_lengths[sample_idx]

                # Extract non-padded hidden states
                sample_hidden = hidden_states[i, :seq_len, :].cpu().numpy().astype(np.float16)

                # Store in flattened array
                all_embeddings[current_offset:current_offset + seq_len] = sample_hidden
                current_offset += seq_len
                offsets.append(current_offset)

    # Save to disk
    if HAS_H5PY:
        embeddings_path = output_dir / "sequence_embeddings.h5"
        logger.info(f"Saving to {embeddings_path} (HDF5 with compression)...")
        with h5py.File(embeddings_path, "w") as hf:
            # Flattened embeddings
            hf.create_dataset(
                "embeddings",
                data=all_embeddings,
                dtype="float16",
                compression="gzip",
                compression_opts=4,
                chunks=(min(10000, total_tokens), hidden_dim),
            )
            # Offsets for reconstruction
            hf.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
            hf.create_dataset("sequence_lengths", data=np.array(sequence_lengths, dtype=np.int32))

            hf.attrs["num_samples"] = num_samples
            hf.attrs["total_tokens"] = total_tokens
            hf.attrs["hidden_dim"] = hidden_dim
            hf.attrs["max_sequence_length"] = max(sequence_lengths)
            hf.attrs["avg_sequence_length"] = total_tokens / num_samples
            hf.attrs["created_at"] = datetime.now().isoformat()
    else:
        # Fallback to numpy
        np.save(output_dir / "sequence_embeddings.npy", all_embeddings)
        np.save(output_dir / "sequence_offsets.npy", np.array(offsets, dtype=np.int64))
        np.save(output_dir / "sequence_lengths.npy", np.array(sequence_lengths, dtype=np.int32))

    metadata = {
        "mode": "full_sequence",
        "num_samples": num_samples,
        "total_tokens": total_tokens,
        "hidden_dim": hidden_dim,
        "max_sequence_length": max(sequence_lengths),
        "avg_sequence_length": total_tokens / num_samples,
        "storage_mb": storage_bytes / 1024 / 1024,
    }

    logger.info(f"Saved full sequence embeddings:")
    logger.info(f"  - Shape: ({total_tokens:,}, {hidden_dim})")
    logger.info(f"  - Storage: {storage_bytes / 1024 / 1024:.1f} MB")
    logger.info(f"  - Max seq len: {max(sequence_lengths)}")

    return metadata


def save_training_data(
    samples: list[dict],
    embeddings: np.ndarray | None,
    output_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    sequence_metadata: dict | None = None,
):
    """Save training data in efficient format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(samples)

    # 1. Save samples as JSONL (without embeddings)
    samples_path = output_dir / "samples.jsonl"
    logger.info(f"Saving samples to {samples_path}...")
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 2. Save pooled embeddings if provided (not in full-sequence mode)
    embeddings_path = None
    if embeddings is not None:
        if HAS_H5PY:
            embeddings_path = output_dir / "embeddings.h5"
            logger.info(f"Saving embeddings to {embeddings_path} (HDF5 with compression)...")
            with h5py.File(embeddings_path, "w") as hf:
                hf.create_dataset(
                    "embeddings",
                    data=embeddings,
                    dtype="float16",
                    compression="gzip",
                    compression_opts=4,
                )
                hf.attrs["num_samples"] = num_samples
                hf.attrs["hidden_dim"] = embeddings.shape[1]
                hf.attrs["created_at"] = datetime.now().isoformat()
        else:
            embeddings_path = output_dir / "embeddings.npy"
            logger.info(f"Saving embeddings to {embeddings_path} (numpy format)...")
            np.save(embeddings_path, embeddings)

    # 3. Create train/val split
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_ratio)

    split = {
        "train_indices": indices[val_size:].tolist(),
        "val_indices": indices[:val_size].tolist(),
        "train_size": num_samples - val_size,
        "val_size": val_size,
        "seed": seed,
    }

    split_path = output_dir / "train_val_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(f"Train/Val split: {split['train_size']:,} / {split['val_size']:,}")

    # 4. Save manifest with statistics
    domain_dist = Counter(s["domain"] for s in samples)
    subdomain_dist = Counter(s["subdomain"] for s in samples)

    # 3-Layer Constitution statistics
    family_value_dist = Counter(s.get("family_value", "universal") for s in samples)
    individual_pref_dist = Counter(s.get("individual_pref", "default") for s in samples)
    situational_context_dist = Counter(s.get("situational_context", "normal") for s in samples)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_samples": num_samples,
        "domain_distribution": dict(domain_dist.most_common()),
        "subdomain_distribution": dict(subdomain_dist.most_common(20)),
        "avg_input_length": float(np.mean([len(s["input_text"]) for s in samples])),
        "avg_output_length": float(np.mean([len(s["counterfactual_full_text"]) for s in samples])),
        "train_size": split["train_size"],
        "val_size": split["val_size"],
        # 3-Layer Constitution distributions
        "constitution": {
            "family_value_distribution": dict(family_value_dist.most_common()),
            "individual_pref_distribution": dict(individual_pref_dist.most_common()),
            "situational_context_distribution": dict(situational_context_dist.most_common()),
            "num_family_values": len(family_value_dist),
            "num_individual_prefs": len(individual_pref_dist),
            "num_situational_contexts": len(situational_context_dist),
        },
    }

    # Add embedding info based on mode
    if sequence_metadata:
        manifest["mode"] = "full_sequence"
        manifest.update(sequence_metadata)
    elif embeddings is not None:
        manifest["mode"] = "pooled"
        manifest["hidden_dim"] = embeddings.shape[1]
        manifest["embeddings_dtype"] = "float16"
        manifest["embeddings_size_mb"] = embeddings.nbytes / 1024 / 1024

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nSaved training data to {output_dir}")
    logger.info(f"  - samples.jsonl: {samples_path.stat().st_size / 1024 / 1024:.1f} MB")
    if embeddings_path and embeddings_path.exists():
        logger.info(f"  - {embeddings_path.name}: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  - Total samples: {num_samples:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare decoder training data with UltraBERT embeddings"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/counterfactual/synthetic"),
        help="Directory with counterfactual shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/counterfactual/training"),
        help="Output directory for training data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,  # Will use bundled weights from familyos_ultrabert
        help="Path to UltraBERT checkpoint. If not provided, uses bundled weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for embeddings (ignored if --full-sequence)",
    )
    parser.add_argument(
        "--full-sequence",
        action="store_true",
        help="Save full sequence embeddings (N, seq_len, 768) for decoder cross-attention. "
             "Uses more storage but enables position-aware attention.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to first N samples (for testing/debugging)",
    )

    args = parser.parse_args()

    # Load samples
    samples = load_counterfactual_samples(args.input_dir)

    if not samples:
        logger.error("No valid samples found!")
        return

    # Limit samples if requested (for testing)
    if args.max_samples and len(samples) > args.max_samples:
        logger.info(f"Limiting to first {args.max_samples} samples (from {len(samples):,})")
        samples = samples[:args.max_samples]

    # Load UltraBERT engine using installed familyos_ultrabert library
    engine = load_ultrabert_encoder(args.model_path, args.device)

    if args.full_sequence:
        # Full sequence mode: save (total_tokens, 768) with offsets
        logger.info("=" * 60)
        logger.info("MODE: Full Sequence Embeddings (for cross-attention)")
        logger.info("=" * 60)

        sequence_metadata = extract_full_sequence_embeddings(
            samples=samples,
            engine=engine,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Save training data (samples + split, no pooled embeddings)
        save_training_data(
            samples=samples,
            embeddings=None,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            sequence_metadata=sequence_metadata,
        )
    else:
        # Pooled mode: save (N, 768)
        logger.info("=" * 60)
        logger.info(f"MODE: Pooled Embeddings ({args.pooling.upper()})")
        logger.info("=" * 60)

        embeddings = extract_embeddings(
            samples=samples,
            engine=engine,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        )

        # Save training data
        save_training_data(
            samples=samples,
            embeddings=embeddings,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
        )

    logger.info("\nDone! Training data is ready for Stage C decoder training.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Embedding Head Bake-Off Training Script

Trains any of the 6 candidate embedding heads from heads_embedding.py
under identical conditions for fair comparison.

Derived from train_embedding_head.py with these additions:
    - Config-driven head selection via embedding_head.head_type
    - Custom head metadata saved per checkpoint for correct reload
    - Shared-encoder joint bake-off mode via --run_all
    - Optional sequential fallback via --run_sequential
    - All other behavior (data loading, loss, freezing, eval) is identical

Usage:
    # Single experiment
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --head_type agreement_gated

    # Train all configured heads together (shared encoder pass)
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --run_all

    # Legacy sequential mode
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --run_sequential

    # Debug mode (small dataset, one head)
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --head_type mean_baseline --debug --max_samples 500

Architecture:
    checkpoint-8000 (ModernBertMultiTaskModel with 12 heads)
        |
        +-- encoder [FROZEN]
        +-- 11 task heads [FROZEN]
        |
        +-- embedding head [REPLACED with candidate, TRAINABLE]

Output: per-head checkpoints and a combined bake-off summary
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from modeling_studio.models.modernbert_multitask import (
    ModernBertMultiTaskModel,
    Capability,
)
from modeling_studio.models.heads import EmbeddingHead, GlobalPointerNERHead, create_globalpointer_head

# Import bake-off head registry and factory
from modeling_studio.models.heads_embedding import (
    EMBEDDING_HEAD_REGISTRY,
    create_embedding_head,
    get_head_constructor_params,
)

from familyos_ultrabert.models.losses import FamilyContrastiveLoss

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tokenizers").setLevel(logging.WARNING)


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# =============================================================================
# Dataset - Unified loader for triplets AND pairs
# =============================================================================

# Prefixes to skip when scanning JSONL files in a directory
_SKIP_PREFIXES = ("hash_index", "manifest", "stats", "metadata")


class EmbeddingDataset(Dataset):
    """Unified dataset that loads triplets AND query-document pairs.

    Handles three record schemas automatically:
        A) Standard triplets:  anchor / positive / negative
        B) Mined triplets:     anchor / positive / negative + hard_negative_type
        C) Query-doc pairs:    query / document (no negative)

    Every record is normalized to a common dict so downstream code only
    sees one shape.

    Args:
        sources: List of dicts, each with keys ``path`` (Path), ``slice``
            (str tag), and ``format`` (``"triplet"`` or ``"pair"``).
            When *None*, falls back to flat ``data_paths`` loading (legacy).
        data_paths: Legacy flat list of directories (used when *sources*
            is None).  All paths treated as triplet format.
        max_samples: Global cap on total samples (for debugging).
    """

    def __init__(
        self,
        sources: list[dict[str, Any]] | None = None,
        data_paths: list[Path] | None = None,
        max_samples: int | None = None,
    ):
        self.samples: list[dict[str, Any]] = []
        self.slice_counts: dict[str, int] = {}

        if sources is not None:
            self._load_sources(sources, max_samples)
        elif data_paths is not None:
            # Legacy mode: treat every path as a triplet source
            legacy_sources = [
                {"path": p, "slice": p.name, "format": "triplet"}
                for p in data_paths
            ]
            self._load_sources(legacy_sources, max_samples)
        else:
            raise ValueError("EmbeddingDataset requires either sources or data_paths")

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        random.shuffle(self.samples)

        # Log summary
        total = len(self.samples)
        logger.info(f"  EmbeddingDataset: {total:,} samples from {len(self.slice_counts)} slices")
        for slice_name, count in sorted(self.slice_counts.items()):
            pct = 100.0 * count / total if total else 0
            logger.info(f"    {slice_name:<25} {count:>8,}  ({pct:5.1f}%)")

    # ------------------------------------------------------------------ #
    # Internal loaders
    # ------------------------------------------------------------------ #

    def _load_sources(
        self,
        sources: list[dict[str, Any]],
        max_samples: int | None,
    ) -> None:
        for src in sources:
            path = Path(src["path"])
            slice_name = src.get("slice", path.name)
            fmt = src.get("format", "triplet")
            if not path.exists():
                logger.warning(f"Data source not found: {path}")
                continue

            count_before = len(self.samples)

            if path.is_dir():
                jsonl_files = sorted(path.glob("*.jsonl"))
                if not jsonl_files:
                    jsonl_files = sorted(path.glob("**/*.jsonl"))
                for jsonl_file in jsonl_files:
                    self._load_jsonl(jsonl_file, slice_name, fmt, max_samples)
                    if max_samples and len(self.samples) >= max_samples:
                        break
            elif path.suffix == ".jsonl":
                self._load_jsonl(path, slice_name, fmt, max_samples)

            loaded = len(self.samples) - count_before
            self.slice_counts[slice_name] = self.slice_counts.get(slice_name, 0) + loaded

            if max_samples and len(self.samples) >= max_samples:
                break

    def _load_jsonl(
        self,
        path: Path,
        slice_name: str,
        fmt: str,
        max_samples: int | None,
    ) -> None:
        if any(path.name.startswith(p) for p in _SKIP_PREFIXES):
            return

        is_hard_neg_dir = "hard_negative" in str(path.parent) or "hard_negative" in path.name
        count_before = len(self.samples)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = self._normalize_record(raw, slice_name, fmt, is_hard_neg_dir)
                if record is not None:
                    self.samples.append(record)

        loaded = len(self.samples) - count_before
        if loaded > 0:
            logger.info(f"    {path.name}: {loaded} ({fmt}, slice={slice_name})")

    @staticmethod
    def _normalize_record(
        raw: dict[str, Any],
        slice_name: str,
        fmt: str,
        is_hard_neg_dir: bool,
    ) -> dict[str, Any] | None:
        """Convert any raw schema to the unified record format."""
        if fmt == "pair":
            # Schema C: query / document
            anchor = raw.get("query", "")
            positive = raw.get("document", "")
            if not anchor or not positive:
                return None
            return {
                "anchor": anchor,
                "positive": positive,
                "negative": None,
                "has_negative": False,
                "is_hard_negative": False,
                "slice": slice_name,
                "difficulty": raw.get("difficulty", "unknown"),
            }
        else:
            # Schema A/B: anchor / positive / negative
            anchor = raw.get("anchor", "")
            positive = raw.get("positive", "")
            negative = raw.get("negative", "")
            if not anchor or not positive or not negative:
                return None
            return {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "has_negative": True,
                "is_hard_negative": is_hard_neg_dir or bool(raw.get("hard_negative_type")),
                "slice": slice_name,
                "difficulty": raw.get("difficulty", "unknown"),
            }

    # ------------------------------------------------------------------ #
    # Per-slice access helpers (used by eval holdout and sampler)
    # ------------------------------------------------------------------ #

    def get_indices_by_slice(self) -> dict[str, list[int]]:
        """Return mapping of slice_name -> list of sample indices."""
        slices: dict[str, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            s = sample["slice"]
            if s not in slices:
                slices[s] = []
            slices[s].append(idx)
        return slices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class EmbeddingCollator:
    """Tokenize mixed triplet + pair batches.

    Produces a batch dict with separate index tensors for triplet and pair
    samples so the training step can route them to different loss paths.
    """

    def __init__(self, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        anchors = [f["anchor"] for f in features]
        positives = [f["positive"] for f in features]
        has_neg = [f.get("has_negative", True) for f in features]
        hard_neg_flags = [f.get("is_hard_negative", False) for f in features]
        slice_tags = [f.get("slice", "unknown") for f in features]

        # Separate triplet vs pair indices
        triplet_indices = [i for i, hn in enumerate(has_neg) if hn]
        pair_indices = [i for i, hn in enumerate(has_neg) if not hn]

        # Tokenize anchor and positive for ALL samples
        anchor_enc = self.tokenizer(
            anchors, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        positive_enc = self.tokenizer(
            positives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        batch = {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "has_negative": torch.tensor(has_neg, dtype=torch.bool),
            "triplet_indices": torch.tensor(triplet_indices, dtype=torch.long),
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
            "slice_tags": slice_tags,
        }

        # Tokenize negatives ONLY for triplet samples
        if triplet_indices:
            negatives = [features[i]["negative"] for i in triplet_indices]
            trip_hard_neg = [hard_neg_flags[i] for i in triplet_indices]
            negative_enc = self.tokenizer(
                negatives, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            batch["negative_input_ids"] = negative_enc["input_ids"]
            batch["negative_attention_mask"] = negative_enc["attention_mask"]
            batch["hard_negative_mask"] = torch.tensor(trip_hard_neg, dtype=torch.bool)
        else:
            # Pure pair batch - no negatives
            batch["negative_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["negative_attention_mask"] = torch.zeros(0, 1, dtype=torch.long)
            batch["hard_negative_mask"] = torch.zeros(0, dtype=torch.bool)

        return batch


# =============================================================================
# Data loading helpers
# =============================================================================


def get_embedding_data_sources(
    data_config: dict[str, Any],
    data_root: Path,
) -> list[dict[str, Any]]:
    """Build source list from new ``data.sources`` schema or legacy fallback.

    Returns:
        List of dicts with keys ``path`` (resolved Path), ``slice``, ``format``.
    """
    sources_cfg = data_config.get("sources")
    if sources_cfg is not None:
        resolved = []
        for src in sources_cfg:
            raw_path = src["path"]
            path = data_root / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
            resolved.append({
                "path": path,
                "slice": src.get("slice", path.name),
                "format": src.get("format", "triplet"),
            })
        return resolved

    # Legacy fallback: comma-separated paths in data.embedding.train
    embedding_config = data_config.get("embedding", {})
    train_paths_str = embedding_config.get("train", "")
    if not train_paths_str:
        raise ValueError("No embedding training data paths in config (need data.sources or data.embedding.train)")
    path_strings = [p.strip() for p in train_paths_str.split(",") if p.strip()]
    resolved = []
    for path_str in path_strings:
        path = data_root / path_str if not Path(path_str).is_absolute() else Path(path_str)
        if path.exists():
            resolved.append({"path": path, "slice": path.name, "format": "triplet"})
        else:
            logger.warning(f"Data path not found: {path}")
    return resolved


def get_embedding_data_paths(data_config: dict, data_root: Path) -> list[Path]:
    """Legacy helper - returns flat path list for backward compat."""
    sources = get_embedding_data_sources(data_config, data_root)
    return [s["path"] for s in sources]


def build_train_val_datasets(
    data_config: dict[str, Any],
    data_root: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple["EmbeddingDataset", "EmbeddingDataset", list[dict[str, Any]]]:
    """Build training and validation datasets with per-slice holdout.

    Instead of a single random split across the whole pool, holds out a
    fixed percentage of each slice so every slice is guaranteed
    representation in the eval set.

    Args:
        data_config: The ``data`` section of the YAML config.
        data_root: Root data directory.
        max_samples: Global cap on total samples (debug).
        seed: Random seed for reproducible splits.

    Returns:
        (train_dataset, val_dataset, query_doc_eval_samples)
        where query_doc_eval_samples is a flat list of raw dicts for
        retrieval eval (Recall@k).
    """
    eval_split = data_config.get("eval_split_per_slice", 0.15)
    sources = get_embedding_data_sources(data_config, data_root)

    full_dataset = EmbeddingDataset(sources=sources, max_samples=max_samples)
    indices_by_slice = full_dataset.get_indices_by_slice()

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    query_doc_eval_samples: list[dict[str, Any]] = []

    logger.info(f"  Per-slice eval holdout (split={eval_split:.0%}):")
    for slice_name in sorted(indices_by_slice.keys()):
        idxs = indices_by_slice[slice_name][:]
        rng.shuffle(idxs)
        n_val = max(1, int(math.ceil(len(idxs) * eval_split)))
        slice_val = idxs[:n_val]
        slice_train = idxs[n_val:]
        val_indices.extend(slice_val)
        train_indices.extend(slice_train)
        logger.info(f"    {slice_name:<25} train={len(slice_train):>6,}  val={len(slice_val):>6,}")

        # Collect query_doc eval samples for retrieval eval
        if slice_name == "query_doc":
            query_doc_eval_samples = [full_dataset.samples[i] for i in slice_val]

    # Build subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    logger.info(f"  Split totals: train={len(train_indices):,}  val={len(val_indices):,}")
    return train_dataset, val_dataset, query_doc_eval_samples


def load_checkpoint_capabilities(
    checkpoint_path: Path,
    exclude_decoder: bool = True,
) -> list[Capability]:
    capabilities_file = checkpoint_path / "capabilities.json"
    if capabilities_file.exists():
        with open(capabilities_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            capabilities = [Capability(value) for value in data]
        else:
            capabilities = [Capability(value) for value in data.get("capabilities", [])]
    else:
        capabilities = list(Capability)
    if exclude_decoder:
        capabilities = [cap for cap in capabilities if cap != Capability.COUNTERFACTUAL]
        logger.info("Excluding GPT-2 decoder (COUNTERFACTUAL) - saves 355M params")
    return capabilities


def restore_checkpoint_head_architecture(
    model: ModernBertMultiTaskModel,
    checkpoint_path: Path,
) -> dict[str, Any] | None:
    metadata_path = checkpoint_path / "globalpointer_metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    head_info = metadata.get("head_info", {})
    hidden_size = getattr(model.config, "hidden_size", 768)
    restored_heads: list[str] = []
    for head_name, info in head_info.items():
        if head_name not in model.heads:
            continue
        if info.get("class") != "GlobalPointerNERHead":
            continue
        head_size = info.get("head_size", 64)
        model.heads[head_name] = create_globalpointer_head(
            capability=head_name, hidden_size=hidden_size, head_size=head_size,
        )
        restored_heads.append(head_name)
    if restored_heads:
        logger.info(f"  Restored checkpoint head classes: {', '.join(restored_heads)}")
        setattr(model, "_checkpoint_globalpointer_metadata", metadata)
    return metadata


def load_model_and_replace_embedding_head(
    checkpoint_path: str | Path,
    head_type: str,
    head_params: dict[str, Any],
    exclude_decoder: bool = True,
    use_flash_attention: bool = False,
) -> ModernBertMultiTaskModel:
    """Load model and replace embedding head with a bake-off candidate.

    Args:
        checkpoint_path: Path to source checkpoint.
        head_type: Registry key for the embedding head (e.g. 'agreement_gated').
        head_params: Head-specific constructor kwargs.
        exclude_decoder: Skip GPT-2 decoder to save memory.
        use_flash_attention: Enable Flash Attention 2.

    Returns:
        Model with the specified embedding head installed.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading model from {checkpoint_path}")

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    if use_flash_attention:
        config.attn_implementation = "flash_attention_2"
        logger.info("  Flash Attention 2: ENABLED")

    capabilities = load_checkpoint_capabilities(checkpoint_path, exclude_decoder)
    model = ModernBertMultiTaskModel(config=config, capabilities=capabilities, freeze_encoder=False)
    restore_checkpoint_head_architecture(model, checkpoint_path)
    model._init_encoder()

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        weights_path = checkpoint_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    # Encoder weights
    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"  Loaded encoder: {len(encoder_state)} tensors")

    # Head weights
    for head_name in model.heads.keys():
        head_prefix = f"heads.{head_name}."
        head_state = {k.replace(head_prefix, ""): v for k, v in state_dict.items() if k.startswith(head_prefix)}
        if head_state:
            try:
                model.heads[head_name].load_state_dict(head_state, strict=True)
            except Exception as e:
                logger.warning(f"Could not load {head_name} head: {e}")

    hidden_size = model.config.hidden_size
    logger.info(f"  Loaded {len(model.heads)} heads, hidden_size={hidden_size}")

    # Replace embedding head with bake-off candidate
    old_class = type(model.heads["embedding"]).__name__
    old_params = sum(p.numel() for p in model.heads["embedding"].parameters())

    new_head = create_embedding_head(
        head_type=head_type,
        hidden_size=hidden_size,
        **head_params,
    )
    model.heads["embedding"] = new_head
    new_params = sum(p.numel() for p in new_head.parameters())

    logger.info(f"  Replaced embedding head:")
    logger.info(f"    Old: {old_class} ({old_params:,} params)")
    logger.info(f"    New: {type(new_head).__name__}[{head_type}] ({new_params:,} params)")

    return model


def freeze_model_except_embedding_head(model: ModernBertMultiTaskModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for name, head in model.heads.items():
        for param in head.parameters():
            param.requires_grad = (name == "embedding")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    emb_params = sum(p.numel() for p in model.heads["embedding"].parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_heads = [n for n in model.heads.keys() if n != "embedding"]
    logger.info(f"  Encoder: {encoder_params:,} params (frozen)")
    logger.info(f"  Frozen heads: {', '.join(frozen_heads)} ({len(frozen_heads)} heads)")
    logger.info(f"  Embedding head: {emb_params:,} params (TRAINABLE)")
    logger.info(f"  Total trainable: {trainable_total:,} params")


def get_trainable_params(model: ModernBertMultiTaskModel) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def freeze_base_model_for_joint_bakeoff(model: ModernBertMultiTaskModel) -> None:
    """Freeze encoder and all built-in heads for shared multi-head training."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for head in model.heads.values():
        for param in head.parameters():
            param.requires_grad = False

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    frozen_heads = list(model.heads.keys())
    logger.info(f"  Encoder: {encoder_params:,} params (frozen)")
    logger.info(f"  Built-in frozen heads: {', '.join(frozen_heads)} ({len(frozen_heads)} heads)")


def count_parameters(module: nn.Module) -> int:
    """Count parameters in a module."""
    return sum(param.numel() for param in module.parameters())


def merge_head_params(
    default_params: dict[str, Any],
    experiment_params: dict[str, Any],
) -> dict[str, Any]:
    """Merge top-level and experiment-specific head parameters."""
    merged_params = {**default_params, **experiment_params}
    merged_params.pop("head_type", None)
    merged_params.pop("pooling", None)
    return merged_params


def get_configured_head_experiments(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return configured head experiments with merged parameters."""
    experiments_config = config.get("experiments", {})
    default_params = config.get("embedding_head", {})
    experiments = experiments_config.get("heads", [])
    return [
        (exp["head_type"], merge_head_params(default_params, exp.get("params", {})))
        for exp in experiments
    ]


def encode_triplet_batch(
    model: ModernBertMultiTaskModel,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Encode anchor/positive/negative text once with the frozen encoder."""
    anchor_ids = batch["anchor_input_ids"].to(device)
    anchor_mask = batch["anchor_attention_mask"].to(device)
    positive_ids = batch["positive_input_ids"].to(device)
    positive_mask = batch["positive_attention_mask"].to(device)
    negative_ids = batch["negative_input_ids"].to(device)
    negative_mask = batch["negative_attention_mask"].to(device)
    hard_neg_mask = batch["hard_negative_mask"].to(device)

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    with amp_context:
        with torch.no_grad():
            enc_out = model.encoder(input_ids=anchor_ids, attention_mask=anchor_mask)
            anchor_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)
            enc_out = model.encoder(input_ids=positive_ids, attention_mask=positive_mask)
            positive_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)
            enc_out = model.encoder(input_ids=negative_ids, attention_mask=negative_mask)
            negative_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

    return {
        "anchor_hidden": anchor_hidden,
        "anchor_mask": anchor_mask,
        "positive_hidden": positive_hidden,
        "positive_mask": positive_mask,
        "negative_hidden": negative_hidden,
        "negative_mask": negative_mask,
        "hard_neg_mask": hard_neg_mask,
    }


# =============================================================================
# Training Step
# =============================================================================


def train_step(
    model: ModernBertMultiTaskModel,
    batch: dict,
    loss_fn: FamilyContrastiveLoss,
    device: torch.device,
    debug: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    matryoshka_dims: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
    anchor_hidden = encoded["anchor_hidden"]
    anchor_mask = encoded["anchor_mask"]
    positive_hidden = encoded["positive_hidden"]
    positive_mask = encoded["positive_mask"]
    negative_hidden = encoded["negative_hidden"]
    negative_mask = encoded["negative_mask"]
    hard_neg_mask = encoded["hard_neg_mask"]

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    with amp_context:

        embedding_head = model.heads["embedding"]
        anchor_emb = embedding_head(anchor_hidden, anchor_mask)
        positive_emb = embedding_head(positive_hidden, positive_mask)
        negative_emb = embedding_head(negative_hidden, negative_mask)

        if matryoshka_dims:
            total_loss = 0.0
            for dim in matryoshka_dims:
                a_d = F.normalize(anchor_emb[:, :dim], p=2, dim=-1)
                p_d = F.normalize(positive_emb[:, :dim], p=2, dim=-1)
                n_d = F.normalize(negative_emb[:, :dim], p=2, dim=-1).unsqueeze(1)
                hn_mask = hard_neg_mask.unsqueeze(1)
                dim_loss = loss_fn(anchor=a_d, positive=p_d, negatives=n_d, hard_negative_mask=hn_mask)
                total_loss = total_loss + dim_loss
            loss = total_loss / len(matryoshka_dims)
        else:
            negatives = negative_emb.unsqueeze(1)
            hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)
            loss = loss_fn(
                anchor=anchor_emb, positive=positive_emb,
                negatives=negatives, hard_negative_mask=hard_neg_mask_expanded,
            )

    with torch.no_grad():
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb).mean().item()
        margin = pos_sim - neg_sim

    result = {"total_loss": loss, "pos_sim": pos_sim, "neg_sim": neg_sim, "margin": margin}
    if debug:
        logger.debug(f"  loss={loss.item():.4f} pos_sim={pos_sim:.4f} neg_sim={neg_sim:.4f} margin={margin:.4f}")
    return result


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(
    model: ModernBertMultiTaskModel,
    val_loader: DataLoader,
    loss_fn: FamilyContrastiveLoss,
    device: torch.device,
    debug: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_correct = 0
    total_samples = 0
    total_hard_neg_correct = 0
    total_hard_neg_samples = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
        anchor_hidden = encoded["anchor_hidden"]
        anchor_mask = encoded["anchor_mask"]
        positive_hidden = encoded["positive_hidden"]
        positive_mask = encoded["positive_mask"]
        negative_hidden = encoded["negative_hidden"]
        negative_mask = encoded["negative_mask"]
        hard_neg_mask = encoded["hard_neg_mask"]

        with amp_context:
            embedding_head = model.heads["embedding"]
            anchor_emb = embedding_head(anchor_hidden, anchor_mask)
            positive_emb = embedding_head(positive_hidden, positive_mask)
            negative_emb = embedding_head(negative_hidden, negative_mask)

            negatives = negative_emb.unsqueeze(1)
            hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)
            batch_loss = loss_fn(
                anchor=anchor_emb, positive=positive_emb,
                negatives=negatives, hard_negative_mask=hard_neg_mask_expanded,
            )

        pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
        correct = (pos_sim > neg_sim).float()
        batch_size = anchor_emb.size(0)
        total_loss += batch_loss.item() * batch_size
        total_pos_sim += pos_sim.sum().item()
        total_neg_sim += neg_sim.sum().item()
        total_correct += correct.sum().item()
        total_samples += batch_size
        if hard_neg_mask.any():
            total_hard_neg_correct += correct[hard_neg_mask].sum().item()
            total_hard_neg_samples += hard_neg_mask.sum().item()

    model.train()
    if total_samples == 0:
        return {"val_loss": 0, "pos_sim": 0, "neg_sim": 0, "margin": 0, "accuracy": 0}

    avg_pos_sim = total_pos_sim / total_samples
    avg_neg_sim = total_neg_sim / total_samples
    metrics = {
        "val_loss": total_loss / total_samples,
        "pos_sim": avg_pos_sim,
        "neg_sim": avg_neg_sim,
        "margin": avg_pos_sim - avg_neg_sim,
        "accuracy": total_correct / total_samples,
        "hard_neg_accuracy": total_hard_neg_correct / total_hard_neg_samples if total_hard_neg_samples > 0 else 0.0,
        "hard_neg_samples": total_hard_neg_samples,
        "total_samples": total_samples,
    }
    logger.info(
        f"  val_loss={metrics['val_loss']:.4f} | pos_sim={avg_pos_sim:.4f} neg_sim={avg_neg_sim:.4f} "
        f"margin={metrics['margin']:.4f} | acc={metrics['accuracy']:.4f} "
        f"hard_neg_acc={metrics['hard_neg_accuracy']:.4f}"
    )
    return metrics


@torch.no_grad()
def evaluate_all_heads(
    model: ModernBertMultiTaskModel,
    candidate_heads: nn.ModuleDict,
    loss_modules: nn.ModuleDict,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all candidate heads side-by-side on the same validation set."""
    model.eval()
    candidate_heads.eval()

    totals: dict[str, dict[str, float]] = {
        head_type: {
            "loss": 0.0,
            "pos_sim": 0.0,
            "neg_sim": 0.0,
            "correct": 0.0,
            "samples": 0.0,
            "hard_neg_correct": 0.0,
            "hard_neg_samples": 0.0,
        }
        for head_type in candidate_heads.keys()
    }

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
        anchor_hidden = encoded["anchor_hidden"]
        anchor_mask = encoded["anchor_mask"]
        positive_hidden = encoded["positive_hidden"]
        positive_mask = encoded["positive_mask"]
        negative_hidden = encoded["negative_hidden"]
        negative_mask = encoded["negative_mask"]
        hard_neg_mask = encoded["hard_neg_mask"]

        with amp_context:
            for head_type, head in candidate_heads.items():
                anchor_emb = head(anchor_hidden, anchor_mask)
                positive_emb = head(positive_hidden, positive_mask)
                negative_emb = head(negative_hidden, negative_mask)

                negatives = negative_emb.unsqueeze(1)
                hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)
                batch_loss = loss_modules[head_type](
                    anchor=anchor_emb,
                    positive=positive_emb,
                    negatives=negatives,
                    hard_negative_mask=hard_neg_mask_expanded,
                )

                pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
                neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
                correct = (pos_sim > neg_sim).float()
                batch_size = anchor_emb.size(0)

                totals[head_type]["loss"] += batch_loss.item() * batch_size
                totals[head_type]["pos_sim"] += pos_sim.sum().item()
                totals[head_type]["neg_sim"] += neg_sim.sum().item()
                totals[head_type]["correct"] += correct.sum().item()
                totals[head_type]["samples"] += batch_size

                if hard_neg_mask.any():
                    totals[head_type]["hard_neg_correct"] += correct[hard_neg_mask].sum().item()
                    totals[head_type]["hard_neg_samples"] += hard_neg_mask.sum().item()

    model.train()
    candidate_heads.train()

    metrics_by_head: dict[str, dict[str, Any]] = {}
    for head_type, total in totals.items():
        sample_count = int(total["samples"])
        if sample_count == 0:
            metrics_by_head[head_type] = {
                "val_loss": 0.0,
                "pos_sim": 0.0,
                "neg_sim": 0.0,
                "margin": 0.0,
                "accuracy": 0.0,
                "hard_neg_accuracy": 0.0,
                "hard_neg_samples": 0,
                "total_samples": 0,
            }
            continue

        avg_pos_sim = total["pos_sim"] / sample_count
        avg_neg_sim = total["neg_sim"] / sample_count
        hard_neg_samples = int(total["hard_neg_samples"])
        metrics_by_head[head_type] = {
            "val_loss": total["loss"] / sample_count,
            "pos_sim": avg_pos_sim,
            "neg_sim": avg_neg_sim,
            "margin": avg_pos_sim - avg_neg_sim,
            "accuracy": total["correct"] / sample_count,
            "hard_neg_accuracy": total["hard_neg_correct"] / hard_neg_samples if hard_neg_samples > 0 else 0.0,
            "hard_neg_samples": hard_neg_samples,
            "total_samples": sample_count,
        }

    return metrics_by_head


def log_head_leaderboard(
    metrics_by_head: dict[str, dict[str, Any]],
    title: str,
) -> None:
    """Log a compact per-head leaderboard."""
    def metric_value(metrics: dict[str, Any], primary: str, fallback: str) -> float:
        value = metrics.get(primary, metrics.get(fallback, 0.0))
        return float(value)

    log_section(title)
    logger.info(f"{'Rank':<6}{'Head':<25}{'Margin':<10}{'Acc':<10}{'HardNeg':<10}{'ValLoss':<10}")
    logger.info("-" * 71)
    sorted_heads = sorted(
        metrics_by_head.items(),
        key=lambda item: metric_value(item[1], "margin", "best_margin"),
        reverse=True,
    )
    for rank, (head_type, metrics) in enumerate(sorted_heads, 1):
        margin = metric_value(metrics, "margin", "best_margin")
        accuracy = metric_value(metrics, "accuracy", "best_accuracy")
        hard_neg_accuracy = metric_value(metrics, "hard_neg_accuracy", "best_hard_neg_accuracy")
        val_loss = metric_value(metrics, "val_loss", "best_val_loss")
        logger.info(
            f"{rank:<6}{head_type:<25}{margin:<10.4f}{accuracy:<10.4f}"
            f"{hard_neg_accuracy:<10.4f}{val_loss:<10.4f}"
        )


# =============================================================================
# Training Loop
# =============================================================================


def train(
    model: ModernBertMultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: FamilyContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    num_epochs: int,
    output_dir: Path,
    tokenizer: Any = None,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 50,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    debug: bool = False,
    use_amp: bool = True,
    use_bf16: bool = False,
    use_ema: bool = True,
    early_stopping_patience: int = 5,
    max_eval_batches: int | None = 100,
    curriculum_config: dict | None = None,
    matryoshka_dims: list[int] | None = None,
    base_hard_negative_weight: float = 1.5,
    head_type: str = "mean_baseline",
    head_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = use_amp and device.type == "cuda" and not use_bf16
    scaler = GradScaler("cuda", enabled=use_scaler)

    ema_model = None
    if use_ema:
        ema_model = AveragedModel(model)

    model.train()
    global_step = 0
    best_margin = -1.0
    no_improve_count = 0
    history: dict[str, list] = {
        "train_loss": [], "train_pos_sim": [], "train_neg_sim": [],
        "train_margin": [], "eval_metrics": [],
    }

    log_section(f"TRAINING: {head_type}")
    logger.info(f"  Epochs: {num_epochs} | Batches: {len(train_loader)} | Grad accum: {gradient_accumulation_steps}")

    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        if curriculum_config and curriculum_config.get("enabled", False):
            warmup_epochs = curriculum_config.get("warmup_epochs", num_epochs)
            scale = min(1.0, epoch / max(warmup_epochs - 1, 1))
            current_hn_weight = base_hard_negative_weight * scale
            loss_fn.hard_negative_weight = current_hn_weight
            logger.info(f"  Curriculum: hard_negative_weight={current_hn_weight:.3f}")

        if loss_fn.log_temperature.requires_grad:
            logger.info(f"  Learned temperature: {loss_fn.log_temperature.exp().item():.4f}")

        epoch_loss = 0.0
        epoch_pos_sim = 0.0
        epoch_neg_sim = 0.0
        epoch_steps = 0
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            step_debug = debug and (step < 5 or step % 100 == 0)
            losses = train_step(
                model, batch, loss_fn, device, debug=step_debug,
                use_amp=use_amp, amp_dtype=amp_dtype, matryoshka_dims=matryoshka_dims,
            )
            loss = losses["total_loss"]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                trainable_params = get_trainable_params(model)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if ema_model is not None:
                    ema_model.update_parameters(model)

                if global_step > 0 and global_step % save_steps == 0:
                    logger.info(f"\nSaving checkpoint at step {global_step}...")
                    save_bakeoff_checkpoint(model, output_dir / f"checkpoint-{global_step}", tokenizer, head_type, head_params, optimizer, scheduler)
                    if ema_model is not None:
                        save_bakeoff_checkpoint(ema_model.module, output_dir / f"checkpoint-{global_step}-ema", tokenizer, head_type, head_params)

                if global_step > 0 and global_step % eval_steps == 0 and val_loader is not None:
                    logger.info(f"\n--- Eval @ step {global_step} ---")
                    eval_metrics = evaluate(model, val_loader, loss_fn, device, debug=debug, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=max_eval_batches)
                    current_margin = eval_metrics["margin"]
                    history["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, **eval_metrics})

                    if current_margin > best_margin:
                        best_margin = current_margin
                        no_improve_count = 0
                        logger.info(f"New best margin={best_margin:.4f}! Saving...")
                        save_bakeoff_checkpoint(model, output_dir / "best", tokenizer, head_type, head_params, optimizer, scheduler)
                        if ema_model is not None:
                            save_bakeoff_checkpoint(ema_model.module, output_dir / "best-ema", tokenizer, head_type, head_params)
                    else:
                        no_improve_count += 1
                        logger.info(f"No improvement ({no_improve_count}/{early_stopping_patience})")

                    if no_improve_count >= early_stopping_patience:
                        logger.info(f"Early stopping triggered")
                        save_bakeoff_checkpoint(model, output_dir / "early-stop", tokenizer, head_type, head_params, optimizer, scheduler)
                        return history
                    model.train()

            epoch_loss += losses["total_loss"].item()
            epoch_pos_sim += losses["pos_sim"]
            epoch_neg_sim += losses["neg_sim"]
            epoch_steps += 1

            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{losses['total_loss'].item():.4f}", margin=f"{losses['margin']:.3f}", lr=f"{lr:.2e}")

            if global_step > 0 and global_step % logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_pos = epoch_pos_sim / epoch_steps
                avg_neg = epoch_neg_sim / epoch_steps
                logger.info(f"  Step {global_step}: loss={avg_loss:.4f} pos_sim={avg_pos:.4f} neg_sim={avg_neg:.4f} margin={avg_pos - avg_neg:.4f} lr={lr:.2e}")

        # Epoch summary
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            avg_pos = epoch_pos_sim / epoch_steps
            avg_neg = epoch_neg_sim / epoch_steps
            history["train_loss"].append(avg_loss)
            history["train_pos_sim"].append(avg_pos)
            history["train_neg_sim"].append(avg_neg)
            history["train_margin"].append(avg_pos - avg_neg)
            logger.info(f"Epoch {epoch + 1} summary: loss={avg_loss:.4f} margin={avg_pos - avg_neg:.4f}")

        # Full eval at epoch end
        if val_loader is not None:
            logger.info(f"--- Full eval epoch {epoch + 1} ---")
            eval_metrics = evaluate(model, val_loader, loss_fn, device, debug=debug, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=None)
            history["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, "full_eval": True, **eval_metrics})
            current_margin = eval_metrics["margin"]
            if current_margin > best_margin:
                best_margin = current_margin
                no_improve_count = 0
                logger.info(f"New best margin={best_margin:.4f}! Saving...")
                save_bakeoff_checkpoint(model, output_dir / "best", tokenizer, head_type, head_params, optimizer, scheduler)
                if ema_model is not None:
                    save_bakeoff_checkpoint(ema_model.module, output_dir / "best-ema", tokenizer, head_type, head_params)
            else:
                no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered")
                break

    log_section("TRAINING COMPLETE")
    logger.info(f"  Best margin: {best_margin:.4f}")
    save_bakeoff_checkpoint(model, output_dir / "final", tokenizer, head_type, head_params, optimizer, scheduler)
    if ema_model is not None:
        save_bakeoff_checkpoint(ema_model.module, output_dir / "final-ema", tokenizer, head_type, head_params)
    return history


# =============================================================================
# Checkpoint Saving (with bake-off metadata)
# =============================================================================


def save_bakeoff_checkpoint(
    model: ModernBertMultiTaskModel,
    checkpoint_dir: Path,
    tokenizer: Any = None,
    head_type: str = "mean_baseline",
    head_params: dict[str, Any] | None = None,
    optimizer: Any = None,
    scheduler: Any = None,
    embedding_head_override: nn.Module | None = None,
) -> None:
    """Save full model checkpoint with bake-off embedding head metadata."""
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {}

    for name, param in model.encoder.state_dict().items():
        state_dict[f"encoder.{name}"] = param
    for head_name, head in model.heads.items():
        current_head = embedding_head_override if head_name == "embedding" and embedding_head_override is not None else head
        for name, param in current_head.state_dict().items():
            state_dict[f"heads.{head_name}.{name}"] = param

    save_file(state_dict, checkpoint_dir / "model.safetensors")
    model.config.save_pretrained(checkpoint_dir)

    # Capabilities
    capabilities_payload = {
        "capabilities": [
            capability.value if isinstance(capability, Capability) else str(capability)
            for capability in getattr(model, "capabilities", [])
        ],
        "decoder_type": None,
        "epic_5_0": {
            "shared_pooler": getattr(model, "_shared_pooler_type", None),
            "use_adapters": getattr(model, "_use_adapters", False),
            "adapter_bottleneck_size": getattr(model, "_adapter_bottleneck_size", 64),
            "use_pair_encoder": getattr(model, "_use_pair_encoder", False),
            "pair_encoder_num_layers": getattr(model, "_pair_encoder_num_layers", 1),
        },
    }
    with open(checkpoint_dir / "capabilities.json", "w", encoding="utf-8") as f:
        json.dump(capabilities_payload, f, indent=2)

    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Bake-off embedding metadata (the key addition)
    embedding_head = embedding_head_override if embedding_head_override is not None else (model.heads["embedding"] if "embedding" in model.heads else None)
    head_constructor_params = get_head_constructor_params(embedding_head) if embedding_head is not None else {}

    embedding_metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "embedding_heads_bakeoff",
        "bakeoff": {
            "head_type": head_type,
            "head_class": type(embedding_head).__name__ if embedding_head else None,
            "head_params": head_params or {},
            "head_constructor_params": head_constructor_params,
        },
        "head_info": {},
        "trained_head": "embedding",
    }
    for hn, h in model.heads.items():
        current_head = embedding_head_override if hn == "embedding" and embedding_head_override is not None else h
        info = {"class": type(current_head).__name__}
        if hn == "embedding":
            info.update({
                "head_type": head_type,
                "pooling": getattr(current_head, "pooling", None),
                "output_dim": getattr(current_head, "output_dim", None),
                "hidden_size": getattr(current_head, "hidden_size", None),
                "normalize": getattr(current_head, "normalize", None),
            })
        embedding_metadata["head_info"][hn] = info

    with open(checkpoint_dir / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(embedding_metadata, f, indent=2)

    # Preserve GlobalPointer metadata
    checkpoint_gp = getattr(model, "_checkpoint_globalpointer_metadata", None)
    if checkpoint_gp is not None:
        with open(checkpoint_dir / "globalpointer_metadata.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint_gp, f, indent=2)
    else:
        gp_info = {}
        for hn, h in model.heads.items():
            if isinstance(h, GlobalPointerNERHead):
                gp_info[hn] = {"class": type(h).__name__, "num_labels": getattr(h, "num_labels", None), "head_size": getattr(h, "head_size", None)}
        if gp_info:
            with open(checkpoint_dir / "globalpointer_metadata.json", "w", encoding="utf-8") as f:
                json.dump({"replaced_heads": list(gp_info.keys()), "head_architecture": "GlobalPointerNERHead", "head_info": gp_info}, f, indent=2)

    logger.info(f"Saved: {checkpoint_dir.name} (head_type={head_type})")


def run_joint_bakeoff(
    config: dict[str, Any],
    output_base: Path,
    debug: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Train all configured embedding heads together with a shared encoder pass."""
    head_experiments = get_configured_head_experiments(config)
    if not head_experiments:
        raise ValueError("No experiments defined in config under experiments.heads")

    encoder_config = config.get("encoder", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})

    checkpoint_path = encoder_config.get("checkpoint", "checkpoints/checkpoint-8000")
    data_root = Path(data_config.get("root", "data"))
    learning_rate = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_epochs = training_config.get("num_epochs", 3)
    batch_size = training_config.get("batch_size", 128)
    max_length = data_config.get("max_length", 128)
    warmup_steps = training_config.get("warmup_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    logging_steps = training_config.get("logging_steps", 50)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    val_split = training_config.get("val_split", 0.1)
    num_workers = data_config.get("num_workers", 8)
    max_eval_batches = training_config.get("max_eval_batches", 100)
    use_bf16 = training_config.get("bf16", False)
    use_tf32 = training_config.get("tf32", False)
    use_flash_attention = training_config.get("flash_attention", False)
    ema_config = training_config.get("ema", {})
    use_ema = ema_config.get("enabled", True)
    es_config = training_config.get("early_stopping", {})
    early_stopping_enabled = es_config.get("enabled", True)
    early_stopping_patience = es_config.get("patience", 5)
    lr_scheduler_type = training_config.get("lr_scheduler_type", "cosine")
    temperature = loss_config.get("temperature", 0.07)
    hard_negative_weight = loss_config.get("hard_negative_weight", 1.5)
    learnable_temperature = loss_config.get("learnable_temperature", False)
    temperature_lr = loss_config.get("temperature_lr", 1e-3)
    curriculum_config = training_config.get("curriculum", {})

    if debug:
        max_samples = max_samples or 500
        eval_steps = 50

    seed = training_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: {gpu_name}")
        if use_tf32 and ("A100" in gpu_name or "H100" in gpu_name):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    log_section("JOINT BAKE-OFF MODEL")
    base_model = load_model_and_replace_embedding_head(
        checkpoint_path,
        head_type="mean_baseline",
        head_params={},
        use_flash_attention=use_flash_attention,
    )
    freeze_base_model_for_joint_bakeoff(base_model)
    base_model = base_model.to(device)

    hidden_size = base_model.config.hidden_size
    candidate_heads = nn.ModuleDict({
        head_type: create_embedding_head(head_type=head_type, hidden_size=hidden_size, **head_params)
        for head_type, head_params in head_experiments
    }).to(device)

    logger.info("  Candidate heads loaded together:")
    total_head_params = 0
    for head_type, head in candidate_heads.items():
        param_count = count_parameters(head)
        total_head_params += param_count
        logger.info(f"    {head_type:<25} {type(head).__name__:<28} {param_count:>10,} params")
    logger.info(f"  Total candidate head params: {total_head_params:,}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    log_section("DATA")
    train_dataset, val_dataset, query_doc_eval_samples = build_train_val_datasets(
        data_config=data_config,
        data_root=data_root,
        max_samples=max_samples,
        val_split=val_split,
        seed=seed,
    )
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset)} samples "
                f"(train={len(train_dataset)}, val={len(val_dataset)})")
    if query_doc_eval_samples:
        logger.info(f"  Query-doc eval pairs: {len(query_doc_eval_samples)}")

    collator = EmbeddingCollator(tokenizer=tokenizer, max_length=max_length)
    effective_workers = 0 if platform.system() == "Windows" else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
    )

    loss_modules = nn.ModuleDict({
        head_type: FamilyContrastiveLoss(
            temperature=temperature,
            hard_negative_weight=hard_negative_weight,
            use_hard_negatives=True,
            normalize=False,
        )
        for head_type, _ in head_experiments
    }).to(device)
    if learnable_temperature:
        for loss_module in loss_modules.values():
            loss_module.log_temperature.requires_grad_(True)

    param_groups = []
    for head_type, head in candidate_heads.items():
        head_params_list = [param for param in head.parameters() if param.requires_grad]
        if head_params_list:
            param_groups.append({"params": head_params_list, "lr": learning_rate, "weight_decay": weight_decay})
        if learnable_temperature:
            param_groups.append({"params": [loss_modules[head_type].log_temperature], "lr": temperature_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        eps=training_config.get("adam_epsilon", 1e-8),
    )

    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    adaptive_warmup = max(10, int(num_training_steps * 0.05))
    effective_warmup = min(warmup_steps, adaptive_warmup) if num_training_steps < warmup_steps else warmup_steps
    if lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=effective_warmup, num_training_steps=num_training_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=effective_warmup, num_training_steps=num_training_steps)

    logger.info(f"  Steps: {num_training_steps} total, {effective_warmup} warmup")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = device.type == "cuda" and not use_bf16
    scaler = GradScaler("cuda", enabled=use_scaler)
    ema_heads = nn.ModuleDict({head_type: copy.deepcopy(head) for head_type, head in candidate_heads.items()}).to(device) if use_ema else None
    if ema_heads is not None:
        ema_heads.eval()

    history: dict[str, Any] = {
        "mode": "joint_multi_head",
        "heads": {
            head_type: {
                "params": head_params,
                "train_loss": [],
                "train_pos_sim": [],
                "train_neg_sim": [],
                "train_margin": [],
                "eval_metrics": [],
            }
            for head_type, head_params in head_experiments
        },
    }
    best_margin = {head_type: -1.0 for head_type, _ in head_experiments}
    no_improve_count = {head_type: 0 for head_type, _ in head_experiments}
    trainable_heads = {
        head_type
        for head_type, head in candidate_heads.items()
        if count_parameters(head) > 0 or learnable_temperature
    }
    active_heads = set(trainable_heads)
    global_step = 0

    log_section("JOINT TRAINING")
    logger.info(f"  Heads trained together: {', '.join(candidate_heads.keys())}")
    logger.info(f"  Epochs: {num_epochs} | Batches: {len(train_loader)} | Grad accum: {gradient_accumulation_steps}")
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        logger.info(f"\n--- Joint epoch {epoch + 1}/{num_epochs} ---")

        if curriculum_config and curriculum_config.get("enabled", False):
            warmup_epochs = curriculum_config.get("warmup_epochs", num_epochs)
            scale = min(1.0, epoch / max(warmup_epochs - 1, 1))
            current_hn_weight = hard_negative_weight * scale
            for loss_module in loss_modules.values():
                loss_module.hard_negative_weight = current_hn_weight
            logger.info(f"  Curriculum: hard_negative_weight={current_hn_weight:.3f}")

        if learnable_temperature:
            temp_line = ", ".join(
                f"{head_type}={loss_modules[head_type].log_temperature.exp().item():.4f}"
                for head_type in candidate_heads.keys()
            )
            logger.info(f"  Learned temperatures: {temp_line}")

        epoch_totals = {
            head_type: {"loss": 0.0, "pos": 0.0, "neg": 0.0, "steps": 0}
            for head_type in candidate_heads.keys()
        }
        progress = tqdm(train_loader, desc=f"Joint epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            encoded = encode_triplet_batch(base_model, batch, device, use_amp=use_bf16, amp_dtype=amp_dtype)
            anchor_hidden = encoded["anchor_hidden"]
            anchor_mask = encoded["anchor_mask"]
            positive_hidden = encoded["positive_hidden"]
            positive_mask = encoded["positive_mask"]
            negative_hidden = encoded["negative_hidden"]
            negative_mask = encoded["negative_mask"]
            hard_neg_mask = encoded["hard_neg_mask"]

            amp_context = (
                autocast("cuda", dtype=amp_dtype, enabled=use_bf16)
                if device.type == "cuda"
                else autocast("cpu", enabled=False)
            )

            with amp_context:
                head_losses = {}
                batch_metrics = {}
                for head_type, head in candidate_heads.items():
                    anchor_emb = head(anchor_hidden, anchor_mask)
                    positive_emb = head(positive_hidden, positive_mask)
                    negative_emb = head(negative_hidden, negative_mask)

                    if config.get("training", {}).get("matryoshka", {}).get("enabled", False):
                        dims = config["training"]["matryoshka"].get("dims", [hidden_size])
                        total_loss = 0.0
                        for dim in dims:
                            a_d = F.normalize(anchor_emb[:, :dim], p=2, dim=-1)
                            p_d = F.normalize(positive_emb[:, :dim], p=2, dim=-1)
                            n_d = F.normalize(negative_emb[:, :dim], p=2, dim=-1).unsqueeze(1)
                            hn_mask = hard_neg_mask.unsqueeze(1)
                            total_loss = total_loss + loss_modules[head_type](
                                anchor=a_d,
                                positive=p_d,
                                negatives=n_d,
                                hard_negative_mask=hn_mask,
                            )
                        loss = total_loss / len(dims)
                    else:
                        negatives = negative_emb.unsqueeze(1)
                        hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)
                        loss = loss_modules[head_type](
                            anchor=anchor_emb,
                            positive=positive_emb,
                            negatives=negatives,
                            hard_negative_mask=hard_neg_mask_expanded,
                        )

                    pos_sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
                    neg_sim = F.cosine_similarity(anchor_emb, negative_emb).mean().item()
                    margin = pos_sim - neg_sim

                    head_losses[head_type] = loss
                    batch_metrics[head_type] = {
                        "loss": loss.item(),
                        "pos_sim": pos_sim,
                        "neg_sim": neg_sim,
                        "margin": margin,
                    }

            optim_losses = [head_losses[head_type] for head_type in active_heads if head_type in head_losses]
            if optim_losses:
                total_optim_loss = torch.stack(optim_losses).mean()
                if gradient_accumulation_steps > 1:
                    total_optim_loss = total_optim_loss / gradient_accumulation_steps
                if use_scaler:
                    scaler.scale(total_optim_loss).backward()
                else:
                    total_optim_loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    trainable_params = [param for param_group in optimizer.param_groups for param in param_group["params"]]
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if ema_heads is not None:
                        with torch.no_grad():
                            for head_type in candidate_heads.keys():
                                ema_state = ema_heads[head_type].state_dict()
                                current_state = candidate_heads[head_type].state_dict()
                                for key in ema_state.keys():
                                    ema_state[key].mul_(0.999).add_(current_state[key], alpha=0.001)

            best_batch_head = max(batch_metrics.items(), key=lambda item: item[1]["margin"])
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(best=f"{best_batch_head[0]}:{best_batch_head[1]['margin']:.3f}", lr=f"{lr:.2e}")

            for head_type, metrics in batch_metrics.items():
                epoch_totals[head_type]["loss"] += metrics["loss"]
                epoch_totals[head_type]["pos"] += metrics["pos_sim"]
                epoch_totals[head_type]["neg"] += metrics["neg_sim"]
                epoch_totals[head_type]["steps"] += 1

            if global_step > 0 and global_step % logging_steps == 0:
                top_heads = sorted(batch_metrics.items(), key=lambda item: item[1]["margin"], reverse=True)[:3]
                top_summary = " | ".join(
                    f"{head_type}: margin={metrics['margin']:.4f} loss={metrics['loss']:.4f}"
                    for head_type, metrics in top_heads
                )
                logger.info(f"  Step {global_step}: {top_summary} | lr={lr:.2e}")

            if global_step > 0 and global_step % eval_steps == 0 and len(val_loader) > 0:
                logger.info(f"\n--- Joint eval @ step {global_step} ---")
                eval_metrics = evaluate_all_heads(
                    model=base_model,
                    candidate_heads=ema_heads if ema_heads is not None else candidate_heads,
                    loss_modules=loss_modules,
                    val_loader=val_loader,
                    device=device,
                    use_amp=use_bf16,
                    amp_dtype=amp_dtype,
                    max_batches=max_eval_batches,
                )
                log_head_leaderboard(eval_metrics, f"JOINT LEADERBOARD @ STEP {global_step}")

                for head_type, metrics in eval_metrics.items():
                    history["heads"][head_type]["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, **metrics})
                    if metrics["margin"] > best_margin[head_type]:
                        best_margin[head_type] = metrics["margin"]
                        no_improve_count[head_type] = 0
                        save_bakeoff_checkpoint(
                            base_model,
                            output_base / head_type / "best",
                            tokenizer=tokenizer,
                            head_type=head_type,
                            head_params=dict(history["heads"][head_type]["params"]),
                            embedding_head_override=(ema_heads if ema_heads is not None else candidate_heads)[head_type],
                        )
                    else:
                        no_improve_count[head_type] += 1
                        if early_stopping_enabled and head_type in active_heads and no_improve_count[head_type] >= early_stopping_patience:
                            active_heads.discard(head_type)
                            logger.info(f"  Deactivating head {head_type} after {no_improve_count[head_type]} non-improving evals")

                if early_stopping_enabled and not active_heads:
                    logger.info("All trainable heads reached early stopping. Ending joint run.")
                    break

        for head_type, totals in epoch_totals.items():
            if totals["steps"] == 0:
                continue
            avg_loss = totals["loss"] / totals["steps"]
            avg_pos = totals["pos"] / totals["steps"]
            avg_neg = totals["neg"] / totals["steps"]
            history["heads"][head_type]["train_loss"].append(avg_loss)
            history["heads"][head_type]["train_pos_sim"].append(avg_pos)
            history["heads"][head_type]["train_neg_sim"].append(avg_neg)
            history["heads"][head_type]["train_margin"].append(avg_pos - avg_neg)

        epoch_train_snapshot = {
            head_type: {
                "margin": head_history["train_margin"][-1],
                "accuracy": 0.0,
                "hard_neg_accuracy": 0.0,
                "val_loss": head_history["train_loss"][-1],
            }
            for head_type, head_history in history["heads"].items()
            if head_history["train_margin"]
        }
        if epoch_train_snapshot:
            log_head_leaderboard(epoch_train_snapshot, f"TRAIN MARGINS AFTER EPOCH {epoch + 1}")

        if len(val_loader) > 0:
            logger.info(f"--- Full joint eval epoch {epoch + 1} ---")
            eval_metrics = evaluate_all_heads(
                model=base_model,
                candidate_heads=ema_heads if ema_heads is not None else candidate_heads,
                loss_modules=loss_modules,
                val_loader=val_loader,
                device=device,
                use_amp=use_bf16,
                amp_dtype=amp_dtype,
                max_batches=None,
            )
            log_head_leaderboard(eval_metrics, f"FULL EVAL LEADERBOARD EPOCH {epoch + 1}")
            for head_type, metrics in eval_metrics.items():
                history["heads"][head_type]["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, "full_eval": True, **metrics})
                if metrics["margin"] > best_margin[head_type]:
                    best_margin[head_type] = metrics["margin"]
                    no_improve_count[head_type] = 0
                    save_bakeoff_checkpoint(
                        base_model,
                        output_base / head_type / "best",
                        tokenizer=tokenizer,
                        head_type=head_type,
                        head_params=dict(history["heads"][head_type]["params"]),
                        embedding_head_override=(ema_heads if ema_heads is not None else candidate_heads)[head_type],
                    )
                else:
                    no_improve_count[head_type] += 1
                    if early_stopping_enabled and head_type in active_heads and no_improve_count[head_type] >= early_stopping_patience:
                        active_heads.discard(head_type)
                        logger.info(f"  Deactivating head {head_type} after {no_improve_count[head_type]} non-improving evals")

        if early_stopping_enabled and not active_heads:
            logger.info("All trainable heads reached early stopping. Ending joint run.")
            break

    summary = {}
    for head_type, head_history in history["heads"].items():
        evals = head_history.get("eval_metrics", [])
        best_eval = max(evals, key=lambda item: item.get("margin", -1.0)) if evals else {}
        summary[head_type] = {
            "best_margin": best_eval.get("margin", 0.0),
            "best_accuracy": best_eval.get("accuracy", 0.0),
            "best_hard_neg_accuracy": best_eval.get("hard_neg_accuracy", 0.0),
            "best_val_loss": best_eval.get("val_loss", 0.0),
            "final_train_loss": head_history["train_loss"][-1] if head_history["train_loss"] else 0.0,
        }
        save_bakeoff_checkpoint(
            base_model,
            output_base / head_type / "final",
            tokenizer=tokenizer,
            head_type=head_type,
            head_params=dict(head_history["params"]),
            embedding_head_override=(ema_heads if ema_heads is not None else candidate_heads)[head_type],
        )

    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "bakeoff_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_base / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log_head_leaderboard(summary, "FINAL JOINT BAKE-OFF LEADERBOARD")
    return history


# =============================================================================
# Single Experiment Runner
# =============================================================================


def run_experiment(
    config: dict[str, Any],
    head_type: str,
    head_params: dict[str, Any],
    output_dir: Path,
    debug: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run a single bake-off experiment for one head type."""
    log_section(f"EXPERIMENT: {head_type}")

    encoder_config = config.get("encoder", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})

    checkpoint_path = encoder_config.get("checkpoint", "checkpoints/checkpoint-8000")
    data_root = Path(data_config.get("root", "data"))

    # Training params
    learning_rate = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_epochs = training_config.get("num_epochs", 3)
    batch_size = training_config.get("batch_size", 128)
    max_length = data_config.get("max_length", 128)
    warmup_steps = training_config.get("warmup_steps", 500)
    save_steps = training_config.get("save_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    logging_steps = training_config.get("logging_steps", 50)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    val_split = training_config.get("val_split", 0.1)
    num_workers = data_config.get("num_workers", 8)
    max_eval_batches = training_config.get("max_eval_batches", 100)
    use_bf16 = training_config.get("bf16", False)
    use_tf32 = training_config.get("tf32", False)
    use_flash_attention = training_config.get("flash_attention", False)
    ema_config = training_config.get("ema", {})
    use_ema = ema_config.get("enabled", True)
    es_config = training_config.get("early_stopping", {})
    early_stopping_patience = es_config.get("patience", 5) if es_config.get("enabled", True) else 100000
    lr_scheduler_type = training_config.get("lr_scheduler_type", "cosine")
    temperature = loss_config.get("temperature", 0.07)
    hard_negative_weight = loss_config.get("hard_negative_weight", 1.5)
    learnable_temperature = loss_config.get("learnable_temperature", False)
    temperature_lr = loss_config.get("temperature_lr", 1e-3)
    curriculum_config = training_config.get("curriculum", {})
    matryoshka_config = training_config.get("matryoshka", {})
    matryoshka_dims = matryoshka_config.get("dims", None) if matryoshka_config.get("enabled", False) else None

    if debug:
        max_samples = max_samples or 500
        save_steps = 50
        eval_steps = 50

    seed = training_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: {gpu_name}")
        if use_tf32 and "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load model
    log_section("MODEL")
    model = load_model_and_replace_embedding_head(
        checkpoint_path, head_type=head_type, head_params=head_params,
        use_flash_attention=use_flash_attention,
    )
    freeze_model_except_embedding_head(model)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Data
    log_section("DATA")
    data_paths = get_embedding_data_paths(data_config, data_root)
    logger.info(f"  Data sources: {len(data_paths)}")
    for p in data_paths:
        logger.info(f"    {p}")

    full_dataset = TripletDataset(data_paths=data_paths, max_samples=max_samples)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    logger.info(f"  Total: {len(full_dataset)} triplets (train={train_size}, val={val_size})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)
    effective_workers = 0 if platform.system() == "Windows" else num_workers

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
        num_workers=effective_workers, pin_memory=True,
        persistent_workers=effective_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=effective_workers, pin_memory=True,
        persistent_workers=effective_workers > 0,
    )

    # Loss
    loss_fn = FamilyContrastiveLoss(
        temperature=temperature, hard_negative_weight=hard_negative_weight,
        use_hard_negatives=True, normalize=False,
    )
    if learnable_temperature:
        loss_fn.log_temperature.requires_grad_(True)

    # Optimizer
    trainable_params = get_trainable_params(model)
    param_groups = [{"params": trainable_params, "lr": learning_rate, "weight_decay": weight_decay}]
    if learnable_temperature:
        param_groups.append({"params": [loss_fn.log_temperature], "lr": temperature_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        eps=training_config.get("adam_epsilon", 1e-8),
    )

    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    adaptive_warmup = max(10, int(num_training_steps * 0.05))
    effective_warmup = min(warmup_steps, adaptive_warmup) if num_training_steps < warmup_steps else warmup_steps

    if lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=effective_warmup, num_training_steps=num_training_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=effective_warmup, num_training_steps=num_training_steps)

    logger.info(f"  Steps: {num_training_steps} total, {effective_warmup} warmup")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Train
    history = train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        device=device, num_epochs=num_epochs, output_dir=output_dir,
        tokenizer=tokenizer, save_steps=save_steps, eval_steps=eval_steps,
        logging_steps=logging_steps, gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm, debug=debug,
        use_amp=use_bf16, use_bf16=use_bf16, use_ema=use_ema,
        early_stopping_patience=early_stopping_patience,
        max_eval_batches=max_eval_batches, curriculum_config=curriculum_config,
        matryoshka_dims=matryoshka_dims,
        base_hard_negative_weight=hard_negative_weight,
        head_type=head_type, head_params=head_params,
    )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Experiment {head_type} complete -> {output_dir}")
    return history


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding Head Bake-Off Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--head_type", type=str, default=None, help="Single head type to train (e.g. agreement_gated)")
    parser.add_argument("--run_all", action="store_true", help="Train all configured heads together with a shared encoder pass")
    parser.add_argument("--run_sequential", action="store_true", help="Legacy mode: run configured heads one by one")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max total samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiments_config = config.get("experiments", {})
    output_base = Path(config.get("output", {}).get("dir", "outputs/embedding-bakeoff"))

    if args.run_all:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# RUN MODE: JOINT MULTI-HEAD BAKE-OFF (--run_all)")
        logger.info("# Shared encoder pass | all configured heads trained together")
        logger.info("#" * 70)
        run_joint_bakeoff(
            config=config,
            output_base=output_base,
            debug=args.debug,
            max_samples=args.max_samples,
        )

    elif args.run_sequential:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# RUN MODE: SEQUENTIAL BAKE-OFF (--run_sequential)")
        logger.info("# Heads are trained one by one")
        logger.info("#" * 70)
        experiments = get_configured_head_experiments(config)
        if not experiments:
            logger.error("No experiments defined in config under experiments.heads")
            return
        results = {}
        for exp_head_type, exp_params in experiments:
            exp_output = output_base / exp_head_type
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# BAKE-OFF EXPERIMENT: {exp_head_type}")
            logger.info(f"{'#' * 70}")
            history = run_experiment(
                config=config,
                head_type=exp_head_type,
                head_params=exp_params,
                output_dir=exp_output,
                debug=args.debug,
                max_samples=args.max_samples,
            )
            results[exp_head_type] = history

        summary_path = output_base / "bakeoff_summary.json"
        summary = {}
        for ht, hist in results.items():
            evals = hist.get("eval_metrics", [])
            best_eval = max(evals, key=lambda e: e.get("margin", -1)) if evals else {}
            summary[ht] = {
                "best_margin": best_eval.get("margin", 0),
                "best_accuracy": best_eval.get("accuracy", 0),
                "best_hard_neg_accuracy": best_eval.get("hard_neg_accuracy", 0),
                "best_val_loss": best_eval.get("val_loss", 0),
                "final_train_loss": hist["train_loss"][-1] if hist.get("train_loss") else 0,
            }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nBake-off summary -> {summary_path}")
        log_head_leaderboard(summary, "SEQUENTIAL BAKE-OFF LEADERBOARD")

    elif args.head_type:
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"# RUN MODE: SINGLE HEAD (--head_type {args.head_type})")
        logger.info("# Only one embedding head will be trained")
        logger.info("#" * 70)
        # Single experiment from CLI
        # Check if this head has experiment-specific params in config
        merged_params = config.get("embedding_head", {})
        for exp_head_type, exp_params in get_configured_head_experiments(config):
            if exp_head_type == args.head_type:
                merged_params = exp_params
                break

        exp_output = output_base / args.head_type
        run_experiment(
            config=config, head_type=args.head_type, head_params=merged_params,
            output_dir=exp_output, debug=args.debug, max_samples=args.max_samples,
        )
    else:
        logger.error("Specify --head_type <name>, --run_all, or --run_sequential")
        logger.info(f"Available heads: {', '.join(sorted(EMBEDDING_HEAD_REGISTRY.keys()))}")


if __name__ == "__main__":
    main()

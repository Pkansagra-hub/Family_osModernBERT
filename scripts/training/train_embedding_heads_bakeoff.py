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
import hashlib
import json
import logging
import math
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

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

from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

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


def _deep_update_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def resolve_workspace_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve config paths robustly relative to cwd, base_dir, or project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(project_root / path)
    candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else path


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


def load_checkpoint_tokenizer(checkpoint_path: str | Path) -> Any:
    """Load tokenizer with fallback for custom tokenizer_class metadata."""
    checkpoint_path = resolve_workspace_path(checkpoint_path)
    try:
        return AutoTokenizer.from_pretrained(checkpoint_path)
    except ValueError as exc:
        tokenizer_json = checkpoint_path / "tokenizer.json"
        if not tokenizer_json.exists():
            raise

        logger.warning(
            f"AutoTokenizer failed for {checkpoint_path.name}; falling back to PreTrainedTokenizerFast ({exc})"
        )
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))


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
        per_source_cap = max(1, math.ceil(max_samples / len(sources))) if max_samples and sources else None
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
                    remaining_for_source = None
                    if per_source_cap is not None:
                        loaded_for_source = len(self.samples) - count_before
                        remaining_for_source = max(0, per_source_cap - loaded_for_source)
                        if remaining_for_source == 0:
                            break
                    self._load_jsonl(
                        jsonl_file,
                        slice_name,
                        fmt,
                        max_samples,
                        max_source_samples=remaining_for_source,
                    )
                    if max_samples and len(self.samples) >= max_samples:
                        break
                    if per_source_cap is not None and (len(self.samples) - count_before) >= per_source_cap:
                        break
            elif path.suffix == ".jsonl":
                self._load_jsonl(
                    path,
                    slice_name,
                    fmt,
                    max_samples,
                    max_source_samples=per_source_cap,
                )

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
        max_source_samples: int | None = None,
    ) -> None:
        if any(path.name.startswith(p) for p in _SKIP_PREFIXES):
            return

        is_hard_neg_dir = "hard_negative" in str(path.parent) or "hard_negative" in path.name
        count_before = len(self.samples)

        with open(path, encoding="utf-8") as f:
            loaded_from_file = 0
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                if max_source_samples is not None and loaded_from_file >= max_source_samples:
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
                    loaded_from_file += 1

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
                "anchor_safety_label": raw.get("safety_label_anchor"),
                "negative_safety_label": raw.get("safety_label_negative"),
                "anchor_emotion_label": raw.get("emotion_label_anchor"),
                "negative_emotion_label": raw.get("emotion_label_negative"),
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
                "anchor_safety_label": raw.get("safety_label_anchor"),
                "negative_safety_label": raw.get("safety_label_negative"),
                "anchor_emotion_label": raw.get("emotion_label_anchor"),
                "negative_emotion_label": raw.get("emotion_label_negative"),
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
        anchor_safety_labels = [f.get("anchor_safety_label") for f in features]
        anchor_emotion_labels = [f.get("anchor_emotion_label") for f in features]

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
            "anchor_texts": anchors,
            "positive_texts": positives,
            "has_negative": torch.tensor(has_neg, dtype=torch.bool),
            "triplet_indices": torch.tensor(triplet_indices, dtype=torch.long),
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
            "slice_tags": slice_tags,
            "anchor_safety_labels": anchor_safety_labels,
            "anchor_emotion_labels": anchor_emotion_labels,
        }

        # Tokenize negatives ONLY for triplet samples
        if triplet_indices:
            negatives = [features[i]["negative"] for i in triplet_indices]
            trip_hard_neg = [hard_neg_flags[i] for i in triplet_indices]
            negative_safety_labels = [features[i].get("negative_safety_label") for i in triplet_indices]
            negative_emotion_labels = [features[i].get("negative_emotion_label") for i in triplet_indices]
            negative_enc = self.tokenizer(
                negatives, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            batch["negative_input_ids"] = negative_enc["input_ids"]
            batch["negative_attention_mask"] = negative_enc["attention_mask"]
            batch["hard_negative_mask"] = torch.tensor(trip_hard_neg, dtype=torch.bool)
            batch["negative_texts"] = negatives
            batch["negative_safety_labels"] = negative_safety_labels
            batch["negative_emotion_labels"] = negative_emotion_labels
        else:
            # Pure pair batch - no negatives
            batch["negative_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["negative_attention_mask"] = torch.zeros(0, 1, dtype=torch.long)
            batch["hard_negative_mask"] = torch.zeros(0, dtype=torch.bool)
            batch["negative_texts"] = []
            batch["negative_safety_labels"] = []
            batch["negative_emotion_labels"] = []

        return batch


class SliceBalancedSampler(torch.utils.data.Sampler):
    """Weighted per-slice sampler that rebalances training distribution.

    Prevents large slices (e.g. 261K silver_synthetic) from drowning
    small but critical slices (e.g. 4K wrong_person).

    Each epoch, draws ``slice_weight * slice_count`` samples per slice
    (normalised to the target epoch size), shuffles, and yields indices.
    Small slices are upsampled with replacement; large slices may be
    subsampled.

    Works with both ``EmbeddingDataset`` and ``torch.utils.data.Subset``
    wrapping one.

    Args:
        dataset: The training dataset (EmbeddingDataset or Subset).
        slice_weights: Mapping of slice_name -> sampling weight.
            Slices not listed default to weight 1.0.
        epoch_size: Total samples per epoch.  Defaults to ``len(dataset)``.
        seed: Base random seed (combined with epoch counter).
    """

    def __init__(
        self,
        dataset: Dataset,
        slice_weights: dict[str, float],
        epoch_size: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.slice_weights = slice_weights
        self.seed = seed
        self.epoch = 0

        # Build local-index groups by slice
        self._indices_by_slice: dict[str, list[int]] = {}

        if isinstance(dataset, torch.utils.data.Subset):
            underlying = dataset.dataset
            for local_idx, global_idx in enumerate(dataset.indices):
                slice_name = underlying.samples[global_idx]["slice"]
                self._indices_by_slice.setdefault(slice_name, []).append(local_idx)
        elif hasattr(dataset, "samples"):
            for idx, sample in enumerate(dataset.samples):
                slice_name = sample["slice"]
                self._indices_by_slice.setdefault(slice_name, []).append(idx)
        else:
            raise TypeError("Dataset must be EmbeddingDataset or Subset thereof")

        # Compute per-slice draw counts
        self._compute_epoch_counts(epoch_size)

        # Log effective distribution
        logger.info(f"  SliceBalancedSampler: {self._total_size:,} samples/epoch")
        for sn in sorted(self._epoch_counts):
            cnt = self._epoch_counts[sn]
            pool = len(self._indices_by_slice[sn])
            pct = 100.0 * cnt / self._total_size if self._total_size else 0
            mode = "upsample" if cnt > pool else "subsample" if cnt < pool else "exact"
            logger.info(f"    {sn:<25} {cnt:>8,} / {pool:>8,} pool  ({pct:5.1f}%)  [{mode}]")

    def _compute_epoch_counts(self, epoch_size: int | None) -> None:
        """Compute how many samples to draw from each slice per epoch."""
        raw_weights: dict[str, float] = {}
        for slice_name, indices in self._indices_by_slice.items():
            w = self.slice_weights.get(slice_name, 1.0)
            raw_weights[slice_name] = w * len(indices)

        total_weight = sum(raw_weights.values())
        target_size = epoch_size or len(self.dataset)

        self._epoch_counts: dict[str, int] = {}
        for slice_name, raw_w in raw_weights.items():
            self._epoch_counts[slice_name] = max(1, int(round(target_size * raw_w / total_weight)))

        self._total_size = sum(self._epoch_counts.values())

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []

        for slice_name, count in self._epoch_counts.items():
            pool = self._indices_by_slice[slice_name]
            if count <= len(pool):
                sampled = rng.sample(pool, count)
            else:
                # Upsample with replacement
                sampled = [rng.choice(pool) for _ in range(count)]
            indices.extend(sampled)

        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return self._total_size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch index for deterministic shuffling."""
        self.epoch = epoch


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
    resolved_data_root = resolve_workspace_path(data_root)
    if sources_cfg is not None:
        resolved = []
        for src in sources_cfg:
            raw_path = src["path"]
            path = resolve_workspace_path(raw_path, resolved_data_root)
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
        path = resolve_workspace_path(path_str, resolved_data_root)
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


def resolve_data_config(
    data_config: dict[str, Any],
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Resolve an optional stage-specific data profile from the YAML config."""
    resolved = copy.deepcopy(data_config)
    if not profile_name:
        return resolved

    profiles = data_config.get("profiles", {})
    profile = profiles.get(profile_name)
    if not profile:
        return resolved

    for key in ("sources", "sampling", "eval_split_per_slice"):
        if key in profile:
            resolved[key] = copy.deepcopy(profile[key])

    logger.info(f"Using data profile '{profile_name}' with {len(resolved.get('sources', []))} sources")
    return resolved


# =============================================================================
# Teacher Cache Build Helpers (Milestone 2)
# =============================================================================


def iter_source_jsonl_files(path: Path) -> list[Path]:
    """Return JSONL files for a source path, skipping metadata helpers."""
    if not path.exists():
        return []
    if path.is_file() and path.suffix == ".jsonl":
        return [path] if not any(path.name.startswith(prefix) for prefix in _SKIP_PREFIXES) else []

    jsonl_files = sorted(path.glob("*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(path.glob("**/*.jsonl"))
    return [jsonl_file for jsonl_file in jsonl_files if not any(jsonl_file.name.startswith(prefix) for prefix in _SKIP_PREFIXES)]


def iter_jsonl_records(path: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield decoded JSONL records from a file or directory."""
    for jsonl_file in iter_source_jsonl_files(path):
        with open(jsonl_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield jsonl_file, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _normalize_text_value(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None

    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        text = " ".join(item.strip() for item in value if item.strip()).strip()
        return text if text else None

    return None


def _teacher_entry_id(mode: str, text: str) -> str:
    digest = hashlib.sha256(f"{mode}\n{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _add_teacher_entry(
    entries: list[dict[str, Any]],
    dedupe_index: dict[tuple[str, str], int],
    text: str,
    mode: str,
    slice_name: str,
    source_path: Path,
    text_key: str,
) -> None:
    dedupe_key = (mode, text)
    if dedupe_key in dedupe_index:
        existing = entries[dedupe_index[dedupe_key]]
        existing.setdefault("slices", [])
        if slice_name not in existing["slices"]:
            existing["slices"].append(slice_name)
        return

    dedupe_index[dedupe_key] = len(entries)
    entries.append({
        "id": _teacher_entry_id(mode, text),
        "text": text,
        "mode": mode,
        "slice": slice_name,
        "slices": [slice_name],
        "source_path": str(source_path),
        "text_key": text_key,
    })


def collect_teacher_entries(
    config: dict[str, Any],
    max_texts: int | None = None,
) -> list[dict[str, Any]]:
    """Collect deduplicated teacher-cache text entries from configured sources."""
    teacher_config = config.get("teacher", {})
    data_config = config.get("data", {})
    data_root = resolve_workspace_path(data_config.get("root", "data"))

    source_sets = teacher_config.get("source_sets", {})
    use_stage_b_profile = source_sets.get("use_stage_b_profile", False)
    active_data_config = resolve_data_config(
        data_config,
        config.get("stage_b", {}).get("data_profile") if use_stage_b_profile else None,
    )

    configured_sources: list[dict[str, Any]] = []
    if source_sets.get("use_data_sources", True):
        configured_sources.extend(get_embedding_data_sources(active_data_config, data_root))

    for source in teacher_config.get("corpus_sources", []):
        raw_path = source["path"]
        resolved_path = resolve_workspace_path(raw_path, data_root)
        configured_sources.append({
            "path": resolved_path,
            "slice": source.get("slice", resolved_path.name),
            "format": source.get("format", "text"),
            "text_keys": source.get("text_keys", ["text"]),
            "mode": source.get("mode", "document"),
        })

    entries: list[dict[str, Any]] = []
    dedupe_index: dict[tuple[str, str], int] = {}

    for source in configured_sources:
        source_path = Path(source["path"])
        source_format = source.get("format", "triplet")
        slice_name = source.get("slice", source_path.name)
        text_keys = source.get("text_keys", ["text"])
        default_mode = source.get("mode", "document")

        if not source_path.exists():
            logger.warning(f"Teacher source not found: {source_path}")
            continue

        logger.info(f"  Collecting teacher texts: slice={slice_name} format={source_format} path={source_path}")
        before_count = len(entries)
        for jsonl_file, record in iter_jsonl_records(source_path):
            if source_format == "pair":
                query_text = _normalize_text_value(record.get("query") or record.get("anchor"))
                document_text = _normalize_text_value(record.get("document") or record.get("positive"))
                if query_text is not None:
                    _add_teacher_entry(entries, dedupe_index, query_text, "query", slice_name, jsonl_file, "query")
                if document_text is not None:
                    _add_teacher_entry(entries, dedupe_index, document_text, "document", slice_name, jsonl_file, "document")
            elif source_format == "triplet":
                for key in ("anchor", "positive", "negative"):
                    text = _normalize_text_value(record.get(key))
                    if text is not None:
                        _add_teacher_entry(entries, dedupe_index, text, default_mode, slice_name, jsonl_file, key)
            elif source_format == "text":
                for key in text_keys:
                    text = _normalize_text_value(record.get(key))
                    if text is not None:
                        _add_teacher_entry(entries, dedupe_index, text, default_mode, slice_name, jsonl_file, key)
            else:
                raise ValueError(f"Unsupported teacher source format: {source_format}")

            if max_texts is not None and len(entries) >= max_texts:
                break

        added = len(entries) - before_count
        logger.info(f"    Added {added:,} unique entries from {slice_name}")
        if max_texts is not None and len(entries) >= max_texts:
            logger.info(f"  Reached max_texts limit ({max_texts:,}); stopping collection")
            break

    logger.info(f"Collected {len(entries):,} deduplicated teacher entries")
    return entries


def _teacher_prompt_text(text: str, mode: str, teacher_config: dict[str, Any]) -> str:
    prompts_config = teacher_config.get("prompts", {})
    use_query_prompt = prompts_config.get("use_query_prompt_for_query_doc", True)
    if mode == "query" and use_query_prompt:
        instruction = prompts_config.get(
            "query_instruction",
            "Given a user query, retrieve the most relevant FamilyOS memory or passage.",
        )
        return f"Instruct: {instruction}\nQuery: {text}"
    return text


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token embedding."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def load_teacher_model(teacher_config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    """Load teacher tokenizer/model for cache generation."""
    model_name = teacher_config.get("model_name")
    if not model_name:
        raise ValueError("teacher.model_name is required for teacher cache mode")

    dtype = _resolve_dtype(teacher_config.get("dtype", "bfloat16"))
    attn_implementation = teacher_config.get("attn_implementation")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if attn_implementation and device.type == "cuda":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except ImportError as exc:
        if "FlashAttention2" not in str(exc):
            raise

        logger.warning(
            "FlashAttention2 unavailable for teacher load; retrying without attn_implementation"
        )
        model_kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(model_name, **model_kwargs)

    model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_teacher_entries(
    entries: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    teacher_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Encode teacher entries into a single CPU tensor."""
    if not entries:
        return torch.empty((0, 0), dtype=torch.float32)

    max_length = int(teacher_config.get("max_length", 256))
    batch_size = int(teacher_config.get("batch_size", 64))
    normalize = bool(teacher_config.get("normalize", True))

    dtype = _resolve_dtype(teacher_config.get("dtype", "bfloat16"))
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_context = (
        autocast("cuda", dtype=dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    all_embeddings: list[torch.Tensor] = []
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start : start + batch_size]
        texts = [_teacher_prompt_text(entry["text"], entry["mode"], teacher_config) for entry in batch_entries]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with amp_context:
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = last_token_pool(hidden_states, encoded["attention_mask"])
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

        all_embeddings.append(pooled.float().cpu())
        if (start // batch_size) % 20 == 0:
            logger.info(f"  Teacher encoded {min(start + len(batch_entries), len(entries)):,}/{len(entries):,} entries")

    return torch.cat(all_embeddings, dim=0)


def _resolve_teacher_cache_storage_dtype(
    cache_config: dict[str, Any],
    teacher_config: dict[str, Any],
) -> tuple[torch.dtype, str]:
    save_dtype_name = str(cache_config.get("save_dtype", teacher_config.get("dtype", "float32"))).strip().lower()
    save_dtype = _resolve_dtype(save_dtype_name)
    if save_dtype is None:
        return torch.float32, "float32"
    return save_dtype, save_dtype_name


def _prepare_teacher_cache_output(
    entries: list[dict[str, Any]],
    teacher_config: dict[str, Any],
) -> tuple[Path, dict[str, Any], int, bool, torch.dtype]:
    cache_config = teacher_config.get("cache", {})
    cache_dir = resolve_workspace_path(cache_config.get("dir", "outputs/teacher-cache"))
    overwrite = bool(cache_config.get("overwrite", False))
    shard_size = int(cache_config.get("shard_size", 50000))
    save_text_index = bool(cache_config.get("save_text_index", True))
    save_dtype, save_dtype_name = _resolve_teacher_cache_storage_dtype(cache_config, teacher_config)

    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_artifacts = [
        *cache_dir.glob("shard_*.pt"),
        *cache_dir.glob(".shard_*.pt.tmp"),
    ]
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.jsonl"
    if manifest_path.exists():
        existing_artifacts.append(manifest_path)
    if index_path.exists():
        existing_artifacts.append(index_path)

    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"Teacher cache already exists in {cache_dir}; set teacher.cache.overwrite=true to rebuild"
        )

    for artifact in existing_artifacts:
        artifact.unlink()

    manifest = {
        "model_name": teacher_config.get("model_name"),
        "dtype": teacher_config.get("dtype", "bfloat16"),
        "max_length": teacher_config.get("max_length", 256),
        "normalize": teacher_config.get("normalize", True),
        "num_entries": len(entries),
        "embedding_dim": 0,
        "shard_size": shard_size,
        "storage_dtype": save_dtype_name,
        "payload_format": "id_embedding_only",
        "shards": [],
    }
    return cache_dir, manifest, shard_size, save_text_index, save_dtype


def _write_teacher_cache_shard(
    cache_dir: Path,
    shard_entries: list[dict[str, Any]],
    shard_embeddings: torch.Tensor,
    shard_idx: int,
    save_dtype: torch.dtype,
) -> dict[str, Any]:
    shard_tensor = shard_embeddings.to(dtype=save_dtype).contiguous()
    shard_path = cache_dir / f"shard_{shard_idx:05d}.pt"
    temp_shard_path = cache_dir / f".{shard_path.name}.tmp"
    payload = {
        "ids": [entry["id"] for entry in shard_entries],
        "embeddings": shard_tensor,
    }
    try:
        torch.save(
            payload,
            temp_shard_path,
            _use_new_zipfile_serialization=False,
        )
        temp_shard_path.replace(shard_path)
    except Exception as exc:
        if temp_shard_path.exists():
            temp_shard_path.unlink()
        raise RuntimeError(
            "Failed to write teacher cache shard. This is often caused by Colab filesystem write issues; "
            "retry with a smaller teacher.cache.shard_size (for example 10000) or keep outputs on local /content storage."
        ) from exc

    return {
        "path": shard_path.name,
        "count": len(shard_entries),
    }


def save_teacher_cache(
    entries: list[dict[str, Any]],
    embeddings: torch.Tensor,
    teacher_config: dict[str, Any],
) -> Path:
    """Persist teacher-cache shards and manifest."""
    cache_dir, manifest, shard_size, save_text_index, save_dtype = _prepare_teacher_cache_output(entries, teacher_config)
    manifest["embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.numel() > 0 else 0

    for shard_idx, start in enumerate(range(0, len(entries), shard_size)):
        shard_entries = entries[start : start + shard_size]
        shard_embeddings = embeddings[start : start + shard_size]
        shard_meta = _write_teacher_cache_shard(cache_dir, shard_entries, shard_embeddings, shard_idx, save_dtype)
        manifest["shards"].append({
            "path": shard_meta["path"],
            "count": shard_meta["count"],
            "start": start,
            "end": start + len(shard_entries),
        })

    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if save_text_index:
        with open(cache_dir / "index.jsonl", "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved teacher cache -> {cache_dir}")
    return cache_dir


def build_teacher_cache(
    config: dict[str, Any],
    max_samples: int | None = None,
) -> Path:
    """Build a deduplicated teacher embedding cache using existing data sources."""
    teacher_config = config.get("teacher", {})
    if not teacher_config:
        raise ValueError("teacher configuration is required for --build_teacher_cache")

    log_section("TEACHER CACHE BUILD")
    logger.info(f"  Teacher model: {teacher_config.get('model_name')}")

    entries = collect_teacher_entries(config, max_texts=max_samples)
    if not entries:
        raise ValueError("No teacher entries collected; check teacher/source configuration")

    device_name = teacher_config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for teacher cache but not available; falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    tokenizer, model = load_teacher_model(teacher_config, device)
    cache_dir, manifest, shard_size, save_text_index, save_dtype = _prepare_teacher_cache_output(entries, teacher_config)
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.jsonl"
    embedding_dim = 0

    index_handle = open(index_path, "w", encoding="utf-8") if save_text_index else None
    try:
        for shard_idx, start in enumerate(range(0, len(entries), shard_size)):
            shard_entries = entries[start : start + shard_size]
            shard_embeddings = encode_teacher_entries(shard_entries, tokenizer, model, teacher_config, device)
            if embedding_dim == 0 and shard_embeddings.ndim == 2 and shard_embeddings.numel() > 0:
                embedding_dim = int(shard_embeddings.shape[1])
                manifest["embedding_dim"] = embedding_dim

            shard_meta = _write_teacher_cache_shard(cache_dir, shard_entries, shard_embeddings, shard_idx, save_dtype)
            manifest["shards"].append({
                "path": shard_meta["path"],
                "count": shard_meta["count"],
                "start": start,
                "end": start + len(shard_entries),
            })

            if index_handle is not None:
                for entry in shard_entries:
                    index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                index_handle.flush()

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)

            logger.info(
                f"  Saved teacher shard {shard_idx + 1}/{math.ceil(len(entries) / shard_size)} "
                f"({start + len(shard_entries):,}/{len(entries):,} entries persisted)"
            )
            del shard_embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if index_handle is not None:
            index_handle.close()

    logger.info(
        f"Teacher cache complete: {len(entries):,} entries | dim={embedding_dim}"
    )
    return cache_dir


class TeacherEmbeddingCache:
    """In-memory teacher embedding cache loaded from sharded artifacts."""

    def __init__(self, embedding_dim: int, embeddings_by_id: dict[str, torch.Tensor]):
        self.embedding_dim = embedding_dim
        self.embeddings_by_id = embeddings_by_id

    @classmethod
    def load(cls, cache_dir: str | Path) -> "TeacherEmbeddingCache":
        cache_path = resolve_workspace_path(cache_dir)
        manifest_path = cache_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Teacher cache manifest not found: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        embedding_dim = int(manifest.get("embedding_dim", 0))
        embeddings_by_id: dict[str, torch.Tensor] = {}
        skipped_shards = 0
        skipped_entries = 0
        for shard_meta in manifest.get("shards", []):
            shard_path = cache_path / shard_meta["path"]
            if not shard_path.exists():
                skipped_shards += 1
                skipped_entries += int(shard_meta.get("count", 0))
                logger.warning(f"Teacher cache shard missing; skipping: {shard_path}")
                continue

            try:
                shard_payload = torch.load(shard_path, map_location="cpu")
            except Exception as exc:
                skipped_shards += 1
                skipped_entries += int(shard_meta.get("count", 0))
                logger.warning(f"Teacher cache shard unreadable; skipping: {shard_path} ({exc})")
                continue

            shard_embeddings = shard_payload["embeddings"].float()
            shard_ids = shard_payload.get("ids")
            if shard_ids is not None:
                for index, entry_id in enumerate(shard_ids):
                    embeddings_by_id[str(entry_id)] = shard_embeddings[index]
                continue

            shard_texts = shard_payload.get("texts")
            shard_modes = shard_payload.get("modes")
            if shard_texts is None or shard_modes is None:
                raise ValueError(f"Teacher cache shard missing ids/texts/modes: {shard_path}")
            for index, (text, mode) in enumerate(zip(shard_texts, shard_modes)):
                embeddings_by_id[_teacher_entry_id(str(mode), str(text))] = shard_embeddings[index]

        if skipped_shards:
            logger.warning(
                f"Teacher cache loaded with missing/unreadable shards: {skipped_shards} shard(s), "
                f"~{skipped_entries:,} entry/entries skipped"
            )

        logger.info(
            f"Loaded teacher cache: {len(embeddings_by_id):,} entries | dim={embedding_dim} | path={cache_path}"
        )
        return cls(embedding_dim=embedding_dim, embeddings_by_id=embeddings_by_id)

    def lookup(self, texts: list[str], modes: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if len(texts) != len(modes):
            raise ValueError("texts and modes must have the same length for teacher cache lookup")

        if not texts:
            empty_emb = torch.zeros((0, self.embedding_dim), dtype=torch.float32, device=device)
            empty_mask = torch.zeros((0,), dtype=torch.bool, device=device)
            return empty_emb, empty_mask

        found_vectors: list[torch.Tensor] = []
        found_mask: list[bool] = []
        for text, mode in zip(texts, modes):
            vector = self.embeddings_by_id.get(_teacher_entry_id(mode, text))
            if vector is None and mode != "document":
                vector = self.embeddings_by_id.get(_teacher_entry_id("document", text))
            if vector is None:
                vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
                found_mask.append(False)
            else:
                found_mask.append(True)
            found_vectors.append(vector)

        stacked = torch.stack(found_vectors, dim=0).to(device)
        mask = torch.tensor(found_mask, dtype=torch.bool, device=device)
        return stacked, mask


class TeacherProjection(nn.Module):
    """Optional trainable projection from teacher space into student space."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.linear = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


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


def load_model_checkpoint(
    checkpoint_path: str | Path,
    exclude_decoder: bool = True,
    use_flash_attention: bool = False,
) -> ModernBertMultiTaskModel:
    """Load a checkpoint and preserve the existing embedding head weights."""
    from safetensors.torch import load_file
    from transformers import AutoConfig

    checkpoint_path = resolve_workspace_path(checkpoint_path)
    logger.info(f"Loading checkpoint model from {checkpoint_path}")

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    if use_flash_attention:
        config.attn_implementation = "flash_attention_2"
        logger.info("  Flash Attention 2: ENABLED")

    capabilities = load_checkpoint_capabilities(checkpoint_path, exclude_decoder)
    model = ModernBertMultiTaskModel(config=config, capabilities=capabilities, freeze_encoder=False)
    restore_checkpoint_head_architecture(model, checkpoint_path)
    model._init_encoder()

    embedding_metadata_path = checkpoint_path / "embedding_metadata.json"
    if embedding_metadata_path.exists():
        with open(embedding_metadata_path, encoding="utf-8") as handle:
            embedding_metadata = json.load(handle)
        bakeoff_info = embedding_metadata.get("bakeoff", {})
        head_type = bakeoff_info.get("head_type")
        head_params = bakeoff_info.get("head_params") or {}
        if head_type:
            hidden_size = model.config.hidden_size
            model.heads["embedding"] = create_embedding_head(
                head_type=head_type,
                hidden_size=hidden_size,
                **head_params,
            )
            logger.info(f"  Restored embedding head architecture from metadata: {head_type}")

    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        weights_path = checkpoint_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state, strict=True)

    loaded_heads: list[str] = []
    for head_name in model.heads.keys():
        head_prefix = f"heads.{head_name}."
        head_state = {k.replace(head_prefix, ""): v for k, v in state_dict.items() if k.startswith(head_prefix)}
        if head_state:
            try:
                model.heads[head_name].load_state_dict(head_state, strict=True)
                loaded_heads.append(head_name)
            except Exception as exc:
                logger.warning(f"Could not load {head_name} head: {exc}")

    logger.info(f"  Loaded checkpoint heads: {', '.join(loaded_heads)}")
    return model


def load_model_with_trained_head(
    bakeoff_checkpoint: str | Path,
    exclude_decoder: bool = True,
    use_flash_attention: bool = False,
) -> tuple[ModernBertMultiTaskModel, str, dict[str, Any]]:
    """Load model from a Stage A bakeoff checkpoint, keeping the trained embedding head.

    Unlike load_model_and_replace_embedding_head, this does NOT create a fresh
    head. The embedding head weights from the bakeoff are preserved as-is.

    Args:
        bakeoff_checkpoint: Path to bakeoff winner checkpoint (e.g. outputs/embedding-bakeoff/agreement_gated_v2/best).
        exclude_decoder: Skip GPT-2 decoder to save memory.
        use_flash_attention: Enable Flash Attention 2.

    Returns:
        Tuple of (model, head_type, head_params) where head_type/params are read from embedding_metadata.json.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    bakeoff_checkpoint = Path(bakeoff_checkpoint)
    logger.info(f"Loading Stage B model from bakeoff checkpoint: {bakeoff_checkpoint}")

    # Read embedding metadata to get head type and params
    emb_meta_path = bakeoff_checkpoint / "embedding_metadata.json"
    if not emb_meta_path.exists():
        raise FileNotFoundError(f"No embedding_metadata.json in {bakeoff_checkpoint}")
    with open(emb_meta_path, encoding="utf-8") as f:
        emb_meta = json.load(f)

    bakeoff_info = emb_meta.get("bakeoff", {})
    head_type = bakeoff_info.get("head_type", "agreement_gated_v2")
    head_constructor_params = bakeoff_info.get("head_constructor_params", {})
    head_params = bakeoff_info.get("head_params", {})

    if head_type not in EMBEDDING_HEAD_REGISTRY:
        inferred_head_type = head_constructor_params.get("head_type")
        if inferred_head_type not in EMBEDDING_HEAD_REGISTRY:
            head_class = bakeoff_info.get("head_class")
            class_to_head_type = {
                "MeanBaselineHead": "mean_baseline",
                "ResidualMLPMeanHead": "residual_mlp_mean",
                "LatentResidualHead": "latent_residual",
                "AgreementGatedHead": "agreement_gated",
                "AgreementGatedHeadV2": "agreement_gated_v2",
                "MultiPoolLowRankHead": "multi_pool_low_rank",
                "AnisotropyCorrectedHead": "anisotropy_corrected",
            }
            inferred_head_type = class_to_head_type.get(str(head_class))

        if inferred_head_type in EMBEDDING_HEAD_REGISTRY:
            logger.warning(
                f"  Checkpoint metadata head_type '{head_type}' is not a registered embedding head; "
                f"recovering as '{inferred_head_type}'"
            )
            head_type = inferred_head_type
            if not head_params:
                recovered_params = dict(head_constructor_params)
                recovered_params.pop("head_type", None)
                recovered_params.pop("hidden_size", None)
                head_params = recovered_params

    logger.info(f"  Bakeoff winner: {head_type} ({bakeoff_info.get('head_class', 'unknown')})")

    config = AutoConfig.from_pretrained(bakeoff_checkpoint, trust_remote_code=True)
    if use_flash_attention:
        config.attn_implementation = "flash_attention_2"
        logger.info("  Flash Attention 2: ENABLED")

    capabilities = load_checkpoint_capabilities(bakeoff_checkpoint, exclude_decoder)
    model = ModernBertMultiTaskModel(config=config, capabilities=capabilities, freeze_encoder=False)
    restore_checkpoint_head_architecture(model, bakeoff_checkpoint)
    model._init_encoder()

    # Create the correct embedding head architecture (so state_dict keys match)
    hidden_size = model.config.hidden_size
    new_head = create_embedding_head(
        head_type=head_type,
        hidden_size=hidden_size,
        **head_params,
    )
    model.heads["embedding"] = new_head

    # Load ALL weights including the trained embedding head
    weights_path = bakeoff_checkpoint / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        weights_path = bakeoff_checkpoint / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {bakeoff_checkpoint}")

    # Encoder weights
    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"  Loaded encoder: {len(encoder_state)} tensors")

    # ALL head weights including the trained embedding head
    loaded_heads = []
    for head_name in model.heads.keys():
        head_prefix = f"heads.{head_name}."
        head_state = {k.replace(head_prefix, ""): v for k, v in state_dict.items() if k.startswith(head_prefix)}
        if head_state:
            try:
                model.heads[head_name].load_state_dict(head_state, strict=True)
                loaded_heads.append(head_name)
            except Exception as e:
                logger.warning(f"  Could not load {head_name} head: {e}")

    emb_params = sum(p.numel() for p in model.heads["embedding"].parameters())
    logger.info(f"  Loaded {len(loaded_heads)} heads (including trained embedding head)")
    logger.info(f"  Embedding head: {type(model.heads['embedding']).__name__} ({emb_params:,} params, TRAINED weights preserved)")

    return model, head_type, head_params


def load_model_for_stage_b_v2(
    config: dict[str, Any],
    bakeoff_checkpoint: str | Path,
    exclude_decoder: bool = True,
    use_flash_attention: bool = False,
) -> tuple[ModernBertMultiTaskModel, str, dict[str, Any]]:
    """Load a bakeoff checkpoint and install AgreementGatedHeadV2 for Stage B."""
    stage_b_config = config.get("stage_b", {})
    target_head_type = stage_b_config.get("head_type", "agreement_gated_v2")
    reuse_checkpoint_head_as_is = stage_b_config.get("reuse_checkpoint_head_as_is", True)
    target_head_params = stage_b_config.get("head_params") or get_head_params_from_config(config, target_head_type)

    model, source_head_type, source_head_params = load_model_with_trained_head(
        bakeoff_checkpoint=bakeoff_checkpoint,
        exclude_decoder=exclude_decoder,
        use_flash_attention=use_flash_attention,
    )

    if source_head_type == target_head_type and reuse_checkpoint_head_as_is:
        logger.info(
            f"  Stage B reusing checkpoint head as-is: {target_head_type} "
            f"(no head reinitialization, full checkpoint weights preserved)"
        )
        return model, source_head_type, source_head_params

    if source_head_type == target_head_type and (not target_head_params or target_head_params == source_head_params):
        logger.info(f"  Stage B head already matches target {target_head_type}; keeping trained head as-is")
        return model, source_head_type, source_head_params

    hidden_size = model.config.hidden_size
    source_embedding_state = model.heads["embedding"].state_dict()
    stage_b_head = create_embedding_head(
        head_type=target_head_type,
        hidden_size=hidden_size,
        **target_head_params,
    )
    matched, skipped = load_matching_state_dict(stage_b_head, source_embedding_state)
    model.heads["embedding"] = stage_b_head

    logger.info(
        f"  Stage B head upgraded: {source_head_type} -> {target_head_type} "
        f"(matched {matched} tensors, skipped {skipped})"
    )
    return model, target_head_type, target_head_params


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


def get_head_params_from_config(
    config: dict[str, Any],
    head_type: str,
) -> dict[str, Any]:
    """Return merged params for a configured head type."""
    for configured_head_type, params in get_configured_head_experiments(config):
        if configured_head_type == head_type:
            return copy.deepcopy(params)
    return {}


def head_supports_mode_routing(head: nn.Module) -> bool:
    """Return whether the embedding head supports query/document routing."""
    return getattr(head, "pooling", None) == "agreement_gated_v2"


def forward_embedding_head(
    head: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mode: str = "document",
    return_aux: bool = False,
) -> torch.Tensor | dict[str, Any]:
    """Forward helper that preserves compatibility with older heads."""
    if head_supports_mode_routing(head):
        return head(hidden_states, attention_mask, mode=mode, return_aux=return_aux)

    embedding = head(hidden_states, attention_mask)
    if return_aux:
        return {"embedding": embedding}
    return embedding


def load_matching_state_dict(
    module: nn.Module,
    source_state_dict: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Load only keys whose names and shapes match the target module."""
    target_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0

    for key, value in source_state_dict.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped += 1

    target_state.update(matched)
    module.load_state_dict(target_state)
    return len(matched), skipped


def get_embedding_modes(stage_config: dict[str, Any] | None = None) -> dict[str, str]:
    """Return anchor/positive/negative routing modes for embedding forward."""
    stage_cfg = stage_config or {}
    return {
        "anchor": stage_cfg.get("anchor_mode", "document"),
        "positive": stage_cfg.get("positive_mode", "document"),
        "negative": stage_cfg.get("negative_mode", "document"),
    }


def resolve_mode_routing_for_batch(
    slice_tags: list[str],
    mode_routing_config: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Resolve per-sample query/document routing modes for a mixed batch."""
    config = mode_routing_config or {}
    defaults = get_embedding_modes({
        "anchor_mode": config.get("default_anchor_mode", config.get("anchor_mode", "document")),
        "positive_mode": config.get("default_positive_mode", config.get("positive_mode", "document")),
        "negative_mode": config.get("default_negative_mode", config.get("negative_mode", "document")),
    })

    anchor_query_slices = set(config.get("anchor_query_slices", config.get("query_slices", [])))
    positive_query_slices = set(config.get("positive_query_slices", []))
    negative_query_slices = set(config.get("negative_query_slices", []))

    anchor_modes = ["query" if tag in anchor_query_slices else defaults["anchor"] for tag in slice_tags]
    positive_modes = ["query" if tag in positive_query_slices else defaults["positive"] for tag in slice_tags]
    negative_modes = ["query" if tag in negative_query_slices else defaults["negative"] for tag in slice_tags]

    return {
        "anchor": anchor_modes,
        "positive": positive_modes,
        "negative": negative_modes,
    }


def _reorder_batched_output(
    chunks: list[Any],
    index_chunks: list[list[int]],
) -> Any:
    """Restore original batch order after grouped mode forwards."""
    if not chunks:
        raise ValueError("Cannot reorder empty output chunks")

    ordered_indices = [index for chunk in index_chunks for index in chunk]
    first_chunk = chunks[0]

    if torch.is_tensor(first_chunk):
        concat = torch.cat(chunks, dim=0)
        inverse_order = torch.tensor(ordered_indices, dtype=torch.long, device=concat.device).argsort()
        return concat.index_select(0, inverse_order)

    if isinstance(first_chunk, dict):
        return {
            key: _reorder_batched_output([chunk[key] for chunk in chunks], index_chunks)
            for key in first_chunk
        }

    raise TypeError(f"Unsupported batched output type: {type(first_chunk)!r}")


def forward_embedding_head_batch(
    head: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sample_modes: list[str] | None = None,
    return_aux: bool = False,
) -> torch.Tensor | dict[str, Any]:
    """Forward an embedding head with optional per-sample mode routing."""
    batch_size = hidden_states.size(0)
    if batch_size == 0:
        raise ValueError("Cannot forward an empty batch through embedding head")

    if not sample_modes:
        return forward_embedding_head(
            head,
            hidden_states,
            attention_mask,
            mode="document",
            return_aux=return_aux,
        )

    if not head_supports_mode_routing(head):
        return forward_embedding_head(
            head,
            hidden_states,
            attention_mask,
            mode=sample_modes[0],
            return_aux=return_aux,
        )

    unique_modes = sorted(set(sample_modes))
    if len(unique_modes) == 1:
        return forward_embedding_head(
            head,
            hidden_states,
            attention_mask,
            mode=unique_modes[0],
            return_aux=return_aux,
        )

    output_chunks: list[torch.Tensor | dict[str, Any]] = []
    index_chunks: list[list[int]] = []
    for mode in unique_modes:
        batch_indices = [i for i, sample_mode in enumerate(sample_modes) if sample_mode == mode]
        if not batch_indices:
            continue

        index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=hidden_states.device)
        mode_hidden = hidden_states.index_select(0, index_tensor)
        mode_mask = attention_mask.index_select(0, index_tensor) if attention_mask is not None else None
        mode_output = forward_embedding_head(
            head,
            mode_hidden,
            mode_mask,
            mode=mode,
            return_aux=return_aux,
        )

        output_chunks.append(mode_output)
        index_chunks.append(batch_indices)

    if not output_chunks:
        raise RuntimeError("Failed to route embedding head outputs for mixed-mode batch")
    return _reorder_batched_output(output_chunks, index_chunks)


def _salience_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy over token-salience weights."""
    clamped = weights.clamp(min=eps)
    return -(clamped * clamped.log()).sum(dim=-1)


def compute_stage_b_auxiliary_loss(
    anchor_aux: dict[str, Any],
    slice_tags: list[str],
    device: torch.device,
    aux_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Stage B salience-sharpening auxiliary losses for V2."""
    config = aux_config or {}
    if not config.get("enabled", False):
        return torch.tensor(0.0, device=device), {}

    salience = anchor_aux.get("salience")
    if not salience:
        return torch.tensor(0.0, device=device), {}

    total_loss = torch.tensor(0.0, device=device)
    metrics: dict[str, float] = {}

    slice_specs = [
        ("wrong_person", "role", salience.get("role_weights"), config.get("role_entropy_weight", 0.0)),
        ("wrong_time", "temporal", salience.get("temporal_weights"), config.get("temporal_entropy_weight", 0.0)),
        ("safety_emotion", "safety", salience.get("safety_weights"), config.get("safety_entropy_weight", 0.0)),
    ]

    for slice_name, metric_name, weights, loss_weight in slice_specs:
        if weights is None or loss_weight <= 0:
            continue
        indices = [i for i, tag in enumerate(slice_tags) if tag == slice_name]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        entropy = _salience_entropy(weights[index_tensor]).mean()
        total_loss = total_loss + (float(loss_weight) * entropy)
        metrics[f"{metric_name}_entropy"] = entropy.item()

    return total_loss, metrics


def encode_triplet_batch(
    model: ModernBertMultiTaskModel,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor | None]:
    """Encode anchor/positive/negative text once with the frozen encoder.

    Skips negative encoding entirely when the batch contains no triplet
    samples (pure pair batch), saving ~33% encoder forward cost.
    """
    anchor_ids = batch["anchor_input_ids"].to(device)
    anchor_mask = batch["anchor_attention_mask"].to(device)
    positive_ids = batch["positive_input_ids"].to(device)
    positive_mask = batch["positive_attention_mask"].to(device)
    hard_neg_mask = batch["hard_negative_mask"].to(device)

    triplet_indices = batch["triplet_indices"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    has_triplets = triplet_indices.numel() > 0

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

            if has_triplets:
                negative_ids = batch["negative_input_ids"].to(device)
                negative_mask = batch["negative_attention_mask"].to(device)
                enc_out = model.encoder(input_ids=negative_ids, attention_mask=negative_mask)
                negative_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)
            else:
                negative_hidden = None
                negative_mask = None

    return {
        "anchor_hidden": anchor_hidden,
        "anchor_mask": anchor_mask,
        "positive_hidden": positive_hidden,
        "positive_mask": positive_mask,
        "negative_hidden": negative_hidden,
        "negative_mask": negative_mask,
        "hard_neg_mask": hard_neg_mask,
        "triplet_indices": triplet_indices,
        "pair_indices": pair_indices,
    }


def compute_teacher_distillation_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor | None,
    batch: dict[str, Any],
    triplet_idx: torch.Tensor,
    pair_idx: torch.Tensor,
    routed_modes: dict[str, list[str]],
    teacher_cache: TeacherEmbeddingCache | None,
    distillation_config: dict[str, Any] | None,
    teacher_projection: nn.Module | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute optional teacher-guided vector and ranking losses."""
    if teacher_cache is None or not distillation_config or not distillation_config.get("enabled", False):
        return torch.tensor(0.0, device=device), {}

    losses_config = distillation_config.get("losses", {})
    vector_weight = float(losses_config.get("teacher_vector_weight", 0.0))
    ranking_weight = float(losses_config.get("teacher_ranking_weight", 0.0))
    teacher_temperature = float(losses_config.get("teacher_temperature", 0.05))

    teacher_anchor, anchor_found = teacher_cache.lookup(batch.get("anchor_texts", []), routed_modes["anchor"], device)
    teacher_positive, positive_found = teacher_cache.lookup(batch.get("positive_texts", []), routed_modes["positive"], device)

    negative_modes = [routed_modes["negative"][i] for i in triplet_idx.detach().cpu().tolist()] if triplet_idx.numel() > 0 else []
    teacher_negative, negative_found = teacher_cache.lookup(batch.get("negative_texts", []), negative_modes, device)

    total_loss = torch.tensor(0.0, device=device)
    metrics: dict[str, float] = {
        "teacher_anchor_found_rate": anchor_found.float().mean().item() if anchor_found.numel() > 0 else 0.0,
        "teacher_positive_found_rate": positive_found.float().mean().item() if positive_found.numel() > 0 else 0.0,
        "teacher_negative_found_rate": negative_found.float().mean().item() if negative_found.numel() > 0 else 0.0,
    }

    def _project_teacher(teacher_tensor: torch.Tensor, student_dim: int) -> torch.Tensor | None:
        if teacher_projection is not None:
            return teacher_projection(teacher_tensor)
        if teacher_tensor.shape[-1] == student_dim:
            return teacher_tensor
        return None

    if vector_weight > 0:
        vector_terms: list[torch.Tensor] = []
        teacher_anchor_proj = _project_teacher(teacher_anchor, anchor_emb.shape[-1])
        teacher_positive_proj = _project_teacher(teacher_positive, positive_emb.shape[-1])
        teacher_negative_proj = _project_teacher(teacher_negative, negative_emb.shape[-1]) if negative_emb is not None and teacher_negative.numel() > 0 else None

        if teacher_anchor_proj is not None and anchor_found.any():
            vector_terms.append(1.0 - F.cosine_similarity(
                F.normalize(anchor_emb[anchor_found], p=2, dim=-1),
                F.normalize(teacher_anchor_proj[anchor_found], p=2, dim=-1),
                dim=-1,
            ).mean())
        if teacher_positive_proj is not None and positive_found.any():
            vector_terms.append(1.0 - F.cosine_similarity(
                F.normalize(positive_emb[positive_found], p=2, dim=-1),
                F.normalize(teacher_positive_proj[positive_found], p=2, dim=-1),
                dim=-1,
            ).mean())
        if teacher_negative_proj is not None and negative_emb is not None and negative_found.any():
            vector_terms.append(1.0 - F.cosine_similarity(
                F.normalize(negative_emb[negative_found], p=2, dim=-1),
                F.normalize(teacher_negative_proj[negative_found], p=2, dim=-1),
                dim=-1,
            ).mean())

        if vector_terms:
            vector_loss = torch.stack(vector_terms).mean()
            total_loss = total_loss + (vector_weight * vector_loss)
            metrics["teacher_vector_loss"] = vector_loss.item()
        else:
            metrics["teacher_vector_loss"] = 0.0

    if ranking_weight > 0:
        ranking_terms: list[torch.Tensor] = []

        if pair_idx.numel() > 1:
            pair_mask = anchor_found[pair_idx] & positive_found[pair_idx]
            if pair_mask.sum().item() >= 2:
                pair_indices = pair_idx[pair_mask]
                student_pair_scores = F.normalize(anchor_emb[pair_indices], p=2, dim=-1) @ F.normalize(positive_emb[pair_indices], p=2, dim=-1).T
                teacher_pair_scores = F.normalize(teacher_anchor[pair_indices], p=2, dim=-1) @ F.normalize(teacher_positive[pair_indices], p=2, dim=-1).T
                ranking_terms.append(F.mse_loss(student_pair_scores, teacher_pair_scores))

        if triplet_idx.numel() > 0 and negative_emb is not None and negative_found.numel() > 0:
            triplet_mask = anchor_found[triplet_idx] & positive_found[triplet_idx] & negative_found
            if triplet_mask.any():
                triplet_indices = triplet_idx[triplet_mask]
                student_logits = torch.stack([
                    F.cosine_similarity(anchor_emb[triplet_indices], positive_emb[triplet_indices], dim=-1),
                    F.cosine_similarity(anchor_emb[triplet_indices], negative_emb[triplet_mask], dim=-1),
                ], dim=-1) / teacher_temperature
                teacher_logits = torch.stack([
                    F.cosine_similarity(teacher_anchor[triplet_indices], teacher_positive[triplet_indices], dim=-1),
                    F.cosine_similarity(teacher_anchor[triplet_indices], teacher_negative[triplet_mask], dim=-1),
                ], dim=-1) / teacher_temperature
                ranking_terms.append(
                    F.kl_div(
                        F.log_softmax(student_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction="batchmean",
                    )
                )

        if ranking_terms:
            ranking_loss = torch.stack(ranking_terms).mean()
            total_loss = total_loss + (ranking_weight * ranking_loss)
            metrics["teacher_ranking_loss"] = ranking_loss.item()
        else:
            metrics["teacher_ranking_loss"] = 0.0

    return total_loss, metrics


def compute_distillation_label_auxiliary_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor | None,
    batch: dict[str, Any],
    triplet_idx: torch.Tensor,
    device: torch.device,
    aux_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply label-aware margin pressure for safety/emotion near-miss negatives."""
    config = aux_config or {}
    if not config.get("enabled", False) or negative_emb is None or triplet_idx.numel() == 0:
        return torch.tensor(0.0, device=device), {}

    apply_slices = set(config.get("apply_slices", ["safety_emotion"]))
    if not apply_slices:
        return torch.tensor(0.0, device=device), {}

    safety_weight = float(config.get("safety_label_weight", 0.0))
    emotion_weight = float(config.get("emotion_label_weight", 0.0))
    if safety_weight <= 0 and emotion_weight <= 0:
        return torch.tensor(0.0, device=device), {}

    base_margin = float(config.get("base_margin", 0.05))
    safety_margin_scale = float(config.get("safety_margin_scale", 0.03))
    emotion_margin = float(config.get("emotion_margin", 0.02))

    triplet_anchor = anchor_emb[triplet_idx]
    triplet_positive = positive_emb[triplet_idx]
    positive_similarity = F.cosine_similarity(triplet_anchor, triplet_positive, dim=-1)
    negative_similarity = F.cosine_similarity(triplet_anchor, negative_emb, dim=-1)
    observed_margin = positive_similarity - negative_similarity

    slice_tags = batch.get("slice_tags", [])
    anchor_safety_labels = batch.get("anchor_safety_labels", [])
    negative_safety_labels = batch.get("negative_safety_labels", [])
    anchor_emotion_labels = batch.get("anchor_emotion_labels", [])
    negative_emotion_labels = batch.get("negative_emotion_labels", [])
    triplet_indices = triplet_idx.detach().cpu().tolist()

    safety_losses: list[torch.Tensor] = []
    emotion_losses: list[torch.Tensor] = []
    safety_gap_sum = 0.0
    safety_count = 0
    emotion_count = 0

    for local_idx, batch_idx in enumerate(triplet_indices):
        if batch_idx >= len(slice_tags) or slice_tags[batch_idx] not in apply_slices:
            continue

        if safety_weight > 0 and local_idx < len(negative_safety_labels) and batch_idx < len(anchor_safety_labels):
            anchor_safety_score = _coerce_safety_label_score(anchor_safety_labels[batch_idx])
            negative_safety_score = _coerce_safety_label_score(negative_safety_labels[local_idx])
            if anchor_safety_score is not None and negative_safety_score is not None:
                severity_gap = abs(negative_safety_score - anchor_safety_score)
                if severity_gap > 0:
                    target_margin = base_margin + (safety_margin_scale * severity_gap)
                    safety_losses.append(F.relu(target_margin - observed_margin[local_idx]))
                    safety_gap_sum += severity_gap
                    safety_count += 1

        if emotion_weight > 0 and local_idx < len(negative_emotion_labels) and batch_idx < len(anchor_emotion_labels):
            anchor_emotion = anchor_emotion_labels[batch_idx]
            negative_emotion = negative_emotion_labels[local_idx]
            if anchor_emotion and negative_emotion and str(anchor_emotion).strip().lower() != str(negative_emotion).strip().lower():
                target_margin = base_margin + emotion_margin
                emotion_losses.append(F.relu(target_margin - observed_margin[local_idx]))
                emotion_count += 1

    total_loss = torch.tensor(0.0, device=device)
    metrics: dict[str, float] = {}

    if safety_losses:
        safety_loss = torch.stack(safety_losses).mean()
        total_loss = total_loss + (safety_weight * safety_loss)
        metrics["safety_label_loss"] = safety_loss.item()
        metrics["safety_label_samples"] = float(safety_count)
        metrics["safety_label_avg_gap"] = safety_gap_sum / max(safety_count, 1)

    if emotion_losses:
        emotion_loss = torch.stack(emotion_losses).mean()
        total_loss = total_loss + (emotion_weight * emotion_loss)
        metrics["emotion_label_loss"] = emotion_loss.item()
        metrics["emotion_label_samples"] = float(emotion_count)

    return total_loss, metrics


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
    mode_routing_config: dict[str, Any] | None = None,
    aux_objectives_config: dict[str, Any] | None = None,
    teacher_cache: TeacherEmbeddingCache | None = None,
    distillation_config: dict[str, Any] | None = None,
    teacher_projection: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
    anchor_hidden = encoded["anchor_hidden"]
    anchor_mask = encoded["anchor_mask"]
    positive_hidden = encoded["positive_hidden"]
    positive_mask = encoded["positive_mask"]
    negative_hidden = encoded["negative_hidden"]
    negative_mask = encoded["negative_mask"]
    hard_neg_mask = encoded["hard_neg_mask"]
    triplet_idx = encoded["triplet_indices"]
    pair_idx = encoded["pair_indices"]

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    with amp_context:
        embedding_head = model.heads["embedding"]
        slice_tags = batch.get("slice_tags", [])
        routed_modes = resolve_mode_routing_for_batch(slice_tags, mode_routing_config)
        triplet_batch_indices = triplet_idx.detach().cpu().tolist() if triplet_idx.numel() > 0 else []
        negative_modes = [routed_modes["negative"][i] for i in triplet_batch_indices]
        request_aux = bool(aux_objectives_config and aux_objectives_config.get("enabled", False))

        anchor_out = forward_embedding_head_batch(
            embedding_head,
            anchor_hidden,
            anchor_mask,
            sample_modes=routed_modes["anchor"],
            return_aux=request_aux,
        )
        positive_out = forward_embedding_head_batch(
            embedding_head,
            positive_hidden,
            positive_mask,
            sample_modes=routed_modes["positive"],
            return_aux=False,
        )
        anchor_emb = anchor_out["embedding"] if isinstance(anchor_out, dict) else anchor_out
        positive_emb = positive_out["embedding"] if isinstance(positive_out, dict) else positive_out

        loss = torch.tensor(0.0, device=device)
        loss_count = 0
        negative_emb = None

        # --- Triplet sub-batch: explicit negatives ---
        if triplet_idx.numel() > 0 and negative_hidden is not None:
            negative_out = forward_embedding_head_batch(
                embedding_head,
                negative_hidden,
                negative_mask,
                sample_modes=negative_modes,
                return_aux=False,
            )
            negative_emb = negative_out["embedding"] if isinstance(negative_out, dict) else negative_out

            if matryoshka_dims:
                trip_loss = torch.tensor(0.0, device=device)
                for dim in matryoshka_dims:
                    a_d = F.normalize(anchor_emb[triplet_idx, :dim], p=2, dim=-1)
                    p_d = F.normalize(positive_emb[triplet_idx, :dim], p=2, dim=-1)
                    n_d = F.normalize(negative_emb[:, :dim], p=2, dim=-1).unsqueeze(1)
                    hn_mask = hard_neg_mask.unsqueeze(1)
                    trip_loss = trip_loss + loss_fn(
                        anchor=a_d, positive=p_d, negatives=n_d, hard_negative_mask=hn_mask,
                    )
                loss = loss + trip_loss / len(matryoshka_dims)
            else:
                a_trip = anchor_emb[triplet_idx]
                p_trip = positive_emb[triplet_idx]
                negatives = negative_emb.unsqueeze(1)
                hn_mask = hard_neg_mask.unsqueeze(1)
                loss = loss + loss_fn(
                    anchor=a_trip, positive=p_trip,
                    negatives=negatives, hard_negative_mask=hn_mask,
                )
            loss_count += 1

        # --- Pair sub-batch: in-batch negatives only ---
        if pair_idx.numel() > 0:
            a_pair = anchor_emb[pair_idx]
            p_pair = positive_emb[pair_idx]

            if matryoshka_dims:
                pair_loss = torch.tensor(0.0, device=device)
                for dim in matryoshka_dims:
                    a_d = F.normalize(a_pair[:, :dim], p=2, dim=-1)
                    p_d = F.normalize(p_pair[:, :dim], p=2, dim=-1)
                    pair_loss = pair_loss + loss_fn(anchor=a_d, positive=p_d, negatives=None)
                loss = loss + pair_loss / len(matryoshka_dims)
            else:
                loss = loss + loss_fn(anchor=a_pair, positive=p_pair, negatives=None)
            loss_count += 1

        if loss_count > 1:
            loss = loss / loss_count

        aux_loss = torch.tensor(0.0, device=device)
        aux_metrics: dict[str, float] = {}
        if isinstance(anchor_out, dict):
            aux_loss, aux_metrics = compute_stage_b_auxiliary_loss(
                anchor_aux=anchor_out,
                slice_tags=batch.get("slice_tags", []),
                device=device,
                aux_config=aux_objectives_config,
            )
            loss = loss + aux_loss

        teacher_loss, teacher_metrics = compute_teacher_distillation_loss(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negative_emb=negative_emb,
            batch=batch,
            triplet_idx=triplet_idx,
            pair_idx=pair_idx,
            routed_modes=routed_modes,
            teacher_cache=teacher_cache,
            distillation_config=distillation_config,
            teacher_projection=teacher_projection,
            device=device,
        )
        loss = loss + teacher_loss

        distillation_aux_loss, distillation_aux_metrics = compute_distillation_label_auxiliary_loss(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negative_emb=negative_emb,
            batch=batch,
            triplet_idx=triplet_idx,
            device=device,
            aux_config=(distillation_config or {}).get("auxiliary_losses"),
        )
        loss = loss + distillation_aux_loss

    # Metrics: pos_sim always available; neg_sim only for triplet sub-batch
    with torch.no_grad():
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
        if triplet_idx.numel() > 0 and negative_emb is not None:
            neg_sim = F.cosine_similarity(anchor_emb[triplet_idx], negative_emb).mean().item()
            margin = pos_sim - neg_sim
        else:
            neg_sim = 0.0
            margin = pos_sim

    result = {
        "total_loss": loss,
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "margin": margin,
        "aux_loss": aux_loss,
        "teacher_loss": teacher_loss,
        "distillation_aux_loss": distillation_aux_loss,
    }
    for metric_name, metric_value in aux_metrics.items():
        result[f"aux_{metric_name}"] = torch.tensor(metric_value, device=device)
    for metric_name, metric_value in teacher_metrics.items():
        result[f"distill_{metric_name}"] = torch.tensor(metric_value, device=device)
    for metric_name, metric_value in distillation_aux_metrics.items():
        result[f"distill_{metric_name}"] = torch.tensor(metric_value, device=device)
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
    mode_routing_config: dict[str, Any] | None = None,
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
    total_triplet_samples = 0
    total_pair_samples = 0
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
        triplet_idx = encoded["triplet_indices"]
        pair_idx = encoded["pair_indices"]
        slice_tags = batch.get("slice_tags", [])
        routed_modes = resolve_mode_routing_for_batch(slice_tags, mode_routing_config)
        triplet_batch_indices = triplet_idx.detach().cpu().tolist() if triplet_idx.numel() > 0 else []
        negative_modes = [routed_modes["negative"][i] for i in triplet_batch_indices]

        with amp_context:
            embedding_head = model.heads["embedding"]
            anchor_emb = forward_embedding_head_batch(
                embedding_head,
                anchor_hidden,
                anchor_mask,
                sample_modes=routed_modes["anchor"],
                return_aux=False,
            )
            positive_emb = forward_embedding_head_batch(
                embedding_head,
                positive_hidden,
                positive_mask,
                sample_modes=routed_modes["positive"],
                return_aux=False,
            )

            batch_loss = torch.tensor(0.0, device=device)
            loss_count = 0

            # Triplet sub-batch
            if triplet_idx.numel() > 0 and negative_hidden is not None:
                negative_emb = forward_embedding_head_batch(
                    embedding_head,
                    negative_hidden,
                    negative_mask,
                    sample_modes=negative_modes,
                    return_aux=False,
                )
                a_trip = anchor_emb[triplet_idx]
                p_trip = positive_emb[triplet_idx]
                negatives = negative_emb.unsqueeze(1)
                hn_mask = hard_neg_mask.unsqueeze(1)
                batch_loss = batch_loss + loss_fn(
                    anchor=a_trip, positive=p_trip,
                    negatives=negatives, hard_negative_mask=hn_mask,
                )
                loss_count += 1

                # Triplet metrics
                trip_pos_sim = F.cosine_similarity(a_trip, p_trip)
                trip_neg_sim = F.cosine_similarity(a_trip, negative_emb)
                correct = (trip_pos_sim > trip_neg_sim).float()
                n_trip = triplet_idx.numel()
                total_pos_sim += trip_pos_sim.sum().item()
                total_neg_sim += trip_neg_sim.sum().item()
                total_correct += correct.sum().item()
                total_triplet_samples += n_trip

                if hard_neg_mask.any():
                    total_hard_neg_correct += correct[hard_neg_mask].sum().item()
                    total_hard_neg_samples += hard_neg_mask.sum().item()

            # Pair sub-batch
            if pair_idx.numel() > 0:
                a_pair = anchor_emb[pair_idx]
                p_pair = positive_emb[pair_idx]
                batch_loss = batch_loss + loss_fn(anchor=a_pair, positive=p_pair, negatives=None)
                loss_count += 1

                pair_pos_sim = F.cosine_similarity(a_pair, p_pair)
                total_pos_sim += pair_pos_sim.sum().item()
                total_pair_samples += pair_idx.numel()

            if loss_count > 1:
                batch_loss = batch_loss / loss_count

        batch_size = anchor_emb.size(0)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    model.train()
    if total_samples == 0:
        return {"val_loss": 0, "pos_sim": 0, "neg_sim": 0, "margin": 0, "accuracy": 0}

    avg_pos_sim = total_pos_sim / total_samples
    avg_neg_sim = total_neg_sim / total_triplet_samples if total_triplet_samples > 0 else 0.0
    triplet_accuracy = total_correct / total_triplet_samples if total_triplet_samples > 0 else 0.0
    metrics = {
        "val_loss": total_loss / total_samples,
        "pos_sim": avg_pos_sim,
        "neg_sim": avg_neg_sim,
        "margin": avg_pos_sim - avg_neg_sim,
        "accuracy": triplet_accuracy,
        "hard_neg_accuracy": total_hard_neg_correct / total_hard_neg_samples if total_hard_neg_samples > 0 else 0.0,
        "hard_neg_samples": total_hard_neg_samples,
        "total_samples": total_samples,
        "triplet_samples": total_triplet_samples,
        "pair_samples": total_pair_samples,
    }
    logger.info(
        f"  val_loss={metrics['val_loss']:.4f} | pos_sim={avg_pos_sim:.4f} neg_sim={avg_neg_sim:.4f} "
        f"margin={metrics['margin']:.4f} | acc={triplet_accuracy:.4f} "
        f"hard_neg_acc={metrics['hard_neg_accuracy']:.4f} "
        f"(triplets={total_triplet_samples} pairs={total_pair_samples})"
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
            "triplet_samples": 0.0,
            "pair_samples": 0.0,
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
        triplet_idx = encoded["triplet_indices"]
        pair_idx = encoded["pair_indices"]
        has_triplets = triplet_idx.numel() > 0 and negative_hidden is not None

        with amp_context:
            for head_type, head in candidate_heads.items():
                anchor_emb = head(anchor_hidden, anchor_mask)
                positive_emb = head(positive_hidden, positive_mask)

                batch_loss = torch.tensor(0.0, device=anchor_emb.device)
                loss_count = 0

                # Triplet sub-batch
                if has_triplets:
                    negative_emb = head(negative_hidden, negative_mask)
                    a_trip = anchor_emb[triplet_idx]
                    p_trip = positive_emb[triplet_idx]
                    negatives = negative_emb.unsqueeze(1)
                    hn_mask = hard_neg_mask.unsqueeze(1)
                    batch_loss = batch_loss + loss_modules[head_type](
                        anchor=a_trip, positive=p_trip,
                        negatives=negatives, hard_negative_mask=hn_mask,
                    )
                    loss_count += 1

                    trip_pos_sim = F.cosine_similarity(a_trip, p_trip)
                    trip_neg_sim = F.cosine_similarity(a_trip, negative_emb)
                    correct = (trip_pos_sim > trip_neg_sim).float()
                    n_trip = triplet_idx.numel()
                    totals[head_type]["pos_sim"] += trip_pos_sim.sum().item()
                    totals[head_type]["neg_sim"] += trip_neg_sim.sum().item()
                    totals[head_type]["correct"] += correct.sum().item()
                    totals[head_type]["triplet_samples"] += n_trip

                    if hard_neg_mask.any():
                        totals[head_type]["hard_neg_correct"] += correct[hard_neg_mask].sum().item()
                        totals[head_type]["hard_neg_samples"] += hard_neg_mask.sum().item()

                # Pair sub-batch
                if pair_idx.numel() > 0:
                    a_pair = anchor_emb[pair_idx]
                    p_pair = positive_emb[pair_idx]
                    batch_loss = batch_loss + loss_modules[head_type](
                        anchor=a_pair, positive=p_pair, negatives=None,
                    )
                    loss_count += 1

                    pair_pos_sim = F.cosine_similarity(a_pair, p_pair)
                    totals[head_type]["pos_sim"] += pair_pos_sim.sum().item()
                    totals[head_type]["pair_samples"] += pair_idx.numel()

                if loss_count > 1:
                    batch_loss = batch_loss / loss_count

                batch_size = anchor_emb.size(0)
                totals[head_type]["loss"] += batch_loss.item() * batch_size
                totals[head_type]["samples"] += batch_size

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
        triplet_count = int(total["triplet_samples"])
        avg_neg_sim = total["neg_sim"] / triplet_count if triplet_count > 0 else 0.0
        hard_neg_samples = int(total["hard_neg_samples"])
        triplet_accuracy = total["correct"] / triplet_count if triplet_count > 0 else 0.0
        metrics_by_head[head_type] = {
            "val_loss": total["loss"] / sample_count,
            "pos_sim": avg_pos_sim,
            "neg_sim": avg_neg_sim,
            "margin": avg_pos_sim - avg_neg_sim,
            "accuracy": triplet_accuracy,
            "hard_neg_accuracy": total["hard_neg_correct"] / hard_neg_samples if hard_neg_samples > 0 else 0.0,
            "hard_neg_samples": hard_neg_samples,
            "total_samples": sample_count,
            "triplet_samples": triplet_count,
            "pair_samples": int(total["pair_samples"]),
        }

    return metrics_by_head


@torch.no_grad()
def compute_per_slice_metrics(
    batch: dict[str, Any],
    model: ModernBertMultiTaskModel,
    loss_fn: FamilyContrastiveLoss,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    head: nn.Module | None = None,
    mode_routing_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-slice loss and pos_sim for a single batch.

    Called at logging steps only (not every step) to avoid overhead.
    Returns ``{slice_name: {"loss": ..., "pos_sim": ..., "count": ...}}``.
    """
    slice_tags: list[str] = batch.get("slice_tags", [])
    if not slice_tags:
        return {}

    encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
    anchor_hidden = encoded["anchor_hidden"]
    anchor_mask = encoded["anchor_mask"]
    positive_hidden = encoded["positive_hidden"]
    positive_mask = encoded["positive_mask"]
    negative_hidden = encoded["negative_hidden"]
    negative_mask = encoded["negative_mask"]
    hard_neg_mask = encoded["hard_neg_mask"]
    triplet_idx = encoded["triplet_indices"]
    routed_modes = resolve_mode_routing_for_batch(slice_tags, mode_routing_config)
    triplet_batch_indices = triplet_idx.detach().cpu().tolist() if triplet_idx.numel() > 0 else []
    negative_modes = [routed_modes["negative"][i] for i in triplet_batch_indices]

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    embedding_head = head if head is not None else model.heads["embedding"]

    with amp_context:
        anchor_emb = forward_embedding_head_batch(
            embedding_head,
            anchor_hidden,
            anchor_mask,
            sample_modes=routed_modes["anchor"],
            return_aux=False,
        )
        positive_emb = forward_embedding_head_batch(
            embedding_head,
            positive_hidden,
            positive_mask,
            sample_modes=routed_modes["positive"],
            return_aux=False,
        )

    # Build per-slice index groups
    slice_indices: dict[str, list[int]] = {}
    for i, tag in enumerate(slice_tags):
        slice_indices.setdefault(tag, []).append(i)

    # Build triplet set for quick lookup
    triplet_set = set(triplet_idx.cpu().tolist()) if triplet_idx.numel() > 0 else set()

    # Encode negatives once if any triplets exist
    negative_emb = None
    if triplet_idx.numel() > 0 and negative_hidden is not None:
        with amp_context:
            negative_emb = forward_embedding_head_batch(
                embedding_head,
                negative_hidden,
                negative_mask,
                sample_modes=negative_modes,
                return_aux=False,
            )

    results: dict[str, dict[str, float]] = {}
    for slice_name, indices in slice_indices.items():
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        a_slice = anchor_emb[idx_t]
        p_slice = positive_emb[idx_t]

        pos_sim = F.cosine_similarity(a_slice, p_slice).mean().item()

        # Determine if this slice has triplets or pairs
        slice_triplet_indices = [i for i in indices if i in triplet_set]
        if slice_triplet_indices and negative_emb is not None:
            # Map batch-level triplet indices to negative tensor indices
            triplet_list = triplet_idx.cpu().tolist()
            neg_indices = [triplet_list.index(i) for i in slice_triplet_indices]
            neg_idx_t = torch.tensor(neg_indices, dtype=torch.long, device=device)
            trip_idx_t = torch.tensor(slice_triplet_indices, dtype=torch.long, device=device)
            with amp_context:
                slice_loss = loss_fn(
                    anchor=anchor_emb[trip_idx_t],
                    positive=positive_emb[trip_idx_t],
                    negatives=negative_emb[neg_idx_t].unsqueeze(1),
                    hard_negative_mask=hard_neg_mask[neg_idx_t].unsqueeze(1),
                ).item()
        elif len(indices) >= 2:
            # Pair sub-batch needs at least 2 for in-batch negatives
            with amp_context:
                slice_loss = loss_fn(anchor=a_slice, positive=p_slice, negatives=None).item()
        else:
            slice_loss = 0.0

        results[slice_name] = {
            "loss": slice_loss,
            "pos_sim": pos_sim,
            "count": len(indices),
        }

    return results


def format_per_slice_log(slice_metrics: dict[str, dict[str, float]]) -> str:
    """Format per-slice metrics into a compact log line."""
    parts = []
    for sn in sorted(slice_metrics):
        m = slice_metrics[sn]
        parts.append(f"{sn}={m['loss']:.3f}")
    return " ".join(parts)


# =============================================================================
# Per-Slice Evaluation (Epic 3.1)
# =============================================================================


@torch.no_grad()
def evaluate_by_slice(
    model: ModernBertMultiTaskModel,
    val_loader: DataLoader,
    loss_fn: FamilyContrastiveLoss,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
    head: nn.Module | None = None,
    mode_routing_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate each held-out slice separately and return per-slice metrics.

    Args:
        model: The multi-task model (encoder + heads).
        val_loader: Validation DataLoader with mixed triplet/pair batches.
        loss_fn: The contrastive loss function.
        device: Torch device.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: AMP data type.
        max_batches: Cap on number of eval batches (None = all).
        head: Override embedding head (used for multi-head bakeoff).

    Returns:
        Dict mapping slice names to metric dicts, plus an ``_aggregate`` key.
        Triplet slices: val_loss, pos_sim, neg_sim, margin, accuracy,
                        hard_neg_accuracy, sample_count.
        Pair slices (query_doc): val_loss, pair_accuracy, pos_sim, sample_count.
    """
    model.eval()
    embedding_head = head if head is not None else model.heads["embedding"]

    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )

    # Per-slice accumulators
    slice_acc: dict[str, dict[str, float]] = {}

    def _ensure_slice(name: str) -> dict[str, float]:
        if name not in slice_acc:
            slice_acc[name] = {
                "loss_sum": 0.0,
                "pos_sim_sum": 0.0,
                "neg_sim_sum": 0.0,
                "correct": 0.0,
                "hard_neg_correct": 0.0,
                "hard_neg_count": 0.0,
                "triplet_count": 0.0,
                "pair_count": 0.0,
                "pair_correct": 0.0,
                "total": 0.0,
            }
        return slice_acc[name]

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        slice_tags: list[str] = batch.get("slice_tags", [])
        if not slice_tags:
            continue

        encoded = encode_triplet_batch(model, batch, device, use_amp=use_amp, amp_dtype=amp_dtype)
        anchor_hidden = encoded["anchor_hidden"]
        anchor_mask = encoded["anchor_mask"]
        positive_hidden = encoded["positive_hidden"]
        positive_mask = encoded["positive_mask"]
        negative_hidden = encoded["negative_hidden"]
        negative_mask = encoded["negative_mask"]
        hard_neg_mask = encoded["hard_neg_mask"]
        triplet_idx = encoded["triplet_indices"]
        pair_idx = encoded["pair_indices"]
        routed_modes = resolve_mode_routing_for_batch(slice_tags, mode_routing_config)
        triplet_batch_indices = triplet_idx.detach().cpu().tolist() if triplet_idx.numel() > 0 else []
        negative_modes = [routed_modes["negative"][i] for i in triplet_batch_indices]

        with amp_context:
            anchor_emb = forward_embedding_head_batch(
                embedding_head,
                anchor_hidden,
                anchor_mask,
                sample_modes=routed_modes["anchor"],
                return_aux=False,
            )
            positive_emb = forward_embedding_head_batch(
                embedding_head,
                positive_hidden,
                positive_mask,
                sample_modes=routed_modes["positive"],
                return_aux=False,
            )

            negative_emb = None
            if triplet_idx.numel() > 0 and negative_hidden is not None:
                negative_emb = forward_embedding_head_batch(
                    embedding_head,
                    negative_hidden,
                    negative_mask,
                    sample_modes=negative_modes,
                    return_aux=False,
                )

        # Build per-sample slice mapping
        triplet_set = set(triplet_idx.cpu().tolist()) if triplet_idx.numel() > 0 else set()
        pair_set = set(pair_idx.cpu().tolist()) if pair_idx.numel() > 0 else set()
        triplet_list = triplet_idx.cpu().tolist() if triplet_idx.numel() > 0 else []

        # Group batch indices by slice
        slice_indices: dict[str, list[int]] = {}
        for i, tag in enumerate(slice_tags):
            slice_indices.setdefault(tag, []).append(i)

        for slice_name, indices in slice_indices.items():
            acc = _ensure_slice(slice_name)

            # Separate triplet vs pair indices for this slice
            s_triplet = [i for i in indices if i in triplet_set]
            s_pair = [i for i in indices if i in pair_set]

            # --- Triplet metrics ---
            if s_triplet and negative_emb is not None:
                # Map batch-level triplet indices to negative tensor indices
                neg_indices = [triplet_list.index(i) for i in s_triplet]
                trip_t = torch.tensor(s_triplet, dtype=torch.long, device=device)
                neg_t = torch.tensor(neg_indices, dtype=torch.long, device=device)

                a_trip = anchor_emb[trip_t]
                p_trip = positive_emb[trip_t]
                n_trip = negative_emb[neg_t]

                pos_sim = F.cosine_similarity(a_trip, p_trip)
                neg_sim = F.cosine_similarity(a_trip, n_trip)
                correct = (pos_sim > neg_sim).float()

                acc["pos_sim_sum"] += pos_sim.sum().item()
                acc["neg_sim_sum"] += neg_sim.sum().item()
                acc["correct"] += correct.sum().item()
                acc["triplet_count"] += len(s_triplet)
                acc["total"] += len(s_triplet)

                # Per-slice loss
                with amp_context:
                    hn_slice = hard_neg_mask[neg_t]
                    s_loss = loss_fn(
                        anchor=a_trip, positive=p_trip,
                        negatives=n_trip.unsqueeze(1),
                        hard_negative_mask=hn_slice.unsqueeze(1),
                    ).item()
                acc["loss_sum"] += s_loss * len(s_triplet)

                # Hard negative accuracy
                if hn_slice.any():
                    acc["hard_neg_correct"] += correct[hn_slice].sum().item()
                    acc["hard_neg_count"] += hn_slice.sum().item()

            # --- Pair metrics (query_doc style: in-batch accuracy) ---
            if s_pair:
                pair_t = torch.tensor(s_pair, dtype=torch.long, device=device)
                a_pair = anchor_emb[pair_t]
                p_pair = positive_emb[pair_t]

                acc["pos_sim_sum"] += F.cosine_similarity(a_pair, p_pair).sum().item()
                acc["total"] += len(s_pair)
                acc["pair_count"] += len(s_pair)

                # In-batch pair loss
                if len(s_pair) >= 2:
                    with amp_context:
                        p_loss = loss_fn(anchor=a_pair, positive=p_pair, negatives=None).item()
                    acc["loss_sum"] += p_loss * len(s_pair)

                    # Pair accuracy: for each anchor, correct positive has highest sim
                    sim_matrix = a_pair @ p_pair.T  # [n_pair, n_pair]
                    targets = torch.arange(len(s_pair), device=device)
                    predicted = sim_matrix.argmax(dim=1)
                    acc["pair_correct"] += (predicted == targets).float().sum().item()

    model.train()

    # Build final per-slice metrics
    results: dict[str, dict[str, float]] = {}

    # Aggregate accumulators
    agg = {
        "loss_sum": 0.0, "pos_sim_sum": 0.0, "neg_sim_sum": 0.0,
        "correct": 0.0, "hard_neg_correct": 0.0, "hard_neg_count": 0.0,
        "triplet_count": 0.0, "pair_count": 0.0, "pair_correct": 0.0, "total": 0.0,
    }

    for slice_name, acc in sorted(slice_acc.items()):
        total = acc["total"]
        if total == 0:
            continue

        # Accumulate into aggregate
        for k in agg:
            agg[k] += acc[k]

        trip_n = acc["triplet_count"]
        pair_n = acc["pair_count"]

        if trip_n > 0:
            # Triplet slice
            avg_pos = acc["pos_sim_sum"] / total
            avg_neg = acc["neg_sim_sum"] / trip_n
            results[slice_name] = {
                "val_loss": acc["loss_sum"] / total,
                "pos_sim": avg_pos,
                "neg_sim": avg_neg,
                "margin": avg_pos - avg_neg,
                "accuracy": acc["correct"] / trip_n,
                "hard_neg_accuracy": (
                    acc["hard_neg_correct"] / acc["hard_neg_count"]
                    if acc["hard_neg_count"] > 0 else 0.0
                ),
                "sample_count": int(total),
            }
        else:
            # Pair-only slice (query_doc)
            results[slice_name] = {
                "val_loss": acc["loss_sum"] / total if total > 0 else 0.0,
                "pair_accuracy": acc["pair_correct"] / pair_n if pair_n > 0 else 0.0,
                "pos_sim": acc["pos_sim_sum"] / total,
                "sample_count": int(total),
            }

    # Aggregate row
    agg_total = agg["total"]
    if agg_total > 0:
        agg_trip = agg["triplet_count"]
        agg_pos_sim = agg["pos_sim_sum"] / agg_total
        agg_neg_sim = agg["neg_sim_sum"] / agg_trip if agg_trip > 0 else 0.0
        results["_aggregate"] = {
            "val_loss": agg["loss_sum"] / agg_total,
            "pos_sim": agg_pos_sim,
            "neg_sim": agg_neg_sim,
            "margin": agg_pos_sim - agg_neg_sim,
            "accuracy": agg["correct"] / agg_trip if agg_trip > 0 else 0.0,
            "hard_neg_accuracy": (
                agg["hard_neg_correct"] / agg["hard_neg_count"]
                if agg["hard_neg_count"] > 0 else 0.0
            ),
            "sample_count": int(agg_total),
        }

    return results


@torch.no_grad()
def evaluate_all_heads_by_slice(
    model: ModernBertMultiTaskModel,
    candidate_heads: nn.ModuleDict,
    loss_modules: nn.ModuleDict,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-slice evaluation for all candidate heads.

    Returns:
        Dict mapping head_type -> slice_name -> metrics dict.
        Each head gets the same structure as evaluate_by_slice().
    """
    model.eval()
    candidate_heads.eval()

    results: dict[str, dict[str, dict[str, float]]] = {}
    for head_type, head in candidate_heads.items():
        results[head_type] = evaluate_by_slice(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_modules[head_type],
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=max_batches,
            head=head,
        )

    model.train()
    candidate_heads.train()
    return results


def log_slice_eval_table(
    slice_metrics: dict[str, dict[str, float]],
    title: str,
) -> None:
    """Log a formatted per-slice evaluation table.

    Prints the leaderboard-style table from the Epic 3.1 spec with
    Margin, Acc, HardNeg, Loss, and N columns. Pair-only slices show
    ``--`` for margin and hard_neg_accuracy.
    """
    log_section(title)
    header = f"{'Slice':<22}{'Margin':<10}{'Acc':<10}{'HardNeg':<10}{'Loss':<10}{'N':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    # Sort: named slices first (alphabetical), then _aggregate last
    ordered = sorted(
        (k for k in slice_metrics if not k.startswith("_")),
    )
    if "_aggregate" in slice_metrics:
        ordered.append("_aggregate")

    for slice_name in ordered:
        m = slice_metrics[slice_name]
        n = int(m.get("sample_count", 0))
        loss_str = f"{m.get('val_loss', 0.0):.4f}"

        if "margin" in m:
            # Triplet slice or aggregate
            margin_str = f"{m['margin']:.4f}"
            acc_str = f"{m['accuracy']:.4f}"
            hn_str = f"{m.get('hard_neg_accuracy', 0.0):.4f}"
        else:
            # Pair-only slice (query_doc)
            margin_str = "--"
            acc_str = f"{m.get('pair_accuracy', 0.0):.4f}"
            hn_str = "--"

        logger.info(
            f"{slice_name:<22}{margin_str:<10}{acc_str:<10}"
            f"{hn_str:<10}{loss_str:<10}{n:>8,}"
        )


# =============================================================================
# Retrieval Eval for Query-Doc (Epic 3.2)
# =============================================================================


def _embed_texts(
    texts: list[str],
    model: ModernBertMultiTaskModel,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    head: nn.Module | None = None,
    mode: str = "document",
) -> torch.Tensor:
    """Encode a list of texts into embeddings using the embedding head."""
    embedding_head = head if head is not None else model.heads["embedding"]
    amp_context = (
        autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device.type == "cuda"
        else autocast("cpu", enabled=False)
    )
    all_embs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with amp_context:
            hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state
            embs = forward_embedding_head(
                embedding_head,
                hidden,
                attention_mask,
                mode=mode,
                return_aux=False,
            )
        all_embs.append(embs.float())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def evaluate_retrieval(
    model: ModernBertMultiTaskModel,
    query_doc_eval_samples: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 256,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    head: nn.Module | None = None,
    query_mode: str = "query",
    document_mode: str = "document",
) -> dict[str, float]:
    """Compute retrieval metrics (Recall@k, MRR) on query-doc eval pairs.

    Args:
        model: The multi-task model.
        query_doc_eval_samples: List of normalized dicts with 'anchor' (query)
            and 'positive' (document) fields.
        tokenizer: Tokenizer for encoding texts.
        device: Torch device.
        max_length: Max token length for encoding.
        batch_size: Encoding batch size.
        use_amp: Use automatic mixed precision.
        amp_dtype: AMP dtype.
        head: Override embedding head (for multi-head bakeoff).

    Returns:
        {"recall@1": ..., "recall@5": ..., "recall@10": ..., "mrr": ..., "n_pairs": ...}
    """
    if not query_doc_eval_samples:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_pairs": 0}

    model.eval()
    queries = [s["anchor"] for s in query_doc_eval_samples]
    documents = [s["positive"] for s in query_doc_eval_samples]

    # Embed queries and documents
    q_embs = _embed_texts(
        queries, model, tokenizer, device, max_length, batch_size,
        use_amp=use_amp, amp_dtype=amp_dtype, head=head, mode=query_mode,
    )
    d_embs = _embed_texts(
        documents, model, tokenizer, device, max_length, batch_size,
        use_amp=use_amp, amp_dtype=amp_dtype, head=head, mode=document_mode,
    )

    # Normalize for cosine similarity
    q_embs = F.normalize(q_embs, p=2, dim=-1)
    d_embs = F.normalize(d_embs, p=2, dim=-1)

    # Similarity matrix [N_q, N_d] — each query's correct doc is at same index
    sim_matrix = q_embs @ d_embs.T  # [N, N]
    n = sim_matrix.size(0)

    # Rank of correct document for each query
    # For query i, correct doc is document i
    targets = torch.arange(n, device=device)
    # Sort similarities descending per query row
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)  # [N, N]
    # Find rank of target (0-indexed)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].float()
    # Convert to 1-indexed
    ranks = ranks + 1.0

    recall_at_1 = (ranks <= 1).float().mean().item()
    recall_at_5 = (ranks <= 5).float().mean().item()
    recall_at_10 = (ranks <= 10).float().mean().item()
    mrr = (1.0 / ranks).mean().item()

    metrics = {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr": mrr,
        "n_pairs": n,
    }
    logger.info(
        f"  Retrieval eval: R@1={recall_at_1:.4f} R@5={recall_at_5:.4f} "
        f"R@10={recall_at_10:.4f} MRR={mrr:.4f} (n={n})"
    )
    model.train()
    return metrics


# Default composite weights for best-model selection (Epic 3.3)
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "aggregate_margin": 0.35,
    "hard_neg_accuracy": 0.15,
    "wrong_person_accuracy": 0.15,
    "wrong_time_accuracy": 0.10,
    "safety_emotion_accuracy": 0.10,
    "query_doc_recall_at_5": 0.15,
}


def compute_composite_score(
    eval_metrics: dict[str, Any],
    slice_eval: dict[str, dict[str, float]] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite score for best-model selection.

    Falls back to aggregate margin when slice or retrieval data is missing.

    Args:
        eval_metrics: Overall eval metrics dict (must have 'margin',
            'hard_neg_accuracy').
        slice_eval: Per-slice metrics from evaluate_by_slice().
        retrieval_metrics: Retrieval metrics from evaluate_retrieval().
        weights: Configurable weight dict. Uses DEFAULT_COMPOSITE_WEIGHTS
            if not provided.

    Returns:
        Weighted composite score (higher is better).
    """
    w = weights or DEFAULT_COMPOSITE_WEIGHTS

    aggregate_margin = eval_metrics.get("margin", 0.0)
    hard_neg_acc = eval_metrics.get("hard_neg_accuracy", 0.0)

    # Per-slice accuracies from slice eval
    wrong_person_acc = 0.0
    wrong_time_acc = 0.0
    safety_emotion_acc = 0.0
    if slice_eval:
        wp = slice_eval.get("wrong_person", {})
        wrong_person_acc = wp.get("accuracy", 0.0)
        wt = slice_eval.get("wrong_time", {})
        wrong_time_acc = wt.get("accuracy", 0.0)
        se = slice_eval.get("safety_emotion", {})
        safety_emotion_acc = se.get("accuracy", 0.0)

    # Retrieval recall@5
    recall_at_5 = 0.0
    if retrieval_metrics:
        recall_at_5 = retrieval_metrics.get("recall@5", 0.0)

    score = (
        w.get("aggregate_margin", 0.35) * aggregate_margin
        + w.get("hard_neg_accuracy", 0.15) * hard_neg_acc
        + w.get("wrong_person_accuracy", 0.15) * wrong_person_acc
        + w.get("wrong_time_accuracy", 0.10) * wrong_time_acc
        + w.get("safety_emotion_accuracy", 0.10) * safety_emotion_acc
        + w.get("query_doc_recall_at_5", 0.15) * recall_at_5
    )
    return score


DEFAULT_SELECTION_METRIC = "composite"
SAFETY_LABEL_SCORES: dict[str, int] = {
    "GREEN": 0,
    "AMBER": 1,
    "RED": 2,
    "CRISIS": 3,
}


def _normalize_metric_name(metric_name: str) -> str:
    """Normalize metric names so config aliases resolve cleanly."""
    return (
        str(metric_name)
        .strip()
        .lower()
        .replace("@", "_at_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def build_selection_metric_map(
    eval_metrics: dict[str, Any],
    slice_eval: dict[str, dict[str, float]] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
    composite_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build a flat metric map for configurable checkpoint selection."""
    composite_score = compute_composite_score(
        eval_metrics,
        slice_eval,
        retrieval_metrics,
        composite_weights,
    )

    wrong_person_acc = float((slice_eval or {}).get("wrong_person", {}).get("accuracy", 0.0))
    wrong_time_acc = float((slice_eval or {}).get("wrong_time", {}).get("accuracy", 0.0))
    safety_emotion_acc = float((slice_eval or {}).get("safety_emotion", {}).get("accuracy", 0.0))
    recall_at_1 = float((retrieval_metrics or {}).get("recall@1", 0.0))
    recall_at_5 = float((retrieval_metrics or {}).get("recall@5", 0.0))
    recall_at_10 = float((retrieval_metrics or {}).get("recall@10", 0.0))
    mrr = float((retrieval_metrics or {}).get("mrr", 0.0))

    return {
        "composite": composite_score,
        "aggregate_margin": float(eval_metrics.get("margin", 0.0)),
        "margin": float(eval_metrics.get("margin", 0.0)),
        "hard_neg_accuracy": float(eval_metrics.get("hard_neg_accuracy", 0.0)),
        "wrong_person_accuracy": wrong_person_acc,
        "wrong_time_accuracy": wrong_time_acc,
        "safety_emotion_accuracy": safety_emotion_acc,
        "query_doc_recall_at_1": recall_at_1,
        "query_doc_recall_at_5": recall_at_5,
        "query_doc_recall_at_10": recall_at_10,
        "query_doc_mrr": mrr,
        "retrieval_recall_at_1": recall_at_1,
        "retrieval_recall_at_5": recall_at_5,
        "retrieval_recall_at_10": recall_at_10,
        "retrieval_mrr": mrr,
    }


def resolve_selection_score(
    selection_metric: str | None,
    eval_metrics: dict[str, Any],
    slice_eval: dict[str, dict[str, float]] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
    composite_weights: dict[str, float] | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Resolve the configured checkpoint-selection metric into a concrete score."""
    metric_map = build_selection_metric_map(
        eval_metrics,
        slice_eval,
        retrieval_metrics,
        composite_weights,
    )
    requested_metric = selection_metric or DEFAULT_SELECTION_METRIC
    normalized_target = _normalize_metric_name(requested_metric)

    for metric_name, metric_value in metric_map.items():
        if _normalize_metric_name(metric_name) == normalized_target:
            return metric_name, metric_value, metric_map

    logger.warning(
        f"Unknown evaluation.selection_metric='{requested_metric}'; falling back to '{DEFAULT_SELECTION_METRIC}'"
    )
    return DEFAULT_SELECTION_METRIC, metric_map[DEFAULT_SELECTION_METRIC], metric_map


def _coerce_safety_label_score(label: Any) -> int | None:
    """Map ordinal safety labels onto an integer scale for margin shaping."""
    if label is None:
        return None
    normalized = str(label).strip().upper()
    if not normalized:
        return None
    return SAFETY_LABEL_SCORES.get(normalized)


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
    query_doc_eval_samples: list[dict[str, Any]] | None = None,
    composite_weights: dict[str, float] | None = None,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    train_sampler: SliceBalancedSampler | None = None,
    mode_routing_config: dict[str, Any] | None = None,
    aux_objectives_config: dict[str, Any] | None = None,
    retrieval_modes: dict[str, str] | None = None,
    teacher_cache: TeacherEmbeddingCache | None = None,
    distillation_config: dict[str, Any] | None = None,
    teacher_projection: nn.Module | None = None,
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
    best_score = float("-inf")
    no_improve_count = 0
    history: dict[str, list] = {
        "train_loss": [], "train_pos_sim": [], "train_neg_sim": [],
        "train_margin": [], "eval_metrics": [],
    }

    log_section(f"TRAINING: {head_type}")
    logger.info(f"  Epochs: {num_epochs} | Batches: {len(train_loader)} | Grad accum: {gradient_accumulation_steps}")
    logger.info(f"  Selection metric: {selection_metric}")

    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
        epoch_teacher_loss = 0.0
        epoch_distillation_aux_loss = 0.0
        epoch_teacher_anchor_found = 0.0
        epoch_teacher_positive_found = 0.0
        epoch_teacher_negative_found = 0.0
        epoch_steps = 0
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            step_debug = debug and (step < 5 or step % 100 == 0)
            losses = train_step(
                model, batch, loss_fn, device, debug=step_debug,
                use_amp=use_amp, amp_dtype=amp_dtype, matryoshka_dims=matryoshka_dims,
                mode_routing_config=mode_routing_config,
                aux_objectives_config=aux_objectives_config,
                teacher_cache=teacher_cache,
                distillation_config=distillation_config,
                teacher_projection=teacher_projection,
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
                    eval_metrics = evaluate(
                        model,
                        val_loader,
                        loss_fn,
                        device,
                        debug=debug,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        max_batches=max_eval_batches,
                        mode_routing_config=mode_routing_config,
                    )
                    current_margin = eval_metrics["margin"]
                    history["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, **eval_metrics})

                    # Per-slice eval table
                    slice_eval = evaluate_by_slice(
                        model, val_loader, loss_fn, device,
                        use_amp=use_amp, amp_dtype=amp_dtype,
                        max_batches=max_eval_batches,
                        mode_routing_config=mode_routing_config,
                    )
                    log_slice_eval_table(slice_eval, f"SLICE EVAL @ STEP {global_step} ({head_type})")

                    # Retrieval eval on query_doc
                    retrieval_result = None
                    if query_doc_eval_samples:
                        retrieval_result = evaluate_retrieval(
                            model, query_doc_eval_samples, tokenizer, device,
                            use_amp=use_amp, amp_dtype=amp_dtype,
                            query_mode=(retrieval_modes or {}).get("query", "query"),
                            document_mode=(retrieval_modes or {}).get("document", "document"),
                        )
                        eval_metrics.update({f"retrieval_{k}": v for k, v in retrieval_result.items()})

                    resolved_selection_metric, selection_score, metric_map = resolve_selection_score(
                        selection_metric,
                        eval_metrics,
                        slice_eval,
                        retrieval_result,
                        composite_weights,
                    )
                    eval_metrics["composite_score"] = metric_map["composite"]
                    eval_metrics["selection_metric"] = resolved_selection_metric
                    eval_metrics["selection_score"] = selection_score
                    logger.info(
                        f"  Selection metric ({resolved_selection_metric}): {selection_score:.4f} | "
                        f"Composite score: {metric_map['composite']:.4f}"
                    )

                    if selection_score > best_score:
                        best_score = selection_score
                        no_improve_count = 0
                        logger.info(f"New best {resolved_selection_metric}={best_score:.4f}! Saving...")
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
            epoch_teacher_loss += losses.get("teacher_loss", torch.tensor(0.0, device=device)).item()
            epoch_distillation_aux_loss += losses.get("distillation_aux_loss", torch.tensor(0.0, device=device)).item()
            epoch_teacher_anchor_found += losses.get("distill_teacher_anchor_found_rate", torch.tensor(0.0, device=device)).item()
            epoch_teacher_positive_found += losses.get("distill_teacher_positive_found_rate", torch.tensor(0.0, device=device)).item()
            epoch_teacher_negative_found += losses.get("distill_teacher_negative_found_rate", torch.tensor(0.0, device=device)).item()
            epoch_steps += 1

            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{losses['total_loss'].item():.4f}", margin=f"{losses['margin']:.3f}", lr=f"{lr:.2e}")

            if global_step > 0 and global_step % logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_pos = epoch_pos_sim / epoch_steps
                avg_neg = epoch_neg_sim / epoch_steps
                avg_teacher_loss = epoch_teacher_loss / epoch_steps
                avg_distillation_aux_loss = epoch_distillation_aux_loss / epoch_steps
                avg_teacher_anchor_found = epoch_teacher_anchor_found / epoch_steps
                avg_teacher_positive_found = epoch_teacher_positive_found / epoch_steps
                avg_teacher_negative_found = epoch_teacher_negative_found / epoch_steps
                log_message = (
                    f"  Step {global_step}: loss={avg_loss:.4f} pos_sim={avg_pos:.4f} "
                    f"neg_sim={avg_neg:.4f} margin={avg_pos - avg_neg:.4f} lr={lr:.2e}"
                )
                if teacher_cache is not None and distillation_config and distillation_config.get("enabled", False):
                    log_message += (
                        f" teacher_loss={avg_teacher_loss:.4f}"
                        f" distill_aux={avg_distillation_aux_loss:.4f}"
                        f" teacher_found(a/p/n)={avg_teacher_anchor_found:.2%}/{avg_teacher_positive_found:.2%}/{avg_teacher_negative_found:.2%}"
                    )
                logger.info(log_message)
                # Per-slice diagnostics
                slice_metrics = compute_per_slice_metrics(
                    batch, model, loss_fn, device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                    mode_routing_config=mode_routing_config,
                )
                if slice_metrics:
                    logger.info(f"    Per-slice loss: {format_per_slice_log(slice_metrics)}")
                model.train()

        # Epoch summary
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            avg_pos = epoch_pos_sim / epoch_steps
            avg_neg = epoch_neg_sim / epoch_steps
            avg_teacher_loss = epoch_teacher_loss / epoch_steps
            avg_distillation_aux_loss = epoch_distillation_aux_loss / epoch_steps
            avg_teacher_anchor_found = epoch_teacher_anchor_found / epoch_steps
            avg_teacher_positive_found = epoch_teacher_positive_found / epoch_steps
            avg_teacher_negative_found = epoch_teacher_negative_found / epoch_steps
            history["train_loss"].append(avg_loss)
            history["train_pos_sim"].append(avg_pos)
            history["train_neg_sim"].append(avg_neg)
            history["train_margin"].append(avg_pos - avg_neg)
            epoch_summary = f"Epoch {epoch+1} summary: loss={avg_loss:.4f} margin={avg_pos - avg_neg:.4f}"
            if teacher_cache is not None and distillation_config and distillation_config.get("enabled", False):
                epoch_summary += (
                    f" teacher_loss={avg_teacher_loss:.4f}"
                    f" distill_aux={avg_distillation_aux_loss:.4f}"
                    f" teacher_found(a/p/n)={avg_teacher_anchor_found:.2%}/{avg_teacher_positive_found:.2%}/{avg_teacher_negative_found:.2%}"
                )
            logger.info(epoch_summary)

        # Full eval at epoch end
        if val_loader is not None:
            logger.info(f"--- Full eval epoch {epoch + 1} ---")
            eval_metrics = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                debug=debug,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=None,
                mode_routing_config=mode_routing_config,
            )
            history["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, "full_eval": True, **eval_metrics})

            # Per-slice eval table (full)
            slice_eval = evaluate_by_slice(
                model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                max_batches=None,
                mode_routing_config=mode_routing_config,
            )
            log_slice_eval_table(slice_eval, f"SLICE EVAL EPOCH {epoch + 1} ({head_type})")

            # Retrieval eval on query_doc (full)
            retrieval_result = None
            if query_doc_eval_samples:
                retrieval_result = evaluate_retrieval(
                    model, query_doc_eval_samples, tokenizer, device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                    query_mode=(retrieval_modes or {}).get("query", "query"),
                    document_mode=(retrieval_modes or {}).get("document", "document"),
                )
                eval_metrics.update({f"retrieval_{k}": v for k, v in retrieval_result.items()})

            resolved_selection_metric, selection_score, metric_map = resolve_selection_score(
                selection_metric,
                eval_metrics,
                slice_eval,
                retrieval_result,
                composite_weights,
            )
            eval_metrics["composite_score"] = metric_map["composite"]
            eval_metrics["selection_metric"] = resolved_selection_metric
            eval_metrics["selection_score"] = selection_score
            logger.info(
                f"  Selection metric ({resolved_selection_metric}): {selection_score:.4f} | "
                f"Composite score: {metric_map['composite']:.4f}"
            )

            current_margin = eval_metrics["margin"]
            if selection_score > best_score:
                best_score = selection_score
                no_improve_count = 0
                logger.info(f"New best {resolved_selection_metric}={best_score:.4f}! Saving...")
                save_bakeoff_checkpoint(model, output_dir / "best", tokenizer, head_type, head_params, optimizer, scheduler)
                if ema_model is not None:
                    save_bakeoff_checkpoint(ema_model.module, output_dir / "best-ema", tokenizer, head_type, head_params)
            else:
                no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered")
                break

    log_section("TRAINING COMPLETE")
    logger.info(f"  Best {selection_metric}: {best_score:.4f}")
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
    bakeoff_config = config.get("bakeoff", {})
    bakeoff_data_profile = bakeoff_config.get("data_profile")
    resolved_data_config = resolve_data_config(data_config, bakeoff_data_profile)

    checkpoint_path = encoder_config.get("checkpoint", "checkpoints/checkpoint-8000")
    data_root = Path(resolved_data_config.get("root", "data"))
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

    eval_config = config.get("evaluation", {})
    composite_weights = eval_config.get("composite_weights", None)
    selection_metric = eval_config.get("selection_metric", DEFAULT_SELECTION_METRIC)

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

    tokenizer = load_checkpoint_tokenizer(checkpoint_path)

    log_section("DATA")
    if bakeoff_data_profile:
        logger.info(f"  Bake-off data profile: {bakeoff_data_profile}")
    train_dataset, val_dataset, query_doc_eval_samples = build_train_val_datasets(
        data_config=resolved_data_config,
        data_root=data_root,
        max_samples=max_samples,
        seed=seed,
    )
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset)} samples "
                f"(train={len(train_dataset)}, val={len(val_dataset)})")
    if query_doc_eval_samples:
        logger.info(f"  Query-doc eval pairs: {len(query_doc_eval_samples)}")

    collator = EmbeddingCollator(tokenizer=tokenizer, max_length=max_length)
    effective_workers = 0 if platform.system() == "Windows" else num_workers
    drop_last_train = len(train_dataset) >= batch_size
    if not drop_last_train:
        logger.info(f"  Train dataset smaller than batch_size ({len(train_dataset)} < {batch_size}); using drop_last=False")

    # Slice-balanced sampling
    sampling_config = resolved_data_config.get("sampling", {})
    slice_weights = sampling_config.get("slice_weights", {})
    if slice_weights:
        train_sampler = SliceBalancedSampler(
            dataset=train_dataset,
            slice_weights=slice_weights,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=effective_workers > 0,
            drop_last=drop_last_train,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=effective_workers > 0,
            drop_last=drop_last_train,
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
    best_score = {head_type: -1.0 for head_type, _ in head_experiments}
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

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            triplet_idx = encoded["triplet_indices"]
            pair_idx = encoded["pair_indices"]
            has_triplets = triplet_idx.numel() > 0 and negative_hidden is not None

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

                    loss = torch.tensor(0.0, device=device)
                    loss_count = 0

                    # Triplet sub-batch
                    if has_triplets:
                        negative_emb = head(negative_hidden, negative_mask)

                        if config.get("training", {}).get("matryoshka", {}).get("enabled", False):
                            dims = config["training"]["matryoshka"].get("dims", [hidden_size])
                            trip_loss = torch.tensor(0.0, device=device)
                            for dim in dims:
                                a_d = F.normalize(anchor_emb[triplet_idx, :dim], p=2, dim=-1)
                                p_d = F.normalize(positive_emb[triplet_idx, :dim], p=2, dim=-1)
                                n_d = F.normalize(negative_emb[:, :dim], p=2, dim=-1).unsqueeze(1)
                                hn_mask = hard_neg_mask.unsqueeze(1)
                                trip_loss = trip_loss + loss_modules[head_type](
                                    anchor=a_d, positive=p_d,
                                    negatives=n_d, hard_negative_mask=hn_mask,
                                )
                            loss = loss + trip_loss / len(dims)
                        else:
                            negatives = negative_emb.unsqueeze(1)
                            hn_mask = hard_neg_mask.unsqueeze(1)
                            loss = loss + loss_modules[head_type](
                                anchor=anchor_emb[triplet_idx], positive=positive_emb[triplet_idx],
                                negatives=negatives, hard_negative_mask=hn_mask,
                            )
                        loss_count += 1

                    # Pair sub-batch
                    if pair_idx.numel() > 0:
                        a_pair = anchor_emb[pair_idx]
                        p_pair = positive_emb[pair_idx]
                        if config.get("training", {}).get("matryoshka", {}).get("enabled", False):
                            dims = config["training"]["matryoshka"].get("dims", [hidden_size])
                            pair_loss = torch.tensor(0.0, device=device)
                            for dim in dims:
                                a_d = F.normalize(a_pair[:, :dim], p=2, dim=-1)
                                p_d = F.normalize(p_pair[:, :dim], p=2, dim=-1)
                                pair_loss = pair_loss + loss_modules[head_type](
                                    anchor=a_d, positive=p_d, negatives=None,
                                )
                            loss = loss + pair_loss / len(dims)
                        else:
                            loss = loss + loss_modules[head_type](
                                anchor=a_pair, positive=p_pair, negatives=None,
                            )
                        loss_count += 1

                    if loss_count > 1:
                        loss = loss / loss_count

                    pos_sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
                    if has_triplets:
                        neg_sim = F.cosine_similarity(anchor_emb[triplet_idx], negative_emb).mean().item()
                    else:
                        neg_sim = 0.0
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
                # Per-slice diagnostics (use first active head)
                first_head_type = next(iter(candidate_heads))
                slice_metrics = compute_per_slice_metrics(
                    batch, base_model, loss_modules[first_head_type], device,
                    use_amp=use_bf16, amp_dtype=amp_dtype,
                    head=candidate_heads[first_head_type],
                )
                if slice_metrics:
                    logger.info(f"    Per-slice loss: {format_per_slice_log(slice_metrics)}")
                base_model.train()
                candidate_heads.train()

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

                # Per-slice eval for all heads
                slice_eval_all = evaluate_all_heads_by_slice(
                    model=base_model,
                    candidate_heads=ema_heads if ema_heads is not None else candidate_heads,
                    loss_modules=loss_modules,
                    val_loader=val_loader,
                    device=device,
                    use_amp=use_bf16,
                    amp_dtype=amp_dtype,
                    max_batches=max_eval_batches,
                )
                for ht, slice_eval in slice_eval_all.items():
                    log_slice_eval_table(slice_eval, f"SLICE EVAL @ STEP {global_step} ({ht})")

                # Retrieval eval on query_doc per head
                if query_doc_eval_samples:
                    eval_heads = ema_heads if ema_heads is not None else candidate_heads
                    for ht in eval_heads:
                        retrieval_metrics = evaluate_retrieval(
                            base_model, query_doc_eval_samples, tokenizer, device,
                            use_amp=use_bf16, amp_dtype=amp_dtype, head=eval_heads[ht],
                        )
                        eval_metrics[ht].update({f"retrieval_{k}": v for k, v in retrieval_metrics.items()})

                for head_type, metrics in eval_metrics.items():
                    # Compute composite score per head
                    head_slice_eval = slice_eval_all.get(head_type)
                    head_retrieval = {k.replace("retrieval_", ""): v for k, v in metrics.items() if k.startswith("retrieval_")} or None
                    head_score = compute_composite_score(metrics, head_slice_eval, head_retrieval, composite_weights)
                    metrics["composite_score"] = head_score

                    history["heads"][head_type]["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, **metrics})
                    if head_score > best_score[head_type]:
                        best_score[head_type] = head_score
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

            # Per-slice eval for all heads (full)
            slice_eval_all = evaluate_all_heads_by_slice(
                model=base_model,
                candidate_heads=ema_heads if ema_heads is not None else candidate_heads,
                loss_modules=loss_modules,
                val_loader=val_loader,
                device=device,
                use_amp=use_bf16,
                amp_dtype=amp_dtype,
                max_batches=None,
            )
            for ht, slice_eval in slice_eval_all.items():
                log_slice_eval_table(slice_eval, f"SLICE EVAL EPOCH {epoch + 1} ({ht})")

            # Retrieval eval on query_doc per head (full)
            if query_doc_eval_samples:
                eval_heads = ema_heads if ema_heads is not None else candidate_heads
                for ht in eval_heads:
                    retrieval_metrics = evaluate_retrieval(
                        base_model, query_doc_eval_samples, tokenizer, device,
                        use_amp=use_bf16, amp_dtype=amp_dtype, head=eval_heads[ht],
                    )
                    eval_metrics[ht].update({f"retrieval_{k}": v for k, v in retrieval_metrics.items()})

            for head_type, metrics in eval_metrics.items():
                history["heads"][head_type]["eval_metrics"].append({"step": global_step, "epoch": epoch + 1, "full_eval": True, **metrics})
                # Compute composite score per head
                head_slice_eval = slice_eval_all.get(head_type)
                head_retrieval = {k.replace("retrieval_", ""): v for k, v in metrics.items() if k.startswith("retrieval_")} or None
                head_score = compute_composite_score(metrics, head_slice_eval, head_retrieval, composite_weights)
                metrics["composite_score"] = head_score

                if head_score > best_score[head_type]:
                    best_score[head_type] = head_score
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
        best_eval = max(evals, key=lambda item: item.get("composite_score", item.get("margin", -1.0))) if evals else {}
        summary[head_type] = {
            "best_composite_score": best_eval.get("composite_score", 0.0),
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
    bakeoff_config = config.get("bakeoff", {})
    bakeoff_data_profile = bakeoff_config.get("data_profile")
    resolved_data_config = resolve_data_config(data_config, bakeoff_data_profile)

    checkpoint_path = encoder_config.get("checkpoint", "checkpoints/checkpoint-8000")
    data_root = Path(resolved_data_config.get("root", "data"))

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

    eval_config = config.get("evaluation", {})
    composite_weights = eval_config.get("composite_weights", None)

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

    tokenizer = load_checkpoint_tokenizer(checkpoint_path)

    # Data
    log_section("DATA")
    if bakeoff_data_profile:
        logger.info(f"  Bake-off data profile: {bakeoff_data_profile}")
    train_dataset, val_dataset, query_doc_eval_samples = build_train_val_datasets(
        data_config=resolved_data_config,
        data_root=data_root,
        max_samples=max_samples,
        seed=seed,
    )
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset)} samples "
                f"(train={len(train_dataset)}, val={len(val_dataset)})")
    if query_doc_eval_samples:
        logger.info(f"  Query-doc eval pairs: {len(query_doc_eval_samples)}")

    collator = EmbeddingCollator(tokenizer=tokenizer, max_length=max_length)
    effective_workers = 0 if platform.system() == "Windows" else num_workers
    drop_last_train = len(train_dataset) >= batch_size
    if not drop_last_train:
        logger.info(f"  Train dataset smaller than batch_size ({len(train_dataset)} < {batch_size}); using drop_last=False")

    # Slice-balanced sampling
    sampling_config = resolved_data_config.get("sampling", {})
    slice_weights = sampling_config.get("slice_weights", {})
    if slice_weights:
        train_sampler = SliceBalancedSampler(
            dataset=train_dataset,
            slice_weights=slice_weights,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collator,
            num_workers=effective_workers, pin_memory=True,
            persistent_workers=effective_workers > 0, drop_last=drop_last_train,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
            num_workers=effective_workers, pin_memory=True,
            persistent_workers=effective_workers > 0, drop_last=drop_last_train,
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
        query_doc_eval_samples=query_doc_eval_samples,
        composite_weights=composite_weights,
        selection_metric=selection_metric,
        train_sampler=train_sampler,
        mode_routing_config=None,
        aux_objectives_config=None,
        retrieval_modes=None,
    )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Experiment {head_type} complete -> {output_dir}")
    return history


# =============================================================================
# Stage B: AgreementGatedHeadV2 specialization
# =============================================================================


def run_stage_b(
    config: dict[str, Any],
    bakeoff_checkpoint: str | Path,
    output_dir: Path,
    debug: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Stage B: specialize AgreementGatedHeadV2 on FamilyOS retrieval slices.

    Loads a Stage A bakeoff checkpoint, upgrades or preserves the embedding
    head as ``agreement_gated_v2``, and trains it on the dedicated Stage B
    profile with query/document asymmetry and auxiliary objectives.

    Encoder remains frozen. Only the embedding head is trained.

    Args:
        config: Full YAML config dict.
        bakeoff_checkpoint: Path to the bakeoff winner checkpoint dir
            (e.g. outputs/embedding-bakeoff/agreement_gated_v2/best).
        output_dir: Where to save Stage B outputs.
        debug: Enable debug mode (small dataset).
        max_samples: Limit total samples loaded.

    Returns:
        Training history dict.
    """
    log_section("STAGE B: DOMAIN ADAPTATION")
    logger.info(f"  Bakeoff checkpoint: {bakeoff_checkpoint}")
    logger.info(f"  Output: {output_dir}")

    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})
    stage_b_config = config.get("stage_b", {})
    training_config = copy.deepcopy(training_config)
    stage_b_training_overrides = stage_b_config.get("training_overrides", {})
    if stage_b_training_overrides:
        training_config = _deep_update_dict(training_config, stage_b_training_overrides)
        logger.info("Applying Stage B training overrides")

    evaluation_config = copy.deepcopy(config.get("evaluation", {}))
    stage_b_evaluation_overrides = stage_b_config.get("evaluation_overrides", {})
    if stage_b_evaluation_overrides:
        evaluation_config = _deep_update_dict(evaluation_config, stage_b_evaluation_overrides)
        logger.info("Applying Stage B evaluation overrides")

    stage_b_data_profile = stage_b_config.get("data_profile", "stage_b_v2")
    stage_b_data_config = resolve_data_config(data_config, stage_b_data_profile)
    stage_b_data_overrides = stage_b_config.get("data_overrides", {})
    if stage_b_data_overrides:
        stage_b_data_config = _deep_update_dict(stage_b_data_config, stage_b_data_overrides)
        logger.info("Applying Stage B data overrides")

    mode_routing_config = stage_b_config.get("mode_routing", {})
    aux_objectives_config = stage_b_config.get("aux_objectives", {})
    retrieval_modes = stage_b_config.get("retrieval_modes", {"query": "query", "document": "document"})

    data_root = Path(stage_b_data_config.get("root", "data"))

    # Training params
    learning_rate = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_epochs = training_config.get("num_epochs", 7)
    batch_size = training_config.get("batch_size", 128)
    max_length = data_config.get("max_length", 128)
    warmup_steps = training_config.get("warmup_steps", 200)
    save_steps = training_config.get("save_steps", 500)
    eval_steps = training_config.get("eval_steps", 500)
    logging_steps = training_config.get("logging_steps", 50)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    num_workers = data_config.get("num_workers", 0)
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

    composite_weights = evaluation_config.get("composite_weights", None)
    selection_metric = evaluation_config.get("selection_metric", DEFAULT_SELECTION_METRIC)

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

    # Load model with Stage B target head from the bakeoff checkpoint
    log_section("MODEL (Stage B)")
    model, head_type, head_params = load_model_for_stage_b_v2(
        config=config,
        bakeoff_checkpoint=bakeoff_checkpoint,
        use_flash_attention=use_flash_attention,
    )
    freeze_model_except_embedding_head(model)
    model = model.to(device)

    tokenizer = load_checkpoint_tokenizer(bakeoff_checkpoint)

    # Data -- dedicated Stage B profile
    log_section("DATA (Stage B profile)")
    train_dataset, val_dataset, query_doc_eval_samples = build_train_val_datasets(
        data_config=stage_b_data_config,
        data_root=data_root,
        max_samples=max_samples,
        seed=seed,
    )
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset)} samples "
                f"(train={len(train_dataset)}, val={len(val_dataset)})")
    if query_doc_eval_samples:
        logger.info(f"  Query-doc eval pairs: {len(query_doc_eval_samples)}")

    collator = EmbeddingCollator(tokenizer=tokenizer, max_length=max_length)
    effective_workers = 0 if platform.system() == "Windows" else num_workers
    drop_last_train = len(train_dataset) >= batch_size
    if not drop_last_train:
        logger.info(f"  Train dataset smaller than batch_size ({len(train_dataset)} < {batch_size}); using drop_last=False")

    # Slice-balanced sampling
    sampling_config = stage_b_data_config.get("sampling", {})
    slice_weights = sampling_config.get("slice_weights", {})
    if slice_weights:
        train_sampler = SliceBalancedSampler(
            dataset=train_dataset,
            slice_weights=slice_weights,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collator,
            num_workers=effective_workers, pin_memory=True,
            persistent_workers=effective_workers > 0, drop_last=drop_last_train,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
            num_workers=effective_workers, pin_memory=True,
            persistent_workers=effective_workers > 0, drop_last=drop_last_train,
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
    log_section("STAGE B TRAINING")
    logger.info(f"  Head: {head_type} (trained weights from Stage A)")
    logger.info(f"  Encoder: FROZEN")
    logger.info(f"  Stage B data profile: {stage_b_data_profile}")
    logger.info(f"  Data slices: {len(stage_b_data_config.get('sources', []))} sources")

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
        query_doc_eval_samples=query_doc_eval_samples,
        composite_weights=composite_weights,
        selection_metric=selection_metric,
        train_sampler=train_sampler,
        mode_routing_config=mode_routing_config,
        aux_objectives_config=aux_objectives_config,
        retrieval_modes=retrieval_modes,
    )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save Stage B metadata
    stage_b_meta = {
        "stage": "B",
        "source_checkpoint": str(bakeoff_checkpoint),
        "head_type": head_type,
        "head_params": head_params,
        "training_type": "stage_b_domain_adaptation",
        "selection_metric": selection_metric,
        "training_overrides": stage_b_training_overrides,
        "data_overrides": stage_b_data_overrides,
    }
    with open(output_dir / "stage_b_metadata.json", "w") as f:
        json.dump(stage_b_meta, f, indent=2)

    logger.info(f"Stage B complete -> {output_dir}")
    return history


def run_distillation(
    config: dict[str, Any],
    output_dir: Path,
    debug: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Milestone 3: distill a student embedding head from the teacher cache."""
    log_section("DISTILLATION")

    runtime_config = config.get("runtime", {})
    distillation_config = copy.deepcopy(config.get("distillation", {}))
    training_config = copy.deepcopy(config.get("training", {}))
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})
    stage_b_config = config.get("stage_b", {})
    evaluation_config = copy.deepcopy(config.get("evaluation", {}))

    training_overrides = distillation_config.get("training_overrides", {})
    if training_overrides:
        training_config = _deep_update_dict(training_config, training_overrides)
        logger.info("Applying distillation training overrides")

    evaluation_overrides = distillation_config.get("evaluation_overrides", {})
    if evaluation_overrides:
        evaluation_config = _deep_update_dict(evaluation_config, evaluation_overrides)
        logger.info("Applying distillation evaluation overrides")

    student_checkpoint = runtime_config.get("student_checkpoint")
    if not student_checkpoint:
        raise ValueError("runtime.student_checkpoint is required for distillation mode")

    train_sources_config = distillation_config.get("train_sources", {})
    use_stage_b_profile = train_sources_config.get("use_stage_b_profile", True)
    data_profile = stage_b_config.get("data_profile") if use_stage_b_profile else config.get("bakeoff", {}).get("data_profile")
    resolved_data_config = resolve_data_config(data_config, data_profile)
    data_overrides = distillation_config.get("data_overrides", {})
    if data_overrides:
        resolved_data_config = _deep_update_dict(resolved_data_config, data_overrides)
        logger.info("Applying distillation data overrides")
    data_root = resolve_workspace_path(resolved_data_config.get("root", "data"))

    seed = training_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = training_config.get("bf16", False)
    use_tf32 = training_config.get("tf32", False)
    use_flash_attention = training_config.get("flash_attention", False)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: {gpu_name}")
        if use_tf32 and "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    teacher_cache_dir = distillation_config.get("teacher_cache_dir")
    if not teacher_cache_dir:
        raise ValueError("distillation.teacher_cache_dir is required for distillation mode")
    teacher_cache = TeacherEmbeddingCache.load(teacher_cache_dir)

    checkpoint_head_type = stage_b_config.get("head_type", "agreement_gated_v2")
    checkpoint_head_params = get_head_params_from_config(config, checkpoint_head_type)
    student_embedding_metadata_path = resolve_workspace_path(student_checkpoint) / "embedding_metadata.json"
    if student_embedding_metadata_path.exists():
        with open(student_embedding_metadata_path, encoding="utf-8") as handle:
            student_embedding_metadata = json.load(handle)
        student_bakeoff_info = student_embedding_metadata.get("bakeoff", {})
        metadata_head_type = student_bakeoff_info.get("head_type")
        if metadata_head_type in EMBEDDING_HEAD_REGISTRY:
            checkpoint_head_type = metadata_head_type
        metadata_head_params = student_bakeoff_info.get("head_params") or {}
        if metadata_head_params:
            checkpoint_head_params = metadata_head_params

    log_section("MODEL (Distillation)")
    model = load_model_checkpoint(
        checkpoint_path=student_checkpoint,
        use_flash_attention=use_flash_attention,
    )
    freeze_model_except_embedding_head(model)
    model = model.to(device)

    teacher_projection: nn.Module | None = None
    projection_config = distillation_config.get("projection", {})
    if projection_config.get("enabled", False):
        teacher_projection = TeacherProjection(
            teacher_dim=int(projection_config.get("teacher_dim", teacher_cache.embedding_dim)),
            student_dim=int(projection_config.get("student_dim", model.heads["embedding"].output_dim if hasattr(model.heads["embedding"], "output_dim") and model.heads["embedding"].output_dim is not None else model.config.hidden_size)),
        ).to(device)
        logger.info(
            f"  Teacher projection enabled: {projection_config.get('teacher_dim', teacher_cache.embedding_dim)} -> "
            f"{projection_config.get('student_dim', model.config.hidden_size)}"
        )

    tokenizer = load_checkpoint_tokenizer(student_checkpoint)

    log_section("DATA (Distillation)")
    train_dataset, val_dataset, query_doc_eval_samples = build_train_val_datasets(
        data_config=resolved_data_config,
        data_root=data_root,
        max_samples=max_samples,
        seed=seed,
    )
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset)} samples (train={len(train_dataset)}, val={len(val_dataset)})")
    if query_doc_eval_samples:
        logger.info(f"  Query-doc eval pairs: {len(query_doc_eval_samples)}")

    collator = EmbeddingCollator(tokenizer=tokenizer, max_length=resolved_data_config.get("max_length", 128))
    effective_workers = 0 if platform.system() == "Windows" else resolved_data_config.get("num_workers", 8)
    batch_size = training_config.get("batch_size", 128)
    drop_last_train = len(train_dataset) >= batch_size

    sampling_config = resolved_data_config.get("sampling", {})
    slice_weights = sampling_config.get("slice_weights", {})
    if slice_weights:
        train_sampler = SliceBalancedSampler(dataset=train_dataset, slice_weights=slice_weights, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=effective_workers > 0,
            drop_last=drop_last_train,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=effective_workers > 0,
            drop_last=drop_last_train,
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

    loss_fn = FamilyContrastiveLoss(
        temperature=loss_config.get("temperature", 0.07),
        hard_negative_weight=loss_config.get("hard_negative_weight", 1.5),
        use_hard_negatives=True,
        normalize=False,
    )
    if loss_config.get("learnable_temperature", False):
        loss_fn.log_temperature.requires_grad_(True)

    trainable_params = get_trainable_params(model)
    param_groups = [{
        "params": trainable_params,
        "lr": training_config.get("learning_rate", 2e-4),
        "weight_decay": training_config.get("weight_decay", 0.01),
    }]
    if teacher_projection is not None:
        param_groups.append({
            "params": [param for param in teacher_projection.parameters() if param.requires_grad],
            "lr": training_config.get("learning_rate", 2e-4),
            "weight_decay": training_config.get("weight_decay", 0.01),
        })
    if loss_config.get("learnable_temperature", False):
        param_groups.append({
            "params": [loss_fn.log_temperature],
            "lr": loss_config.get("temperature_lr", 1e-3),
            "weight_decay": 0.0,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        eps=training_config.get("adam_epsilon", 1e-8),
    )

    num_epochs = training_config.get("num_epochs", 12)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = training_config.get("warmup_steps", 200)
    adaptive_warmup = max(10, int(num_training_steps * 0.05))
    effective_warmup = min(warmup_steps, adaptive_warmup) if num_training_steps < warmup_steps else warmup_steps

    if training_config.get("lr_scheduler_type", "cosine") == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=effective_warmup,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=effective_warmup,
            num_training_steps=num_training_steps,
        )

    distillation_config["enabled"] = True
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        logging_steps=training_config.get("logging_steps", 50),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        debug=debug,
        use_amp=use_bf16,
        use_bf16=use_bf16,
        use_ema=training_config.get("ema", {}).get("enabled", True),
        early_stopping_patience=training_config.get("early_stopping", {}).get("patience", 5),
        max_eval_batches=training_config.get("max_eval_batches", 100),
        curriculum_config=training_config.get("curriculum", {}),
        matryoshka_dims=training_config.get("matryoshka", {}).get("dims") if training_config.get("matryoshka", {}).get("enabled", False) else None,
        base_hard_negative_weight=loss_config.get("hard_negative_weight", 1.5),
        head_type=checkpoint_head_type,
        head_params=checkpoint_head_params,
        query_doc_eval_samples=query_doc_eval_samples,
        composite_weights=evaluation_config.get("composite_weights"),
        selection_metric=evaluation_config.get("selection_metric", DEFAULT_SELECTION_METRIC),
        train_sampler=train_sampler,
        mode_routing_config=stage_b_config.get("mode_routing") if use_stage_b_profile else None,
        aux_objectives_config=stage_b_config.get("aux_objectives") if use_stage_b_profile else None,
        retrieval_modes=stage_b_config.get("retrieval_modes") if use_stage_b_profile else None,
        teacher_cache=teacher_cache,
        distillation_config=distillation_config,
        teacher_projection=teacher_projection,
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(output_dir / "distillation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({
            "student_checkpoint": str(student_checkpoint),
            "teacher_cache_dir": str(teacher_cache_dir),
            "data_profile": data_profile,
            "projection_enabled": teacher_projection is not None,
            "selection_metric": evaluation_config.get("selection_metric", DEFAULT_SELECTION_METRIC),
            "training_overrides": training_overrides,
            "data_overrides": data_overrides,
        }, handle, indent=2)

    logger.info(f"Distillation complete -> {output_dir}")
    return history


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding Head Bake-Off Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--build_teacher_cache", action="store_true",
                        help="Milestone 2: build teacher embedding cache using configured teacher/data sources")
    parser.add_argument("--distill_teacher_cache", action="store_true",
                        help="Milestone 3: load runtime.student_checkpoint and distill using distillation.teacher_cache_dir")
    parser.add_argument("--head_type", type=str, default=None, help="Single head type to train (e.g. agreement_gated)")
    parser.add_argument("--run_all", action="store_true", help="Train all configured heads together with a shared encoder pass")
    parser.add_argument("--run_sequential", action="store_true", help="Legacy mode: run configured heads one by one")
    parser.add_argument("--stage_b", type=str, default=None,
                        help="Stage B: path to bakeoff winner checkpoint (e.g. outputs/embedding-bakeoff/agreement_gated_v2/best). "
                             "Loads trained head weights and continues training with all data slices.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max total samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiments_config = config.get("experiments", {})
    output_base = Path(config.get("output", {}).get("dir", "outputs/embedding-bakeoff"))

    if args.build_teacher_cache:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# RUN MODE: BUILD TEACHER CACHE (--build_teacher_cache)")
        logger.info("# Existing source loader + Qwen teacher cache generation")
        logger.info("#" * 70)
        build_teacher_cache(
            config=config,
            max_samples=args.max_samples,
        )

    elif args.distill_teacher_cache:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# RUN MODE: DISTILL FROM TEACHER CACHE (--distill_teacher_cache)")
        logger.info("# Existing student checkpoint + cached Qwen teacher supervision")
        logger.info("#" * 70)
        config.setdefault("distillation", {})["enabled"] = True
        distill_output = output_base / "distill"
        distill_output.mkdir(parents=True, exist_ok=True)
        run_distillation(
            config=config,
            output_dir=distill_output,
            debug=args.debug,
            max_samples=args.max_samples,
        )

    elif args.stage_b:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# RUN MODE: STAGE B - DOMAIN ADAPTATION")
        logger.info("# Trained head from bakeoff + AgreementGatedHeadV2 specialization")
        logger.info("#" * 70)
        stage_b_output = output_base / "stage_b"
        stage_b_output.mkdir(parents=True, exist_ok=True)
        run_stage_b(
            config=config,
            bakeoff_checkpoint=args.stage_b,
            output_dir=stage_b_output,
            debug=args.debug,
            max_samples=args.max_samples,
        )

    elif args.run_all:
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

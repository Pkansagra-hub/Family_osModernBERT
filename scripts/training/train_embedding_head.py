#!/usr/bin/env python
"""
Embedding Head Retraining - SOTA Attentive EmbeddingHead with InfoNCE

Loads ModernBertMultiTaskModel from checkpoint-8000 (12 heads),
replaces EmbeddingHead with SOTA attentive version (NV-Embed style
latent cross-attention + SwiGLU gated projection), trains with
FamilyContrastiveLoss (InfoNCE) on hard-negative triplets.

All other 11 heads + encoder remain FROZEN.

Usage:
    python scripts/training/train_embedding_head.py \
        --config configs/training/embedding_head_retrain.yaml

    # Debug mode
    python scripts/training/train_embedding_head.py \
        --config configs/training/embedding_head_retrain.yaml \
        --debug --max_samples 500

Architecture:
    checkpoint-8000 (ModernBertMultiTaskModel with 12 heads)
        |
        +-- encoder [FROZEN]
        +-- emotions head [FROZEN]
        +-- safety_generic head [FROZEN]
        +-- safety_familyos head [FROZEN]
        +-- sentiment head [FROZEN]
        +-- intent head [FROZEN]
        +-- ingress head [FROZEN]
        +-- nli head [FROZEN]
        +-- relation head [FROZEN]
        +-- counterfactual head [FROZEN]
        +-- ner_general head [FROZEN]
        +-- ner_family head [FROZEN]
        +-- temporal head [FROZEN]
        |
        +-- embedding head [REPLACED with SOTA attentive, TRAINABLE]

Output: ONE checkpoint with all capabilities intact

Author: FamilyOS Team
Date: March 2026
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

# Use centralized model exports from modernbert_multitask (Epic 3.3.1-3.3.2)
from modeling_studio.models.modernbert_multitask import (
    ModernBertMultiTaskModel,
    Capability,
)

# Import SOTA EmbeddingHead
from modeling_studio.models.heads import EmbeddingHead

# Import FamilyContrastiveLoss (InfoNCE)
from familyos_ultrabert.models.losses import FamilyContrastiveLoss

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tokenizers").setLevel(logging.WARNING)


def log_section(title: str) -> None:
    """Log a section header."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def log_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> None:
    """Log a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    logger.info(header_line)
    logger.info("-" * sum(col_widths))
    for row in rows:
        row_line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        logger.info(row_line)


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


# =============================================================================
# Dataset - Triplet Format for Contrastive Learning
# =============================================================================


class TripletDataset(Dataset):
    """
    Dataset that loads triplet data for contrastive embedding training.

    Each sample is a dict with:
        - anchor: str
        - positive: str
        - negative: str
        - is_hard_negative: bool (True if from hard_negatives dir)

    Supports multiple data directories with automatic JSONL loading.
    """

    def __init__(
        self,
        data_paths: list[Path],
        max_samples: int | None = None,
    ):
        """
        Args:
            data_paths: List of directories or JSONL files containing triplets
            max_samples: Max total samples (for debugging)
        """
        self.samples = []

        for path in data_paths:
            if path.is_dir():
                jsonl_files = sorted(path.glob("*.jsonl"))
                if not jsonl_files:
                    jsonl_files = sorted(path.glob("**/*.jsonl"))
                for jsonl_file in jsonl_files:
                    self._load_jsonl(jsonl_file, max_samples)
                    if max_samples and len(self.samples) >= max_samples:
                        break
            elif path.exists() and path.suffix == ".jsonl":
                self._load_jsonl(path, max_samples)
            else:
                logger.warning(f"Data path not found: {path}")

            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # Shuffle
        random.shuffle(self.samples)
        logger.info(f"  TripletDataset: {len(self.samples)} samples from {len(data_paths)} sources")

    def _load_jsonl(self, path: Path, max_samples: int | None = None) -> None:
        """Load triplets from a JSONL file."""
        # Skip non-triplet files (hash indexes, metadata, etc.)
        if "hash_index" in path.name or not path.name.startswith("triplets"):
            return

        count_before = len(self.samples)
        is_hard_neg_dir = "hard_negative" in str(path.parent) or "hard_negative" in path.name

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

                anchor = raw.get("anchor", "")
                positive = raw.get("positive", "")
                negative = raw.get("negative", "")

                if not anchor or not positive or not negative:
                    continue

                self.samples.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "is_hard_negative": is_hard_neg_dir or bool(raw.get("hard_negative_type")),
                })

        loaded = len(self.samples) - count_before
        if loaded > 0:
            logger.info(f"    {path.name}: {loaded} triplets (hard_neg={is_hard_neg_dir})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class TripletCollator:
    """
    Collator that tokenizes anchor, positive, and negative texts for contrastive training.

    Returns a batch dict with:
        - anchor_input_ids, anchor_attention_mask
        - positive_input_ids, positive_attention_mask
        - negative_input_ids, negative_attention_mask
        - hard_negative_mask: bool tensor [batch_size]
    """

    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        anchors = [f["anchor"] for f in features]
        positives = [f["positive"] for f in features]
        negatives = [f["negative"] for f in features]
        hard_neg_flags = [f.get("is_hard_negative", False) for f in features]

        anchor_enc = self.tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive_enc = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        negative_enc = self.tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids": negative_enc["input_ids"],
            "negative_attention_mask": negative_enc["attention_mask"],
            "hard_negative_mask": torch.tensor(hard_neg_flags, dtype=torch.bool),
        }


# =============================================================================
# Model Functions
# =============================================================================


def encode_texts(
    model: ModernBertMultiTaskModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Encode texts through frozen encoder + trainable embedding head.

    Args:
        model: The multi-task model
        input_ids: Token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]

    Returns:
        Embeddings [batch, output_dim]
    """
    # Encoder forward (frozen, no grad needed)
    encoder_output = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if hasattr(encoder_output, "last_hidden_state"):
        hidden_states = encoder_output.last_hidden_state
    elif isinstance(encoder_output, tuple):
        hidden_states = encoder_output[0]
    else:
        hidden_states = encoder_output

    # Embedding head forward (trainable)
    embeddings = model.heads["embedding"](
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )
    return embeddings


def load_model_and_replace_embedding_head(
    checkpoint_path: str | Path,
    embedding_config: dict,
    exclude_decoder: bool = True,
    use_flash_attention: bool = False,
) -> ModernBertMultiTaskModel:
    """
    Load ModernBertMultiTaskModel and replace embedding head with SOTA attentive version.

    Loads the full checkpoint (encoder + all 12 heads), then replaces ONLY
    the embedding head with the new SOTA architecture. All other heads preserved.

    Args:
        checkpoint_path: Path to checkpoint-8000
        embedding_config: Dict with embedding head parameters
        exclude_decoder: If True, skip loading GPT-2 decoder (saves 355M params)
        use_flash_attention: If True, enable Flash Attention 2 (requires A100+)

    Returns:
        Model with replaced embedding head
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Enable Flash Attention 2 if requested (A100/H100 feature)
    if use_flash_attention:
        config.attn_implementation = "flash_attention_2"
        logger.info("  Flash Attention 2: ENABLED")

    # Determine which capabilities to load
    # Exclude COUNTERFACTUAL (GPT-2 decoder) to save 355M params
    if exclude_decoder:
        capabilities = [cap for cap in Capability if cap != Capability.COUNTERFACTUAL]
        logger.info("Excluding GPT-2 decoder (COUNTERFACTUAL) - saves 355M params")
    else:
        capabilities = list(Capability)

    # Create model instance
    model = ModernBertMultiTaskModel(
        config=config,
        capabilities=capabilities,
        freeze_encoder=False,  # Will freeze later
    )

    # Force encoder initialization
    model._init_encoder()

    # Load saved weights from checkpoint
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        weights_path = checkpoint_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    # Load encoder weights (keys start with "encoder.")
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"  Loaded encoder: {len(encoder_state)} tensors")

    # Load head weights (keys start with "heads.")
    for head_name in model.heads.keys():
        head_prefix = f"heads.{head_name}."
        head_state = {
            k.replace(head_prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(head_prefix)
        }
        if head_state:
            try:
                model.heads[head_name].load_state_dict(head_state, strict=True)
            except Exception as e:
                logger.warning(f"Could not load {head_name} head: {e}")

    hidden_size = model.config.hidden_size
    logger.info(f"  Loaded {len(model.heads)} heads, hidden_size={hidden_size}")

    # Replace embedding head with SOTA attentive version
    pooling = embedding_config.get("pooling", "attentive")
    output_dim = embedding_config.get("output_dim", None)
    normalize = embedding_config.get("normalize", True)
    num_latents = embedding_config.get("num_latents", 8)
    num_attn_heads = embedding_config.get("num_attn_heads", 8)

    new_embedding_head = EmbeddingHead(
        hidden_size=hidden_size,
        output_dim=output_dim,
        pooling=pooling,
        normalize=normalize,
        num_latents=num_latents,
        num_attn_heads=num_attn_heads,
    )

    old_class = type(model.heads["embedding"]).__name__
    old_params = sum(p.numel() for p in model.heads["embedding"].parameters())
    model.heads["embedding"] = new_embedding_head
    new_params = sum(p.numel() for p in new_embedding_head.parameters())

    logger.info(f"  Replaced embedding head:")
    logger.info(f"    Old: {old_class} ({old_params:,} params)")
    logger.info(f"    New: EmbeddingHead(pooling={pooling}, output_dim={output_dim or hidden_size}, "
                f"latents={num_latents}, attn_heads={num_attn_heads}) ({new_params:,} params)")

    return model


def freeze_model_except_embedding_head(model: ModernBertMultiTaskModel) -> None:
    """
    Freeze everything except the embedding head.

    Args:
        model: The model
    """
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Freeze all heads except embedding
    for name, head in model.heads.items():
        for param in head.parameters():
            param.requires_grad = (name == "embedding")

    # Log summary
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    emb_params = sum(p.numel() for p in model.heads["embedding"].parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_heads = [n for n in model.heads.keys() if n != "embedding"]

    logger.info(f"  Encoder: {encoder_params:,} params (frozen)")
    logger.info(f"  Frozen heads: {', '.join(frozen_heads)} ({len(frozen_heads)} heads)")
    logger.info(f"  Embedding head: {emb_params:,} params (TRAINABLE)")
    logger.info(f"  Total trainable: {trainable_total:,} params")


def get_trainable_params(model: ModernBertMultiTaskModel) -> list[nn.Parameter]:
    """Get all trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


# =============================================================================
# Data Loading
# =============================================================================


def get_embedding_data_paths(data_config: dict, data_root: Path) -> list[Path]:
    """
    Get data paths for embedding training from config.

    Supports comma-separated paths, e.g.:
        train: "familyos/embeddings/silver_synthetic,familyos/embeddings/hard_negatives"

    Args:
        data_config: Config dict with embedding data paths
        data_root: Root data directory

    Returns:
        List of resolved paths
    """
    embedding_config = data_config.get("embedding", {})
    train_paths_str = embedding_config.get("train", "")

    if not train_paths_str:
        raise ValueError("No embedding training data paths in config")

    path_strings = [p.strip() for p in train_paths_str.split(",") if p.strip()]
    resolved_paths = []

    for path_str in path_strings:
        path = data_root / path_str if not Path(path_str).is_absolute() else Path(path_str)
        if path.exists():
            resolved_paths.append(path)
        else:
            logger.warning(f"Data path not found: {path}")

    return resolved_paths


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
    """
    Single training step: encode anchor/positive/negative, compute InfoNCE loss.

    Supports Matryoshka Representation Learning (Kusupati et al., 2022):
    when matryoshka_dims is provided, computes InfoNCE at multiple truncated
    dimensions and averages the losses, enabling flexible deployment.

    Args:
        model: The model (encoder frozen, embedding head trainable)
        batch: Batch from TripletCollator
        loss_fn: FamilyContrastiveLoss instance
        device: Device
        debug: Enable verbose debug logging
        use_amp: Use automatic mixed precision
        amp_dtype: AMP data type
        matryoshka_dims: List of dimensions for Matryoshka loss (e.g. [768, 512, 256, 128])

    Returns:
        Dict with total_loss and similarity metrics
    """
    # Move batch to device
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
        # Encoder forward is frozen - no grad needed for encoder
        with torch.no_grad():
            # Encode anchor
            enc_out = model.encoder(input_ids=anchor_ids, attention_mask=anchor_mask)
            anchor_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

            # Encode positive
            enc_out = model.encoder(input_ids=positive_ids, attention_mask=positive_mask)
            positive_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

            # Encode negative
            enc_out = model.encoder(input_ids=negative_ids, attention_mask=negative_mask)
            negative_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

        # Embedding head forward (trainable - grads flow here)
        embedding_head = model.heads["embedding"]
        anchor_emb = embedding_head(anchor_hidden, anchor_mask)
        positive_emb = embedding_head(positive_hidden, positive_mask)
        negative_emb = embedding_head(negative_hidden, negative_mask)

        # Compute loss (with optional Matryoshka multi-dim)
        if matryoshka_dims:
            # Matryoshka Representation Learning (Kusupati et al., 2022):
            # compute InfoNCE at multiple truncated dimensions, average losses.
            # Enables deployment at any dim (768/512/256/128) with one model.
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
            # Standard single-dim loss
            negatives = negative_emb.unsqueeze(1)  # [B, 1, dim]
            hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)  # [B, 1]
            loss = loss_fn(
                anchor=anchor_emb,
                positive=positive_emb,
                negatives=negatives,
                hard_negative_mask=hard_neg_mask_expanded,
            )

    # Compute similarity metrics for logging (no grad)
    with torch.no_grad():
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb).mean().item()
        margin = pos_sim - neg_sim

    result = {
        "total_loss": loss,
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "margin": margin,
    }

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
    """
    Evaluate embedding model on validation set.

    Computes:
        - Mean positive similarity (higher is better)
        - Mean negative similarity (lower is better)
        - Mean margin (pos_sim - neg_sim, higher is better)
        - Accuracy (fraction where pos_sim > neg_sim)
        - Validation loss

    Args:
        model: The model
        val_loader: Validation dataloader
        loss_fn: FamilyContrastiveLoss for computing val loss
        device: Device
        debug: Enable verbose debug logging
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype
        max_batches: Max batches to evaluate (None = all)

    Returns:
        Dict with evaluation metrics
    """
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

        anchor_ids = batch["anchor_input_ids"].to(device)
        anchor_mask = batch["anchor_attention_mask"].to(device)
        positive_ids = batch["positive_input_ids"].to(device)
        positive_mask = batch["positive_attention_mask"].to(device)
        negative_ids = batch["negative_input_ids"].to(device)
        negative_mask = batch["negative_attention_mask"].to(device)
        hard_neg_mask = batch["hard_negative_mask"].to(device)

        with amp_context:
            # Encode all three
            enc_out = model.encoder(input_ids=anchor_ids, attention_mask=anchor_mask)
            anchor_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

            enc_out = model.encoder(input_ids=positive_ids, attention_mask=positive_mask)
            positive_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

            enc_out = model.encoder(input_ids=negative_ids, attention_mask=negative_mask)
            negative_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else (enc_out[0] if isinstance(enc_out, tuple) else enc_out)

            embedding_head = model.heads["embedding"]
            anchor_emb = embedding_head(anchor_hidden, anchor_mask)
            positive_emb = embedding_head(positive_hidden, positive_mask)
            negative_emb = embedding_head(negative_hidden, negative_mask)

            # Compute val loss
            negatives = negative_emb.unsqueeze(1)
            hard_neg_mask_expanded = hard_neg_mask.unsqueeze(1)
            batch_loss = loss_fn(
                anchor=anchor_emb,
                positive=positive_emb,
                negatives=negatives,
                hard_negative_mask=hard_neg_mask_expanded,
            )

        # Similarity metrics
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb)  # [B]
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb)  # [B]
        correct = (pos_sim > neg_sim).float()

        batch_size = anchor_emb.size(0)
        total_loss += batch_loss.item() * batch_size
        total_pos_sim += pos_sim.sum().item()
        total_neg_sim += neg_sim.sum().item()
        total_correct += correct.sum().item()
        total_samples += batch_size

        # Track hard negative accuracy separately
        if hard_neg_mask.any():
            hard_correct = correct[hard_neg_mask].sum().item()
            total_hard_neg_correct += hard_correct
            total_hard_neg_samples += hard_neg_mask.sum().item()

    model.train()

    if total_samples == 0:
        return {"val_loss": 0, "pos_sim": 0, "neg_sim": 0, "margin": 0, "accuracy": 0}

    avg_pos_sim = total_pos_sim / total_samples
    avg_neg_sim = total_neg_sim / total_samples
    avg_margin = avg_pos_sim - avg_neg_sim
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples

    hard_neg_accuracy = (
        total_hard_neg_correct / total_hard_neg_samples
        if total_hard_neg_samples > 0
        else 0.0
    )

    metrics = {
        "val_loss": avg_loss,
        "pos_sim": avg_pos_sim,
        "neg_sim": avg_neg_sim,
        "margin": avg_margin,
        "accuracy": accuracy,
        "hard_neg_accuracy": hard_neg_accuracy,
        "hard_neg_samples": total_hard_neg_samples,
        "total_samples": total_samples,
    }

    logger.info(f"  val_loss={avg_loss:.4f} | pos_sim={avg_pos_sim:.4f} neg_sim={avg_neg_sim:.4f} "
                f"margin={avg_margin:.4f} | acc={accuracy:.4f} hard_neg_acc={hard_neg_accuracy:.4f}")

    return metrics


# =============================================================================
# Training Loop
# =============================================================================


def train(
    model: ModernBertMultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: FamilyContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    output_dir: Path,
    tokenizer=None,
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
) -> dict[str, Any]:
    """
    Training loop for embedding head retraining with InfoNCE loss.

    Features:
    - Mixed precision training (AMP) for 2x speedup
    - BFloat16 support for A100 (better numerical stability)
    - Exponential Moving Average (EMA) for smoother checkpoints
    - Early stopping based on margin metric
    - Gradient accumulation for effective large batch sizes
    - Curriculum learning: ramp hard negative weight from 0 to full over warmup epochs
    - Matryoshka loss: multi-dim InfoNCE for flexible deployment (768/512/256/128)
    - Learnable temperature: temperature parameter updated via gradient descent

    Args:
        model: The model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: FamilyContrastiveLoss instance
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        num_epochs: Number of epochs
        output_dir: Output directory
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        gradient_accumulation_steps: Gradient accumulation
        max_grad_norm: Max gradient norm
        debug: Enable verbose debug logging
        use_amp: Use automatic mixed precision
        use_bf16: Use bfloat16 instead of float16
        use_ema: Use exponential moving average
        early_stopping_patience: Stop after N evals without improvement
        max_eval_batches: Max batches during mid-training eval (None = all)

    Returns:
        Training history
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine AMP dtype
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Setup AMP GradScaler (disabled for bf16 - not needed)
    use_scaler = use_amp and device.type == "cuda" and not use_bf16
    scaler = GradScaler("cuda", enabled=use_scaler)
    logger.info(f"  Mixed Precision: {'bf16' if use_bf16 else 'fp16' if use_amp else 'disabled'}")
    logger.info(f"  GradScaler: {'enabled' if use_scaler else 'disabled (bf16 mode)'}")

    # Setup EMA model for smoother checkpoints
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(model)
        logger.info(f"  EMA: enabled (decay via AveragedModel)")

    model.train()
    global_step = 0
    best_margin = -1.0  # Track best margin (pos_sim - neg_sim)
    no_improve_count = 0
    history = {
        "train_loss": [],
        "train_pos_sim": [],
        "train_neg_sim": [],
        "train_margin": [],
        "eval_metrics": [],
    }

    log_section("TRAINING")
    logger.info(f"  Epochs: {num_epochs} | Batches: {len(train_loader)} | Grad accum: {gradient_accumulation_steps}")
    logger.info(f"  Save: every {save_steps} steps | Eval: every {eval_steps} steps")
    logger.info(f"  Early stopping: patience={early_stopping_patience}")

    for epoch in range(num_epochs):
        logger.info("")
        logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        # Curriculum learning: ramp hard negative weight from 0 to full
        if curriculum_config and curriculum_config.get("enabled", False):
            warmup_epochs = curriculum_config.get("warmup_epochs", num_epochs)
            if warmup_epochs > 1:
                scale = min(1.0, epoch / (warmup_epochs - 1))
            else:
                scale = 1.0
            current_hn_weight = base_hard_negative_weight * scale
            loss_fn.hard_negative_weight = current_hn_weight
            logger.info(f"  Curriculum: hard_negative_weight={current_hn_weight:.3f} (scale={scale:.2f})")

        # Log learnable temperature if enabled
        if loss_fn.log_temperature.requires_grad:
            current_temp = loss_fn.log_temperature.exp().item()
            logger.info(f"  Learned temperature: {current_temp:.4f}")

        epoch_loss = 0.0
        epoch_pos_sim = 0.0
        epoch_neg_sim = 0.0
        epoch_steps = 0

        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            step_debug = debug and (step < 5 or step % 100 == 0)

            # Forward pass
            losses = train_step(
                model, batch, loss_fn, device,
                debug=step_debug,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                matryoshka_dims=matryoshka_dims,
            )
            loss = losses["total_loss"]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # Unscale before gradient clipping
                if use_scaler:
                    scaler.unscale_(optimizer)

                # Clip gradients
                trainable_params = get_trainable_params(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                if step_debug:
                    logger.debug(f"Gradient norm (before clip): {grad_norm:.4f}")

                # Optimizer step
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update EMA model
                if ema_model is not None:
                    ema_model.update_parameters(model)

                # Save checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    logger.info(f"\nSaving checkpoint at step {global_step}...")
                    save_checkpoint(model, output_dir / f"checkpoint-{global_step}", tokenizer, optimizer, scheduler)
                    if ema_model is not None:
                        save_checkpoint(ema_model.module, output_dir / f"checkpoint-{global_step}-ema", tokenizer)

                # Mid-epoch evaluation
                if global_step > 0 and global_step % eval_steps == 0 and val_loader is not None:
                    logger.info(f"")
                    logger.info(f"--- Eval @ step {global_step} ---")

                    eval_metrics = evaluate(
                        model, val_loader, loss_fn, device,
                        debug=debug,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        max_batches=max_eval_batches,
                    )

                    current_margin = eval_metrics["margin"]
                    history["eval_metrics"].append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        **eval_metrics,
                    })

                    # Save best model based on margin
                    if current_margin > best_margin:
                        best_margin = current_margin
                        no_improve_count = 0
                        logger.info(f"New best margin={best_margin:.4f}! Saving...")
                        save_checkpoint(model, output_dir / "best", tokenizer, optimizer, scheduler)
                        if ema_model is not None:
                            save_checkpoint(ema_model.module, output_dir / "best-ema", tokenizer)
                    else:
                        no_improve_count += 1
                        logger.info(f"No improvement ({no_improve_count}/{early_stopping_patience})")

                    # Early stopping
                    if no_improve_count >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {no_improve_count} evals without improvement")
                        save_checkpoint(model, output_dir / "early-stop", tokenizer, optimizer, scheduler)
                        if ema_model is not None:
                            save_checkpoint(ema_model.module, output_dir / "early-stop-ema", tokenizer)
                        return history

                    model.train()

            # Accumulate epoch stats
            epoch_loss += losses["total_loss"].item()
            epoch_pos_sim += losses["pos_sim"]
            epoch_neg_sim += losses["neg_sim"]
            epoch_steps += 1

            # Update progress bar
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                margin=f"{losses['margin']:.3f}",
                lr=f"{lr:.2e}",
            )

            # Periodic logging
            if global_step > 0 and global_step % logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_pos = epoch_pos_sim / epoch_steps
                avg_neg = epoch_neg_sim / epoch_steps
                logger.info(f"  Step {global_step}: loss={avg_loss:.4f} pos_sim={avg_pos:.4f} "
                            f"neg_sim={avg_neg:.4f} margin={avg_pos - avg_neg:.4f} lr={lr:.2e}")

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        avg_pos = epoch_pos_sim / epoch_steps if epoch_steps > 0 else 0
        avg_neg = epoch_neg_sim / epoch_steps if epoch_steps > 0 else 0
        avg_margin = avg_pos - avg_neg

        history["train_loss"].append(avg_loss)
        history["train_pos_sim"].append(avg_pos)
        history["train_neg_sim"].append(avg_neg)
        history["train_margin"].append(avg_margin)

        logger.info(f"Epoch {epoch + 1} summary: loss={avg_loss:.4f} pos_sim={avg_pos:.4f} "
                    f"neg_sim={avg_neg:.4f} margin={avg_margin:.4f}")

        # Full evaluation at epoch end
        if val_loader is not None:
            logger.info(f"--- Full eval epoch {epoch + 1} ---")
            eval_metrics = evaluate(
                model, val_loader, loss_fn, device,
                debug=debug,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=None,  # Full eval at epoch end
            )
            history["eval_metrics"].append({
                "step": global_step,
                "epoch": epoch + 1,
                "full_eval": True,
                **eval_metrics,
            })

            current_margin = eval_metrics["margin"]
            if current_margin > best_margin:
                best_margin = current_margin
                no_improve_count = 0
                logger.info(f"New best margin={best_margin:.4f}! Saving...")
                save_checkpoint(model, output_dir / "best", tokenizer, optimizer, scheduler)
                if ema_model is not None:
                    save_checkpoint(ema_model.module, output_dir / "best-ema", tokenizer)
            else:
                no_improve_count += 1
                logger.info(f"No improvement ({no_improve_count}/{early_stopping_patience})")

            if no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {no_improve_count} evals without improvement")
                break

    # Save final checkpoint
    log_section("TRAINING COMPLETE")
    logger.info(f"  Best margin: {best_margin:.4f}")
    logger.info(f"  Output: {output_dir}")
    save_checkpoint(model, output_dir / "final", tokenizer, optimizer, scheduler)
    if ema_model is not None:
        save_checkpoint(ema_model.module, output_dir / "final-ema", tokenizer)
        logger.info(f"  EMA checkpoints saved")

    return history


def save_checkpoint(
    model: ModernBertMultiTaskModel,
    checkpoint_dir: Path,
    tokenizer=None,
    optimizer=None,
    scheduler=None,
) -> None:
    """
    Save full model checkpoint with all heads, tokenizer, and optionally optimizer/scheduler.

    Args:
        model: The model
        checkpoint_dir: Output directory
        tokenizer: Tokenizer to save
        optimizer: Optimizer to save (optional)
        scheduler: Scheduler to save (optional)
    """
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build full state dict with proper prefixes
    state_dict = {}

    # Encoder weights
    for name, param in model.encoder.state_dict().items():
        state_dict[f"encoder.{name}"] = param

    # Head weights (ALL heads, not just embedding)
    for head_name, head in model.heads.items():
        for name, param in head.state_dict().items():
            state_dict[f"heads.{head_name}.{name}"] = param

    # Save weights
    save_file(state_dict, checkpoint_dir / "model.safetensors")

    # Save config
    model.config.save_pretrained(checkpoint_dir)

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler for resume
    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Save head architecture info
    head_info = {}
    for head_name, head in model.heads.items():
        info = {
            "class": type(head).__name__,
        }
        if head_name == "embedding":
            info.update({
                "pooling": getattr(head, "pooling", None),
                "output_dim": getattr(head, "output_dim", None),
                "hidden_size": getattr(head, "hidden_size", None),
                "normalize": getattr(head, "normalize", None),
            })
        head_info[head_name] = info

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "embedding_head_retrain",
        "head_info": head_info,
        "trained_head": "embedding",
    }
    with open(checkpoint_dir / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved: {checkpoint_dir.name}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain embedding head with SOTA attentive architecture + InfoNCE loss",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/training/embedding_head_retrain.yaml)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset, frequent saves)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max total samples for debugging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Extract settings
    encoder_config = config.get("encoder", {})
    embedding_head_config = config.get("embedding_head", {})
    loss_config = config.get("loss", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    checkpoint_path = encoder_config.get("checkpoint", "checkpoints/checkpoint-8000")
    output_dir = Path(output_config.get("dir", "outputs/embedding-head-v1"))
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

    # A100 optimization settings
    use_bf16 = training_config.get("bf16", False)
    use_tf32 = training_config.get("tf32", False)
    use_flash_attention = training_config.get("flash_attention", False)

    # EMA and early stopping
    ema_config = training_config.get("ema", {})
    use_ema = ema_config.get("enabled", True)

    es_config = training_config.get("early_stopping", {})
    early_stopping_patience = es_config.get("patience", 5) if es_config.get("enabled", True) else 100000

    # Scheduler type
    lr_scheduler_type = training_config.get("lr_scheduler_type", "cosine")

    # Loss config
    temperature = loss_config.get("temperature", 0.07)
    hard_negative_weight = loss_config.get("hard_negative_weight", 1.5)
    learnable_temperature = loss_config.get("learnable_temperature", False)
    temperature_lr = loss_config.get("temperature_lr", 1e-3)

    # Curriculum learning config
    curriculum_config = training_config.get("curriculum", {})

    # Matryoshka Representation Learning config
    matryoshka_config = training_config.get("matryoshka", {})
    matryoshka_dims = matryoshka_config.get("dims", None) if matryoshka_config.get("enabled", False) else None

    # Debug mode
    max_samples = args.max_samples
    debug_mode = args.debug
    if debug_mode:
        max_samples = max_samples or 500
        save_steps = 50
        eval_steps = 50
        logger.info(f"DEBUG MODE: max_samples={max_samples}")

    # Seed for reproducibility
    seed = training_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: {gpu_name} ({gpu_mem:.1f}GB)")

        # Enable TF32 for A100 (massive speedup for matmuls)
        if use_tf32 and "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("  TF32: ENABLED (A100 detected)")
        elif use_tf32:
            logger.info("  TF32: requested but non-A100 GPU")
    else:
        logger.info(f"Device: CPU")

    # Load model and replace embedding head
    log_section("MODEL")
    model = load_model_and_replace_embedding_head(
        checkpoint_path,
        embedding_config=embedding_head_config,
        use_flash_attention=use_flash_attention,
    )

    # Freeze everything except embedding head
    freeze_model_except_embedding_head(model)

    # Move to device
    model = model.to(device)
    if use_bf16:
        logger.info("  Mixed precision: bf16 (via autocast)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load data
    log_section("DATA")
    data_paths = get_embedding_data_paths(data_config, data_root)
    logger.info(f"  Data sources: {len(data_paths)}")
    for p in data_paths:
        logger.info(f"    {p}")

    full_dataset = TripletDataset(
        data_paths=data_paths,
        max_samples=max_samples,
    )

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    logger.info(f"  Total: {len(full_dataset)} triplets (train={train_size}, val={val_size})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create collator
    collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)

    # Create dataloaders
    effective_workers = 0 if platform.system() == "Windows" else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        drop_last=True,  # Important for contrastive learning - avoid tiny last batch
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

    # Create loss function
    loss_fn = FamilyContrastiveLoss(
        temperature=temperature,
        hard_negative_weight=hard_negative_weight,
        use_hard_negatives=True,
        normalize=False,  # EmbeddingHead already normalizes
    )

    # Enable learnable temperature if configured
    if learnable_temperature:
        loss_fn.log_temperature.requires_grad_(True)
        logger.info(f"  Learnable temperature: ENABLED (init={temperature}, lr={temperature_lr})")

    # Create optimizer with param groups
    trainable_params = get_trainable_params(model)
    total_trainable = sum(p.numel() for p in trainable_params)

    log_section("OPTIMIZER")
    logger.info(f"  Trainable parameters: {total_trainable:,}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")

    param_groups = [
        {
            "params": trainable_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
    ]
    if learnable_temperature:
        param_groups.append({
            "params": [loss_fn.log_temperature],
            "lr": temperature_lr,
            "weight_decay": 0.0,
        })
        logger.info(f"  Temperature LR: {temperature_lr} (separate param group, no weight decay)")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(
            training_config.get("adam_beta1", 0.9),
            training_config.get("adam_beta2", 0.999),
        ),
        eps=training_config.get("adam_epsilon", 1e-8),
    )

    # Create scheduler
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

    # Adaptive warmup
    adaptive_warmup = max(10, int(num_training_steps * 0.05))
    effective_warmup = min(warmup_steps, adaptive_warmup) if num_training_steps < warmup_steps else warmup_steps

    if lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=effective_warmup,
            num_training_steps=num_training_steps,
        )
        logger.info(f"  Using cosine LR schedule")
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=effective_warmup,
            num_training_steps=num_training_steps,
        )
        logger.info(f"  Using linear LR schedule")

    logger.info(f"  Batches: train={len(train_loader)}, val={len(val_loader)}")
    logger.info(f"  Steps: {num_training_steps} total, {effective_warmup} warmup")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Log loss settings
    log_section("LOSS")
    logger.info(f"  Type: InfoNCE (FamilyContrastiveLoss)")
    logger.info(f"  Temperature: {temperature} ({'learnable' if learnable_temperature else 'fixed'})")
    logger.info(f"  Hard negative weight: {hard_negative_weight}")

    # Log curriculum settings
    if curriculum_config and curriculum_config.get("enabled", False):
        logger.info(f"  Curriculum: ENABLED (warmup_epochs={curriculum_config.get('warmup_epochs', num_epochs)})")
    else:
        logger.info(f"  Curriculum: disabled")

    # Log Matryoshka settings
    if matryoshka_dims:
        logger.info(f"  Matryoshka dims: {matryoshka_dims}")
    else:
        logger.info(f"  Matryoshka: disabled (single-dim)")

    # Log optimization settings
    log_section("OPTIMIZATION")
    logger.info(f"  BF16: {use_bf16}")
    logger.info(f"  TF32: {use_tf32}")
    logger.info(f"  Flash Attention: {use_flash_attention}")
    logger.info(f"  AMP: {use_bf16}")
    logger.info(f"  EMA: {use_ema}")
    logger.info(f"  Early stopping: {early_stopping_patience}")

    # Train
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
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        debug=debug_mode,
        use_amp=use_bf16,  # AMP enabled when bf16 is set
        use_bf16=use_bf16,
        use_ema=use_ema,
        early_stopping_patience=early_stopping_patience,
        max_eval_batches=max_eval_batches,
        curriculum_config=curriculum_config,
        matryoshka_dims=matryoshka_dims,
        base_hard_negative_weight=hard_negative_weight,
    )

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training history saved to {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

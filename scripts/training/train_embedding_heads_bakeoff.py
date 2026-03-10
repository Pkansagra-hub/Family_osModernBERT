#!/usr/bin/env python
"""
Embedding Head Bake-Off Training Script

Trains any of the 6 candidate embedding heads from heads_embedding.py
under identical conditions for fair comparison.

Derived from train_embedding_head.py with these additions:
    - Config-driven head selection via embedding_head.head_type
    - Custom head metadata saved per checkpoint for correct reload
    - Support for running multiple experiments sequentially via --experiments
    - All other behavior (data loading, loss, freezing, eval) is identical

Usage:
    # Single experiment
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --head_type agreement_gated

    # All experiments sequentially
    python scripts/training/train_embedding_heads_bakeoff.py \
        --config configs/training/embedding_heads_bakeoff.yaml \
        --run_all

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

Output: ONE checkpoint per experiment with all capabilities intact
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
# Dataset (identical to original)
# =============================================================================


class TripletDataset(Dataset):
    """Loads triplet data for contrastive embedding training."""

    def __init__(self, data_paths: list[Path], max_samples: int | None = None):
        self.samples: list[dict[str, Any]] = []
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
        random.shuffle(self.samples)
        logger.info(f"  TripletDataset: {len(self.samples)} samples from {len(data_paths)} sources")

    def _load_jsonl(self, path: Path, max_samples: int | None = None) -> None:
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
    def __init__(self, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        anchors = [f["anchor"] for f in features]
        positives = [f["positive"] for f in features]
        negatives = [f["negative"] for f in features]
        hard_neg_flags = [f.get("is_hard_negative", False) for f in features]
        anchor_enc = self.tokenizer(anchors, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        positive_enc = self.tokenizer(positives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        negative_enc = self.tokenizer(negatives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
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


# =============================================================================
# Data Loading
# =============================================================================


def get_embedding_data_paths(data_config: dict, data_root: Path) -> list[Path]:
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
        anchor_ids = batch["anchor_input_ids"].to(device)
        anchor_mask = batch["anchor_attention_mask"].to(device)
        positive_ids = batch["positive_input_ids"].to(device)
        positive_mask = batch["positive_attention_mask"].to(device)
        negative_ids = batch["negative_input_ids"].to(device)
        negative_mask = batch["negative_attention_mask"].to(device)
        hard_neg_mask = batch["hard_negative_mask"].to(device)

        with amp_context:
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
) -> None:
    """Save full model checkpoint with bake-off embedding head metadata."""
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {}

    for name, param in model.encoder.state_dict().items():
        state_dict[f"encoder.{name}"] = param
    for head_name, head in model.heads.items():
        for name, param in head.state_dict().items():
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
    embedding_head = model.heads["embedding"] if "embedding" in model.heads else None
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
        info = {"class": type(h).__name__}
        if hn == "embedding":
            info.update({
                "head_type": head_type,
                "pooling": getattr(h, "pooling", None),
                "output_dim": getattr(h, "output_dim", None),
                "hidden_size": getattr(h, "hidden_size", None),
                "normalize": getattr(h, "normalize", None),
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
    parser.add_argument("--run_all", action="store_true", help="Run all experiments defined in config")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max total samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiments_config = config.get("experiments", {})
    output_base = Path(config.get("output", {}).get("dir", "outputs/embedding-bakeoff"))

    if args.run_all:
        # Run all experiments defined in config
        experiments = experiments_config.get("heads", [])
        if not experiments:
            logger.error("No experiments defined in config under experiments.heads")
            return
        results = {}
        for exp in experiments:
            exp_head_type = exp["head_type"]
            exp_params = exp.get("params", {})
            exp_output = output_base / exp_head_type
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# BAKE-OFF EXPERIMENT: {exp_head_type}")
            logger.info(f"{'#' * 70}")
            history = run_experiment(
                config=config, head_type=exp_head_type, head_params=exp_params,
                output_dir=exp_output, debug=args.debug, max_samples=args.max_samples,
            )
            results[exp_head_type] = history

        # Save combined summary
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
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nBake-off summary -> {summary_path}")

        # Print leaderboard
        log_section("BAKE-OFF LEADERBOARD")
        sorted_heads = sorted(summary.items(), key=lambda x: x[1]["best_margin"], reverse=True)
        logger.info(f"{'Rank':<6}{'Head':<25}{'Margin':<10}{'Accuracy':<10}{'HardNeg Acc':<12}")
        logger.info("-" * 63)
        for rank, (ht, metrics) in enumerate(sorted_heads, 1):
            logger.info(f"{rank:<6}{ht:<25}{metrics['best_margin']:<10.4f}{metrics['best_accuracy']:<10.4f}{metrics['best_hard_neg_accuracy']:<12.4f}")

    elif args.head_type:
        # Single experiment from CLI
        # Check if this head has experiment-specific params in config
        exp_params = {}
        for exp in experiments_config.get("heads", []):
            if exp["head_type"] == args.head_type:
                exp_params = exp.get("params", {})
                break

        # Also merge top-level embedding_head params as defaults
        top_level_params = config.get("embedding_head", {})
        merged_params = {**top_level_params, **exp_params}
        # Remove non-constructor keys
        merged_params.pop("head_type", None)
        merged_params.pop("pooling", None)

        exp_output = output_base / args.head_type
        run_experiment(
            config=config, head_type=args.head_type, head_params=merged_params,
            output_dir=exp_output, debug=args.debug, max_samples=args.max_samples,
        )
    else:
        logger.error("Specify --head_type <name> or --run_all")
        logger.info(f"Available heads: {', '.join(sorted(EMBEDDING_HEAD_REGISTRY.keys()))}")


if __name__ == "__main__":
    main()

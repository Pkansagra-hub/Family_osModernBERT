#!/usr/bin/env python
"""
GlobalPointer Head Training - Unified Model

Loads ModernBertMultiTaskModel from checkpoint-18000 (12 heads),
replaces 3 NER heads with GlobalPointer, trains them in parallel,
saves ONE unified checkpoint with all 12 capabilities.

Usage:
    python scripts/training/train_globalpointer_unified.py \
        --config configs/training/globalpointer_heads.yaml

    # Debug mode
    python scripts/training/train_globalpointer_unified.py \
        --config configs/training/globalpointer_heads.yaml \
        --debug --max_samples 100

Architecture:
    checkpoint-18000 (ModernBertMultiTaskModel with 12 heads)
        |
        +-- encoder [FROZEN]
        +-- embedding head [FROZEN]
        +-- emotions head [FROZEN]
        +-- safety_generic head [FROZEN]
        +-- safety_familyos head [FROZEN]
        +-- sentiment head [FROZEN]
        +-- intent head [FROZEN]
        +-- ingress head [FROZEN]
        +-- nli head [FROZEN]
        +-- relation head [FROZEN]
        +-- counterfactual head [FROZEN]
        |
        +-- ner_general head [REPLACED with GlobalPointer, TRAINABLE]
        +-- ner_family head [REPLACED with GlobalPointer, TRAINABLE]
        +-- temporal head [REPLACED with GlobalPointer, TRAINABLE]

Output: ONE checkpoint with all 12 capabilities

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from modeling_studio.data.globalpointer_collator import (
    GlobalPointerCollator,
    NER_GENERAL_LABELS,
    NER_FAMILY_LABELS,
    TEMPORAL_LABELS,
)
from modeling_studio.models.heads import GlobalPointerNERHead
from modeling_studio.models.modernbert_multitask import (
    ModernBertMultiTaskModel,
    Capability,
)

# Configure logging with force flush for Colab compatibility
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,  # Reset any existing handlers
)
logger = logging.getLogger(__name__)

# Force immediate output in Colab/notebooks
import sys
for handler in logging.root.handlers:
    handler.stream = sys.stdout
    handler.flush = lambda: sys.stdout.flush()


# =============================================================================
# Configuration
# =============================================================================

LABEL_CONFIGS = {
    "ner_general": NER_GENERAL_LABELS,
    "ner_family": NER_FAMILY_LABELS,
    "temporal": TEMPORAL_LABELS,
}

# Heads to replace with GlobalPointer
HEADS_TO_REPLACE = ["ner_general", "ner_family", "temporal"]


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
# Dataset
# =============================================================================


class MultiHeadSpanDataset(Dataset):
    """
    Dataset that loads span-format data for multiple heads.
    Each sample includes the head name for routing during training.
    """

    def __init__(
        self,
        data_paths: dict[str, list[Path]],
        max_samples_per_head: int | None = None,
    ):
        """
        Args:
            data_paths: Dict mapping head_name -> list of JSONL file paths
            max_samples_per_head: Max samples per head (for debugging)
        """
        self.samples = []

        for head_name, paths in data_paths.items():
            head_samples = []

            for path in paths:
                if not path.exists():
                    logger.warning(f"Data file not found: {path}")
                    continue

                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if max_samples_per_head and len(head_samples) >= max_samples_per_head:
                            break

                        raw = json.loads(line.strip())
                        sample = self._extract_sample(raw, head_name)

                        if sample is not None:
                            head_samples.append(sample)

                if max_samples_per_head and len(head_samples) >= max_samples_per_head:
                    break

            logger.info(f"Loaded {len(head_samples)} samples for {head_name}")
            self.samples.extend(head_samples)

        # Shuffle all samples together
        random.shuffle(self.samples)
        logger.info(f"Total samples: {len(self.samples)}")

    def _extract_sample(self, raw: dict, head_name: str) -> dict | None:
        """Extract text and entities, tag with head_name.

        Only returns samples that have at least one entity for this head.
        """
        # Direct format: {"text": "...", "entities": [...]}
        if "entities" in raw and "text" in raw:
            entities = raw["entities"]
            if not entities:  # Skip samples with no entities
                return None
            return {
                "text": raw["text"],
                "entities": entities,
                "head_name": head_name,
            }

        # Unified format: {"text": "...", "tasks": {"ner_family": [...], ...}}
        if "tasks" in raw and "text" in raw:
            tasks = raw["tasks"]
            entities = tasks.get(head_name, [])
            if not entities:  # Skip samples with no entities for this head
                return None
            return {
                "text": raw["text"],
                "entities": entities,
                "head_name": head_name,
            }

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class MultiHeadCollator:
    """
    Collator that handles multiple heads in same batch.
    Groups samples by head_name and creates per-head span labels.
    """

    def __init__(
        self,
        tokenizer,
        label_configs: dict[str, dict[str, int]],
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.label_configs = label_configs
        self.max_length = max_length

        # Create per-head collators
        self.collators = {}
        for head_name, labels in label_configs.items():
            self.collators[head_name] = GlobalPointerCollator(
                tokenizer=tokenizer,
                label_to_id=labels,
                max_length=max_length,
            )

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """
        Collate batch with multiple heads.

        Returns:
            {
                "input_ids": (B, L),
                "attention_mask": (B, L),
                "head_names": list of head names per sample,
                "span_labels": {head_name: (B, num_labels, L, L)},
            }
        """
        texts = [f["text"] for f in features]
        head_names = [f["head_name"] for f in features]

        # Tokenize all texts together
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        batch_size = len(texts)
        seq_len = encoding["input_ids"].shape[1]

        # Build span labels per head
        span_labels = {}
        for head_name, labels in self.label_configs.items():
            num_labels = len(labels)
            span_labels[head_name] = torch.zeros(batch_size, num_labels, seq_len, seq_len)

        # Fill in entity spans
        offset_mappings = encoding["offset_mapping"]

        for b, feature in enumerate(features):
            head_name = feature["head_name"]
            labels = self.label_configs[head_name]
            offset_mapping = offset_mappings[b].tolist()

            for entity in feature.get("entities", []):
                char_start = entity["start"]
                char_end = entity["end"]
                label = entity.get("label", entity.get("type"))

                if label not in labels:
                    continue

                label_id = labels[label]

                # Map char span to token span
                tok_start, tok_end = self._char_to_token_span(
                    offset_mapping, char_start, char_end
                )

                if tok_start is not None and tok_end is not None:
                    span_labels[head_name][b, label_id, tok_start, tok_end] = 1.0

        # Remove offset_mapping from output
        del encoding["offset_mapping"]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "head_names": head_names,
            "span_labels": span_labels,
        }

    def _char_to_token_span(
        self,
        offset_mapping: list[tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> tuple[int | None, int | None]:
        """Convert character span to token span."""
        tok_start = tok_end = None

        for i, (cs, ce) in enumerate(offset_mapping):
            if cs == 0 and ce == 0:
                continue
            if cs < char_end and ce > char_start:
                if tok_start is None:
                    tok_start = i
                tok_end = i

        return tok_start, tok_end


# =============================================================================
# Model Functions
# =============================================================================


def load_model_and_replace_heads(
    checkpoint_path: str | Path,
    head_size: int = 64,
    dropout: float = 0.1,
    exclude_decoder: bool = True,
) -> ModernBertMultiTaskModel:
    """
    Load ModernBertMultiTaskModel and replace NER heads with GlobalPointer.

    Properly loads the full checkpoint including trained encoder and heads,
    then replaces the 3 NER heads with GlobalPointer architecture.

    Args:
        checkpoint_path: Path to checkpoint-18000
        head_size: GlobalPointer head dimension
        dropout: Dropout probability
        exclude_decoder: If True, skip loading GPT-2 decoder (saves 355M params)

    Returns:
        Model with replaced heads
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

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
    logger.info(f"Loaded encoder weights: {len(encoder_state)} tensors")

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
                logger.info(f"Loaded {head_name} head: {len(head_state)} tensors")
            except Exception as e:
                logger.warning(f"Could not load {head_name} head: {e}")

    hidden_size = model.config.hidden_size
    logger.info(f"Loaded model with hidden_size={hidden_size}")

    # Log original heads
    logger.info("Original heads:")
    for name, head in model.heads.items():
        param_count = sum(p.numel() for p in head.parameters())
        logger.info(f"  {name}: {type(head).__name__} ({param_count:,} params)")

    # Replace NER heads with GlobalPointer
    for head_name in HEADS_TO_REPLACE:
        if head_name not in model.heads:
            logger.warning(f"Head {head_name} not found in model, skipping")
            continue

        labels = LABEL_CONFIGS[head_name]
        new_head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=len(labels),
            head_size=head_size,
            use_rope=True,
            dropout=dropout,
            loss_type="globalpointer",
        )

        # Replace in ModuleDict
        model.heads[head_name] = new_head

        logger.info(f"Replaced {head_name} with GlobalPointerNERHead ({len(labels)} labels)")

    return model


def freeze_model_except_heads(
    model: ModernBertMultiTaskModel,
    trainable_heads: list[str],
) -> None:
    """
    Freeze everything except specified heads.

    Args:
        model: The model
        trainable_heads: List of head names to keep trainable
    """
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Freeze all heads
    for name, head in model.heads.items():
        for param in head.parameters():
            param.requires_grad = name in trainable_heads

    # Log status
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

    logger.info(f"Encoder: {encoder_params:,} params, {encoder_trainable:,} trainable")

    for name, head in model.heads.items():
        total = sum(p.numel() for p in head.parameters())
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        status = "TRAINABLE" if name in trainable_heads else "FROZEN"
        logger.info(f"Head {name}: {total:,} params, {trainable:,} trainable [{status}]")


def get_trainable_params(model: ModernBertMultiTaskModel) -> list[nn.Parameter]:
    """Get all trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


# =============================================================================
# Data Loading
# =============================================================================


def get_data_paths(data_config: dict, data_root: Path) -> dict[str, list[Path]]:
    """
    Get data paths for each head from config.

    Supports multiple comma-separated paths per head, e.g.:
        train: "path1,path2,path3"

    Args:
        data_config: Config dict with per-head data paths
        data_root: Root data directory

    Returns:
        Dict mapping head_name -> list of paths
    """
    paths = {}

    for head_name in HEADS_TO_REPLACE:
        head_config = data_config.get(head_name, {})
        train_paths_str = head_config.get("train", "")

        if not train_paths_str:
            logger.warning(f"No train path for {head_name}")
            paths[head_name] = []
            continue

        # Support comma-separated paths
        train_path_list = [p.strip() for p in train_paths_str.split(",") if p.strip()]
        head_paths = []

        for train_path in train_path_list:
            train_dir = data_root / train_path if not Path(train_path).is_absolute() else Path(train_path)

            if train_dir.is_dir():
                # Collect all JSONL files
                dir_paths = list(train_dir.glob("*.jsonl"))
                if not dir_paths:
                    dir_paths = list(train_dir.glob("**/*.jsonl"))
                head_paths.extend(dir_paths)
                logger.info(f"  {head_name}: {len(dir_paths)} files from {train_dir}")
            elif train_dir.exists():
                head_paths.append(train_dir)
                logger.info(f"  {head_name}: 1 file from {train_dir}")
            else:
                logger.warning(f"Train path not found for {head_name}: {train_dir}")

        paths[head_name] = head_paths
        logger.info(f"{head_name}: {len(head_paths)} total files")

    return paths


# =============================================================================
# Training Loop
# =============================================================================


def train_step(
    model: ModernBertMultiTaskModel,
    batch: dict,
    device: torch.device,
    debug: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Single training step with parallel head forward.

    Args:
        model: The model
        batch: Batch with input_ids, attention_mask, head_names, span_labels
        device: Device
        debug: Enable verbose debug logging

    Returns:
        Dict with total_loss and per-head losses
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    head_names = batch["head_names"]
    span_labels = {k: v.to(device) for k, v in batch["span_labels"].items()}

    if debug:
        logger.debug(f"Batch input_ids shape: {input_ids.shape}")
        logger.debug(f"Batch head distribution: {dict((h, head_names.count(h)) for h in set(head_names))}")
        for h, labels in span_labels.items():
            positive_count = (labels > 0).sum().item()
            logger.debug(f"  {h} span_labels: shape={labels.shape}, positives={positive_count}")

    # Get encoder hidden states ONCE
    with torch.no_grad():
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

    # Compute loss for each head in parallel
    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    for head_name in HEADS_TO_REPLACE:
        head = model.heads[head_name]
        head_labels = span_labels[head_name]

        # Check if any samples in batch belong to this head
        mask = torch.tensor([n == head_name for n in head_names], device=device)
        if not mask.any():
            continue

        # Forward through head
        output = head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            span_labels=head_labels,
        )

        if "loss" in output and output["loss"] is not None:
            # Weight by number of samples for this head
            weight = mask.sum().float() / len(head_names)
            head_loss = output["loss"] * weight
            losses[head_name] = head_loss
            total_loss = total_loss + head_loss

            if debug:
                logger.debug(f"  {head_name}: raw_loss={output['loss'].item():.4f}, weight={weight:.2f}, weighted_loss={head_loss.item():.4f}")
                logger.debug(f"  {head_name} logits: min={output['logits'].min().item():.2f}, max={output['logits'].max().item():.2f}, mean={output['logits'].mean().item():.2f}")

    if debug:
        logger.debug(f"Total loss: {total_loss.item():.4f}")

    return {"total_loss": total_loss, **losses}


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(
    model: ModernBertMultiTaskModel,
    val_loader: DataLoader,
    device: torch.device,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Evaluate model on validation set using decode_with_nms.

    Computes precision, recall, F1 per head and overall.

    Args:
        model: The model
        val_loader: Validation dataloader
        device: Device
        debug: Enable verbose debug logging

    Returns:
        Dict with metrics per head and overall
    """
    model.eval()

    # Collect predictions and gold for each head
    all_preds = {h: [] for h in HEADS_TO_REPLACE}
    all_golds = {h: [] for h in HEADS_TO_REPLACE}

    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        head_names = batch["head_names"]
        span_labels = {k: v.to(device) for k, v in batch["span_labels"].items()}

        # Get encoder hidden states
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

        # Evaluate each head
        for head_name in HEADS_TO_REPLACE:
            head = model.heads[head_name]
            head_labels = span_labels[head_name]
            id2label = {v: k for k, v in LABEL_CONFIGS[head_name].items()}

            # Forward
            output = head(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            logits = output["logits"]

            # Decode predictions using FAST batch decode method
            # MUST pass id2label so labels match gold format
            preds = head.decode_batch_efficient(
                logits,
                attention_mask=attention_mask,
                threshold=0.0,
                id2label=id2label,
            )

            # Extract gold spans from labels
            batch_size, num_labels, seq_len, _ = head_labels.shape
            for b in range(batch_size):
                # Only evaluate samples that belong to this head
                if head_names[b] != head_name:
                    continue

                # Predicted entities - decode_batch_efficient returns list of dicts
                pred_set = set()
                for entity in preds[b]:
                    pred_set.add((entity["start"], entity["end"], entity["label"]))

                # Gold entities from span_labels
                gold_set = set()
                for label_id in range(num_labels):
                    positions = torch.where(head_labels[b, label_id] > 0)
                    for i, j in zip(positions[0].tolist(), positions[1].tolist()):
                        gold_set.add((i, j, id2label[label_id]))

                all_preds[head_name].append(pred_set)
                all_golds[head_name].append(gold_set)

                if debug and batch_idx == 0 and b < 2:
                    logger.debug(f"Sample {b} ({head_name}):")
                    logger.debug(f"  Predicted: {pred_set}")
                    logger.debug(f"  Gold: {gold_set}")

    model.train()

    # Compute metrics per head
    metrics = {}
    total_pred = total_gold = total_correct = 0

    for head_name in HEADS_TO_REPLACE:
        preds_list = all_preds[head_name]
        golds_list = all_golds[head_name]

        if not preds_list:
            metrics[head_name] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
            continue

        head_pred = sum(len(p) for p in preds_list)
        head_gold = sum(len(g) for g in golds_list)
        head_correct = sum(len(p & g) for p, g in zip(preds_list, golds_list))

        precision = head_correct / head_pred if head_pred > 0 else 0.0
        recall = head_correct / head_gold if head_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[head_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(preds_list),
            "predicted": head_pred,
            "gold": head_gold,
            "correct": head_correct,
        }

        total_pred += head_pred
        total_gold += head_gold
        total_correct += head_correct

        print(f"  {head_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (pred={head_pred}, gold={head_gold}, correct={head_correct})", flush=True)
        logger.info(f"  {head_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (pred={head_pred}, gold={head_gold}, correct={head_correct})")
        sys.stdout.flush()  # Force flush for Colab

    # Overall metrics
    overall_p = total_correct / total_pred if total_pred > 0 else 0.0
    overall_r = total_correct / total_gold if total_gold > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    metrics["overall"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "predicted": total_pred,
        "gold": total_gold,
        "correct": total_correct,
    }

    print(f"  OVERALL: P={overall_p:.4f}, R={overall_r:.4f}, F1={overall_f1:.4f}", flush=True)
    logger.info(f"  OVERALL: P={overall_p:.4f}, R={overall_r:.4f}, F1={overall_f1:.4f}")
    sys.stdout.flush()  # Force flush for Colab

    return metrics


def train(
    model: ModernBertMultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    output_dir: Path,
    save_steps: int = 1000,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Training loop for parallel head training with evaluation after each epoch.

    Args:
        model: The model
        train_loader: Training dataloader
        val_loader: Validation dataloader (optional)
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        num_epochs: Number of epochs
        output_dir: Output directory
        save_steps: Save checkpoint every N steps
        gradient_accumulation_steps: Gradient accumulation
        max_grad_norm: Max gradient norm
        debug: Enable verbose debug logging

    Returns:
        Training history
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    best_f1 = 0.0
    history = {
        "train_loss": [],
        "per_head_loss": {h: [] for h in HEADS_TO_REPLACE},
        "eval_metrics": [],
    }

    logger.info("=" * 80)
    logger.info("TRAINING START")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader) if val_loader else 'None'}")
    logger.info(f"  Save steps: {save_steps}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Max grad norm: {max_grad_norm}")
    logger.info(f"  Debug mode: {debug}")
    logger.info("=" * 80)

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*80}")

        epoch_loss = 0.0
        epoch_head_losses = {h: 0.0 for h in HEADS_TO_REPLACE}
        epoch_steps = 0

        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            # Debug every 100 steps or first 5 steps
            step_debug = debug and (step < 5 or step % 100 == 0)

            if step_debug:
                logger.debug(f"\n--- Step {step} ---")
                logger.debug(f"Batch size: {len(batch['head_names'])}")
                logger.debug(f"Head distribution: {batch['head_names']}")

            losses = train_step(model, batch, device, debug=step_debug)
            loss = losses["total_loss"]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    get_trainable_params(model),
                    max_grad_norm,
                )
                if step_debug:
                    logger.debug(f"Gradient norm (before clip): {grad_norm:.4f}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += losses["total_loss"].item()
            for h in HEADS_TO_REPLACE:
                if h in losses:
                    epoch_head_losses[h] += losses[h].item()
            epoch_steps += 1

            # Update progress bar
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{losses['total_loss'].item():.4f}", lr=f"{lr:.2e}")

            # Log every 50 steps
            if debug and step > 0 and step % 50 == 0:
                avg_so_far = epoch_loss / epoch_steps
                logger.info(f"  Step {step}: avg_loss={avg_so_far:.4f}, lr={lr:.2e}")

            # Save checkpoint every save_steps
            if global_step > 0 and global_step % save_steps == 0:
                logger.info(f"\nSaving checkpoint at step {global_step}...")
                save_checkpoint(model, output_dir / f"checkpoint-{global_step}")

            # Mid-epoch evaluation every 500 steps
            if global_step > 0 and global_step % 500 == 0 and val_loader is not None:
                print(f"\n=== Evaluation at Step {global_step} ===", flush=True)
                logger.info(f"\n--- Evaluation at Step {global_step} ---")
                sys.stdout.flush()
                eval_metrics = evaluate(model, val_loader, device, debug=debug)

                overall_f1 = eval_metrics["overall"]["f1"]
                print(f"Step {global_step} F1: {overall_f1:.4f}", flush=True)
                logger.info(f"Step {global_step} F1: {overall_f1:.4f}")
                sys.stdout.flush()

                # Save best model
                if overall_f1 > best_f1:
                    best_f1 = overall_f1
                    print(f"New best F1! Saving best checkpoint...", flush=True)
                    logger.info(f"New best F1! Saving best checkpoint...")
                    save_checkpoint(model, output_dir / "best")

                # Back to train mode
                model.train()

        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        history["train_loss"].append(avg_loss)

        # Per-head average losses
        for h in HEADS_TO_REPLACE:
            avg_h_loss = epoch_head_losses[h] / epoch_steps if epoch_steps > 0 else 0
            history["per_head_loss"][h].append(avg_h_loss)

        logger.info(f"\nEpoch {epoch + 1} Training Complete:")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        for h in HEADS_TO_REPLACE:
            logger.info(f"  {h} avg loss: {epoch_head_losses[h] / epoch_steps:.4f}")

        # Evaluate after each epoch
        if val_loader is not None:
            logger.info(f"\n--- Evaluation Epoch {epoch + 1} ---")
            eval_metrics = evaluate(model, val_loader, device, debug=debug)
            history["eval_metrics"].append(eval_metrics)

            overall_f1 = eval_metrics["overall"]["f1"]
            logger.info(f"Overall F1: {overall_f1:.4f}")

            # Save best model
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                logger.info(f"New best F1! Saving best checkpoint...")
                save_checkpoint(model, output_dir / "best")

    # Save final checkpoint
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"{'='*80}")
    save_checkpoint(model, output_dir / "final")

    return history


def save_checkpoint(model: ModernBertMultiTaskModel, checkpoint_dir: Path) -> None:
    """
    Save full model checkpoint with all 12 heads.

    Args:
        model: The model
        checkpoint_dir: Output directory
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save using HuggingFace save_pretrained
    model.save_pretrained(checkpoint_dir)

    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "replaced_heads": HEADS_TO_REPLACE,
        "head_architecture": "GlobalPointerNERHead",
    }
    with open(checkpoint_dir / "globalpointer_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GlobalPointer heads with frozen encoder (unified model)",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per head for debugging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Extract settings
    encoder_config = config.get("encoder", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})
    heads_config = config.get("heads", {})

    checkpoint_path = encoder_config.get("checkpoint", "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000")
    output_dir = Path(output_config.get("dir", "outputs/globalpointer-unified-v1"))
    data_root = Path(data_config.get("root", "data"))

    # Training params
    learning_rate = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_epochs = training_config.get("num_epochs", 3)
    batch_size = training_config.get("batch_size", 16)
    max_length = data_config.get("max_length", 256)
    warmup_steps = training_config.get("warmup_steps", 500)
    save_steps = training_config.get("save_steps", 1000)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    val_split = training_config.get("val_split", 0.1)

    # Head architecture
    head_size = heads_config.get("architecture", {}).get("head_size", 64)
    dropout = heads_config.get("architecture", {}).get("dropout", 0.1)

    # Debug mode
    max_samples = args.max_samples
    debug_mode = args.debug
    if debug_mode:
        max_samples = max_samples or 500
        save_steps = 100
        logger.info("=" * 80)
        logger.info("DEBUG MODE ENABLED")
        logger.info("  Max samples per head: {}".format(max_samples))
        logger.info("  Num epochs: {}".format(num_epochs))
        logger.info("  Save steps: {}".format(save_steps))
        logger.info("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and replace heads
    logger.info("\n" + "=" * 80)
    logger.info("LOADING MODEL AND REPLACING HEADS")
    logger.info("=" * 80)
    model = load_model_and_replace_heads(
        checkpoint_path,
        head_size=head_size,
        dropout=dropout,
    )

    # Freeze everything except the 3 new heads
    freeze_model_except_heads(model, HEADS_TO_REPLACE)

    # Move to device
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Get data paths
    data_paths = get_data_paths(data_config, data_root)
    logger.info("\nData paths:")
    for head_name, path in data_paths.items():
        logger.info(f"  {head_name}: {path}")

    # Create full dataset
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    full_dataset = MultiHeadSpanDataset(
        data_paths=data_paths,
        max_samples_per_head=max_samples,
    )
    logger.info(f"Total samples: {len(full_dataset)}")

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    logger.info(f"Train/Val split: {train_size}/{val_size} ({1-val_split:.0%}/{val_split:.0%})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create collator
    collator = MultiHeadCollator(
        tokenizer=tokenizer,
        label_configs=LABEL_CONFIGS,
        max_length=max_length,
    )

    # Create dataloaders
    # Note: num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create optimizer (only trainable params)
    trainable_params = get_trainable_params(model)
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"\nTotal trainable parameters: {total_trainable:,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Create scheduler
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Num epochs: {num_epochs}")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        output_dir=output_dir,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        debug=debug_mode,
    )

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete!")
    logger.info(f"Output saved to {output_dir}")
    logger.info(f"Final checkpoint: {output_dir}/final")
    logger.info(f"Best checkpoint: {output_dir}/best")


if __name__ == "__main__":
    main()

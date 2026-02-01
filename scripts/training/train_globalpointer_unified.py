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

# Use centralized collator exports from trainers.collators (Epic 3.3)
from modeling_studio.trainers.collators import (
    GlobalPointerCollator,
    get_globalpointer_collator,
    GLOBALPOINTER_COLLATOR_MAPPING,
    NER_GENERAL_LABELS,
    NER_FAMILY_LABELS,
    TEMPORAL_LABELS,
)

# Use centralized model exports from modernbert_multitask (Epic 3.3.1-3.3.2)
from modeling_studio.models.modernbert_multitask import (
    ModernBertMultiTaskModel,
    Capability,
    GlobalPointerNERHead,
    create_globalpointer_head,
)

# Import Intent/Ingress V2 heads for classification training
from modeling_studio.models.heads import IntentHeadV2, IngressHeadV2

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

# NER head label configs (span-based)
LABEL_CONFIGS = {
    "ner_general": NER_GENERAL_LABELS,
    "ner_family": NER_FAMILY_LABELS,
    "temporal": TEMPORAL_LABELS,
}

# Classification head label configs (sequence-level)
CLASSIFICATION_LABEL_CONFIGS = {
    "intent_v2": {label: i for i, label in enumerate(IntentHeadV2.INTENT_LABELS)},
    "ingress_v2": {label: i for i, label in enumerate(IngressHeadV2.INGRESS_LABELS)},
}

# Head types for routing
SPAN_HEADS = ["ner_general", "ner_family", "temporal"]
CLASSIFICATION_HEADS = ["intent_v2", "ingress_v2"]

# Heads to replace with GlobalPointer (backward compatible default)
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

            self.samples.extend(head_samples)

        # Shuffle all samples together
        random.shuffle(self.samples)

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


class MultiHeadClassificationDataset(Dataset):
    """
    Dataset that loads classification data for Intent/Ingress V2 heads.
    Supports multi-label format (list of labels) or single-label (string).

    Data format:
        {"text": "...", "intent": "label" or ["label1", "label2"], ...}
        {"text": "...", "ingress": "label" or ["label1", "label2"], ...}
    """

    def __init__(
        self,
        data_paths: dict[str, list[Path]],
        label_configs: dict[str, dict[str, int]],
        max_samples_per_head: int | None = None,
    ):
        """
        Args:
            data_paths: Dict mapping head_name -> list of JSONL file paths
            label_configs: Dict mapping head_name -> label_to_id dict
            max_samples_per_head: Max samples per head (for debugging)
        """
        self.samples = []
        self.label_configs = label_configs

        # Map head_name to data field name
        field_map = {
            "intent_v2": "intent",
            "ingress_v2": "ingress",
        }

        for head_name, paths in data_paths.items():
            field_name = field_map.get(head_name, head_name.replace("_v2", ""))
            label_to_id = label_configs.get(head_name, {})
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
                        sample = self._extract_sample(raw, head_name, field_name, label_to_id)

                        if sample is not None:
                            head_samples.append(sample)

                if max_samples_per_head and len(head_samples) >= max_samples_per_head:
                    break

            logger.info(f"  {head_name}: loaded {len(head_samples)} samples from {len(paths)} files")
            self.samples.extend(head_samples)

        # Shuffle all samples together
        random.shuffle(self.samples)

    def _extract_sample(
        self,
        raw: dict,
        head_name: str,
        field_name: str,
        label_to_id: dict[str, int],
    ) -> dict | None:
        """Extract text and labels, supporting multi-label."""
        text = raw.get("text", "")
        if not text:
            return None

        label_value = raw.get(field_name)
        if label_value is None:
            return None

        # Support both single-label (string) and multi-label (list)
        if isinstance(label_value, str):
            labels = [label_value]
        elif isinstance(label_value, list):
            labels = label_value
        else:
            return None

        # Convert to label IDs, skip unknown labels
        label_ids = []
        for lbl in labels:
            if lbl in label_to_id:
                label_ids.append(label_to_id[lbl])

        if not label_ids:
            return None

        return {
            "text": text,
            "labels": label_ids,  # List of label IDs for multi-label
            "head_name": head_name,
        }

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


class MultiHeadClassificationCollator:
    """
    Collator that handles multiple classification heads (Intent/Ingress V2).
    Creates multi-label targets for each head.
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

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """
        Collate batch for classification heads.

        Returns:
            {
                "input_ids": (B, L),
                "attention_mask": (B, L),
                "head_names": list of head names per sample,
                "classification_labels": {head_name: (B, num_labels) multi-hot},
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
            return_tensors="pt",
        )

        batch_size = len(texts)

        # Build multi-hot labels per head
        classification_labels = {}
        for head_name, labels in self.label_configs.items():
            num_labels = len(labels)
            classification_labels[head_name] = torch.zeros(batch_size, num_labels)

        # Fill in labels
        for b, feature in enumerate(features):
            head_name = feature["head_name"]
            label_ids = feature.get("labels", [])

            for label_id in label_ids:
                classification_labels[head_name][b, label_id] = 1.0

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "head_names": head_names,
            "classification_labels": classification_labels,
        }


class UnifiedMultiHeadCollator:
    """
    Unified collator that handles BOTH span-based (NER) and classification heads.
    Routes samples to appropriate processing based on head type.
    """

    def __init__(
        self,
        tokenizer,
        span_label_configs: dict[str, dict[str, int]],
        classification_label_configs: dict[str, dict[str, int]],
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.span_label_configs = span_label_configs
        self.classification_label_configs = classification_label_configs
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """
        Collate batch with both span and classification heads.

        Returns combined batch with both label types.
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

        # Build span labels for NER heads
        span_labels = {}
        for head_name, labels in self.span_label_configs.items():
            num_labels = len(labels)
            span_labels[head_name] = torch.zeros(batch_size, num_labels, seq_len, seq_len)

        # Build classification labels for Intent/Ingress V2 heads
        classification_labels = {}
        for head_name, labels in self.classification_label_configs.items():
            num_labels = len(labels)
            classification_labels[head_name] = torch.zeros(batch_size, num_labels)

        # Fill in labels based on head type
        offset_mappings = encoding["offset_mapping"]

        for b, feature in enumerate(features):
            head_name = feature["head_name"]

            # Span-based (NER) heads
            if head_name in self.span_label_configs:
                labels = self.span_label_configs[head_name]
                offset_mapping = offset_mappings[b].tolist()

                for entity in feature.get("entities", []):
                    char_start = entity["start"]
                    char_end = entity["end"]
                    label = entity.get("label", entity.get("type"))

                    if label not in labels:
                        continue

                    label_id = labels[label]
                    tok_start, tok_end = self._char_to_token_span(
                        offset_mapping, char_start, char_end
                    )

                    if tok_start is not None and tok_end is not None:
                        span_labels[head_name][b, label_id, tok_start, tok_end] = 1.0

            # Classification heads (Intent/Ingress V2)
            elif head_name in self.classification_label_configs:
                label_ids = feature.get("labels", [])
                for label_id in label_ids:
                    classification_labels[head_name][b, label_id] = 1.0

        # Remove offset_mapping from output
        del encoding["offset_mapping"]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "head_names": head_names,
            "span_labels": span_labels,
            "classification_labels": classification_labels,
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

    # Replace NER heads with GlobalPointer using factory function (Epic 3.3)
    span_heads_to_replace = [h for h in HEADS_TO_REPLACE if h in SPAN_HEADS]
    if span_heads_to_replace:
        logger.info(f"  Replacing span heads: {', '.join(span_heads_to_replace)}")
        for head_name in span_heads_to_replace:
            if head_name not in model.heads:
                logger.warning(f"Head {head_name} not found in model, skipping")
                continue

            # Use factory function for consistent head creation
            new_head = create_globalpointer_head(
                capability=head_name,
                hidden_size=hidden_size,
                head_size=head_size,
                dropout=dropout,
                use_rope=True,
                loss_type="globalpointer",
            )

            # Replace in ModuleDict
            model.heads[head_name] = new_head
            num_labels = len(LABEL_CONFIGS[head_name])
            logger.info(f"    {head_name} -> GlobalPointerNERHead ({num_labels} labels)")

    # Replace classification heads with V2 (Label-Description Embedding) architecture
    classification_heads_to_replace = [h for h in HEADS_TO_REPLACE if h in CLASSIFICATION_HEADS]
    if classification_heads_to_replace:
        logger.info(f"  Replacing classification heads: {', '.join(classification_heads_to_replace)}")
        for head_name in classification_heads_to_replace:
            # Create V2 head with multi-label support
            if head_name == "intent_v2":
                new_head = IntentHeadV2(
                    hidden_size=hidden_size,
                    multi_label=True,  # FamilyOS requires multi-label
                    dropout=dropout,
                )
                # Replace the old "intent" head
                old_name = "intent"
            elif head_name == "ingress_v2":
                new_head = IngressHeadV2(
                    hidden_size=hidden_size,
                    multi_label=True,  # FamilyOS requires multi-label
                    dropout=dropout,
                )
                # Replace the old "ingress" head
                old_name = "ingress"
            else:
                logger.warning(f"Unknown classification head: {head_name}")
                continue

            # Replace old head with V2
            if old_name in model.heads:
                del model.heads[old_name]
            model.heads[head_name] = new_head
            num_labels = len(CLASSIFICATION_LABEL_CONFIGS[head_name])
            logger.info(f"    {old_name} -> {type(new_head).__name__} ({num_labels} labels, multi-label)")

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

    # Log summary
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = {h: sum(p.numel() for p in model.heads[h].parameters()) for h in trainable_heads}

    logger.info(f"  Encoder: {encoder_params:,} params (frozen)")
    logger.info(f"  Trainable heads: {', '.join(f'{h}={head_params[h]:,}' for h in trainable_heads)}")
    logger.info(f"  Total trainable: {trainable_total:,} params")


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
            elif train_dir.exists():
                head_paths.append(train_dir)
            else:
                logger.warning(f"Path not found: {train_dir}")

        paths[head_name] = head_paths

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
    Supports both span-based (NER) and classification (Intent/Ingress V2) heads.

    Args:
        model: The model
        batch: Batch with input_ids, attention_mask, head_names, span_labels, classification_labels
        device: Device
        debug: Enable verbose debug logging

    Returns:
        Dict with total_loss and per-head losses
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    head_names = batch["head_names"]

    # Handle both label types
    span_labels = {k: v.to(device) for k, v in batch.get("span_labels", {}).items()}
    classification_labels = {k: v.to(device) for k, v in batch.get("classification_labels", {}).items()}

    if debug:
        logger.debug(f"Batch input_ids shape: {input_ids.shape}")
        logger.debug(f"Batch head distribution: {dict((h, head_names.count(h)) for h in set(head_names))}")
        for h, labels in span_labels.items():
            positive_count = (labels > 0).sum().item()
            logger.debug(f"  {h} span_labels: shape={labels.shape}, positives={positive_count}")
        for h, labels in classification_labels.items():
            positive_count = (labels > 0).sum().item()
            logger.debug(f"  {h} classification_labels: shape={labels.shape}, positives={positive_count}")

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
        if head_name not in model.heads:
            continue
        head = model.heads[head_name]

        # Check if any samples in batch belong to this head
        mask = torch.tensor([n == head_name for n in head_names], device=device)
        if not mask.any():
            continue

        # Route to appropriate forward based on head type
        if head_name in SPAN_HEADS:
            # Span-based (NER) head
            head_labels = span_labels.get(head_name)
            if head_labels is None:
                continue
            output = head(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                span_labels=head_labels,
            )
        elif head_name in CLASSIFICATION_HEADS:
            # Classification head (Intent/Ingress V2)
            head_labels = classification_labels.get(head_name)
            if head_labels is None:
                continue
            output = head(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                labels=head_labels,
            )
        else:
            continue

        if "loss" in output and output["loss"] is not None:
            # Weight by number of samples for this head
            weight = mask.sum().float() / len(head_names)
            head_loss = output["loss"] * weight
            losses[head_name] = head_loss
            total_loss = total_loss + head_loss

            if debug:
                logger.debug(f"  {head_name}: raw_loss={output['loss'].item():.4f}, weight={weight:.2f}, weighted_loss={head_loss.item():.4f}")
                if "logits" in output:
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
    threshold: float = 2.0,  # Logit threshold: 2.0 = prob > 0.88, reduces FP for untrained models
) -> dict[str, Any]:
    """
    Evaluate model on validation set using decode_batch_efficient.

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
        span_labels = {k: v.to(device) for k, v in batch.get("span_labels", {}).items()}
        classification_labels = {k: v.to(device) for k, v in batch.get("classification_labels", {}).items()}

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
            if head_name not in model.heads:
                continue
            head = model.heads[head_name]

            # Route based on head type
            if head_name in SPAN_HEADS:
                # Span-based (NER) evaluation
                head_labels = span_labels.get(head_name)
                if head_labels is None:
                    continue
                id2label = {v: k for k, v in LABEL_CONFIGS[head_name].items()}

                # Forward
                output = head(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                )
                logits = output["logits"]

                # Decode predictions using FAST batch decode method
                preds = head.decode_batch_efficient(
                    logits,
                    attention_mask=attention_mask,
                    threshold=threshold,
                    id2label=id2label,
                )

                # Extract gold spans from labels
                batch_size, num_labels, seq_len, _ = head_labels.shape
                for b in range(batch_size):
                    if head_names[b] != head_name:
                        continue

                    pred_set = set()
                    for entity in preds[b]:
                        pred_set.add((entity["start"], entity["end"], entity["label"]))

                    gold_set = set()
                    for label_id in range(num_labels):
                        positions = torch.where(head_labels[b, label_id] > 0)
                        for i, j in zip(positions[0].tolist(), positions[1].tolist()):
                            gold_set.add((i, j, id2label[label_id]))

                    all_preds[head_name].append(pred_set)
                    all_golds[head_name].append(gold_set)

            elif head_name in CLASSIFICATION_HEADS:
                # Classification (Intent/Ingress V2) evaluation - multi-label
                head_labels = classification_labels.get(head_name)
                if head_labels is None:
                    continue
                id2label = {v: k for k, v in CLASSIFICATION_LABEL_CONFIGS[head_name].items()}

                # Forward
                output = head(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                )
                probs = output["probabilities"]  # (B, num_labels)

                # Multi-label threshold (0.5 for sigmoid)
                batch_size = probs.shape[0]
                for b in range(batch_size):
                    if head_names[b] != head_name:
                        continue

                    # Predicted labels (above 0.5 threshold)
                    pred_set = set()
                    pred_indices = torch.where(probs[b] > 0.5)[0].tolist()
                    for idx in pred_indices:
                        pred_set.add(id2label[idx])

                    # Gold labels
                    gold_set = set()
                    gold_indices = torch.where(head_labels[b] > 0)[0].tolist()
                    for idx in gold_indices:
                        gold_set.add(id2label[idx])

                    all_preds[head_name].append(pred_set)
                    all_golds[head_name].append(gold_set)

                if debug and batch_idx == 0:
                    logger.debug(f"Classification eval {head_name}: probs shape={probs.shape}")

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

        logger.info(f"  {head_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} | pred={head_pred} gold={head_gold} correct={head_correct}")

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

    logger.info(f"  {'OVERALL':12} P={overall_p:.3f} R={overall_r:.3f} F1={overall_f1:.3f} | pred={total_pred} gold={total_gold} correct={total_correct}")

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
    tokenizer=None,
    save_steps: int = 1000,
    eval_steps: int = 500,
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

    log_section("TRAINING")
    logger.info(f"  Epochs: {num_epochs} | Batches: {len(train_loader)} | Grad accum: {gradient_accumulation_steps}")
    logger.info(f"  Save: every {save_steps} steps | Eval: every {eval_steps} steps")

    for epoch in range(num_epochs):
        logger.info("")
        logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")

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

                # Save checkpoint every save_steps (INSIDE accumulation block)
                if global_step > 0 and global_step % save_steps == 0:
                    logger.info(f"\nSaving checkpoint at step {global_step}...")
                    save_checkpoint(model, output_dir / f"checkpoint-{global_step}", tokenizer, optimizer, scheduler)

                # Mid-epoch evaluation (INSIDE accumulation block)
                if global_step > 0 and global_step % eval_steps == 0 and val_loader is not None:
                    logger.info(f"")
                    logger.info(f"--- Eval @ step {global_step} ---")
                    eval_metrics = evaluate(model, val_loader, device, debug=debug)

                    overall_f1 = eval_metrics["overall"]["f1"]
                    logger.info(f"Overall F1: {overall_f1:.4f}")

                    # Save best model
                    if overall_f1 > best_f1:
                        best_f1 = overall_f1
                        logger.info(f"New best F1={best_f1:.4f}! Saving...")
                        save_checkpoint(model, output_dir / "best", tokenizer, optimizer, scheduler)

                    # Back to train mode
                    model.train()

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

        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        history["train_loss"].append(avg_loss)

        # Per-head average losses
        for h in HEADS_TO_REPLACE:
            avg_h_loss = epoch_head_losses[h] / epoch_steps if epoch_steps > 0 else 0
            history["per_head_loss"][h].append(avg_h_loss)

        # Build loss summary
        head_losses_str = " | ".join(f"{h}={epoch_head_losses[h]/epoch_steps:.3f}" for h in HEADS_TO_REPLACE)
        logger.info(f"Epoch {epoch + 1} loss: {avg_loss:.4f} ({head_losses_str})")

        # Evaluate after each epoch
        if val_loader is not None:
            logger.info(f"--- Eval epoch {epoch + 1} ---")
            eval_metrics = evaluate(model, val_loader, device, debug=debug)
            history["eval_metrics"].append(eval_metrics)

            overall_f1 = eval_metrics["overall"]["f1"]

            # Save best model
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                logger.info(f"New best F1={best_f1:.4f}! Saving...")
                save_checkpoint(model, output_dir / "best", tokenizer, optimizer, scheduler)

    # Save final checkpoint
    log_section("TRAINING COMPLETE")
    logger.info(f"  Best F1: {best_f1:.4f}")
    logger.info(f"  Output: {output_dir}")
    save_checkpoint(model, output_dir / "final", tokenizer, optimizer, scheduler)

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

    Saves heads individually with class type info so GlobalPointer heads can be restored.

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

    # Head weights
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

    # Save head architecture info for proper loading
    head_info = {}
    for head_name, head in model.heads.items():
        head_info[head_name] = {
            "class": type(head).__name__,
            "num_labels": getattr(head, "num_labels", None),
            "head_size": getattr(head, "head_size", None),
            "multi_label": getattr(head, "multi_label", None),
        }

    # Collect label configs for all trained heads
    all_label_configs = {}
    for h in HEADS_TO_REPLACE:
        if h in LABEL_CONFIGS:
            all_label_configs[h] = dict(LABEL_CONFIGS[h])
        elif h in CLASSIFICATION_LABEL_CONFIGS:
            all_label_configs[h] = dict(CLASSIFICATION_LABEL_CONFIGS[h])

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "replaced_heads": HEADS_TO_REPLACE,
        "span_heads": [h for h in HEADS_TO_REPLACE if h in SPAN_HEADS],
        "classification_heads": [h for h in HEADS_TO_REPLACE if h in CLASSIFICATION_HEADS],
        "head_info": head_info,
        "label_configs": all_label_configs,
    }
    with open(checkpoint_dir / "globalpointer_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved: {checkpoint_dir.name}")


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

    # Update global HEADS_TO_REPLACE from config
    global HEADS_TO_REPLACE
    enabled_heads = heads_config.get("enabled", HEADS_TO_REPLACE)
    HEADS_TO_REPLACE = enabled_heads
    logger.info(f"Heads to train: {HEADS_TO_REPLACE}")

    # Training params
    learning_rate = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_epochs = training_config.get("num_epochs", 3)
    batch_size = training_config.get("batch_size", 16)
    max_length = data_config.get("max_length", 256)
    warmup_steps = training_config.get("warmup_steps", 500)
    save_steps = training_config.get("save_steps", 1000)
    eval_steps = training_config.get("eval_steps", 500)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    val_split = training_config.get("val_split", 0.1)
    num_workers = data_config.get("num_workers", 4)

    # Head architecture
    head_size = heads_config.get("architecture", {}).get("head_size", 64)
    dropout = heads_config.get("architecture", {}).get("dropout", 0.1)

    # Debug mode
    max_samples = args.max_samples
    debug_mode = args.debug
    if debug_mode:
        max_samples = max_samples or 500
        save_steps = 100
        logger.info(f"DEBUG MODE: max_samples={max_samples}, epochs={num_epochs}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Device: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    else:
        logger.info(f"Device: CPU")

    # Load model and replace heads
    log_section("MODEL")
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

    # Determine which head types we're training
    span_heads_active = [h for h in HEADS_TO_REPLACE if h in SPAN_HEADS]
    classification_heads_active = [h for h in HEADS_TO_REPLACE if h in CLASSIFICATION_HEADS]

    log_section("HEAD CONFIGURATION")
    logger.info(f"  Span heads: {span_heads_active or 'none'}")
    logger.info(f"  Classification heads: {classification_heads_active or 'none'}")

    # Get data paths for active heads
    data_paths = get_data_paths(data_config, data_root)

    log_section("DATA SOURCES")
    for head_name, paths in data_paths.items():
        logger.info(f"  {head_name}: {len(paths)} files")

    # Create datasets based on head types
    datasets = []

    # Span-based (NER) dataset
    if span_heads_active:
        span_paths = {h: data_paths[h] for h in span_heads_active if h in data_paths}
        if span_paths:
            span_dataset = MultiHeadSpanDataset(
                data_paths=span_paths,
                max_samples_per_head=max_samples,
            )
            datasets.append(span_dataset)
            logger.info(f"  Span dataset: {len(span_dataset)} samples")

    # Classification (Intent/Ingress V2) dataset
    if classification_heads_active:
        classification_paths = {h: data_paths[h] for h in classification_heads_active if h in data_paths}
        if classification_paths:
            classification_dataset = MultiHeadClassificationDataset(
                data_paths=classification_paths,
                label_configs=CLASSIFICATION_LABEL_CONFIGS,
                max_samples_per_head=max_samples,
            )
            datasets.append(classification_dataset)
            logger.info(f"  Classification dataset: {len(classification_dataset)} samples")

    # Combine datasets
    if len(datasets) == 0:
        raise ValueError("No data found for any head!")
    elif len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    logger.info(f"  Total: {len(full_dataset)} samples (train={train_size}, val={val_size})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create unified collator that handles both head types
    span_label_configs = {h: LABEL_CONFIGS[h] for h in span_heads_active} if span_heads_active else {}
    classification_label_configs = {h: CLASSIFICATION_LABEL_CONFIGS[h] for h in classification_heads_active} if classification_heads_active else {}

    collator = UnifiedMultiHeadCollator(
        tokenizer=tokenizer,
        span_label_configs=span_label_configs,
        classification_label_configs=classification_label_configs,
        max_length=max_length,
    )

    # Create dataloaders
    import platform
    effective_workers = 0 if platform.system() == "Windows" else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
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

    # Create optimizer (only trainable params)
    trainable_params = get_trainable_params(model)
    total_trainable = sum(p.numel() for p in trainable_params)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Create scheduler
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

    # Use adaptive warmup: max(configured steps, 5% of total steps, 10)
    # This ensures warmup isn't longer than training in debug mode
    adaptive_warmup = max(10, int(num_training_steps * 0.05))
    effective_warmup = min(warmup_steps, adaptive_warmup) if num_training_steps < warmup_steps else warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=effective_warmup,
        num_training_steps=num_training_steps,
    )

    logger.info(f"  Batches: train={len(train_loader)}, val={len(val_loader)}")
    logger.info(f"  Steps: {num_training_steps} total, {effective_warmup} warmup")

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
        tokenizer=tokenizer,
        save_steps=save_steps,
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        debug=debug_mode,
    )

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

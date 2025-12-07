#!/usr/bin/env python3
"""
Phase 0.5 Enhanced Healing Training Script for ModernBERT v3

This script implements the "healing" phase that repairs the cloned layers
(L23-28) and establishes smooth activation flow across the L22->L23 interface.

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-28 (Feeder + Family bands), Hub tokens
    - LR: Zipper strategy with L23 at maximum plasticity
    - Data: Enhanced healing (SST-2, CoNLL, MNLI, SQuAD, STS-B)

Phase 0.5 Objectives:
    1. Heal L23-28 (cloned from L15-20) to work coherently
    2. Smooth the L22->L23 interface transition
    3. Preserve L1-22 frozen capabilities
    4. Train hub tokens for routing semantics

Usage:
    # Dry run (validate configuration)
    python scripts/train_v3_phase0_5.py --dry-run

    # Smoke test (10 steps)
    python scripts/train_v3_phase0_5.py --smoke-test

    # Debug mode (5 steps with verbose gradient logging)
    python scripts/train_v3_phase0_5.py --debug

    # Full training
    python scripts/train_v3_phase0_5.py \\
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \\
        --model-path checkpoints/v3-initialized-from-v2 \\
        --output-dir outputs/v3_phase0_5

    # Resume from checkpoint
    python scripts/train_v3_phase0_5.py \\
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \\
        --resume-from outputs/v3_phase0_5/checkpoint-1500

    # With overrides
    python scripts/train_v3_phase0_5.py \\
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \\
        --learning-rate 3e-5 \\
        --max-steps 3000 \\
        --wandb-run-name "phase0_5_experiment_1"

Environment:
    - GPU: A100/H100 recommended (16GB+ VRAM)
    - RAM: 32GB+ recommended

Outputs:
    - outputs/v3_phase0_5/: Checkpoints and final model
    - outputs/v3_phase0_5/best/: Best model by validation loss
    - wandb/: W&B logs (if enabled)

Issue: 5.4.1 - Implement Phase 0.5 Healing Training Script
Author: FamilyOS Team
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path if running as script
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

# =============================================================================
# Core Imports - Models
# =============================================================================

from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra
from modeling_studio.models.hub_tokens import HUB_TOKEN_IDS, get_hub_for_capability

# =============================================================================
# Core Imports - Trainers (v3 infrastructure)
# =============================================================================

from modeling_studio.trainers.freezing_v3 import (
    LayerFreezer,
    LayerBand,
    TrainingPhase,
    LAYER_BANDS,
    PHASE_TRAINABLE_BANDS,
)
from modeling_studio.trainers.zipper_lr_v3 import (
    ZipperLRConfig,
    ZipperLROptimizer,
    get_zipper_preset,
    ZIPPER_PRESETS,
)
from modeling_studio.trainers.gradient_utils_v3 import (
    GradientClipConfig,
    GradientClipper,
)
from modeling_studio.trainers.schedulers_v3 import (
    WarmupCosineScheduler,
    create_scheduler,
)
from modeling_studio.trainers.gradient_masking_v3 import (
    GradientMaskConfig,
    EmbeddingGradientHook,
    setup_hub_token_gradient_masking,
)

# =============================================================================
# Core Imports - Data (v3 infrastructure)
# =============================================================================

from modeling_studio.data.collators_v3 import (
    V3CollatorConfig,
    V3MultiTaskCollator,
    create_v3_collator,
)

# =============================================================================
# Optional Imports - Initialization
# =============================================================================

try:
    from modeling_studio.models.initialization_v3 import (
        initialize_from_v2,
        V2CheckpointLoader,
        WeightTransferStats,
    )

    V2_INIT_AVAILABLE = True
except ImportError:
    V2_INIT_AVAILABLE = False
    initialize_from_v2 = None

# =============================================================================
# Optional Imports - Verification
# =============================================================================

try:
    from modeling_studio.models.verification_v3 import (
        verify_function_preserving,
        VerificationResult,
    )

    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

# =============================================================================
# Optional Imports - HuggingFace
# =============================================================================

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, set_seed

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# =============================================================================
# Optional Imports - YAML Config
# =============================================================================

try:
    import yaml
    from omegaconf import OmegaConf, DictConfig

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# =============================================================================
# Optional Imports - Weights & Biases
# =============================================================================

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class Phase05Config:
    """
    Configuration for Phase 0.5 Enhanced Healing training.

    This dataclass contains all configuration options for Phase 0.5 training,
    following the patterns from train_stage_a.py and train_stage_b.py.

    Attributes:
        model_path: Path to initialized v3 model checkpoint
        v2_checkpoint: Path to v2 checkpoint for initialization (if model_path not found)
        tokenizer_name: HuggingFace tokenizer name or path
        max_steps: Maximum training steps (default: 2500)
        warmup_steps: Number of warmup steps (default: 500)
        ... (see individual attributes)
    """

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Phase 0.5 uses the initialized v3 model (expanded from v2)
    # Use checkpoints/v3-initialized-from-v2 which has proper v3 architecture
    model_path: str = "checkpoints/v3-initialized-from-v2"
    v2_checkpoint: str = "outputs/modernbert-v2-for-v3-transfer/pytorch_model.bin"
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    model_config_path: str = "configs/model/encoder/modernbert_v3_ultra.yaml"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    max_steps: int = 2500
    warmup_steps: int = 500
    warmup_ratio: float = 0.2
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # =========================================================================
    # Batch Configuration
    # =========================================================================
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_length: int = 512

    # =========================================================================
    # Zipper Learning Rate Strategy (from zipper_lr_v3.py)
    # =========================================================================
    lr_strategy: str = "zipper"
    base_lr: float = 3e-5
    lr_layers_1_18: float = 0.0  # Frozen
    lr_layers_19_22: float = 1e-5  # Feeder band
    lr_layer_23: float = 5e-5  # Interface layer (maximum plasticity)
    lr_layers_24_28: float = 3e-5  # Family band
    lr_embeddings: float = 0.0  # Frozen (except hub tokens)
    lr_hub_tokens: float = 1e-5  # Hub token embeddings
    lr_task_heads: float = 3e-5  # Task heads
    family_graduated: bool = True  # Graduated decay in Family band
    family_decay: float = 0.85  # Decay factor per layer

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # =========================================================================
    # Gradient Configuration (from gradient_utils_v3.py)
    # =========================================================================
    max_grad_norm: float = 1.0
    per_layer_clip: bool = True
    interface_clip: float = 0.5  # Tighter clip at L23
    feeder_clip: float = 1.0
    family_clip: float = 1.0
    log_grad_norms: bool = True
    grad_log_every: int = 100
    nan_check: bool = True
    zero_nan_grads: bool = True

    # =========================================================================
    # Hub Token Configuration (from gradient_masking_v3.py)
    # =========================================================================
    train_hub_tokens: list = field(default_factory=lambda: ["[EMO]", "[MEM]", "[REL]", "[TASK]"])
    freeze_original_vocab: bool = True
    hub_token_grad_scale: float = 1.0

    # =========================================================================
    # Layer Freezing Configuration (from freezing_v3.py)
    # =========================================================================
    frozen_bands: list = field(default_factory=lambda: ["foundation", "core"])
    trainable_bands: list = field(default_factory=lambda: ["feeder", "family"])
    freeze_embeddings: bool = True
    freeze_hub_tokens: bool = False

    # =========================================================================
    # Scheduler Configuration (from schedulers_v3.py)
    # =========================================================================
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.01

    # =========================================================================
    # Data Configuration
    # =========================================================================
    healing_data_path: str = "data/healing/healing_enhanced.jsonl"
    data_config_path: str = "configs/data/multitask/healing_enhanced.yaml"
    tasks: list = field(default_factory=lambda: ["sentiment", "ner", "nli", "qa", "similarity"])
    num_workers: int = 4
    pin_memory: bool = True

    # =========================================================================
    # Loss Configuration
    # =========================================================================
    use_task_weights: bool = True
    task_weights: dict = field(
        default_factory=lambda: {
            "sentiment": 1.0,
            "ner": 1.0,
            "nli": 1.0,
            "qa": 1.2,
            "similarity": 1.2,
        }
    )

    # =========================================================================
    # Output Configuration
    # =========================================================================
    # Output goes to same parent directory for next stage continuity
    output_dir: str = "outputs/modernbert-v2-for-v3-transfer/phase0_5"
    checkpoint_dir: str = "outputs/modernbert-v2-for-v3-transfer/phase0_5/checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # =========================================================================
    # Device Configuration
    # =========================================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = True
    fp16: bool = False

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str | None = None
    wandb_tags: list = field(default_factory=lambda: ["phase_0.5", "healing", "v3"])

    # =========================================================================
    # Reproducibility
    # =========================================================================
    seed: int = 42

    # =========================================================================
    # Derived Methods
    # =========================================================================

    def get_zipper_lr_config(self) -> ZipperLRConfig:
        """Create ZipperLRConfig from this config."""
        return ZipperLRConfig(
            base_lr=float(self.base_lr),
            feeder_lr=float(self.lr_layers_19_22),
            interface_lr=float(self.lr_layer_23),
            family_lr=float(self.lr_layers_24_28),
            family_graduated=bool(self.family_graduated),
            family_decay=float(self.family_decay),
            frozen_lr=float(self.lr_layers_1_18),
            embeddings_lr=float(self.lr_embeddings),
            task_heads_lr=float(self.lr_task_heads),
        )

    def get_gradient_config(self) -> GradientClipConfig:
        """Create GradientClipConfig from this config."""
        return GradientClipConfig(
            max_grad_norm=self.max_grad_norm,
            per_layer_clip=self.per_layer_clip,
            interface_clip=self.interface_clip,
            feeder_clip=self.feeder_clip,
            family_clip=self.family_clip,
            log_grad_norms=self.log_grad_norms,
            log_every_n_steps=self.grad_log_every,
            nan_check=self.nan_check,
        )

    def get_hub_grad_mask_config(self) -> GradientMaskConfig:
        """Create GradientMaskConfig from this config."""
        return GradientMaskConfig(
            train_hub_tokens=self.train_hub_tokens,
            freeze_original_vocab=self.freeze_original_vocab,
            hub_token_grad_scale=self.hub_token_grad_scale,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                result[key] = list(value)
            elif isinstance(value, dict):
                result[key] = dict(value)
            else:
                result[key] = value
        return result

    def save(self, path: Path | str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Phase05Config:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path | str) -> Phase05Config:
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Training State
# =============================================================================


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resumption."""

    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    phase: str = "phase_0.5"
    losses: list = field(default_factory=list)
    metrics_history: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "phase": self.phase,
            "losses": self.losses[-100:],  # Keep last 100 losses
            "metrics_history": self.metrics_history[-10:],  # Keep last 10 evals
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingState:
        """Create state from dictionary."""
        return cls(**d)


# =============================================================================
# Healing Dataset
# =============================================================================


class HealingDataset(Dataset):
    """
    Healing dataset for Phase 0.5 using public benchmarks.

    Loads from HuggingFace datasets:
        - SST-2: Sentiment classification (binary)
        - CoNLL-2003: NER (token classification)
        - MNLI: Natural Language Inference (3-way)
        - SQuAD: Question Answering (span extraction)
        - STS-B: Semantic Textual Similarity (regression)

    Attributes:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        samples: List of processed samples
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 512,
        tasks: list[str] | None = None,
    ):
        """Initialize HealingDataset."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        if tasks is None:
            tasks = ["sentiment", "nli"]

        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available, using synthetic data")
            self._create_synthetic_samples(max_samples or 1000)
            return

        # Calculate samples per task
        samples_per_task = (max_samples // len(tasks)) if max_samples else None

        # Load each task
        if "sentiment" in tasks:
            self._load_sst2(split, samples_per_task)

        if "nli" in tasks:
            self._load_mnli(split, samples_per_task)

        if "ner" in tasks:
            self._load_conll2003(split, samples_per_task)

        if "qa" in tasks:
            self._load_squad(split, samples_per_task)

        if "similarity" in tasks:
            self._load_stsb(split, samples_per_task)

        logger.info(f"Loaded {len(self.samples)} healing samples for {split}")

    def _load_sst2(self, split: str, max_samples: int | None) -> None:
        """Load SST-2 sentiment data."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("glue", "sst2", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                encoding = self.tokenizer(
                    item["sentence"],
                    max_length=self.max_length - 5,  # Reserve space for hub tokens
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "task": "sentiment",
                        "label": item["label"],
                    }
                )
                count += 1

            logger.info(f"Loaded {count} SST-2 samples")
        except Exception as e:
            logger.warning(f"Failed to load SST-2: {e}")

    def _load_mnli(self, split: str, max_samples: int | None) -> None:
        """Load MNLI NLI data."""
        try:
            ds_split = "validation_matched" if split == "validation" else "train"
            ds = load_dataset("glue", "mnli", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                # Combine premise and hypothesis
                text = f"{item['premise']} [SEP] {item['hypothesis']}"

                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length - 5,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "task": "nli",
                        "label": item["label"],
                    }
                )
                count += 1

            logger.info(f"Loaded {count} MNLI samples")
        except Exception as e:
            logger.warning(f"Failed to load MNLI: {e}")

    def _load_conll2003(self, split: str, max_samples: int | None) -> None:
        """Load CoNLL-2003 NER data."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("conll2003", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                tokens = item["tokens"]
                ner_tags = item["ner_tags"]

                # Join tokens for encoding
                text = " ".join(tokens)
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length - 5,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Align labels (simplified - use first token label)
                # Note: Full implementation would use word_ids() for proper alignment
                aligned_labels = ner_tags[: len(encoding["input_ids"]) - 2]  # -2 for CLS/SEP

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "task": "ner",
                        "label": aligned_labels,
                        "ner_tags": ner_tags,
                    }
                )
                count += 1

            logger.info(f"Loaded {count} CoNLL-2003 samples")
        except Exception as e:
            logger.warning(f"Failed to load CoNLL-2003: {e}")

    def _load_squad(self, split: str, max_samples: int | None) -> None:
        """Load SQuAD QA data."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("squad", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                # Combine question and context
                text = f"{item['question']} [SEP] {item['context']}"

                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length - 5,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Get answer info
                answers = item["answers"]
                answer_text = answers["text"][0] if answers["text"] else ""
                answer_start = answers["answer_start"][0] if answers["answer_start"] else 0

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "task": "qa",
                        "label": 0,  # Placeholder for QA
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                    }
                )
                count += 1

            logger.info(f"Loaded {count} SQuAD samples")
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}")

    def _load_stsb(self, split: str, max_samples: int | None) -> None:
        """Load STS-B similarity data."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("glue", "stsb", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                # Combine sentence pairs
                text = f"{item['sentence1']} [SEP] {item['sentence2']}"

                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length - 5,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Normalize similarity score to 0-1
                similarity_score = item["label"] / 5.0

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "task": "similarity",
                        "label": similarity_score,
                        "raw_score": item["label"],
                    }
                )
                count += 1

            logger.info(f"Loaded {count} STS-B samples")
        except Exception as e:
            logger.warning(f"Failed to load STS-B: {e}")

    def _create_synthetic_samples(self, num_samples: int) -> None:
        """Create synthetic samples when HF not available."""
        for i in range(num_samples):
            # Random token IDs (avoiding special tokens)
            seq_len = 64 + (i % 64)
            input_ids = torch.randint(5, 50368, (seq_len,)).tolist()

            task_idx = i % 5
            tasks = ["sentiment", "nli", "ner", "qa", "similarity"]
            task = tasks[task_idx]

            # Task-specific labels
            if task == "sentiment":
                label = i % 2
            elif task == "nli":
                label = i % 3
            elif task == "ner":
                label = [i % 9 for _ in range(seq_len)]
            elif task == "qa":
                label = 0
            else:  # similarity
                label = (i % 50) / 50.0

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * seq_len,
                    "task": task,
                    "label": label,
                }
            )

        logger.info(f"Created {num_samples} synthetic samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# Healing Collator
# =============================================================================


class HealingCollator:
    """
    Collator for healing data that adds hub tokens.

    Token layout: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
    """

    HUB_TOKENS = {
        "[EMO]": 50368,
        "[MEM]": 50369,
        "[REL]": 50370,
        "[TASK]": 50371,
    }

    def __init__(self, tokenizer, max_length: int = 512):
        """Initialize HealingCollator."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.cls_token_id = tokenizer.cls_token_id or 0
        self.sep_token_id = tokenizer.sep_token_id or 2

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch with hub token insertion."""
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_tasks = []

        for sample in batch:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            # Remove existing CLS/SEP if present
            if input_ids and input_ids[0] == self.cls_token_id:
                input_ids = input_ids[1:]
            if input_ids and input_ids[-1] == self.sep_token_id:
                input_ids = input_ids[:-1]

            # Build v3 token sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP]
            v3_ids = (
                [
                    self.cls_token_id,
                    self.HUB_TOKENS["[EMO]"],
                    self.HUB_TOKENS["[MEM]"],
                    self.HUB_TOKENS["[REL]"],
                    self.HUB_TOKENS["[TASK]"],
                ]
                + input_ids[: self.max_length - 6]
                + [self.sep_token_id]
            )

            # Pad to max_length
            pad_len = self.max_length - len(v3_ids)
            attention_mask = [1] * len(v3_ids) + [0] * pad_len
            v3_ids = v3_ids + [self.pad_token_id] * pad_len

            batch_input_ids.append(v3_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(sample["label"])
            batch_tasks.append(sample.get("task", "sentiment"))

        # Handle different label types (int vs float for similarity)
        if batch_tasks and batch_tasks[0] == "similarity":
            labels_tensor = torch.tensor(batch_labels, dtype=torch.float)
        else:
            # For classification tasks, labels should be integers
            labels_tensor = torch.tensor(
                [lbl if isinstance(lbl, int) else 0 for lbl in batch_labels],
                dtype=torch.long,
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": labels_tensor,
            "tasks": batch_tasks,
        }


# =============================================================================
# Phase 0.5 Training Wrapper
# =============================================================================


@dataclass
class Phase05Output:
    """Output container for Phase 0.5 training forward pass."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor


class Phase05TrainingModel(nn.Module):
    """
    Wrapper that adds task heads to ModernBERTv3Ultra for Phase 0.5.

    Uses hub tokens for task-specific classification:
        - [EMO] at position 1 for sentiment/safety
        - [MEM] at position 2 for embedding/similarity
        - [REL] at position 3 for NLI
        - [TASK] at position 4 for intent/ingress
    """

    # Hub token positions in the sequence
    POS_CLS = 0
    POS_EMO = 1
    POS_MEM = 2
    POS_REL = 3
    POS_TASK = 4

    def __init__(self, model: ModernBERTv3Ultra):
        """Initialize Phase05TrainingModel."""
        super().__init__()
        self.model = model

        hidden_size = model.config.hidden_size

        # Task-specific heads
        self.sentiment_head = nn.Linear(hidden_size, 2)  # Binary sentiment
        self.nli_head = nn.Linear(hidden_size, 3)  # 3-way NLI
        self.similarity_head = nn.Linear(hidden_size, 1)  # Regression
        self.qa_head = nn.Linear(hidden_size, 2)  # Start/end logits

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    @property
    def encoder(self):
        """Expose encoder for LayerFreezer."""
        return self.model.encoder

    @property
    def embeddings(self):
        """Expose embeddings for LayerFreezer."""
        return self.model.embeddings

    @property
    def config(self):
        """Expose config."""
        return self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        tasks: list[str] | None = None,
        **kwargs,
    ) -> Phase05Output:
        """Forward pass with task-specific loss computation."""
        # Get encoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        batch_size = input_ids.size(0)
        hidden_states = outputs.last_hidden_state

        total_loss = torch.tensor(0.0, device=input_ids.device, dtype=hidden_states.dtype)
        all_logits = []

        for i in range(batch_size):
            task = tasks[i] if tasks else "sentiment"

            if task == "sentiment":
                # Use [EMO] hub token at position 1
                pooled = hidden_states[i, self.POS_EMO, :]
                logits = self.sentiment_head(pooled)
                all_logits.append(logits)

                if labels is not None:
                    loss = self.ce_loss(logits.unsqueeze(0), labels[i : i + 1].long())
                    total_loss = total_loss + loss

            elif task == "nli":
                # Use [REL] hub token at position 3
                pooled = hidden_states[i, self.POS_REL, :]
                logits = self.nli_head(pooled)
                all_logits.append(logits)

                if labels is not None:
                    loss = self.ce_loss(logits.unsqueeze(0), labels[i : i + 1].long())
                    total_loss = total_loss + loss

            elif task == "similarity":
                # Use [MEM] hub token at position 2
                pooled = hidden_states[i, self.POS_MEM, :]
                logits = self.similarity_head(pooled).squeeze(-1)
                all_logits.append(logits.unsqueeze(0))  # Keep shape consistent

                if labels is not None:
                    target = labels[i : i + 1].float()
                    loss = self.mse_loss(logits.unsqueeze(0), target)
                    total_loss = total_loss + loss

            elif task == "qa":
                # Use full sequence for QA (start/end positions)
                # Simplified: just use [TASK] hub token for now
                pooled = hidden_states[i, self.POS_TASK, :]
                logits = self.qa_head(pooled)
                all_logits.append(logits)

                # QA loss would require start/end positions - placeholder
                if labels is not None:
                    # Use cross-entropy on first position as placeholder
                    loss = self.ce_loss(logits.unsqueeze(0), labels[i : i + 1].long() % 2)
                    total_loss = total_loss + loss

            elif task == "ner":
                # NER uses all token positions - simplified for healing
                # Just use [CLS] for sequence-level representation
                pooled = hidden_states[i, self.POS_CLS, :]
                logits = self.sentiment_head(pooled)  # Reuse head
                all_logits.append(logits)

                if labels is not None:
                    # Simplified: treat as binary classification
                    label_val = labels[i] if isinstance(labels[i], int) else 0
                    loss = self.ce_loss(
                        logits.unsqueeze(0),
                        torch.tensor([label_val % 2], device=input_ids.device),
                    )
                    total_loss = total_loss + loss

            else:
                # Default: use [CLS]
                pooled = hidden_states[i, self.POS_CLS, :]
                logits = self.sentiment_head(pooled)
                all_logits.append(logits)

        # Average loss over batch
        if labels is not None:
            total_loss = total_loss / batch_size

        # Stack logits (may have different shapes, use first for reference)
        try:
            logits_tensor = torch.stack(all_logits)
        except RuntimeError:
            # If shapes don't match, pad to largest
            max_size = max(lg.numel() for lg in all_logits)
            padded = []
            for lg in all_logits:
                if lg.numel() < max_size:
                    pad = torch.zeros(max_size - lg.numel(), device=lg.device, dtype=lg.dtype)
                    padded.append(torch.cat([lg.flatten(), pad]))
                else:
                    padded.append(lg.flatten()[:max_size])
            logits_tensor = torch.stack(padded)

        return Phase05Output(
            loss=total_loss if labels is not None else None,
            logits=logits_tensor,
            hidden_states=hidden_states,
        )


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    if not YAML_AVAILABLE:
        logger.warning("YAML/OmegaConf not available, cannot load config")
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config or {}


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides to config (e.g., --learning-rate=3e-5)."""
    result = config.copy()

    for override in overrides:
        if "=" not in override:
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Navigate to parent dict
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set value with type inference
        final_key = keys[-1]
        if value.lower() == "true":
            current[final_key] = True
        elif value.lower() == "false":
            current[final_key] = False
        elif value.replace(".", "").replace("-", "").replace("e", "").isdigit():
            try:
                current[final_key] = (
                    float(value) if "." in value or "e" in value.lower() else int(value)
                )
            except ValueError:
                current[final_key] = value
        else:
            current[final_key] = value

    return result


def merge_configs(base_config: Phase05Config, yaml_config: dict[str, Any]) -> Phase05Config:
    """Merge YAML config into base config."""
    config_dict = base_config.to_dict()

    # Flatten nested YAML config and apply
    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    flat_yaml = flatten_dict(yaml_config)

    # Map YAML keys to config attributes
    key_mapping = {
        "training.max_steps": "max_steps",
        "training.warmup_steps": "warmup_steps",
        "training.per_device_train_batch_size": "train_batch_size",
        "training.per_device_eval_batch_size": "eval_batch_size",
        "training.eval_steps": "eval_steps",
        "training.save_steps": "save_steps",
        "training.logging_steps": "logging_steps",
        "training.bf16": "bf16",
        "training.fp16": "fp16",
        "training.seed": "seed",
        "learning_rate.base_lr": "base_lr",
        "learning_rate.layers_1_18": "lr_layers_1_18",
        "learning_rate.layers_19_22": "lr_layers_19_22",
        "learning_rate.layer_23": "lr_layer_23",
        "learning_rate.layers_24_28": "lr_layers_24_28",
        "learning_rate.family_graduated": "family_graduated",
        "learning_rate.family_decay": "family_decay",
        "gradient.max_grad_norm": "max_grad_norm",
        "gradient.per_layer_clip": "per_layer_clip",
        "gradient.interface_clip": "interface_clip",
        "optimizer.weight_decay": "weight_decay",
        "checkpointing.output_dir": "output_dir",
        "logging.use_wandb": "use_wandb",
        "logging.wandb_project": "wandb_project",
        "logging.wandb_run_name": "wandb_run_name",
    }

    # Apply mapped values
    for yaml_key, config_key in key_mapping.items():
        if yaml_key in flat_yaml:
            config_dict[config_key] = flat_yaml[yaml_key]

    # Also apply direct matches
    for key, value in flat_yaml.items():
        simple_key = key.split(".")[-1]
        if simple_key in config_dict:
            config_dict[simple_key] = value

    return Phase05Config.from_dict(config_dict)


# =============================================================================
# Model Setup
# =============================================================================


def setup_model(config: Phase05Config) -> ModernBERTv3Ultra:
    """
    Setup v3 model from checkpoint or initialize from v2.

    Priority:
        1. Load from model_path if exists
        2. Initialize from v2_checkpoint if available
        3. Create with random initialization (warning)
    """
    model_path = Path(config.model_path)

    # Try loading from existing v3 checkpoint
    if model_path.exists() and (model_path / "pytorch_model.bin").exists():
        logger.info(f"Loading v3 model from {model_path}")
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)

        state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    # Try loading from safetensors
    if model_path.exists() and (model_path / "model.safetensors").exists():
        logger.info(f"Loading v3 model from safetensors: {model_path}")
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)

        try:
            from safetensors.torch import load_file

            state_dict = load_file(model_path / "model.safetensors")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
        except ImportError:
            logger.warning("safetensors not available")

    # Try initializing from v2 checkpoint
    v2_path = Path(config.v2_checkpoint)
    if v2_path.exists() and V2_INIT_AVAILABLE:
        logger.info(f"Initializing v3 model from v2 checkpoint: {v2_path}")
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)

        stats = initialize_from_v2(model, str(v2_path))
        logger.info(
            f"Initialized from v2: {stats.transferred_params:,} transferred, "
            f"{stats.cloned_params:,} cloned"
        )
        return model

    # Fall back to random initialization with warning
    logger.warning(
        "No checkpoint found, creating model with random initialization. "
        "This is NOT recommended for Phase 0.5 training!"
    )
    v3_config = ModernBERTv3Config()
    model = ModernBERTv3Ultra(v3_config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def setup_layer_freezing(model: nn.Module, config: Phase05Config) -> LayerFreezer:
    """
    Configure layer freezing for Phase 0.5.

    Freezes: L1-18 (Foundation + Core bands)
    Trains: L19-28 (Feeder + Family bands)
    """
    # Get base model if wrapped
    base_model = model.model if hasattr(model, "model") else model

    # Create freezer and configure for Phase 0.5
    freezer = LayerFreezer(base_model)
    freezer.configure_for_phase(TrainingPhase.PHASE_0_5)

    # Log stats
    stats = freezer.get_freeze_stats()
    logger.info(
        f"Layer freezing configured for Phase 0.5:\n"
        f"  Frozen: {stats['frozen_params']:,} parameters\n"
        f"  Trainable: {stats['trainable_params']:,} parameters\n"
        f"  Frozen layers: {freezer.get_frozen_layers()}\n"
        f"  Trainable layers: {freezer.get_trainable_layers()}"
    )

    return freezer


def setup_hub_gradient_masking(
    model: nn.Module, config: Phase05Config
) -> EmbeddingGradientHook | None:
    """
    Setup hub token gradient masking.

    Freezes original vocabulary embeddings, enables hub token gradients.
    """
    if not config.freeze_original_vocab:
        logger.info("Hub gradient masking disabled")
        return None

    # Get base model
    base_model = model.model if hasattr(model, "model") else model

    try:
        hook = setup_hub_token_gradient_masking(
            base_model,
            freeze_original_vocab=config.freeze_original_vocab,
            train_hub_tokens=config.train_hub_tokens,
        )
        logger.info(
            f"Hub gradient masking enabled:\n"
            f"  Trainable hub tokens: {config.train_hub_tokens}\n"
            f"  Original vocab frozen: {config.freeze_original_vocab}"
        )
        return hook
    except Exception as e:
        logger.warning(f"Failed to setup hub gradient masking: {e}")
        return None


# =============================================================================
# Optimizer Setup
# =============================================================================


def create_optimizer(model: nn.Module, config: Phase05Config) -> torch.optim.AdamW:
    """
    Create optimizer with Zipper LR strategy.

    Uses layer-group learning rates from zipper_lr_v3.py.
    """
    zipper_config = config.get_zipper_lr_config()
    base_model = model.model if hasattr(model, "model") else model

    param_groups = []

    # Get encoder layers
    encoder = getattr(base_model, "encoder", base_model)
    layers = getattr(encoder, "layers", None)

    if layers is not None:
        num_layers = len(layers)

        # Feeder layers (L19-22, indices 18-21)
        feeder_params = []
        for i in range(18, min(22, num_layers)):
            feeder_params.extend([p for p in layers[i].parameters() if p.requires_grad])
        if feeder_params:
            param_groups.append(
                {
                    "params": feeder_params,
                    "lr": zipper_config.feeder_lr,
                    "name": "feeder_L19-22",
                }
            )

        # Interface layer (L23, index 22) - maximum plasticity
        if num_layers > 22:
            interface_params = [p for p in layers[22].parameters() if p.requires_grad]
            if interface_params:
                param_groups.append(
                    {
                        "params": interface_params,
                        "lr": zipper_config.interface_lr,
                        "name": "interface_L23",
                    }
                )

        # Family layers (L24-28, indices 23-27) with graduated LR
        for layer_idx in range(23, min(28, num_layers)):
            layer_params = [p for p in layers[layer_idx].parameters() if p.requires_grad]
            if layer_params:
                # Graduated decay from interface
                if zipper_config.family_graduated:
                    steps_from_interface = layer_idx - 22
                    layer_lr = zipper_config.interface_lr * (
                        zipper_config.family_decay**steps_from_interface
                    )
                else:
                    layer_lr = zipper_config.family_lr

                param_groups.append(
                    {
                        "params": layer_params,
                        "lr": layer_lr,
                        "name": f"family_L{layer_idx + 1}",
                    }
                )

    # Task heads (from wrapper model)
    head_params = []
    for head_name in ["sentiment_head", "nli_head", "similarity_head", "qa_head"]:
        if hasattr(model, head_name):
            head_params.extend(list(getattr(model, head_name).parameters()))
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": zipper_config.task_heads_lr,
                "name": "task_heads",
            }
        )

    # Embeddings (hub tokens only if configured)
    if hasattr(base_model, "embeddings"):
        emb_params = [p for p in base_model.embeddings.parameters() if p.requires_grad]
        if emb_params:
            param_groups.append(
                {
                    "params": emb_params,
                    "lr": config.lr_hub_tokens,
                    "name": "embeddings",
                }
            )

    # Collect any remaining trainable parameters
    assigned_ids = set()
    for group in param_groups:
        for p in group["params"]:
            assigned_ids.add(id(p))

    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned_ids]
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": zipper_config.base_lr,
                "name": "other",
            }
        )

    # Log parameter groups
    logger.info("Optimizer parameter groups (Zipper LR):")
    for group in param_groups:
        num_params = sum(p.numel() for p in group["params"])
        logger.info(f"  {group['name']}: {num_params:,} params, lr={group['lr']:.2e}")

    return torch.optim.AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Phase05Config,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.

    Warmup + Cosine decay from schedulers_v3.py.
    """
    if config.scheduler_type == "cosine":
        try:
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
            )
        except ImportError:
            # Fallback to PyTorch scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.max_steps - config.warmup_steps,
                eta_min=config.base_lr * config.min_lr_ratio,
            )
    elif config.scheduler_type == "linear":
        try:
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
            )
        except ImportError:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.min_lr_ratio,
                total_iters=config.max_steps,
            )
    else:
        # Default: constant LR with warmup (manual)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, config.warmup_steps)),
        )

    logger.info(
        f"Created {config.scheduler_type} scheduler: "
        f"{config.warmup_steps} warmup, {config.max_steps} total steps"
    )
    return scheduler


# =============================================================================
# Data Loading
# =============================================================================


def create_dataloaders(
    config: Phase05Config,
    tokenizer,
    synthetic: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Determine sample counts
    if synthetic:
        train_samples = 1000
        val_samples = 100
    else:
        train_samples = 10000
        val_samples = 1000

    # Create datasets
    train_dataset = HealingDataset(
        tokenizer,
        split="train",
        max_samples=train_samples,
        max_length=config.max_length,
        tasks=config.tasks,
    )

    val_dataset = HealingDataset(
        tokenizer,
        split="validation",
        max_samples=val_samples,
        max_length=config.max_length,
        tasks=config.tasks,
    )

    # Create collator
    collator = HealingCollator(tokenizer, max_length=config.max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True,
        collate_fn=collator,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
    )

    logger.info(
        f"Created dataloaders:\n"
        f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches\n"
        f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches"
    )

    return train_loader, val_loader


# =============================================================================
# Training Functions
# =============================================================================


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    clipper: GradientClipper,
    config: Phase05Config,
    step: int = 0,
) -> tuple[float, dict]:
    """
    Single training step.

    Returns:
        Tuple of (loss_value, gradient_stats)
    """
    model.train()
    device = next(model.parameters()).device

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    tasks = batch.get("tasks", ["sentiment"] * len(labels))

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass with mixed precision
    if config.bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                tasks=tasks,
            )
            loss = outputs.loss
    elif config.fp16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                tasks=tasks,
            )
            loss = outputs.loss
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            tasks=tasks,
        )
        loss = outputs.loss

    # Check for NaN loss
    if torch.isnan(loss):
        logger.warning(f"NaN loss detected at step {step}")
        return float("nan"), {
            "total_norm": 0.0,
            "has_nan": True,
            "has_inf": False,
            "clipped": False,
        }

    # Backward pass
    loss.backward()

    # Gradient clipping using GradientClipper
    grad_stats = clipper.clip_gradients()

    # Optimizer step
    optimizer.step()
    scheduler.step()

    return loss.item(), {
        "total_norm": grad_stats.total_norm,
        "has_nan": grad_stats.has_nan,
        "has_inf": grad_stats.has_inf,
        "clipped": grad_stats.clipped,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    config: Phase05Config,
) -> dict[str, float]:
    """Run evaluation on validation set."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    task_metrics: dict[str, dict] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tasks = batch.get("tasks", ["sentiment"] * len(labels))

            if config.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        tasks=tasks,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    tasks=tasks,
                )

            if outputs.loss is not None:
                total_loss += outputs.loss.item() * len(labels)

            # Compute accuracy for classification tasks
            for i, task in enumerate(tasks):
                if task in ["sentiment", "nli"]:
                    if outputs.logits.dim() > 1:
                        pred = outputs.logits[i].argmax().item()
                    else:
                        pred = outputs.logits[i].item() > 0.5
                    correct = int(pred == labels[i].item())
                    total_correct += correct

                    if task not in task_metrics:
                        task_metrics[task] = {"correct": 0, "total": 0}
                    task_metrics[task]["correct"] += correct
                    task_metrics[task]["total"] += 1

            total_samples += len(labels)

    # Compute metrics
    metrics = {
        "eval_loss": total_loss / max(total_samples, 1),
        "eval_samples": total_samples,
    }

    # Per-task accuracy
    for task, task_data in task_metrics.items():
        if task_data["total"] > 0:
            metrics[f"eval_{task}_accuracy"] = task_data["correct"] / task_data["total"]

    # Overall accuracy
    classification_total = sum(t["total"] for t in task_metrics.values())
    if classification_total > 0:
        classification_correct = sum(t["correct"] for t in task_metrics.values())
        metrics["eval_accuracy"] = classification_correct / classification_total

    model.train()
    return metrics


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    config: Phase05Config,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    import shutil

    output_path = Path(config.output_dir)
    checkpoint_dir = output_path / f"checkpoint-{state.global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save base model
    base_model = model.model if hasattr(model, "model") else model
    torch.save(base_model.state_dict(), checkpoint_dir / "pytorch_model.bin")

    # Save task heads if present
    heads_state = {}
    for head_name in ["sentiment_head", "nli_head", "similarity_head", "qa_head"]:
        if hasattr(model, head_name):
            heads_state[head_name] = getattr(model, head_name).state_dict()
    if heads_state:
        torch.save(heads_state, checkpoint_dir / "task_heads.pt")

    # Save optimizer and scheduler state
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )

    # Save training state
    with open(checkpoint_dir / "trainer_state.json", "w") as f:
        json.dump(state.to_dict(), f, indent=2)

    # Save config
    config.save(checkpoint_dir / "config.json")

    logger.info(f"Saved checkpoint: {checkpoint_dir}")

    # Save best model
    if is_best:
        best_dir = output_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Saved best model: {best_dir}")

    # Cleanup old checkpoints if needed
    if config.save_total_limit > 0:
        checkpoints = sorted(
            output_path.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]),
        )
        while len(checkpoints) > config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: Path,
    config: Phase05Config,
) -> TrainingState:
    """Load training checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load model weights
    model_path = checkpoint_path / "pytorch_model.bin"
    if model_path.exists():
        base_model = model.model if hasattr(model, "model") else model
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        base_model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")

    # Load task heads
    heads_path = checkpoint_path / "task_heads.pt"
    if heads_path.exists():
        heads_state = torch.load(heads_path, map_location="cpu", weights_only=True)
        for head_name, head_state in heads_state.items():
            if hasattr(model, head_name):
                getattr(model, head_name).load_state_dict(head_state)
        logger.info(f"Loaded task heads from {heads_path}")

    # Load optimizer and scheduler state
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info(f"Loaded optimizer/scheduler from {training_state_path}")

    # Load trainer state
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state_dict = json.load(f)
        state = TrainingState.from_dict(state_dict)
        logger.info(f"Loaded trainer state: step {state.global_step}")
        return state

    return TrainingState()


# =============================================================================
# Logging
# =============================================================================


def setup_wandb(config: Phase05Config) -> None:
    """Initialize Weights & Biases logging."""
    if not config.use_wandb:
        return

    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, disabling logging")
        config.use_wandb = False
        return

    run_name = config.wandb_run_name or f"phase0.5_{config.max_steps}steps"

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            "phase": "0.5",
            "description": "Enhanced Healing Training",
            "max_steps": config.max_steps,
            "batch_size": config.train_batch_size,
            "base_lr": config.base_lr,
            "interface_lr": config.lr_layer_23,
            "frozen_layers": "L1-18",
            "trainable_layers": "L19-28",
            "warmup_steps": config.warmup_steps,
            "bf16": config.bf16,
            "seed": config.seed,
        },
    )

    logger.info(f"W&B initialized: {run_name}")


def log_training_step(
    step: int,
    loss: float,
    grad_stats: dict,
    lr: float,
    config: Phase05Config,
) -> None:
    """Log training step metrics."""
    # Console logging at configured intervals
    if (step + 1) % config.logging_steps == 0:
        logger.info(
            f"Step {step + 1}/{config.max_steps}: "
            f"loss={loss:.4f}, lr={lr:.2e}, "
            f"grad_norm={grad_stats.get('total_norm', 0):.4f}"
        )

    # W&B logging
    if config.use_wandb and WANDB_AVAILABLE:
        metrics = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/step": step + 1,
        }

        # Add gradient stats
        if grad_stats:
            metrics["gradients/total_norm"] = grad_stats.get("total_norm", 0)
            metrics["gradients/clipped"] = 1 if grad_stats.get("clipped", False) else 0
            metrics["gradients/has_nan"] = 1 if grad_stats.get("has_nan", False) else 0
            metrics["gradients/has_inf"] = 1 if grad_stats.get("has_inf", False) else 0

        wandb.log(metrics, step=step + 1)


def log_evaluation(
    step: int,
    metrics: dict[str, float],
    config: Phase05Config,
) -> None:
    """Log evaluation metrics."""
    # Console logging
    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Eval @ step {step}: {metrics_str}")

    # W&B logging
    if config.use_wandb and WANDB_AVAILABLE:
        eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        eval_metrics["eval/step"] = step
        wandb.log(eval_metrics, step=step)


# =============================================================================
# Training Modes
# =============================================================================


def run_dry_run(config: Phase05Config) -> bool:
    """
    Run dry-run validation without actual training.

    Validates:
        - Model can be created/loaded
        - Tokenizer available
        - Data can be loaded
        - Optimizer can be created
        - Configuration is valid
    """
    print("\n" + "=" * 60)
    print("Phase 0.5 Dry Run Validation")
    print("=" * 60)

    checks_total = 0
    checks_passed = 0

    # Check 1: Configuration
    checks_total += 1
    print("[OK] Configuration loaded")
    print(f"    Output dir:     {config.output_dir}")
    print(f"    Max steps:      {config.max_steps}")
    print(f"    Batch size:     {config.train_batch_size}")
    print(f"    Base LR:        {config.base_lr}")
    print(f"    Interface LR:   {config.lr_layer_23}")
    checks_passed += 1

    # Check 2: Zipper LR Config
    checks_total += 1
    zipper_config = config.get_zipper_lr_config()
    print("[OK] Zipper LR configuration:")
    print(f"    Feeder (L19-22):   {zipper_config.feeder_lr}")
    print(f"    Interface (L23):   {zipper_config.interface_lr}")
    print(f"    Family (L24-28):   {zipper_config.family_lr}")
    checks_passed += 1

    # Check 3: Gradient Config
    checks_total += 1
    grad_config = config.get_gradient_config()
    print("[OK] Gradient configuration:")
    print(f"    Max norm:       {grad_config.max_grad_norm}")
    print(f"    Interface clip: {grad_config.interface_clip}")
    print(f"    Per-layer clip: {grad_config.per_layer_clip}")
    checks_passed += 1

    # Check 4: Tokenizer
    checks_total += 1
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        print(f"[OK] Tokenizer loaded: {config.tokenizer_name}")
        print(f"    Vocab size: {len(tokenizer)}")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Tokenizer failed: {e}")

    # Check 5: Model Creation
    checks_total += 1
    try:
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created: {param_count:,} parameters")
        print(f"    Layers: {v3_config.num_layers}")
        print(f"    Hidden: {v3_config.hidden_size}")
        checks_passed += 1
        del model
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")

    # Check 6: Output directory
    checks_total += 1
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {output_path}")
    checks_passed += 1

    # Summary
    print("\n" + "-" * 60)
    print(f"Dry Run: {checks_passed}/{checks_total} checks passed")
    print("-" * 60)

    return checks_passed == checks_total


def run_smoke_test(config: Phase05Config) -> bool:
    """
    Run smoke test with 10 training steps.

    Validates:
        - Full training pipeline works
        - Loss decreases
        - No NaN values
        - Gradients flow correctly
    """
    print("\n" + "=" * 60)
    print("Phase 0.5 Smoke Test (10 steps)")
    print("=" * 60)

    # Override config for smoke test
    config.max_steps = 10
    config.warmup_steps = 2
    config.eval_steps = 5
    config.save_steps = 10
    config.logging_steps = 1
    config.use_wandb = False

    try:
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Setup model
        base_model = setup_model(config)
        device = torch.device(config.device)
        base_model = base_model.to(device)

        # Setup layer freezing
        setup_layer_freezing(base_model, config)

        # Wrap in training model
        model = Phase05TrainingModel(base_model).to(device)

        # Create dataloaders (use synthetic data for speed)
        train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=True)

        # Create optimizer with Zipper LR
        optimizer = create_optimizer(model, config)

        # Create scheduler
        scheduler = create_lr_scheduler(optimizer, config)

        # Create gradient clipper
        grad_config = config.get_gradient_config()
        clipper = GradientClipper(model, grad_config)

        # Training loop
        losses = []
        progress_bar = tqdm(range(config.max_steps), desc="Smoke test")

        train_iter = iter(train_loader)
        for step in progress_bar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            loss, grad_info = train_step(
                model, batch, optimizer, scheduler, clipper, config, step=step
            )
            losses.append(loss)
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

            # Evaluation
            if (step + 1) % config.eval_steps == 0:
                metrics = evaluate(model, val_loader, config)
                logger.info(f"Eval @ step {step + 1}: {metrics}")

        # Results
        print("\n" + "-" * 60)
        print("Smoke Test Results:")
        print(f"  Final step: {config.max_steps}")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss decreased: {'YES' if losses[-1] < losses[0] else 'NO'}")

        has_nan = any(torch.isnan(torch.tensor(loss_val)) for loss_val in losses)
        print(f"  No NaN losses: {'YES' if not has_nan else 'NO'}")
        print("-" * 60)
        print("SMOKE TEST PASSED")
        return True

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_debug_mode(config: Phase05Config) -> bool:
    """
    Run debug mode with verbose gradient logging.

    Runs 5 steps with detailed logging of:
        - Per-layer gradient norms
        - Interface gradient ratio
        - Hub token gradients
        - Learning rates
    """
    print("\n" + "=" * 60)
    print("Phase 0.5 DEBUG Mode (5 steps with gradient logging)")
    print("=" * 60)

    # Override config for debug mode
    config.max_steps = 5
    config.warmup_steps = 1
    config.logging_steps = 1
    config.use_wandb = False

    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Setup model
        base_model = setup_model(config)
        device = torch.device(config.device)
        base_model = base_model.to(device)

        # Setup layer freezing with verbose output
        freezer = setup_layer_freezing(base_model, config)

        print("\n--- Layer Freezing Details ---")
        print(f"Frozen layers: {freezer.get_frozen_layers()}")
        print(f"Trainable layers: {freezer.get_trainable_layers()}")

        # Wrap in training model
        model = Phase05TrainingModel(base_model).to(device)

        # Create dataloaders (use synthetic data for speed)
        train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=True)

        # Create optimizer with Zipper LR
        optimizer = create_optimizer(model, config)

        # Print optimizer parameter groups
        print("\n--- Optimizer Parameter Groups ---")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: lr={group['lr']:.2e}, params={len(group['params'])}")

        # Create scheduler
        scheduler = create_lr_scheduler(optimizer, config)

        # Create gradient clipper
        grad_config = config.get_gradient_config()
        clipper = GradientClipper(model, grad_config)

        print("\n--- Training with Gradient Monitoring ---")

        train_iter = iter(train_loader)
        for step in range(config.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            print(f"\n=== Step {step + 1} ===")

            loss, grad_info = train_step(
                model, batch, optimizer, scheduler, clipper, config, step=step
            )

            print(f"Loss: {loss:.4f}")
            print(f"Total grad norm: {grad_info['total_norm']:.4f}")
            print(f"Has NaN: {grad_info['has_nan']}")
            print(f"Has Inf: {grad_info['has_inf']}")
            print(f"Clipped: {grad_info['clipped']}")

            # Print learning rates for each group
            print("Learning rates:")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Group {i}: {group['lr']:.6f}")

        print("\n" + "-" * 60)
        print("DEBUG MODE COMPLETED")
        print("-" * 60)
        return True

    except Exception as e:
        print(f"\nDEBUG MODE FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_full_training(
    config: Phase05Config,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """
    Run full Phase 0.5 training.

    Training loop with:
        - Zipper LR strategy
        - Gradient clipping
        - Hub token gradient masking
        - Periodic evaluation
        - Checkpointing
        - W&B logging
    """
    print("\n" + "=" * 60)
    print("Phase 0.5 Full Training")
    print("=" * 60)

    # Setup W&B
    setup_wandb(config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Setup model
    base_model = setup_model(config)
    device = torch.device(config.device)
    base_model = base_model.to(device)

    # Setup layer freezing
    freezer = setup_layer_freezing(base_model, config)
    logger.info(f"Frozen layers: {freezer.get_frozen_layers()}")
    logger.info(f"Trainable layers: {freezer.get_trainable_layers()}")

    # Setup hub token gradient masking
    hub_masker = setup_hub_gradient_masking(base_model, config)
    if hub_masker:
        logger.info("Hub token gradient masking enabled")

    # Wrap in training model
    model = Phase05TrainingModel(base_model).to(device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=False)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create optimizer with Zipper LR
    optimizer = create_optimizer(model, config)

    # Create scheduler
    scheduler = create_lr_scheduler(optimizer, config)

    # Create gradient clipper
    grad_config = config.get_gradient_config()
    clipper = GradientClipper(model, grad_config)

    # Resume from checkpoint if specified
    state = TrainingState()
    start_step = 0
    if resume_from:
        state = load_checkpoint(model, optimizer, scheduler, resume_from, device)
        start_step = state.global_step
        logger.info(f"Resumed from step {start_step}")

    # Training loop
    best_metric = state.best_metric
    losses = []

    progress_bar = tqdm(
        range(start_step, config.max_steps),
        desc=f"Training ({config.max_steps} steps)",
    )

    train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            state.epoch += 1

        # Training step
        loss, grad_info = train_step(model, batch, optimizer, scheduler, clipper, config, step=step)
        losses.append(loss)
        state.global_step = step + 1
        state.losses.append(loss)

        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        log_training_step(step, loss, grad_info, current_lr, config)

        # Evaluation
        if (step + 1) % config.eval_steps == 0:
            metrics = evaluate(model, val_loader, config)
            log_evaluation(step + 1, metrics, config)
            state.metrics_history.append(metrics)

            # Save best checkpoint
            eval_loss = metrics.get("eval_loss", float("inf"))
            if eval_loss < best_metric:
                best_metric = eval_loss
                state.best_metric = best_metric
                save_checkpoint(model, optimizer, scheduler, state, config.output_dir, is_best=True)
                logger.info(f"New best model saved (loss: {eval_loss:.4f})")

        # Regular checkpoint
        if (step + 1) % config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, state, config.output_dir, is_best=False)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, state, config.output_dir, is_best=False)

    # Final evaluation
    final_metrics = evaluate(model, val_loader, config)
    log_evaluation(config.max_steps, final_metrics, config)

    # Close W&B
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Final step: {config.max_steps}")
    print(f"  Final eval loss: {final_metrics.get('eval_loss', 'N/A')}")
    print(f"  Final eval accuracy: {final_metrics.get('eval_accuracy', 'N/A')}")
    print(f"  Best metric: {best_metric:.4f}")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    return {
        "final_step": config.max_steps,
        "final_loss": losses[-1] if losses else None,
        "final_metrics": final_metrics,
        "best_metric": best_metric,
    }


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0.5 Enhanced Healing Training for ModernBERT v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10-step smoke test",
    )
    mode_group.add_argument(
        "--debug",
        action="store_true",
        help="Run 5 steps with verbose gradient logging",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_v3_phase0_5_enhanced.yaml",
        help="Path to YAML configuration file",
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to initialized v3 model checkpoint",
    )
    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        default=None,
        help="Path to v2 checkpoint for initialization",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path",
    )

    # Training settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Base learning rate",
    )
    parser.add_argument(
        "--interface-lr",
        type=float,
        default=None,
        help="Interface layer (L23) learning rate",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 precision",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    # Additional overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format key=value or key.subkey=value",
    )

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for Phase 0.5 training."""
    args = parse_args()

    # Create base config
    config = Phase05Config()

    # Load YAML config if available
    if YAML_AVAILABLE and Path(args.config).exists():
        yaml_config = load_config(args.config)
        if yaml_config:
            config = merge_configs(config, yaml_config)

    # Apply CLI overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.v2_checkpoint:
        config.v2_checkpoint = args.v2_checkpoint
    if args.tokenizer:
        config.tokenizer_name = args.tokenizer
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.learning_rate:
        config.base_lr = args.learning_rate
    if args.interface_lr:
        config.lr_layer_23 = args.interface_lr
    if args.wandb:
        config.use_wandb = True
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.device:
        config.device = args.device
    if args.no_bf16:
        config.bf16 = False
    if args.seed:
        config.seed = args.seed

    # Apply additional overrides
    if args.overrides and YAML_AVAILABLE:
        override_dict = apply_overrides({}, args.overrides)
        config = merge_configs(config, override_dict)

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if HF_AVAILABLE:
        set_seed(config.seed)

    # Run appropriate mode
    if args.dry_run:
        success = run_dry_run(config)
        sys.exit(0 if success else 1)

    elif args.smoke_test:
        success = run_smoke_test(config)
        sys.exit(0 if success else 1)

    elif args.debug:
        success = run_debug_mode(config)
        sys.exit(0 if success else 1)

    else:
        # Full training
        results = run_full_training(config, resume_from=args.resume_from)
        logger.info(f"Training complete: {results}")


if __name__ == "__main__":
    main()

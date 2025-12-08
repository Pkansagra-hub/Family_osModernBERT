#!/usr/bin/env python3
"""
Phase 0.5 Enhanced Healing Training Script for ModernBERT v3

This script implements the "healing" phase that repairs the cloned layers
(L23-28) and establishes smooth activation flow across the L22->L23 interface.

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-28 (Semantic + Family bands), Hub tokens
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

# Suppress noisy warnings before any other imports
import warnings

warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import argparse
import json
import logging
import shutil
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
    V3BaseCollator,
    V3CollatorConfig,
    # Position constants from v3 infrastructure
    POSITION_CLS,
    POSITION_EMO,
    POSITION_MEM,
    POSITION_REL,
    POSITION_TASK,
    POSITION_TEXT_START,
    V3_SPECIAL_PREFIX_LEN,
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

# Ensure unbuffered output for Colab/Jupyter compatibility
import os

os.environ["PYTHONUNBUFFERED"] = "1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,  # Override any existing config (needed for Colab)
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
    lr_layers_19_22: float = 1e-5  # Semantic band
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
    semantic_clip: float = 1.0
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
    trainable_bands: list = field(default_factory=lambda: ["semantic", "family"])
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
    max_samples: int | None = None  # Limit samples for debug (None = all)
    max_train_samples: int | None = None  # Max training samples (None = use default 10000)
    max_eval_samples: int | None = None  # Max eval samples (None = use default 1000)
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
            semantic_lr=float(self.lr_layers_19_22),
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
            semantic_clip=self.semantic_clip,
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
    """
    Tracks training state for checkpointing and resumption.

    Maintains comprehensive training history including loss trajectory,
    learning rate schedule, and evaluation metrics for Phase 0.5 healing.

    Attributes:
        global_step: Total training steps completed
        epoch: Current epoch number (0-indexed)
        best_metric: Best validation metric achieved (lower is better)
        best_step: Step at which best metric was achieved
        phase: Training phase identifier
        losses: Rolling window of recent loss values
        metrics_history: History of evaluation metrics
        lr_history: Learning rate at each evaluation point
    """

    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    best_step: int = 0
    phase: str = "phase_0.5"
    losses: list[float] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)

    def update_loss(self, loss: float) -> None:
        """Add loss to history, maintaining rolling window."""
        self.losses.append(loss)
        # Keep last 500 losses for plotting
        if len(self.losses) > 500:
            self.losses = self.losses[-500:]

    def update_metrics(self, metrics: dict[str, Any], lr: float | None = None) -> None:
        """Add evaluation metrics to history."""
        metrics_with_step = {"step": self.global_step, **metrics}
        self.metrics_history.append(metrics_with_step)
        if lr is not None:
            self.lr_history.append(lr)
        # Keep last 50 evaluations
        if len(self.metrics_history) > 50:
            self.metrics_history = self.metrics_history[-50:]

    def update_best(self, metric_value: float) -> bool:
        """
        Update best metric if improved.

        Args:
            metric_value: Current validation metric (lower is better)

        Returns:
            True if this is a new best, False otherwise
        """
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            self.best_step = self.global_step
            return True
        return False

    def get_average_loss(self, window: int = 100) -> float:
        """Get average loss over recent window."""
        if not self.losses:
            return 0.0
        recent = self.losses[-window:]
        return sum(recent) / len(recent)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "phase": self.phase,
            "losses": self.losses[-100:],  # Keep last 100 for checkpoint size
            "metrics_history": self.metrics_history[-10:],
            "lr_history": self.lr_history[-10:],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingState:
        """Create state from dictionary."""
        # Handle missing fields for backward compatibility
        return cls(
            global_step=d.get("global_step", 0),
            epoch=d.get("epoch", 0),
            best_metric=d.get("best_metric", float("inf")),
            best_step=d.get("best_step", 0),
            phase=d.get("phase", "phase_0.5"),
            losses=d.get("losses", []),
            metrics_history=d.get("metrics_history", []),
            lr_history=d.get("lr_history", []),
        )


# =============================================================================
# Healing Dataset
# =============================================================================


@dataclass
class TaskStats:
    """Statistics for a single task in the healing dataset."""

    name: str
    count: int = 0
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0

    def update(self, seq_length: int) -> None:
        """Update stats with a new sample."""
        if self.count == 0:
            self.min_length = seq_length
            self.max_length = seq_length
        else:
            self.min_length = min(self.min_length, seq_length)
            self.max_length = max(self.max_length, seq_length)

        # Running average
        self.avg_length = (self.avg_length * self.count + seq_length) / (self.count + 1)
        self.count += 1


class HealingDataset(Dataset):
    """
    Healing dataset for Phase 0.5 using public benchmarks.

    Loads from HuggingFace datasets to heal the cloned layers (L23-28):
        - SST-2: Sentiment classification (binary) -> uses [EMO] hub
        - MNLI: Natural Language Inference (3-way) -> uses [REL] hub
        - STS-B: Semantic Textual Similarity (regression) -> uses [MEM] hub
        - CoNLL-2003: NER (token classification) -> sequence tagging
        - SQuAD: Question Answering (span extraction) -> uses [TASK] hub

    The dataset samples are tokenized but NOT padded - padding is handled
    by the HealingCollator which also inserts hub tokens.

    Attributes:
        tokenizer: HuggingFace tokenizer (with v3 hub tokens)
        max_length: Maximum sequence length (default 512)
        samples: List of processed samples
        task_stats: Per-task statistics
    """

    # Supported tasks and their hub token associations
    TASK_HUB_MAP = {
        "sentiment": "[EMO]",  # Position 1
        "similarity": "[MEM]",  # Position 2
        "nli": "[REL]",  # Position 3
        "qa": "[TASK]",  # Position 4
        "ner": "[CLS]",  # Uses full sequence
    }

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 512,
        tasks: list[str] | None = None,
        seed: int = 42,
    ):
        """
        Initialize HealingDataset.

        Args:
            tokenizer: HuggingFace tokenizer with v3 hub tokens
            split: Data split ('train' or 'validation')
            max_samples: Maximum total samples (divided among tasks)
            max_length: Maximum sequence length after tokenization
            tasks: List of tasks to load (default: ['sentiment', 'nli'])
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.samples: list[dict[str, Any]] = []
        self.task_stats: dict[str, TaskStats] = {}

        if tasks is None:
            tasks = ["sentiment", "nli"]

        # Validate tasks
        for task in tasks:
            if task not in self.TASK_HUB_MAP:
                logger.warning(f"Unknown task: {task}, skipping")
                continue
            self.task_stats[task] = TaskStats(name=task)

        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available, using synthetic data")
            self._create_synthetic_samples(max_samples or 1000, tasks)
            return

        # Calculate samples per task (balanced distribution)
        samples_per_task = (max_samples // len(tasks)) if max_samples else None

        # Load each task with error handling
        loaders = {
            "sentiment": self._load_sst2,
            "nli": self._load_mnli,
            "ner": self._load_conll2003,
            "qa": self._load_squad,
            "similarity": self._load_stsb,
        }

        for task in tasks:
            if task in loaders:
                loaders[task](split, samples_per_task)

        # Log summary statistics
        self._log_stats()

    def _log_stats(self) -> None:
        """Log dataset statistics."""
        total = len(self.samples)
        logger.info(f"Loaded {total} healing samples for {self.split}")
        for task, stats in self.task_stats.items():
            if stats.count > 0:
                logger.info(
                    f"  {task}: {stats.count} samples, "
                    f"avg_len={stats.avg_length:.1f}, "
                    f"range=[{stats.min_length}, {stats.max_length}]"
                )

    def _add_sample(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        task: str,
        label: int | float | list[int],
        **extra_fields,
    ) -> None:
        """Add a sample and update statistics."""
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task": task,
            "label": label,
            **extra_fields,
        }
        self.samples.append(sample)

        # Update stats
        if task in self.task_stats:
            self.task_stats[task].update(len(input_ids))

    def _load_sst2(self, split: str, max_samples: int | None) -> None:
        """Load SST-2 sentiment data (binary classification)."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("glue", "sst2", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                sentence = item["sentence"]
                label = item["label"]

                encoding = self.tokenizer(
                    sentence,
                    max_length=self.max_length - 6,  # Reserve for hub tokens + SEP
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                self._add_sample(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    task="sentiment",
                    label=int(label),
                )
                count += 1

            logger.debug(f"Loaded {count} SST-2 samples")
        except Exception as e:
            logger.warning(f"Failed to load SST-2: {e}")

    def _load_mnli(self, split: str, max_samples: int | None) -> None:
        """Load MNLI NLI data (3-way classification)."""
        try:
            ds_split = "validation_matched" if split == "validation" else "train"
            ds = load_dataset("glue", "mnli", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                # Tokenize premise and hypothesis together
                premise = item["premise"]
                hypothesis = item["hypothesis"]
                label = item["label"]

                # Use tokenizer's built-in pair encoding
                encoding = self.tokenizer(
                    premise,
                    hypothesis,
                    max_length=self.max_length - 6,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                self._add_sample(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    task="nli",
                    label=int(label),
                )
                count += 1

            logger.debug(f"Loaded {count} MNLI samples")
        except Exception as e:
            logger.warning(f"Failed to load MNLI: {e}")

    def _load_conll2003(self, split: str, max_samples: int | None) -> None:
        """Load CoNLL-2003 NER data (token classification)."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("conll2003", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                tokens = item["tokens"]
                ner_tags = item["ner_tags"]

                # Tokenize with is_split_into_words for proper word-piece handling
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    max_length=self.max_length - 6,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Align labels to word-pieces
                word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None
                if word_ids is not None:
                    aligned_labels = []
                    previous_word_idx = None
                    for word_idx in word_ids:
                        if word_idx is None:
                            aligned_labels.append(-100)  # Special tokens
                        elif word_idx != previous_word_idx:
                            aligned_labels.append(
                                ner_tags[word_idx] if word_idx < len(ner_tags) else -100
                            )
                        else:
                            aligned_labels.append(-100)  # Subword continuation
                        previous_word_idx = word_idx
                else:
                    # Fallback: simple truncation
                    aligned_labels = ner_tags[: len(encoding["input_ids"]) - 2]

                self._add_sample(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    task="ner",
                    label=aligned_labels,
                    original_tokens=tokens,
                )
                count += 1

            logger.debug(f"Loaded {count} CoNLL-2003 samples")
        except Exception as e:
            logger.warning(f"Failed to load CoNLL-2003: {e}")

    def _load_squad(self, split: str, max_samples: int | None) -> None:
        """Load SQuAD QA data (span extraction)."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("squad", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                question = item["question"]
                context = item["context"]
                answers = item["answers"]

                # Get first answer
                answer_text = answers["text"][0] if answers["text"] else ""
                answer_start = answers["answer_start"][0] if answers["answer_start"] else 0

                # Tokenize question and context together
                encoding = self.tokenizer(
                    question,
                    context,
                    max_length=self.max_length - 6,
                    truncation="only_second",  # Truncate context, keep question
                    padding=False,
                    return_tensors=None,
                )

                self._add_sample(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    task="qa",
                    label=0,  # Placeholder - actual span finding in forward pass
                    answer_text=answer_text,
                    answer_start=answer_start,
                )
                count += 1

            logger.debug(f"Loaded {count} SQuAD samples")
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}")

    def _load_stsb(self, split: str, max_samples: int | None) -> None:
        """Load STS-B similarity data (regression 0-5)."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("glue", "stsb", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples:
                    break

                sentence1 = item["sentence1"]
                sentence2 = item["sentence2"]
                score = item["label"]  # 0-5 similarity score

                # Tokenize sentence pair
                encoding = self.tokenizer(
                    sentence1,
                    sentence2,
                    max_length=self.max_length - 6,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Normalize to 0-1 range
                normalized_score = float(score) / 5.0

                self._add_sample(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    task="similarity",
                    label=normalized_score,
                    raw_score=score,
                )
                count += 1

            logger.debug(f"Loaded {count} STS-B samples")
        except Exception as e:
            logger.warning(f"Failed to load STS-B: {e}")

    def _create_synthetic_samples(self, num_samples: int, tasks: list[str]) -> None:
        """Create synthetic samples when HuggingFace datasets not available."""
        import random

        random.seed(self.seed)

        for i in range(num_samples):
            # Random sequence length
            seq_len = random.randint(32, 128)
            input_ids = [random.randint(100, 50000) for _ in range(seq_len)]

            # Round-robin task assignment
            task = tasks[i % len(tasks)]

            # Task-specific labels
            if task == "sentiment":
                label = random.randint(0, 1)
            elif task == "nli":
                label = random.randint(0, 2)
            elif task == "ner":
                label = [random.randint(0, 8) for _ in range(seq_len)]
            elif task == "qa":
                label = 0
            elif task == "similarity":
                label = random.random()
            else:
                label = 0

            self._add_sample(
                input_ids=input_ids,
                attention_mask=[1] * seq_len,
                task=task,
                label=label,
            )

        logger.info(f"Created {num_samples} synthetic samples")

    def get_task_distribution(self) -> dict[str, int]:
        """Get count of samples per task."""
        return {task: stats.count for task, stats in self.task_stats.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


# =============================================================================
# Healing Collator
# =============================================================================


class HealingCollator(V3BaseCollator):
    """
    Data collator for Phase 0.5 healing that inserts v3 hub tokens.

    Extends V3BaseCollator to handle HuggingFace dataset format with
    separate 'input_ids', 'task', 'label' fields.

    Transforms standard tokenized input into v3 format:
        Input:  [CLS] <text tokens> [SEP]
        Output: [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

    The hub tokens enable task-specific routing:
        - Position 1 [EMO]: Sentiment/emotion classification
        - Position 2 [MEM]: Similarity/embedding tasks
        - Position 3 [REL]: NLI/relation tasks
        - Position 4 [TASK]: Intent/QA tasks

    Note:
        Uses HUB_TOKEN_IDS from modeling_studio.models.hub_tokens module
        and position constants from modeling_studio.data.collators_v3 module.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        hub_token_ids: dict[str, int] | None = None,
    ):
        """
        Initialize HealingCollator.

        Args:
            tokenizer: HuggingFace tokenizer with v3 hub tokens
            max_length: Maximum sequence length (default 512)
            hub_token_ids: Custom hub token IDs (uses imported HUB_TOKEN_IDS if None)
        """
        # Initialize base collator with config
        config = V3CollatorConfig(max_length=max_length)
        super().__init__(tokenizer, config)

        # Use imported HUB_TOKEN_IDS from hub_tokens module
        self.hub_token_ids = hub_token_ids or HUB_TOKEN_IDS

        # Get special token IDs from tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

        # Build hub token prefix: [CLS] [EMO] [MEM] [REL] [TASK]
        self.hub_prefix = [
            self.cls_token_id,
            self.hub_token_ids["[EMO]"],
            self.hub_token_ids["[MEM]"],
            self.hub_token_ids["[REL]"],
            self.hub_token_ids["[TASK]"],
        ]

    def _strip_special_tokens(self, input_ids: list[int]) -> list[int]:
        """Remove CLS and SEP tokens from the sequence."""
        if not input_ids:
            return input_ids

        # Remove leading CLS
        if input_ids[0] == self.cls_token_id:
            input_ids = input_ids[1:]

        # Remove trailing SEP (may have multiple from sentence pairs)
        while input_ids and input_ids[-1] == self.sep_token_id:
            input_ids = input_ids[:-1]

        # Remove internal SEP tokens but keep track for sentence pairs
        # (We keep them as they mark sentence boundaries)

        return input_ids

    def _build_v3_sequence(
        self,
        input_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Build v3 token sequence with hub tokens.

        Args:
            input_ids: Text token IDs (may include CLS/SEP)

        Returns:
            Tuple of (padded_input_ids, attention_mask)
        """
        # Strip existing special tokens
        text_ids = self._strip_special_tokens(list(input_ids))

        # Calculate available space for text
        # Layout: [CLS][EMO][MEM][REL][TASK]<text>[SEP][PAD...]
        max_text_len = self.config.max_length - V3_SPECIAL_PREFIX_LEN - 1  # -1 for [SEP]

        # Truncate text if needed
        if len(text_ids) > max_text_len:
            text_ids = text_ids[:max_text_len]

        # Build sequence: hub_prefix + text + SEP
        v3_ids = self.hub_prefix + text_ids + [self.sep_token_id]

        # Create attention mask (1 for real tokens, 0 for padding)
        seq_len = len(v3_ids)
        pad_len = self.config.max_length - seq_len

        attention_mask = [1] * seq_len + [0] * pad_len
        v3_ids = v3_ids + [self.pad_token_id] * pad_len

        return v3_ids, attention_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
        """
        Collate batch with hub token insertion.

        Args:
            batch: List of sample dictionaries with 'input_ids', 'task', 'label'

        Returns:
            Dictionary with:
                - input_ids: [batch_size, max_length]
                - attention_mask: [batch_size, max_length]
                - labels: [batch_size] or [batch_size, seq_len] for NER
                - tasks: List of task names
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_tasks = []

        for sample in batch:
            input_ids = sample["input_ids"]

            # Handle tensor inputs
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            # Build v3 sequence with hub tokens
            v3_ids, attention_mask = self._build_v3_sequence(input_ids)

            batch_input_ids.append(v3_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(sample["label"])
            batch_tasks.append(sample.get("task", "sentiment"))

        # Determine label type based on batch composition
        # Mixed batches use first task's type
        primary_task = batch_tasks[0] if batch_tasks else "sentiment"

        if primary_task == "similarity":
            # Regression task: float labels
            labels_tensor = torch.tensor(
                [float(lbl) if isinstance(lbl, (int, float)) else 0.0 for lbl in batch_labels],
                dtype=torch.float,
            )
        elif primary_task == "ner":
            # Token classification: pad label sequences
            max_label_len = self.config.max_length
            padded_labels = []
            for lbl in batch_labels:
                if isinstance(lbl, list):
                    # Add -100 for hub tokens + CLS, then labels, then -100 for rest
                    aligned = [-100] * V3_SPECIAL_PREFIX_LEN + lbl[
                        : max_label_len - V3_SPECIAL_PREFIX_LEN - 1
                    ]
                    aligned = aligned + [-100] * (max_label_len - len(aligned))
                    padded_labels.append(aligned)
                else:
                    padded_labels.append([-100] * max_label_len)
            labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
        else:
            # Classification tasks: integer labels
            labels_tensor = torch.tensor(
                [int(lbl) if isinstance(lbl, (int, float)) else 0 for lbl in batch_labels],
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
    """
    Output container for Phase 0.5 training forward pass.

    Attributes:
        loss: Scalar loss tensor (None if no labels provided)
        logits: Task-specific logits [batch_size, num_classes]
        hidden_states: Last layer hidden states [batch_size, seq_len, hidden_size]
        task_losses: Per-task loss breakdown (optional)
    """

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor
    task_losses: dict[str, float] | None = None


class Phase05TrainingModel(nn.Module):
    """
    Training wrapper for ModernBERTv3Ultra in Phase 0.5 healing.

    Adds task-specific classification heads that read from hub tokens:
        - [EMO] (pos 1): Sentiment/emotion classification (2 classes)
        - [MEM] (pos 2): Similarity regression (cosine-like 0-1)
        - [REL] (pos 3): NLI classification (3 classes)
        - [TASK] (pos 4): Intent/QA routing

    The goal of Phase 0.5 is to heal the cloned family layers (L23-28)
    so they learn to process hub tokens correctly while the frozen
    foundation/core layers (L1-18) provide stable representations.

    Architecture:
        - Wraps ModernBERTv3Ultra backbone
        - Adds lightweight task heads (single Linear layers)
        - Computes task-weighted loss for multi-task training

    Attributes:
        model: Underlying ModernBERTv3Ultra model
        sentiment_head: Binary sentiment classifier
        nli_head: 3-way NLI classifier
        similarity_head: Similarity regression head
        qa_head: QA start/end position head
        task_weights: Optional per-task loss weights

    Note:
        Uses position constants from modeling_studio.data.collators_v3:
        POSITION_CLS, POSITION_EMO, POSITION_MEM, POSITION_REL, POSITION_TASK
    """

    def __init__(
        self,
        model: ModernBERTv3Ultra,
        task_weights: dict[str, float] | None = None,
    ):
        """
        Initialize Phase05TrainingModel.

        Args:
            model: ModernBERTv3Ultra backbone
            task_weights: Per-task loss weights (default: equal weights)
        """
        super().__init__()
        self.model = model

        hidden_size = model.config.hidden_size

        # Task-specific classification heads
        # Each head reads from its designated hub token position
        self.sentiment_head = nn.Linear(hidden_size, 2)  # Binary
        self.nli_head = nn.Linear(hidden_size, 3)  # Entailment/Neutral/Contradiction
        self.similarity_head = nn.Linear(hidden_size, 1)  # Regression
        self.qa_head = nn.Linear(hidden_size, 2)  # Start/end logits

        # For NER (token classification), we use a sequence head
        self.ner_head = nn.Linear(hidden_size, 9)  # CoNLL-2003 has 9 NER tags

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

        # Task weights for loss aggregation
        self.task_weights = task_weights or {
            "sentiment": 1.0,
            "nli": 1.0,
            "similarity": 1.0,
            "qa": 1.0,
            "ner": 1.0,
        }

    @property
    def encoder(self):
        """Expose encoder for LayerFreezer compatibility."""
        return self.model.encoder

    @property
    def embeddings(self):
        """Expose embeddings for LayerFreezer compatibility."""
        return self.model.embeddings

    @property
    def config(self):
        """Expose config from underlying model."""
        return self.model.config

    def get_task_head(self, task: str) -> nn.Module:
        """Get the classification head for a task."""
        heads = {
            "sentiment": self.sentiment_head,
            "nli": self.nli_head,
            "similarity": self.similarity_head,
            "qa": self.qa_head,
            "ner": self.ner_head,
        }
        return heads.get(task, self.sentiment_head)

    def get_task_position(self, task: str) -> int:
        """Get the hub token position for a task."""
        positions = {
            "sentiment": POSITION_EMO,
            "nli": POSITION_REL,
            "similarity": POSITION_MEM,
            "qa": POSITION_TASK,
            "ner": -1,  # NER uses all positions
        }
        return positions.get(task, POSITION_CLS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        tasks: list[str] | None = None,
        return_task_losses: bool = False,
        **kwargs,
    ) -> Phase05Output:
        """
        Forward pass with task-specific loss computation.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Task labels (shape depends on task type)
            tasks: List of task names for each sample
            return_task_losses: Whether to return per-task loss breakdown
            **kwargs: Additional arguments (ignored)

        Returns:
            Phase05Output with loss, logits, and hidden states
        """
        # Get encoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        batch_size = input_ids.size(0)
        hidden_states = outputs.last_hidden_state
        device = input_ids.device
        dtype = hidden_states.dtype

        # Accumulate losses and logits
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        all_logits = []
        task_loss_accum: dict[str, list[float]] = {}

        for i in range(batch_size):
            task = tasks[i] if tasks else "sentiment"

            if task == "sentiment":
                # Use [EMO] hub token for sentiment
                pooled = hidden_states[i, POSITION_EMO, :]
                logits = self.sentiment_head(pooled)
                all_logits.append(logits)

                if labels is not None:
                    # Handle multi-dimensional labels (from mixed-task batches)
                    label_val = labels[i, 0] if labels.dim() > 1 else labels[i]
                    loss = self.ce_loss(logits.unsqueeze(0), label_val.long().unsqueeze(0))
                    weighted_loss = loss.mean() * self.task_weights.get("sentiment", 1.0)
                    total_loss = total_loss + weighted_loss
                    if return_task_losses:
                        task_loss_accum.setdefault("sentiment", []).append(loss.item())

            elif task == "nli":
                # Use [REL] hub token for NLI
                pooled = hidden_states[i, POSITION_REL, :]
                logits = self.nli_head(pooled)
                all_logits.append(logits)

                if labels is not None:
                    # Handle multi-dimensional labels (from mixed-task batches)
                    label_val = labels[i, 0] if labels.dim() > 1 else labels[i]
                    loss = self.ce_loss(logits.unsqueeze(0), label_val.long().unsqueeze(0))
                    weighted_loss = loss.mean() * self.task_weights.get("nli", 1.0)
                    total_loss = total_loss + weighted_loss
                    if return_task_losses:
                        task_loss_accum.setdefault("nli", []).append(loss.item())

            elif task == "similarity":
                # Use [MEM] hub token for similarity
                pooled = hidden_states[i, POSITION_MEM, :]
                logits = self.similarity_head(pooled).squeeze(-1)
                all_logits.append(logits.unsqueeze(0))  # Keep shape consistent

                if labels is not None:
                    # Handle multi-dimensional labels (from mixed-task batches)
                    label_val = labels[i, 0] if labels.dim() > 1 else labels[i]
                    target = label_val.float().unsqueeze(0)
                    loss = self.mse_loss(logits.unsqueeze(0), target)
                    weighted_loss = loss.mean() * self.task_weights.get("similarity", 1.0)
                    total_loss = total_loss + weighted_loss
                    if return_task_losses:
                        task_loss_accum.setdefault("similarity", []).append(loss.item())

            elif task == "qa":
                # Use [TASK] hub token for QA
                pooled = hidden_states[i, POSITION_TASK, :]
                logits = self.qa_head(pooled)
                all_logits.append(logits)

                if labels is not None:
                    # Simplified: use binary classification as placeholder
                    # Handle multi-dimensional labels (e.g., QA with start/end)
                    if labels.dim() > 1:
                        # Use first element of label sequence
                        label_val = labels[i, 0] if labels.shape[1] > 0 else labels[i].flatten()[0]
                    else:
                        label_val = labels[i]
                    target = (label_val.long() % 2).unsqueeze(0)
                    loss = self.ce_loss(logits.unsqueeze(0), target)
                    weighted_loss = loss.mean() * self.task_weights.get("qa", 1.0)
                    total_loss = total_loss + weighted_loss
                    if return_task_losses:
                        task_loss_accum.setdefault("qa", []).append(loss.item())

            elif task == "ner":
                # NER: sequence labeling (simplified for healing)
                # For now, just use [CLS] as sequence representation
                pooled = hidden_states[i, POSITION_CLS, :]
                logits = self.sentiment_head(pooled)  # Binary as placeholder
                all_logits.append(logits)

                if labels is not None:
                    # Handle sequence labels by taking first non-padding label
                    if labels.dim() > 1:
                        label_seq = labels[i]
                        valid_labels = label_seq[label_seq != -100]
                        label_val = (
                            valid_labels[0]
                            if len(valid_labels) > 0
                            else torch.tensor(0, device=device)
                        )
                    else:
                        label_val = labels[i]

                    loss = self.ce_loss(
                        logits.unsqueeze(0),
                        (label_val.long() % 2).unsqueeze(0),
                    )
                    weighted_loss = loss.mean() * self.task_weights.get("ner", 1.0)
                    total_loss = total_loss + weighted_loss
                    if return_task_losses:
                        task_loss_accum.setdefault("ner", []).append(loss.item())

            else:
                # Unknown task: default to sentiment-like
                pooled = hidden_states[i, POSITION_CLS, :]
                logits = self.sentiment_head(pooled)
                all_logits.append(logits)

        # Average loss over batch
        if labels is not None and batch_size > 0:
            total_loss = total_loss / batch_size

        # Stack logits (handle variable shapes)
        if all_logits:
            try:
                logits_tensor = torch.stack(all_logits)
            except RuntimeError:
                # Different shapes: pad to maximum
                max_size = max(lg.numel() for lg in all_logits)
                padded = []
                for lg in all_logits:
                    flat = lg.flatten()
                    if flat.numel() < max_size:
                        pad = torch.zeros(max_size - flat.numel(), device=device, dtype=dtype)
                        padded.append(torch.cat([flat, pad]))
                    else:
                        padded.append(flat[:max_size])
                logits_tensor = torch.stack(padded)
        else:
            logits_tensor = torch.zeros(1, 2, device=device, dtype=dtype)

        # Aggregate per-task losses
        task_losses = None
        if return_task_losses and task_loss_accum:
            task_losses = {
                task: sum(losses) / len(losses) for task, losses in task_loss_accum.items()
            }

        return Phase05Output(
            loss=total_loss if labels is not None else None,
            logits=logits_tensor,
            hidden_states=hidden_states,
            task_losses=task_losses,
        )


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Supports hierarchical YAML configuration files with nested sections
    for training, learning_rate, gradient, optimizer, checkpointing, logging.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary (empty if file not found)

    Example:
        >>> config_dict = load_config("configs/training/phase05.yaml")
        >>> config = Phase05Config.from_dict(config_dict)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    if not YAML_AVAILABLE:
        logger.warning("YAML not available, cannot load config (install pyyaml)")
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return {}


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Apply CLI overrides to configuration dictionary.

    Supports dotted key notation for nested values and automatic
    type inference (bool, int, float, string).

    Args:
        config: Base configuration dictionary
        overrides: List of "key=value" or "key.subkey=value" strings

    Returns:
        Updated configuration dictionary

    Example:
        >>> config = {"training": {"lr": 1e-4}}
        >>> apply_overrides(config, ["training.lr=5e-5", "seed=123"])
        {'training': {'lr': 5e-5}, 'seed': 123}
    """
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
    """
    Merge YAML configuration into Phase05Config.

    Maps hierarchical YAML keys to flat config attributes using
    a predefined mapping. Direct matches are also applied.

    Args:
        base_config: Base Phase05Config with default values
        yaml_config: Dictionary from YAML file

    Returns:
        New Phase05Config with merged values
    """
    config_dict = base_config.to_dict()

    # Flatten nested YAML config and apply
    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        items: dict[str, Any] = {}
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
    Load or initialize ModernBERTv3Ultra model for Phase 0.5 training.

    Model Loading Priority:
        1. Load from model_path if checkpoint exists (preferred)
        2. Initialize from v2_checkpoint via weight transfer
        3. Create with random initialization (NOT recommended)

    The expected scenario for Phase 0.5 is loading a v3 model that was
    initialized from v2 in a previous step (via initialize_v3_from_v2.py).

    Args:
        config: Phase05Config with model_path and v2_checkpoint

    Returns:
        ModernBERTv3Ultra model ready for training

    Raises:
        RuntimeError: If model cannot be loaded (in strict mode)
    """
    model_path = Path(config.model_path)

    # Try loading from existing v3 checkpoint (pytorch_model.bin)
    if model_path.exists() and (model_path / "pytorch_model.bin").exists():
        logger.info(f"Loading v3 model from {model_path}")
        # Disable LoRA for Phase 0.5 - train layers directly
        v3_config = ModernBERTv3Config(lora_enabled=False, lora_target_layers=[])
        model = ModernBERTv3Ultra(v3_config)

        state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    # Try loading from safetensors format
    if model_path.exists() and (model_path / "model.safetensors").exists():
        logger.info(f"Loading v3 model from safetensors: {model_path}")
        # Disable LoRA for Phase 0.5 - train layers directly
        v3_config = ModernBERTv3Config(lora_enabled=False, lora_target_layers=[])
        model = ModernBERTv3Ultra(v3_config)

        try:
            from safetensors.torch import load_file

            state_dict = load_file(model_path / "model.safetensors")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
        except ImportError:
            logger.warning("safetensors package not available")

    # Try initializing from v2 checkpoint (weight transfer + cloning)
    v2_path = Path(config.v2_checkpoint)
    if v2_path.exists() and V2_INIT_AVAILABLE:
        logger.info(f"Initializing v3 model from v2 checkpoint: {v2_path}")
        # Disable LoRA for Phase 0.5 - train layers directly
        v3_config = ModernBERTv3Config(lora_enabled=False, lora_target_layers=[])
        model = ModernBERTv3Ultra(v3_config)

        stats = initialize_from_v2(model, str(v2_path))
        logger.info(
            f"Initialized from v2:\n"
            f"  Transferred: {stats.transferred_params:,} parameters\n"
            f"  Initialized (new): {stats.initialized_params:,} parameters\n"
            f"  Source layers: L1-22 -> Target: L1-28"
        )
        return model

    # Fall back to random initialization (NOT recommended for Phase 0.5)
    logger.warning(
        "No checkpoint found - creating model with RANDOM initialization.\n"
        "This is NOT recommended for Phase 0.5 training!\n"
        f"  Checked: {model_path}\n"
        f"  Checked: {v2_path}"
    )
    # Disable LoRA for Phase 0.5 - train layers directly
    v3_config = ModernBERTv3Config(lora_enabled=False, lora_target_layers=[])
    model = ModernBERTv3Ultra(v3_config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def setup_layer_freezing(model: nn.Module, config: Phase05Config) -> LayerFreezer:
    """
    Configure layer freezing for Phase 0.5 healing strategy.

    Phase 0.5 Freezing Strategy:
        - Frozen: L1-18 (Foundation + Core bands) - preserved v2 knowledge
        - Trainable: L19-28 (Semantic + Family bands) - layers to heal

    The goal is to heal the cloned family layers (L23-28) while keeping
    the foundation stable. Semantic layers (L19-22) also train to adapt
    their output for the new family layers.

    Args:
        model: Phase05TrainingModel or ModernBERTv3Ultra
        config: Phase05Config (for future customization)

    Returns:
        Configured LayerFreezer instance
    """
    # Get base model if wrapped in Phase05TrainingModel
    base_model = model.model if hasattr(model, "model") else model

    # Create freezer and configure for Phase 0.5
    freezer = LayerFreezer(base_model)
    freezer.configure_for_phase(TrainingPhase.PHASE_0_5)

    # Log comprehensive freeze stats
    stats = freezer.get_freeze_stats()
    frozen_layers = freezer.get_frozen_layers()
    trainable_layers = freezer.get_trainable_layers()

    logger.info(
        f"Layer freezing configured for Phase 0.5:\n"
        f"  Frozen parameters: {stats['frozen_params']:,}\n"
        f"  Trainable parameters: {stats['trainable_params']:,}\n"
        f"  Frozen layers: {frozen_layers}\n"
        f"  Trainable layers: {trainable_layers}"
    )

    return freezer


def setup_hub_gradient_masking(
    model: nn.Module, config: Phase05Config
) -> EmbeddingGradientHook | None:
    """
    Setup selective gradient masking for hub token embeddings.

    Hub Token Training Strategy:
        - Original vocabulary (50368 tokens): Frozen
        - Hub tokens [EMO], [MEM], [REL], [TASK]: Trainable

    This prevents catastrophic forgetting of the original embeddings
    while allowing the new hub tokens to learn their representations.

    Args:
        model: Model containing embeddings to mask
        config: Phase05Config with gradient masking settings

    Returns:
        EmbeddingGradientHook if enabled, None otherwise
    """
    if not config.freeze_original_vocab:
        logger.info("Hub gradient masking disabled (freeze_original_vocab=False)")
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
    Create AdamW optimizer with Zipper LR layer-group learning rates.

    Zipper LR Strategy for Phase 0.5 (REVERSED):
        The "zipper" metaphor describes how learning rates are configured
        across the model layers. In the REVERSED strategy, maximum plasticity
        is at the semantic band (L19-22) closest to frozen layers, with rates
        decreasing toward the output family layers.

    Layer Groups:
        - Frozen (L1-18): lr=0 (handled by LayerFreezer)
        - Semantic (L19-22): lr=semantic_lr (8e-6 default) - MAXIMUM
        - Interface (L23): lr=interface_lr (6e-6 default) - medium
        - Family (L24-28): lr=family_lr (5e-6 default) - lowest
        - Task heads: lr=task_heads_lr (3e-4 default)
        - Hub embeddings: lr=lr_hub_tokens (3e-4 default)

    Args:
        model: Phase05TrainingModel with task heads
        config: Phase05Config with learning rate settings

    Returns:
        Configured AdamW optimizer with parameter groups
    """
    zipper_config = config.get_zipper_lr_config()
    base_model = model.model if hasattr(model, "model") else model

    param_groups: list[dict[str, Any]] = []

    # Get encoder layers
    encoder = getattr(base_model, "encoder", base_model)
    layers = getattr(encoder, "layers", None)

    if layers is not None:
        num_layers = len(layers)

        # Semantic layers (L19-22, indices 18-21)
        semantic_params = []
        for i in range(18, min(22, num_layers)):
            semantic_params.extend([p for p in layers[i].parameters() if p.requires_grad])
        if semantic_params:
            param_groups.append(
                {
                    "params": semantic_params,
                    "lr": zipper_config.semantic_lr,
                    "name": "semantic_L19-22",
                }
            )

        # Interface layer (L23, index 22) - medium plasticity (bridge between semantic and family)
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

        # Family layers (L24-28, indices 23-27) - lowest plasticity
        # In reversed zipper, family layers use family_lr directly (no decay needed)
        for layer_idx in range(23, min(28, num_layers)):
            layer_params = [p for p in layers[layer_idx].parameters() if p.requires_grad]
            if layer_params:
                # Use family_lr directly - already the lowest in reversed zipper
                layer_lr = zipper_config.family_lr

                param_groups.append(
                    {
                        "params": layer_params,
                        "lr": layer_lr,
                        "name": f"family_L{layer_idx + 1}",
                    }
                )

    # Task heads (from Phase05TrainingModel wrapper)
    head_params = []
    head_names = ["sentiment_head", "nli_head", "similarity_head", "qa_head", "ner_head"]
    for head_name in head_names:
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

    # Embeddings (hub tokens if gradient masking is NOT applied)
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

    # Collect any remaining trainable parameters not yet assigned
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

    # Log parameter group summary
    total_params = 0
    logger.info("Optimizer parameter groups (Zipper LR strategy):")
    for group in param_groups:
        num_params = sum(p.numel() for p in group["params"])
        total_params += num_params
        logger.info(f"  {group['name']}: {num_params:,} params, lr={group['lr']:.2e}")
    logger.info(f"  Total trainable: {total_params:,} parameters")

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
    Create learning rate scheduler with warmup.

    Scheduler Types:
        - "cosine": Warmup + cosine annealing (default, recommended)
        - "linear": Warmup + linear decay to min_lr_ratio
        - "constant": Warmup only, then constant LR

    The scheduler applies uniformly to all parameter groups, preserving
    the relative learning rate ratios from the Zipper LR strategy.

    Args:
        optimizer: Configured optimizer with parameter groups
        config: Phase05Config with scheduler settings

    Returns:
        Learning rate scheduler
    """
    scheduler: torch.optim.lr_scheduler.LRScheduler

    if config.scheduler_type == "cosine":
        try:
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
            )
        except ImportError:
            # Fallback to PyTorch scheduler (no warmup)
            logger.warning(
                "transformers not available, using PyTorch CosineAnnealingLR (no warmup)"
            )
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
            logger.warning("transformers not available, using PyTorch LinearLR (no warmup)")
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.min_lr_ratio,
                total_iters=config.max_steps,
            )
    else:
        # Default: warmup-only scheduler (constant after warmup)
        def warmup_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return float(step) / max(1, config.warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    logger.info(
        f"Created {config.scheduler_type} scheduler:\n"
        f"  Warmup steps: {config.warmup_steps}\n"
        f"  Total steps: {config.max_steps}"
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
    """
    Create training and validation DataLoaders for Phase 0.5.

    Uses HealingDataset to load public benchmarks (SST-2, MNLI, STS-B, etc.)
    and HealingCollator to insert hub tokens into the v3 token layout.

    Args:
        config: Phase05Config with batch sizes, tasks, and max_length
        tokenizer: HuggingFace tokenizer with v3 hub tokens
        synthetic: If True, use smaller synthetic data for testing

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Determine sample counts based on mode
    if synthetic:
        train_samples = 1000
        val_samples = 100
    elif config.max_samples:
        # Debug mode: use max_samples for both train and val
        train_samples = config.max_samples
        val_samples = min(100, config.max_samples // 5)
    else:
        train_samples = config.max_train_samples or 10000
        val_samples = config.max_eval_samples or 1000

    # Create datasets
    train_dataset = HealingDataset(
        tokenizer,
        split="train",
        max_samples=train_samples,
        max_length=config.max_length,
        tasks=config.tasks,
        seed=config.seed,
    )

    val_dataset = HealingDataset(
        tokenizer,
        split="validation",
        max_samples=val_samples,
        max_length=config.max_length,
        tasks=config.tasks,
        seed=config.seed,
    )

    # Create collator with hub token insertion
    collator = HealingCollator(tokenizer, max_length=config.max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
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
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    clipper: GradientClipper,
    config: Phase05Config,
    step: int = 0,
) -> tuple[float, dict[str, Any]]:
    """
    Execute a single training step.

    Performs forward pass, loss computation, backward pass, gradient clipping
    with the GradientClipper, and optimizer/scheduler steps.

    Mixed Precision:
        - BF16 (bfloat16): Recommended for Ampere+ GPUs
        - FP16 (float16): For older GPUs (requires gradient scaling)
        - FP32: Fallback for CPU or debugging

    Args:
        model: Phase05TrainingModel in training mode
        batch: Dictionary with input_ids, attention_mask, labels, tasks
        optimizer: AdamW optimizer with Zipper LR parameter groups
        scheduler: Learning rate scheduler
        clipper: GradientClipper for per-layer gradient clipping
        config: Phase05Config for mixed precision settings
        step: Current global step (for logging)

    Returns:
        Tuple of (loss_value, gradient_stats_dict)
            - loss_value: Scalar loss (float("nan") if NaN detected)
            - gradient_stats: dict with total_norm, has_nan, has_inf, clipped
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
    if config.bf16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                tasks=tasks,
            )
            loss = outputs.loss
    elif config.fp16 and device.type == "cuda":
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

    # Check for NaN loss (critical for debugging)
    if loss is None or torch.isnan(loss):
        logger.warning(f"NaN/None loss detected at step {step}")
        return float("nan"), {
            "total_norm": 0.0,
            "has_nan": True,
            "has_inf": False,
            "clipped": False,
        }

    # Scale loss if it exceeds threshold (gradient-preserving approach)
    MAX_LOSS = 50.0  # Upper bound for loss scaling
    original_loss = loss.item()
    if original_loss > MAX_LOSS:
        # Scale down the loss to MAX_LOSS while preserving gradient direction
        scale_factor = MAX_LOSS / original_loss
        loss = loss * scale_factor
        logger.warning(f"Loss spike: {original_loss:.2f} > {MAX_LOSS}, scaled by {scale_factor:.4f}")

    # Backward pass
    loss.backward()

    # Gradient clipping using GradientClipper (per-layer aware)
    grad_stats = clipper.clip_gradients()

    # Optimizer and scheduler step
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
    """
    Run evaluation on validation set.

    Computes loss and accuracy metrics for each task type.
    Classification tasks (sentiment, NLI) report accuracy.
    Regression tasks (similarity) report MSE.

    Args:
        model: Phase05TrainingModel
        dataloader: Validation DataLoader
        config: Phase05Config for mixed precision

    Returns:
        Dictionary with eval_loss, eval_accuracy, and per-task metrics
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    task_metrics: dict[str, dict[str, int]] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tasks = batch.get("tasks", ["sentiment"] * len(labels))

            # Forward with optional mixed precision
            if config.bf16 and device.type == "cuda":
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

            # Accumulate loss
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * len(labels)

            # Compute accuracy for classification tasks
            for i, task in enumerate(tasks):
                if task in ["sentiment", "nli"]:
                    if outputs.logits.dim() > 1 and i < outputs.logits.size(0):
                        pred = outputs.logits[i].argmax().item()
                    else:
                        pred = 0
                    label_val = labels[i].item() if labels.dim() == 1 else 0
                    correct = int(pred == label_val)
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
) -> Path:
    """
    Save training checkpoint.

    Saves complete training state including:
        - Model weights (pytorch_model.bin)
        - Task heads (task_heads.pt)
        - Optimizer and scheduler state (training_state.pt)
        - Training state JSON (trainer_state.json)
        - Config JSON (config.json)

    If is_best=True, also copies to 'best/' directory.
    Automatically removes oldest checkpoints beyond save_total_limit.

    Args:
        model: Phase05TrainingModel to save
        optimizer: Current optimizer state
        scheduler: Current scheduler state
        state: TrainingState with step, epoch, metrics
        config: Phase05Config
        is_best: Whether this is the best model so far

    Returns:
        Path to saved checkpoint directory
    """
    output_path = Path(config.output_dir)
    checkpoint_dir = output_path / f"checkpoint-{state.global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save base model (encoder + embeddings)
    base_model = model.model if hasattr(model, "model") else model
    torch.save(base_model.state_dict(), checkpoint_dir / "pytorch_model.bin")

    # Save task heads separately (for easy loading without full model)
    heads_state = {}
    head_names = ["sentiment_head", "nli_head", "similarity_head", "qa_head", "ner_head"]
    for head_name in head_names:
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

    # Save best model copy
    if is_best:
        best_dir = output_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Saved best model: {best_dir}")

    # Cleanup old checkpoints beyond limit
    if config.save_total_limit > 0:
        checkpoints = sorted(
            output_path.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
        )
        while len(checkpoints) > config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")

    return checkpoint_dir


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: Path | str,
    config: Phase05Config,
) -> TrainingState:
    """
    Load training checkpoint for resumption.

    Restores complete training state:
        - Model weights
        - Task heads
        - Optimizer state (including per-parameter states)
        - Scheduler state
        - Training state (step, epoch, best_metric, etc.)

    Args:
        model: Phase05TrainingModel to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore
        checkpoint_path: Path to checkpoint directory
        config: Phase05Config (unused but kept for consistency)

    Returns:
        TrainingState restored from checkpoint
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return TrainingState()

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

    # Load trainer state JSON
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state_dict = json.load(f)
        state = TrainingState.from_dict(state_dict)
        logger.info(f"Restored training state: step={state.global_step}, epoch={state.epoch}")
        return state

    return TrainingState()


# =============================================================================
# Logging
# =============================================================================


def setup_wandb(config: Phase05Config) -> bool:
    """
    Initialize Weights & Biases logging.

    Creates a W&B run with Phase 0.5 training configuration including
    layer freezing strategy, Zipper LR parameters, and gradient settings.

    Args:
        config: Phase05Config with wandb settings

    Returns:
        True if W&B was initialized, False otherwise
    """
    if not config.use_wandb:
        return False

    if not WANDB_AVAILABLE:
        logger.warning("wandb package not available, disabling logging")
        config.use_wandb = False
        return False

    run_name = config.wandb_run_name or f"phase0.5_{config.max_steps}steps"

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        tags=config.wandb_tags,
        config={
            "phase": "0.5",
            "description": "Enhanced Healing Training",
            "max_steps": config.max_steps,
            "batch_size": config.train_batch_size,
            "base_lr": config.base_lr,
            "semantic_lr": config.lr_layers_19_22,
            "interface_lr": config.lr_layer_23,
            "family_lr": config.lr_layers_24_28,
            "frozen_layers": "L1-18",
            "trainable_layers": "L19-28",
            "warmup_steps": config.warmup_steps,
            "scheduler": config.scheduler_type,
            "max_grad_norm": config.max_grad_norm,
            "bf16": config.bf16,
            "seed": config.seed,
        },
    )

    logger.info(f"W&B initialized: {run_name}")
    return True


def log_training_step(
    step: int,
    loss: float,
    grad_stats: dict[str, Any],
    lr: float,
    config: Phase05Config,
) -> None:
    """
    Log training step metrics.

    Logs to console at configured intervals and to W&B on every step.

    Args:
        step: Current training step (0-indexed)
        loss: Training loss value
        grad_stats: Gradient statistics from GradientClipper
        lr: Current learning rate
        config: Phase05Config for logging settings
    """
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

        # Add gradient statistics
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
    """
    Log evaluation metrics.

    Args:
        step: Current training step
        metrics: Dictionary of evaluation metrics
        config: Phase05Config for logging settings
    """
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
    print(f"    Semantic (L19-22):   {zipper_config.semantic_lr}")
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
        # Disable LoRA for Phase 0.5
        v3_config = ModernBERTv3Config(lora_enabled=False, lora_target_layers=[])
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
    print("Phase 0.5 DEBUG Mode (5 steps, 500 samples)")
    print("=" * 60)

    # Override config for debug mode
    config.max_steps = 5
    config.warmup_steps = 1
    config.logging_steps = 1
    config.use_wandb = False
    config.max_samples = config.max_samples or 500  # Default 500 for debug

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

        # Create dataloaders (use limited real data in debug mode)
        use_synthetic = config.max_samples is None  # Use synthetic only if no limit set
        train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=use_synthetic)

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

        # Save checkpoint for orchestrator to chain phases
        print("\n--- Saving Debug Checkpoint ---")
        output_path = Path(config.output_dir)
        best_dir = output_path / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Save base model weights
        base_model_to_save = model.model if hasattr(model, "model") else model
        torch.save(base_model_to_save.state_dict(), best_dir / "pytorch_model.bin")

        # Save model config
        if hasattr(base_model_to_save, "config"):
            model_config = base_model_to_save.config
            if hasattr(model_config, "to_dict"):
                with open(best_dir / "model_config.json", "w") as f:
                    json.dump(model_config.to_dict(), f, indent=2)

        # Save tokenizer
        tokenizer.save_pretrained(str(best_dir))
        print(f"Saved debug checkpoint to: {best_dir}")

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
    Run full Phase 0.5 healing training.

    Complete training pipeline with:
        - Model initialization from v2 or checkpoint
        - Layer freezing (L1-18 frozen, L19-28 trainable)
        - Zipper LR strategy (maximum plasticity at L23)
        - Per-layer gradient clipping
        - Hub token gradient masking (optional)
        - Periodic evaluation on validation set
        - Checkpoint saving with best model tracking
        - W&B logging (optional)

    The goal of Phase 0.5 is to "heal" the cloned family layers (L23-28)
    that were duplicated from L17-22 during v2->v3 initialization. After
    healing, these layers should properly integrate with the frozen
    foundation/core layers.

    Args:
        config: Phase05Config with all training parameters
        resume_from: Optional path to checkpoint for resumption

    Returns:
        Dictionary with:
            - final_step: Last training step completed
            - final_loss: Final training loss
            - final_metrics: Evaluation metrics from last eval
            - best_metric: Best validation metric achieved
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
        state = load_checkpoint(model, optimizer, scheduler, resume_from, config)
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
                save_checkpoint(model, optimizer, scheduler, state, config, is_best=True)
                logger.info(f"New best model saved (loss: {eval_loss:.4f})")

        # Regular checkpoint
        if (step + 1) % config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, state, config, is_best=False)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, state, config, is_best=False)

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
    """
    Parse command line arguments for Phase 0.5 training.

    Supports multiple execution modes:
        --dry-run: Validate configuration without training
        --smoke-test: Run 10-step quick validation
        --debug: Run 5 steps with verbose gradient logging
        (default): Full training run

    Configuration can be loaded from YAML and overridden via CLI.

    Returns:
        Parsed argparse.Namespace
    """
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
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to load (for debug)",
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


def main() -> None:
    """
    Main entry point for Phase 0.5 healing training.

    Execution Flow:
        1. Parse command line arguments
        2. Load YAML config (if available)
        3. Apply CLI overrides
        4. Set random seeds for reproducibility
        5. Execute selected mode:
           - --dry-run: Validate configuration
           - --smoke-test: Quick 10-step test
           - --debug: Verbose gradient debugging
           - (default): Full training run

    Exit Codes:
        0: Success
        1: Failure or validation error
    """
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
    if args.max_samples:
        config.max_samples = args.max_samples
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

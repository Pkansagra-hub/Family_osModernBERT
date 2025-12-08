#!/usr/bin/env python3
"""
Phase 1 Multi-Task FamilyOS Training Script for ModernBERT v3

This script implements full multi-task training on unified FamilyOS data
with hub routing for per-sample task activation.

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-28 (Semantic + Family bands), Hub tokens, All task heads
    - LR: Zipper strategy (phase_1_multitask preset)
    - Data: Unified FamilyOS shards with hub_routing
    - Loss: Hub-weighted multi-task loss
    - Replay: 15% healing data for forgetting prevention
    - LoRA: Applied to layers 23-28 (r=16, alpha=16)

Phase 1 Objectives:
    1. Train on all 8 FamilyOS task types simultaneously
    2. Use hub routing for per-sample task activation
    3. Apply hub-weighted loss scaling
    4. Integrate replay sampling to prevent forgetting
    5. Monitor per-task and per-hub metrics

Usage:
    # Dry run (validate configuration)
    python scripts/train_v3_phase1.py --dry-run

    # Smoke test (10 steps)
    python scripts/train_v3_phase1.py --smoke-test

    # Debug mode (5 steps with verbose logging)
    python scripts/train_v3_phase1.py --debug

    # Full training
    python scripts/train_v3_phase1.py \\
        --config configs/training/multitask/stage_v3_phase1.yaml \\
        --model-path outputs/v3_phase0_5/best_model \\
        --output-dir outputs/v3_phase1

    # Resume training
    python scripts/train_v3_phase1.py \\
        --resume-from outputs/v3_phase1/checkpoint-5000

    # With overrides
    python scripts/train_v3_phase1.py \\
        --config configs/training/multitask/stage_v3_phase1.yaml \\
        --learning-rate 2e-5 \\
        --replay-ratio 0.15 \\
        --max-steps 10000 \\
        --wandb-run-name "phase1_experiment_1"

Environment:
    - GPU: A100/H100 recommended (16GB+ VRAM)
    - RAM: 32GB+ recommended

Outputs:
    - outputs/v3_phase1/: Checkpoints and final model
    - outputs/v3_phase1/best_model/: Best model by validation loss
    - wandb/: W&B logs (if enabled)

Issue: 5.4.2 - Implement Phase 1 Multi-Task Training Script
Author: FamilyOS Team
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
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
from modeling_studio.models.hub_tokens import (
    HUB_TOKEN_IDS,
    get_hub_for_capability,
    get_all_hub_tokens,
)

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
    POSITION_CLS,
    POSITION_EMO,
    POSITION_MEM,
    POSITION_REL,
    POSITION_TASK,
    POSITION_TEXT_START,
    V3_SPECIAL_PREFIX_LEN,
)
from modeling_studio.data.loaders_v3 import (
    TaskType,
    HubRouting,
    UnifiedSample,
    HubRoutingParser,
)

# =============================================================================
# Core Imports - Training (v3 infrastructure)
# =============================================================================

from modeling_studio.training.losses_v3 import (
    HubLossConfig,
    HubLossWeightCalculator,
    HubWeightedMultiTaskLoss,
)

# =============================================================================
# Optional Imports - LoRA
# =============================================================================

try:
    from modeling_studio.models.lora_v3 import (
        LoRAConfig,
        apply_lora_to_model,
        merge_lora_weights,
        get_lora_params,
    )

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

# =============================================================================
# Optional Imports - Shard Loading
# =============================================================================

try:
    from modeling_studio.data.shard_loader_v3 import (
        ShardConfig,
        StreamingShardDataset,
        create_streaming_dataset,
    )

    SHARD_LOADER_AVAILABLE = True
except ImportError:
    SHARD_LOADER_AVAILABLE = False

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
class Phase1Config:
    """
    Configuration for Phase 1 Multi-Task FamilyOS training.

    This dataclass contains all configuration options for Phase 1 training,
    which trains on unified FamilyOS data with hub routing for task activation.

    Attributes:
        model_path: Path to Phase 0.5 trained model checkpoint
        tokenizer_name: HuggingFace tokenizer name or path
        max_steps: Maximum training steps (default: 10000)
        warmup_steps: Number of warmup steps (default: 1000)
        ... (see individual attributes)
    """

    # =========================================================================
    # Model Configuration
    # =========================================================================
    model_path: str = "outputs/v3_phase0_5/best_model"
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    model_config_path: str = "configs/model/encoder/modernbert_v3_ultra.yaml"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    max_steps: int = 10000
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # =========================================================================
    # Batch Configuration
    # =========================================================================
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    max_length: int = 512

    # =========================================================================
    # Zipper Learning Rate Strategy (Phase 1 preset)
    # =========================================================================
    lr_strategy: str = "zipper"
    base_lr: float = 2e-5  # Lower than Phase 0.5
    lr_layers_1_18: float = 0.0  # Frozen
    lr_layers_19_22: float = 2e-5  # Conservative (v2 originals)
    lr_layers_23_28: float = 5e-5  # Higher (cloned layers)
    lr_embeddings: float = 0.0  # Frozen (except hub tokens)
    lr_hub_tokens: float = 1e-4  # Hub token embeddings
    lr_task_heads: float = 1e-4  # Task heads
    lr_lora: float = 1e-4  # LoRA adapters

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # =========================================================================
    # Gradient Configuration
    # =========================================================================
    max_grad_norm: float = 1.0
    per_layer_clip: bool = True
    log_grad_norms: bool = True
    grad_log_every: int = 100
    nan_check: bool = True
    zero_nan_grads: bool = True

    # =========================================================================
    # Hub Token Configuration
    # =========================================================================
    train_hub_tokens: list = field(default_factory=lambda: ["[EMO]", "[MEM]", "[REL]", "[TASK]"])
    freeze_original_vocab: bool = True
    hub_token_grad_scale: float = 1.0

    # =========================================================================
    # Layer Freezing Configuration
    # =========================================================================
    frozen_bands: list = field(default_factory=lambda: ["foundation", "core"])
    trainable_bands: list = field(default_factory=lambda: ["semantic", "family"])
    freeze_embeddings: bool = True
    freeze_hub_tokens: bool = False

    # =========================================================================
    # LoRA Configuration (Applied to layers 23-28)
    # =========================================================================
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_layers: list = field(default_factory=lambda: [22, 23, 24, 25, 26, 27])  # 0-indexed

    # =========================================================================
    # Scheduler Configuration
    # =========================================================================
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.01

    # =========================================================================
    # Data Configuration
    # =========================================================================
    familyos_data_dir: str = "data/familyos/unified/output"
    familyos_shard_pattern: str = "shard_*.jsonl"
    healing_data_path: str = "data/healing/healing_enhanced.jsonl"
    replay_ratio: float = 0.15  # 15% healing replay
    num_workers: int = 4
    pin_memory: bool = True

    # Task types to train on
    tasks: list = field(
        default_factory=lambda: [
            "emotions",
            "sentiment",
            "safety_familyos",
            "intent",
            "ingress",
            "ner_family",
            "temporal",
            "relations",
        ]
    )

    # =========================================================================
    # Hub-Weighted Loss Configuration
    # =========================================================================
    hub_active_weight: float = 1.0
    hub_inactive_weight: float = 0.3
    safety_multiplier: float = 2.0
    always_train_safety: bool = True

    task_base_weights: dict = field(
        default_factory=lambda: {
            "emotions": 1.0,
            "sentiment": 1.0,
            "safety_familyos": 1.0,
            "intent": 0.8,
            "ingress": 0.8,
            "ner_family": 1.0,
            "temporal": 1.0,
            "relations": 1.2,
        }
    )

    # =========================================================================
    # Output Configuration
    # =========================================================================
    output_dir: str = "outputs/v3_phase1"
    checkpoint_dir: str = "outputs/v3_phase1/checkpoints"
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
    wandb_tags: list = field(default_factory=lambda: ["phase_1", "multitask", "familyos", "v3"])

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
            interface_lr=float(self.lr_layers_23_28),  # L23 gets same as family
            family_lr=float(self.lr_layers_23_28),
            family_graduated=False,  # No graduated decay in Phase 1
            frozen_lr=float(self.lr_layers_1_18),
            embeddings_lr=float(self.lr_embeddings),
            task_heads_lr=float(self.lr_task_heads),
        )

    def get_gradient_config(self) -> GradientClipConfig:
        """Create GradientClipConfig from this config."""
        return GradientClipConfig(
            max_grad_norm=self.max_grad_norm,
            per_layer_clip=self.per_layer_clip,
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

    def get_hub_loss_config(self) -> HubLossConfig:
        """Create HubLossConfig from this config."""
        return HubLossConfig(
            active_weight=self.hub_active_weight,
            inactive_weight=self.hub_inactive_weight,
            safety_multiplier=self.safety_multiplier,
            always_train_safety=self.always_train_safety,
            task_base_weights=self.task_base_weights,
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
    def from_dict(cls, d: dict[str, Any]) -> Phase1Config:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path | str) -> Phase1Config:
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
    learning rate schedule, per-task metrics, and hub activation statistics.
    """

    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    best_step: int = 0
    phase: str = "phase_1"
    losses: list[float] = field(default_factory=list)
    task_losses: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    hub_activations: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)

    def update_loss(self, loss: float, task_losses: dict[str, float] | None = None) -> None:
        """Add loss to history, maintaining rolling window."""
        self.losses.append(loss)
        if len(self.losses) > 500:
            self.losses = self.losses[-500:]

        # Track per-task losses
        if task_losses:
            for task, task_loss in task_losses.items():
                self.task_losses[task].append(task_loss)
                if len(self.task_losses[task]) > 500:
                    self.task_losses[task] = self.task_losses[task][-500:]

    def update_hub_activations(self, hub_routing: HubRouting) -> None:
        """Track hub activation counts."""
        for hub in hub_routing.active_hubs:
            self.hub_activations[hub] += 1

    def update_metrics(self, metrics: dict[str, Any], lr: float | None = None) -> None:
        """Add evaluation metrics to history."""
        metrics_with_step = {"step": self.global_step, **metrics}
        self.metrics_history.append(metrics_with_step)
        if lr is not None:
            self.lr_history.append(lr)
        if len(self.metrics_history) > 50:
            self.metrics_history = self.metrics_history[-50:]

    def update_best(self, metric_value: float) -> bool:
        """Update best metric if improved. Returns True if new best."""
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

    def get_task_average_losses(self, window: int = 100) -> dict[str, float]:
        """Get average loss per task over recent window."""
        return {
            task: sum(losses[-window:]) / len(losses[-window:])
            for task, losses in self.task_losses.items()
            if losses
        }

    def get_hub_activation_ratios(self) -> dict[str, float]:
        """Get hub activation ratios."""
        total = sum(self.hub_activations.values())
        if total == 0:
            return {}
        return {hub: count / total for hub, count in self.hub_activations.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "phase": self.phase,
            "losses": self.losses[-100:],
            "task_losses": {k: v[-100:] for k, v in self.task_losses.items()},
            "hub_activations": dict(self.hub_activations),
            "metrics_history": self.metrics_history[-10:],
            "lr_history": self.lr_history[-10:],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingState:
        """Create state from dictionary."""
        state = cls(
            global_step=d.get("global_step", 0),
            epoch=d.get("epoch", 0),
            best_metric=d.get("best_metric", float("inf")),
            best_step=d.get("best_step", 0),
            phase=d.get("phase", "phase_1"),
            losses=d.get("losses", []),
            metrics_history=d.get("metrics_history", []),
            lr_history=d.get("lr_history", []),
        )
        # Restore task losses
        for task, losses in d.get("task_losses", {}).items():
            state.task_losses[task] = losses
        # Restore hub activations
        for hub, count in d.get("hub_activations", {}).items():
            state.hub_activations[hub] = count
        return state


# =============================================================================
# FamilyOS Dataset
# =============================================================================


class FamilyOSDataset(Dataset):
    """
    FamilyOS dataset for Phase 1 multi-task training.

    Loads unified JSONL samples with hub_routing and 8 task types.
    Supports both file-based and shard-based loading.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        """
        Initialize FamilyOSDataset.

        Args:
            data_path: Path to JSONL file or directory with shards
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_samples: Maximum samples to load (None = all)
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[UnifiedSample] = []

        data_path = Path(data_path)

        if data_path.is_dir():
            # Load from shard directory
            self._load_shards(data_path, max_samples)
        elif data_path.exists():
            # Load from single file
            self._load_file(data_path, max_samples)
        else:
            logger.warning(f"Data path not found: {data_path}")
            self._create_synthetic_samples(max_samples or 100)

        logger.info(f"Loaded {len(self.samples)} FamilyOS samples")

    def _load_file(self, path: Path, max_samples: int | None) -> None:
        """Load samples from a single JSONL file."""
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                if line.strip():
                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)
                    self.samples.append(sample)
                    count += 1

    def _load_shards(self, shard_dir: Path, max_samples: int | None) -> None:
        """Load samples from shard directory."""
        import glob

        shard_files = sorted(glob.glob(str(shard_dir / "shard_*.jsonl")))

        if not shard_files:
            shard_files = sorted(glob.glob(str(shard_dir / "*.jsonl")))

        count = 0
        for shard_path in shard_files:
            if max_samples and count >= max_samples:
                break
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    if max_samples and count >= max_samples:
                        break
                    if line.strip():
                        data = json.loads(line)
                        sample = UnifiedSample.from_json(data)
                        self.samples.append(sample)
                        count += 1

    def _create_synthetic_samples(self, num_samples: int) -> None:
        """Create synthetic samples for testing."""
        import random

        hub_configs = [
            {"EMO": True, "REL": False, "MEM": False, "TASK": False},
            {"EMO": False, "REL": True, "MEM": False, "TASK": False},
            {"EMO": False, "REL": False, "MEM": True, "TASK": False},
            {"EMO": False, "REL": False, "MEM": False, "TASK": True},
            {"EMO": True, "REL": True, "MEM": False, "TASK": False},
        ]

        for i in range(num_samples):
            hub_config = random.choice(hub_configs)
            sample_data = {
                "id": f"synthetic_{i}",
                "text": f"This is synthetic sample number {i} for testing.",
                "hub_routing": hub_config,
                "tasks": {
                    "sentiment": random.choice(["positive", "negative", "neutral"]),
                    "emotions": random.sample(["joy", "sadness", "anger"], k=random.randint(1, 2)),
                },
            }
            sample = UnifiedSample.from_json(sample_data)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length - V3_SPECIAL_PREFIX_LEN - 1,  # Reserve for hub tokens + SEP
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "sample": sample,
        }


# =============================================================================
# Healing Replay Dataset
# =============================================================================


class HealingReplayDataset(Dataset):
    """
    Healing dataset for replay sampling during Phase 1.

    Loads healing data from Phase 0.5 to prevent catastrophic forgetting.
    Uses 15% replay ratio as specified in enhanced_design_v3.md.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
        max_samples: int | None = None,
    ):
        """
        Initialize HealingReplayDataset.

        Args:
            data_path: Path to healing JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_samples: Maximum samples to load
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, Any]] = []

        data_path = Path(data_path)
        if data_path.exists():
            self._load_file(data_path, max_samples)
        else:
            logger.warning(f"Healing data not found: {data_path}")

        logger.info(f"Loaded {len(self.samples)} healing replay samples")

    def _load_file(self, path: Path, max_samples: int | None) -> None:
        """Load samples from JSONL file."""
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                if line.strip():
                    self.samples.append(json.loads(line))
                    count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.get("text", ""),
            max_length=self.max_length - V3_SPECIAL_PREFIX_LEN - 1,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "task": sample.get("task", "healing"),
            "label": sample.get("label", 0),
            "is_replay": True,
        }


# =============================================================================
# Phase 1 Collator
# =============================================================================


class Phase1Collator(V3BaseCollator):
    """
    Collator for Phase 1 multi-task training.

    Extends V3BaseCollator to:
    - Handle UnifiedSample objects with hub_routing
    - Extract task labels from samples
    - Support mixed batches (FamilyOS + healing replay)
    - Compute hub gradient masks
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        hub_routing_parser: HubRoutingParser | None = None,
    ):
        """Initialize Phase1Collator."""
        config = V3CollatorConfig(
            max_length=max_length,
            padding=padding,
            include_hub_tokens=True,
        )
        super().__init__(tokenizer=tokenizer, config=config)

        self.hub_routing_parser = hub_routing_parser or HubRoutingParser()

    def _build_v3_sequence(
        self,
        input_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Build v3 sequence with hub tokens.

        Input:  [CLS] <text> [SEP] [PAD]...
        Output: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...

        Args:
            input_ids: Original token IDs

        Returns:
            Tuple of (new_input_ids, attention_mask)
        """
        # Create attention mask (1 for non-pad, 0 for pad)
        pad_id = self.tokenizer.pad_token_id
        attention_mask = [1 if tok != pad_id else 0 for tok in input_ids]

        # Add hub tokens using base class method
        new_ids, new_mask = self._add_hub_tokens(input_ids, attention_mask)

        # Pad/truncate to max_length
        max_len = self.config.max_length
        if len(new_ids) < max_len:
            pad_len = max_len - len(new_ids)
            new_ids = new_ids + [pad_id] * pad_len
            new_mask = new_mask + [0] * pad_len
        elif len(new_ids) > max_len:
            new_ids = new_ids[:max_len]
            new_mask = new_mask[:max_len]
            # Ensure SEP at end
            new_ids[-1] = self.tokenizer.sep_token_id

        return new_ids, new_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate batch of samples.

        Args:
            batch: List of samples from dataset

        Returns:
            Collated batch with:
            - input_ids, attention_mask
            - hub_routings: List of HubRouting objects
            - task_labels: Dict mapping task -> labels tensor
            - hub_masks: Tensor [batch_size, 4] for hub activation
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_hub_routings = []
        batch_samples = []
        batch_is_replay = []

        for item in batch:
            input_ids = item["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            # Build v3 sequence with hub tokens
            v3_ids, attention_mask = self._build_v3_sequence(input_ids)

            batch_input_ids.append(v3_ids)
            batch_attention_mask.append(attention_mask)

            # Handle FamilyOS samples
            if "sample" in item:
                sample: UnifiedSample = item["sample"]
                batch_hub_routings.append(sample.hub_routing)
                batch_samples.append(sample)
                batch_is_replay.append(False)
            else:
                # Healing replay sample - create default hub routing
                batch_hub_routings.append(HubRouting(emo=True))  # Default to EMO for healing
                batch_samples.append(None)
                batch_is_replay.append(True)

        # Build hub masks tensor
        hub_masks = torch.stack([hr.to_tensor() for hr in batch_hub_routings])

        # Build task labels
        task_labels = self._extract_task_labels(batch_samples, batch)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "hub_routings": batch_hub_routings,
            "hub_masks": hub_masks,
            "task_labels": task_labels,
            "is_replay": batch_is_replay,
        }

    def _extract_task_labels(
        self,
        samples: list[UnifiedSample | None],
        raw_batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract task labels from samples."""
        labels = {
            "sentiment": [],
            "emotions": [],
            "safety_familyos": [],
            "intent": [],
            "ingress": [],
            "ner_family": [],
            "temporal": [],
            "relations": [],
        }

        # Label vocabularies (simplified - should use V3LabelVocabularies)
        sentiment_vocab = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "very_negative": 3,
            "very_positive": 4,
        }
        safety_vocab = {"safe": 0, "caution": 1, "crisis": 2}
        intent_vocab = {
            "query": 0,
            "command": 1,
            "statement": 2,
            "greeting": 3,
            "farewell": 4,
            "thanks": 5,
            "apology": 6,
            "other": 7,
        }
        ingress_vocab = {
            "text": 0,
            "voice": 1,
            "reminder": 2,
            "photo": 3,
            "location": 4,
            "schedule": 5,
            "contact": 6,
            "health": 7,
            "finance": 8,
            "shopping": 9,
            "weather": 10,
            "other": 11,
        }

        for i, sample in enumerate(samples):
            if sample is None:
                # Healing replay - use raw batch label
                raw_item = raw_batch[i]
                task = raw_item.get("task", "sentiment")
                label = raw_item.get("label", 0)
                if task == "sentiment":
                    labels["sentiment"].append(label if isinstance(label, int) else 0)
                else:
                    labels["sentiment"].append(-100)  # Ignore
                # Mark all other tasks as ignore for replay samples
                for task_name in [
                    "emotions",
                    "safety_familyos",
                    "intent",
                    "ingress",
                    "ner_family",
                    "temporal",
                    "relations",
                ]:
                    labels[task_name].append(-100 if task_name != task else label)
            else:
                # FamilyOS sample - extract from UnifiedSample
                # Sentiment
                if sample.sentiment:
                    labels["sentiment"].append(sentiment_vocab.get(sample.sentiment, -100))
                else:
                    labels["sentiment"].append(-100)

                # Emotions (multi-label - store as list, will need special handling)
                labels["emotions"].append(sample.emotions if sample.emotions else [])

                # Safety
                if sample.safety_familyos:
                    labels["safety_familyos"].append(safety_vocab.get(sample.safety_familyos, -100))
                else:
                    labels["safety_familyos"].append(-100)

                # Intent
                if sample.intent:
                    labels["intent"].append(intent_vocab.get(sample.intent, -100))
                else:
                    labels["intent"].append(-100)

                # Ingress
                if sample.ingress:
                    labels["ingress"].append(ingress_vocab.get(sample.ingress, -100))
                else:
                    labels["ingress"].append(-100)

                # Span annotations (NER, Temporal) - store as lists
                labels["ner_family"].append(sample.ner_family)
                labels["temporal"].append(sample.temporal)

                # Relations - store as lists
                labels["relations"].append(sample.relations)

        # Convert classification labels to tensors
        for task in ["sentiment", "safety_familyos", "intent", "ingress"]:
            labels[task] = torch.tensor(labels[task], dtype=torch.long)

        return labels


# =============================================================================
# Combined Dataset with Replay
# =============================================================================


class CombinedReplayDataset(Dataset):
    """
    Combined dataset with replay sampling for Phase 1.

    Implements 85% FamilyOS + 15% healing replay ratio.
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        replay_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize combined dataset.

        Args:
            primary_dataset: Main FamilyOS dataset
            replay_dataset: Healing replay dataset
            replay_ratio: Ratio of replay samples (0.15 = 15%)
            seed: Random seed
        """
        self.primary = primary_dataset
        self.replay = replay_dataset
        self.replay_ratio = replay_ratio
        self.rng = torch.Generator().manual_seed(seed)

        # Calculate effective length
        primary_len = len(primary_dataset)
        replay_samples_needed = int(primary_len * replay_ratio / (1 - replay_ratio))
        self.total_len = primary_len + min(replay_samples_needed, len(replay_dataset))

        logger.info(
            f"Combined dataset: {primary_len} primary + {replay_samples_needed} replay "
            f"(ratio={replay_ratio:.1%})"
        )

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Determine if this should be a replay sample
        if torch.rand(1, generator=self.rng).item() < self.replay_ratio and len(self.replay) > 0:
            replay_idx = idx % len(self.replay)
            return self.replay[replay_idx]
        else:
            primary_idx = idx % len(self.primary)
            return self.primary[primary_idx]


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_familyos_dataset(config: Phase1Config, tokenizer) -> FamilyOSDataset:
    """Load FamilyOS dataset from configured path."""
    return FamilyOSDataset(
        data_path=config.familyos_data_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )


def load_healing_replay_dataset(config: Phase1Config, tokenizer) -> HealingReplayDataset:
    """Load healing replay dataset."""
    return HealingReplayDataset(
        data_path=config.healing_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )


def create_phase1_dataloader(
    config: Phase1Config,
    tokenizer,
    split: str = "train",
) -> DataLoader:
    """
    Create DataLoader for Phase 1 training.

    Args:
        config: Phase1Config
        tokenizer: HuggingFace tokenizer
        split: Data split ('train' or 'eval')

    Returns:
        DataLoader with combined FamilyOS + healing replay
    """
    # Load datasets
    familyos_dataset = load_familyos_dataset(config, tokenizer)
    healing_dataset = load_healing_replay_dataset(config, tokenizer)

    # Create combined dataset with replay
    if len(healing_dataset) > 0 and config.replay_ratio > 0:
        combined_dataset = CombinedReplayDataset(
            primary_dataset=familyos_dataset,
            replay_dataset=healing_dataset,
            replay_ratio=config.replay_ratio,
            seed=config.seed,
        )
    else:
        combined_dataset = familyos_dataset

    # Create collator
    collator = Phase1Collator(
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    # Create dataloader
    batch_size = config.train_batch_size if split == "train" else config.eval_batch_size

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=(split == "train"),
    )

    return dataloader


# =============================================================================
# Phase 1 Training Model
# =============================================================================


class Phase1TrainingModel(nn.Module):
    """
    Training wrapper for ModernBERTv3Ultra in Phase 1 multi-task training.

    Adds task-specific heads for all 8 FamilyOS task types:
        - [EMO] hub: emotions (multi-label), sentiment, safety_familyos
        - [REL] hub: relations
        - [MEM] hub: temporal, ner_family
        - [TASK] hub: intent, ingress

    Architecture:
        - Wraps ModernBERTv3Ultra backbone
        - Task heads read from designated hub token positions
        - Computes hub-weighted multi-task loss
    """

    def __init__(
        self,
        model: ModernBERTv3Ultra,
        config: Phase1Config,
    ):
        """
        Initialize Phase1TrainingModel.

        Args:
            model: ModernBERTv3Ultra backbone
            config: Phase1Config with task settings
        """
        super().__init__()
        self.model = model
        self.config = config

        hidden_size = model.config.hidden_size

        # =================================================================
        # Task Heads - organized by hub token
        # =================================================================

        # [EMO] Hub (position 1) - Affective tasks
        self.sentiment_head = nn.Linear(hidden_size, 5)  # 5-class sentiment
        self.emotions_head = nn.Linear(hidden_size, 44)  # Multi-label emotions
        self.safety_head = nn.Linear(hidden_size, 3)  # Safe/Caution/Crisis

        # [REL] Hub (position 3) - Relational tasks
        self.relations_head = nn.Linear(hidden_size, 15)  # Relation types

        # [MEM] Hub (position 2) - Memory tasks
        # NER and temporal use token-level heads
        self.ner_head = nn.Linear(hidden_size, 21)  # NER family tags
        self.temporal_head = nn.Linear(hidden_size, 12)  # Temporal tags

        # [TASK] Hub (position 4) - Task routing
        self.intent_head = nn.Linear(hidden_size, 8)  # Intent classes
        self.ingress_head = nn.Linear(hidden_size, 12)  # Ingress types

        # =================================================================
        # Loss Functions
        # =================================================================
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

        # Hub loss weight calculator
        self.hub_loss_calc = HubLossWeightCalculator(config.get_hub_loss_config())

    @property
    def encoder(self):
        """Expose encoder for LayerFreezer compatibility."""
        return self.model.encoder

    @property
    def embeddings(self):
        """Expose embeddings for gradient masking."""
        return self.model.embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hub_routings: list[HubRouting] | None = None,
        hub_masks: torch.Tensor | None = None,
        task_labels: dict[str, Any] | None = None,
        return_task_losses: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass with hub-weighted multi-task loss.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            hub_routings: List of HubRouting objects per sample
            hub_masks: Hub activation mask [batch_size, 4]
            task_labels: Dict mapping task -> labels
            return_task_losses: Whether to return per-task losses

        Returns:
            Dict with loss, logits, and per-task losses
        """
        # Forward through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Extract hub token representations
        emo_repr = hidden_states[:, POSITION_EMO, :]  # [batch, hidden]
        mem_repr = hidden_states[:, POSITION_MEM, :]
        rel_repr = hidden_states[:, POSITION_REL, :]
        task_repr = hidden_states[:, POSITION_TASK, :]

        # =================================================================
        # Compute logits for all tasks
        # =================================================================
        logits = {
            "sentiment": self.sentiment_head(emo_repr),
            "emotions": self.emotions_head(emo_repr),
            "safety_familyos": self.safety_head(emo_repr),
            "relations": self.relations_head(rel_repr),
            "intent": self.intent_head(task_repr),
            "ingress": self.ingress_head(task_repr),
        }

        # Token-level tasks (NER, Temporal) use full sequence
        # Skip hub token positions for token classification
        token_hidden = hidden_states[:, POSITION_TEXT_START:, :]  # Skip [CLS] + hubs
        logits["ner_family"] = self.ner_head(token_hidden)
        logits["temporal"] = self.temporal_head(token_hidden)

        # =================================================================
        # Compute hub-weighted loss if labels provided
        # =================================================================
        total_loss = torch.tensor(0.0, device=device)
        task_losses = {}

        if task_labels is not None and hub_routings is not None:
            # Classification tasks with hub weighting
            for task_name in ["sentiment", "safety_familyos", "intent", "ingress"]:
                if task_name in task_labels:
                    labels = task_labels[task_name]
                    if isinstance(labels, torch.Tensor) and labels.device != device:
                        labels = labels.to(device)

                    # Compute per-sample weights based on hub routing
                    has_labels = [lbl.item() != -100 for lbl in labels]
                    weights = self.hub_loss_calc.compute_batch_weights(
                        task_name, hub_routings, has_labels
                    ).to(device)

                    # Compute loss with weights
                    raw_loss = self.ce_loss(logits[task_name], labels)
                    weighted_loss = (raw_loss * weights).sum() / (weights.sum() + 1e-8)

                    task_losses[task_name] = weighted_loss.item()
                    total_loss = total_loss + weighted_loss

            # Multi-label emotions (special handling)
            if "emotions" in task_labels:
                emotions_labels = task_labels["emotions"]
                # Convert emotion lists to binary tensor
                # This requires emotion vocabulary mapping
                # For now, skip if not properly formatted
                pass  # TODO: Implement multi-label emotion loss

        result = {
            "loss": total_loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

        if return_task_losses:
            result["task_losses"] = task_losses

        return result


# =============================================================================
# Training Loop
# =============================================================================


def train_phase_1(
    model: Phase1TrainingModel,
    train_loader: DataLoader,
    config: Phase1Config,
    state: TrainingState,
) -> dict[str, Any]:
    """
    Execute Phase 1 multi-task training loop.

    Args:
        model: Phase1TrainingModel
        train_loader: Training DataLoader
        config: Phase1Config
        state: TrainingState for resumption

    Returns:
        Dict with training results
    """
    device = torch.device(config.device)
    model = model.to(device)

    # =========================================================================
    # 1. Setup Layer Freezing
    # =========================================================================
    logger.info("Setting up layer freezing for Phase 1...")

    freezer = LayerFreezer(model)

    # Freeze L1-18 (Foundation + Core bands)
    for band_name in config.frozen_bands:
        band = LayerBand[band_name.upper()]
        freezer.freeze_band(band)
        logger.info(f"Frozen band: {band_name}")

    # Unfreeze L19-28 (Semantic + Family bands)
    for band_name in config.trainable_bands:
        band = LayerBand[band_name.upper()]
        freezer.unfreeze_band(band)
        logger.info(f"Trainable band: {band_name}")

    freeze_stats = freezer.get_freeze_stats()
    frozen_count = freeze_stats["frozen_params"]
    trainable_count = freeze_stats["trainable_params"]
    logger.info(f"Frozen: {frozen_count:,} | Trainable: {trainable_count:,}")
    logger.info(f"Trainable ratio: {100 * trainable_count / (frozen_count + trainable_count):.1f}%")

    # =========================================================================
    # 2. Setup Hub Token Gradient Masking
    # =========================================================================
    logger.info("Setting up hub token gradient masking...")
    setup_hub_token_gradient_masking(
        model,
        train_hub_tokens=config.train_hub_tokens,
        freeze_original_vocab=config.freeze_original_vocab,
    )

    # =========================================================================
    # 3. Apply LoRA to layers 23-28 (if enabled)
    # =========================================================================
    if config.use_lora and LORA_AVAILABLE:
        logger.info(f"Applying LoRA to layers {config.lora_target_layers}")
        lora_config = LoRAConfig(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_layers=config.lora_target_layers,
        )
        apply_lora_to_model(model.model, lora_config)
        logger.info(f"LoRA applied (r={config.lora_r}, alpha={config.lora_alpha})")
    elif config.use_lora and not LORA_AVAILABLE:
        logger.warning("LoRA requested but not available")

    # =========================================================================
    # 4. Setup Optimizer with Zipper LR
    # =========================================================================
    logger.info("Creating optimizer with Zipper LR...")
    zipper_config = config.get_zipper_lr_config()
    zipper_builder = ZipperLROptimizer(model, zipper_config, weight_decay=config.weight_decay)
    optimizer = zipper_builder.create_optimizer()
    logger.info(f"Optimizer created with {len(optimizer.param_groups)} parameter groups")

    # =========================================================================
    # 5. Setup Scheduler
    # =========================================================================
    total_steps = config.max_steps
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config.scheduler_type,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.min_lr_ratio,
    )
    logger.info(f"Scheduler: {config.scheduler_type} with {config.warmup_steps} warmup steps")

    # =========================================================================
    # 6. Setup Gradient Clipping
    # =========================================================================
    grad_config = config.get_gradient_config()
    grad_clipper = GradientClipper(model, grad_config)

    # =========================================================================
    # 7. Initialize W&B
    # =========================================================================
    if config.use_wandb and WANDB_AVAILABLE:
        run_name = config.wandb_run_name or f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.to_dict(),
            tags=config.wandb_tags,
            resume="allow" if state.global_step > 0 else None,
        )
        logger.info(f"W&B initialized: {run_name}")

    # =========================================================================
    # 8. Training Loop
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Starting Phase 1 Multi-Task FamilyOS Training")
    logger.info("=" * 60)
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Batch size: {config.train_batch_size} x {config.gradient_accumulation_steps}")
    logger.info(f"Replay ratio: {config.replay_ratio:.1%}")
    logger.info("=" * 60)

    model.train()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / "config.json")

    pbar = tqdm(total=config.max_steps - state.global_step, desc="Phase 1 Training")
    accumulation_step = 0
    accumulated_loss = 0.0

    data_iter = iter(train_loader)

    while state.global_step < config.max_steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            state.epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hub_routings = batch["hub_routings"]
        hub_masks = batch["hub_masks"].to(device)
        task_labels = batch["task_labels"]

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float32):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hub_routings=hub_routings,
                hub_masks=hub_masks,
                task_labels=task_labels,
            )
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        scaled_loss = loss / config.gradient_accumulation_steps
        scaled_loss.backward()

        accumulated_loss += loss.item()
        accumulation_step += 1

        # Update hub activation tracking
        for hr in hub_routings:
            state.update_hub_activations(hr)

        # Optimizer step after accumulation
        if accumulation_step >= config.gradient_accumulation_steps:
            # Gradient clipping
            grad_stats = grad_clipper.clip_gradients()
            grad_norm = grad_stats.total_norm if grad_stats else 0.0

            # Check for NaN gradients
            has_nan = False
            if config.nan_check:
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan = True
                        if config.zero_nan_grads:
                            p.grad.zero_()

            if not has_nan:
                optimizer.step()
                scheduler.step()

            optimizer.zero_grad()

            # Update state
            avg_loss = accumulated_loss / config.gradient_accumulation_steps
            state.update_loss(avg_loss, outputs.get("task_losses"))
            state.global_step += 1

            accumulation_step = 0
            accumulated_loss = 0.0

            pbar.update(1)

            # Logging
            if state.global_step % config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_recent_loss = state.get_average_loss(100)
                task_avg_losses = state.get_task_average_losses(100)
                hub_ratios = state.get_hub_activation_ratios()

                log_dict = {
                    "train/loss": avg_recent_loss,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm,
                    "train/step": state.global_step,
                    "train/epoch": state.epoch,
                }

                # Add per-task losses
                for task, task_loss in task_avg_losses.items():
                    log_dict[f"train/loss_{task}"] = task_loss

                # Add hub activation ratios
                for hub, ratio in hub_ratios.items():
                    log_dict[f"hub/{hub}_ratio"] = ratio

                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(log_dict, step=state.global_step)

                pbar.set_postfix(
                    loss=f"{avg_recent_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    grad=f"{grad_norm:.2f}",
                )

            # Save checkpoint
            if state.global_step % config.save_steps == 0:
                checkpoint_path = output_dir / f"checkpoint-{state.global_step}"
                save_checkpoint(model, optimizer, scheduler, state, config, checkpoint_path)

    pbar.close()

    # =========================================================================
    # 9. Final Save
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 1 Training Complete!")
    logger.info("=" * 60)

    # Save final model
    final_path = output_dir / "final_model"
    save_checkpoint(model, optimizer, scheduler, state, config, final_path)

    # Log final metrics
    final_metrics = {
        "total_steps": state.global_step,
        "final_loss": state.get_average_loss(100),
        "task_losses": state.get_task_average_losses(100),
        "hub_activations": state.get_hub_activation_ratios(),
    }

    logger.info(f"Final loss: {final_metrics['final_loss']:.4f}")
    logger.info("Task losses:")
    for task, loss in final_metrics["task_losses"].items():
        logger.info(f"  {task}: {loss:.4f}")
    logger.info("Hub activation ratios:")
    for hub, ratio in final_metrics["hub_activations"].items():
        logger.info(f"  {hub}: {ratio:.1%}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({"final": final_metrics})
        wandb.finish()

    return {
        "total_steps": state.global_step,
        "final_loss": final_metrics["final_loss"],
        "task_losses": final_metrics["task_losses"],
        "hub_activations": final_metrics["hub_activations"],
        "output_dir": str(output_dir),
    }


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    model: Phase1TrainingModel,
    optimizer: Any,
    scheduler: Any,
    state: TrainingState,
    config: Phase1Config,
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), path / "pytorch_model.bin")

    # Save optimizer and scheduler if provided
    if optimizer is not None:
        opt_state = {"optimizer": optimizer.state_dict()}
        if scheduler is not None:
            opt_state["scheduler"] = scheduler.state_dict()
        torch.save(opt_state, path / "optimizer.pt")

    # Save training state
    with open(path / "training_state.json", "w") as f:
        json.dump(state.to_dict(), f, indent=2)

    # Save config
    config.save(path / "config.json")

    # Save model config if available
    if hasattr(model.model, "config"):
        model_config = model.model.config
        if hasattr(model_config, "save_pretrained"):
            model_config.save_pretrained(path)
        elif hasattr(model_config, "to_dict"):
            # For V3Config objects, serialize to JSON
            with open(path / "model_config.json", "w") as f:
                json.dump(model_config.to_dict(), f, indent=2)

    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model: Phase1TrainingModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> TrainingState:
    """Load training checkpoint."""
    # Load model weights
    state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)

    # Load optimizer and scheduler if provided
    if optimizer is not None and (path / "optimizer.pt").exists():
        opt_state = torch.load(path / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(opt_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(opt_state["scheduler"])

    # Load training state
    with open(path / "training_state.json") as f:
        state = TrainingState.from_dict(json.load(f))

    logger.info(f"Checkpoint loaded from {path} (step {state.global_step})")
    return state


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Multi-Task FamilyOS Training for ModernBERT v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (validate configuration)
    python scripts/train_v3_phase1.py --dry-run

    # Smoke test (10 steps)
    python scripts/train_v3_phase1.py --smoke-test

    # Debug mode (5 steps with verbose logging)
    python scripts/train_v3_phase1.py --debug

    # Full training
    python scripts/train_v3_phase1.py \\
        --config configs/training/multitask/stage_v3_phase1.yaml \\
        --model-path outputs/v3_phase0_5/best_model \\
        --output-dir outputs/v3_phase1

    # Resume training
    python scripts/train_v3_phase1.py \\
        --resume-from outputs/v3_phase1/checkpoint-5000
        """,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_v3_phase1.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to Phase 0.5 trained model (overrides config)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint directory",
    )

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Base learning rate")
    parser.add_argument("--replay-ratio", type=float, help="Healing replay ratio")
    parser.add_argument("--batch-size", type=int, help="Training batch size")

    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Logging
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Quick test modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without training",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10 steps for quick validation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run 5 steps with verbose logging",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 precision",
    )

    return parser.parse_args()


def load_config_from_args(args: argparse.Namespace) -> Phase1Config:
    """Load and merge configuration from args and YAML."""
    # Try loading YAML config
    config_dict = {}
    if YAML_AVAILABLE and Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            # Flatten nested config
            config_dict = flatten_config(yaml_config)
            logger.info(f"Loaded config from {args.config}")

    # Create config from dict or defaults
    config = Phase1Config.from_dict(config_dict) if config_dict else Phase1Config()

    # Apply CLI overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.base_lr = args.learning_rate
    if args.replay_ratio is not None:
        config.replay_ratio = args.replay_ratio
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config.use_wandb = False
    if args.device:
        config.device = args.device
    if args.no_bf16:
        config.bf16 = False

    # Handle test modes
    if args.smoke_test:
        config.max_steps = 10
        config.logging_steps = 2
        config.save_steps = 10
        config.eval_steps = 10
        config.use_wandb = False
        config.warmup_steps = 3  # Scale warmup for smoke test
        logger.info("Smoke test mode: 10 steps")

    if args.debug:
        config.max_steps = 5
        config.logging_steps = 1
        config.save_steps = 5
        config.eval_steps = 5
        config.use_wandb = False
        config.log_grad_norms = True
        config.grad_log_every = 1
        config.warmup_steps = 2  # Scale warmup for debug mode
        logger.info("Debug mode: 5 steps with verbose logging")

    return config


def flatten_config(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dict to flat dict with underscores."""
    result = {}
    for key, value in nested.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_config(value, flat_key))
        else:
            # Convert keys like training_max_steps to max_steps
            # by trying both formats
            result[flat_key] = value
            result[key] = value
    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for Phase 1 training."""
    args = parse_args()

    print()
    print("=" * 60)
    print("ModernBERT v3 - Phase 1 Multi-Task FamilyOS Training")
    print("=" * 60)

    # Load configuration
    config = load_config_from_args(args)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if HF_AVAILABLE:
        set_seed(config.seed)

    print(f"Config: {args.config}")
    print(f"Model: {config.model_path}")
    print(f"Data: {config.familyos_data_dir}")
    print(f"Replay: {config.replay_ratio:.1%}")
    print(f"Output: {config.output_dir}")
    print(f"Max steps: {config.max_steps}")
    print(f"Device: {config.device}")
    print(f"BF16: {config.bf16}")
    print()

    # Dry run - just validate config
    if args.dry_run:
        print("Dry run mode - validating configuration...")
        print()
        print("Configuration validated successfully!")
        print(f"  Model path: {config.model_path}")
        print(f"  FamilyOS data: {config.familyos_data_dir}")
        print(f"  Healing data: {config.healing_data_path}")
        print(f"  Replay ratio: {config.replay_ratio:.1%}")
        print(f"  Max steps: {config.max_steps}")
        print(f"  Batch size: {config.train_batch_size}")
        print(f"  Learning rate: {config.base_lr}")
        print(f"  LoRA: {config.use_lora} (r={config.lora_r}, alpha={config.lora_alpha})")
        print()
        return 0

    # Check model path
    model_path = Path(config.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run Phase 0.5 training first:")
        logger.error(
            "  python scripts/train_v3_phase0_5.py --model-path checkpoints/v3-initialized-from-v2"
        )
        return 1

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        # Add hub tokens if not present
        hub_tokens = get_all_hub_tokens()
        existing_tokens = set(tokenizer.get_vocab().keys())
        new_tokens = [t for t in hub_tokens if t not in existing_tokens]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info(f"Added hub tokens: {new_tokens}")
    else:
        logger.error("HuggingFace transformers not available")
        return 1

    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        # Try loading as ModernBERTv3Ultra
        model_config = ModernBERTv3Config()
        if (model_path / "config.json").exists():
            with open(model_path / "config.json") as f:
                config_dict = json.load(f)
            model_config = ModernBERTv3Config(
                **{
                    k: v
                    for k, v in config_dict.items()
                    if k in ModernBERTv3Config.__dataclass_fields__
                }
            )

        backbone = ModernBERTv3Ultra(model_config)

        # Load weights
        weights_path = model_path / "pytorch_model.bin"
        if not weights_path.exists():
            weights_path = model_path / "model.safetensors"

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            # Handle nested state dict from Phase05TrainingModel
            if "model.embeddings" in str(list(state_dict.keys())[:5]):
                # Strip "model." prefix if present
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
            backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning("No model weights found, using initialized model")

        # Resize embeddings for hub tokens
        backbone.resize_token_embeddings(len(tokenizer))

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Create training model wrapper
    model = Phase1TrainingModel(backbone, config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create data loader
    logger.info("Creating data loader...")
    train_loader = create_phase1_dataloader(config, tokenizer, split="train")
    logger.info(f"Data loader created with {len(train_loader)} batches")

    # Initialize training state
    state = TrainingState()

    # Resume from checkpoint if specified
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            logger.info(f"Resuming from {resume_path}")
            state = load_checkpoint(resume_path, model)
        else:
            logger.warning(f"Checkpoint not found: {resume_path}")

    # Run training
    try:
        results = train_phase_1(model, train_loader, config, state)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = Path(config.output_dir) / "emergency_checkpoint"
        save_checkpoint(model, None, None, state, config, emergency_path)
        return 130

    # Print results
    print()
    print("=" * 60)
    print("Phase 1 Training Complete!")
    print("=" * 60)
    print(f"Total steps: {results['total_steps']}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Output: {results['output_dir']}")
    print()
    print("Task losses:")
    for task, loss in results.get("task_losses", {}).items():
        print(f"  {task}: {loss:.4f}")
    print()
    print("Hub activation ratios:")
    for hub, ratio in results.get("hub_activations", {}).items():
        print(f"  {hub}: {ratio:.1%}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

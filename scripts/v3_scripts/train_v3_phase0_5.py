#!/usr/bin/env python
"""
Phase 0.5 Enhanced Healing Training Script for ModernBERT v3

This script implements the "healing" phase that repairs the cloned layers
(L23-28) and establishes smooth activation flow across the L22->L23 interface.

Phase 0.5 Objectives:
    1. Heal L23-28 (cloned from L15-20) to work coherently
    2. Smooth the L22->L23 interface transition
    3. Preserve L1-22 frozen capabilities
    4. Train hub tokens for routing semantics

Training Strategy:
    - Freeze: L1-18 (Foundation + Core bands)
    - Train: L19-22 (SEMANTIC), L23-28 (Family), Hub tokens
    - LR: Zipper strategy with L23 at maximum plasticity
    - Data: Enhanced healing (SST-2, CoNLL, MNLI, SQuAD, STS-B)

Layer Band Architecture (v3.3 Ultra - 28 layers):
    - Foundation (L1-6):   window=64,  FROZEN in Phase 0.5
    - Core (L7-18):        window=128, FROZEN in Phase 0.5
    - SEMANTIC (L19-22):     window=256, trainable (low LR)
    - Family (L23-28):     window=512, trainable (L23 highest LR)

Zipper LR Strategy:
    L19-22 (SEMANTIC):   1e-5   - gentle adaptation
    L23 (Interface):   5e-5   - maximum plasticity
    L24-28 (Family):   3e-5   - graduated decay

Usage:
    # Dry run (validate configuration)
    python scripts/v3_scripts/train_v3_phase0_5.py --dry-run

    # Smoke test (10 steps)
    python scripts/v3_scripts/train_v3_phase0_5.py --smoke-test

    # Debug mode (5 steps with gradient logging)
    python scripts/v3_scripts/train_v3_phase0_5.py --debug

    # Full training
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml \\
        --model-path checkpoints/v3-initialized-from-v2 \\
        --output-dir outputs/v3_phase0_5

    # Resume from checkpoint
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --resume-from outputs/v3_phase0_5/checkpoint-1000

    # With overrides
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --learning-rate 3e-5 \\
        --max-steps 3000 \\
        --wandb-run-name "phase0_5_experiment_1"

Environment:
    - GPU: 24GB+ VRAM recommended (16GB with gradient checkpointing)
    - RAM: 32GB+ recommended
    - Python: 3.10+
    - PyTorch: 2.0+

Outputs:
    - outputs/v3_phase0_5/: Checkpoints and final model
    - outputs/v3_phase0_5/best/: Best model by validation loss
    - wandb/: W&B logs (if enabled)

Post-Training:
    After Phase 0.5, proceed to Phase 1:
    python scripts/v3_scripts/train_v3_phase1.py \\
        --model-path outputs/v3_phase0_5/best

Author: FamilyOS Team
Date: December 2025
Version: 3.0 (Production Quality)
Epic: 5.4.1 - Phase 0.5 Healing Training Script
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# v3 Infrastructure Imports
# =============================================================================

# Models
from modeling_studio.models.config_v3 import ModernBERTv3Config
from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra

# Trainers
from modeling_studio.trainers.freezing_v3 import LayerFreezer, TrainingPhase
from modeling_studio.trainers.zipper_lr_v3 import ZipperLRConfig, ZIPPER_PRESETS
from modeling_studio.trainers.gradient_utils_v3 import GradientClipConfig, GradientClipper

# Optional: v2 initialization
try:
    from modeling_studio.models.initialization_v3 import initialize_from_v2

    V2_INIT_AVAILABLE = True
except ImportError:
    V2_INIT_AVAILABLE = False

# Optional: W&B logging
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Transformers for public datasets
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, set_seed

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    set_seed = None

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Hub token IDs for v3 architecture
HUB_TOKEN_IDS = {
    "[EMO]": 50368,
    "[MEM]": 50369,
    "[REL]": 50370,
    "[TASK]": 50371,
}

# Layer band boundaries (0-indexed)
LAYER_BANDS = {
    "foundation": list(range(0, 6)),  # L1-6
    "core": list(range(6, 18)),  # L7-18
    "SEMANTIC": list(range(18, 22)),  # L19-22
    "family": list(range(22, 28)),  # L23-28
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Phase05Config:
    """
    Configuration for Phase 0.5 healing training.

    This configuration integrates:
        - ZipperLRConfig for layer-wise learning rates
        - GradientClipConfig for gradient management
        - Layer freezing for Phase 0.5
        - Training hyperparameters

    Attributes:
        model_path: Path to initialized v3 model checkpoint
        v2_checkpoint: Path to v2 checkpoint for initialization
        tokenizer_name: Tokenizer to use
        max_steps: Maximum training steps
        warmup_steps: Number of warmup steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        lr_frozen: LR for frozen layers (L1-18)
        lr_SEMANTIC: LR for SEMANTIC layers (L19-22)
        lr_interface: LR for interface layer (L23)
        lr_family: LR for family layers (L24-28)
        lr_embeddings: LR for embeddings
        lr_heads: LR for task heads
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm (global)
        interface_clip: Tighter clip at L23 interface
        per_layer_clip: Whether to use per-layer clipping
        output_dir: Output directory for checkpoints
        device: Device to use (cuda/cpu)
        bf16: Whether to use bf16 mixed precision
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        seed: Random seed
    """

    # Model paths
    model_path: str = "checkpoints/v3-initialized-from-v2"
    v2_checkpoint: str = (
        "checkpoints/modernbert-v2-for-v3-transfer/checkpoint-4000/model.safetensors"
    )
    tokenizer_name: str = "answerdotai/ModernBERT-base"

    # Training hyperparameters
    max_steps: int = 2500
    warmup_steps: int = 500
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # Batch settings
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_length: int = 512

    # Zipper LR strategy (from zipper_lr_v3.py)
    # Phase 0.5 Zipper Profile:
    #   L1-18:  0.0    (frozen)
    #   L19-22: 1e-5   (SEMANTIC - gentle adaptation)
    #   L23:    5e-5   (interface - maximum plasticity)
    #   L24-28: 3e-5   (family - learning new patterns)
    lr_frozen: float = 0.0  # L1-18 frozen
    lr_SEMANTIC: float = 1e-5  # L19-22 SEMANTIC
    lr_interface: float = 5e-5  # L23 interface (max plasticity)
    lr_family: float = 3e-5  # L24-28 family
    lr_embeddings: float = 0.0  # Embeddings frozen
    lr_heads: float = 3e-5  # Task heads

    # Optimizer settings
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Gradient clipping (from gradient_utils_v3.py)
    max_grad_norm: float = 1.0
    interface_clip: float = 0.5  # Tighter clip at L23
    per_layer_clip: bool = True
    log_grad_norms: bool = True
    grad_log_every: int = 50

    # Output
    output_dir: str = "outputs/v3_phase0_5"
    checkpoint_dir: str = "checkpoints/v3_phase0_5"

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False

    # Logging
    use_wandb: bool = False
    wandb_project: str = "modernbert-v3"
    wandb_run_name: str | None = None

    # Data
    healing_data_path: str = "data/healing/healing_enhanced.jsonl"
    use_synthetic_data: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Seed
    seed: int = 42

    def get_zipper_lr_config(self) -> ZipperLRConfig:
        """Create ZipperLRConfig from this config."""
        return ZipperLRConfig(
            base_lr=self.lr_family,
            SEMANTIC_lr=self.lr_SEMANTIC,
            interface_lr=self.lr_interface,
            family_lr=self.lr_family,
            frozen_lr=self.lr_frozen,
            embeddings_lr=self.lr_embeddings,
            task_heads_lr=self.lr_heads,
        )

    def get_gradient_config(self) -> GradientClipConfig:
        """Create GradientClipConfig from this config."""
        return GradientClipConfig(
            max_grad_norm=self.max_grad_norm,
            per_layer_clip=self.per_layer_clip,
            interface_clip=self.interface_clip,
            log_grad_norms=self.log_grad_norms,
            log_every_n_steps=self.grad_log_every,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            "model_path": self.model_path,
            "v2_checkpoint": self.v2_checkpoint,
            "tokenizer_name": self.tokenizer_name,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "lr_frozen": self.lr_frozen,
            "lr_SEMANTIC": self.lr_SEMANTIC,
            "lr_interface": self.lr_interface,
            "lr_family": self.lr_family,
            "lr_embeddings": self.lr_embeddings,
            "lr_heads": self.lr_heads,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "interface_clip": self.interface_clip,
            "per_layer_clip": self.per_layer_clip,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "device": self.device,
            "bf16": self.bf16,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "seed": self.seed,
        }


def load_config_from_yaml(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config or {}


def apply_config_overrides(config: Phase05Config, yaml_config: dict) -> Phase05Config:
    """Apply YAML config overrides to Phase05Config."""
    # Training section
    training = yaml_config.get("training", {})
    if "max_steps" in training:
        config.max_steps = training["max_steps"]
    if "warmup_steps" in training:
        config.warmup_steps = training["warmup_steps"]
    if "eval_steps" in training:
        config.eval_steps = training["eval_steps"]
    if "save_steps" in training:
        config.save_steps = training["save_steps"]
    if "logging_steps" in training:
        config.logging_steps = training["logging_steps"]
    if "per_device_train_batch_size" in training:
        config.train_batch_size = training["per_device_train_batch_size"]
    if "per_device_eval_batch_size" in training:
        config.eval_batch_size = training["per_device_eval_batch_size"]
    if "gradient_accumulation_steps" in training:
        config.gradient_accumulation_steps = training["gradient_accumulation_steps"]
    if "seed" in training:
        config.seed = training["seed"]
    if "bf16" in training:
        config.bf16 = training["bf16"]
    if "gradient_checkpointing" in training:
        config.gradient_checkpointing = training["gradient_checkpointing"]

    # Learning rate section
    lr_config = yaml_config.get("learning_rate", {})
    if "SEMANTIC_lr" in lr_config:
        config.lr_SEMANTIC = float(lr_config["SEMANTIC_lr"])
    if "interface_lr" in lr_config:
        config.lr_interface = float(lr_config["interface_lr"])
    if "family_lr" in lr_config:
        config.lr_family = float(lr_config["family_lr"])
    if "embeddings_lr" in lr_config:
        config.lr_embeddings = float(lr_config["embeddings_lr"])
    if "task_heads_lr" in lr_config:
        config.lr_heads = float(lr_config["task_heads_lr"])

    # Gradient section
    gradient = yaml_config.get("gradient", {})
    if "max_grad_norm" in gradient:
        config.max_grad_norm = float(gradient["max_grad_norm"])
    if "interface_clip" in gradient:
        config.interface_clip = float(gradient["interface_clip"])
    if "per_layer_clip" in gradient:
        config.per_layer_clip = gradient["per_layer_clip"]

    # Optimizer section
    optimizer = yaml_config.get("optimizer", {})
    if "weight_decay" in optimizer:
        config.weight_decay = float(optimizer["weight_decay"])

    # Data section
    data = yaml_config.get("data", {})
    if "healing_data_path" in data:
        config.healing_data_path = data["healing_data_path"]

    # Output section
    output = yaml_config.get("checkpointing", {})
    if "output_dir" in output:
        config.output_dir = output["output_dir"]
    if "checkpoint_dir" in output:
        config.checkpoint_dir = output["checkpoint_dir"]

    # Logging section
    logging_cfg = yaml_config.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    if "enabled" in wandb_cfg:
        config.use_wandb = wandb_cfg["enabled"]
    if "project" in wandb_cfg:
        config.wandb_project = wandb_cfg["project"]
    if "run_name" in wandb_cfg:
        config.wandb_run_name = wandb_cfg["run_name"]

    return config


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0.5 Enhanced Healing Training for ModernBERT v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (validate configuration)
    python scripts/v3_scripts/train_v3_phase0_5.py --dry-run

    # Smoke test (10 steps)
    python scripts/v3_scripts/train_v3_phase0_5.py --smoke-test

    # Debug mode (5 steps with gradient logging)
    python scripts/v3_scripts/train_v3_phase0_5.py --debug

    # Full training with config file
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --config configs/training/multitask/stage_v3_phase0_5_enhanced.yaml

    # Full training with overrides
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --model-path checkpoints/v3-initialized-from-v2 \\
        --output-dir outputs/v3_phase0_5 \\
        --learning-rate 5e-5 \\
        --max-steps 3000

    # Resume from checkpoint
    python scripts/v3_scripts/train_v3_phase0_5.py \\
        --resume-from outputs/v3_phase0_5/checkpoint-1000
""",
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

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML file",
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/v3-initialized-from-v2",
        help="Path to initialized v3 model checkpoint",
    )
    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        default="checkpoints/modernbert-v2-for-v3-transfer/checkpoint-4000/model.safetensors",
        help="Path to v2 checkpoint for initialization if model not found",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v3_phase0_5",
        help="Output directory for checkpoints and final model",
    )

    # Training overrides
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        dest="learning_rate",
        help="Interface layer learning rate (L23)",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 mixed precision",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Phase05Config:
    """Create Phase05Config from parsed arguments."""
    config = Phase05Config()

    # Load YAML config if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        config = apply_config_overrides(config, yaml_config)

    # Apply CLI overrides (take precedence over YAML)
    if args.model_path:
        config.model_path = args.model_path
    if args.v2_checkpoint:
        config.v2_checkpoint = args.v2_checkpoint
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.train_batch_size = args.batch_size
    if args.learning_rate is not None:
        config.lr_interface = args.learning_rate
    if args.wandb:
        config.use_wandb = True
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.device:
        config.device = args.device
    if args.no_bf16:
        config.bf16 = False
    if args.seed is not None:
        config.seed = args.seed

    return config


# =============================================================================
# Healing Dataset with Public Data
# =============================================================================


class HealingDataset(Dataset):
    """
    Healing dataset using public benchmarks for Phase 0.5.

    Loads from HuggingFace datasets:
        - SST-2: Sentiment (binary)
        - MNLI: NLI (3-way)
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 512,
        tasks: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        if tasks is None:
            tasks = ["sentiment", "nli"]

        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available, using synthetic data")
            self._create_synthetic_samples(max_samples or 1000)
            return

        # Load public datasets
        if "sentiment" in tasks:
            self._load_sst2(split, max_samples)

        if "nli" in tasks:
            self._load_mnli(split, max_samples)

        logger.info(f"Loaded {len(self.samples)} healing samples for {split}")

    def _load_sst2(self, split: str, max_samples: int | None) -> None:
        """Load SST-2 sentiment data."""
        try:
            ds_split = "validation" if split == "validation" else "train"
            ds = load_dataset("glue", "sst2", split=ds_split, trust_remote_code=True)

            count = 0
            for item in ds:
                if max_samples and count >= max_samples // 2:
                    break

                encoding = self.tokenizer(
                    item["sentence"],
                    max_length=self.max_length - 5,
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
                if max_samples and count >= max_samples // 2:
                    break

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

    def _create_synthetic_samples(self, num_samples: int) -> None:
        """Create synthetic samples when HF not available."""
        for i in range(num_samples):
            input_ids = torch.randint(5, 50368, (128,)).tolist()
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * 128,
                    "task": "sentiment" if i % 2 == 0 else "nli",
                    "label": i % 2 if i % 2 == 0 else i % 3,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


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

            if input_ids[0] == self.cls_token_id:
                input_ids = input_ids[1:]
            if input_ids[-1] == self.sep_token_id:
                input_ids = input_ids[:-1]

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

            pad_len = self.max_length - len(v3_ids)
            attention_mask = [1] * len(v3_ids) + [0] * pad_len
            v3_ids = v3_ids + [self.pad_token_id] * pad_len

            batch_input_ids.append(v3_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(sample["label"])
            batch_tasks.append(sample.get("task", "sentiment"))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "tasks": batch_tasks,
        }


# =============================================================================
# Training Wrapper with Loss Computation
# =============================================================================


@dataclass
class Phase05Output:
    """Output container for Phase 0.5 training."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor


class Phase05TrainingModel(nn.Module):
    """
    Wrapper that makes ModernBERTv3Ultra compatible with trainer.

    Uses hub tokens for task-specific classification:
    - [EMO] at position 1 for sentiment/safety
    - [REL] at position 3 for NLI
    """

    def __init__(self, model: ModernBERTv3Ultra):
        super().__init__()
        self.model = model

        hidden_size = model.config.hidden_size
        self.sentiment_head = nn.Linear(hidden_size, 2)
        self.nli_head = nn.Linear(hidden_size, 3)

        self.loss_fn = nn.CrossEntropyLoss()

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
        """Forward pass that computes loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        batch_size = input_ids.size(0)

        total_loss = torch.tensor(0.0, device=input_ids.device)
        all_logits = []

        for i in range(batch_size):
            task = tasks[i] if tasks else "sentiment"

            if task == "sentiment":
                pooled = outputs.last_hidden_state[i, 1, :]
                logits = self.sentiment_head(pooled)
            else:
                pooled = outputs.last_hidden_state[i, 3, :]
                logits = self.nli_head(pooled)

            all_logits.append(logits)

            if labels is not None:
                loss = self.loss_fn(logits.unsqueeze(0), labels[i : i + 1])
                total_loss = total_loss + loss

        if labels is not None:
            total_loss = total_loss / batch_size

        logits_tensor = torch.stack(all_logits)

        return Phase05Output(
            loss=total_loss if labels is not None else None,
            logits=logits_tensor,
            hidden_states=outputs.last_hidden_state,
        )


# =============================================================================
# Model Setup
# =============================================================================


def setup_model(config: Phase05Config) -> ModernBERTv3Ultra:
    """Setup v3 model from checkpoint or initialize from v2."""
    model_path = Path(config.model_path)

    if model_path.exists() and (model_path / "pytorch_model.bin").exists():
        logger.info(f"Loading model from {model_path}")
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)

        state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    else:
        logger.info("Initializing v3 model from v2 checkpoint...")
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)

        v2_path = Path(config.v2_checkpoint)
        if v2_path.exists() and V2_INIT_AVAILABLE:
            stats = initialize_from_v2(model, str(v2_path))
            logger.info(f"Initialized {stats.transferred_params:,} parameters from v2")
        else:
            logger.warning("v2 checkpoint not found or init not available, using random init")

    return model


def setup_layer_freezing(model: nn.Module) -> LayerFreezer:
    """Configure layer freezing for Phase 0.5 using freezing_v3.py."""
    base_model = model.model if hasattr(model, "model") else model

    freezer = LayerFreezer(base_model)
    freezer.configure_for_phase(TrainingPhase.PHASE_0_5)

    stats = freezer.get_freeze_stats()
    logger.info(f"Frozen: {stats['frozen_params']:,} | Trainable: {stats['trainable_params']:,}")

    return freezer


def create_optimizer_with_zipper_lr(
    model: nn.Module,
    config: Phase05Config,
) -> torch.optim.AdamW:
    """Create optimizer with Zipper LR strategy from zipper_lr_v3.py."""
    zipper_config = config.get_zipper_lr_config()
    base_model = model.model if hasattr(model, "model") else model

    param_groups = []

    # SEMANTIC layers (L19-22, indices 18-21)
    SEMANTIC_params = []
    for i in range(18, 22):
        if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layers"):
            if len(base_model.encoder.layers) > i:
                SEMANTIC_params.extend(
                    [p for p in base_model.encoder.layers[i].parameters() if p.requires_grad]
                )

    if SEMANTIC_params:
        param_groups.append(
            {
                "params": SEMANTIC_params,
                "lr": zipper_config.SEMANTIC_lr,
                "name": "SEMANTIC_L19-22",
            }
        )

    # Interface layer (L23, index 22)
    interface_params = []
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layers"):
        if len(base_model.encoder.layers) > 22:
            interface_params = [
                p for p in base_model.encoder.layers[22].parameters() if p.requires_grad
            ]

    if interface_params:
        param_groups.append(
            {
                "params": interface_params,
                "lr": zipper_config.interface_lr,
                "name": "interface_L23",
            }
        )

    # Family layers (L24-28, indices 23-27)
    family_params = []
    for i in range(23, 28):
        if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layers"):
            if len(base_model.encoder.layers) > i:
                family_params.extend(
                    [p for p in base_model.encoder.layers[i].parameters() if p.requires_grad]
                )

    if family_params:
        param_groups.append(
            {
                "params": family_params,
                "lr": zipper_config.family_lr,
                "name": "family_L24-28",
            }
        )

    # Task heads
    head_params = []
    if hasattr(model, "sentiment_head"):
        head_params.extend(list(model.sentiment_head.parameters()))
    if hasattr(model, "nli_head"):
        head_params.extend(list(model.nli_head.parameters()))

    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": zipper_config.task_heads_lr,
                "name": "task_heads",
            }
        )

    # Other trainable params
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

    logger.info("Optimizer parameter groups (Zipper LR):")
    for group in param_groups:
        num_params = sum(p.numel() for p in group["params"])
        logger.info(f"  {group['name']}: {num_params:,} params, lr={group['lr']}")

    return torch.optim.AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


# =============================================================================
# Data Loading
# =============================================================================


def create_dataloaders(
    config: Phase05Config,
    tokenizer,
    synthetic: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    if synthetic:
        train_dataset = HealingDataset(
            tokenizer,
            split="train",
            max_samples=1000,
            max_length=config.max_length,
        )
        val_dataset = HealingDataset(
            tokenizer,
            split="validation",
            max_samples=100,
            max_length=config.max_length,
        )
    else:
        train_dataset = HealingDataset(
            tokenizer,
            split="train",
            max_samples=10000,
            max_length=config.max_length,
        )
        val_dataset = HealingDataset(
            tokenizer,
            split="validation",
            max_samples=1000,
            max_length=config.max_length,
        )

    collator = HealingCollator(tokenizer, max_length=config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    return train_loader, val_loader


# =============================================================================
# Training Loop with GradientClipper
# =============================================================================


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    clipper: GradientClipper,
    config: Phase05Config,
    step: int = 0,
) -> tuple[float, dict]:
    """Single training step with gradient monitoring via GradientClipper."""
    model.train()

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    tasks = batch.get("tasks", ["sentiment"] * len(labels))

    optimizer.zero_grad()

    if config.bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

    loss.backward()

    # Use GradientClipper from gradient_utils_v3.py
    grad_stats = clipper.clip_gradients()

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
    """Run evaluation."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

            total_loss += outputs.loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    return {
        "eval_loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "eval_accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    output_dir: str,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    import shutil

    output_path = Path(output_dir)
    checkpoint_dir = output_path / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_model = model.model if hasattr(model, "model") else model
    torch.save(base_model.state_dict(), checkpoint_dir / "pytorch_model.bin")

    if hasattr(model, "sentiment_head"):
        torch.save(
            {
                "sentiment_head": model.sentiment_head.state_dict(),
                "nli_head": model.nli_head.state_dict() if hasattr(model, "nli_head") else None,
            },
            checkpoint_dir / "heads.pt",
        )

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_dir / "training_state.pt",
    )

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"step": step, "is_best": is_best}, f)

    logger.info(f"Saved checkpoint: {checkpoint_dir}")

    if is_best:
        best_dir = output_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Saved best model: {best_dir}")


# =============================================================================
# Training Modes
# =============================================================================


def run_dry_run(config: Phase05Config) -> bool:
    """Run dry-run validation without actual training."""
    print("\n" + "=" * 60)
    print("Phase 0.5 Dry Run Validation")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    checks_total += 1
    model_path = Path(config.model_path)
    v2_path = Path(config.v2_checkpoint)
    if model_path.exists() or v2_path.exists():
        print("[OK] Model source available")
        checks_passed += 1
    else:
        print("[FAIL] No model source found")

    checks_total += 1
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU available: {gpu_name} ({gpu_mem:.1f}GB)")
        checks_passed += 1
    else:
        print("[WARN] No GPU available, using CPU")
        checks_passed += 1

    checks_total += 1
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        print(f"[OK] Tokenizer loaded: {config.tokenizer_name}")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Tokenizer failed: {e}")

    checks_total += 1
    zipper_config = config.get_zipper_lr_config()
    print("[OK] Zipper LR configuration:")
    print(f"    SEMANTIC (L19-22):   {zipper_config.SEMANTIC_lr}")
    print(f"    Interface (L23):   {zipper_config.interface_lr}")
    print(f"    Family (L24-28):   {zipper_config.family_lr}")
    checks_passed += 1

    checks_total += 1
    grad_config = config.get_gradient_config()
    print("[OK] Gradient configuration:")
    print(f"    Max norm:       {grad_config.max_grad_norm}")
    print(f"    Interface clip: {grad_config.interface_clip}")
    print(f"    Per-layer clip: {grad_config.per_layer_clip}")
    checks_passed += 1

    checks_total += 1
    try:
        v3_config = ModernBERTv3Config()
        model = ModernBERTv3Ultra(v3_config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created: {param_count:,} parameters")
        checks_passed += 1
        del model
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")

    print("\n" + "-" * 60)
    print(f"Dry Run: {checks_passed}/{checks_total} checks passed")
    print("-" * 60)

    return checks_passed == checks_total


def run_smoke_test(config: Phase05Config) -> bool:
    """Run smoke test with 10 steps."""
    print("\n" + "=" * 60)
    print("Phase 0.5 Smoke Test (10 steps)")
    print("=" * 60)

    config.max_steps = 10
    config.warmup_steps = 2
    config.eval_steps = 5
    config.save_steps = 10
    config.logging_steps = 1
    config.use_wandb = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        base_model = setup_model(config)
        device = torch.device(config.device)
        base_model = base_model.to(device)

        setup_layer_freezing(base_model)

        model = Phase05TrainingModel(base_model).to(device)

        train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=True)

        optimizer = create_optimizer_with_zipper_lr(model, config)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps, eta_min=0
        )

        # Create GradientClipper from gradient_utils_v3.py
        grad_config = config.get_gradient_config()
        clipper = GradientClipper(model, grad_config)

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

            if (step + 1) % config.eval_steps == 0:
                metrics = evaluate(model, val_loader, config)
                logger.info(f"Eval @ step {step + 1}: {metrics}")

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

    Uses GradientClipper for per-step gradient monitoring.
    """
    print("\n" + "=" * 60)
    print("Phase 0.5 DEBUG Mode (5 steps with gradient logging)")
    print("=" * 60)

    config.max_steps = 5
    config.warmup_steps = 1
    config.logging_steps = 1
    config.grad_log_every = 1
    config.log_grad_norms = True
    config.use_wandb = False

    logging.getLogger().setLevel(logging.DEBUG)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        base_model = setup_model(config)
        device = torch.device(config.device)
        base_model = base_model.to(device)

        freezer = setup_layer_freezing(base_model)

        print("\n--- Layer Freezing Details ---")
        print(f"Frozen layers: {freezer.get_frozen_layers()}")
        print(f"Trainable layers: {freezer.get_trainable_layers()}")

        model = Phase05TrainingModel(base_model).to(device)

        train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=True)

        optimizer = create_optimizer_with_zipper_lr(model, config)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps, eta_min=0
        )

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

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current LR: {current_lr:.6f}")

        print("\n" + "-" * 60)
        print("DEBUG MODE COMPLETED")
        print("-" * 60)
        return True

    except Exception as e:
        print(f"\nDEBUG MODE FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_full_training(config: Phase05Config, resume_from: str | None = None) -> dict:
    """Run full Phase 0.5 training."""
    print("\n" + "=" * 60)
    print("Phase 0.5 Full Training")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    base_model = setup_model(config)
    device = torch.device(config.device)
    base_model = base_model.to(device)

    setup_layer_freezing(base_model)

    model = Phase05TrainingModel(base_model).to(device)

    train_loader, val_loader = create_dataloaders(config, tokenizer, synthetic=False)

    optimizer = create_optimizer_with_zipper_lr(model, config)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=0
    )

    grad_config = config.get_gradient_config()
    clipper = GradientClipper(model, grad_config)

    start_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if (checkpoint_path / "training_state.pt").exists():
            state = torch.load(checkpoint_path / "training_state.pt", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]
            logger.info(f"Resumed from step {start_step}")

    best_loss = float("inf")
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

        loss, _ = train_step(model, batch, optimizer, scheduler, clipper, config, step=step)
        losses.append(loss)
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        if (step + 1) % config.logging_steps == 0:
            avg_loss = sum(losses[-config.logging_steps :]) / config.logging_steps
            logger.info(f"Step {step + 1}: avg_loss={avg_loss:.4f}")

        if (step + 1) % config.eval_steps == 0:
            metrics = evaluate(model, val_loader, config)
            logger.info(f"Eval @ step {step + 1}: {metrics}")

            if metrics["eval_loss"] < best_loss:
                best_loss = metrics["eval_loss"]
                save_checkpoint(
                    model, optimizer, scheduler, step + 1, config.output_dir, is_best=True
                )

        if (step + 1) % config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, config.output_dir, is_best=False)

    save_checkpoint(model, optimizer, scheduler, config.max_steps, config.output_dir, is_best=False)

    final_metrics = evaluate(model, val_loader, config)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Final step: {config.max_steps}")
    print(f"  Final eval loss: {final_metrics['eval_loss']:.4f}")
    print(f"  Final eval accuracy: {final_metrics['eval_accuracy']:.4f}")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    return {
        "final_step": config.max_steps,
        "final_loss": losses[-1],
        "final_metrics": final_metrics,
    }


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0.5 Enhanced Healing Training for ModernBERT v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="Validate configuration")
    mode_group.add_argument("--smoke-test", action="store_true", help="Run 10-step smoke test")
    mode_group.add_argument(
        "--debug", action="store_true", help="Run 5 steps with gradient logging"
    )

    parser.add_argument("--model-path", type=str, default="checkpoints/v3-initialized-from-v2")
    parser.add_argument(
        "--v2-checkpoint",
        type=str,
        default="checkpoints/modernbert-v2-for-v3-transfer/checkpoint-4000/model.safetensors",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/v3_phase0_5")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5, help="Interface layer LR")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    args = parse_args()

    config = Phase05Config(
        model_path=args.model_path,
        v2_checkpoint=args.v2_checkpoint,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        train_batch_size=args.batch_size,
        lr_interface=args.lr,
        use_wandb=args.wandb,
        wandb_run_name=args.wandb_run_name,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

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
        run_full_training(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Stage A Training Script: Generic Multi-Task ModernBERT

This script trains ModernBERT-base on public datasets for generic NLU tasks.
Output: modernbert-multitask-v0

Tasks trained:
    - ner_general: CoNLL-2003, OntoNotes
    - sentiment: SST-2
    - emotions: GoEmotions
    - safety_generic: Jigsaw toxicity
    - nli: MNLI, SNLI
    - embedding: STS-B, NLI pairs

Usage:
    python scripts/train_stage_a.py --config configs/training/multitask/stage_a_generic.yaml

    # Override config values
    python scripts/train_stage_a.py \
        --config configs/training/multitask/stage_a_generic.yaml \
        --training.learning_rate 3e-5 \
        --training.num_train_epochs 10

    # Resume from checkpoint
    python scripts/train_stage_a.py \
        --config configs/training/multitask/stage_a_generic.yaml \
        --resume_from_checkpoint checkpoints/modernbert-multitask-v0/checkpoint-5000

Environment:
    - GPU: A100/H100 recommended (40GB+ VRAM for full batch)
    - CPU: 32+ cores for data loading
    - RAM: 64GB+ recommended

    # Multi-GPU with accelerate
    accelerate launch scripts/train_stage_a.py --config ...

    # DeepSpeed
    deepspeed scripts/train_stage_a.py --deepspeed configs/training/deepspeed/ds_z2.json

Outputs:
    - checkpoints/modernbert-multitask-v0/: Training checkpoints
    - outputs/modernbert-multitask-v0/: Final model, logs, eval results
    - wandb/: W&B logs (if enabled)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer, TrainingArguments, set_seed

# Add project root to path if running as script
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modeling_studio.data.labels import (Capability,  # noqa: E402
                                         get_num_labels)
from modeling_studio.data.loaders import load_stage_a_datasets  # noqa: E402
from modeling_studio.models.modernbert_multitask import \
    ModernBertMultiTaskModel  # noqa: E402
from modeling_studio.trainers.collators import MultiTaskCollator  # noqa: E402
from modeling_studio.trainers.ema import EMAModel  # noqa: E402
from modeling_studio.trainers.multitask_trainer import \
    MultiTaskTrainer  # noqa: E402

# Note: UncertaintyWeighting is handled internally by MultiTaskTrainer via args.use_uncertainty_weighting

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Management
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle defaults (simplified - full implementation would use OmegaConf)
    if "defaults" in config:
        # Load base configs
        for default in config.get("defaults", []):
            if isinstance(default, str) and default.startswith("/"):
                base_path = config_path.parent.parent / default.lstrip("/").replace(".", "/")
                base_path = base_path.with_suffix(".yaml")
                if base_path.exists():
                    with open(base_path) as f:
                        base_config = yaml.safe_load(f)
                    # Merge base into config (config takes precedence)
                    config = _deep_merge(base_config, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply command-line overrides to config."""
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        keys = key.split(".")

        # Navigate to the right place in config
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Parse value type
        try:
            parsed_value = yaml.safe_load(value)
        except Exception:
            parsed_value = value

        current[keys[-1]] = parsed_value

    return config


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ModernBERT multi-task model on Stage A (generic) datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_a_generic.yaml",
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/multitask/stage_a_datasets.yaml",
        help="Path to data configuration file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory from config",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller datasets, more logging)",
    )

    parser.add_argument(
        "--ignore_optimizer_state",
        action="store_true",
        help="When resuming from checkpoint, skip loading optimizer/scheduler state. "
        "Useful when optimizer configuration has changed (e.g., new parameter groups).",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and data loading without training",
    )

    # Accept additional overrides as key=value pairs
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format key.subkey=value",
    )

    return parser.parse_args()


# =============================================================================
# Model Initialization
# =============================================================================


def init_model(config: dict[str, Any]) -> ModernBertMultiTaskModel:
    """Initialize the multi-task model from config."""
    model_config = config.get("model", {})
    heads_config = config.get("heads", {})

    model_name = model_config.get("name_or_path", "answerdotai/ModernBERT-base")
    torch_dtype_str = model_config.get("torch_dtype", "bfloat16")
    use_flash_attention = model_config.get("use_flash_attention_2", False)

    # Determine enabled capabilities from heads config
    capabilities = []
    for head_name, head_cfg in heads_config.items():
        if head_cfg.get("enabled", True):
            try:
                cap = Capability(head_name)
                capabilities.append(cap)
            except ValueError:
                logger.warning(f"Unknown capability: {head_name}, skipping")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Enabled capabilities: {[c.value for c in capabilities]}")

    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    # Determine attention implementation
    # Check if flash attention is available
    attn_implementation = "sdpa"  # Default to SDPA (PyTorch native)
    if use_flash_attention:
        try:
            import flash_attn  # noqa: F401, F811

            _ = flash_attn  # Suppress unused import warning
            attn_implementation = "flash_attention_2"
            logger.info("Using Flash Attention 2.0")
        except ImportError:
            logger.warning("Flash Attention 2.0 not available, falling back to SDPA")
            attn_implementation = "sdpa"
    else:
        logger.info("Using SDPA (Scaled Dot-Product Attention)")

    # Load model
    model = ModernBertMultiTaskModel.from_pretrained(
        model_name,
        capabilities=capabilities,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    return model


def configure_head_loss(
    model: ModernBertMultiTaskModel,
    head_name: str,
    train_dataset,
    heads_config: dict[str, Any],
) -> None:
    """
    Configure loss function for a head (ASL, focal loss, class weights, label smoothing).

    This is called after model initialization to set up:
    - use_asl: Whether to use Asymmetric Loss (SOTA for multi-label, better than focal)
    - use_focal_loss: Whether to use focal loss (helps with class imbalance)
    - focal_gamma: Focal loss gamma parameter
    - class_weights: Inverse frequency weights computed from training data
    - label_smoothing: Label smoothing factor for regularization

    Args:
        model: The multi-task model
        head_name: Name of the head (e.g., "emotions", "safety_generic")
        train_dataset: Training dataset for computing class weights
        heads_config: Head configuration from YAML
    """
    head_cfg = heads_config.get(head_name, {})

    # Check if ASL, focal loss, or class weights are requested
    use_asl = head_cfg.get("use_asl", False)
    use_focal_loss = head_cfg.get("use_focal_loss", False)
    focal_gamma = float(head_cfg.get("focal_gamma", 2.0))
    compute_class_weights = head_cfg.get("compute_class_weights", False)
    label_smoothing = float(head_cfg.get("label_smoothing", 0.0))

    if not use_asl and not use_focal_loss and not compute_class_weights and label_smoothing == 0.0:
        return

    # Get the head
    try:
        head = model.get_head(head_name)
    except KeyError:
        logger.warning(f"Head '{head_name}' not found in model, skipping loss configuration")
        return

    # Set ASL parameters (takes priority over focal loss)
    if use_asl:
        head.use_asl = True
        head.asl_gamma_neg = float(head_cfg.get("asl_gamma_neg", 4.0))
        head.asl_gamma_pos = float(head_cfg.get("asl_gamma_pos", 1.0))
        head.asl_clip = float(head_cfg.get("asl_clip", 0.05))
        logger.info(
            f"  {head_name}: enabled ASL (γ-={head.asl_gamma_neg}, γ+={head.asl_gamma_pos}, clip={head.asl_clip})"
        )
    # Set focal loss parameters (only if ASL not enabled)
    elif use_focal_loss:
        head.use_focal_loss = True
        head.focal_gamma = focal_gamma
        logger.info(f"  {head_name}: enabled focal loss (gamma={focal_gamma})")

    # Set pos_weight for positive sample upweighting (helps with sparse multi-label)
    pos_weight = head_cfg.get("pos_weight")
    if pos_weight is not None:
        device = next(head.parameters()).device
        dtype = next(head.parameters()).dtype
        num_labels = head_cfg.get("num_labels", head.num_labels)
        if isinstance(pos_weight, (int, float)):
            pos_weight_tensor = torch.tensor([pos_weight] * num_labels, device=device, dtype=dtype)
        else:
            pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=dtype)
        # Just assign directly - BaseHead always registers pos_weight as a buffer
        head.pos_weight = pos_weight_tensor
        logger.info(
            f"  {head_name}: enabled pos_weight={pos_weight} for positive sample upweighting"
        )

    # Set label smoothing
    if label_smoothing > 0.0:
        head.label_smoothing = label_smoothing
        logger.info(f"  {head_name}: enabled label smoothing ({label_smoothing})")

    # Compute class weights from training data
    if compute_class_weights and train_dataset is not None:
        class_weights = _compute_class_weights_from_dataset(
            train_dataset,
            head_cfg.get("num_labels", 44),
        )
        if class_weights is not None:
            # Move to same device/dtype as head parameters
            device = next(head.parameters()).device
            dtype = next(head.parameters()).dtype
            class_weights = class_weights.to(device=device, dtype=dtype)
            head.register_buffer("class_weights", class_weights)
            head.class_weights = class_weights
            logger.info(
                f"  {head_name}: computed class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})"
            )


def _compute_class_weights_from_dataset(dataset, num_labels: int) -> torch.Tensor | None:
    """
    Compute inverse frequency class weights for multi-label classification.

    For multi-label, we count how often each label appears and compute:
        weight[i] = total_samples / (num_classes * count[i])

    This gives rare classes higher weights.
    """
    try:
        import numpy as np

        # Count label frequencies
        label_counts = np.zeros(num_labels, dtype=np.float32)

        for example in dataset:
            labels = example.get("labels")
            if labels is None:
                continue

            # Handle different label formats
            if isinstance(labels, (list, np.ndarray)):
                labels_array = np.array(labels)
                if labels_array.dtype == bool or (
                    labels_array.max() <= 1 and len(labels_array) == num_labels
                ):
                    # Multi-hot format
                    label_counts += labels_array.astype(np.float32)
                else:
                    # List of label indices
                    for idx in labels_array:
                        if 0 <= idx < num_labels:
                            label_counts[int(idx)] += 1
            elif isinstance(labels, torch.Tensor):
                labels_np = labels.numpy()
                if len(labels_np) == num_labels:
                    label_counts += labels_np.astype(np.float32)
                else:
                    for idx in labels_np:
                        if 0 <= idx < num_labels:
                            label_counts[int(idx)] += 1

        # Compute inverse frequency weights
        # Avoid division by zero
        label_counts = np.maximum(label_counts, 1.0)
        total_samples = len(dataset)
        weights = total_samples / (num_labels * label_counts)

        # Normalize to mean 1.0
        weights = weights / weights.mean()

        # Clip extreme weights to avoid instability
        weights = np.clip(weights, 0.1, 10.0)

        logger.info(
            f"    Class weight distribution: mean={weights.mean():.3f}, std={weights.std():.3f}"
        )

        return torch.from_numpy(weights)

    except Exception as e:
        logger.warning(f"Failed to compute class weights: {e}")
        return None


def init_tokenizer(config: dict[str, Any]) -> AutoTokenizer:
    """Initialize tokenizer from config."""
    model_config = config.get("model", {})
    model_name = model_config.get("name_or_path", "answerdotai/ModernBERT-base")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =============================================================================
# Dataset Loading
# =============================================================================


def load_datasets(
    config: dict[str, Any],
    data_config_path: str | Path,
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and preprocess Stage A datasets."""
    logger.info("Loading Stage A datasets...")

    # Get enabled tasks from heads config
    heads_config = config.get("heads", {})
    enabled_tasks = [task for task, cfg in heads_config.items() if cfg.get("enabled", True)]

    # Load train datasets with tokenization
    train_datasets = load_stage_a_datasets(
        split="train",
        config_path=data_config_path,
        tokenizer=tokenizer,
        apply_tokenization=True,
    )

    # Load validation datasets with tokenization
    eval_datasets = load_stage_a_datasets(
        split="validation",
        config_path=data_config_path,
        tokenizer=tokenizer,
        apply_tokenization=True,
    )

    # Filter to enabled tasks
    train_datasets = {k: v for k, v in train_datasets.items() if k in enabled_tasks}
    eval_datasets = {k: v for k, v in eval_datasets.items() if k in enabled_tasks}

    # In debug mode, take smaller subsets
    if debug:
        logger.info("Debug mode: using smaller dataset subsets")
        for task in train_datasets:
            if len(train_datasets[task]) > 1000:
                train_datasets[task] = train_datasets[task].select(range(1000))
        for task in eval_datasets:
            if len(eval_datasets[task]) > 200:
                eval_datasets[task] = eval_datasets[task].select(range(200))

    logger.info(f"Loaded {len(train_datasets)} training datasets")
    for task, ds in train_datasets.items():
        logger.info(f"  - {task}: {len(ds)} samples")

    logger.info(f"Loaded {len(eval_datasets)} evaluation datasets")
    for task, ds in eval_datasets.items():
        logger.info(f"  - {task}: {len(ds)} samples")

    return train_datasets, eval_datasets


# =============================================================================
# Training Arguments
# =============================================================================


def create_training_args(
    config: dict[str, Any],
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
    debug: bool = False,
) -> TrainingArguments:
    """Create MultiTaskTrainingArguments from config.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
        debug: If True, use smaller batch sizes for local debugging
    """
    from modeling_studio.trainers.multitask_trainer import \
        MultiTaskTrainingArguments

    training_config = config.get("training", {})
    output_config = config.get("output", {})
    mixing_config = config.get("mixing", {})

    # Determine output directory
    if output_dir is None:
        output_dir = output_config.get("output_dir", "outputs/modernbert-multitask-v0")

    # Check if bf16 is supported (CUDA device with compute capability >= 8.0)
    import torch

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bf16_config = training_config.get("bf16", True)
    fp16_config = training_config.get("fp16", False)

    # Fall back to fp32 if bf16 not supported and fp16 not explicitly enabled
    if bf16_config and not bf16_supported:
        logger.warning("bf16 not supported on this device, falling back to fp32")
        bf16_config = False
        fp16_config = False

    # Get batch sizes - override for debug mode to fit in smaller GPUs
    train_batch_size = training_config.get("per_device_train_batch_size", 16)
    eval_batch_size = training_config.get("per_device_eval_batch_size", 32)
    gradient_checkpointing = training_config.get("gradient_checkpointing", True)

    if debug:
        # Use smaller batch sizes for debug mode to fit in consumer GPUs
        train_batch_size = min(train_batch_size, 8)
        eval_batch_size = min(eval_batch_size, 16)
        gradient_checkpointing = True  # Enable to save memory
        logger.info(
            f"Debug mode: using batch_size={train_batch_size}, eval_batch_size={eval_batch_size}, gradient_checkpointing=True"
        )

    # Create MultiTaskTrainingArguments (extends TrainingArguments with V2 features)
    args = MultiTaskTrainingArguments(
        output_dir=output_dir,
        # Optimization
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        optim=training_config.get("optim", "adamw_torch_fused"),
        # Schedule
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        # Duration
        num_train_epochs=training_config.get("num_train_epochs", 5),
        max_steps=training_config.get("max_steps", -1),
        # Batch size - use debug-adjusted values
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        # Evaluation & saving
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 1000),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_avg_score"),
        greater_is_better=training_config.get("greater_is_better", True),
        # Logging
        logging_steps=training_config.get("logging_steps", 100),
        logging_first_step=True,
        report_to=training_config.get("report_to", ["tensorboard"]),
        # Mixed precision - use auto-detected values
        bf16=bf16_config,
        fp16=fp16_config,
        # Memory optimization
        gradient_checkpointing=gradient_checkpointing,
        # Misc
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        remove_unused_columns=False,  # Important for multi-task
        # Resume
        resume_from_checkpoint=resume_from_checkpoint,
        # Run name for W&B
        run_name=f"stage-a-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        # === V2 FEATURES: Multi-task specific ===
        sampling_strategy=mixing_config.get("strategy", "proportional"),
        sampling_temperature=mixing_config.get("temperature", 2.0),
        use_uncertainty_weighting=training_config.get("use_uncertainty_weighting", False),
        # === SOTA FEATURES ===
        # R-Drop regularization
        use_rdrop=training_config.get("use_rdrop", False),
        rdrop_alpha=training_config.get("rdrop_alpha", 0.5),
        # Adversarial training (FGM/PGD)
        use_adversarial=training_config.get("use_adversarial", False),
        adversarial_type=training_config.get("adversarial_type", "fgm"),
        adversarial_epsilon=training_config.get("adversarial_epsilon", 1.0),
        pgd_steps=training_config.get("pgd_steps", 3),
        pgd_alpha=training_config.get("pgd_alpha", 0.3),
        # Mixup augmentation
        use_mixup=training_config.get("use_mixup", False),
        mixup_alpha=training_config.get("mixup_alpha", 0.4),
        mixup_prob=training_config.get("mixup_prob", 0.5),
    )

    return args


# =============================================================================
# Evaluation and Metrics
# =============================================================================


def compute_metrics_factory(task_names: list[str]):
    """Create a compute_metrics function for multi-task evaluation."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred

        # Handle different output shapes
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=-1)

        # Compute basic metrics
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        f1 = f1_score(labels.flatten(), predictions.flatten(), average="weighted", zero_division=0)
        precision = precision_score(
            labels.flatten(), predictions.flatten(), average="weighted", zero_division=0
        )
        recall = recall_score(
            labels.flatten(), predictions.flatten(), average="weighted", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    return compute_metrics


# =============================================================================
# Checkpointing and Model Saving
# =============================================================================


def save_model_and_artifacts(
    model: ModernBertMultiTaskModel,
    tokenizer: AutoTokenizer,
    output_dir: str | Path,
    config: dict[str, Any],
    training_args: TrainingArguments,
    eval_results: dict[str, float] | None = None,
) -> None:
    """Save the final model and all artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")

    # Save model
    model.save_pretrained(str(output_dir))

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Save training args
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=2, default=str)

    # Save capabilities
    capabilities_info = {
        "capabilities": [c.value for c in model.capabilities],
        "num_labels": {c.value: get_num_labels(c) for c in model.capabilities},
    }
    with open(output_dir / "capabilities.json", "w") as f:
        json.dump(capabilities_info, f, indent=2)

    # Save eval results
    if eval_results:
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

    logger.info("Model and artifacts saved successfully")


# =============================================================================
# Main Training Function
# =============================================================================


def train(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run the full training pipeline."""
    # Set seed for reproducibility
    seed = args.seed if args.seed else config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Initialize tokenizer
    tokenizer = init_tokenizer(config)
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Initialize model
    model = init_model(config)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load datasets
    train_datasets, eval_datasets = load_datasets(
        config=config,
        data_config_path=args.data_config,
        tokenizer=tokenizer,
        debug=args.debug,
    )

    # === Configure head loss functions (focal loss, class weights, label smoothing) ===
    heads_config = config.get("heads", {})
    for head_name, head_cfg in heads_config.items():
        if head_cfg.get("enabled", True):
            # Check if this head needs special loss configuration
            if (
                head_cfg.get("use_focal_loss")
                or head_cfg.get("compute_class_weights")
                or head_cfg.get("label_smoothing")
            ):
                train_ds = train_datasets.get(head_name)
                configure_head_loss(model, head_name, train_ds, heads_config)

    # Dry run: validate without training
    if args.dry_run:
        logger.info("Dry run complete - config and data validated successfully")
        return {}

    # Create training arguments
    output_dir = args.output_dir or config.get("output", {}).get(
        "output_dir", "outputs/modernbert-multitask-v0"
    )
    checkpoint_dir = args.checkpoint_dir or config.get("output", {}).get(
        "checkpoint_dir", "checkpoints/modernbert-multitask-v0"
    )

    training_args = create_training_args(
        config=config,
        output_dir=checkpoint_dir,  # Checkpoints go to checkpoint_dir
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug=args.debug,
    )

    # Get task weights from config
    task_weights = config.get("task_weights", {})

    # Get training config for v2 features
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})

    # === V2 FEATURE: Head-wise Learning Rates + Layer Decay ===
    custom_optimizer = None
    custom_scheduler = None
    if optimizer_config:
        # Note: YAML safe_load parses scientific notation (e.g., 2e-5) as strings
        # So we need to convert them to float explicitly
        encoder_lr = float(optimizer_config.get("encoder_lr", 2e-5))
        head_lr = float(optimizer_config.get("head_lr", 1e-4))
        token_head_lr = float(optimizer_config.get("token_head_lr", 5e-5))
        layer_decay = float(optimizer_config.get("layer_decay", 0.95))

        logger.info("=" * 60)
        logger.info("V2 FEATURE: Head-wise Learning Rates + Layer Decay")
        logger.info(f"  encoder_lr: {encoder_lr}")
        logger.info(f"  head_lr: {head_lr}")
        logger.info(f"  token_head_lr: {token_head_lr}")
        logger.info(f"  layer_decay: {layer_decay}")
        logger.info("=" * 60)

        # Create custom optimizer with head-wise LRs AND layer-wise decay
        from modeling_studio.trainers.optimizer import \
            create_optimizer_with_layer_decay

        custom_optimizer = create_optimizer_with_layer_decay(
            model,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            token_head_lr=token_head_lr,
            layer_decay=layer_decay,
            num_layers=22,  # ModernBERT-base has 22 layers
            weight_decay=float(training_config.get("weight_decay", 0.01)),
        )

        # CRITICAL: Create scheduler for custom optimizer
        # Without this, LR stays at 0 during warmup!
        from transformers import get_scheduler

        # Calculate total training steps
        total_train_samples = sum(len(ds) for ds in train_datasets.values())
        train_batch_size = training_args.per_device_train_batch_size
        grad_accum = training_args.gradient_accumulation_steps
        num_epochs = training_args.num_train_epochs
        steps_per_epoch = total_train_samples // (train_batch_size * grad_accum)
        total_steps = int(steps_per_epoch * num_epochs)
        warmup_steps = int(total_steps * training_args.warmup_ratio)

        custom_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=custom_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(f"  Scheduler: {training_args.lr_scheduler_type}")
        logger.info(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # === V2 FEATURE: EMA Model ===
    use_ema = training_config.get("use_ema", False)
    ema_decay = float(training_config.get("ema_decay", 0.999))
    ema_model = None
    if use_ema:
        logger.info("=" * 60)
        logger.info(f"V2 FEATURE: EMA Model (decay={ema_decay})")
        logger.info("=" * 60)

    # === V2 FEATURE: Uncertainty Weighting ===
    # Note: Uncertainty weighting is now handled via MultiTaskTrainingArguments
    # and applied within the trainer's compute_loss method
    use_uncertainty_weighting = training_config.get("use_uncertainty_weighting", False)
    if use_uncertainty_weighting:
        num_tasks = len(train_datasets)
        logger.info("=" * 60)
        logger.info(f"V2 FEATURE: Uncertainty Weighting ({num_tasks} tasks)")
        logger.info("=" * 60)

    # === V2 FEATURE: Embedding Hard Negatives ===
    embedding_hard_negatives = training_config.get("embedding_hard_negatives", 0)
    if embedding_hard_negatives > 0:
        logger.info("=" * 60)
        logger.info(f"V2 FEATURE: Embedding Hard Negatives (n={embedding_hard_negatives})")
        logger.info("=" * 60)

    # Create data collator
    data_collator = MultiTaskCollator(tokenizer=tokenizer)  # type: ignore[arg-type]

    # Initialize trainer with V2 features
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        task_weights=task_weights,
        sampling_strategy=config.get("mixing", {}).get("strategy", "proportional"),
        sampling_temperature=config.get("mixing", {}).get("temperature", 2.0),
        tokenizer=tokenizer,  # type: ignore[arg-type]
        data_collator=data_collator,
        # V2 features: custom optimizer with head-wise LRs + scheduler
        optimizers=(custom_optimizer, custom_scheduler) if custom_optimizer else (None, None),
    )

    # === V2 FEATURE: Initialize EMA after trainer setup ===
    if use_ema:
        ema_model = EMAModel(model, decay=ema_decay)
        trainer.ema_model = ema_model  # Attach to trainer for updates

    # Log training info
    logger.info("=" * 60)
    logger.info("Starting Stage A Training (V2 COMPLIANT)")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Tasks: {list(train_datasets.keys())}")
    logger.info(f"Total training samples: {sum(len(ds) for ds in train_datasets.values()):,}")
    logger.info(f"Total eval samples: {sum(len(ds) for ds in eval_datasets.values()):,}")
    logger.info("--- V2 Features ---")
    logger.info(f"  Head-wise LR: {bool(optimizer_config)}")
    logger.info(f"  EMA: {use_ema} (decay={ema_decay if use_ema else 'N/A'})")
    logger.info(f"  Uncertainty Weighting: {use_uncertainty_weighting}")
    logger.info(f"  Embedding Hard Negatives: {embedding_hard_negatives}")
    logger.info("=" * 60)

    # Handle --ignore_optimizer_state flag
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint and getattr(args, "ignore_optimizer_state", False):
        # Remove optimizer/scheduler state files so they won't be loaded
        from pathlib import Path as Pth

        ckpt_path = Pth(resume_checkpoint)
        for state_file in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            state_path = ckpt_path / state_file
            if state_path.exists():
                logger.info(f"Ignoring optimizer state: removing {state_path}")
                state_path.unlink()
        logger.warning(
            "Optimizer state ignored - training will restart optimizer from scratch "
            "but model weights are preserved from checkpoint."
        )

    # Train
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # === V2 FEATURE: Update EMA after training ===
    if use_ema and ema_model is not None:
        logger.info("Applying EMA weights for final model...")
        ema_model.apply_shadow(model)

    # Log training results
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training steps: {train_result.global_step}")

    # Evaluate on all tasks
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate(eval_dataset=eval_datasets, metric_key_prefix="eval")

    # Compute average metrics
    f1_scores = [v for k, v in eval_results.items() if "f1" in k and isinstance(v, float)]
    if f1_scores:
        eval_results["eval_avg_f1"] = sum(f1_scores) / len(f1_scores)

    # Log eval results
    logger.info("Final evaluation results:")
    for metric, value in sorted(eval_results.items()):
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")

    # Save final model and artifacts
    save_model_and_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        config=config,
        training_args=training_args,
        eval_results=eval_results,
    )

    return eval_results


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command-line overrides
    if args.overrides:
        logger.info(f"Applying overrides: {args.overrides}")
        config = apply_overrides(config, args.overrides)

    # Apply argument overrides
    if args.seed:
        config.setdefault("training", {})["seed"] = args.seed

    # Log configuration
    logger.info(f"Config file: {args.config}")
    logger.info(f"Data config: {args.data_config}")

    # Log key training parameters
    training_cfg = config.get("training", {})
    logger.info(
        f"Training params: batch_size={training_cfg.get('per_device_train_batch_size')}, "
        f"grad_accum={training_cfg.get('gradient_accumulation_steps')}, "
        f"epochs={training_cfg.get('num_train_epochs')}"
    )
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from: {args.resume_from_checkpoint}")
    if args.debug:
        logger.info("Debug mode enabled")

    # Run training
    try:
        _eval_results = train(config, args)  # noqa: F841
        logger.info("Training completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())

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

from modeling_studio.data.labels import Capability, get_num_labels
from modeling_studio.data.loaders import load_stage_a_datasets
from modeling_studio.models.modernbert_multitask import \
    ModernBertMultiTaskModel
from modeling_studio.trainers.collators import MultiTaskCollator
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

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
            import flash_attn  # noqa: F401

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
) -> TrainingArguments:
    """Create TrainingArguments from config."""
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    # Determine output directory
    if output_dir is None:
        output_dir = output_config.get("output_dir", "outputs/modernbert-multitask-v0")

    # Create TrainingArguments
    args = TrainingArguments(
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
        # Batch size
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 32),
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
        # Mixed precision
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        # Memory optimization
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        # Misc
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        remove_unused_columns=False,  # Important for multi-task
        # Resume
        resume_from_checkpoint=resume_from_checkpoint,
        # Run name for W&B
        run_name=f"stage-a-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

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
    )

    # Get task weights from config
    task_weights = config.get("task_weights", {})

    # Create data collator
    data_collator = MultiTaskCollator(tokenizer=tokenizer)

    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        task_weights=task_weights,
        sampling_strategy=config.get("mixing", {}).get("strategy", "proportional"),
        sampling_temperature=config.get("mixing", {}).get("temperature", 2.0),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Log training info
    logger.info("=" * 60)
    logger.info("Starting Stage A Training")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Tasks: {list(train_datasets.keys())}")
    logger.info(f"Total training samples: {sum(len(ds) for ds in train_datasets.values()):,}")
    logger.info(f"Total eval samples: {sum(len(ds) for ds in eval_datasets.values()):,}")
    logger.info("=" * 60)

    # Train
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
        config = apply_overrides(config, args.overrides)

    # Apply argument overrides
    if args.seed:
        config.setdefault("training", {})["seed"] = args.seed

    # Log configuration
    logger.info(f"Config file: {args.config}")
    logger.info(f"Data config: {args.data_config}")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from: {args.resume_from_checkpoint}")
    if args.debug:
        logger.info("Debug mode enabled")

    # Run training
    try:
        eval_results = train(config, args)
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
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())

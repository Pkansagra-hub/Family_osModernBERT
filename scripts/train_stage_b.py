#!/usr/bin/env python
"""
Stage B Training Script: FamilyOS Domain Adaptation

This script fine-tunes modernbert-multitask-v0 with FamilyOS-specific data
using LoRA adapters to preserve generic capabilities.
Output: familyos-modernbert-unified-v1

New tasks added:
    - ner_family: Family-specific NER (kinship, nicknames)
    - ingress: Domain classification (DIARY, TASK, HEALTH, etc.)
    - safety_familyos: Policy bands (GREEN, AMBER, RED, CRISIS)
    - relation: Family relationship extraction
    - intent: User intent classification

Existing tasks (replay for anti-forgetting):
    - ner_general, sentiment, emotions, safety_generic, nli, embedding, temporal

Usage:
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml

    # Start from specific Stage A checkpoint
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_familyos.yaml \
        --model.name_or_path outputs/modernbert-multitask-v0/best

    # Adjust LoRA rank
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_familyos.yaml \
        --peft.lora.r 64

    # Debug mode (smaller batches, subset of data)
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_familyos.yaml \
        --debug

Environment:
    - GPU: Single GPU sufficient due to LoRA (16GB+ VRAM for A100)
    - RAM: 32GB+ recommended

Outputs:
    - checkpoints/familyos-modernbert-unified-v1/: Training checkpoints
    - outputs/familyos-modernbert-unified-v1/: Final merged model
    - outputs/familyos-modernbert-unified-v1-lora/: LoRA adapters only

Post-Training:
    After training, run threshold calibration:
    python scripts/calibrate_safety.py --model outputs/familyos-modernbert-unified-v1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, TrainingArguments

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data import load_stage_b_datasets
from modeling_studio.data.labels import Capability
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer, MultiTaskTrainingArguments

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Stage B capabilities (FamilyOS domain)
STAGE_B_CAPABILITIES = [
    Capability.NER_FAMILY,
    Capability.INGRESS,
    Capability.SAFETY_FAMILYOS,
    Capability.RELATION,
    Capability.INTENT,
]

# Stage A capabilities (for replay)
STAGE_A_CAPABILITIES = [
    Capability.NER_GENERAL,
    Capability.SENTIMENT,
    Capability.EMOTIONS,
    Capability.SAFETY_GENERIC,
    Capability.NLI,
    Capability.EMBEDDING,
    Capability.TEMPORAL,
]

# All capabilities for unified model
ALL_CAPABILITIES = STAGE_A_CAPABILITIES + STAGE_B_CAPABILITIES

# Safety oversampling factors (from v2 plan)
SAFETY_OVERSAMPLING = {
    "CRISIS": 20,  # 20x oversampling for CRISIS
    "RED": 5,  # 5x oversampling for RED
}


# =============================================================================
# Config Loading
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides to config (e.g., --model.name_or_path=...)"""
    omega_conf = OmegaConf.create(config)

    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            key = key.lstrip("-")
            # Try to parse value as YAML for proper typing
            try:
                parsed_value = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed_value = value
            OmegaConf.update(omega_conf, key, parsed_value, merge=True)

    return OmegaConf.to_container(omega_conf, resolve=True)


# =============================================================================
# Model Initialization
# =============================================================================


def load_stage_a_model(
    model_path: str | Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ModernBertMultiTaskModel:
    """
    Load Stage A checkpoint as base model.

    Args:
        model_path: Path to Stage A model/checkpoint
        device: Device to load model on

    Returns:
        ModernBertMultiTaskModel with Stage A weights
    """
    model_path = Path(model_path)

    logger.info(f"Loading Stage A model from {model_path}")

    # Load using checkpoint loader (handles safetensors + capabilities.json)
    model = ModernBertMultiTaskModel.load_checkpoint(
        checkpoint_path=str(model_path),
        device=device,
    )

    logger.info(f"Loaded model with capabilities: {[c.value for c in model.capabilities]}")
    return model


def add_stage_b_heads(
    model: ModernBertMultiTaskModel,
    config: dict[str, Any],
) -> ModernBertMultiTaskModel:
    """
    Add FamilyOS-specific heads to the model.

    Args:
        model: Base model with Stage A heads
        config: Configuration with familyos_heads settings

    Returns:
        Model with Stage B heads added
    """
    from modeling_studio.data.labels import get_num_labels
    from modeling_studio.models import CAPABILITY_TO_HEAD_TYPE, get_problem_type
    from modeling_studio.models.heads import SafetyHead

    familyos_heads = config.get("familyos_heads", {})
    hidden_size = model.config.hidden_size

    for cap in STAGE_B_CAPABILITIES:
        cap_name = cap.value
        head_config = familyos_heads.get(cap_name, {})

        if not head_config.get("enabled", True):
            logger.info(f"Skipping disabled head: {cap_name}")
            continue

        # Skip if already exists (shouldn't happen for Stage B heads)
        if cap_name in model.heads:
            logger.info(f"Head already exists: {cap_name}")
            continue

        # Get head class and parameters
        head_cls = CAPABILITY_TO_HEAD_TYPE.get(cap)
        num_labels = get_num_labels(cap)
        problem_type = get_problem_type(cap)
        dropout = head_config.get("dropout", 0.1)

        logger.info(f"Adding head: {cap_name} (num_labels={num_labels})")

        # Create head based on capability type
        if cap == Capability.SAFETY_FAMILYOS:
            head = SafetyHead(
                hidden_size=hidden_size,
                num_bands=4,  # GREEN, AMBER, RED, CRISIS
                dropout=dropout,
                problem_type=problem_type,
            )
        else:
            head = head_cls(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout,
                problem_type=problem_type,
            )

        model.heads[cap_name] = head

    # Update capabilities list
    model.capabilities = ALL_CAPABILITIES

    logger.info(f"Model now has {len(model.heads)} heads: {list(model.heads.keys())}")
    return model


def apply_lora(
    model: ModernBertMultiTaskModel,
    config: dict[str, Any],
) -> PeftModel:
    """
    Apply LoRA adapters to the model encoder.

    Per v2 plan:
        - r=32, alpha=64
        - target: q, k, v, o projections
        - Heads remain full-precision trainable

    Args:
        model: Multi-task model
        config: Configuration with peft settings

    Returns:
        PeftModel with LoRA adapters
    """
    peft_config = config.get("peft", {})
    lora_config = peft_config.get("lora", peft_config)  # Handle nested or flat

    # Extract LoRA parameters
    r = lora_config.get("r", 32)
    lora_alpha = lora_config.get("lora_alpha", 64)
    lora_dropout = lora_config.get("lora_dropout", 0.05)
    bias = lora_config.get("bias", "none")
    target_modules = lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    logger.info(f"Applying LoRA: r={r}, alpha={lora_alpha}, targets={target_modules}")

    # Create LoRA config
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,  # For encoder models
    )

    # Apply LoRA to model
    # Note: PEFT applies to the full model, so heads remain trainable
    peft_model = get_peft_model(model, lora_cfg)

    # Enable input gradients for gradient checkpointing compatibility
    # This is required when using gradient checkpointing with PEFT
    peft_model.enable_input_require_grads()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return peft_model


def freeze_stage_a_heads(
    model: ModernBertMultiTaskModel,
    config: dict[str, Any],
) -> None:
    """
    Freeze Stage A heads that should not be updated.

    Per v2 plan:
        - ner_general: frozen (generic NER preserved)
        - safety_generic: frozen (baseline safety)
        - nli: frozen
        - Others: fine-tuned with small LR
    """
    heads_config = config.get("heads", {})

    for cap in STAGE_A_CAPABILITIES:
        cap_name = cap.value
        head_cfg = heads_config.get(cap_name, {})

        if cap_name not in model.heads:
            continue

        freeze = head_cfg.get("freeze", False)

        if freeze:
            logger.info(f"Freezing head: {cap_name}")
            for param in model.heads[cap_name].parameters():
                param.requires_grad = False
        else:
            logger.info(f"Head trainable: {cap_name}")


# =============================================================================
# Data Loading
# =============================================================================


def load_datasets_for_stage_b(
    config: dict[str, Any],
    data_config_path: str | Path,
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load Stage B datasets: FamilyOS + replay data.

    Args:
        config: Training config
        data_config_path: Path to Stage B data config
        tokenizer: Tokenizer for preprocessing
        debug: If True, use smaller subsets

    Returns:
        Tuple of (train_datasets, eval_datasets)
    """
    logger.info("Loading Stage B datasets...")
    logger.info(f"Data config: {data_config_path}")

    # Load all Stage B datasets (FamilyOS + replay)
    train_datasets = load_stage_b_datasets(
        split="train",
        config_path=data_config_path,
        tokenizer=tokenizer,
        apply_tokenization=True,
    )

    eval_datasets = load_stage_b_datasets(
        split="validation",
        config_path=data_config_path,
        tokenizer=tokenizer,
        apply_tokenization=True,
    )

    # In debug mode, use smaller subsets
    if debug:
        logger.info("=" * 60)
        logger.info("DEBUG MODE: Using smaller dataset subsets")
        logger.info("  - Train: max 500 samples per task")
        logger.info("  - Eval: max 100 samples per task")
        logger.info("=" * 60)
        for task in train_datasets:
            original_size = len(train_datasets[task])
            if original_size > 500:
                train_datasets[task] = train_datasets[task].select(range(500))
                logger.info(f"  {task}: {original_size} -> 500 samples")
        for task in eval_datasets:
            original_size = len(eval_datasets[task])
            if original_size > 100:
                eval_datasets[task] = eval_datasets[task].select(range(100))

    # Apply safety oversampling (skip in debug mode to keep data small)
    if not debug:
        train_datasets = apply_safety_oversampling(train_datasets)
    else:
        logger.info("DEBUG MODE: Skipping safety oversampling")

    logger.info(f"Loaded {len(train_datasets)} training datasets:")
    total_train = 0
    for task, ds in train_datasets.items():
        logger.info(f"  - {task}: {len(ds):,} samples")
        total_train += len(ds)
    logger.info(f"  TOTAL: {total_train:,} training samples")

    logger.info(f"Loaded {len(eval_datasets)} evaluation datasets:")
    total_eval = 0
    for task, ds in eval_datasets.items():
        logger.info(f"  - {task}: {len(ds):,} samples")
        total_eval += len(ds)
    logger.info(f"  TOTAL: {total_eval:,} evaluation samples")

    return train_datasets, eval_datasets


def apply_safety_oversampling(
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply oversampling to safety datasets for CRISIS and RED samples.

    Per v2 plan:
        - CRISIS: 20x oversampling
        - RED: 5x oversampling
    """
    if "safety_familyos" not in datasets:
        return datasets

    from datasets import concatenate_datasets

    ds = datasets["safety_familyos"]

    # Filter by label (assuming label field exists)
    # Labels: GREEN=0, AMBER=1, RED=2, CRISIS=3
    try:
        crisis_samples = ds.filter(lambda x: x.get("label", x.get("labels", -1)) == 3)
        red_samples = ds.filter(lambda x: x.get("label", x.get("labels", -1)) == 2)

        # Oversample
        oversampled = [ds]

        if len(crisis_samples) > 0:
            for _ in range(SAFETY_OVERSAMPLING["CRISIS"] - 1):
                oversampled.append(crisis_samples)
            logger.info(
                f"Oversampled CRISIS: {len(crisis_samples)} x {SAFETY_OVERSAMPLING['CRISIS']}"
            )

        if len(red_samples) > 0:
            for _ in range(SAFETY_OVERSAMPLING["RED"] - 1):
                oversampled.append(red_samples)
            logger.info(f"Oversampled RED: {len(red_samples)} x {SAFETY_OVERSAMPLING['RED']}")

        datasets["safety_familyos"] = concatenate_datasets(oversampled)
        logger.info(
            f"Safety dataset after oversampling: {len(datasets['safety_familyos'])} samples"
        )

    except Exception as e:
        logger.warning(f"Could not apply safety oversampling: {e}")

    return datasets


# =============================================================================
# Training Arguments
# =============================================================================


def create_training_args(
    config: dict[str, Any],
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
    debug: bool = False,
) -> TrainingArguments:
    """
    Create training arguments for Stage B.

    Args:
        config: Configuration dictionary
        output_dir: Output directory override
        resume_from_checkpoint: Checkpoint to resume from
        debug: If True, use smaller batches

    Returns:
        MultiTaskTrainingArguments
    """
    training_config = config.get("training", {})
    output_config = config.get("output", {})
    task_weights = config.get("task_weights", {})

    # Determine output directory
    if output_dir is None:
        output_dir = output_config.get("output_dir", "outputs/familyos-modernbert-unified-v1")

    # Check bf16 support
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bf16_config = training_config.get("bf16", True)
    fp16_config = training_config.get("fp16", False)

    if bf16_config and not bf16_supported:
        logger.warning("bf16 not supported, falling back to fp32")
        bf16_config = False
        fp16_config = False

    # Batch sizes
    train_batch_size = training_config.get("per_device_train_batch_size", 8)
    eval_batch_size = training_config.get("per_device_eval_batch_size", 16)
    gradient_checkpointing = training_config.get("gradient_checkpointing", True)

    if debug:
        train_batch_size = min(train_batch_size, 4)
        eval_batch_size = min(eval_batch_size, 8)
        gradient_checkpointing = True
        logger.info(f"Debug mode: batch_size={train_batch_size}, eval_batch_size={eval_batch_size}")

    args = MultiTaskTrainingArguments(
        output_dir=output_dir,
        # Optimization (LoRA needs higher LR)
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=training_config.get("weight_decay", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 0.5),
        optim=training_config.get("optim", "adamw_torch_fused"),
        # Schedule
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        # Duration
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        # Batch size
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        # Evaluation & saving
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 5),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get(
            "metric_for_best_model", "eval_safety_familyos_f1"
        ),
        greater_is_better=training_config.get("greater_is_better", True),
        # Logging
        logging_steps=training_config.get("logging_steps", 50),
        logging_first_step=True,
        report_to=training_config.get("report_to", ["tensorboard"]),
        # Mixed precision
        bf16=bf16_config,
        fp16=fp16_config,
        # Memory
        gradient_checkpointing=gradient_checkpointing,
        # Misc
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        remove_unused_columns=False,
        # Resume
        resume_from_checkpoint=resume_from_checkpoint,
        # Run name
        run_name=f"stage-b-familyos-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        # Multi-task settings
        sampling_strategy="temperature",  # Better for imbalanced FamilyOS + replay
        sampling_temperature=2.0,
    )

    return args


# =============================================================================
# Save and Merge
# =============================================================================


def save_merged_model(
    peft_model: PeftModel,
    output_dir: str | Path,
    tokenizer: AutoTokenizer,
) -> None:
    """
    Merge LoRA adapters into base model and save.

    Saves:
        - outputs/.../: Merged model (standalone)
        - outputs/...-lora/: LoRA adapters only
    """
    output_dir = Path(output_dir)

    # Save LoRA adapters separately
    lora_dir = output_dir.parent / f"{output_dir.name}-lora"
    logger.info(f"Saving LoRA adapters to {lora_dir}")
    peft_model.save_pretrained(str(lora_dir))

    # Merge LoRA into base model
    logger.info("Merging LoRA adapters into base model...")
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save capabilities
    caps_path = output_dir / "capabilities.json"
    with open(caps_path, "w") as f:
        json.dump(
            {
                "capabilities": [c.value for c in ALL_CAPABILITIES],
                "stage": "B",
                "base_model": "modernbert-multitask-v0",
            },
            f,
            indent=2,
        )

    logger.info(f"Saved merged model with {len(ALL_CAPABILITIES)} capabilities")


# =============================================================================
# Main Training Function
# =============================================================================


def train_stage_b(
    config_path: str | Path,
    data_config_path: str | Path = "configs/data/multitask/stage_b_datasets.yaml",
    output_dir: str | None = None,
    checkpoint_dir: str | None = None,
    overrides: list[str] | None = None,
    debug: bool = False,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    seed: int | None = None,
) -> None:
    """
    Main Stage B training function.

    Args:
        config_path: Path to training configuration file
        data_config_path: Path to data configuration file
        output_dir: Override output directory
        checkpoint_dir: Override checkpoint directory
        overrides: List of config overrides (key=value)
        debug: Enable debug mode (smaller data, batches)
        dry_run: Validate config and data without training
        resume_from_checkpoint: Path to checkpoint to resume from
        seed: Random seed override

    Steps:
        1. Load Stage A model as base
        2. Add FamilyOS heads
        3. Apply LoRA adapters
        4. Load FamilyOS + replay datasets
        5. Train with MultiTaskTrainer
        6. Merge adapters and save
    """
    from transformers import set_seed

    logger.info("=" * 60)
    logger.info("Stage B Training: FamilyOS Domain Adaptation")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)

    # Set seed for reproducibility
    training_seed = seed if seed else config.get("training", {}).get("seed", 42)
    set_seed(training_seed)
    logger.info(f"Random seed: {training_seed}")

    # Get model path
    model_config = config.get("model", {})
    stage_a_path = model_config.get("name_or_path", "outputs/modernbert-multitask-v0/best")

    # Check Stage A model exists
    if not Path(stage_a_path).exists():
        # Try checkpoint directory
        alt_path = Path("checkpoints/modernbert-multitask-v0/best")
        if alt_path.exists():
            stage_a_path = str(alt_path)
        else:
            raise FileNotFoundError(
                f"Stage A model not found at {stage_a_path}. "
                "Run Stage A training first or specify --model.name_or_path"
            )

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(stage_a_path)

    # Load Stage A model
    model = load_stage_a_model(stage_a_path)

    # Add Stage B heads
    model = add_stage_b_heads(model, config)

    # Freeze Stage A heads as configured
    freeze_stage_a_heads(model, config)

    # Apply LoRA
    peft_model = apply_lora(model, config)

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device)
    logger.info(f"Model on device: {device}")

    # Load datasets
    train_datasets, eval_datasets = load_datasets_for_stage_b(
        config=config,
        data_config_path=data_config_path,
        tokenizer=tokenizer,
        debug=debug,
    )

    # Dry run: validate without training
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE")
        logger.info("=" * 60)
        logger.info("✅ Configuration loaded and validated")
        logger.info("✅ Stage A model loaded successfully")
        logger.info("✅ Stage B heads added")
        logger.info("✅ LoRA adapters applied")
        logger.info(f"✅ {len(train_datasets)} training datasets loaded")
        logger.info(f"✅ {len(eval_datasets)} evaluation datasets loaded")
        logger.info("")
        logger.info("Ready to train! Remove --dry_run to start training.")
        return

    # Get task weights
    task_weights = config.get("task_weights", {})
    # Ensure all tasks have weights
    for task in train_datasets:
        if task not in task_weights:
            # FamilyOS tasks get higher weight
            if task in [c.value for c in STAGE_B_CAPABILITIES]:
                task_weights[task] = 1.0
            else:
                task_weights[task] = 0.2  # Replay tasks get lower weight

    # Safety gets extra weight per v2 plan
    if "safety_familyos" in task_weights:
        task_weights["safety_familyos"] = task_weights.get("safety_familyos", 1.0) * 1.5

    logger.info(f"Task weights: {task_weights}")

    # Determine output directories
    final_output_dir = output_dir or config.get("output", {}).get(
        "output_dir", "outputs/familyos-modernbert-unified-v1"
    )
    final_checkpoint_dir = checkpoint_dir or config.get("output", {}).get(
        "checkpoint_dir", "checkpoints/familyos-modernbert-unified-v1"
    )

    # Create training arguments
    training_args = create_training_args(
        config=config,
        output_dir=final_checkpoint_dir,  # Checkpoints go here during training
        resume_from_checkpoint=resume_from_checkpoint,
        debug=debug,
    )

    # Create trainer
    trainer = MultiTaskTrainer(
        model=peft_model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        task_weights=task_weights,
        sampling_strategy="temperature",
        sampling_temperature=2.0,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting Stage B training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log training results
    logger.info(f"Training completed in {train_result.metrics.get('train_runtime', 0):.2f}s")
    logger.info(f"Final training loss: {train_result.metrics.get('train_loss', 0):.4f}")

    # Save best model
    checkpoint_output_dir = Path(training_args.output_dir)
    best_dir = checkpoint_output_dir / "best"

    logger.info("Saving best model...")
    trainer.save_model(str(best_dir))

    # Merge and save final model to the actual output directory
    final_output_path = Path(final_output_dir)
    save_merged_model(peft_model, final_output_path, tokenizer)

    # Save training config
    config_save_path = final_output_path / "training_config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Final summary
    logger.info("=" * 60)
    logger.info("🎉 Stage B Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Merged model: {final_output_path}")
    logger.info(f"LoRA adapters: {final_output_path.parent / f'{final_output_path.name}-lora'}")
    logger.info(f"Checkpoints: {checkpoint_output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Calibrate safety thresholds:")
    logger.info(f"     python scripts/calibrate_safety.py --model {final_output_path}")
    logger.info("  2. Run forgetting evaluation:")
    logger.info(f"     python scripts/evaluate_stage_a.py --model {final_output_path}")
    logger.info("  3. Run Stage B evaluation:")
    logger.info(f"     python scripts/evaluate_stage_b.py --model {final_output_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage B Training: FamilyOS Domain Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml

    # Specify Stage A checkpoint
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml \\
        --model.name_or_path outputs/modernbert-multitask-v0/best

    # Adjust LoRA rank
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml \\
        --peft.lora.r 64 --peft.lora.lora_alpha 128

    # Debug mode (smaller datasets, reduced batches)
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml --debug

    # Dry run (validate config and data loading without training)
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml --dry_run
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/multitask/stage_b_familyos.yaml",
        help="Path to Stage B config file",
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/multitask/stage_b_datasets.yaml",
        help="Path to Stage B data configuration file",
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
        "--debug",
        action="store_true",
        help="Debug mode: smaller batches, subset of data, more logging",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and data loading without training",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    # Allow arbitrary config overrides as key=value pairs
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format key.subkey=value",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Debug mode banner
    if args.debug:
        logger.info("=" * 60)
        logger.info("🐛 DEBUG MODE ENABLED")
        logger.info("  - Smaller dataset subsets (500 train, 100 eval per task)")
        logger.info("  - Reduced batch sizes (max 4 train, 8 eval)")
        logger.info("  - Gradient checkpointing enabled")
        logger.info("  - Safety oversampling disabled")
        logger.info("=" * 60)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("🧪 DRY RUN MODE - Validating config and data only")
        logger.info("=" * 60)

    train_stage_b(
        config_path=args.config,
        data_config_path=args.data_config,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        overrides=args.overrides if args.overrides else [],
        debug=args.debug,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

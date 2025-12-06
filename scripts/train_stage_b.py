#!/usr/bin/env python
"""
Stage B Training Script: FamilyOS Domain Adaptation

This script fine-tunes modernbert-multitask-v0 with FamilyOS-specific data.
Supports two modes:
    1. LoRA adapters (default) - preserves generic capabilities
    2. Full fine-tuning (v3 prep) - trains encoder layers for v3 transfer

Output: familyos-modernbert-unified-v1 (or modernbert-v2-for-v3-transfer)

Epic 5.0 Enhancements:
    - Shared CLSMeanPooler for consistent sequence pooling
    - CrossAttentionPairEncoder for NLI and Relation heads
    - Task-group adapters (optional, alongside PEFT LoRA)

v3 Preparation Mode:
    - Full fine-tuning of encoder layers (no LoRA)
    - Layer-wise learning rates (higher for layers 15-20)
    - Trains encoder weights that will transfer to v3

New tasks added:
    - ner_family: Family-specific NER (kinship, nicknames)
    - ingress: Domain classification (DIARY, TASK, HEALTH, etc.)
    - safety_familyos: Policy bands (GREEN, AMBER, RED, CRISIS)
    - relation: Family relationship extraction
    - intent: User intent classification

Existing tasks (replay for anti-forgetting):
    - ner_general, sentiment, emotions, safety_generic, nli, embedding, temporal

Usage:
    # Standard LoRA training
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_familyos.yaml

    # v3 Preparation (full fine-tuning)
    python scripts/train_stage_b.py --config configs/training/multitask/stage_b_for_v3_prep.yaml

    # Use unified FamilyOS synthetic data
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_for_v3_prep.yaml \
        --use_unified_loader

    # Start from specific Stage A checkpoint
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_familyos.yaml \
        --model.name_or_path outputs/modernbert-multitask-v0/best

    # Debug mode (smaller batches, subset of data)
    python scripts/train_stage_b.py \
        --config configs/training/multitask/stage_b_familyos.yaml \
        --debug

Environment:
    - GPU: Single GPU sufficient for LoRA (16GB+ VRAM)
    - GPU: Full fine-tune needs 24GB+ VRAM (or gradient checkpointing)
    - RAM: 32GB+ recommended

Outputs:
    - checkpoints/...: Training checkpoints
    - outputs/...: Final model (merged if LoRA, direct if full FT)
    - outputs/...-lora/: LoRA adapters only (if LoRA mode)

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
from modeling_studio.data.loaders import (
    load_embedding_triplets,
    load_familyos_unified,
    load_familyos_unified_for_training,
)
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer, MultiTaskTrainingArguments

# Epic 5.0 imports (optional enhancements)
try:
    from modeling_studio.models.pair_encoder import CrossAttentionPairEncoder
    from modeling_studio.models.poolers import CLSMeanPooler, get_pooler

    EPIC_5_AVAILABLE = True
except ImportError:
    EPIC_5_AVAILABLE = False
    CrossAttentionPairEncoder = None
    CLSMeanPooler = None
    get_pooler = None

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

    with open(config_path, encoding="utf-8") as f:
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


def reinitialize_emotions_head_for_stage_b(
    model: ModernBertMultiTaskModel,
    heads_config: dict[str, Any],
) -> None:
    """
    Reinitialize emotions head from Stage A (7 super-labels) to Stage B (44 labels).

    Stage A trains with 7 super-labels for faster curriculum learning.
    Stage B needs the full 44 FamilyOS emotion labels for fine-grained classification.

    Args:
        model: The multi-task model loaded from Stage A
        heads_config: Head configuration from YAML (heads section)
    """
    from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS
    from modeling_studio.models.heads import HierarchicalEmotionHead

    emotions_cfg = heads_config.get("emotions", {})
    if not emotions_cfg.get("enabled", True):
        return

    # Stage B expects 44 labels (or whatever is configured)
    target_num_labels = emotions_cfg.get("num_labels", 44)

    try:
        current_head = model.get_head("emotions")
        current_num_labels = getattr(
            current_head, "num_emotions", getattr(current_head, "num_labels", 44)
        )

        if current_num_labels == target_num_labels:
            logger.info(
                f"Emotions head already has {target_num_labels} labels, no reinitialization needed"
            )
            return

        logger.info(
            f"Reinitializing emotions head for Stage B: {current_num_labels} -> {target_num_labels} labels"
        )

        # Get emotion labels for 44-class schema
        emotion_labels = list(EMOTIONS_FAMILYOS_LABELS.label2id.keys())

        # Stage B uses multi-label classification (44 emotions, BCE loss)
        problem_type = emotions_cfg.get("problem_type", "multi_label_classification")

        # Create new head with 44 labels
        hidden_size = model.config.hidden_size
        new_head = HierarchicalEmotionHead(
            hidden_size=hidden_size,
            num_emotions=target_num_labels,
            num_secondary=emotions_cfg.get("num_secondary", 3),
            dropout=emotions_cfg.get("dropout", 0.1),
            pooling=emotions_cfg.get("pooling", "cls"),
            use_intensity=emotions_cfg.get("use_intensity", True),
            use_valence_arousal=emotions_cfg.get("use_valence_arousal", False),
            use_familyos=True,  # Use FamilyOS 44-emotion schema
            emotion_labels=emotion_labels,
            problem_type=problem_type,
            use_asl=emotions_cfg.get("use_asl", False),
            use_hierarchical_loss=emotions_cfg.get("use_hierarchical_loss", False),
            use_label_correlation=emotions_cfg.get("use_label_correlation", False),
            use_emotion_attention=emotions_cfg.get("use_emotion_attention", False),
            use_dynamic_thresholds=emotions_cfg.get("use_dynamic_thresholds", False),
            use_mixup=emotions_cfg.get("use_mixup", False),
            label_smoothing=emotions_cfg.get("label_smoothing", 0.0),
        )

        # Move to same device/dtype as old head
        device = next(current_head.parameters()).device
        dtype = next(current_head.parameters()).dtype
        new_head = new_head.to(device=device, dtype=dtype)

        # Replace head in model
        model.heads["emotions"] = new_head

        logger.info(
            f"Emotions head reinitialized for Stage B with {target_num_labels} labels, "
            f"problem_type={problem_type}"
        )

    except KeyError:
        logger.warning("Emotions head not found in model, skipping reinitialization")


def add_stage_b_heads(
    model: ModernBertMultiTaskModel,
    config: dict[str, Any],
) -> ModernBertMultiTaskModel:
    """
    Add FamilyOS-specific heads to the model.

    Epic 5.0 Enhancements:
        - Uses shared pooler (CLSMeanPooler) for consistent sequence pooling
        - Uses CrossAttentionPairEncoder for NLI and Relation heads
        - All enhancements are optional and backward-compatible

    Args:
        model: Base model with Stage A heads
        config: Configuration with familyos_heads settings

    Returns:
        Model with Stage B heads added
    """
    from modeling_studio.data.labels import get_num_labels
    from modeling_studio.models import CAPABILITY_TO_HEAD_TYPE, get_problem_type
    from modeling_studio.models.heads import NLIHead, RelationHead, SafetyHead

    familyos_heads = config.get("familyos_heads", {})
    epic5_config = config.get("epic5", {})
    hidden_size = model.config.hidden_size

    # Epic 5.0: Create shared pooler if enabled
    shared_pooler = None
    if epic5_config.get("use_shared_pooler", False) and EPIC_5_AVAILABLE:
        pooler_type = epic5_config.get("shared_pooler_type", "cls_mean")
        shared_pooler = get_pooler(pooler_type, hidden_size=hidden_size)
        logger.info(f"Epic 5.0: Created shared pooler ({pooler_type})")

    # Epic 5.0: Create pair encoder for NLI/Relation if enabled
    pair_encoder = None
    if epic5_config.get("use_pair_encoder", False) and EPIC_5_AVAILABLE:
        pair_encoder_layers = epic5_config.get("pair_encoder_num_layers", 2)
        pair_encoder = CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=8,
            num_layers=pair_encoder_layers,
            pooling_strategy="attention",
        )
        logger.info(f"Epic 5.0: Created CrossAttentionPairEncoder ({pair_encoder_layers} layers)")

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
        elif cap == Capability.RELATION:
            # Epic 5.0: Use pair encoder for RelationHead
            head = RelationHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout,
                problem_type=problem_type,
                pair_encoder=pair_encoder,  # Epic 5.0 enhancement
            )
            if pair_encoder:
                logger.info(f"  → {cap_name}: Using CrossAttentionPairEncoder")
        elif cap == Capability.NLI:
            # Epic 5.0: Use pair encoder and shared pooler for NLIHead
            head = NLIHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout,
                problem_type=problem_type,
                external_pooler=shared_pooler,  # Epic 5.0 enhancement
                pair_encoder=pair_encoder,  # Epic 5.0 enhancement
            )
            if pair_encoder or shared_pooler:
                logger.info(f"  → {cap_name}: Using Epic 5.0 enhancements")
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

    # Store Epic 5.0 components on model for later access
    if shared_pooler is not None:
        model.shared_pooler = shared_pooler
    if pair_encoder is not None:
        model.pair_encoder = pair_encoder

    logger.info(f"Model now has {len(model.heads)} heads: {list(model.heads.keys())}")
    return model


def apply_lora(
    model: ModernBertMultiTaskModel,
    config: dict[str, Any],
) -> PeftModel | ModernBertMultiTaskModel:
    """
    Apply LoRA adapters to the model encoder, or return model for full fine-tuning.

    Supports two training modes:
        1. LoRA (default): Applies parameter-efficient LoRA adapters
        2. Full fine-tune: When peft.method="none", returns model directly

    For v3 preparation, full fine-tuning is recommended to properly update
    encoder layers 15-20 that will be cloned to v3's Family Band.

    Args:
        model: Multi-task model
        config: Configuration with peft settings

    Returns:
        PeftModel with LoRA adapters, or original model for full fine-tuning
    """
    peft_config = config.get("peft", {})
    method = peft_config.get("method", "lora")

    # Full fine-tuning mode (no PEFT)
    if method == "none" or method is None:
        logger.info("=" * 60)
        logger.info("FULL FINE-TUNING MODE")
        logger.info("No LoRA adapters - all encoder parameters trainable")
        logger.info("=" * 60)

        # Enable input gradients for gradient checkpointing compatibility
        model.enable_input_require_grads()

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

        return model

    # LoRA mode
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
    use_unified_loader: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load Stage B datasets: FamilyOS + replay data.

    Supports two loading modes:
        1. Config-based: Uses stage_b_datasets.yaml (default)
        2. Unified loader: Loads from synthetic data directories

    Args:
        config: Training config
        data_config_path: Path to Stage B data config
        tokenizer: Tokenizer for preprocessing
        debug: If True, use smaller subsets
        use_unified_loader: If True, use unified FamilyOS loader

    Returns:
        Tuple of (train_datasets, eval_datasets)
    """
    logger.info("Loading Stage B datasets...")

    # Check if config specifies unified loader
    data_config = config.get("data", {})
    loader_type = data_config.get("loader", "config")

    if use_unified_loader or loader_type == "familyos_unified":
        # Use unified FamilyOS loader for synthetic data
        return _load_unified_familyos_data(config, tokenizer, debug)
    else:
        # Use config-based loader (original behavior)
        return _load_config_based_data(config, data_config_path, tokenizer, debug)


def _load_unified_familyos_data(
    config: dict[str, Any],
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load FamilyOS data using the unified loader.

    This handles your 420K synthetic data format where each sample
    contains labels for all tasks.
    """
    data_config = config.get("data", {})
    safety_config = config.get("safety_oversampling", {})

    # Get data directories
    data_dirs = data_config.get(
        "familyos_data_dirs",
        [
            "data/familyos/unified/output_synthetic",
            "data/familyos/unified/output",
        ],
    )

    # Get tasks to load
    tasks = data_config.get(
        "familyos_tasks",
        [
            "emotions",
            "sentiment",
            "ner_family",
            "safety_familyos",
            "intent",
            "ingress",
            "temporal",
        ],
    )

    # Get split parameters
    validation_ratio = data_config.get("validation_ratio", 0.1)
    seed = data_config.get("seed", 42)

    # Safety oversampling config
    safety_oversampling = {
        "CRISIS": safety_config.get("CRISIS", 20),
        "RED": safety_config.get("RED", 5),
        "AMBER": safety_config.get("AMBER", 1),
        "GREEN": safety_config.get("GREEN", 1),
    }

    logger.info(f"Loading unified FamilyOS data from: {data_dirs}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Safety oversampling: {safety_oversampling}")

    # Limit samples in debug mode
    max_samples = 5000 if debug else None

    # Get max length from training config
    training_config = config.get("training", {})
    max_length = training_config.get("max_length", 512)

    # Load using unified loader with tokenization
    train_datasets, eval_datasets = load_familyos_unified_for_training(
        data_dirs=data_dirs,
        tasks=tasks,
        validation_ratio=validation_ratio,
        seed=seed,
        safety_oversampling=safety_oversampling if not debug else None,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Load embedding triplets if configured
    embedding_config = data_config.get("embedding_familyos", {})
    if embedding_config.get("enabled", False):
        embedding_data_dir = embedding_config.get(
            "data_dir", "data/familyos/embeddings/silver_synthetic"
        )
        logger.info(f"Loading embedding triplets from: {embedding_data_dir}")

        # Load train and validation splits separately
        embedding_train = load_embedding_triplets(
            data_dir=embedding_data_dir,
            split="train",
            validation_ratio=validation_ratio,
            seed=seed,
        )
        embedding_eval = load_embedding_triplets(
            data_dir=embedding_data_dir,
            split="validation",
            validation_ratio=validation_ratio,
            seed=seed,
        )

        # Tokenize the triplets
        def tokenize_triplets(examples):
            """Tokenize anchor, positive, and negative texts."""
            anchor_enc = tokenizer(
                examples["anchor"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            positive_enc = tokenizer(
                examples["positive"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            negative_enc = tokenizer(
                examples["negative"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            return {
                "input_ids": anchor_enc["input_ids"],
                "attention_mask": anchor_enc["attention_mask"],
                "positive_input_ids": positive_enc["input_ids"],
                "positive_attention_mask": positive_enc["attention_mask"],
                "negative_input_ids": negative_enc["input_ids"],
                "negative_attention_mask": negative_enc["attention_mask"],
                "task": examples["task"],
            }

        # Apply tokenization
        embedding_train = embedding_train.map(
            tokenize_triplets,
            batched=True,
            remove_columns=["anchor", "positive", "negative", "anchor_cluster"],
            desc="Tokenizing embedding train triplets",
        )
        embedding_eval = embedding_eval.map(
            tokenize_triplets,
            batched=True,
            remove_columns=["anchor", "positive", "negative", "anchor_cluster"],
            desc="Tokenizing embedding eval triplets",
        )

        # NOTE: Don't set_format("torch") - the collator handles tensor conversion
        # and expects Python lists

        train_datasets["embedding"] = embedding_train
        eval_datasets["embedding"] = embedding_eval
        logger.info(
            f"Loaded {len(embedding_train):,} train / {len(embedding_eval):,} eval embedding triplets"
        )

    # Apply max samples limit in debug mode
    if debug:
        logger.info("=" * 60)
        logger.info("DEBUG MODE: Using smaller dataset subsets")
        logger.info("=" * 60)
        for task in list(train_datasets.keys()):
            if len(train_datasets[task]) > 500:
                train_datasets[task] = train_datasets[task].select(range(500))
        for task in list(eval_datasets.keys()):
            if len(eval_datasets[task]) > 100:
                eval_datasets[task] = eval_datasets[task].select(range(100))

    # Load replay datasets if configured
    replay_config = data_config.get("replay", {})
    if replay_config.get("enabled", False):
        train_datasets, eval_datasets = _add_replay_datasets(
            train_datasets, eval_datasets, replay_config, tokenizer, debug
        )

    _log_dataset_stats(train_datasets, eval_datasets)

    return train_datasets, eval_datasets


def _load_config_based_data(
    config: dict[str, Any],
    data_config_path: str | Path,
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load data using the config-based loader (original behavior).
    """
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

    _log_dataset_stats(train_datasets, eval_datasets)

    return train_datasets, eval_datasets


def _add_replay_datasets(
    train_datasets: dict[str, Any],
    eval_datasets: dict[str, Any],
    replay_config: dict[str, Any],
    tokenizer: AutoTokenizer,
    debug: bool = False,
    max_length: int = 512,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Add Stage A replay datasets to prevent forgetting.

    Loads datasets from HuggingFace and tokenizes them to match
    the expected format for training.
    """
    from datasets import load_dataset
    from modeling_studio.data.tokenization import (
        tokenize_for_nli,
        tokenize_for_token_classification,
    )

    replay_ratio = replay_config.get("ratio", 0.15)
    replay_datasets_config = replay_config.get("datasets", [])

    logger.info(f"Adding {len(replay_datasets_config)} replay datasets (ratio={replay_ratio})")

    # Task to tokenization type mapping
    task_type_map = {
        "sentiment": "classification",
        "ner_general": "token_classification",
        "nli": "nli",
        "emotions": "multilabel",
    }

    for ds_config in replay_datasets_config:
        name = ds_config.get("name")
        task = ds_config.get("task")
        config_name = ds_config.get("config", None)

        if not name or not task:
            continue

        try:
            logger.info(f"  Loading replay: {name} -> {task}")

            # Load from HuggingFace
            ds = load_dataset(name, config_name, split="train", trust_remote_code=True)

            # Calculate how many samples to include based on ratio
            if task in train_datasets:
                target_size = int(len(train_datasets[task]) * replay_ratio)
            else:
                target_size = 10000  # Default

            if debug:
                target_size = min(target_size, 200)

            if len(ds) > target_size:
                ds = ds.shuffle(seed=42).select(range(target_size))

            # Determine tokenization type
            tok_type = task_type_map.get(task, "classification")

            # Capture task in closure properly
            current_task = task

            # Create tokenization wrapper based on task type
            if tok_type == "token_classification":

                def tokenize_fn(example, _task=current_task, _tok=tokenizer, _ml=max_length):
                    # CoNLL2003 format: tokens + ner_tags
                    tags = example.get("ner_tags", [])
                    tokens = example.get("tokens", [])
                    tokenized = tokenize_for_token_classification(
                        tokenizer=_tok,
                        tokens=tokens,
                        ner_tags=tags,
                        max_length=_ml,
                    )
                    # Extract only required fields (BatchEncoding may have extra fields)
                    result = {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "labels": tokenized["labels"],
                        "task": _task,
                    }
                    return result

            elif tok_type == "nli":

                def tokenize_fn(example, _task=current_task, _tok=tokenizer, _ml=max_length):
                    result = tokenize_for_nli(
                        tokenizer=_tok,
                        premise=example.get("premise", ""),
                        hypothesis=example.get("hypothesis", ""),
                        max_length=_ml,
                    )
                    if "label" in example:
                        result["labels"] = example["label"]
                    result["task"] = _task
                    return result

            elif tok_type == "multilabel":

                def tokenize_fn(example, _task=current_task, _tok=tokenizer, _ml=max_length):
                    # For emotions datasets like dynasent
                    text = example.get("text") or example.get("sentence") or ""
                    encoded = _tok(
                        text,
                        max_length=_ml,
                        truncation=True,
                        padding=False,
                    )
                    result = {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "task": _task,
                    }
                    if "label" in example:
                        result["labels"] = example["label"]
                    return result

            else:  # classification

                def tokenize_fn(example, _task=current_task, _tok=tokenizer, _ml=max_length):
                    # SST2 format: sentence + label
                    text = example.get("text") or example.get("sentence") or ""
                    encoded = _tok(
                        text,
                        max_length=_ml,
                        truncation=True,
                        padding=False,
                    )
                    result = {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "task": _task,
                    }
                    if "label" in example:
                        result["labels"] = example["label"]
                    return result

            # Apply tokenization
            column_names = list(ds.column_names) if hasattr(ds, "column_names") else None
            ds = ds.map(
                tokenize_fn,
                remove_columns=column_names,
            )

            # Add to training data
            replay_task_name = f"{task}_replay"
            train_datasets[replay_task_name] = ds
            logger.info(f"    Added {len(ds)} tokenized samples as {replay_task_name}")

        except Exception as e:
            logger.warning(f"    Failed to load {name}: {e}")
            import traceback

            traceback.print_exc()

    return train_datasets, eval_datasets


def _log_dataset_stats(
    train_datasets: dict[str, Any],
    eval_datasets: dict[str, Any],
) -> None:
    """Log dataset statistics."""
    logger.info(f"Loaded {len(train_datasets)} training datasets:")
    total_train = 0
    for task, ds in sorted(train_datasets.items()):
        logger.info(f"  - {task}: {len(ds):,} samples")
        total_train += len(ds)
    logger.info(f"  TOTAL: {total_train:,} training samples")

    logger.info(f"Loaded {len(eval_datasets)} evaluation datasets:")
    total_eval = 0
    for task, ds in sorted(eval_datasets.items()):
        logger.info(f"  - {task}: {len(ds):,} samples")
        total_eval += len(ds)
    logger.info(f"  TOTAL: {total_eval:,} evaluation samples")


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
# Layer-wise Learning Rate
# =============================================================================


def create_layer_wise_optimizer(
    model: ModernBertMultiTaskModel | PeftModel,
    config: dict[str, Any],
    base_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """
    Create optimizer with layer-wise learning rates for v3 preparation.

    The key insight: v3 will clone v2 layers 15-20 to layers 23-28.
    So we want to train those layers STRONGLY on FamilyOS data.

    Layer-wise LR Strategy:
        - Layers 1-6 (Foundation):  5e-6  (preserve general understanding)
        - Layers 7-14 (Context):    1e-5  (preserve context processing)
        - Layers 15-20 (Target):    3e-5  (highest! will become v3 Family Band)
        - Layers 21-22 (Semantic):  2e-5  (will become v3 upper semantic layers)
        - Heads:                    1e-4  (high LR for heads)

    Args:
        model: The model to create optimizer for
        config: Config with layer_wise_lr settings
        base_lr: Base learning rate (used if layer-wise LR disabled)
        weight_decay: Weight decay for AdamW

    Returns:
        AdamW optimizer with layer-wise param groups
    """
    from torch.optim import AdamW

    layer_lr_config = config.get("layer_wise_lr", {})

    if not layer_lr_config.get("enabled", False):
        # Standard optimizer - all params same LR
        logger.info(f"Layer-wise LR disabled. Using uniform LR: {base_lr}")
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    logger.info("=" * 60)
    logger.info("LAYER-WISE LEARNING RATE ENABLED")
    logger.info("Strategy: v3 preparation (train layers 15-20 strongly)")
    logger.info("=" * 60)

    # Extract learning rates from config (convert to float in case they're strings)
    lr_1_6 = float(layer_lr_config.get("layers_1_6", {}).get("learning_rate", 5e-6))
    lr_7_14 = float(layer_lr_config.get("layers_7_14", {}).get("learning_rate", 1e-5))
    lr_15_20 = float(layer_lr_config.get("layers_15_20", {}).get("learning_rate", 3e-5))
    lr_21_22 = float(layer_lr_config.get("layers_21_22", {}).get("learning_rate", 2e-5))
    head_lr = float(layer_lr_config.get("head_lr", base_lr))

    # Build param groups
    param_groups = []
    encoder_params_assigned = set()

    # Get the base model (unwrap PEFT if needed)
    base_model = model.base_model if hasattr(model, "base_model") else model

    # Access encoder layers - try multiple paths
    encoder_layers = None

    # Path 1: model.encoder.layers (ModernBertMultiTaskModel)
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layers"):
        encoder_layers = list(base_model.encoder.layers)
    # Path 2: model.model.layers (wrapped model)
    elif hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        encoder_layers = list(base_model.model.layers)
    # Path 3: model.encoder.model.layers
    elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "model"):
        if hasattr(base_model.encoder.model, "layers"):
            encoder_layers = list(base_model.encoder.model.layers)

    if encoder_layers is None or len(encoder_layers) == 0:
        logger.warning("Could not find encoder layers. Using uniform LR.")
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    num_layers = len(encoder_layers)
    logger.info(f"Found {num_layers} encoder layers")

    # Group 1: Layers 1-6 (Foundation Band) - indices 0-5
    foundation_params = []
    for i in range(min(6, num_layers)):
        for p in encoder_layers[i].parameters():
            if p.requires_grad:
                foundation_params.append(p)
                encoder_params_assigned.add(id(p))

    if foundation_params:
        param_groups.append(
            {
                "params": foundation_params,
                "lr": lr_1_6,
                "name": "layers_1-6_foundation",
            }
        )
        logger.info(
            f"  Layers 1-6 (Foundation):  {len(foundation_params):,} params @ lr={lr_1_6:.1e}"
        )

    # Group 2: Layers 7-14 (Context Band) - indices 6-13
    context_params = []
    for i in range(6, min(14, num_layers)):
        for p in encoder_layers[i].parameters():
            if p.requires_grad:
                context_params.append(p)
                encoder_params_assigned.add(id(p))

    if context_params:
        param_groups.append(
            {
                "params": context_params,
                "lr": lr_7_14,
                "name": "layers_7-14_context",
            }
        )
        logger.info(
            f"  Layers 7-14 (Context):    {len(context_params):,} params @ lr={lr_7_14:.1e}"
        )

    # Group 3: Layers 15-20 (Target for v3 cloning!) - indices 14-19
    target_params = []
    for i in range(14, min(20, num_layers)):
        for p in encoder_layers[i].parameters():
            if p.requires_grad:
                target_params.append(p)
                encoder_params_assigned.add(id(p))

    if target_params:
        param_groups.append(
            {
                "params": target_params,
                "lr": lr_15_20,
                "name": "layers_15-20_v3_target",
            }
        )
        logger.info(
            f"  Layers 15-20 (v3 Target): {len(target_params):,} params @ lr={lr_15_20:.1e} ⭐"
        )

    # Group 4: Layers 21-22 (Upper Semantic) - indices 20-21
    semantic_params = []
    for i in range(20, min(22, num_layers)):
        for p in encoder_layers[i].parameters():
            if p.requires_grad:
                semantic_params.append(p)
                encoder_params_assigned.add(id(p))

    if semantic_params:
        param_groups.append(
            {
                "params": semantic_params,
                "lr": lr_21_22,
                "name": "layers_21-22_semantic",
            }
        )
        logger.info(
            f"  Layers 21-22 (Semantic):  {len(semantic_params):,} params @ lr={lr_21_22:.1e}"
        )

    # Group 5: All heads and other parameters
    other_params = []
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in encoder_params_assigned:
            other_params.append(p)

    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": head_lr,
                "name": "heads_and_other",
            }
        )
        logger.info(f"  Heads & Other:            {len(other_params):,} params @ lr={head_lr:.1e}")

    # Create optimizer
    optimizer = AdamW(
        param_groups,
        lr=base_lr,  # Base LR (overridden by param groups)
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_params = sum(len(g["params"]) for g in param_groups)
    logger.info(f"  TOTAL: {total_params:,} trainable parameters in {len(param_groups)} groups")

    return optimizer


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
    peft_model: PeftModel | ModernBertMultiTaskModel,
    output_dir: str | Path,
    tokenizer: AutoTokenizer,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Merge LoRA adapters into base model and save (or save full fine-tuned model).

    Handles two cases:
        1. PeftModel: Merge LoRA adapters and save merged model + adapters
        2. ModernBertMultiTaskModel: Save directly (full fine-tuning mode)

    Saves:
        - outputs/.../: Merged or full model (standalone)
        - outputs/...-lora/: LoRA adapters only (if PeftModel)
        - capabilities.json: Capabilities + Epic 5.0 config
    """
    from peft import PeftModel as PeftModelClass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is a PEFT model or full fine-tuned model
    is_peft_model = isinstance(peft_model, PeftModelClass)

    if is_peft_model:
        # Save LoRA adapters separately
        lora_dir = output_dir.parent / f"{output_dir.name}-lora"
        logger.info(f"Saving LoRA adapters to {lora_dir}")
        peft_model.save_pretrained(str(lora_dir))

        # Merge LoRA into base model
        logger.info("Merging LoRA adapters into base model...")
        merged_model = peft_model.merge_and_unload()

        # Save merged model
        logger.info(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(str(output_dir))
    else:
        # Full fine-tuned model - save directly
        logger.info(f"Saving full fine-tuned model to {output_dir}")
        peft_model.save_pretrained(str(output_dir))

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    # Build capabilities.json with Epic 5.0 config
    epic5_config = config.get("epic5", {}) if config else {}
    peft_config = config.get("peft", {}) if config else {}

    caps_data = {
        "capabilities": [c.value for c in ALL_CAPABILITIES],
        "stage": "B",
        "base_model": "modernbert-multitask-v0",
        "training_mode": "full_finetune" if peft_config.get("method") == "none" else "lora",
        "v3_ready": peft_config.get("method") == "none",  # Full fine-tune prepares for v3
        "epic_5_0": {
            "use_shared_pooler": epic5_config.get("use_shared_pooler", False),
            "shared_pooler_type": epic5_config.get("shared_pooler_type", None),
            "use_pair_encoder": epic5_config.get("use_pair_encoder", False),
            "pair_encoder_num_layers": epic5_config.get("pair_encoder_num_layers", 2),
        },
    }

    # Save capabilities
    caps_path = output_dir / "capabilities.json"
    with open(caps_path, "w") as f:
        json.dump(caps_data, f, indent=2)

    logger.info(f"Saved model with {len(ALL_CAPABILITIES)} capabilities")
    if is_peft_model:
        logger.info("  → Training mode: LoRA (adapters merged)")
    else:
        logger.info("  → Training mode: Full fine-tuning (v3 ready)")
    if epic5_config.get("use_pair_encoder") or epic5_config.get("use_shared_pooler"):
        logger.info("  → Epic 5.0 enhancements: enabled")


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

    # Reinitialize emotions head from 7 super-labels (Stage A) to 44 labels (Stage B)
    heads_config = config.get("heads", {})
    reinitialize_emotions_head_for_stage_b(model, heads_config)

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

    # Check if unified loader should be used
    data_config = config.get("data", {})
    use_unified_loader = data_config.get("use_unified_loader", False)

    # Load datasets
    train_datasets, eval_datasets = load_datasets_for_stage_b(
        config=config,
        data_config_path=data_config_path,
        tokenizer=tokenizer,
        debug=debug,
        use_unified_loader=use_unified_loader,
    )

    # Dry run: validate without training
    if dry_run:
        epic5_config = config.get("epic5", {})
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE")
        logger.info("=" * 60)
        logger.info("✅ Configuration loaded and validated")
        logger.info("✅ Stage A model loaded successfully")
        logger.info("✅ Stage B heads added")
        logger.info("✅ LoRA adapters applied")
        logger.info(f"✅ {len(train_datasets)} training datasets loaded")
        logger.info(f"✅ {len(eval_datasets)} evaluation datasets loaded")

        # Epic 5.0 status
        if epic5_config.get("use_shared_pooler") or epic5_config.get("use_pair_encoder"):
            logger.info("✅ Epic 5.0 enhancements enabled:")
            if epic5_config.get("use_shared_pooler"):
                logger.info(
                    f"   - Shared pooler: {epic5_config.get('shared_pooler_type', 'cls_mean')}"
                )
            if epic5_config.get("use_pair_encoder"):
                logger.info(
                    f"   - Pair encoder: {epic5_config.get('pair_encoder_num_layers', 2)} layers"
                )
        else:
            logger.info("ℹ️  Epic 5.0 enhancements: disabled (set epic5.use_* to enable)")

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

    # Create layer-wise optimizer if configured (for v3 preparation)
    layer_wise_config = config.get("layer_wise_lr", {})
    if layer_wise_config.get("enabled", False):
        training_config = config.get("training", {})
        base_lr = float(training_config.get("learning_rate", 1e-4))
        weight_decay = training_config.get("weight_decay", 0.01)

        optimizer = create_layer_wise_optimizer(
            model=peft_model,
            config=config,
            base_lr=base_lr,
            weight_decay=weight_decay,
        )

        # Create scheduler
        from transformers import get_scheduler

        num_training_steps = (
            len(train_datasets.get(list(train_datasets.keys())[0], []))
            * training_args.num_train_epochs
            // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            )
        )
        warmup_steps = int(num_training_steps * training_args.warmup_ratio)

        scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        optimizers = (optimizer, scheduler)
        logger.info(f"Using layer-wise optimizer with {len(optimizer.param_groups)} param groups")
    else:
        optimizers = (None, None)

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
        optimizers=optimizers,
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
    save_merged_model(peft_model, final_output_path, tokenizer, config)

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

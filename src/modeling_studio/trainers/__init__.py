"""
Trainers Module

This module provides training infrastructure for multi-task learning,
with specialized components for the ModernBERT unified encoder.

Components:
    - multitask_trainer: Main trainer class extending HuggingFace Trainer
    - task_sampler: Strategies for sampling tasks during training
    - collators: Data collators for different task types
    - callbacks: Training callbacks for monitoring and control

Training Stages:
    Stage A (Generic):
        Train on public datasets (CoNLL, SST-2, GoEmotions, etc.)
        Output: modernbert-multitask-v0

    Stage B (FamilyOS):
        Domain adaptation with LoRA on FamilyOS data
        Output: familyos-modernbert-unified-v1

Key Features:
    - Task sampling with multiple strategies
    - Per-task gradient scaling
    - Dynamic task weighting
    - Multi-task evaluation
    - Checkpoint management
"""

# TODO: Export trainer classes
# from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer

# Task samplers for multi-task learning
# Collators for different task types
from modeling_studio.trainers.collators import (
    BaseCollator,
    EmbeddingCollator,
    MultiLabelCollator,
    MultiTaskCollator,
    NLICollator,
    RelationCollator,
    SequenceClassificationCollator,
    TokenClassificationCollator,
)

# TODO: Export callbacks
# from modeling_studio.trainers.callbacks import (
#     TaskMetricsCallback,
#     GradientMonitorCallback,
# )
# EMA model for smoother training
from modeling_studio.trainers.ema import EMAModel

# Layer freezing for v3 phase-based training
from modeling_studio.trainers.freezing_v3 import (
    LAYER_BANDS,
    PHASE_TRAINABLE_BANDS,
    LayerBand,
    LayerFreezer,
    TrainingPhase,
    configure_model_for_phase,
    get_band_for_layer,
    get_layers_for_band,
    get_trainable_bands_for_phase,
)

# Phase-aware trainer for v3
from modeling_studio.trainers.trainer_v3 import (
    ModernBERTv3Trainer,
    TrainingConfig,
    TrainingState,
)

# LoRA adapters for v3
from modeling_studio.trainers.lora_v3 import (
    LoRAConfig,
    LoRALinear,
    LoRAManager,
    apply_lora_to_family_band,
    get_lora_param_count,
)

# Layer-group learning rates for v3
from modeling_studio.trainers.lr_groups_v3 import (
    LAYER_BAND_RANGES,
    PHASE_LR_CONFIGS,
    LayerGroupLRConfig,
    LayerGroupOptimizer,
    create_layer_group_optimizer,
    get_phase_config,
    print_lr_summary,
)

# Optimizer with head-wise learning rates
from modeling_studio.trainers.optimizer import create_optimizer_with_head_lr, create_param_groups
from modeling_studio.trainers.task_sampler import (
    CurriculumSampler,
    ProportionalSampler,
    SequentialSampler,
    TaskSampler,
    TemperatureSampler,
    UniformSampler,
    create_sampler,
)

# Uncertainty-based task weighting
from modeling_studio.trainers.task_weighting import UncertaintyWeighting

__all__ = [
    # Task samplers
    "TaskSampler",
    "ProportionalSampler",
    "TemperatureSampler",
    "UniformSampler",
    "SequentialSampler",
    "CurriculumSampler",
    "create_sampler",
    # Collators
    "BaseCollator",
    "SequenceClassificationCollator",
    "MultiLabelCollator",
    "TokenClassificationCollator",
    "NLICollator",
    "EmbeddingCollator",
    "RelationCollator",
    "MultiTaskCollator",
    # Training utilities
    "EMAModel",
    "create_optimizer_with_head_lr",
    "create_param_groups",
    "UncertaintyWeighting",
    # Layer freezing (v3)
    "LayerBand",
    "TrainingPhase",
    "LayerFreezer",
    "LAYER_BANDS",
    "PHASE_TRAINABLE_BANDS",
    "configure_model_for_phase",
    "get_band_for_layer",
    "get_layers_for_band",
    "get_trainable_bands_for_phase",
    # Phase-aware trainer (v3)
    "ModernBERTv3Trainer",
    "TrainingConfig",
    "TrainingState",
    # LoRA adapters (v3)
    "LoRAConfig",
    "LoRALinear",
    "LoRAManager",
    "apply_lora_to_family_band",
    "get_lora_param_count",
    # Layer-group learning rates (v3)
    "LayerGroupLRConfig",
    "LayerGroupOptimizer",
    "PHASE_LR_CONFIGS",
    "LAYER_BAND_RANGES",
    "create_layer_group_optimizer",
    "get_phase_config",
    "print_lr_summary",
]

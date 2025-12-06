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

# Hub token gradient masking for v3
from modeling_studio.trainers.gradient_masking_v3 import (
    EmbeddingGradientHook,
    GradientMaskConfig,
    HubTokenGradientManager,
    HUB_TOKEN_POSITIONS,
    V2_VOCAB_SIZE,
    V3_VOCAB_SIZE,
    get_hub_token_positions,
    get_vocab_layout,
    setup_hub_token_gradient_masking,
)

# Zipper Learning Rate strategy for v3
from modeling_studio.trainers.zipper_lr_v3 import (
    ZIPPER_PRESETS,
    ZipperLRConfig,
    ZipperLROptimizer,
    compare_zipper_presets,
    create_zipper_optimizer,
    get_zipper_preset,
    list_zipper_presets,
    print_zipper_lr_profile,
    validate_zipper_config,
    ZIPPER_LR_QUICK_REF,
)

# Learning rate schedulers for v3
from modeling_studio.trainers.schedulers_v3 import (
    DEFAULT_PHASE_SCHEDULER_CONFIGS,
    PhaseAwareScheduler,
    VALID_SCHEDULER_TYPES,
    WarmupConstantScheduler,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    compute_warmup_steps,
    create_phase_aware_scheduler,
    create_scheduler,
    get_lr_at_step,
    print_scheduler_profile,
)

# Gradient clipping and monitoring for v3
from modeling_studio.trainers.gradient_utils_v3 import (
    GradientClipConfig,
    GradientClipper,
    GradientStats,
    InterfaceGradientMonitor,
    clip_gradients,
    create_gradient_clipper,
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
    # Hub token gradient masking (v3)
    "GradientMaskConfig",
    "EmbeddingGradientHook",
    "HubTokenGradientManager",
    "HUB_TOKEN_POSITIONS",
    "V2_VOCAB_SIZE",
    "V3_VOCAB_SIZE",
    "setup_hub_token_gradient_masking",
    "get_hub_token_positions",
    "get_vocab_layout",
    # Zipper Learning Rate strategy (v3)
    "ZipperLRConfig",
    "ZipperLROptimizer",
    "ZIPPER_PRESETS",
    "create_zipper_optimizer",
    "get_zipper_preset",
    "list_zipper_presets",
    "print_zipper_lr_profile",
    "compare_zipper_presets",
    "validate_zipper_config",
    "ZIPPER_LR_QUICK_REF",
    # Learning rate schedulers (v3)
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "WarmupConstantScheduler",
    "PhaseAwareScheduler",
    "create_scheduler",
    "create_phase_aware_scheduler",
    "DEFAULT_PHASE_SCHEDULER_CONFIGS",
    "VALID_SCHEDULER_TYPES",
    "compute_warmup_steps",
    "get_lr_at_step",
    "print_scheduler_profile",
    # Gradient clipping and monitoring (v3)
    "GradientClipConfig",
    "GradientClipper",
    "GradientStats",
    "InterfaceGradientMonitor",
    "clip_gradients",
    "create_gradient_clipper",
]

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
]

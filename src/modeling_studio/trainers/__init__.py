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

# TODO: Export sampler
# from modeling_studio.trainers.task_sampler import TaskSampler

# TODO: Export collators
# from modeling_studio.trainers.collators import (
#     MultiTaskCollator,
#     SequenceClassificationCollator,
#     TokenClassificationCollator,
# )

# TODO: Export callbacks
# from modeling_studio.trainers.callbacks import (
#     TaskMetricsCallback,
#     GradientMonitorCallback,
# )

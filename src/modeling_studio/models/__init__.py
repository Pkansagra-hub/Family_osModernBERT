"""
Models Module - Enhanced v2

This module provides model architectures for multi-task learning,
with a focus on ModernBERT-based unified encoders.

Components:
    - modernbert_multitask: Main multi-task model architecture
    - heads: Task-specific classification/embedding heads
    - poolers: Pooling strategies for sequence representations
    - losses: Specialized loss functions for multi-task training

Primary Model:
    ModernBertMultiTaskModel - Unified encoder with multiple task heads

Supported Tasks (12 capabilities):
    Generic (Stage A):
    - ner_general: Extended named entity recognition (17 BIO tags)
    - sentiment: 5-point sentiment classification
    - emotions: Multi-label emotion detection (32 classes)
    - safety_generic: Toxicity detection (8 types)
    - nli: Natural language inference
    - embedding: Dense vector representations
    - temporal: Temporal expression extraction (NEW)

    FamilyOS (Stage B):
    - ner_family: Family-specific NER (21 BIO tags)
    - safety_familyos: Policy band classification
    - ingress: Domain/topic classification (12 domains)
    - relation: Family relationship extraction (NEW)
    - intent: User intent classification (NEW)
"""

# Export main model class
# Export head classes
from modeling_studio.models.heads import (
    BaseHead,
    EmbeddingHead,
    IntentHead,
    NLIHead,
    RelationHead,
    SafetyHead,
    SequenceClassificationHead,
    TemporalHead,
    TokenClassificationHead,
)
from modeling_studio.models.modernbert_multitask import (
    CAPABILITY_TO_HEAD_TYPE,
    ModernBertMultiTaskModel,
    MultiTaskOutput,
    get_problem_type,
)

__all__ = [
    # Main model
    "ModernBertMultiTaskModel",
    "MultiTaskOutput",
    "CAPABILITY_TO_HEAD_TYPE",
    "get_problem_type",
    # Heads
    "BaseHead",
    "SequenceClassificationHead",
    "TokenClassificationHead",
    "EmbeddingHead",
    "NLIHead",
    "SafetyHead",
    "RelationHead",  # NEW
    "IntentHead",  # NEW
    "TemporalHead",  # NEW
]

# TODO: Export poolers
# from modeling_studio.models.poolers import (
#     CLSPooler,
#     MeanPooler,
#     MaxPooler,
# )

# TODO: Export losses
# from modeling_studio.models.losses import (
#     FocalLoss,
#     MultipleNegativesRankingLoss,
#     MultiTaskLoss,
# )

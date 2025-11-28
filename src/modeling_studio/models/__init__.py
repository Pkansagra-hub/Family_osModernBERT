"""
Models Module

This module provides model architectures for multi-task learning,
with a focus on ModernBERT-based unified encoders.

Components:
    - modernbert_multitask: Main multi-task model architecture
    - heads: Task-specific classification/embedding heads
    - poolers: Pooling strategies for sequence representations
    - losses: Specialized loss functions for multi-task training

Primary Model:
    ModernBertMultiTaskModel - Unified encoder with multiple task heads
    
Supported Tasks:
    - ner_general: General named entity recognition
    - ner_family: FamilyOS-specific NER (kinship, nicknames)
    - sentiment: Sentiment classification
    - emotions: Multi-label emotion detection
    - safety: Toxicity and policy band classification
    - ingress: Domain/topic classification
    - embedding: Dense vector representations
    - nli: Natural language inference
"""

# TODO: Export main model class
# from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

# TODO: Export head classes
# from modeling_studio.models.heads import (
#     SequenceClassificationHead,
#     TokenClassificationHead,
#     EmbeddingHead,
#     NLIHead,
#     SafetyHead,
# )

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

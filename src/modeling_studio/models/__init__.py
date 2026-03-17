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
# Import adapters (Epic 5.0)
from modeling_studio.models.adapters import (
    AdaptedLinear,
    AdapterConfig,
    BottleneckAdapter,
    LoRAAdapter,
    ParallelAdapter,
    TaskGroupAdapter,
    TaskGroupConfig,
    create_adapter,
)

# Import attention mechanisms (Milestone 11)
from modeling_studio.models.attention import (
    CrossAttention,
    GroupedQueryAttention,
    RotaryEmbedding,
)

# Import decoder components (Stage C - GPT-2 based)
# NOTE: MoE decoder deprecated due to Chinchilla scaling failure
from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

# Legacy MoE components (deprecated - kept for backward compatibility)
from modeling_studio.models.decoder_config import DecoderMoEConfig
from modeling_studio.models.decoder_moe import (
    CounterfactualDecoderHead,  # DEPRECATED: Use GPT2DecoderHead
    DecoderBlock,
    EncoderProjection,
)
from modeling_studio.models.heads import (
    BaseHead,
    EmbeddingHead,
    GlobalPointerNERHead,  # NEW - v2 SOTA span-based NER
    create_globalpointer_head,  # Factory function
    IntentHead,
    MultiGranularityRelevanceHead,  # MGRH — SOTA cross-encoder re-ranking
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

# Import ranking losses (MGRH — Milestone 1)
from modeling_studio.models.losses_ranking import (
    CombinedRankingLoss,
    LambdaRankLoss,
)

# Import MoE components (Milestone 10)
from modeling_studio.models.moe_components import (
    MoELayer,
    SwiGLUExpert,
    TopKRouter,
)

# Import pair encoder (Epic 5.0)
from modeling_studio.models.pair_encoder import (
    AttentionPooling,
    BidirectionalCrossAttentionBlock,
    ConcatPairEncoder,
    CrossAttentionLayer,
    CrossAttentionPairEncoder,
    PairEncoderConfig,
    create_pair_encoder,
)

# Import verification (Issue 4.2.1)
from modeling_studio.models.verification_v3 import (
    FunctionPreservingVerifier,
    LayerComparisonResult,
    VerificationResult,
    WeightComparisonResult,
    create_verification_inputs,
    verify_embedding_transfer,
    verify_function_preserving,
    verify_weight_transfer,
)

# Import poolers (Epic 5.0)
from modeling_studio.models.poolers import (
    AttentionPooler,
    CLSMeanPooler,
    CLSPooler,
    MeanPooler,
    get_pooler,
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
    "GlobalPointerNERHead",  # NEW - v2 SOTA span-based NER
    "create_globalpointer_head",  # Factory function
    "EmbeddingHead",
    "NLIHead",
    "MultiGranularityRelevanceHead",  # MGRH — SOTA cross-encoder re-ranking
    "SafetyHead",
    "RelationHead",  # NEW
    "IntentHead",  # NEW
    "TemporalHead",  # NEW
    # Decoder components (Stage C - GPT-2 based)
    "GPT2DecoderHead",  # PRIMARY: Pre-trained GPT-2 with prefix injection
    "GPT2DecoderConfig",
    # Legacy decoder components (deprecated)
    "DecoderMoEConfig",  # DEPRECATED
    "CounterfactualDecoderHead",  # DEPRECATED: Use GPT2DecoderHead
    "DecoderBlock",
    "EncoderProjection",
    # MoE components (Milestone 10)
    "MoELayer",
    "TopKRouter",
    "SwiGLUExpert",
    # Attention mechanisms (Milestone 11)
    "GroupedQueryAttention",
    "CrossAttention",
    "RotaryEmbedding",
    # Poolers (Epic 5.0)
    "CLSPooler",
    "MeanPooler",
    "CLSMeanPooler",
    "AttentionPooler",
    "get_pooler",
    # Pair Encoder (Epic 5.0)
    "PairEncoderConfig",
    "CrossAttentionPairEncoder",
    "CrossAttentionLayer",
    "BidirectionalCrossAttentionBlock",
    "AttentionPooling",
    "ConcatPairEncoder",
    "create_pair_encoder",
    # Ranking losses (MGRH — Milestone 1)
    "LambdaRankLoss",
    "CombinedRankingLoss",
    # Adapters (Epic 5.0)
    "AdapterConfig",
    "TaskGroupConfig",
    "BottleneckAdapter",
    "TaskGroupAdapter",
    "ParallelAdapter",
    "LoRAAdapter",
    "AdaptedLinear",
    "create_adapter",
    # Verification (Issue 4.2.1)
    "FunctionPreservingVerifier",
    "VerificationResult",
    "LayerComparisonResult",
    "WeightComparisonResult",
    "verify_function_preserving",
    "verify_weight_transfer",
    "verify_embedding_transfer",
    "create_verification_inputs",
]

# TODO: Export losses
# from modeling_studio.models.losses import (
#     FocalLoss,
#     MultipleNegativesRankingLoss,
#     MultiTaskLoss,
# )

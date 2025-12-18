"""FamilyOS UltraBERT - Models subpackage.

Contains the multi-task model architecture and all task-specific heads.
"""

from familyos_ultrabert.models.modernbert_multitask import (
    ModernBertMultiTaskModel,
    MultiTaskOutput,
)
from familyos_ultrabert.models.heads import (
    EmbeddingHead,
    EnhancedSafetyHead,
    HierarchicalEmotionHead,
    IntentHead,
    NLIHead,
    RelationHead,
    SafetyHead,
    SequenceClassificationHead,
    TemporalHead,
    TokenClassificationHead,
)

# Decoder exports (v3)
from familyos_ultrabert.models.decoder_gpt2 import GPT2DecoderHead, EncoderProjection
from familyos_ultrabert.models.decoder_gpt2_config import (
    GPT2DecoderConfig,
    get_edge_config,
    get_small_config,
    get_quality_config,
)

__all__ = [
    # Core model
    "ModernBertMultiTaskModel",
    "MultiTaskOutput",
    # Task heads
    "EmbeddingHead",
    "EnhancedSafetyHead",
    "HierarchicalEmotionHead",
    "IntentHead",
    "NLIHead",
    "RelationHead",
    "SafetyHead",
    "SequenceClassificationHead",
    "TemporalHead",
    "TokenClassificationHead",
    # Decoder (v3)
    "GPT2DecoderHead",
    "EncoderProjection",
    "GPT2DecoderConfig",
    "get_edge_config",
    "get_small_config",
    "get_quality_config",
]

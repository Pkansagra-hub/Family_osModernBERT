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

__all__ = [
    "ModernBertMultiTaskModel",
    "MultiTaskOutput",
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
]

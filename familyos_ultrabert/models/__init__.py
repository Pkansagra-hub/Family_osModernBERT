"""FamilyOS UltraBERT - Models subpackage.

Contains the multi-task model architecture and all task-specific heads.
"""

from familyos_ultrabert.models.modernbert_multitask import (
    CAPABILITY_TO_HEAD_TYPE,
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
    GlobalPointerNERHead,
    create_globalpointer_head,
)

# Loss functions
from familyos_ultrabert.models.losses import (
    GlobalPointerLoss,
    FocalGlobalPointerLoss,
)

__all__ = [
    # Core model
    "ModernBertMultiTaskModel",
    "MultiTaskOutput",
    "CAPABILITY_TO_HEAD_TYPE",
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
    # GlobalPointer (v4 heads)
    "GlobalPointerNERHead",
    "create_globalpointer_head",
    "GlobalPointerLoss",
    "FocalGlobalPointerLoss",
]

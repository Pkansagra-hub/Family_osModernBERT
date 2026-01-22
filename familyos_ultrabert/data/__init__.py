"""FamilyOS UltraBERT - Data subpackage.

Contains label definitions for all 12 capabilities.
"""

from familyos_ultrabert.data.labels import (
    Capability,
    LabelSchema,
    CAPABILITY_TO_LABELS,
    get_labels_for_capability,
    get_num_labels,
)

# GlobalPointer collator and labels
from familyos_ultrabert.data.globalpointer_collator import (
    GlobalPointerCollator,
    create_ner_general_collator,
    create_ner_family_collator,
    create_temporal_collator,
    NER_GENERAL_LABELS,
    NER_FAMILY_LABELS,
    TEMPORAL_LABELS,
)

# Alias for compatibility
get_labels = get_labels_for_capability

__all__ = [
    "Capability",
    "LabelSchema",
    "CAPABILITY_TO_LABELS",
    "get_labels",
    "get_labels_for_capability",
    "get_num_labels",
    # GlobalPointer
    "GlobalPointerCollator",
    "create_ner_general_collator",
    "create_ner_family_collator",
    "create_temporal_collator",
    "NER_GENERAL_LABELS",
    "NER_FAMILY_LABELS",
    "TEMPORAL_LABELS",
]

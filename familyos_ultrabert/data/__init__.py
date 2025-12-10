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

# Alias for compatibility
get_labels = get_labels_for_capability

__all__ = [
    "Capability",
    "LabelSchema",
    "CAPABILITY_TO_LABELS",
    "get_labels",
    "get_labels_for_capability",
    "get_num_labels",
]

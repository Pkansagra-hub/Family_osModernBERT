"""
Data Module - Enhanced v2

This module provides data loading, preprocessing, and dataset management
for multi-task learning with ModernBERT.

Components:
    - multitask_dataset: Combined dataset for multi-task training
    - loaders: Dataset loading from various sources
    - preprocessing: Text cleaning and normalization
    - tokenization: Tokenization utilities with subword alignment
    - labels: Label schema definitions for all tasks

Data Flow:
    1. Load raw data (HuggingFace, local files)
    2. Apply task-specific preprocessing
    3. Tokenize with appropriate strategy
    4. Combine into MultiTaskDataset
    5. Use with MultiTaskTrainer

Supported Tasks (12 capabilities):
    Generic (Stage A):
        - NER General (17 BIO tags)
        - Sentiment (5 classes)
        - Emotions (32 multi-label)
        - Safety Generic (8 types)
        - NLI (3 classes)
        - Embedding (contrastive)
        - Temporal (13 BIO tags) - NEW

    FamilyOS (Stage B):
        - Family NER (21 BIO tags)
        - Ingress (12 domains)
        - Safety FamilyOS (4 bands)
        - Relation (15 types) - NEW
        - Intent (8 classes) - NEW

Configuration:
    See configs/data/multitask/*.yaml for dataset configurations.
"""

# Export label schemas
from modeling_studio.data.labels import (  # Mappings and helpers; FamilyOS labels; Generic labels; Capability enum; Base class; NEW labels
    ALL_LABEL_SCHEMAS,
    BAND_TO_SUBCATEGORY_IDS,
    CAPABILITY_TO_LABELS,
    EMOTIONS_LABELS,
    EMOTIONS_REDUCED_LABELS,
    INGRESS_LABELS,
    INTENT_LABELS,
    NER_FAMILY_LABELS,
    NER_GENERAL_LABELS,
    NLI_LABELS,
    RELATION_LABELS,
    SAFETY_FAMILYOS_LABELS,
    SAFETY_GENERIC_LABELS,
    SAFETY_SUBCATEGORIES,
    SENTIMENT_LABELS,
    SUBCATEGORY_TO_BAND_ID,
    TEMPORAL_LABELS,
    Capability,
    LabelSchema,
    get_labels_for_capability,
    get_num_labels,
)

__all__ = [
    # Label schema base
    "LabelSchema",
    "Capability",
    # V3 Extractor vocabularies
    "LabelVocabulary",
    "V3LabelVocabularies",
    "ExtractedLabels",
    "MultiTaskExtractor",
    "collate_classification_labels",
    "collate_multi_label",
    "collate_token_labels",
    # Generic
    "NER_GENERAL_LABELS",
    "SENTIMENT_LABELS",
    "EMOTIONS_LABELS",
    "EMOTIONS_REDUCED_LABELS",
    "SAFETY_GENERIC_LABELS",
    "NLI_LABELS",
    "TEMPORAL_LABELS",  # NEW
    # FamilyOS
    "NER_FAMILY_LABELS",
    "INGRESS_LABELS",
    "SAFETY_FAMILYOS_LABELS",
    "SAFETY_SUBCATEGORIES",  # Issue 3.6.8
    "SUBCATEGORY_TO_BAND_ID",  # Issue 3.6.8
    "BAND_TO_SUBCATEGORY_IDS",  # Issue 3.6.8
    "RELATION_LABELS",  # NEW
    "INTENT_LABELS",  # NEW
    # Mappings
    "CAPABILITY_TO_LABELS",
    "ALL_LABEL_SCHEMAS",
    "get_labels_for_capability",
    "get_num_labels",
    # Loaders
    "load_ner_dataset",
    "load_classification_dataset",
    "load_multilabel_dataset",
    "load_nli_dataset",
    "load_embedding_dataset",
    "load_familyos_ner",
    "load_familyos_ingress",
    "load_familyos_safety",
    "load_familyos_relations",
    "load_familyos_intents",
    "load_familyos_temporal",
    "load_from_config",
    "load_stage_a_datasets",
    "load_stage_b_datasets",
    "UnifiedFamilyOSDataset",
    "IterableUnifiedFamilyOSDataset",
    "TaskType",
    "HubType",
    "HubRouting",
    "HubTaskMapping",
    "HubRoutingParser",
    "SpanAnnotation",
    "RelationTriple",
    "UnifiedSample",
    # Multi-task dataset
    "TaskDataset",
    "MultiTaskDataset",
    "create_multitask_dataset",
    # Tokenization
    "load_tokenizer",
    "tokenize_for_classification",
    "tokenize_for_token_classification",
    "tokenize_for_nli",
    "tokenize_for_embedding",
    "get_tokenize_function",
    # Indian English Support (Issue 3.6.7)
    "INDIAN_ENGLISH_MAPPINGS",
    "INDIAN_VENTING_PATTERNS",
    "KINSHIP_VARIANTS",
    "FAMILY_STRUCTURE_TYPES",
    "FamilyStructureType",
    "IndianEnglishNormalizer",
    "normalize_indian_english",
    "is_venting",
    "get_kinship_variants",
]


# Export loaders
# Export Indian English support (Issue 3.6.7)
from modeling_studio.data.cultural_mappings import (
    FAMILY_STRUCTURE_TYPES,
    INDIAN_ENGLISH_MAPPINGS,
    INDIAN_VENTING_PATTERNS,
    KINSHIP_VARIANTS,
    FamilyStructureType,
    IndianEnglishNormalizer,
    get_kinship_variants,
    is_venting,
    normalize_indian_english,
)
from modeling_studio.data.loaders import (
    load_classification_dataset,
    load_embedding_dataset,
    load_familyos_ingress,
    load_familyos_intents,
    load_familyos_ner,
    load_familyos_relations,
    load_familyos_safety,
    load_familyos_temporal,
    load_from_config,
    load_multilabel_dataset,
    load_ner_dataset,
    load_nli_dataset,
    load_stage_a_datasets,
    load_stage_b_datasets,
)
from modeling_studio.data.loaders_v3 import (
    HubRouting,
    HubRoutingParser,
    HubTaskMapping,
    HubType,
    IterableUnifiedFamilyOSDataset,
    RelationTriple,
    SpanAnnotation,
    TaskType,
    UnifiedFamilyOSDataset,
    UnifiedSample,
)
from modeling_studio.data.extractors_v3 import (
    ExtractedLabels,
    LabelVocabulary,
    MultiTaskExtractor,
    V3LabelVocabularies,
    collate_classification_labels,
    collate_multi_label,
    collate_token_labels,
)

# Export multi-task dataset
from modeling_studio.data.multitask_dataset import (
    MultiTaskDataset,
    TaskDataset,
    create_multitask_dataset,
)

# Export tokenization utilities
from modeling_studio.data.tokenization import (
    get_tokenize_function,
    load_tokenizer,
    tokenize_for_classification,
    tokenize_for_embedding,
    tokenize_for_nli,
    tokenize_for_token_classification,
)

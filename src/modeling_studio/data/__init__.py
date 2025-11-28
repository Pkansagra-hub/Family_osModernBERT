"""
Data Module

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

Supported Tasks:
    Generic:
        - NER (token classification)
        - Sentiment (sequence classification)
        - Emotions (multi-label classification)
        - Safety (multi-label classification)
        - NLI (pair classification)
        - Embedding (contrastive learning)

    FamilyOS:
        - Family NER
        - Ingress classification
        - Safety policy bands

Configuration:
    See configs/data/multitask/*.yaml for dataset configurations.
"""

# TODO: Export dataset classes
# from modeling_studio.data.multitask_dataset import (
#     MultiTaskDataset,
#     TaskDataset,
# )

# TODO: Export loaders
# from modeling_studio.data.loaders import (
#     load_ner_dataset,
#     load_classification_dataset,
#     load_nli_dataset,
#     load_from_config,
# )

# TODO: Export preprocessing
# from modeling_studio.data.preprocessing import TextPreprocessor

# TODO: Export tokenization
# from modeling_studio.data.tokenization import (
#     load_tokenizer,
#     get_tokenize_function,
# )

# TODO: Export label schemas
# from modeling_studio.data.labels import (
#     NER_GENERAL_LABELS,
#     SENTIMENT_LABELS,
#     EMOTIONS_LABELS,
#     NER_FAMILY_LABELS,
#     INGRESS_LABELS,
#     SAFETY_FAMILYOS_LABELS,
# )

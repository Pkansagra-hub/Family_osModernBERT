"""
Label Schema Definitions

This module contains label definitions and mappings for all tasks
in the multi-task model.

Label Categories:
    Generic Tasks:
        - NER_GENERAL: Standard NER labels (PER, ORG, LOC, MISC)
        - SENTIMENT: Sentiment labels (negative, neutral, positive)
        - EMOTIONS: Emotion labels (GoEmotions reduced set)
        - SAFETY_GENERIC: Toxicity labels (Jigsaw)
        - NLI: NLI labels (entailment, neutral, contradiction)
    
    FamilyOS Tasks:
        - NER_FAMILY: Family-specific NER (kinship, nicknames, etc.)
        - INGRESS: Domain labels (DIARY, TASK, HEALTH, etc.)
        - SAFETY_FAMILYOS: Policy bands (GREEN, AMBER, RED, CRISIS)

Schema Format:
    Each schema is a dataclass containing:
    - label2id: Dict mapping label name to ID
    - id2label: Dict mapping ID to label name
    - num_labels: Total number of labels
    - description: Human-readable descriptions

Usage:
    from modeling_studio.data.labels import NER_GENERAL_LABELS
    
    label_id = NER_GENERAL_LABELS.label2id["B-PER"]
    label_name = NER_GENERAL_LABELS.id2label[1]
"""

# TODO: Define NER_GENERAL_LABELS
#   - BIO tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
#   - num_labels: 9

# TODO: Define SENTIMENT_LABELS
#   - negative: 0, neutral: 1, positive: 2
#   - num_labels: 3

# TODO: Define EMOTIONS_LABELS
#   - GoEmotions reduced set (12 emotions)
#   - Multi-label (can have multiple emotions)

# TODO: Define SAFETY_GENERIC_LABELS
#   - Jigsaw toxicity labels
#   - Multi-label (toxic, severe_toxic, obscene, etc.)

# TODO: Define NLI_LABELS
#   - entailment: 0, neutral: 1, contradiction: 2
#   - num_labels: 3

# TODO: Define NER_FAMILY_LABELS (FamilyOS)
#   - Extended NER for family context
#   - PERSON, KINSHIP, NICKNAME, PET, HOME_LOCATION, etc.
#   - BIO format

# TODO: Define INGRESS_LABELS (FamilyOS)
#   - Domain classification
#   - DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META

# TODO: Define SAFETY_FAMILYOS_LABELS (FamilyOS)
#   - Policy bands: GREEN, AMBER, RED, CRISIS
#   - Single-label classification

# TODO: Implement LabelSchema dataclass
#   - label2id, id2label, num_labels
#   - Validation methods
#   - Serialization to/from config

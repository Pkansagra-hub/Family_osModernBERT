"""
FamilyOS NLP - Label Definitions

This module contains label schemas for all 12 capabilities.
Each schema maps label names to IDs and back, enabling
proper output interpretation.

Usage:
    from familyos_nlp.labels import get_labels, Capability

    schema = get_labels(Capability.SENTIMENT)
    print(schema.id2label[3])  # "positive"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


@dataclass
class LabelSchema:
    """Label schema for a capability."""

    name: str
    label2id: Dict[str, int]
    problem_type: str = "single_label_classification"
    description: str = ""

    _id2label: Dict[int, str] = field(init=False, repr=False)
    _num_labels: int = field(init=False, repr=False)

    def __post_init__(self):
        self._id2label = {v: k for k, v in self.label2id.items()}
        self._num_labels = len(self.label2id)

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def decode(self, label_id: int) -> str:
        return self._id2label[label_id]

    def encode(self, label: str) -> int:
        return self.label2id[label]


class Capability(str, Enum):
    """All 13 capabilities supported by FamilyOS NLP."""

    # Generic capabilities
    NER_GENERAL = "ner_general"
    SENTIMENT = "sentiment"
    EMOTIONS = "emotions"
    SAFETY_GENERIC = "safety_generic"
    NLI = "nli"
    EMBEDDING = "embedding"
    TEMPORAL = "temporal"

    # FamilyOS-specific capabilities
    NER_FAMILY = "ner_family"
    INGRESS = "ingress"
    SAFETY_FAMILYOS = "safety_familyos"
    RELATION = "relation"
    INTENT = "intent"

    # v3: Generation capability (requires decoder)
    COUNTERFACTUAL = "counterfactual"

    def __str__(self) -> str:
        return self.value


# All capability names
CAPABILITIES = [c.value for c in Capability]


# =============================================================================
# Label Schemas
# =============================================================================

NER_GENERAL_LABELS = LabelSchema(
    name="ner_general",
    label2id={
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8,
        "B-DATE": 9, "I-DATE": 10, "B-TIME": 11, "I-TIME": 12,
        "B-EVENT": 13, "I-EVENT": 14, "B-PRODUCT": 15, "I-PRODUCT": 16,
    },
    problem_type="token_classification",
    description="General NER (17 BIO tags)",
)

SENTIMENT_LABELS = LabelSchema(
    name="sentiment",
    label2id={
        "very_negative": 0, "negative": 1, "neutral": 2,
        "positive": 3, "very_positive": 4,
    },
    problem_type="single_label_classification",
    description="5-point sentiment scale",
)

EMOTIONS_LABELS = LabelSchema(
    name="emotions",
    label2id={
        # Core (8)
        "neutral": 0, "joy": 1, "sadness": 2, "anger": 3,
        "fear": 4, "surprise": 5, "love": 6, "disgust": 7,
        # Positive (12)
        "admiration": 8, "amusement": 9, "approval": 10, "caring": 11,
        "excitement": 12, "gratitude": 13, "optimism": 14, "pride": 15,
        "relief": 16, "contentment": 17, "hope": 18, "tenderness": 19,
        # Negative (10)
        "annoyance": 20, "disappointment": 21, "disapproval": 22, "embarrassment": 23,
        "grief": 24, "nervousness": 25, "remorse": 26, "frustration": 27,
        "overwhelmed": 28, "emptiness": 29,
        # Family-specific (14)
        "nostalgia": 30, "protectiveness": 31, "togetherness": 32, "longing": 33,
        "warmth": 34, "playfulness": 35, "celebration": 36, "belonging": 37,
        "parental_pride": 38, "parental_guilt": 39, "patience": 40, "worry": 41,
        "bittersweet": 42, "homesickness": 43,
    },
    problem_type="multi_label_classification",
    description="44 emotions including family-specific",
)

SAFETY_GENERIC_LABELS = LabelSchema(
    name="safety_generic",
    label2id={
        "toxic": 0, "severe_toxic": 1, "obscene": 2, "threat": 3,
        "insult": 4, "identity_hate": 5, "self_harm": 6, "dangerous_advice": 7,
    },
    problem_type="multi_label_classification",
    description="8-type toxicity detection",
)

NLI_LABELS = LabelSchema(
    name="nli",
    label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    problem_type="single_label_classification",
    description="Natural language inference",
)

TEMPORAL_LABELS = LabelSchema(
    name="temporal",
    label2id={
        "O": 0,
        "B-DATE_ABS": 1, "I-DATE_ABS": 2, "B-DATE_REL": 3, "I-DATE_REL": 4,
        "B-TIME": 5, "I-TIME": 6, "B-DURATION": 7, "I-DURATION": 8,
        "B-FREQUENCY": 9, "I-FREQUENCY": 10, "B-AGE": 11, "I-AGE": 12,
    },
    problem_type="token_classification",
    description="Temporal expression extraction (13 BIO tags)",
)

NER_FAMILY_LABELS = LabelSchema(
    name="ner_family",
    label2id={
        "O": 0,
        "B-PERSON": 1, "I-PERSON": 2, "B-KINSHIP": 3, "I-KINSHIP": 4,
        "B-NICKNAME": 5, "I-NICKNAME": 6, "B-PET": 7, "I-PET": 8,
        "B-HOME_LOC": 9, "I-HOME_LOC": 10, "B-FAMILY_EVENT": 11, "I-FAMILY_EVENT": 12,
        "B-ROUTINE": 13, "I-ROUTINE": 14, "B-TRADITION": 15, "I-TRADITION": 16,
        "B-MILESTONE": 17, "I-MILESTONE": 18, "B-HEIRLOOM": 19, "I-HEIRLOOM": 20,
    },
    problem_type="token_classification",
    description="Family NER (21 BIO tags)",
)

INGRESS_LABELS = LabelSchema(
    name="ingress",
    label2id={
        "DIARY": 0, "TASK": 1, "HEALTH": 2, "FINANCE": 3,
        "RELATIONSHIP": 4, "WORK": 5, "META": 6, "MEMORY": 7,
        "PLANNING": 8, "CELEBRATION": 9, "CONCERN": 10, "GRATITUDE": 11,
    },
    problem_type="single_label_classification",
    description="12 domain categories",
)

SAFETY_FAMILYOS_LABELS = LabelSchema(
    name="safety_familyos",
    label2id={"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3},
    problem_type="single_label_classification",
    description="Safety policy bands",
)

RELATION_LABELS = LabelSchema(
    name="relation",
    label2id={
        "no_relation": 0,
        "parent_of": 1, "child_of": 2, "spouse_of": 3, "sibling_of": 4,
        "grandparent_of": 5, "grandchild_of": 6, "aunt_uncle_of": 7, "niece_nephew_of": 8,
        "cousin_of": 9, "pet_of": 10, "friend_of": 11, "colleague_of": 12,
        "lives_at": 13, "owns": 14,
    },
    problem_type="multi_label_classification",
    description="15 relationship types",
)

INTENT_LABELS = LabelSchema(
    name="intent",
    label2id={
        "log_memory": 0, "query_memory": 1, "set_reminder": 2, "express_feeling": 3,
        "seek_advice": 4, "share_news": 5, "reflect": 6, "other": 7,
    },
    problem_type="single_label_classification",
    description="8 user intents",
)


# Counterfactual generation schema (v3)
COUNTERFACTUAL_LABELS = LabelSchema(
    name="counterfactual",
    label2id={},  # Generation task, no fixed labels
    problem_type="generation",
    description="Counterfactual text generation (requires decoder)",
)


# Mapping from capability to labels
CAPABILITY_TO_LABELS: Dict[Capability, Optional[LabelSchema]] = {
    Capability.NER_GENERAL: NER_GENERAL_LABELS,
    Capability.SENTIMENT: SENTIMENT_LABELS,
    Capability.EMOTIONS: EMOTIONS_LABELS,
    Capability.SAFETY_GENERIC: SAFETY_GENERIC_LABELS,
    Capability.NLI: NLI_LABELS,
    Capability.EMBEDDING: None,
    Capability.TEMPORAL: TEMPORAL_LABELS,
    Capability.NER_FAMILY: NER_FAMILY_LABELS,
    Capability.INGRESS: INGRESS_LABELS,
    Capability.SAFETY_FAMILYOS: SAFETY_FAMILYOS_LABELS,
    Capability.RELATION: RELATION_LABELS,
    Capability.INTENT: INTENT_LABELS,
    Capability.COUNTERFACTUAL: COUNTERFACTUAL_LABELS,
}


# Capabilities that require the decoder (v3)
DECODER_CAPABILITIES = {Capability.COUNTERFACTUAL}


def get_labels(capability: Capability | str) -> Optional[LabelSchema]:
    """Get label schema for a capability."""
    if isinstance(capability, str):
        capability = Capability(capability)
    return CAPABILITY_TO_LABELS.get(capability)

"""
Label Schema Definitions - Enhanced v2

This module contains label definitions and mappings for all tasks
in the multi-task model.

Based on: Latest research in multi-task learning, family NLP,
emotion AI, and safety classification (2023-2025).

Label Categories:
    Generic Tasks:
        - NER_GENERAL: Extended NER (PER, ORG, LOC, MISC, DATE, TIME, EVENT, PRODUCT)
        - SENTIMENT: 5-point sentiment scale
        - EMOTIONS: 44 emotions (FamilyOS schema – default for Stage A/B)
        - SAFETY_GENERIC: 8 toxicity types (Jigsaw + self-harm + dangerous advice)
        - NLI: NLI labels (entailment, neutral, contradiction)
        - TEMPORAL: Temporal expression extraction

    FamilyOS Tasks:
        - NER_FAMILY: Family-specific NER (21 BIO tags)
        - INGRESS: 12 domain labels
        - SAFETY_FAMILYOS: Policy bands (GREEN, AMBER, RED, CRISIS)
        - RELATION: Family relationship extraction (15 relations)
        - INTENT: User intent classification (8 intents)

Schema Format:
    Each schema is a dataclass containing:
    - label2id: Dict mapping label name to ID
    - id2label: Dict mapping ID to label name
    - num_labels: Total number of labels
    - description: Human-readable descriptions

Usage:
    from familyos_ultrabert.data.labels import NER_GENERAL_LABELS

    label_id = NER_GENERAL_LABELS.label2id["B-PER"]
    label_name = NER_GENERAL_LABELS.id2label[1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# Label Schema Base Class
# =============================================================================


@dataclass
class LabelSchema:
    """
    Base class for label schemas.

    Attributes:
        name: Schema identifier
        label2id: Mapping from label string to integer ID
        problem_type: Classification type (single_label, multi_label, token)
        description: Human-readable description of the label set
    """

    name: str
    label2id: dict[str, int]
    problem_type: str = "single_label_classification"
    description: str = ""

    # Computed fields
    _id2label: dict[int, str] = field(init=False, repr=False)
    _num_labels: int = field(init=False, repr=False)

    def __post_init__(self):
        self._id2label = {v: k for k, v in self.label2id.items()}
        self._num_labels = len(self.label2id)

    @property
    def id2label(self) -> dict[int, str]:
        """Mapping from integer ID to label string."""
        return self._id2label

    @property
    def num_labels(self) -> int:
        """Total number of labels."""
        return self._num_labels

    def encode(self, label: str) -> int:
        """Convert label string to ID."""
        return self.label2id[label]

    def decode(self, label_id: int) -> str:
        """Convert label ID to string."""
        return self._id2label[label_id]

    def to_dict(self) -> dict:
        """Serialize schema to dictionary."""
        return {
            "name": self.name,
            "label2id": self.label2id,
            "id2label": self._id2label,
            "num_labels": self._num_labels,
            "problem_type": self.problem_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LabelSchema:
        """Deserialize schema from dictionary."""
        return cls(
            name=data["name"],
            label2id=data["label2id"],
            problem_type=data.get("problem_type", "single_label_classification"),
            description=data.get("description", ""),
        )


# =============================================================================
# Generic Task Labels (Stage A - Public Datasets)
# =============================================================================


# -----------------------------------------------------------------------------
# NER General Labels (Extended: 9 → 17 BIO tags)
# -----------------------------------------------------------------------------
NER_GENERAL_LABELS = LabelSchema(
    name="ner_general",
    label2id={
        "O": 0,
        # Person
        "B-PER": 1,
        "I-PER": 2,
        # Organization
        "B-ORG": 3,
        "I-ORG": 4,
        # Location/Place
        "B-LOC": 5,
        "I-LOC": 6,
        # Miscellaneous
        "B-MISC": 7,
        "I-MISC": 8,
        # Date (explicit dates)
        "B-DATE": 9,
        "I-DATE": 10,
        # Time (explicit times)
        "B-TIME": 11,
        "I-TIME": 12,
        # Event (named events)
        "B-EVENT": 13,
        "I-EVENT": 14,
        # Product/Food/Item
        "B-PRODUCT": 15,
        "I-PRODUCT": 16,
    },
    problem_type="token_classification",
    description="Extended general NER with temporal and product entities (BIO format)",
)


# -----------------------------------------------------------------------------
# Sentiment Labels (Enhanced: 3 → 5 classes)
# -----------------------------------------------------------------------------
SENTIMENT_LABELS = LabelSchema(
    name="sentiment",
    label2id={
        "very_negative": 0,  # Strong negative (angry, frustrated, devastated)
        "negative": 1,  # Mild negative (disappointed, sad)
        "neutral": 2,  # Neutral/factual
        "positive": 3,  # Mild positive (happy, content)
        "very_positive": 4,  # Strong positive (ecstatic, overjoyed)
    },
    problem_type="single_label_classification",
    description="5-point sentiment scale for nuanced analysis",
)


# -----------------------------------------------------------------------------
# Emotion Labels (GoEmotions 28 remapped to 32 classes – legacy compatibility)
# NOTE: FamilyOS 44-emotion schema (Stage A/B default) is defined in EMOTIONS_FAMILYOS_LABELS below
# -----------------------------------------------------------------------------
EMOTIONS_LABELS = LabelSchema(
    name="emotions",
    label2id={
        # GoEmotions 28 labels (remapped to contiguous 0-27)
        "admiration": 0,
        "amusement": 1,
        "anger": 2,
        "annoyance": 3,
        "approval": 4,
        "caring": 5,
        "confusion": 6,
        "curiosity": 7,
        "desire": 8,
        "disappointment": 9,
        "disapproval": 10,
        "disgust": 11,
        "embarrassment": 12,
        "excitement": 13,
        "fear": 14,
        "gratitude": 15,
        "grief": 16,
        "joy": 17,
        "love": 18,
        "nervousness": 19,
        "optimism": 20,
        "pride": 21,
        "realization": 22,
        "relief": 23,
        "remorse": 24,
        "sadness": 25,
        "surprise": 26,
        "neutral": 27,
        # Family-specific additions (28-31)
        "nostalgia": 28,
        "togetherness": 29,
        "protectiveness": 30,
        "longing": 31,
    },
    problem_type="multi_label_classification",
    description="Legacy GoEmotions 28 + 4 family-specific = 32 emotions (deprecated)",
)

# -----------------------------------------------------------------------------
# FamilyOS Emotion Labels (44 emotions - Stage A/B default schema)
# -----------------------------------------------------------------------------
EMOTIONS_FAMILYOS_LABELS = LabelSchema(
    name="emotions_familyos",
    label2id={
        # Core Emotions (8)
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "love": 6,
        "disgust": 7,
        # Positive Emotions (12)
        "admiration": 8,
        "amusement": 9,
        "approval": 10,
        "caring": 11,
        "excitement": 12,
        "gratitude": 13,
        "optimism": 14,
        "pride": 15,
        "relief": 16,
        "contentment": 17,
        "hope": 18,
        "tenderness": 19,
        # Negative Emotions (10)
        "annoyance": 20,
        "disappointment": 21,
        "disapproval": 22,
        "embarrassment": 23,
        "grief": 24,
        "nervousness": 25,
        "remorse": 26,
        "frustration": 27,
        "overwhelmed": 28,
        "emptiness": 29,
        # Family-Specific Emotions (14)
        "nostalgia": 30,  # "Remember our first family trip?"
        "protectiveness": 31,  # "Just want to shield them from hurt"
        "togetherness": 32,  # "Love when we're all together"
        "longing": 33,  # "Wish mom was here to see this"
        "warmth": 34,  # "Sunday dinners are the best"
        "playfulness": 35,  # "Had a pillow fight with the kids"
        "celebration": 36,  # "Time to celebrate this milestone!"
        "belonging": 37,  # "This is where I'm meant to be"
        "parental_pride": 38,  # "Look how far they've come"
        "parental_guilt": 39,  # "Should have spent more time with them"
        "patience": 40,  # "Deep breaths, they're just kids"
        "worry": 41,  # "Can't stop thinking about dad's health"
        "bittersweet": 42,  # "They're growing up so fast"
        "homesickness": 43,  # "Wish I could be there with everyone"
    },
    problem_type="multi_label_classification",
    description="FamilyOS emotions - 44 classes with family-specific feelings (Stage A/B default)",
)


# Reduced emotion set (12 classes) for simpler tasks
EMOTIONS_REDUCED_LABELS = LabelSchema(
    name="emotions_reduced",
    label2id={
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
        "love": 7,
        "curiosity": 8,
        "confusion": 9,
        "gratitude": 10,
        "disappointment": 11,
    },
    problem_type="multi_label_classification",
    description="Reduced emotion set (12 classes)",
)


# -----------------------------------------------------------------------------
# Super-Label Emotion Schema (7 classes - Stage A Curriculum Learning)
# -----------------------------------------------------------------------------
# Stage A trains on broad emotion categories before Stage B fine-tunes on 44 labels.
# This curriculum learning approach helps the model learn valence/arousal first.
# See TRAINING_STRATEGY.md for details.

EMOTIONS_SUPER_LABELS = LabelSchema(
    name="emotions_super",
    label2id={
        "JOY": 0,  # joy, excitement, celebration, pride, relief, amusement, hope, optimism, surprise
        "AFFECTION": 1,  # love, warmth, caring, gratitude, tenderness, admiration, parental_pride, protectiveness, playfulness
        "SADNESS": 2,  # sadness, grief, disappointment, longing, emptiness, remorse, parental_guilt
        "ANXIETY": 3,  # worry, overwhelmed, frustration, annoyance, nervousness, fear, anger, disgust, disapproval, embarrassment
        "NOSTALGIA": 4,  # nostalgia, bittersweet, homesickness
        "CONTENTMENT": 5,  # contentment, belonging, togetherness, patience, approval
        "NEUTRAL": 6,  # neutral
    },
    problem_type="single_label_classification",  # Stage A uses CrossEntropyLoss
    description="7 super-label emotion categories for Stage A curriculum learning",
)


# Mapping from 44 FamilyOS emotions to 7 super-labels
# Used by map_to_super_labels() to collapse granular emotions during Stage A training
EMOTION_TO_SUPER_LABEL: dict[str, str] = {
    # JOY cluster - High positive valence, high arousal
    "joy": "JOY",
    "excitement": "JOY",
    "celebration": "JOY",
    "pride": "JOY",
    "relief": "JOY",
    "amusement": "JOY",
    "hope": "JOY",
    "optimism": "JOY",
    # AFFECTION cluster - Positive interpersonal, warm
    "love": "AFFECTION",
    "warmth": "AFFECTION",
    "caring": "AFFECTION",
    "gratitude": "AFFECTION",
    "tenderness": "AFFECTION",
    "admiration": "AFFECTION",
    "parental_pride": "AFFECTION",
    "protectiveness": "AFFECTION",
    "playfulness": "AFFECTION",
    # SADNESS cluster - Low valence, low arousal
    "sadness": "SADNESS",
    "grief": "SADNESS",
    "disappointment": "SADNESS",
    "longing": "SADNESS",
    "emptiness": "SADNESS",
    "remorse": "SADNESS",
    "parental_guilt": "SADNESS",
    # ANXIETY cluster - Negative valence, high arousal
    "worry": "ANXIETY",
    "overwhelmed": "ANXIETY",
    "frustration": "ANXIETY",
    "annoyance": "ANXIETY",
    "nervousness": "ANXIETY",
    "fear": "ANXIETY",
    "anger": "ANXIETY",
    "disgust": "ANXIETY",
    "disapproval": "ANXIETY",
    "embarrassment": "ANXIETY",
    # NOSTALGIA cluster - Mixed valence, reflective
    "nostalgia": "NOSTALGIA",
    "bittersweet": "NOSTALGIA",
    "homesickness": "NOSTALGIA",
    # CONTENTMENT cluster - Positive valence, low arousal
    "contentment": "CONTENTMENT",
    "belonging": "CONTENTMENT",
    "togetherness": "CONTENTMENT",
    "patience": "CONTENTMENT",
    "approval": "CONTENTMENT",
    # NEUTRAL cluster - Neutral valence
    "neutral": "NEUTRAL",
    # Note: 'surprise' maps to JOY because in FamilyOS context, surprise is typically
    # positive (birthday surprises, exciting news). Per Plutchik wheel, surprise is
    # high-energy and in family contexts correlates with excitement/joy.
    "surprise": "JOY",
}


def map_to_super_labels(
    multi_hot_44: list[int],
    source_schema: LabelSchema | None = None,
    target_schema: LabelSchema | None = None,
) -> list[int]:
    """
    Convert a 44-label multi-hot vector to 7 super-label multi-hot.

    This function supports Stage A curriculum learning by collapsing
    granular emotion labels into broader categories.

    Args:
        multi_hot_44: Multi-hot vector of length 44 (or list of emotion names)
        source_schema: Source label schema (44 labels). Defaults to EMOTIONS_FAMILYOS_LABELS.
        target_schema: Target label schema (7 super-labels). Defaults to EMOTIONS_SUPER_LABELS.

    Returns:
        Multi-hot vector of length 7

    Example:
        >>> labels_44 = [0] * 44
        >>> labels_44[1] = 1  # joy (index 1 in EMOTIONS_FAMILYOS_LABELS)
        >>> labels_44[6] = 1  # love (index 6)
        >>> super_labels = map_to_super_labels(labels_44)
        >>> super_labels  # [1, 1, 0, 0, 0, 0, 0] -> JOY + AFFECTION
    """
    # Use defaults if not provided (avoid mutable default args)
    if source_schema is None:
        source_schema = EMOTIONS_FAMILYOS_LABELS
    if target_schema is None:
        target_schema = EMOTIONS_SUPER_LABELS

    multi_hot_7 = [0] * target_schema.num_labels

    for idx, val in enumerate(multi_hot_44):
        if val == 1:
            # Get emotion name from source schema
            if idx < source_schema.num_labels:
                emotion_name = source_schema.id2label.get(idx)
                if emotion_name:
                    # Map to super-label
                    super_label = EMOTION_TO_SUPER_LABEL.get(emotion_name)
                    if super_label and super_label in target_schema.label2id:
                        super_idx = target_schema.label2id[super_label]
                        multi_hot_7[super_idx] = 1

    return multi_hot_7


def map_emotion_names_to_super_labels(
    emotion_names: list[str],
    target_schema: LabelSchema | None = None,
) -> list[int]:
    """
    Convert a list of emotion names to 7 super-label multi-hot.

    This is a convenience function for when you have emotion names
    instead of a multi-hot vector.

    Args:
        emotion_names: List of emotion name strings (e.g., ["joy", "love"])
        target_schema: Target label schema (7 super-labels). Defaults to EMOTIONS_SUPER_LABELS.

    Returns:
        Multi-hot vector of length 7

    Example:
        >>> super_labels = map_emotion_names_to_super_labels(["joy", "love", "nostalgia"])
        >>> super_labels  # [1, 1, 0, 0, 1, 0, 0] -> JOY + AFFECTION + NOSTALGIA
    """
    if target_schema is None:
        target_schema = EMOTIONS_SUPER_LABELS

    multi_hot_7 = [0] * target_schema.num_labels

    for emotion_name in emotion_names:
        super_label = EMOTION_TO_SUPER_LABEL.get(emotion_name)
        if super_label and super_label in target_schema.label2id:
            super_idx = target_schema.label2id[super_label]
            multi_hot_7[super_idx] = 1

    return multi_hot_7


# -----------------------------------------------------------------------------
# Safety Generic Labels (Enhanced: 6 → 8 types)
# -----------------------------------------------------------------------------
SAFETY_GENERIC_LABELS = LabelSchema(
    name="safety_generic",
    label2id={
        # Jigsaw toxicity base (6)
        "toxic": 0,
        "severe_toxic": 1,
        "obscene": 2,
        "threat": 3,
        "insult": 4,
        "identity_hate": 5,
        # Additional safety dimensions (2 new)
        "self_harm": 6,  # Self-harm ideation
        "dangerous_advice": 7,  # Harmful recommendations
    },
    problem_type="multi_label_classification",
    description="Toxicity detection with self-harm and dangerous advice (8 types)",
)


# -----------------------------------------------------------------------------
# NLI Labels (Natural Language Inference)
# -----------------------------------------------------------------------------
NLI_LABELS = LabelSchema(
    name="nli",
    label2id={
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    },
    problem_type="single_label_classification",
    description="Natural Language Inference: entailment, neutral, contradiction",
)


# =============================================================================
# FamilyOS-Specific Labels (Stage B - Domain Adaptation)
# =============================================================================


# -----------------------------------------------------------------------------
# NER Family Labels (Enhanced: 15 → 21 BIO tags)
# -----------------------------------------------------------------------------
NER_FAMILY_LABELS = LabelSchema(
    name="ner_family",
    label2id={
        "O": 0,
        # Person (full names)
        "B-PERSON": 1,
        "I-PERSON": 2,
        # Kinship terms (mom, dad, uncle, nana, bhai, didi)
        "B-KINSHIP": 3,
        "I-KINSHIP": 4,
        # Nicknames (Panda, Bunny, Sweetie, Baby)
        "B-NICKNAME": 5,
        "I-NICKNAME": 6,
        # Pets (Max, Whiskers, our dog)
        "B-PET": 7,
        "I-PET": 8,
        # Home locations (kitchen, Emma's room, backyard)
        "B-HOME_LOC": 9,
        "I-HOME_LOC": 10,
        # Family events (birthday, anniversary, graduation, wedding)
        "B-FAMILY_EVENT": 11,
        "I-FAMILY_EVENT": 12,
        # Routines (school run, dinner time, bedtime story)
        "B-ROUTINE": 13,
        "I-ROUTINE": 14,
        # Family traditions (Sunday brunch, movie night, game night)
        "B-TRADITION": 15,
        "I-TRADITION": 16,
        # Milestone (first steps, first word, lost tooth)
        "B-MILESTONE": 17,
        "I-MILESTONE": 18,
        # Heirloom/Special item (grandma's ring, dad's watch)
        "B-HEIRLOOM": 19,
        "I-HEIRLOOM": 20,
    },
    problem_type="token_classification",
    description="Family-specific NER with traditions, milestones, and heirlooms (21 BIO tags)",
)


# -----------------------------------------------------------------------------
# Ingress Labels (Enhanced: 7 → 12 domains)
# -----------------------------------------------------------------------------
INGRESS_LABELS = LabelSchema(
    name="ingress",
    label2id={
        # Original domains (7)
        "DIARY": 0,  # Personal reflections, journaling
        "TASK": 1,  # To-dos, reminders, action items
        "HEALTH": 2,  # Medical, wellness, fitness
        "FINANCE": 3,  # Money, bills, budgets
        "RELATIONSHIP": 4,  # Family dynamics, social
        "WORK": 5,  # Job, career, professional
        "META": 6,  # System commands, queries about FamilyOS
        # Extended domains (5 new)
        "MEMORY": 7,  # Recalling past events ("Remember when...")
        "PLANNING": 8,  # Future events ("Next week we should...")
        "CELEBRATION": 9,  # Birthdays, achievements, milestones
        "CONCERN": 10,  # Worries, anxieties (feeds into safety)
        "GRATITUDE": 11,  # Appreciation expressions
    },
    problem_type="single_label_classification",
    description="Extended domain classification for family context (12 domains)",
)


# -----------------------------------------------------------------------------
# Safety FamilyOS Labels (Policy Bands)
# -----------------------------------------------------------------------------
SAFETY_FAMILYOS_LABELS = LabelSchema(
    name="safety_familyos",
    label2id={
        "GREEN": 0,  # Safe, routine content
        "AMBER": 1,  # Needs attention, mild concern
        "RED": 2,  # Serious concern, escalate to K1
        "CRISIS": 3,  # Immediate intervention needed
    },
    problem_type="single_label_classification",
    description="FamilyOS safety policy bands: GREEN (safe) to CRISIS (immediate)",
)


# -----------------------------------------------------------------------------
# Safety Subcategories (Issue 3.6.8)
# -----------------------------------------------------------------------------
SAFETY_SUBCATEGORIES = LabelSchema(
    name="safety_subcategories",
    label2id={
        # GREEN subcategory (placeholder - GREEN content is just GREEN)
        "none": 0,  # No safety concern
        # AMBER subcategories (1-4)
        "stress": 1,  # General stress, overwhelm
        "mild_sadness": 2,  # Transient sadness, disappointment
        "frustration": 3,  # Frustration, irritation
        "health_mention": 4,  # Health concerns, medical mentions
        # RED subcategories (5-8)
        "persistent_sadness": 5,  # Prolonged sadness, depressive indicators
        "isolation": 6,  # Social isolation, withdrawal
        "hopelessness": 7,  # Hopelessness, despair
        "substance": 8,  # Substance use concerns
        # CRISIS subcategories (9-11)
        "self_harm_ideation": 9,  # Self-harm thoughts or ideation
        "suicide_ideation": 10,  # Suicidal thoughts or ideation
        "harm_to_others": 11,  # Threats or intent to harm others
        "abuse_disclosure": 12,  # Disclosure of abuse
    },
    problem_type="single_label_classification",
    description="Fine-grained safety subcategories: 12 types grouped by band",
)

# Mapping from subcategory ID to parent band ID
SUBCATEGORY_TO_BAND_ID: dict[int, int] = {
    0: 0,  # none -> GREEN
    1: 1,  # stress -> AMBER
    2: 1,  # mild_sadness -> AMBER
    3: 1,  # frustration -> AMBER
    4: 1,  # health_mention -> AMBER
    5: 2,  # persistent_sadness -> RED
    6: 2,  # isolation -> RED
    7: 2,  # hopelessness -> RED
    8: 2,  # substance -> RED
    9: 3,  # self_harm_ideation -> CRISIS
    10: 3,  # suicide_ideation -> CRISIS
    11: 3,  # harm_to_others -> CRISIS
    12: 3,  # abuse_disclosure -> CRISIS
}

# Mapping from band ID to valid subcategory IDs
BAND_TO_SUBCATEGORY_IDS: dict[int, list[int]] = {
    0: [0],  # GREEN -> none
    1: [1, 2, 3, 4],  # AMBER -> stress, mild_sadness, frustration, health_mention
    2: [5, 6, 7, 8],  # RED -> persistent_sadness, isolation, hopelessness, substance
    3: [
        9,
        10,
        11,
        12,
    ],  # CRISIS -> self_harm_ideation, suicide_ideation, harm_to_others, abuse_disclosure
}


# =============================================================================
# NEW Capabilities (v2 Additions)
# =============================================================================


# -----------------------------------------------------------------------------
# Relation Labels (Family Relationship Extraction)
# -----------------------------------------------------------------------------
RELATION_LABELS = LabelSchema(
    name="relation",
    label2id={
        "no_relation": 0,
        # Family relations
        "parent_of": 1,  # X is parent of Y
        "child_of": 2,  # X is child of Y
        "spouse_of": 3,  # X is married to Y
        "sibling_of": 4,  # X is sibling of Y
        "grandparent_of": 5,  # X is grandparent of Y
        "grandchild_of": 6,  # X is grandchild of Y
        "aunt_uncle_of": 7,  # X is aunt/uncle of Y
        "niece_nephew_of": 8,  # X is niece/nephew of Y
        "cousin_of": 9,  # X is cousin of Y
        "pet_of": 10,  # X is pet of Y
        # Non-family relations
        "friend_of": 11,  # X is friend of Y
        "colleague_of": 12,  # X works with Y
        "lives_at": 13,  # X lives at Y (location)
        "owns": 14,  # X owns Y (heirloom)
    },
    problem_type="multi_label_classification",  # Sentence can have multiple relations
    description="Relationship extraction between entities (15 relations)",
)


# -----------------------------------------------------------------------------
# Intent Labels (User Intent Classification)
# -----------------------------------------------------------------------------
INTENT_LABELS = LabelSchema(
    name="intent",
    label2id={
        "log_memory": 0,  # "Had dinner with family" (store this)
        "query_memory": 1,  # "What did we do last Sunday?"
        "set_reminder": 2,  # "Remind me to call mom"
        "express_feeling": 3,  # "Feeling grateful today"
        "seek_advice": 4,  # "What should I do about..."
        "share_news": 5,  # "Guess what happened!"
        "reflect": 6,  # "Thinking about the past..."
        "other": 7,  # Catch-all
    },
    problem_type="single_label_classification",
    description="User intent classification for FamilyOS interactions (8 intents)",
)


# -----------------------------------------------------------------------------
# Temporal Labels (Temporal Expression Extraction)
# -----------------------------------------------------------------------------
TEMPORAL_LABELS = LabelSchema(
    name="temporal",
    label2id={
        "O": 0,
        # Absolute dates
        "B-DATE_ABS": 1,
        "I-DATE_ABS": 2,  # "January 15, 2024"
        # Relative dates
        "B-DATE_REL": 3,
        "I-DATE_REL": 4,  # "yesterday", "last week"
        # Times
        "B-TIME": 5,
        "I-TIME": 6,  # "3pm", "morning"
        # Durations
        "B-DURATION": 7,
        "I-DURATION": 8,  # "for 2 hours", "all day"
        # Frequency
        "B-FREQUENCY": 9,
        "I-FREQUENCY": 10,  # "every Sunday", "weekly"
        # Age/Period
        "B-AGE": 11,
        "I-AGE": 12,  # "when she was 5", "in my 20s"
    },
    problem_type="token_classification",
    description="Temporal expression extraction for timeline construction (13 BIO tags)",
)


# =============================================================================
# Capability Enum (Maps to Tasks/Heads) - Enhanced v2: 9 → 12 Capabilities
# =============================================================================


class Capability(str, Enum):
    """
    Enumeration of all capabilities supported by the unified encoder.
    Each capability maps to a specific task head.

    Enhanced v2: 9 → 12 capabilities with family-specific additions.
    """

    # Generic capabilities (Stage A)
    NER_GENERAL = "ner_general"
    SENTIMENT = "sentiment"
    EMOTIONS = "emotions"
    SAFETY_GENERIC = "safety_generic"
    NLI = "nli"
    EMBEDDING = "embedding"
    TEMPORAL = "temporal"  # NEW: Temporal expression extraction

    # FamilyOS capabilities (Stage B)
    NER_FAMILY = "ner_family"
    INGRESS = "ingress"
    SAFETY_FAMILYOS = "safety_familyos"
    RELATION = "relation"  # NEW: Family relationship extraction
    INTENT = "intent"  # NEW: User intent classification

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Capability to Label Schema Mapping
# =============================================================================


CAPABILITY_TO_LABELS: dict[Capability, LabelSchema | None] = {
    # Generic capabilities
    Capability.NER_GENERAL: NER_GENERAL_LABELS,
    Capability.SENTIMENT: SENTIMENT_LABELS,
    Capability.EMOTIONS: EMOTIONS_FAMILYOS_LABELS,
    Capability.SAFETY_GENERIC: SAFETY_GENERIC_LABELS,
    Capability.NLI: NLI_LABELS,
    Capability.EMBEDDING: None,  # Embedding has no labels
    Capability.TEMPORAL: TEMPORAL_LABELS,  # NEW
    # FamilyOS capabilities
    Capability.NER_FAMILY: NER_FAMILY_LABELS,
    Capability.INGRESS: INGRESS_LABELS,
    Capability.SAFETY_FAMILYOS: SAFETY_FAMILYOS_LABELS,
    Capability.RELATION: RELATION_LABELS,  # NEW
    Capability.INTENT: INTENT_LABELS,  # NEW
}


def get_labels_for_capability(capability: Capability | str) -> LabelSchema | None:
    """Get the label schema for a given capability."""
    if isinstance(capability, str):
        capability = Capability(capability)
    return CAPABILITY_TO_LABELS.get(capability)


def get_num_labels(capability: Capability | str) -> int:
    """Get the number of labels for a given capability."""
    labels = get_labels_for_capability(capability)
    return labels.num_labels if labels else 0


# =============================================================================
# All Labels Export
# =============================================================================


ALL_LABEL_SCHEMAS: dict[str, LabelSchema] = {
    # Generic labels
    "ner_general": NER_GENERAL_LABELS,
    "sentiment": SENTIMENT_LABELS,
    "emotions": EMOTIONS_FAMILYOS_LABELS,
    "emotions_legacy": EMOTIONS_LABELS,
    "emotions_reduced": EMOTIONS_REDUCED_LABELS,
    "emotions_familyos": EMOTIONS_FAMILYOS_LABELS,
    "emotions_super": EMOTIONS_SUPER_LABELS,  # Stage A curriculum learning
    "safety_generic": SAFETY_GENERIC_LABELS,
    "nli": NLI_LABELS,
    "temporal": TEMPORAL_LABELS,  # NEW
    # FamilyOS labels
    "ner_family": NER_FAMILY_LABELS,
    "ingress": INGRESS_LABELS,
    "safety_familyos": SAFETY_FAMILYOS_LABELS,
    "safety_subcategories": SAFETY_SUBCATEGORIES,  # Issue 3.6.8
    "relation": RELATION_LABELS,  # NEW
    "intent": INTENT_LABELS,  # NEW
}


__all__ = [
    # Base class
    "LabelSchema",
    # Capability enum
    "Capability",
    # Generic labels
    "NER_GENERAL_LABELS",
    "SENTIMENT_LABELS",
    "EMOTIONS_LABELS",
    "EMOTIONS_FAMILYOS_LABELS",
    "EMOTIONS_REDUCED_LABELS",
    "EMOTIONS_SUPER_LABELS",  # Stage A curriculum learning
    "EMOTION_TO_SUPER_LABEL",  # 44 -> 7 mapping dict
    "map_to_super_labels",  # Multi-hot conversion function
    "map_emotion_names_to_super_labels",  # Name-based conversion
    "SAFETY_GENERIC_LABELS",
    "NLI_LABELS",
    "TEMPORAL_LABELS",  # NEW
    # FamilyOS labels
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
    # Helpers
    "get_labels_for_capability",
    "get_num_labels",
]

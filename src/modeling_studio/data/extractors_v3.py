"""
Multi-Task Sample Extractor for v3 Training

Extracts 8 task types from unified samples into training-ready tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from modeling_studio.data.loaders_v3 import RelationTriple, SpanAnnotation, UnifiedSample

logger = logging.getLogger(__name__)


# ============================================================================
# Label Vocabularies
# ============================================================================


@dataclass
class LabelVocabulary:
    """Label vocabulary for a single task."""

    labels: list[str]
    label_to_id: dict[str, int] = field(default_factory=dict)
    id_to_label: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_to_id:
            self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def encode(self, label: str) -> int:
        """Encode a single label to its integer id, returning -1 if unknown."""

        return self.label_to_id.get(label, -1)

    def encode_multi(self, labels: list[str]) -> list[int]:
        """Encode a list of labels to ids, skipping unknown labels."""

        return [self.label_to_id[label] for label in labels if label in self.label_to_id]

    def to_multi_hot(self, labels: list[str]) -> torch.Tensor:
        """Convert label list to a multi-hot tensor."""

        vector = torch.zeros(len(self.labels), dtype=torch.float32)
        for label in labels:
            if label in self.label_to_id:
                vector[self.label_to_id[label]] = 1.0
        return vector

    @property
    def num_labels(self) -> int:
        """Return the number of labels in the vocabulary."""

        return len(self.labels)


class V3LabelVocabularies:
    """Container for all label vocabularies used in v3 FamilyOS tasks."""

    EMOTIONS = LabelVocabulary(
        labels=[
            "neutral",
            "joy",
            "love",
            "gratitude",
            "hope",
            "excitement",
            "contentment",
            "pride",
            "amusement",
            "relief",
            "tenderness",
            "curiosity",
            "surprise",
            "sadness",
            "grief",
            "loneliness",
            "disappointment",
            "fear",
            "anxiety",
            "worry",
            "anger",
            "frustration",
            "annoyance",
            "disgust",
            "guilt",
            "shame",
            "remorse",
            "bittersweet",
        ]
    )

    SENTIMENT = LabelVocabulary(
        labels=[
            "very_negative",
            "negative",
            "neutral",
            "positive",
            "very_positive",
            "mixed",
        ]
    )

    SAFETY = LabelVocabulary(labels=["GREEN", "AMBER", "RED", "CRISIS"])

    INTENT = LabelVocabulary(
        labels=[
            "inform",
            "request",
            "confirm",
            "seek_advice",
            "express_emotion",
            "schedule",
            "remind",
            "plan",
            "reflect",
            "share",
            "ask",
            "command",
            "greet",
            "farewell",
            "thank",
            "apologize",
            "compliment",
            "complain",
            "joke",
            "other",
        ]
    )

    INGRESS = LabelVocabulary(
        labels=[
            "DIARY",
            "CHAT",
            "TODO",
            "CALENDAR",
            "MEMORY",
            "PLANNING",
            "RELATIONSHIP",
            "FINANCE",
            "HEALTH",
            "SHOPPING",
            "RECIPE",
            "TRAVEL",
            "KIDS",
            "PETS",
            "OTHER",
        ]
    )

    NER_FAMILY = LabelVocabulary(
        labels=[
            "O",
            "B-PERSON",
            "I-PERSON",
            "B-KINSHIP",
            "I-KINSHIP",
            "B-PET",
            "I-PET",
            "B-LOCATION",
            "I-LOCATION",
            "B-EVENT",
            "I-EVENT",
            "B-TRADITION",
            "I-TRADITION",
            "B-ORG",
            "I-ORG",
        ]
    )

    TEMPORAL = LabelVocabulary(
        labels=[
            "O",
            "B-DATE_ABS",
            "I-DATE_ABS",
            "B-DATE_REL",
            "I-DATE_REL",
            "B-TIME",
            "I-TIME",
            "B-DURATION",
            "I-DURATION",
            "B-RECURRENCE",
            "I-RECURRENCE",
        ]
    )

    RELATION_PREDICATES = LabelVocabulary(
        labels=[
            "parent_of",
            "child_of",
            "sibling_of",
            "spouse_of",
            "partner_of",
            "friend_of",
            "colleague_of",
            "pet_of",
            "owner_of",
            "lives_with",
            "works_at",
            "member_of",
            "attends",
            "related_to",
            "knows",
        ]
    )


# ============================================================================
# Sample Extractor
# ============================================================================


@dataclass
class ExtractedLabels:
    """Container for all extracted labels for a single sample."""

    emotions: torch.Tensor | None = None
    sentiment: int | None = None
    safety: int | None = None
    intent: int | None = None
    ingress: int | None = None
    ner_family_labels: torch.Tensor | None = None
    temporal_labels: torch.Tensor | None = None
    relation_triples: list[tuple[str, int, str]] | None = None
    ner_spans: list[SpanAnnotation] | None = None
    temporal_spans: list[SpanAnnotation] | None = None


class MultiTaskExtractor:
    """Extracts labels for all tasks from unified samples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocabs: V3LabelVocabularies | None = None,
        max_seq_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocabs = vocabs or V3LabelVocabularies()
        self.max_seq_length = max_seq_length

    def extract(self, sample: UnifiedSample, encoding: Any | None = None) -> ExtractedLabels:
        """Extract all labels from a unified sample."""

        if encoding is None:
            encoding = self.tokenizer(
                sample.text,
                max_length=self.max_seq_length,
                truncation=True,
                return_offsets_mapping=True,
            )

        labels = ExtractedLabels()

        # Classification tasks
        if sample.emotions:
            labels.emotions = self.vocabs.EMOTIONS.to_multi_hot(sample.emotions)

        if sample.sentiment:
            labels.sentiment = self.vocabs.SENTIMENT.encode(sample.sentiment)

        if sample.safety_familyos:
            labels.safety = self.vocabs.SAFETY.encode(sample.safety_familyos)

        if sample.intent:
            labels.intent = self.vocabs.INTENT.encode(sample.intent)

        if sample.ingress:
            labels.ingress = self.vocabs.INGRESS.encode(sample.ingress)

        offset_mapping = self._get_offset_mapping(encoding)

        if offset_mapping is not None:
            if sample.ner_family:
                labels.ner_family_labels = self._extract_bio_labels(
                    sample.ner_family, offset_mapping, self.vocabs.NER_FAMILY
                )
                labels.ner_spans = sample.ner_family

            if sample.temporal:
                labels.temporal_labels = self._extract_bio_labels(
                    sample.temporal, offset_mapping, self.vocabs.TEMPORAL
                )
                labels.temporal_spans = sample.temporal

        if sample.relations:
            labels.relation_triples = self._extract_relations(sample.relations)

        return labels

    def _get_offset_mapping(self, encoding: Any) -> list[tuple[int, int]] | None:
        """Extract offset mapping from tokenizer output or provided encoding."""

        if hasattr(encoding, "offset_mapping"):
            mapping = encoding.offset_mapping
            return list(mapping) if mapping is not None else None

        if isinstance(encoding, dict):
            mapping = encoding.get("offset_mapping")
            return list(mapping) if mapping is not None else None

        return None

    def _extract_bio_labels(
        self,
        spans: list[SpanAnnotation],
        offset_mapping: list[tuple[int, int]],
        vocab: LabelVocabulary,
    ) -> torch.Tensor:
        """Convert character-level spans to BIO token labels."""

        seq_len = len(offset_mapping)
        labels = torch.full((seq_len,), vocab.encode("O"), dtype=torch.long)

        for span in spans:
            char_start = span.start
            char_end = span.end
            label_type = span.label

            is_first = True
            for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == 0 and tok_end == 0:
                    continue

                if tok_start < char_end and tok_end > char_start:
                    bio_label = f"B-{label_type}" if is_first else f"I-{label_type}"
                    is_first = False

                    label_id = vocab.encode(bio_label)
                    if label_id >= 0:
                        labels[token_idx] = label_id

        return labels

    def _extract_relations(self, relations: list[RelationTriple]) -> list[tuple[str, int, str]]:
        """Encode relation triples using predicate vocabulary."""

        triples: list[tuple[str, int, str]] = []

        for relation in relations:
            predicate_id = self.vocabs.RELATION_PREDICATES.encode(relation.predicate)
            if predicate_id >= 0:
                triples.append((relation.subject, predicate_id, relation.object))

        return triples

    def extract_batch(
        self,
        samples: list[UnifiedSample],
        encodings: list[Any] | None = None,
    ) -> list[ExtractedLabels]:
        """Extract labels for a batch of samples."""

        if encodings is None:
            encodings = [None] * len(samples)

        return [
            self.extract(sample, encoding)
            for sample, encoding in zip(samples, encodings, strict=True)
        ]


# ============================================================================
# Batch Collation Helpers
# ============================================================================


def collate_classification_labels(
    labels: list[int | None], ignore_index: int = -100
) -> torch.Tensor:
    """Collate single-label classification targets with ignore_index padding."""

    return torch.tensor(
        [label if label is not None else ignore_index for label in labels], dtype=torch.long
    )


def collate_multi_label(labels: list[torch.Tensor | None], num_labels: int) -> torch.Tensor:
    """Collate multi-label targets, replacing missing entries with zeros."""

    batch: list[torch.Tensor] = []
    for label in labels:
        if label is not None:
            batch.append(label)
        else:
            batch.append(torch.zeros(num_labels, dtype=torch.float32))
    return torch.stack(batch)


def collate_token_labels(
    labels: list[torch.Tensor | None],
    max_len: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Collate token-level labels with padding or truncation."""

    batch: list[torch.Tensor] = []
    for label in labels:
        if label is not None:
            if len(label) < max_len:
                padded = torch.full((max_len,), ignore_index, dtype=torch.long)
                padded[: len(label)] = label
                batch.append(padded)
            else:
                batch.append(label[:max_len])
        else:
            batch.append(torch.full((max_len,), ignore_index, dtype=torch.long))
    return torch.stack(batch)

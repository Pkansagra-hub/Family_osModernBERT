"""
GlobalPointer Collator for Span-Based NER Training.

This module provides a data collator that prepares span-format NER data
for GlobalPointer training. It converts character-level entity spans to
token-level span labels in a (B, num_labels, L, L) format.

The collator inherits from BaseCollator to reuse padding logic and follows
the same dataclass pattern as existing collators in the codebase.

Key Features:
    - Character-to-token span alignment using offset_mapping
    - Efficient span label matrix construction
    - Factory functions for ner_general and ner_family tasks
    - Decode method for inference

Usage:
    from modeling_studio.data.globalpointer_collator import (
        GlobalPointerCollator,
        create_ner_general_collator,
    )

    collator = create_ner_general_collator(tokenizer)
    batch = collator(features)
    # batch["input_ids"]: (B, L)
    # batch["attention_mask"]: (B, L)
    # batch["span_labels"]: (B, num_labels, L, L)

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# =============================================================================
# Label Configurations
# =============================================================================

# Standard NER labels (CoNLL-2003 style)
NER_GENERAL_LABELS = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "MISC": 3,
}

# FamilyOS NER labels (matches BIO data from familyos/ner_family)
NER_FAMILY_LABELS = {
    "PERSON": 0,
    "KINSHIP": 1,
    "NICKNAME": 2,
    "PET": 3,
    "HOME_LOC": 4,
    "FAMILY_EVENT": 5,
    "ROUTINE": 6,
    "TRADITION": 7,
    "MILESTONE": 8,
    "HEIRLOOM": 9,
}

# Temporal labels (matches BIO data from familyos/temporal)
TEMPORAL_LABELS = {
    "DATE_ABS": 0,
    "DATE_REL": 1,
    "TIME": 2,
    "DURATION": 3,
    "FREQUENCY": 4,
    "AGE": 5,
}


# =============================================================================
# GlobalPointer Collator
# =============================================================================


@dataclass
class GlobalPointerCollator:
    """
    Data collator for GlobalPointer NER training.

    Converts span-format data:
        {"text": "...", "entities": [{"start": 0, "end": 4, "label": "PER"}]}

    To batched tensors:
        input_ids: (B, L)
        attention_mask: (B, L)
        span_labels: (B, num_labels, L, L) - 1 where span[tok_start:tok_end] exists

    The span_labels tensor has shape (B, num_labels, L, L) where:
        - B = batch size
        - num_labels = number of entity types
        - L = sequence length
        - span_labels[b, k, i, j] = 1.0 if tokens[i:j+1] is entity of type k

    Note: Only the upper triangle (i <= j) is used since start <= end always.

    Args:
        tokenizer: HuggingFace tokenizer with offset_mapping support
        label_to_id: Mapping from label string to integer id
        max_length: Maximum sequence length
        padding: Padding strategy (True, "max_length", or "longest")
        return_tensors: Return tensor type ("pt" for PyTorch)
    """

    tokenizer: PreTrainedTokenizerBase
    label_to_id: dict[str, int]
    max_length: int = 512
    padding: bool | str = True
    return_tensors: str = "pt"

    # Computed after init
    num_labels: int = field(init=False)
    id_to_label: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        self.num_labels = len(self.label_to_id)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """
        Collate a batch of span-format samples.

        Args:
            features: List of dicts with "text" and "entities" keys.
                Each entity has "start", "end", "label" keys.

        Returns:
            Dictionary with:
                - input_ids: (B, L) token ids
                - attention_mask: (B, L) attention mask
                - span_labels: (B, num_labels, L, L) span labels
        """
        texts = [f["text"] for f in features]

        # Tokenize with offset mapping for char-to-token alignment
        encoding = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors=None,  # Get lists first for processing
        )

        batch_size = len(texts)
        seq_len = len(encoding["input_ids"][0])

        # Initialize span labels: (B, num_labels, L, L)
        # Upper triangular only matters, but initialize full for simplicity
        span_labels = torch.zeros(
            batch_size, self.num_labels, seq_len, seq_len,
            dtype=torch.float32,
        )

        # Fill in entity spans
        for b, feature in enumerate(features):
            offset_mapping = encoding["offset_mapping"][b]
            entities = feature.get("entities", [])

            for entity in entities:
                char_start = entity.get("start")
                char_end = entity.get("end")
                label = entity.get("label", entity.get("type"))

                # Skip if missing required fields
                if char_start is None or char_end is None or label is None:
                    logger.debug(f"Skipping entity with missing fields: {entity}")
                    continue

                # Skip unknown labels
                if label not in self.label_to_id:
                    logger.debug(f"Skipping unknown label: {label}")
                    continue

                label_id = self.label_to_id[label]

                # Map character span to token span
                tok_start, tok_end = self._char_to_token_span(
                    offset_mapping, char_start, char_end
                )

                if tok_start is not None and tok_end is not None:
                    # Set label at (tok_start, tok_end) position
                    # GlobalPointer uses inclusive token indices
                    span_labels[b, label_id, tok_start, tok_end] = 1.0

        # Remove offset_mapping from output (not needed for training)
        encoding.pop("offset_mapping")

        # Convert to tensors
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "span_labels": span_labels,
        }

    def _char_to_token_span(
        self,
        offset_mapping: list[tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> tuple[int | None, int | None]:
        """
        Convert character-level span to token-level span.

        Uses the tokenizer's offset_mapping to find which tokens contain
        the character span. Handles partial overlaps by expanding to
        the full containing tokens.

        Args:
            offset_mapping: List of (start_char, end_char) for each token.
                Special tokens have (0, 0) offset.
            char_start: Character start position (inclusive)
            char_end: Character end position (exclusive)

        Returns:
            (tok_start, tok_end) inclusive token indices, or (None, None)
            if the span falls outside the tokenized sequence.

        Examples:
            text: "New York City"
            tokens: ["New", "York", "City"]
            offset_mapping: [(0, 3), (4, 8), (9, 13)]

            char_span (0, 8) -> "New York" -> token_span (0, 1)
            char_span (4, 13) -> "York City" -> token_span (1, 2)

        Edge Cases:
            - Span in special token region (offset 0,0): Returns (None, None)
            - Span partially overlaps token: Expands to full token
            - Span after truncation: Returns (None, None)
        """
        tok_start = None
        tok_end = None

        for i, (cs, ce) in enumerate(offset_mapping):
            # Skip special tokens (CLS, SEP, PAD have offset (0, 0))
            if cs == 0 and ce == 0:
                continue

            # Check if token overlaps with character span
            # Token overlaps if: token_start < char_end AND token_end > char_start
            if cs < char_end and ce > char_start:
                if tok_start is None:
                    tok_start = i
                tok_end = i  # Keep updating for last overlapping token

        return tok_start, tok_end

    def decode_spans(
        self,
        span_scores: Tensor,
        offset_mapping: list[tuple[int, int]],
        text: str,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Decode GlobalPointer output to span format.

        Converts the span score matrix back to entity annotations.
        Used during inference.

        Args:
            span_scores: (num_labels, L, L) logits from GlobalPointer
            offset_mapping: Token offset mapping from tokenizer
            text: Original text string
            threshold: Minimum score threshold (after sigmoid)

        Returns:
            List of entity dicts: {"start", "end", "label", "text", "score"}
        """
        entities = []

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(span_scores)

        for label_id in range(self.num_labels):
            # Find spans above threshold in upper triangle
            for tok_start in range(probs.shape[1]):
                for tok_end in range(tok_start, probs.shape[2]):
                    score = probs[label_id, tok_start, tok_end].item()
                    if score > threshold:
                        # Convert token span to character span
                        char_start, char_end = self._token_to_char_span(
                            offset_mapping, tok_start, tok_end
                        )

                        if char_start is not None and char_end is not None:
                            entities.append({
                                "start": char_start,
                                "end": char_end,
                                "label": self.id_to_label[label_id],
                                "text": text[char_start:char_end],
                                "score": score,
                            })

        # Sort by start position, then by score (descending)
        entities.sort(key=lambda x: (x["start"], -x["score"]))

        return entities

    def _token_to_char_span(
        self,
        offset_mapping: list[tuple[int, int]],
        tok_start: int,
        tok_end: int,
    ) -> tuple[int | None, int | None]:
        """
        Convert token span back to character span.

        Args:
            offset_mapping: Token offset mapping
            tok_start: Start token index (inclusive)
            tok_end: End token index (inclusive)

        Returns:
            (char_start, char_end) or (None, None) if invalid
        """
        if tok_start >= len(offset_mapping) or tok_end >= len(offset_mapping):
            return None, None

        cs_start, _ = offset_mapping[tok_start]
        _, ce_end = offset_mapping[tok_end]

        # Skip if special tokens (both offsets are 0)
        if cs_start == 0 and ce_end == 0:
            return None, None

        return cs_start, ce_end


# =============================================================================
# Factory Functions
# =============================================================================


def create_ner_general_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> GlobalPointerCollator:
    """
    Factory function to create GlobalPointerCollator for ner_general task.

    Uses standard 4-class NER labels: PER, ORG, LOC, MISC.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Configured GlobalPointerCollator
    """
    return GlobalPointerCollator(
        tokenizer=tokenizer,
        label_to_id=NER_GENERAL_LABELS.copy(),
        max_length=max_length,
    )


def create_ner_family_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> GlobalPointerCollator:
    """
    Factory function to create GlobalPointerCollator for ner_family task.

    Uses FamilyOS NER labels: KINSHIP, MILESTONE, HEIRLOOM, PET, etc.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Configured GlobalPointerCollator
    """
    return GlobalPointerCollator(
        tokenizer=tokenizer,
        label_to_id=NER_FAMILY_LABELS.copy(),
        max_length=max_length,
    )


def create_temporal_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> GlobalPointerCollator:
    """
    Factory function to create GlobalPointerCollator for temporal task.

    Uses temporal labels: DATE_ABS, DATE_REL, TIME, DURATION, RECURRING.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Configured GlobalPointerCollator
    """
    return GlobalPointerCollator(
        tokenizer=tokenizer,
        label_to_id=TEMPORAL_LABELS.copy(),
        max_length=max_length,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GlobalPointerCollator",
    "create_ner_general_collator",
    "create_ner_family_collator",
    "create_temporal_collator",
    "NER_GENERAL_LABELS",
    "NER_FAMILY_LABELS",
    "TEMPORAL_LABELS",
]

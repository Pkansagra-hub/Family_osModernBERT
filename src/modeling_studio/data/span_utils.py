"""
Span Conversion Utilities for NER Data.

This module provides utilities for converting between BIO-tagged token sequences
and character-level span annotations. These are essential for GlobalPointer
training which uses span-based representations instead of BIO tags.

Key Functions:
    - bio_to_spans: Convert BIO tags to character spans
    - flat_to_spans: Convert flat labels to character spans (for Few-NERD)
    - validate_spans: Validate span annotations

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def bio_to_spans(
    tokens: list[str],
    bio_tags: list[int],
    label_names: list[str],
    label_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert BIO-tagged token sequence to span format.

    Takes tokenized text with BIO tag indices and converts to character-level
    span annotations suitable for GlobalPointer training.

    Args:
        tokens: Word tokens, e.g., ["Emma", "lives", "in", "New", "York"]
        bio_tags: BIO tag indices, e.g., [1, 0, 0, 5, 6] for B-PER, O, O, B-LOC, I-LOC
        label_names: Ordered list mapping index to label name,
            e.g., ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", ...]
        label_mapping: Optional mapping to normalize labels,
            e.g., {"PERSON": "PER", "LOCATION": "LOC"}

    Returns:
        Dictionary with:
            - text: Reconstructed text from tokens (space-joined)
            - entities: List of entity spans, each with:
                - start: Character start index (inclusive)
                - end: Character end index (exclusive)
                - label: Entity type (e.g., "PER", "ORG", "LOC", "MISC")
                - text: The entity text

    Example:
        >>> tokens = ["Emma", "lives", "in", "New", "York"]
        >>> bio_tags = [1, 0, 0, 5, 6]
        >>> label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        >>> result = bio_to_spans(tokens, bio_tags, label_names)
        >>> result["text"]
        'Emma lives in New York'
        >>> result["entities"]
        [{'start': 0, 'end': 4, 'label': 'PER', 'text': 'Emma'},
         {'start': 14, 'end': 22, 'label': 'LOC', 'text': 'New York'}]

    Edge Cases:
        - Orphan I-tags (no preceding B-tag): Treated as B-tag (start new entity)
        - Empty tokens: Skipped
        - Special tokens ([CLS], [SEP], [PAD]): Skipped
    """
    if not tokens or not bio_tags:
        return {"text": "", "entities": []}

    if len(tokens) != len(bio_tags):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(bio_tags)} tags"
        )

    # Filter out special tokens
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    filtered_pairs = [
        (tok, tag)
        for tok, tag in zip(tokens, bio_tags)
        if tok not in special_tokens and tok.strip()
    ]

    if not filtered_pairs:
        return {"text": "", "entities": []}

    tokens, bio_tags = zip(*filtered_pairs)
    tokens = list(tokens)
    bio_tags = list(bio_tags)

    # Build text and track character positions
    text = " ".join(tokens)
    entities: list[dict[str, Any]] = []

    # Track character position for each token
    char_positions: list[tuple[int, int]] = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        char_positions.append((start, end))
        current_pos = end + 1  # +1 for space

    # Extract entities
    i = 0
    while i < len(tokens):
        tag_idx = bio_tags[i]

        # Skip O tags
        if tag_idx == 0:
            i += 1
            continue

        # Get label name
        if tag_idx >= len(label_names):
            logger.warning(f"Unknown tag index {tag_idx}, skipping")
            i += 1
            continue

        tag_name = label_names[tag_idx]

        # Parse B- or I- prefix
        if tag_name.startswith("B-"):
            entity_type = tag_name[2:]
            is_beginning = True
        elif tag_name.startswith("I-"):
            entity_type = tag_name[2:]
            is_beginning = False
        else:
            # Unexpected format (e.g., just "PER")
            entity_type = tag_name
            is_beginning = True

        # Handle orphan I-tag: treat as B-tag
        if not is_beginning:
            logger.debug(f"Orphan I-tag '{tag_name}' at position {i}, treating as B-tag")
            is_beginning = True

        # Apply label mapping if provided
        if label_mapping and entity_type in label_mapping:
            entity_type = label_mapping[entity_type]

        # Start entity span
        entity_start_char = char_positions[i][0]
        entity_end_char = char_positions[i][1]
        entity_token_end = i

        # Find corresponding I-tag index
        i_tag_name = f"I-{tag_name[2:]}" if tag_name.startswith("B-") else tag_name
        i_tag_idx = None
        for idx, name in enumerate(label_names):
            if name == i_tag_name:
                i_tag_idx = idx
                break

        # Consume consecutive I-tags
        j = i + 1
        while j < len(tokens):
            next_tag_idx = bio_tags[j]
            if i_tag_idx is not None and next_tag_idx == i_tag_idx:
                entity_end_char = char_positions[j][1]
                entity_token_end = j
                j += 1
            else:
                break

        # Create entity
        entity_text = text[entity_start_char:entity_end_char]
        entities.append({
            "start": entity_start_char,
            "end": entity_end_char,
            "label": entity_type,
            "text": entity_text,
        })

        i = j

    return {"text": text, "entities": entities}


def flat_to_spans(
    tokens: list[str],
    flat_tags: list[int | str],
    label_names: list[str] | None = None,
    label_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert flat-labeled token sequence to span format.

    This handles datasets like Few-NERD that use flat labels (e.g., "person",
    "location") instead of BIO tags. Consecutive tokens with the same label
    are merged into a single entity span.

    Args:
        tokens: Word tokens, e.g., ["John", "Smith", "lives", "in", "London"]
        flat_tags: Flat label indices or strings, e.g., [1, 1, 0, 0, 2] or
            ["person", "person", "O", "O", "location"]
        label_names: Optional mapping from index to label name (if tags are ints)
        label_mapping: Optional mapping to normalize labels,
            e.g., {"person": "PER", "location": "LOC"}

    Returns:
        Dictionary with text and entities (same format as bio_to_spans)

    Example:
        >>> tokens = ["John", "Smith", "lives", "in", "London"]
        >>> flat_tags = ["person", "person", "O", "O", "location"]
        >>> label_mapping = {"person": "PER", "location": "LOC"}
        >>> result = flat_to_spans(tokens, flat_tags, label_mapping=label_mapping)
        >>> result["entities"]
        [{'start': 0, 'end': 10, 'label': 'PER', 'text': 'John Smith'},
         {'start': 20, 'end': 26, 'label': 'LOC', 'text': 'London'}]
    """
    if not tokens or not flat_tags:
        return {"text": "", "entities": []}

    if len(tokens) != len(flat_tags):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(flat_tags)} tags"
        )

    # Convert indices to label names if needed
    if label_names and all(isinstance(t, int) for t in flat_tags):
        flat_tags = [label_names[t] if t < len(label_names) else "O" for t in flat_tags]

    # Filter out special tokens
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    filtered_pairs = [
        (tok, tag)
        for tok, tag in zip(tokens, flat_tags)
        if tok not in special_tokens and tok.strip()
    ]

    if not filtered_pairs:
        return {"text": "", "entities": []}

    tokens, flat_tags = zip(*filtered_pairs)
    tokens = list(tokens)
    flat_tags = list(flat_tags)

    # Build text and track character positions
    text = " ".join(tokens)
    entities: list[dict[str, Any]] = []

    # Track character position for each token
    char_positions: list[tuple[int, int]] = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        char_positions.append((start, end))
        current_pos = end + 1

    # Extract entities by merging consecutive same-type tokens
    i = 0
    while i < len(tokens):
        tag = str(flat_tags[i]).lower()

        # Skip O/outside tags
        if tag in ("o", "0", "outside", ""):
            i += 1
            continue

        # Apply label mapping
        entity_type = tag
        if label_mapping:
            entity_type = label_mapping.get(tag, label_mapping.get(tag.upper(), tag))

        # Start entity span
        entity_start_char = char_positions[i][0]
        entity_end_char = char_positions[i][1]

        # Merge consecutive same-type tokens
        j = i + 1
        while j < len(tokens):
            next_tag = str(flat_tags[j]).lower()
            if next_tag == tag:
                entity_end_char = char_positions[j][1]
                j += 1
            else:
                break

        # Create entity
        entity_text = text[entity_start_char:entity_end_char]
        entities.append({
            "start": entity_start_char,
            "end": entity_end_char,
            "label": entity_type.upper() if entity_type.islower() else entity_type,
            "text": entity_text,
        })

        i = j

    return {"text": text, "entities": entities}


def validate_spans(
    sample: dict[str, Any],
    valid_labels: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a span-annotated sample.

    Checks for common issues in span annotations that could cause training
    problems.

    Args:
        sample: Dictionary with "text" and "entities" keys
        valid_labels: Optional set of valid label names

    Returns:
        Tuple of (is_valid, list of error messages)

    Validation Checks:
        1. Required keys present (text, entities)
        2. Text is non-empty string
        3. Entities is a list
        4. Each entity has start, end, label keys
        5. start < end (valid span)
        6. Span indices within text bounds
        7. Entity text matches text[start:end]
        8. Labels are in valid_labels (if provided)
        9. No overlapping spans
    """
    errors: list[str] = []

    # Check required keys
    if "text" not in sample:
        errors.append("Missing 'text' key")
        return False, errors

    if "entities" not in sample:
        errors.append("Missing 'entities' key")
        return False, errors

    text = sample["text"]
    entities = sample["entities"]

    # Check text
    if not isinstance(text, str):
        errors.append(f"'text' must be string, got {type(text)}")
        return False, errors

    if not text.strip():
        # Empty text is valid if no entities
        if entities:
            errors.append("Empty text but has entities")
            return False, errors
        return True, []

    # Check entities
    if not isinstance(entities, list):
        errors.append(f"'entities' must be list, got {type(entities)}")
        return False, errors

    # Validate each entity
    spans_for_overlap: list[tuple[int, int, str]] = []

    for i, entity in enumerate(entities):
        prefix = f"Entity {i}: "

        if not isinstance(entity, dict):
            errors.append(f"{prefix}must be dict, got {type(entity)}")
            continue

        # Check required keys
        for key in ("start", "end", "label"):
            if key not in entity:
                errors.append(f"{prefix}missing '{key}' key")

        if "start" not in entity or "end" not in entity:
            continue

        start, end = entity["start"], entity["end"]

        # Check types
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{prefix}start/end must be int")
            continue

        # Check valid span
        if start >= end:
            errors.append(f"{prefix}invalid span (start={start} >= end={end})")
            continue

        # Check bounds
        if start < 0:
            errors.append(f"{prefix}start < 0")
        if end > len(text):
            errors.append(f"{prefix}end ({end}) > text length ({len(text)})")
            continue

        # Check text match
        if "text" in entity:
            expected_text = text[start:end]
            actual_text = entity["text"]
            if expected_text != actual_text:
                errors.append(
                    f"{prefix}text mismatch: '{actual_text}' vs '{expected_text}'"
                )

        # Check label
        if "label" in entity and valid_labels:
            if entity["label"] not in valid_labels:
                errors.append(
                    f"{prefix}invalid label '{entity['label']}', "
                    f"expected one of {valid_labels}"
                )

        spans_for_overlap.append((start, end, entity.get("label", "")))

    # Check for overlapping spans
    spans_sorted = sorted(spans_for_overlap, key=lambda x: (x[0], x[1]))
    for i in range(len(spans_sorted) - 1):
        s1, e1, l1 = spans_sorted[i]
        s2, e2, l2 = spans_sorted[i + 1]
        if s2 < e1:  # Overlap
            errors.append(
                f"Overlapping spans: [{s1}:{e1}] ({l1}) and [{s2}:{e2}] ({l2})"
            )

    return len(errors) == 0, errors


def spans_to_bio(
    text: str,
    entities: list[dict[str, Any]],
    tokenizer_fn: callable | None = None,
) -> tuple[list[str], list[str]]:
    """
    Convert span annotations back to BIO format.

    This is the inverse of bio_to_spans, useful for compatibility with
    existing BIO-based evaluation code.

    Args:
        text: The text string
        entities: List of entity spans with start, end, label
        tokenizer_fn: Optional custom tokenizer function. If None, uses
            simple whitespace tokenization.

    Returns:
        Tuple of (tokens, bio_tags) where bio_tags are strings like "B-PER"

    Example:
        >>> text = "Emma lives in New York"
        >>> entities = [
        ...     {"start": 0, "end": 4, "label": "PER"},
        ...     {"start": 14, "end": 22, "label": "LOC"}
        ... ]
        >>> tokens, tags = spans_to_bio(text, entities)
        >>> list(zip(tokens, tags))
        [('Emma', 'B-PER'), ('lives', 'O'), ('in', 'O'),
         ('New', 'B-LOC'), ('York', 'I-LOC')]
    """
    if tokenizer_fn is None:
        # Simple whitespace tokenization with position tracking
        tokens = []
        token_spans = []
        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            token_spans.append((match.start(), match.end()))
    else:
        # Use custom tokenizer
        result = tokenizer_fn(text)
        tokens = result["tokens"]
        token_spans = result["spans"]

    # Initialize all tags as O
    bio_tags = ["O"] * len(tokens)

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    # Assign BIO tags
    for entity in sorted_entities:
        e_start, e_end = entity["start"], entity["end"]
        label = entity["label"]

        is_first = True
        for i, (t_start, t_end) in enumerate(token_spans):
            # Check if token overlaps with entity
            if t_start < e_end and t_end > e_start:
                if is_first:
                    bio_tags[i] = f"B-{label}"
                    is_first = False
                else:
                    bio_tags[i] = f"I-{label}"

    return tokens, bio_tags

"""
Data Collators for Multi-Task Learning

This module provides custom data collators that handle different
task types and their specific padding/batching requirements.

Collators:
    - MultiTaskCollator: Routes to task-specific collators
    - SequenceClassificationCollator: For classification tasks
    - TokenClassificationCollator: For NER with label alignment
    - MultiLabelCollator: For multi-label classification
    - EmbeddingCollator: For contrastive learning pairs
    - NLICollator: For premise-hypothesis pairs

Features:
    - Dynamic padding (pad to longest in batch)
    - Label alignment for subword tokenization
    - Support for multi-label targets (multi-hot encoding)
    - Negative sampling for contrastive learning

Task Routing:
    The MultiTaskCollator routes samples to the appropriate collator
    based on the task's problem_type from labels.py:
    - single_label_classification -> SequenceClassificationCollator
    - multi_label_classification -> MultiLabelCollator
    - token_classification -> TokenClassificationCollator
    - nli -> NLICollator
    - embedding -> EmbeddingCollator

Usage:
    from modeling_studio.trainers.collators import MultiTaskCollator

    collator = MultiTaskCollator(tokenizer=tokenizer)

    batch = collator([
        {"task": "sentiment", "input_ids": [...], "labels": 1},
        {"task": "sentiment", "input_ids": [...], "labels": 0},
    ])

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Label value to ignore in loss computation (special tokens, padding)
IGNORE_INDEX = -100


# =============================================================================
# Base Collator
# =============================================================================


@dataclass
class BaseCollator:
    """
    Base class for all collators with common padding logic.

    Args:
        tokenizer: The tokenizer used for encoding (needed for pad_token_id).
        padding: Padding strategy - "longest" (default), "max_length", or False.
        max_length: Maximum sequence length (only used if padding="max_length").
        pad_to_multiple_of: Pad to a multiple of this value (for hardware efficiency).
        return_tensors: Return type - "pt" for PyTorch (default).
    """

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    padding: str | bool = "longest"
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID from tokenizer."""
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        # Fall back to eos_token_id if pad_token_id is not set
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def _pad_sequence(
        self,
        sequences: list[list[int]],
        padding_value: int,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """
        Pad a list of sequences to the same length.

        Args:
            sequences: List of sequences (each is a list of ints).
            padding_value: Value to use for padding.
            max_length: Maximum length to pad to. If None, uses longest sequence.

        Returns:
            Padded tensor of shape (batch_size, seq_length).
        """
        # Determine target length
        if self.padding == "max_length" and max_length is not None:
            target_length = max_length
        elif self.padding == "longest" or self.padding is True:
            target_length = max(len(seq) for seq in sequences)
        else:
            # No padding - just stack (sequences must be same length)
            return torch.tensor(sequences)

        # Apply pad_to_multiple_of
        if self.pad_to_multiple_of is not None:
            target_length = (
                (target_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad sequences
        padded = []
        for seq in sequences:
            padding_length = target_length - len(seq)
            if padding_length > 0:
                padded_seq = seq + [padding_value] * padding_length
            else:
                padded_seq = seq[:target_length]  # Truncate if needed
            padded.append(padded_seq)

        return torch.tensor(padded)


# =============================================================================
# Sequence Classification Collator
# =============================================================================


@dataclass
class SequenceClassificationCollator(BaseCollator):
    """
    Collator for sequence classification tasks.

    Handles single-label classification tasks like sentiment, ingress,
    safety_familyos, and intent classification.

    Expected input format (each sample):
        {
            "input_ids": list[int],
            "attention_mask": list[int],
            "labels": int,
            "task": str  # optional
        }

    Output format:
        {
            "input_ids": torch.Tensor (batch, seq_len),
            "attention_mask": torch.Tensor (batch, seq_len),
            "labels": torch.Tensor (batch,),
            "task": str  # if present in input
        }

    Example:
        >>> collator = SequenceClassificationCollator(tokenizer=tokenizer)
        >>> batch = collator([
        ...     {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0},
        ...     {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 1},
        ... ])
        >>> batch["input_ids"].shape
        torch.Size([2, 3])
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of classification samples.

        Args:
            features: List of sample dicts with input_ids, attention_mask, labels.

        Returns:
            Batched and padded tensors.
        """
        # Extract sequences and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Preserve task info if present
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# Multi-Label Classification Collator
# =============================================================================


@dataclass
class MultiLabelCollator(BaseCollator):
    """
    Collator for multi-label classification tasks.

    Handles tasks like emotions (GoEmotions) and safety_generic
    where multiple labels can be active simultaneously.

    Expected input format (each sample):
        {
            "input_ids": list[int],
            "attention_mask": list[int],
            "labels": list[float],  # multi-hot encoding
            "task": str  # optional
        }

    Output format:
        {
            "input_ids": torch.Tensor (batch, seq_len),
            "attention_mask": torch.Tensor (batch, seq_len),
            "labels": torch.Tensor (batch, num_labels),  # float for BCE loss
            "task": str  # if present in input
        }

    Example:
        >>> collator = MultiLabelCollator(tokenizer=tokenizer)
        >>> batch = collator([
        ...     {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1.0, 0.0, 1.0]},
        ...     {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0.0, 1.0, 0.0]},
        ... ])
        >>> batch["labels"].shape
        torch.Size([2, 3])
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of multi-label classification samples.

        Args:
            features: List of sample dicts with input_ids, attention_mask, labels.

        Returns:
            Batched and padded tensors.
        """
        # Extract sequences and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Auto-detect single-label vs multi-label based on first sample
        # Single-label: labels is int, Multi-label: labels is list
        first_label = labels[0]
        if isinstance(first_label, (int, float)) and not isinstance(first_label, list):
            # Single-label classification (Stage A super-labels)
            label_dtype = torch.long
        else:
            # Multi-label classification (BCE loss expects float)
            label_dtype = torch.float

        # Pad sequences
        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
            "labels": torch.tensor(labels, dtype=label_dtype),
        }

        # Preserve task info if present
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# Token Classification Collator
# =============================================================================


@dataclass
class TokenClassificationCollator(BaseCollator):
    """
    Collator for token classification tasks (NER, temporal extraction).

    Handles token-level labels with proper padding using IGNORE_INDEX (-100)
    for padding positions so they're ignored in loss computation.

    Expected input format (each sample):
        {
            "input_ids": list[int],
            "attention_mask": list[int],
            "labels": list[int],  # per-token labels, same length as input_ids
            "task": str  # optional
        }

    Output format:
        {
            "input_ids": torch.Tensor (batch, seq_len),
            "attention_mask": torch.Tensor (batch, seq_len),
            "labels": torch.Tensor (batch, seq_len),  # -100 for padding
            "task": str  # if present in input
        }

    Example:
        >>> collator = TokenClassificationCollator(tokenizer=tokenizer)
        >>> batch = collator([
        ...     {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [0, 1, 0]},
        ...     {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0, 2]},
        ... ])
        >>> batch["labels"]
        tensor([[ 0,  1,  0],
                [ 0,  2, -100]])  # -100 for padding position
    """

    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of token classification samples.

        Args:
            features: List of sample dicts with input_ids, attention_mask, labels.

        Returns:
            Batched and padded tensors.
        """
        # Extract sequences and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
            "labels": self._pad_sequence(labels, self.label_pad_token_id, self.max_length),
        }

        # Preserve task info if present
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# NLI Collator
# =============================================================================


@dataclass
class NLICollator(BaseCollator):
    """
    Collator for Natural Language Inference (NLI) tasks.

    Handles premise-hypothesis pairs. The input should already be tokenized
    as a pair (premise + [SEP] + hypothesis format).

    Expected input format (each sample):
        {
            "input_ids": list[int],  # Already concatenated: [CLS] premise [SEP] hypothesis [SEP]
            "attention_mask": list[int],
            "token_type_ids": list[int] | None,  # Optional segment IDs
            "labels": int,  # 0=entailment, 1=neutral, 2=contradiction
            "task": str  # optional
        }

    Output format:
        {
            "input_ids": torch.Tensor (batch, seq_len),
            "attention_mask": torch.Tensor (batch, seq_len),
            "token_type_ids": torch.Tensor (batch, seq_len) | None,
            "labels": torch.Tensor (batch,),
            "task": str  # if present in input
        }

    Example:
        >>> collator = NLICollator(tokenizer=tokenizer)
        >>> batch = collator([
        ...     {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": 0},
        ...     {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 2},
        ... ])
        >>> batch["labels"]
        tensor([0, 2])
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of NLI samples.

        Args:
            features: List of sample dicts with input_ids, attention_mask, labels.

        Returns:
            Batched and padded tensors.
        """
        # Extract sequences and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Handle token_type_ids if present (some models use them)
        if "token_type_ids" in features[0] and features[0]["token_type_ids"] is not None:
            token_type_ids = [f["token_type_ids"] for f in features]
            batch["token_type_ids"] = self._pad_sequence(token_type_ids, 0, self.max_length)

        # Preserve task info if present
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# Embedding Collator
# =============================================================================


@dataclass
class EmbeddingCollator(BaseCollator):
    """
    Collator for embedding/contrastive learning tasks.

    Handles two formats:
    1. Triplets: anchor, positive, negative (for triplet loss)
    2. Pairs with scores: sentence1, sentence2, score (for STS tasks)

    Expected input format - Triplet mode (each sample):
        {
            "anchor_input_ids": list[int],
            "anchor_attention_mask": list[int],
            "positive_input_ids": list[int],
            "positive_attention_mask": list[int],
            "negative_input_ids": list[int],  # Optional
            "negative_attention_mask": list[int],  # Optional
            "task": str  # optional
        }

    Expected input format - Pair mode (each sample):
        {
            "input_ids_1": list[int],
            "attention_mask_1": list[int],
            "input_ids_2": list[int],
            "attention_mask_2": list[int],
            "score": float,  # Similarity score (0.0 to 1.0)
            "task": str  # optional
        }

    Also supports simpler format (auto-detected):
        {
            "input_ids": list[int],
            "attention_mask": list[int],
            "task": str  # For in-batch negative sampling
        }

    Output format depends on input format.
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of embedding samples.

        Args:
            features: List of sample dicts in triplet or pair format.

        Returns:
            Batched and padded tensors.
        """
        # Detect format based on keys present
        sample = features[0]

        # Format 1: Triplet format (anchor, positive, negative)
        if "anchor_input_ids" in sample:
            return self._collate_triplets(features)

        # Format 2: Pair with scores (sentence1, sentence2, score)
        elif "input_ids_1" in sample:
            return self._collate_pairs(features)

        # Format 3: Simple format (for in-batch negatives)
        elif "input_ids" in sample:
            return self._collate_simple(features)

        else:
            raise ValueError(
                f"Unknown embedding format. Expected keys like 'anchor_input_ids', "
                f"'input_ids_1', or 'input_ids'. Got: {list(sample.keys())}"
            )

    def _collate_triplets(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate triplet format samples."""
        # Extract anchor sequences
        anchor_input_ids = [f["anchor_input_ids"] for f in features]
        anchor_attention_mask = [f["anchor_attention_mask"] for f in features]

        # Extract positive sequences
        positive_input_ids = [f["positive_input_ids"] for f in features]
        positive_attention_mask = [f["positive_attention_mask"] for f in features]

        batch = {
            "anchor_input_ids": self._pad_sequence(
                anchor_input_ids, self.pad_token_id, self.max_length
            ),
            "anchor_attention_mask": self._pad_sequence(anchor_attention_mask, 0, self.max_length),
            "positive_input_ids": self._pad_sequence(
                positive_input_ids, self.pad_token_id, self.max_length
            ),
            "positive_attention_mask": self._pad_sequence(
                positive_attention_mask, 0, self.max_length
            ),
        }

        # Use actual labels (similarity scores) if present, else default to 1.0
        # This preserves STS-B scores for proper Spearman/Pearson correlation eval
        if "labels" in features[0] and features[0]["labels"] is not None:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        else:
            # Fallback: dummy labels for contrastive learning (positive pairs assumed similar)
            batch["labels"] = torch.ones(len(features), dtype=torch.float)

        # Add negatives if present
        if "negative_input_ids" in features[0]:
            negative_input_ids = [f["negative_input_ids"] for f in features]
            negative_attention_mask = [f["negative_attention_mask"] for f in features]
            batch["negative_input_ids"] = self._pad_sequence(
                negative_input_ids, self.pad_token_id, self.max_length
            )
            batch["negative_attention_mask"] = self._pad_sequence(
                negative_attention_mask, 0, self.max_length
            )

        # Preserve task info
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch

    def _collate_pairs(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate pair format samples (with similarity scores)."""
        # Extract sentence 1
        input_ids_1 = [f["input_ids_1"] for f in features]
        attention_mask_1 = [f["attention_mask_1"] for f in features]

        # Extract sentence 2
        input_ids_2 = [f["input_ids_2"] for f in features]
        attention_mask_2 = [f["attention_mask_2"] for f in features]

        # Extract scores
        scores = [f["score"] for f in features]

        batch = {
            "input_ids_1": self._pad_sequence(input_ids_1, self.pad_token_id, self.max_length),
            "attention_mask_1": self._pad_sequence(attention_mask_1, 0, self.max_length),
            "input_ids_2": self._pad_sequence(input_ids_2, self.pad_token_id, self.max_length),
            "attention_mask_2": self._pad_sequence(attention_mask_2, 0, self.max_length),
            "labels": torch.tensor(scores, dtype=torch.float),  # Similarity scores
        }

        # Preserve task info
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch

    def _collate_simple(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate simple format (for in-batch negatives)."""
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
        }

        # Use actual labels if present, else default to 1.0 for in-batch negatives
        if "labels" in features[0] and features[0]["labels"] is not None:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        else:
            batch["labels"] = torch.ones(len(features), dtype=torch.float)

        # Preserve task info
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# Relation Extraction Collator
# =============================================================================


@dataclass
class RelationCollator(BaseCollator):
    """
    Collator for relation extraction tasks.

    Handles samples with entity span markers for relation classification
    between two entities.

    Expected input format (each sample):
        {
            "input_ids": list[int],  # With entity markers
            "attention_mask": list[int],
            "entity1_mask": list[int],  # Mask for first entity
            "entity2_mask": list[int],  # Mask for second entity
            "labels": int,  # Relation type
            "task": str  # optional
        }

    Output format:
        {
            "input_ids": torch.Tensor (batch, seq_len),
            "attention_mask": torch.Tensor (batch, seq_len),
            "entity1_mask": torch.Tensor (batch, seq_len),
            "entity2_mask": torch.Tensor (batch, seq_len),
            "labels": torch.Tensor (batch,),
            "task": str  # if present in input
        }
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of relation extraction samples.

        Args:
            features: List of sample dicts with input_ids, entity masks, labels.

        Returns:
            Batched and padded tensors.
        """
        # Extract sequences, masks, and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        entity1_mask = [f["entity1_mask"] for f in features]
        entity2_mask = [f["entity2_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        batch = {
            "input_ids": self._pad_sequence(input_ids, self.pad_token_id, self.max_length),
            "attention_mask": self._pad_sequence(attention_mask, 0, self.max_length),
            "entity1_mask": self._pad_sequence(entity1_mask, 0, self.max_length),
            "entity2_mask": self._pad_sequence(entity2_mask, 0, self.max_length),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Preserve task info if present
        if "task" in features[0]:
            batch["task"] = features[0]["task"]

        return batch


# =============================================================================
# Multi-Task Collator
# =============================================================================


# Task to collator type mapping based on problem_type from labels.py
TASK_COLLATOR_MAPPING: dict[str, type] = {
    # Single-label classification tasks
    "sentiment": SequenceClassificationCollator,
    "ingress": SequenceClassificationCollator,
    "safety_familyos": SequenceClassificationCollator,
    "intent": SequenceClassificationCollator,
    # Multi-label classification tasks
    "emotions": MultiLabelCollator,
    "safety_generic": MultiLabelCollator,
    # Token classification tasks
    "ner_general": TokenClassificationCollator,
    "ner_family": TokenClassificationCollator,
    "temporal": TokenClassificationCollator,
    # NLI task
    "nli": NLICollator,
    # Embedding task
    "embedding": EmbeddingCollator,
    # Relation extraction
    "relation": RelationCollator,
}


@dataclass
class MultiTaskCollator:
    """
    Multi-task collator that routes to task-specific collators.

    Automatically detects the task from the 'task' field in samples
    and routes to the appropriate collator.

    Args:
        tokenizer: The tokenizer used for encoding (passed to sub-collators).
        task_collators: Optional dict mapping task names to custom collators.
            If not provided, uses TASK_COLLATOR_MAPPING defaults.
        default_collator: Fallback collator for unknown tasks.
            Default: SequenceClassificationCollator
        padding: Padding strategy passed to all collators.
        max_length: Maximum sequence length passed to all collators.
        pad_to_multiple_of: Pad to multiple of this value (hardware efficiency).

    Example:
        >>> collator = MultiTaskCollator(tokenizer=tokenizer)
        >>> batch = collator([
        ...     {"task": "sentiment", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 1},
        ...     {"task": "sentiment", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": 0},
        ... ])
        >>> assert "input_ids" in batch
        >>> assert "labels" in batch
    """

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    task_collators: dict[str, BaseCollator] | None = None
    default_collator_cls: type = field(default=SequenceClassificationCollator)
    padding: str | bool = "longest"
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    # Cached collator instances
    _collator_cache: dict[str, BaseCollator] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize task collators from mapping."""
        if self.task_collators is None:
            self.task_collators = {}

    def _get_collator(self, task: str) -> BaseCollator:
        """
        Get or create a collator for a specific task.

        Args:
            task: Task name.

        Returns:
            Collator instance for the task.
        """
        # Check cache first
        if task in self._collator_cache:
            return self._collator_cache[task]

        # Check custom task collators
        if task in self.task_collators:
            collator = self.task_collators[task]
            self._collator_cache[task] = collator
            return collator

        # Handle replay tasks by stripping _replay suffix
        base_task = task
        if task.endswith("_replay"):
            base_task = task[:-7]  # Remove "_replay" suffix

        # Check default mapping (use base_task for replay tasks)
        if base_task in TASK_COLLATOR_MAPPING:
            collator_cls = TASK_COLLATOR_MAPPING[base_task]
        elif task in TASK_COLLATOR_MAPPING:
            collator_cls = TASK_COLLATOR_MAPPING[task]
        else:
            logger.warning(
                f"Unknown task '{task}', using default collator: {self.default_collator_cls.__name__}"
            )
            collator_cls = self.default_collator_cls

        # Create collator instance
        collator = collator_cls(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        self._collator_cache[task] = collator
        return collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of multi-task samples.

        All samples in the batch must be from the same task.

        Args:
            features: List of sample dicts. Each must have a 'task' field.

        Returns:
            Batched and padded tensors.

        Raises:
            ValueError: If samples are from different tasks.
            KeyError: If 'task' field is missing.
        """
        if not features:
            raise ValueError("Cannot collate empty batch")

        # Get task from first sample
        if "task" not in features[0]:
            raise KeyError(
                "Samples must have a 'task' field. " "Make sure you're using TaskDataset wrapper."
            )

        task = features[0]["task"]

        # Verify all samples are from the same task
        for i, f in enumerate(features):
            if f.get("task") != task:
                raise ValueError(
                    f"All samples in a batch must be from the same task. "
                    f"Sample 0 is '{task}', but sample {i} is '{f.get('task')}'."
                )

        # Get the appropriate collator and process
        collator = self._get_collator(task)
        return collator(features)


# =============================================================================
# Factory Function
# =============================================================================


def get_task_collator(
    task: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    padding: str | bool = "longest",
    max_length: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> BaseCollator:
    """
    Factory function to get a collator for a specific task.

    This is useful when you need a single-task collator without the
    MultiTaskCollator wrapper.

    Args:
        task: Task name (e.g., "ner_general", "sentiment").
        tokenizer: The tokenizer for padding.
        padding: Padding strategy - "longest" (default), "max_length", or False.
        max_length: Maximum sequence length.
        pad_to_multiple_of: Pad to a multiple of this value.

    Returns:
        Collator instance appropriate for the task.

    Example:
        >>> collator = get_task_collator("ner_general", tokenizer)
        >>> batch = collator(samples)
    """
    # Handle replay tasks by stripping _replay suffix
    base_task = task
    if task.endswith("_replay"):
        base_task = task[:-7]  # Remove "_replay" suffix

    # Get collator class from mapping (use base_task for replay tasks)
    if base_task in TASK_COLLATOR_MAPPING:
        collator_cls = TASK_COLLATOR_MAPPING[base_task]
    elif task in TASK_COLLATOR_MAPPING:
        collator_cls = TASK_COLLATOR_MAPPING[task]
    else:
        logger.warning(f"Unknown task '{task}', using default SequenceClassificationCollator")
        collator_cls = SequenceClassificationCollator

    # Create and return collator instance
    return collator_cls(
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

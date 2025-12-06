"""
v3 Data Collators with Hub Token Offset Support.

This module provides data collators for the v3 architecture that handle
the special token layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

All position-based labels (NER, etc.) must be offset by +5 to account
for the hub token prefix.

Features:
    - V3ClassificationCollator: Sentiment, safety, intent, etc.
    - V3TokenClassificationCollator: NER with label offsetting
    - V3MultiTaskCollator: Unified multi-task samples
    - Automatic hub token insertion
    - Proper label alignment

Usage:
    from modeling_studio.data.collators_v3 import create_v3_collator

    # Classification
    collator = create_v3_collator(tokenizer, task_type="classification")

    # Token classification
    collator = create_v3_collator(tokenizer, task_type="token_classification")

    # Multi-task
    collator = create_v3_collator(tokenizer, task_type="multitask")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from modeling_studio.models.hub_tokens import HUB_TOKEN_IDS

logger = logging.getLogger(__name__)


# V3 Token Layout Constants
HUB_TOKEN_COUNT = 4
V3_SPECIAL_PREFIX_LEN = 5  # [CLS] + [EMO] + [MEM] + [REL] + [TASK]

# Position mapping
POSITION_CLS = 0
POSITION_EMO = 1
POSITION_MEM = 2
POSITION_REL = 3
POSITION_TASK = 4
POSITION_TEXT_START = 5


@dataclass
class V3CollatorConfig:
    """Configuration for v3 collators."""

    # Tokenizer settings
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True

    # Hub token handling
    include_hub_tokens: bool = True
    hub_token_ids: dict[str, int] | None = None  # Populated from tokenizer

    # Task-specific settings
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        """Set default hub token IDs if not provided."""
        if self.hub_token_ids is None:
            # Use the official hub token IDs from hub_tokens module
            self.hub_token_ids = HUB_TOKEN_IDS.copy()


class V3BaseCollator:
    """
    Base collator for v3 models with hub token support.

    Handles the v3 token layout:
        [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

    All position-based labels (NER, etc.) must be offset by +5 to
    account for the hub token prefix.
    """

    def __init__(
        self,
        tokenizer,
        config: V3CollatorConfig | None = None,
    ):
        """
        Initialize base collator.

        Args:
            tokenizer: Tokenizer with v3 hub tokens
            config: Collator configuration
        """
        self.tokenizer = tokenizer
        self.config = config or V3CollatorConfig()

        # Validate tokenizer has hub tokens
        self._validate_tokenizer()

    def _validate_tokenizer(self) -> None:
        """Ensure tokenizer has v3 hub tokens."""
        if self.config.hub_token_ids is None:
            return
        vocab = self.tokenizer.get_vocab()
        for token_name, expected_id in self.config.hub_token_ids.items():
            if token_name not in vocab:
                logger.warning(f"Hub token {token_name} not in tokenizer vocab")
            elif vocab[token_name] != expected_id:
                logger.warning(
                    f"Hub token {token_name} has ID {vocab[token_name]}, " f"expected {expected_id}"
                )

    def _add_hub_tokens(
        self,
        input_ids: list[int],
        attention_mask: list[int],
    ) -> tuple:
        """
        Insert hub tokens after [CLS].

        Input:  [CLS] <text> [SEP] [PAD]...
        Output: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...

        Args:
            input_ids: Original token IDs
            attention_mask: Original attention mask

        Returns:
            Tuple of (new_input_ids, new_attention_mask)
        """
        cls_id = self.tokenizer.cls_token_id
        if self.config.hub_token_ids is None:
            raise ValueError("Hub token IDs are not configured")
        hub_ids = list(self.config.hub_token_ids.values())

        # Find [CLS] position (should be 0)
        if input_ids[0] != cls_id:
            logger.warning("First token is not [CLS], inserting hub tokens at position 1")
            insert_pos = 1
        else:
            insert_pos = 1  # After [CLS]

        # Insert hub tokens
        new_input_ids = input_ids[:insert_pos] + hub_ids + input_ids[insert_pos:]

        # Extend attention mask (hub tokens are always attended)
        new_attention_mask = (
            attention_mask[:insert_pos] + [1] * HUB_TOKEN_COUNT + attention_mask[insert_pos:]
        )

        # Truncate to max_length if needed
        if len(new_input_ids) > self.config.max_length:
            new_input_ids = new_input_ids[: self.config.max_length]
            new_attention_mask = new_attention_mask[: self.config.max_length]

            # Ensure [SEP] at end
            sep_id = self.tokenizer.sep_token_id
            if new_input_ids[-1] != sep_id:
                new_input_ids[-1] = sep_id

        return new_input_ids, new_attention_mask

    def _offset_labels(
        self,
        labels: list[int],
        offset: int = V3_SPECIAL_PREFIX_LEN,
        add_sep_label: bool = True,
    ) -> list[int]:
        """
        Offset position-based labels for hub token prefix.

        For NER/token classification, label positions must shift by +5.

        Args:
            labels: Original label list (for text tokens only)
            offset: Number of positions to offset (default 5)
            add_sep_label: Whether to add ignore label for [SEP] token

        Returns:
            Offset label list with ignore tokens prepended and appended
        """
        # Prepend ignore labels for [CLS] + hub tokens
        hub_labels = [self.config.label_pad_token_id] * offset
        result = hub_labels + labels

        # Append ignore label for [SEP] token if needed
        if add_sep_label:
            result.append(self.config.label_pad_token_id)

        return result

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate features into batch."""
        raise NotImplementedError("Subclasses must implement __call__")


class V3ClassificationCollator(V3BaseCollator):
    """
    Collator for sequence classification tasks (sentiment, safety, etc.).

    No label offsetting needed - just single label per sequence.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate classification features.

        Args:
            features: List of feature dictionaries

        Returns:
            Batch dictionary with tensors
        """
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(input_ids, attention_mask)

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            if "label" in feature:
                batch["labels"].append(feature["label"])
            elif "labels" in feature:
                batch["labels"].append(feature["labels"])

        # Pad and convert to tensors
        batch = self._pad_batch(batch)
        return batch

    def _pad_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Pad batch to uniform length."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attn_mask in zip(batch["input_ids"], batch["attention_mask"], strict=True):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }

        if batch["labels"]:
            result["labels"] = torch.tensor(batch["labels"])

        return result


class V3TokenClassificationCollator(V3BaseCollator):
    """
    Collator for token classification tasks (NER, etc.).

    Labels must be offset by +5 for hub token prefix.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate token classification features.

        Args:
            features: List of feature dictionaries

        Returns:
            Batch dictionary with tensors
        """
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))
            labels = feature.get("labels", feature.get("ner_tags", []))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(input_ids, attention_mask)
                # Offset labels for hub tokens
                labels = self._offset_labels(labels)

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Pad and convert to tensors
        batch = self._pad_batch(batch)
        return batch

    def _pad_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Pad batch to uniform length."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_id = self.tokenizer.pad_token_id
        label_pad = self.config.label_pad_token_id

        for input_ids, attn_mask, labels in zip(
            batch["input_ids"], batch["attention_mask"], batch["labels"], strict=True
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)
            padded_labels.append(labels + [label_pad] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels),
        }


class V3MultiTaskCollator(V3BaseCollator):
    """
    Collator for multi-task training with multiple label types.

    Handles unified samples with multiple task labels.
    """

    def __init__(
        self,
        tokenizer,
        config: V3CollatorConfig | None = None,
        task_configs: dict[str, dict] | None = None,
    ):
        """
        Initialize multi-task collator.

        Args:
            tokenizer: Tokenizer with v3 hub tokens
            config: Collator configuration
            task_configs: Task-specific configurations
        """
        super().__init__(tokenizer, config)

        # Task-specific label handling
        self.task_configs = task_configs or {
            "sentiment": {"type": "classification", "num_labels": 3},
            "emotions": {"type": "multilabel", "num_labels": 8},
            "safety": {"type": "classification", "num_labels": 3},
            "ner": {"type": "token_classification", "num_labels": 9},
            "intent": {"type": "classification", "num_labels": 12},
            "ingress": {"type": "classification", "num_labels": 4},
        }

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate multi-task features.

        Args:
            features: List of feature dictionaries

        Returns:
            Batch dictionary with tensors for all tasks
        """
        batch = {
            "input_ids": [],
            "attention_mask": [],
        }

        # Initialize task label lists
        for task_name in self.task_configs:
            batch[f"{task_name}_labels"] = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # Add hub tokens
            if self.config.include_hub_tokens:
                input_ids, attention_mask = self._add_hub_tokens(input_ids, attention_mask)

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            # Extract task-specific labels
            tasks = feature.get("tasks", {})
            for task_name, task_config in self.task_configs.items():
                if task_name in tasks:
                    label = tasks[task_name]
                    if task_config["type"] == "token_classification":
                        label = self._offset_labels(label)
                    batch[f"{task_name}_labels"].append(label)
                else:
                    # Missing task - use ignore index
                    batch[f"{task_name}_labels"].append(None)

        # Pad and convert to tensors
        batch = self._pad_multitask_batch(batch)
        return batch

    def _pad_multitask_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Pad multi-task batch."""
        max_len = max(len(ids) for ids in batch["input_ids"])

        # Pad input_ids and attention_mask
        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attn_mask in zip(batch["input_ids"], batch["attention_mask"], strict=True):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }

        # Pad task labels
        for task_name, task_config in self.task_configs.items():
            labels = batch[f"{task_name}_labels"]

            if task_config["type"] == "token_classification":
                # Pad token-level labels
                padded_labels = []
                for label in labels:
                    if label is None:
                        padded_labels.append([self.config.label_pad_token_id] * max_len)
                    else:
                        pad_len = max_len - len(label)
                        padded_labels.append(label + [self.config.label_pad_token_id] * pad_len)
                result[f"{task_name}_labels"] = torch.tensor(padded_labels)
            else:
                # Sequence-level labels
                processed_labels = [
                    label if label is not None else self.config.label_pad_token_id
                    for label in labels
                ]
                result[f"{task_name}_labels"] = torch.tensor(processed_labels)

        return result


def create_v3_collator(
    tokenizer,
    task_type: str = "classification",
    **kwargs,
) -> V3BaseCollator:
    """
    Factory function to create appropriate v3 collator.

    Args:
        tokenizer: Tokenizer with v3 hub tokens
        task_type: "classification", "token_classification", or "multitask"
        **kwargs: Additional config options

    Returns:
        Appropriate collator instance

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        >>> collator = create_v3_collator(tokenizer, task_type="classification")
    """
    config = V3CollatorConfig(**kwargs)

    if task_type == "classification":
        return V3ClassificationCollator(tokenizer, config)
    elif task_type == "token_classification":
        return V3TokenClassificationCollator(tokenizer, config)
    elif task_type == "multitask":
        return V3MultiTaskCollator(tokenizer, config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

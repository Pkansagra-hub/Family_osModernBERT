"""
Tests for v3 Collators with Hub Token Offset Support.

This module tests the data collators that handle the v3 token layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

Tests include:
    - Hub token insertion
    - Label offsetting for token classification
    - Multi-task collation
    - Padding and truncation
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from modeling_studio.data.collators_v3 import (
    HUB_TOKEN_COUNT,
    V3_SPECIAL_PREFIX_LEN,
    V3ClassificationCollator,
    V3CollatorConfig,
    V3MultiTaskCollator,
    V3TokenClassificationCollator,
    create_v3_collator,
)
from modeling_studio.models.hub_tokens import HUB_TOKEN_IDS


@pytest.fixture
def tokenizer():
    """Load ModernBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    # Add hub tokens
    hub_tokens = ["[EMO]", "[MEM]", "[REL]", "[TASK]"]
    tokenizer.add_special_tokens({"additional_special_tokens": hub_tokens})
    return tokenizer


@pytest.fixture
def config():
    """Create default collator config."""
    return V3CollatorConfig()


class TestV3CollatorConfig:
    """Test V3CollatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = V3CollatorConfig()

        assert config.max_length == 512
        assert config.padding == "max_length"
        assert config.truncation is True
        assert config.include_hub_tokens is True
        assert config.label_pad_token_id == -100
        assert config.return_tensors == "pt"

    def test_hub_token_ids_initialized(self):
        """Test hub token IDs are initialized."""
        config = V3CollatorConfig()

        assert config.hub_token_ids is not None
        assert "[EMO]" in config.hub_token_ids
        assert "[MEM]" in config.hub_token_ids
        assert "[REL]" in config.hub_token_ids
        assert "[TASK]" in config.hub_token_ids

        # Check against the actual hub token IDs from the registry
        assert config.hub_token_ids["[EMO]"] == HUB_TOKEN_IDS["[EMO]"]
        assert config.hub_token_ids["[MEM]"] == HUB_TOKEN_IDS["[MEM]"]
        assert config.hub_token_ids["[REL]"] == HUB_TOKEN_IDS["[REL]"]
        assert config.hub_token_ids["[TASK]"] == HUB_TOKEN_IDS["[TASK]"]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = V3CollatorConfig(
            max_length=256,
            padding="longest",
            truncation=False,
            include_hub_tokens=False,
        )

        assert config.max_length == 256
        assert config.padding == "longest"
        assert config.truncation is False
        assert config.include_hub_tokens is False


class TestV3BaseCollator:
    """Test V3BaseCollator base class."""

    def test_hub_token_insertion(self, tokenizer, config):
        """Test hub tokens are inserted after [CLS]."""
        collator = V3ClassificationCollator(tokenizer, config)

        # Simple input: [CLS] hello world [SEP]
        input_ids = [tokenizer.cls_token_id, 100, 101, tokenizer.sep_token_id]
        attention_mask = [1, 1, 1, 1]

        new_input_ids, new_attention_mask = collator._add_hub_tokens(input_ids, attention_mask)

        # Expected: [CLS] [EMO] [MEM] [REL] [TASK] hello world [SEP]
        assert len(new_input_ids) == len(input_ids) + HUB_TOKEN_COUNT
        assert new_input_ids[0] == tokenizer.cls_token_id
        assert new_input_ids[1] == config.hub_token_ids["[EMO]"]
        assert new_input_ids[2] == config.hub_token_ids["[MEM]"]
        assert new_input_ids[3] == config.hub_token_ids["[REL]"]
        assert new_input_ids[4] == config.hub_token_ids["[TASK]"]
        assert new_input_ids[5] == 100  # hello
        assert new_input_ids[6] == 101  # world
        assert new_input_ids[7] == tokenizer.sep_token_id

        # Attention mask should be all 1s
        assert len(new_attention_mask) == len(new_input_ids)
        assert all(mask == 1 for mask in new_attention_mask)

    def test_hub_token_truncation(self, tokenizer, config):
        """Test hub tokens don't break truncation."""
        config.max_length = 10
        collator = V3ClassificationCollator(tokenizer, config)

        # Create input longer than max_length
        input_ids = [tokenizer.cls_token_id] + [100] * 20 + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        new_input_ids, new_attention_mask = collator._add_hub_tokens(input_ids, attention_mask)

        # Should be truncated to max_length
        assert len(new_input_ids) == config.max_length
        assert len(new_attention_mask) == config.max_length
        assert new_input_ids[-1] == tokenizer.sep_token_id  # [SEP] preserved

    def test_label_offsetting(self, tokenizer, config):
        """Test label offsetting for token classification."""
        collator = V3TokenClassificationCollator(tokenizer, config)

        # Original labels for 3 tokens
        labels = [1, 0, 2]

        offset_labels = collator._offset_labels(labels)

        # Should have 5 ignore labels (prefix) + 3 labels + 1 ignore label ([SEP])
        assert len(offset_labels) == V3_SPECIAL_PREFIX_LEN + len(labels) + 1
        assert offset_labels[0] == config.label_pad_token_id  # [CLS]
        assert offset_labels[1] == config.label_pad_token_id  # [EMO]
        assert offset_labels[2] == config.label_pad_token_id  # [MEM]
        assert offset_labels[3] == config.label_pad_token_id  # [REL]
        assert offset_labels[4] == config.label_pad_token_id  # [TASK]
        assert offset_labels[5] == 1  # First token label
        assert offset_labels[6] == 0  # Second token label
        assert offset_labels[7] == 2  # Third token label
        assert offset_labels[8] == config.label_pad_token_id  # [SEP]


class TestV3ClassificationCollator:
    """Test V3ClassificationCollator for sequence classification."""

    def test_classification_collation(self, tokenizer, config):
        """Test basic classification collation."""
        collator = V3ClassificationCollator(tokenizer, config)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, 101, tokenizer.sep_token_id],
                "attention_mask": [1, 1, 1, 1],
                "label": 1,
            },
            {
                "input_ids": [tokenizer.cls_token_id, 200, tokenizer.sep_token_id],
                "attention_mask": [1, 1, 1],
                "label": 0,
            },
        ]

        batch = collator(features)

        # Check batch keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Check shapes
        assert batch["input_ids"].shape[0] == 2  # Batch size
        assert batch["attention_mask"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

        # Check hub tokens inserted
        assert batch["input_ids"][0, 0] == tokenizer.cls_token_id
        assert batch["input_ids"][0, 1] == config.hub_token_ids["[EMO]"]
        assert batch["input_ids"][0, 2] == config.hub_token_ids["[MEM]"]
        assert batch["input_ids"][0, 3] == config.hub_token_ids["[REL]"]
        assert batch["input_ids"][0, 4] == config.hub_token_ids["[TASK]"]

        # Check labels
        assert batch["labels"][0] == 1
        assert batch["labels"][1] == 0

    def test_classification_padding(self, tokenizer, config):
        """Test padding works correctly."""
        collator = V3ClassificationCollator(tokenizer, config)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, 101, 102, tokenizer.sep_token_id],
                "attention_mask": [1, 1, 1, 1, 1],
                "label": 1,
            },
            {
                "input_ids": [tokenizer.cls_token_id, 200, tokenizer.sep_token_id],
                "attention_mask": [1, 1, 1],
                "label": 0,
            },
        ]

        batch = collator(features)

        # Second sequence should be padded to match first
        # First: [CLS] [EMO] [MEM] [REL] [TASK] 100 101 102 [SEP] = 9 tokens
        # Second: [CLS] [EMO] [MEM] [REL] [TASK] 200 [SEP] [PAD] [PAD] = 9 tokens
        assert batch["input_ids"].shape[1] == 9

        # Check padding tokens and attention mask
        assert batch["input_ids"][1, 7] == tokenizer.pad_token_id
        assert batch["input_ids"][1, 8] == tokenizer.pad_token_id
        assert batch["attention_mask"][1, 7] == 0
        assert batch["attention_mask"][1, 8] == 0


class TestV3TokenClassificationCollator:
    """Test V3TokenClassificationCollator for NER."""

    def test_token_classification_collation(self, tokenizer, config):
        """Test token classification with label offsetting."""
        collator = V3TokenClassificationCollator(tokenizer, config)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, 101, tokenizer.sep_token_id],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 0],  # Labels for 2 text tokens
            },
        ]

        batch = collator(features)

        # Check batch keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Check hub tokens inserted
        assert batch["input_ids"][0, 0] == tokenizer.cls_token_id
        assert batch["input_ids"][0, 1] == config.hub_token_ids["[EMO]"]

        # Check labels offset
        # Original: [1, 0] for 2 tokens
        # After offset: [-100, -100, -100, -100, -100, 1, 0, -100]
        #                [CLS] [EMO] [MEM] [REL] [TASK] t1 t2 [SEP]
        assert batch["labels"][0, 0] == config.label_pad_token_id
        assert batch["labels"][0, 1] == config.label_pad_token_id
        assert batch["labels"][0, 2] == config.label_pad_token_id
        assert batch["labels"][0, 3] == config.label_pad_token_id
        assert batch["labels"][0, 4] == config.label_pad_token_id
        assert batch["labels"][0, 5] == 1  # First text token
        assert batch["labels"][0, 6] == 0  # Second text token
        assert batch["labels"][0, 7] == config.label_pad_token_id  # [SEP]

    def test_token_classification_padding(self, tokenizer, config):
        """Test label padding works correctly."""
        collator = V3TokenClassificationCollator(tokenizer, config)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, 101, 102, tokenizer.sep_token_id],
                "labels": [1, 0, 2],
            },
            {
                "input_ids": [tokenizer.cls_token_id, 200, tokenizer.sep_token_id],
                "labels": [1],
            },
        ]

        batch = collator(features)

        # Both sequences should have same length
        assert batch["input_ids"].shape[1] == batch["labels"].shape[1]

        # Padded labels should be -100
        # Second sequence is shorter, so more padding
        last_valid_label_pos = 5 + 1  # 5 hub tokens + 1 text token
        assert batch["labels"][1, last_valid_label_pos + 1] == config.label_pad_token_id


class TestV3MultiTaskCollator:
    """Test V3MultiTaskCollator for unified multi-task samples."""

    def test_multitask_collation(self, tokenizer, config):
        """Test multi-task collation with multiple label types."""
        task_configs = {
            "sentiment": {"type": "classification", "num_labels": 3},
            "ner": {"type": "token_classification", "num_labels": 9},
        }

        collator = V3MultiTaskCollator(tokenizer, config, task_configs)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, 101, tokenizer.sep_token_id],
                "tasks": {
                    "sentiment": 1,
                    "ner": [1, 0],
                },
            },
        ]

        batch = collator(features)

        # Check batch keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "sentiment_labels" in batch
        assert "ner_labels" in batch

        # Check sentiment label (sequence-level)
        assert batch["sentiment_labels"][0] == 1

        # Check NER labels (token-level, offset)
        assert batch["ner_labels"][0, 0] == config.label_pad_token_id  # [CLS]
        assert batch["ner_labels"][0, 5] == 1  # First text token
        assert batch["ner_labels"][0, 6] == 0  # Second text token

    def test_multitask_missing_task(self, tokenizer, config):
        """Test handling of missing task labels."""
        task_configs = {
            "sentiment": {"type": "classification", "num_labels": 3},
            "intent": {"type": "classification", "num_labels": 12},
        }

        collator = V3MultiTaskCollator(tokenizer, config, task_configs)

        features = [
            {
                "input_ids": [tokenizer.cls_token_id, 100, tokenizer.sep_token_id],
                "tasks": {
                    "sentiment": 1,
                    # intent is missing
                },
            },
        ]

        batch = collator(features)

        # Sentiment should have valid label
        assert batch["sentiment_labels"][0] == 1

        # Intent should have ignore label
        assert batch["intent_labels"][0] == config.label_pad_token_id


class TestV3CollatorFactory:
    """Test create_v3_collator factory function."""

    def test_create_classification_collator(self, tokenizer):
        """Test creating classification collator."""
        collator = create_v3_collator(tokenizer, task_type="classification")
        assert isinstance(collator, V3ClassificationCollator)

    def test_create_token_classification_collator(self, tokenizer):
        """Test creating token classification collator."""
        collator = create_v3_collator(tokenizer, task_type="token_classification")
        assert isinstance(collator, V3TokenClassificationCollator)

    def test_create_multitask_collator(self, tokenizer):
        """Test creating multi-task collator."""
        collator = create_v3_collator(tokenizer, task_type="multitask")
        assert isinstance(collator, V3MultiTaskCollator)

    def test_invalid_task_type(self, tokenizer):
        """Test invalid task type raises error."""
        with pytest.raises(ValueError, match="Unknown task type"):
            create_v3_collator(tokenizer, task_type="invalid")

    def test_factory_with_config(self, tokenizer):
        """Test factory with custom config."""
        collator = create_v3_collator(
            tokenizer,
            task_type="classification",
            max_length=256,
            include_hub_tokens=False,
        )
        assert collator.config.max_length == 256
        assert collator.config.include_hub_tokens is False


class TestV3CollatorIntegration:
    """Integration tests for v3 collators."""

    def test_end_to_end_classification(self, tokenizer):
        """Test end-to-end classification pipeline."""
        collator = create_v3_collator(tokenizer, task_type="classification")

        # Simulate real data
        text1 = "Mom is happy today"
        text2 = "I feel sad"

        # Tokenize
        tokens1 = tokenizer(text1, add_special_tokens=False)["input_ids"]
        tokens2 = tokenizer(text2, add_special_tokens=False)["input_ids"]

        # Create features
        features = [
            {
                "input_ids": [tokenizer.cls_token_id] + tokens1 + [tokenizer.sep_token_id],
                "label": 1,  # Positive
            },
            {
                "input_ids": [tokenizer.cls_token_id] + tokens2 + [tokenizer.sep_token_id],
                "label": 0,  # Negative
            },
        ]

        # Collate
        batch = collator(features)

        # Verify structure
        assert isinstance(batch, dict)
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].dim() == 2
        assert batch["labels"].dim() == 1

        # Verify hub tokens present (use the actual token ID from hub_tokens module)
        assert batch["input_ids"][0, 1].item() == HUB_TOKEN_IDS["[EMO]"]

    def test_end_to_end_ner(self, tokenizer):
        """Test end-to-end NER pipeline."""
        collator = create_v3_collator(tokenizer, task_type="token_classification")

        # Simulate NER data
        text = "John works at Google"
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        labels = [1, 0, 0, 5]  # B-PER, O, O, B-ORG

        features = [
            {
                "input_ids": [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id],
                "labels": labels,
            },
        ]

        # Collate
        batch = collator(features)

        # Verify label alignment
        # Labels should start at position 5 (after hub tokens)
        assert batch["labels"][0, 5] == 1  # John -> B-PER
        assert batch["labels"][0, 6] == 0  # works -> O

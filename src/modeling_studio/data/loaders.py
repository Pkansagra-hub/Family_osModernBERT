"""
Dataset Loaders

This module provides functions for loading datasets from various sources
and converting them to a unified format for multi-task training.

Supported Sources:
    - HuggingFace Hub datasets
    - Local files (JSONL, CSV, Parquet)
    - Custom data directories

Loaders by Task:
    - load_ner_dataset(): NER datasets (CoNLL, OntoNotes, custom)
    - load_classification_dataset(): Classification (SST-2, etc.)
    - load_multilabel_dataset(): Multi-label (GoEmotions, Jigsaw)
    - load_nli_dataset(): NLI pairs (MNLI, SNLI, ANLI)
    - load_embedding_dataset(): Sentence pairs/triplets (STS, NLI-pairs)

Data Format Conversion:
    All loaders return HuggingFace Dataset objects with standardized columns:
    - text/tokens: Input text or tokenized input
    - labels: Task-specific labels
    - Additional fields as needed (premise, hypothesis for NLI)

Usage:
    dataset = load_ner_dataset(
        name="conll2003",
        split="train",
        label_map=CONLL_LABELS
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from modeling_studio.data.labels import (
    EMOTIONS_LABELS,
    INGRESS_LABELS,
    INTENT_LABELS,
    NER_FAMILY_LABELS,
    NER_GENERAL_LABELS,
    NLI_LABELS,
    RELATION_LABELS,
    SAFETY_FAMILYOS_LABELS,
    SAFETY_GENERIC_LABELS,
    SENTIMENT_LABELS,
    TEMPORAL_LABELS,
    LabelSchema,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Label Mapping for HuggingFace NER Datasets
# =============================================================================

# CoNLL-2003 original labels → our NER_GENERAL_LABELS mapping
CONLL2003_LABEL_MAP = {
    "O": "O",
    "B-PER": "B-PER",
    "I-PER": "I-PER",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-MISC": "B-MISC",
    "I-MISC": "I-MISC",
}

# OntoNotes 5 labels → our NER_GENERAL_LABELS mapping
ONTONOTES5_LABEL_MAP = {
    "O": "O",
    "B-PERSON": "B-PER",
    "I-PERSON": "I-PER",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "B-GPE": "B-LOC",  # Geo-Political Entity → Location
    "I-GPE": "I-LOC",
    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-DATE": "B-DATE",
    "I-DATE": "I-DATE",
    "B-TIME": "B-TIME",
    "I-TIME": "I-TIME",
    "B-EVENT": "B-EVENT",
    "I-EVENT": "I-EVENT",
    "B-PRODUCT": "B-PRODUCT",
    "I-PRODUCT": "I-PRODUCT",
    # Map other OntoNotes types to MISC
    "B-NORP": "B-MISC",  # Nationalities, religious, political groups
    "I-NORP": "I-MISC",
    "B-FAC": "B-LOC",  # Facilities → Location
    "I-FAC": "I-LOC",
    "B-WORK_OF_ART": "B-MISC",
    "I-WORK_OF_ART": "I-MISC",
    "B-LAW": "B-MISC",
    "I-LAW": "I-MISC",
    "B-LANGUAGE": "B-MISC",
    "I-LANGUAGE": "I-MISC",
    "B-MONEY": "B-MISC",
    "I-MONEY": "I-MISC",
    "B-QUANTITY": "B-MISC",
    "I-QUANTITY": "I-MISC",
    "B-ORDINAL": "B-MISC",
    "I-ORDINAL": "I-MISC",
    "B-CARDINAL": "B-MISC",
    "I-CARDINAL": "I-MISC",
    "B-PERCENT": "B-MISC",
    "I-PERCENT": "I-MISC",
}


# =============================================================================
# NER Dataset Loader
# =============================================================================


def load_ner_dataset(
    name: str,
    split: str | None = None,
    label_schema: LabelSchema = NER_GENERAL_LABELS,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """
    Load an NER dataset from HuggingFace Hub or local JSONL files.

    Supports:
        - HuggingFace datasets: conll2003, ontonotes_5, wnut_17, etc.
        - Local JSONL files with BIO-tagged data

    Args:
        name: Dataset name. Either:
            - HuggingFace dataset name (e.g., "conll2003", "ontonotes_5")
            - Path to local JSONL file (e.g., "data/ner/train.jsonl")
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all splits.
        label_schema: Label schema to apply. Defaults to NER_GENERAL_LABELS.
        data_dir: Data directory for HuggingFace datasets that require it.
        cache_dir: Directory to cache downloaded datasets.

    Returns:
        HuggingFace Dataset with standardized columns:
            - tokens: List[str] - tokenized input
            - ner_tags: List[int] - BIO tag IDs from label_schema

    Raises:
        ValueError: If dataset format is not supported.
        FileNotFoundError: If local file does not exist.

    Example:
        >>> from modeling_studio.data.loaders import load_ner_dataset
        >>> from modeling_studio.data.labels import NER_GENERAL_LABELS
        >>> ds = load_ner_dataset("conll2003", split="train")
        >>> print(ds[0]["tokens"])
        ['EU', 'rejects', 'German', 'call', ...]
        >>> print(ds[0]["ner_tags"])
        [3, 0, 7, 0, ...]
    """
    # Check if it's a local file path
    path = Path(name)
    if path.exists() and path.suffix == ".jsonl":
        return _load_ner_from_jsonl(path, split, label_schema)

    # Check if data_dir is provided and contains JSONL files
    if data_dir is not None:
        data_path = Path(data_dir)
        if data_path.exists():
            return _load_ner_from_directory(data_path, split, label_schema)

    # Load from HuggingFace Hub
    return _load_ner_from_hub(
        name=name,
        split=split,
        label_schema=label_schema,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )


def _load_ner_from_hub(
    name: str,
    split: str | None,
    label_schema: LabelSchema,
    data_dir: str | Path | None,
    cache_dir: str | Path | None,
) -> Dataset | DatasetDict:
    """Load NER dataset from HuggingFace Hub."""
    logger.info(f"Loading NER dataset '{name}' from HuggingFace Hub...")

    # Load dataset from hub with explicit kwargs
    # Note: trust_remote_code=True is needed for some legacy datasets like conll2003
    load_kwargs: dict = {"trust_remote_code": True}
    if data_dir is not None:
        load_kwargs["data_dir"] = str(data_dir)
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    dataset = load_dataset(name, split=split, **load_kwargs)  # type: ignore[arg-type]

    # Get label mapping based on dataset
    if name == "conll2003" or name.startswith("conll"):
        label_map = CONLL2003_LABEL_MAP
        if split:
            original_labels = dataset.features["ner_tags"].feature.names  # type: ignore[union-attr]
        else:
            original_labels = next(iter(dataset.values())).features["ner_tags"].feature.names  # type: ignore[union-attr]
    elif "ontonotes" in name.lower():
        label_map = ONTONOTES5_LABEL_MAP
        # OntoNotes may have different column names
        original_labels = _get_ontonotes_labels(dataset, split)  # type: ignore[arg-type]
    else:
        # For other datasets, try to infer label mapping
        label_map = None
        original_labels = None
        logger.warning(
            f"Unknown dataset '{name}'. Attempting to use labels as-is. "
            "You may need to provide a custom label mapping."
        )

    # Apply label mapping
    if split:
        return _remap_ner_labels(dataset, label_schema, original_labels, label_map)  # type: ignore[arg-type]
    else:
        return DatasetDict(
            {
                s: _remap_ner_labels(d, label_schema, original_labels, label_map)  # type: ignore[arg-type]
                for s, d in dataset.items()  # type: ignore[union-attr]
            }
        )


def _get_ontonotes_labels(dataset: Dataset | DatasetDict, split: str | None) -> list[str]:
    """Extract label names from OntoNotes dataset."""
    if split:
        features = dataset.features
    else:
        features = next(iter(dataset.values())).features

    # OntoNotes might have different column names
    for col in ["ner_tags", "named_entities", "entities"]:
        if col in features:
            return features[col].feature.names
    raise ValueError("Could not find NER tag column in OntoNotes dataset")


def _remap_ner_labels(
    dataset: Dataset,
    label_schema: LabelSchema,
    original_labels: list[str] | None,
    label_map: dict[str, str] | None,
) -> Dataset:
    """Remap NER labels from source dataset to target schema."""

    def remap_tags(example):
        """Remap a single example's NER tags."""
        original_tags = example["ner_tags"]

        if original_labels is not None and label_map is not None:
            # Map through label names
            new_tags = []
            for tag_id in original_tags:
                orig_label = original_labels[tag_id]
                mapped_label = label_map.get(orig_label, "O")  # Default to O if not found
                new_tag_id = label_schema.encode(mapped_label)
                new_tags.append(new_tag_id)
        else:
            # Assume tags are already in correct format or use as-is
            new_tags = [min(tag, label_schema.num_labels - 1) for tag in original_tags]

        return {"ner_tags": new_tags}

    # Ensure we have the required columns
    if "tokens" not in dataset.column_names:
        # Some datasets use "words" instead of "tokens"
        if "words" in dataset.column_names:
            dataset = dataset.rename_column("words", "tokens")
        else:
            raise ValueError(
                f"Dataset must have 'tokens' or 'words' column. " f"Found: {dataset.column_names}"
            )

    # Remap labels
    dataset = dataset.map(remap_tags, desc="Remapping NER labels")

    # Keep only required columns
    columns_to_keep = ["tokens", "ner_tags"]
    columns_to_remove = [c for c in dataset.column_names if c not in columns_to_keep]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def _load_ner_from_jsonl(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
) -> Dataset | DatasetDict:
    """
    Load NER dataset from local JSONL file.

    Expected JSONL format:
        {"tokens": ["John", "lives", "in", "NYC"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}
        {"tokens": [...], "ner_tags": [...]}

    Tags can be either string labels or integer IDs.
    """
    logger.info(f"Loading NER dataset from local file: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not data:
        raise ValueError(f"No valid data found in {path}")

    # Convert string labels to IDs if needed
    processed_data = []
    for item in data:
        tokens = item.get("tokens", item.get("words", []))
        tags = item.get("ner_tags", item.get("tags", []))

        # Convert string tags to IDs
        if tags and isinstance(tags[0], str):
            tags = [label_schema.encode(tag) for tag in tags]

        processed_data.append(
            {
                "tokens": tokens,
                "ner_tags": tags,
            }
        )

    dataset = Dataset.from_list(processed_data)

    if split:
        return dataset
    else:
        # Return as DatasetDict with single split
        return DatasetDict({"train": dataset})


def _load_ner_from_directory(
    data_dir: Path,
    split: str | None,
    label_schema: LabelSchema,
) -> Dataset | DatasetDict:
    """
    Load NER dataset from a directory containing split files.

    Expected structure:
        data_dir/
            train.jsonl
            validation.jsonl (or dev.jsonl or valid.jsonl)
            test.jsonl
    """
    logger.info(f"Loading NER dataset from directory: {data_dir}")

    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": [
            "validation.jsonl",
            "valid.jsonl",
            "dev.jsonl",
            "validation.json",
            "valid.json",
            "dev.json",
        ],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_dir / filename
            if filepath.exists():
                datasets[split_name] = _load_ner_from_jsonl(filepath, split_name, label_schema)
                break

    if not datasets:
        raise FileNotFoundError(f"No valid split files found in {data_dir}")

    if split:
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
        return datasets[split]

    return DatasetDict(datasets)


# =============================================================================
# Classification Dataset Loader
# =============================================================================

# Label mappings for common classification datasets to SENTIMENT_LABELS (5-class)
# SENTIMENT_LABELS: very_negative=0, negative=1, neutral=2, positive=3, very_positive=4

# SST-2: binary (0=negative, 1=positive) -> 5-class
SST2_LABEL_MAP: dict[int, int] = {
    0: 1,  # negative -> negative (1)
    1: 3,  # positive -> positive (3)
}

# IMDB: binary (0=negative, 1=positive) -> 5-class
IMDB_LABEL_MAP: dict[int, int] = {
    0: 1,  # negative -> negative (1)
    1: 3,  # positive -> positive (3)
}

# Amazon Reviews: 5-star (1-5) -> 5-class (shifted by 1)
AMAZON_LABEL_MAP: dict[int, int] = {
    1: 0,  # 1-star -> very_negative (0)
    2: 1,  # 2-star -> negative (1)
    3: 2,  # 3-star -> neutral (2)
    4: 3,  # 4-star -> positive (3)
    5: 4,  # 5-star -> very_positive (4)
}

# Yelp Polarity: binary (0=negative, 1=positive) -> 5-class
YELP_POLARITY_LABEL_MAP: dict[int, int] = {
    0: 1,  # negative -> negative (1)
    1: 3,  # positive -> positive (3)
}

# Yelp Full: 5-star (0-4 or 1-5) -> 5-class
YELP_FULL_LABEL_MAP: dict[int, int] = {
    0: 0,  # 1-star -> very_negative (0)
    1: 1,  # 2-star -> negative (1)
    2: 2,  # 3-star -> neutral (2)
    3: 3,  # 4-star -> positive (3)
    4: 4,  # 5-star -> very_positive (4)
}


def load_classification_dataset(
    name: str,
    split: str | None = None,
    label_schema: LabelSchema = SENTIMENT_LABELS,
    text_column: str | None = None,
    label_column: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """
    Load a text classification dataset from HuggingFace Hub or local files.

    Supports:
        - HuggingFace datasets: sst2, imdb, amazon_polarity, yelp_polarity, etc.
        - Local CSV/JSONL files with text and label columns

    Args:
        name: Dataset name. Either:
            - HuggingFace dataset name (e.g., "sst2", "imdb")
            - Path to local CSV or JSONL file (e.g., "data/sentiment/train.csv")
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all splits.
        label_schema: Label schema to apply. Defaults to SENTIMENT_LABELS (5-class).
        text_column: Name of text column in local files (auto-detected if None).
        label_column: Name of label column in local files (auto-detected if None).
        cache_dir: Directory to cache downloaded datasets.

    Returns:
        HuggingFace Dataset with standardized columns:
            - text: str - input text
            - label: int - class ID from label_schema

    Raises:
        ValueError: If dataset format is not supported.
        FileNotFoundError: If local file does not exist.

    Example:
        >>> from modeling_studio.data.loaders import load_classification_dataset
        >>> from modeling_studio.data.labels import SENTIMENT_LABELS
        >>> ds = load_classification_dataset("sst2", split="train")
        >>> print(ds[0]["text"])
        'hide new secretions from the parental units'
        >>> print(ds[0]["label"])
        1  # negative
    """
    # Check if it's a local file path
    path = Path(name)
    if path.exists():
        if path.suffix in (".csv", ".tsv"):
            return _load_classification_from_csv(
                path, split, label_schema, text_column, label_column
            )
        elif path.suffix in (".jsonl", ".json"):
            return _load_classification_from_jsonl(
                path, split, label_schema, text_column, label_column
            )
        elif path.is_dir():
            return _load_classification_from_directory(
                path, split, label_schema, text_column, label_column
            )

    # Load from HuggingFace Hub
    return _load_classification_from_hub(
        name=name,
        split=split,
        label_schema=label_schema,
        cache_dir=cache_dir,
    )


def _load_classification_from_hub(
    name: str,
    split: str | None,
    label_schema: LabelSchema,
    cache_dir: str | Path | None,
) -> Dataset | DatasetDict:
    """Load classification dataset from HuggingFace Hub."""
    logger.info(f"Loading classification dataset '{name}' from HuggingFace Hub...")

    # Handle GLUE datasets (sst2 is part of GLUE)
    load_kwargs: dict = {"trust_remote_code": True}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    # Determine dataset path and config
    # Normalize name to handle full HuggingFace paths
    name_lower = name.lower().replace("-", "_")
    name_base = name_lower.split("/")[-1]  # Get base name from full path

    if name_base in ("sst2", "sst_2") or name_lower in ("stanfordnlp/sst2", "glue/sst2"):
        dataset = load_dataset("stanfordnlp/sst2", split=split, **load_kwargs)
        text_col = "sentence"
        label_map = SST2_LABEL_MAP
    elif name_base == "imdb":
        dataset = load_dataset("imdb", split=split, **load_kwargs)
        text_col = "text"
        label_map = IMDB_LABEL_MAP
    elif name.lower() in ("amazon_polarity", "amazon-polarity"):
        dataset = load_dataset("amazon_polarity", split=split, **load_kwargs)
        text_col = "content"
        label_map = {0: 1, 1: 3}  # binary: neg=1, pos=3
    elif name.lower() in ("yelp_polarity", "yelp-polarity"):
        dataset = load_dataset("yelp_polarity", split=split, **load_kwargs)
        text_col = "text"
        label_map = YELP_POLARITY_LABEL_MAP
    elif name.lower() in ("yelp_review_full", "yelp-review-full"):
        dataset = load_dataset("yelp_review_full", split=split, **load_kwargs)
        text_col = "text"
        label_map = YELP_FULL_LABEL_MAP
    else:
        # Try to load as generic dataset
        dataset = load_dataset(name, split=split, **load_kwargs)
        text_col = None
        label_map = None
        logger.warning(
            f"Unknown dataset '{name}'. Attempting to auto-detect columns. "
            "You may need to specify text_column and label_column."
        )

    # Apply standardization
    if split:
        return _standardize_classification_dataset(
            dataset, label_schema, text_col, label_map  # type: ignore[arg-type]
        )
    else:
        return DatasetDict(
            {
                s: _standardize_classification_dataset(d, label_schema, text_col, label_map)  # type: ignore[arg-type]
                for s, d in dataset.items()  # type: ignore[union-attr]
            }
        )


def _standardize_classification_dataset(
    dataset: Dataset,
    label_schema: LabelSchema,
    text_column: str | None,
    label_map: dict[int, int] | None,
) -> Dataset:
    """
    Standardize a classification dataset to have 'text' and 'label' columns.

    Remaps labels from source dataset to target label_schema.
    """
    # Find text column if not specified
    if text_column is None:
        text_candidates = ["text", "sentence", "content", "review", "document", "input"]
        for col in text_candidates:
            if col in dataset.column_names:
                text_column = col
                break
        if text_column is None:
            raise ValueError(
                f"Could not auto-detect text column. " f"Available columns: {dataset.column_names}"
            )

    # Find label column
    orig_label_column = "label"
    if orig_label_column not in dataset.column_names:
        label_candidates = ["label", "labels", "sentiment", "class", "target"]
        for col in label_candidates:
            if col in dataset.column_names:
                orig_label_column = col
                break

    # Rename original label column to avoid ClassLabel feature type conflict
    if orig_label_column in dataset.column_names:
        dataset = dataset.rename_column(orig_label_column, "_orig_label")

    # Capture for closure
    text_col_final = text_column

    def standardize(example):
        """Standardize a single example."""
        text = example[text_col_final]

        # Get original label
        original_label = example.get("_orig_label", 0)

        # Apply label mapping if available
        if label_map is not None:
            new_label = label_map.get(original_label, original_label)
        else:
            # Clamp to valid range
            new_label = max(0, min(original_label, label_schema.num_labels - 1))

        return {"text": text, "label": new_label}

    # Apply standardization - use remove_columns to handle feature type conflict
    dataset = dataset.map(
        standardize,
        remove_columns=dataset.column_names,  # Remove all old columns
    )

    return dataset


def _load_classification_from_csv(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
    label_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load classification dataset from local CSV file.

    Expected format:
        text,label
        "This movie was great!",positive
        "Terrible experience.",negative
    """
    logger.info(f"Loading classification dataset from CSV: {path}")

    # Determine delimiter
    delimiter = "\t" if path.suffix == ".tsv" else ","

    csv_dataset = load_dataset(
        "csv",
        data_files=str(path),
        split="train",  # CSV loads as single split
        delimiter=delimiter,
    )

    # Cast to Dataset for type checker (load_dataset with split returns Dataset)
    assert isinstance(csv_dataset, Dataset), "Expected Dataset when split is specified"

    # Auto-detect columns if not specified
    column_names = csv_dataset.column_names
    if text_column is None:
        text_candidates = ["text", "sentence", "content", "review", "document", "input"]
        for col in text_candidates:
            if col in column_names:
                text_column = col
                break
        if text_column is None:
            # Use first non-label column
            for col in column_names:
                if col not in ("label", "labels", "target", "class"):
                    text_column = col
                    break

    if label_column is None:
        label_candidates = ["label", "labels", "sentiment", "class", "target"]
        for col in label_candidates:
            if col in column_names:
                label_column = col
                break

    if text_column is None or label_column is None:
        raise ValueError(
            f"Could not detect text/label columns. "
            f"Available: {column_names}. "
            f"Please specify text_column and label_column."
        )

    # Capture variables for closure
    text_col_final = text_column
    label_col_final = label_column

    def process_csv_example(example):
        """Process a single CSV example."""
        text = example[text_col_final]
        label = example[label_col_final]

        # Convert string label to int if needed
        if isinstance(label, str):
            label = label_schema.encode(label)
        else:
            # Ensure label is in valid range
            label = max(0, min(int(label), label_schema.num_labels - 1))

        return {"text": text, "label": label}

    csv_dataset = csv_dataset.map(process_csv_example)

    # Keep only required columns
    columns_to_keep = ["text", "label"]
    columns_to_remove = [c for c in csv_dataset.column_names if c not in columns_to_keep]
    if columns_to_remove:
        csv_dataset = csv_dataset.remove_columns(columns_to_remove)

    if split:
        return csv_dataset
    else:
        return DatasetDict({"train": csv_dataset})


def _load_classification_from_jsonl(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
    label_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load classification dataset from local JSONL file.

    Expected format:
        {"text": "This movie was great!", "label": "positive"}
        {"text": "Terrible experience.", "label": "negative"}

    Or with integer labels:
        {"text": "This movie was great!", "label": 3}
    """
    logger.info(f"Loading classification dataset from JSONL: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not data:
        raise ValueError(f"No valid data found in {path}")

    # Auto-detect columns from first item
    first_item = data[0]

    if text_column is None:
        text_candidates = ["text", "sentence", "content", "review", "document", "input"]
        for col in text_candidates:
            if col in first_item:
                text_column = col
                break

    if label_column is None:
        label_candidates = ["label", "labels", "sentiment", "class", "target"]
        for col in label_candidates:
            if col in first_item:
                label_column = col
                break

    if text_column is None or label_column is None:
        raise ValueError(
            f"Could not detect text/label columns. "
            f"Found keys: {list(first_item.keys())}. "
            f"Please specify text_column and label_column."
        )

    # Process data
    processed_data = []
    for item in data:
        text = item.get(text_column, "")
        label = item.get(label_column)

        # Convert string label to int if needed
        if isinstance(label, str):
            label = label_schema.encode(label)
        else:
            # Ensure label is in valid range
            label = max(0, min(int(label), label_schema.num_labels - 1))

        processed_data.append({"text": text, "label": label})

    dataset = Dataset.from_list(processed_data)

    if split:
        return dataset
    else:
        return DatasetDict({"train": dataset})


def _load_classification_from_directory(
    data_dir: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
    label_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load classification dataset from a directory containing split files.

    Expected structure:
        data_dir/
            train.csv (or train.jsonl)
            validation.csv (or validation.jsonl)
            test.csv (or test.jsonl)
    """
    logger.info(f"Loading classification dataset from directory: {data_dir}")

    split_files = {
        "train": ["train.csv", "train.tsv", "train.jsonl", "train.json"],
        "validation": [
            "validation.csv",
            "valid.csv",
            "dev.csv",
            "validation.tsv",
            "valid.tsv",
            "dev.tsv",
            "validation.jsonl",
            "valid.jsonl",
            "dev.jsonl",
            "validation.json",
            "valid.json",
            "dev.json",
        ],
        "test": ["test.csv", "test.tsv", "test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_dir / filename
            if filepath.exists():
                if filepath.suffix in (".csv", ".tsv"):
                    ds = _load_classification_from_csv(
                        filepath, split_name, label_schema, text_column, label_column
                    )
                else:
                    ds = _load_classification_from_jsonl(
                        filepath, split_name, label_schema, text_column, label_column
                    )
                # Extract the actual dataset if it's a DatasetDict
                if isinstance(ds, DatasetDict):
                    datasets[split_name] = ds["train"]
                else:
                    datasets[split_name] = ds
                break

    if not datasets:
        raise FileNotFoundError(f"No valid split files found in {data_dir}")

    if split:
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
        return datasets[split]

    return DatasetDict(datasets)


# =============================================================================
# Multi-Label Dataset Loader
# =============================================================================

# GoEmotions original labels (28 emotions)
GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Jigsaw toxicity labels (6 types)
JIGSAW_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_multilabel_dataset(
    name: str,
    split: str | None = None,
    label_schema: LabelSchema = EMOTIONS_LABELS,
    text_column: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """
    Load a multi-label classification dataset from HuggingFace Hub or local files.

    Supports:
        - HuggingFace datasets: go_emotions, jigsaw_toxicity_pred
        - Local CSV/JSONL files with multi-hot labels

    Args:
        name: Dataset name. Either:
            - HuggingFace dataset name (e.g., "go_emotions", "jigsaw_toxicity_pred")
            - Path to local CSV or JSONL file
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all splits.
        label_schema: Label schema to apply. Defaults to EMOTIONS_LABELS (32-class).
        text_column: Name of text column in local files (auto-detected if None).
        cache_dir: Directory to cache downloaded datasets.

    Returns:
        HuggingFace Dataset with standardized columns:
            - text: str - input text
            - labels: List[int] - multi-hot vector of size num_labels

    Raises:
        ValueError: If dataset format is not supported.
        FileNotFoundError: If local file does not exist.

    Example:
        >>> from modeling_studio.data.loaders import load_multilabel_dataset
        >>> from modeling_studio.data.labels import EMOTIONS_LABELS
        >>> ds = load_multilabel_dataset("go_emotions", split="train")
        >>> print(ds[0]["text"])
        'This is great!'
        >>> print(ds[0]["labels"])  # multi-hot vector
        [0, 1, 0, 0, ..., 0]  # 32 elements
    """
    # Check if it's a local file path
    path = Path(name)
    if path.exists():
        if path.suffix in (".csv", ".tsv"):
            return _load_multilabel_from_csv(path, split, label_schema, text_column)
        elif path.suffix in (".jsonl", ".json"):
            return _load_multilabel_from_jsonl(path, split, label_schema, text_column)
        elif path.is_dir():
            return _load_multilabel_from_directory(path, split, label_schema, text_column)

    # Load from HuggingFace Hub
    return _load_multilabel_from_hub(
        name=name,
        split=split,
        label_schema=label_schema,
        cache_dir=cache_dir,
    )


def _load_multilabel_from_hub(
    name: str,
    split: str | None,
    label_schema: LabelSchema,
    cache_dir: str | Path | None,
) -> Dataset | DatasetDict:
    """Load multi-label dataset from HuggingFace Hub."""
    logger.info(f"Loading multi-label dataset '{name}' from HuggingFace Hub...")

    load_kwargs: dict = {"trust_remote_code": True}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    # Determine dataset-specific handling
    # Normalize name to handle full HuggingFace paths
    name_lower = name.lower().replace("-", "_")
    name_base = name_lower.split("/")[-1]  # Get base name from full path

    if name_base in ("go_emotions", "goemotions") or "go_emotions" in name_lower:
        dataset = load_dataset(
            "google-research-datasets/go_emotions", "simplified", split=split, **load_kwargs
        )
        text_col = "text"
        original_labels = GO_EMOTIONS_LABELS
        label_col = "labels"  # List of label indices
    elif name_base in ("civil_comments",) or "civil_comments" in name_lower:
        # Civil Comments toxicity dataset
        dataset = load_dataset("google/civil_comments", split=split, **load_kwargs)
        text_col = "text"
        original_labels = ["toxicity"]  # Binary toxicity score
        label_col = "toxicity"
    elif name_base in ("jigsaw_toxicity_pred", "jigsaw_toxicity", "jigsaw"):
        dataset = load_dataset("jigsaw_toxicity_pred", split=split, **load_kwargs)
        text_col = "comment_text"
        original_labels = JIGSAW_LABELS
        label_col = None  # Labels are individual columns
    else:
        # Try to load as generic dataset
        dataset = load_dataset(name, split=split, **load_kwargs)
        text_col = None
        original_labels = None
        label_col = None
        logger.warning(f"Unknown multi-label dataset '{name}'. Attempting to auto-detect format.")

    # Apply standardization
    if split:
        return _standardize_multilabel_dataset(
            dataset, label_schema, text_col, original_labels, label_col  # type: ignore[arg-type]
        )
    else:
        return DatasetDict(
            {
                s: _standardize_multilabel_dataset(d, label_schema, text_col, original_labels, label_col)  # type: ignore[arg-type]
                for s, d in dataset.items()  # type: ignore[union-attr]
            }
        )


def _standardize_multilabel_dataset(
    dataset: Dataset,
    label_schema: LabelSchema,
    text_column: str | None,
    original_labels: list[str] | None,
    label_column: str | None,
) -> Dataset:
    """
    Standardize a multi-label dataset to have 'text' and 'labels' columns.

    Converts labels to multi-hot encoding matching label_schema.
    """
    # Find text column if not specified
    if text_column is None:
        text_candidates = ["text", "comment_text", "sentence", "content", "input"]
        for col in text_candidates:
            if col in dataset.column_names:
                text_column = col
                break
        if text_column is None:
            raise ValueError(
                f"Could not auto-detect text column. " f"Available columns: {dataset.column_names}"
            )

    # Capture for closure
    text_col_final = text_column
    num_labels = label_schema.num_labels

    # Build mapping from original labels to schema indices
    label_index_map: dict[int, int] = {}
    if original_labels is not None:
        for orig_idx, label_name in enumerate(original_labels):
            if label_name in label_schema.label2id:
                label_index_map[orig_idx] = label_schema.label2id[label_name]

    # Check if this is Jigsaw format (individual columns for each label)
    is_jigsaw_format = label_column is None and any(
        col in dataset.column_names for col in JIGSAW_LABELS
    )

    def standardize(example):
        """Standardize a single example to multi-hot format."""
        text = example[text_col_final]

        # Initialize multi-hot vector
        multi_hot = [0] * num_labels

        if is_jigsaw_format:
            # Jigsaw: each label is a separate column with 0/1 value
            for label_name in JIGSAW_LABELS:
                if label_name in example and example[label_name]:
                    # Check if value is truthy (handles float scores > 0.5)
                    val = example[label_name]
                    is_positive = val >= 0.5 if isinstance(val, float) else bool(val)
                    if is_positive and label_name in label_schema.label2id:
                        multi_hot[label_schema.label2id[label_name]] = 1
        elif label_column and label_column in example:
            # GoEmotions style: list of label indices
            label_indices = example[label_column]
            if isinstance(label_indices, list):
                for orig_idx in label_indices:
                    if orig_idx in label_index_map:
                        multi_hot[label_index_map[orig_idx]] = 1
                    elif orig_idx < num_labels:
                        # Direct index if no mapping
                        multi_hot[orig_idx] = 1
        else:
            # Try to find labels column
            for col in ["labels", "label"]:
                if col in example:
                    val = example[col]
                    if isinstance(val, list):
                        for idx in val:
                            if isinstance(idx, int) and idx < num_labels:
                                multi_hot[idx] = 1
                            elif isinstance(idx, str) and idx in label_schema.label2id:
                                multi_hot[label_schema.label2id[idx]] = 1
                    break

        return {"text": text, "labels": multi_hot}

    # Apply standardization - remove all old columns
    dataset = dataset.map(
        standardize,
        remove_columns=dataset.column_names,
    )

    return dataset


def _load_multilabel_from_csv(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load multi-label dataset from local CSV file.

    Expected formats:
        Format 1 - Multi-hot columns:
            text,toxic,obscene,insult
            "Bad text",1,1,0

        Format 2 - Label list:
            text,labels
            "I love this!","[1, 5, 16]"
    """
    logger.info(f"Loading multi-label dataset from CSV: {path}")

    delimiter = "\t" if path.suffix == ".tsv" else ","

    csv_dataset = load_dataset(
        "csv",
        data_files=str(path),
        split="train",
        delimiter=delimiter,
    )

    assert isinstance(csv_dataset, Dataset), "Expected Dataset when split is specified"

    # Detect format and process
    column_names = csv_dataset.column_names

    # Auto-detect text column
    if text_column is None:
        text_candidates = ["text", "sentence", "content", "comment_text", "input"]
        for col in text_candidates:
            if col in column_names:
                text_column = col
                break
        if text_column is None:
            # Use first column that's not a known label
            for col in column_names:
                if col not in label_schema.label2id:
                    text_column = col
                    break

    if text_column is None:
        raise ValueError(f"Could not detect text column. Available: {column_names}")

    text_col_final = text_column
    num_labels = label_schema.num_labels

    # Check if columns match label names (multi-hot format)
    label_columns = [col for col in column_names if col in label_schema.label2id]

    def process_csv_example(example):
        """Process a single CSV example."""
        text = example[text_col_final]
        multi_hot = [0] * num_labels

        if label_columns:
            # Multi-hot column format
            for col in label_columns:
                if example.get(col):
                    val = example[col]
                    is_positive = val >= 0.5 if isinstance(val, float) else bool(val)
                    if is_positive:
                        multi_hot[label_schema.label2id[col]] = 1
        elif "labels" in example:
            # Labels column with list
            labels = example["labels"]
            if isinstance(labels, str):
                # Parse string representation of list
                import ast

                try:
                    labels = ast.literal_eval(labels)
                except (ValueError, SyntaxError):
                    labels = []
            if isinstance(labels, list):
                for lbl in labels:
                    if isinstance(lbl, int) and lbl < num_labels:
                        multi_hot[lbl] = 1
                    elif isinstance(lbl, str) and lbl in label_schema.label2id:
                        multi_hot[label_schema.label2id[lbl]] = 1

        return {"text": text, "labels": multi_hot}

    csv_dataset = csv_dataset.map(
        process_csv_example,
        remove_columns=csv_dataset.column_names,
    )

    if split:
        return csv_dataset
    else:
        return DatasetDict({"train": csv_dataset})


def _load_multilabel_from_jsonl(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load multi-label dataset from local JSONL file.

    Expected format:
        {"text": "I love this!", "labels": ["joy", "love"]}
        {"text": "This is bad", "labels": [3, 12]}  # integer indices
    """
    logger.info(f"Loading multi-label dataset from JSONL: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not data:
        raise ValueError(f"No valid data found in {path}")

    # Auto-detect text column from first item
    first_item = data[0]
    if text_column is None:
        text_candidates = ["text", "sentence", "content", "input"]
        for col in text_candidates:
            if col in first_item:
                text_column = col
                break

    if text_column is None:
        raise ValueError(f"Could not detect text column. Found keys: {list(first_item.keys())}")

    text_col_final = text_column
    num_labels = label_schema.num_labels

    # Process data
    processed_data = []
    for item in data:
        text = item.get(text_col_final, "")
        multi_hot = [0] * num_labels

        labels = item.get("labels", [])
        if isinstance(labels, list):
            for lbl in labels:
                if isinstance(lbl, int) and lbl < num_labels:
                    multi_hot[lbl] = 1
                elif isinstance(lbl, str) and lbl in label_schema.label2id:
                    multi_hot[label_schema.label2id[lbl]] = 1

        processed_data.append({"text": text, "labels": multi_hot})

    dataset = Dataset.from_list(processed_data)

    if split:
        return dataset
    else:
        return DatasetDict({"train": dataset})


def _load_multilabel_from_directory(
    data_dir: Path,
    split: str | None,
    label_schema: LabelSchema,
    text_column: str | None,
) -> Dataset | DatasetDict:
    """
    Load multi-label dataset from a directory containing split files.

    Expected structure:
        data_dir/
            train.csv (or train.jsonl)
            validation.csv (or validation.jsonl)
            test.csv (or test.jsonl)
    """
    logger.info(f"Loading multi-label dataset from directory: {data_dir}")

    split_files = {
        "train": ["train.csv", "train.tsv", "train.jsonl", "train.json"],
        "validation": [
            "validation.csv",
            "valid.csv",
            "dev.csv",
            "validation.tsv",
            "valid.tsv",
            "dev.tsv",
            "validation.jsonl",
            "valid.jsonl",
            "dev.jsonl",
            "validation.json",
            "valid.json",
            "dev.json",
        ],
        "test": ["test.csv", "test.tsv", "test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_dir / filename
            if filepath.exists():
                if filepath.suffix in (".csv", ".tsv"):
                    ds = _load_multilabel_from_csv(filepath, split_name, label_schema, text_column)
                else:
                    ds = _load_multilabel_from_jsonl(
                        filepath, split_name, label_schema, text_column
                    )
                # Extract the actual dataset if it's a DatasetDict
                if isinstance(ds, DatasetDict):
                    datasets[split_name] = ds["train"]
                else:
                    datasets[split_name] = ds
                break

    if not datasets:
        raise FileNotFoundError(f"No valid split files found in {data_dir}")

    if split:
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
        return datasets[split]

    return DatasetDict(datasets)


# =============================================================================
# NLI Dataset Loader
# =============================================================================

# Standard NLI label mappings
# Most datasets use: 0=entailment, 1=neutral, 2=contradiction
# Some use -1 for unlabeled examples (we filter these out)


def load_nli_dataset(
    name: str,
    split: str | None = None,
    label_schema: LabelSchema = NLI_LABELS,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """
    Load a Natural Language Inference (NLI) dataset from HuggingFace Hub or local files.

    Supports:
        - HuggingFace datasets: multi_nli, snli, anli
        - Local JSONL files with premise/hypothesis pairs

    Args:
        name: Dataset name. Either:
            - HuggingFace dataset name (e.g., "multi_nli", "snli", "anli")
            - Path to local JSONL file
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all splits.
            Note: For multi_nli, validation splits are "validation_matched" and
            "validation_mismatched". Use "validation" to get "validation_matched".
        label_schema: Label schema to apply. Defaults to NLI_LABELS (3-class).
        cache_dir: Directory to cache downloaded datasets.

    Returns:
        HuggingFace Dataset with standardized columns:
            - premise: str - the premise text
            - hypothesis: str - the hypothesis text
            - label: int - 0 (entailment), 1 (neutral), 2 (contradiction)

    Raises:
        ValueError: If dataset format is not supported.
        FileNotFoundError: If local file does not exist.

    Example:
        >>> from modeling_studio.data.loaders import load_nli_dataset
        >>> from modeling_studio.data.labels import NLI_LABELS
        >>> ds = load_nli_dataset("multi_nli", split="train")
        >>> print(ds[0]["premise"])
        'Conceptually cream skimming has two basic dimensions...'
        >>> print(ds[0]["hypothesis"])
        'Product and geography are what cream skimming has...'
        >>> print(NLI_LABELS.decode(ds[0]["label"]))
        'neutral'
    """
    # Check if it's a local file path
    path = Path(name)
    if path.exists():
        if path.suffix in (".jsonl", ".json"):
            return _load_nli_from_jsonl(path, split, label_schema)
        elif path.is_dir():
            return _load_nli_from_directory(path, split, label_schema)

    # Load from HuggingFace Hub
    return _load_nli_from_hub(
        name=name,
        split=split,
        label_schema=label_schema,
        cache_dir=cache_dir,
    )


def _load_nli_from_hub(
    name: str,
    split: str | None,
    label_schema: LabelSchema,
    cache_dir: str | Path | None,
) -> Dataset | DatasetDict:
    """Load NLI dataset from HuggingFace Hub."""
    logger.info(f"Loading NLI dataset '{name}' from HuggingFace Hub...")

    load_kwargs: dict = {"trust_remote_code": True}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    # Handle dataset-specific loading
    # Normalize name to handle full HuggingFace paths
    name_lower = name.lower().replace("-", "_")
    name_base = name_lower.split("/")[-1]  # Get base name from full path

    if (
        name_base in ("multi_nli", "multinli", "mnli")
        or "multi_nli" in name_lower
        or name_lower == "nli"
    ):
        # MultiNLI has special validation split names
        if split == "validation":
            split = "validation_matched"  # Default to matched
        dataset = load_dataset("nyu-mll/multi_nli", split=split, **load_kwargs)
        premise_col = "premise"
        hypothesis_col = "hypothesis"
    elif name_base == "snli" or "snli" in name_lower:
        dataset = load_dataset("stanfordnlp/snli", split=split, **load_kwargs)
        premise_col = "premise"
        hypothesis_col = "hypothesis"
    elif name_base == "anli":
        # ANLI has rounds (r1, r2, r3)
        dataset = load_dataset("facebook/anli", split=split, **load_kwargs)
        premise_col = "premise"
        hypothesis_col = "hypothesis"
    elif name.lower() in ("xnli", "x-nli"):
        # Cross-lingual NLI - English subset
        dataset = load_dataset("facebook/xnli", "en", split=split, **load_kwargs)
        premise_col = "premise"
        hypothesis_col = "hypothesis"
    else:
        # Try to load as generic dataset
        dataset = load_dataset(name, split=split, **load_kwargs)
        premise_col = None
        hypothesis_col = None
        logger.warning(f"Unknown NLI dataset '{name}'. Attempting to auto-detect columns.")

    # Apply standardization
    if split:
        return _standardize_nli_dataset(
            dataset, label_schema, premise_col, hypothesis_col  # type: ignore[arg-type]
        )
    else:
        return DatasetDict(
            {
                s: _standardize_nli_dataset(d, label_schema, premise_col, hypothesis_col)  # type: ignore[arg-type]
                for s, d in dataset.items()  # type: ignore[union-attr]
            }
        )


def _standardize_nli_dataset(
    dataset: Dataset,
    label_schema: LabelSchema,
    premise_column: str | None,
    hypothesis_column: str | None,
) -> Dataset:
    """
    Standardize an NLI dataset to have 'premise', 'hypothesis', 'label' columns.

    Filters out examples with invalid labels (e.g., -1 for unlabeled).
    """
    # Find premise column if not specified
    if premise_column is None:
        premise_candidates = ["premise", "sentence1", "text1", "s1"]
        for col in premise_candidates:
            if col in dataset.column_names:
                premise_column = col
                break
        if premise_column is None:
            raise ValueError(
                f"Could not auto-detect premise column. "
                f"Available columns: {dataset.column_names}"
            )

    # Find hypothesis column if not specified
    if hypothesis_column is None:
        hypothesis_candidates = ["hypothesis", "sentence2", "text2", "s2"]
        for col in hypothesis_candidates:
            if col in dataset.column_names:
                hypothesis_column = col
                break
        if hypothesis_column is None:
            raise ValueError(
                f"Could not auto-detect hypothesis column. "
                f"Available columns: {dataset.column_names}"
            )

    # Capture for closure
    premise_col_final = premise_column
    hypothesis_col_final = hypothesis_column

    def standardize(example):
        """Standardize a single example."""
        premise = example[premise_col_final]
        hypothesis = example[hypothesis_col_final]
        label = example.get("label", example.get("gold_label", -1))

        # Handle string labels
        if isinstance(label, str):
            label_lower = label.lower()
            if label_lower in label_schema.label2id:
                label = label_schema.label2id[label_lower]
            elif label_lower == "e" or label_lower.startswith("entail"):
                label = 0
            elif label_lower == "n" or label_lower.startswith("neutral"):
                label = 1
            elif label_lower == "c" or label_lower.startswith("contrad"):
                label = 2
            else:
                label = -1  # Mark as invalid

        # Ensure label is valid
        if label < 0 or label >= label_schema.num_labels:
            label = -1  # Will be filtered out

        return {"premise": premise, "hypothesis": hypothesis, "label": label}

    # Apply standardization
    dataset = dataset.map(
        standardize,
        remove_columns=dataset.column_names,
    )

    # Filter out invalid labels (label == -1)
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["label"] >= 0)
    filtered_len = len(dataset)

    if filtered_len < original_len:
        logger.info(
            f"Filtered {original_len - filtered_len} examples with invalid labels "
            f"({filtered_len}/{original_len} remaining)"
        )

    return dataset


def _load_nli_from_jsonl(
    path: Path,
    split: str | None,
    label_schema: LabelSchema,
) -> Dataset | DatasetDict:
    """
    Load NLI dataset from local JSONL file.

    Expected format:
        {"premise": "The sky is blue.", "hypothesis": "It is daytime.", "label": "entailment"}
        {"premise": "...", "hypothesis": "...", "label": 0}
    """
    logger.info(f"Loading NLI dataset from JSONL: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not data:
        raise ValueError(f"No valid data found in {path}")

    # Auto-detect columns from first item
    first_item = data[0]

    # Find premise column
    premise_col = None
    for col in ["premise", "sentence1", "text1", "s1"]:
        if col in first_item:
            premise_col = col
            break

    # Find hypothesis column
    hypothesis_col = None
    for col in ["hypothesis", "sentence2", "text2", "s2"]:
        if col in first_item:
            hypothesis_col = col
            break

    if premise_col is None or hypothesis_col is None:
        raise ValueError(
            f"Could not detect premise/hypothesis columns. "
            f"Found keys: {list(first_item.keys())}"
        )

    # Process data
    processed_data = []
    for item in data:
        premise = item.get(premise_col, "")
        hypothesis = item.get(hypothesis_col, "")
        label = item.get("label", item.get("gold_label", -1))

        # Handle string labels
        if isinstance(label, str):
            label_lower = label.lower()
            if label_lower in label_schema.label2id:
                label = label_schema.label2id[label_lower]
            elif label_lower == "e" or label_lower.startswith("entail"):
                label = 0
            elif label_lower == "n" or label_lower.startswith("neutral"):
                label = 1
            elif label_lower == "c" or label_lower.startswith("contrad"):
                label = 2
            else:
                continue  # Skip invalid

        # Skip invalid labels
        if label < 0 or label >= label_schema.num_labels:
            continue

        processed_data.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
            }
        )

    if not processed_data:
        raise ValueError(f"No valid NLI examples found in {path}")

    dataset = Dataset.from_list(processed_data)

    if split:
        return dataset
    else:
        return DatasetDict({"train": dataset})


def _load_nli_from_directory(
    data_dir: Path,
    split: str | None,
    label_schema: LabelSchema,
) -> Dataset | DatasetDict:
    """
    Load NLI dataset from a directory containing split files.

    Expected structure:
        data_dir/
            train.jsonl
            validation.jsonl (or dev.jsonl)
            test.jsonl
    """
    logger.info(f"Loading NLI dataset from directory: {data_dir}")

    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": [
            "validation.jsonl",
            "valid.jsonl",
            "dev.jsonl",
            "validation.json",
            "valid.json",
            "dev.json",
        ],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_dir / filename
            if filepath.exists():
                ds = _load_nli_from_jsonl(filepath, split_name, label_schema)
                if isinstance(ds, DatasetDict):
                    datasets[split_name] = ds["train"]
                else:
                    datasets[split_name] = ds
                break

    if not datasets:
        raise FileNotFoundError(f"No valid split files found in {data_dir}")

    if split:
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
        return datasets[split]

    return DatasetDict(datasets)


# =============================================================================
# Embedding Dataset Loader
# =============================================================================


def load_embedding_dataset(
    name: str,
    split: str | None = None,
    format: str = "pairs",
    config_name: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """
    Load an embedding/similarity dataset from HuggingFace Hub or local files.

    Supports two formats:
        1. Pairs format (default): sentence1, sentence2, score
           - For contrastive learning with similarity scores
           - STS-B, STS benchmarks

        2. Triplets format: anchor, positive, negative
           - For triplet loss training
           - Custom triplet datasets

    Args:
        name: Dataset name. Either:
            - HuggingFace dataset name (e.g., "stsb", "sts-b")
            - Path to local CSV or JSONL file
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all splits.
        format: Output format - "pairs" or "triplets"
        cache_dir: Directory to cache downloaded datasets.

    Returns:
        HuggingFace Dataset with standardized columns:
            Pairs format:
                - sentence1: str - first sentence
                - sentence2: str - second sentence
                - score: float - similarity score (0.0 to 1.0, normalized)

            Triplets format:
                - anchor: str - anchor sentence
                - positive: str - positive (similar) sentence
                - negative: str - negative (dissimilar) sentence

    Raises:
        ValueError: If dataset format is not supported.
        FileNotFoundError: If local file does not exist.

    Example:
        >>> from modeling_studio.data.loaders import load_embedding_dataset
        >>> ds = load_embedding_dataset("stsb", split="train")
        >>> print(ds[0]["sentence1"])
        'A plane is taking off.'
        >>> print(ds[0]["score"])
        0.8  # normalized 0-1
    """
    # Check if it's a local file path
    path = Path(name)
    if path.exists():
        if path.suffix in (".csv", ".tsv"):
            return _load_embedding_from_csv(path, split, format)
        elif path.suffix in (".jsonl", ".json"):
            return _load_embedding_from_jsonl(path, split, format)
        elif path.is_dir():
            return _load_embedding_from_directory(path, split, format)

    # Load from HuggingFace Hub
    return _load_embedding_from_hub(
        name=name,
        split=split,
        format=format,
        config_name=config_name,
        cache_dir=cache_dir,
    )


def _load_embedding_from_hub(
    name: str,
    split: str | None,
    format: str,
    config_name: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    """Load embedding dataset from HuggingFace Hub."""
    logger.info(f"Loading embedding dataset '{name}' from HuggingFace Hub...")

    load_kwargs: dict = {"trust_remote_code": True}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    # Determine dataset-specific handling
    # Normalize name to handle full HuggingFace paths
    name_lower = name.lower().replace("-", "_")
    name_base = name_lower.split("/")[-1]  # Get base name from full path

    if name_base in ("stsb", "sts_b") or "stsb" in name_lower:
        # STS-B - prefer sentence-transformers version
        dataset = load_dataset("sentence-transformers/stsb", split=split, **load_kwargs)
        sentence1_col = "sentence1"
        sentence2_col = "sentence2"
        score_col = "score"  # sentence-transformers uses 'score'
        score_scale = 5.0  # STS-B scores are 0-5, normalize to 0-1
    elif name_base in ("stsb_multi_mt",) or "stsb_multi_mt" in name_lower:
        # Multilingual STS-B
        dataset = load_dataset("stsb_multi_mt", "en", split=split, **load_kwargs)
        sentence1_col = "sentence1"
        sentence2_col = "sentence2"
        score_col = "similarity_score"
        score_scale = 5.0
    elif name.lower() in ("sick", "sick-dataset"):
        # SICK dataset
        dataset = load_dataset("sick", split=split, **load_kwargs)
        sentence1_col = "sentence_A"
        sentence2_col = "sentence_B"
        score_col = "relatedness_score"
        score_scale = 5.0
    elif name.lower() in ("paws", "paws-x"):
        # PAWS paraphrase dataset (binary: 0 or 1)
        dataset = load_dataset("paws", "labeled_final", split=split, **load_kwargs)
        sentence1_col = "sentence1"
        sentence2_col = "sentence2"
        score_col = "label"
        score_scale = 1.0  # Already 0-1
    elif name.lower() in ("mrpc", "glue/mrpc"):
        # Microsoft Research Paraphrase Corpus (binary)
        dataset = load_dataset("glue", "mrpc", split=split, **load_kwargs)
        sentence1_col = "sentence1"
        sentence2_col = "sentence2"
        score_col = "label"
        score_scale = 1.0
    elif name_base in ("qqp",) or "qqp" in name_lower:
        # Quora Question Pairs (binary)
        dataset = load_dataset("glue", "qqp", split=split, **load_kwargs)
        sentence1_col = "question1"
        sentence2_col = "question2"
        score_col = "label"
        score_scale = 1.0
    elif name_base in ("all_nli", "allnli") or "all_nli" in name_lower or "all-nli" in name.lower():
        # sentence-transformers/all-nli - large NLI pairs dataset
        # Requires config: 'pair', 'pair-class', 'pair-score', or 'triplet'
        cfg = config_name or "pair-score"
        dataset = load_dataset("sentence-transformers/all-nli", cfg, split=split, **load_kwargs)
        # pair-score config uses sentence1, sentence2, score columns
        sentence1_col = "sentence1"
        sentence2_col = "sentence2"
        score_col = "score" if cfg == "pair-score" else None
        score_scale = 1.0  # Already normalized
    else:
        # Try to load as generic dataset with optional config
        if config_name:
            dataset = load_dataset(name, config_name, split=split, **load_kwargs)
        else:
            dataset = load_dataset(name, split=split, **load_kwargs)
        sentence1_col = None
        sentence2_col = None
        score_col = None
        score_scale = 1.0
        logger.warning(f"Unknown embedding dataset '{name}'. Attempting to auto-detect columns.")

    # Apply standardization
    if split:
        return _standardize_embedding_dataset(
            dataset, format, sentence1_col, sentence2_col, score_col, score_scale  # type: ignore[arg-type]
        )
    else:
        return DatasetDict(
            {
                s: _standardize_embedding_dataset(d, format, sentence1_col, sentence2_col, score_col, score_scale)  # type: ignore[arg-type]
                for s, d in dataset.items()  # type: ignore[union-attr]
            }
        )


def _standardize_embedding_dataset(
    dataset: Dataset,
    format: str,
    sentence1_column: str | None,
    sentence2_column: str | None,
    score_column: str | None,
    score_scale: float,
) -> Dataset:
    """
    Standardize an embedding dataset to pairs or triplets format.

    Pairs format: sentence1, sentence2, score
    Triplets format: anchor, positive, negative (generated from pairs)
    """
    # Auto-detect columns if not specified
    if sentence1_column is None:
        candidates = ["sentence1", "sentence_A", "text1", "question1", "s1", "anchor"]
        for col in candidates:
            if col in dataset.column_names:
                sentence1_column = col
                break
        if sentence1_column is None:
            raise ValueError(
                f"Could not auto-detect sentence1 column. "
                f"Available columns: {dataset.column_names}"
            )

    if sentence2_column is None:
        candidates = ["sentence2", "sentence_B", "text2", "question2", "s2", "positive"]
        for col in candidates:
            if col in dataset.column_names:
                sentence2_column = col
                break
        if sentence2_column is None:
            raise ValueError(
                f"Could not auto-detect sentence2 column. "
                f"Available columns: {dataset.column_names}"
            )

    if score_column is None:
        candidates = ["score", "label", "similarity_score", "relatedness_score", "similarity"]
        for col in candidates:
            if col in dataset.column_names:
                score_column = col
                break
        if score_column is None:
            # Default to 1.0 for all pairs (assume positive pairs)
            score_column = None
            logger.warning("No score column found, assuming all pairs have similarity 1.0")

    # Capture for closure
    s1_col = sentence1_column
    s2_col = sentence2_column
    sc_col = score_column

    if format == "pairs":

        def standardize_pairs(example):
            """Standardize to pairs format."""
            sentence1 = example[s1_col]
            sentence2 = example[s2_col]

            if sc_col is not None:
                score = float(example[sc_col]) / score_scale
                # Clamp to 0-1 range
                score = max(0.0, min(1.0, score))
            else:
                score = 1.0

            return {"sentence1": sentence1, "sentence2": sentence2, "score": score}

        dataset = dataset.map(
            standardize_pairs,
            remove_columns=dataset.column_names,
        )

    elif format == "triplets":
        # For triplets, we need to generate negatives
        # This is a simplified approach - in practice you might want
        # more sophisticated negative sampling
        def standardize_triplets(example, idx):
            """Generate triplets from pairs by using random negatives."""
            anchor = example[s1_col]
            positive = example[s2_col]

            # For now, use a simple approach: negative is the sentence2
            # from a different example (this should be improved for production)
            # Here we just mark it for later processing
            return {
                "anchor": anchor,
                "positive": positive,
                "negative": "",  # Placeholder - should be filled by collator or preprocessing
            }

        dataset = dataset.map(
            standardize_triplets,
            with_indices=True,
            remove_columns=dataset.column_names,
        )

        logger.warning(
            "Triplet format: 'negative' column is empty and should be filled "
            "during training (e.g., in-batch negatives or hard negative mining)."
        )

    else:
        raise ValueError(f"Unknown format '{format}'. Use 'pairs' or 'triplets'.")

    return dataset


def _load_embedding_from_csv(
    path: Path,
    split: str | None,
    format: str,
) -> Dataset | DatasetDict:
    """
    Load embedding dataset from local CSV file.

    Expected formats:
        Pairs: sentence1,sentence2,score
        Triplets: anchor,positive,negative
    """
    logger.info(f"Loading embedding dataset from CSV: {path}")

    delimiter = "\t" if path.suffix == ".tsv" else ","

    csv_dataset = load_dataset(
        "csv",
        data_files=str(path),
        split="train",
        delimiter=delimiter,
    )

    assert isinstance(csv_dataset, Dataset), "Expected Dataset when split is specified"

    column_names = csv_dataset.column_names

    if format == "pairs":
        # Find columns
        s1_col = None
        for col in ["sentence1", "text1", "s1"]:
            if col in column_names:
                s1_col = col
                break

        s2_col = None
        for col in ["sentence2", "text2", "s2"]:
            if col in column_names:
                s2_col = col
                break

        sc_col = None
        for col in ["score", "similarity", "label"]:
            if col in column_names:
                sc_col = col
                break

        if s1_col is None or s2_col is None:
            raise ValueError(f"Could not detect sentence columns. Available: {column_names}")

        # Capture for closure
        s1_final, s2_final, sc_final = s1_col, s2_col, sc_col

        def process_pairs(example):
            score = float(example[sc_final]) if sc_final else 1.0
            return {
                "sentence1": example[s1_final],
                "sentence2": example[s2_final],
                "score": max(0.0, min(1.0, score)),
            }

        csv_dataset = csv_dataset.map(
            process_pairs,
            remove_columns=csv_dataset.column_names,
        )

    elif format == "triplets":
        # Find columns
        anchor_col = None
        for col in ["anchor", "sentence1", "text1"]:
            if col in column_names:
                anchor_col = col
                break

        pos_col = None
        for col in ["positive", "sentence2", "text2"]:
            if col in column_names:
                pos_col = col
                break

        neg_col = None
        for col in ["negative", "sentence3", "text3"]:
            if col in column_names:
                neg_col = col
                break

        if anchor_col is None or pos_col is None:
            raise ValueError(f"Could not detect anchor/positive columns. Available: {column_names}")

        # Capture for closure
        a_final, p_final, n_final = anchor_col, pos_col, neg_col

        def process_triplets(example):
            return {
                "anchor": example[a_final],
                "positive": example[p_final],
                "negative": example[n_final] if n_final else "",
            }

        csv_dataset = csv_dataset.map(
            process_triplets,
            remove_columns=csv_dataset.column_names,
        )

    if split:
        return csv_dataset
    else:
        return DatasetDict({"train": csv_dataset})


def _load_embedding_from_jsonl(
    path: Path,
    split: str | None,
    format: str,
) -> Dataset | DatasetDict:
    """
    Load embedding dataset from local JSONL file.

    Expected formats:
        Pairs: {"sentence1": "...", "sentence2": "...", "score": 0.8}
        Triplets: {"anchor": "...", "positive": "...", "negative": "..."}
    """
    logger.info(f"Loading embedding dataset from JSONL: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not data:
        raise ValueError(f"No valid data found in {path}")

    first_item = data[0]

    if format == "pairs":
        # Find columns
        s1_col = None
        for col in ["sentence1", "text1", "s1"]:
            if col in first_item:
                s1_col = col
                break

        s2_col = None
        for col in ["sentence2", "text2", "s2"]:
            if col in first_item:
                s2_col = col
                break

        sc_col = None
        for col in ["score", "similarity", "label"]:
            if col in first_item:
                sc_col = col
                break

        if s1_col is None or s2_col is None:
            raise ValueError(
                f"Could not detect sentence columns. Found keys: {list(first_item.keys())}"
            )

        processed_data = []
        for item in data:
            score = float(item.get(sc_col, 1.0)) if sc_col else 1.0
            processed_data.append(
                {
                    "sentence1": item.get(s1_col, ""),
                    "sentence2": item.get(s2_col, ""),
                    "score": max(0.0, min(1.0, score)),
                }
            )

    elif format == "triplets":
        # Find columns
        anchor_col = None
        for col in ["anchor", "sentence1", "text1"]:
            if col in first_item:
                anchor_col = col
                break

        pos_col = None
        for col in ["positive", "sentence2", "text2"]:
            if col in first_item:
                pos_col = col
                break

        neg_col = None
        for col in ["negative", "sentence3", "text3"]:
            if col in first_item:
                neg_col = col
                break

        if anchor_col is None or pos_col is None:
            raise ValueError(
                f"Could not detect anchor/positive columns. Found keys: {list(first_item.keys())}"
            )

        processed_data = []
        for item in data:
            processed_data.append(
                {
                    "anchor": item.get(anchor_col, ""),
                    "positive": item.get(pos_col, ""),
                    "negative": item.get(neg_col, "") if neg_col else "",
                }
            )

    else:
        raise ValueError(f"Unknown format '{format}'. Use 'pairs' or 'triplets'.")

    dataset = Dataset.from_list(processed_data)

    if split:
        return dataset
    else:
        return DatasetDict({"train": dataset})


def _load_embedding_from_directory(
    data_dir: Path,
    split: str | None,
    format: str,
) -> Dataset | DatasetDict:
    """
    Load embedding dataset from a directory containing split files.

    Expected structure:
        data_dir/
            train.csv (or train.jsonl)
            validation.csv (or validation.jsonl)
            test.csv (or test.jsonl)
    """
    logger.info(f"Loading embedding dataset from directory: {data_dir}")

    split_files = {
        "train": ["train.csv", "train.tsv", "train.jsonl", "train.json"],
        "validation": [
            "validation.csv",
            "valid.csv",
            "dev.csv",
            "validation.tsv",
            "valid.tsv",
            "dev.tsv",
            "validation.jsonl",
            "valid.jsonl",
            "dev.jsonl",
            "validation.json",
            "valid.json",
            "dev.json",
        ],
        "test": ["test.csv", "test.tsv", "test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_dir / filename
            if filepath.exists():
                if filepath.suffix in (".csv", ".tsv"):
                    ds = _load_embedding_from_csv(filepath, split_name, format)
                else:
                    ds = _load_embedding_from_jsonl(filepath, split_name, format)
                if isinstance(ds, DatasetDict):
                    datasets[split_name] = ds["train"]
                else:
                    datasets[split_name] = ds
                break

    if not datasets:
        raise FileNotFoundError(f"No valid split files found in {data_dir}")

    if split:
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
        return datasets[split]

    return DatasetDict(datasets)


# =============================================================================
# FamilyOS Dataset Loaders
# =============================================================================


def load_familyos_ner(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/ner_family",
    validate_bio: bool = True,
) -> Dataset | DatasetDict:
    """
    Load FamilyOS family-specific NER dataset.

    This loader handles family-related named entity recognition with 21 BIO tags
    including new v2 entity types: TRADITION, MILESTONE, HEIRLOOM.

    Entity Types (10 types, 21 BIO tags):
        - PERSON: Named individuals (John Smith, Sarah)
        - KINSHIP: Family relationship terms (mom, dad, didi, nana, bhai)
        - NICKNAME: Family nicknames (Panda, Bunny, Sweetie)
        - PET: Pet names (Max, Whiskers, our dog)
        - HOME_LOC: Locations within home (kitchen, Emma's room, backyard)
        - FAMILY_EVENT: Family occasions (birthday party, anniversary)
        - ROUTINE: Regular activities (school run, dinner time)
        - TRADITION: Recurring family rituals (Sunday brunch, movie night) [NEW v2]
        - MILESTONE: Life events to remember (first steps, graduation) [NEW v2]
        - HEIRLOOM: Sentimental objects (grandma's necklace, dad's watch) [NEW v2]

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS NER data directory.
            Default: "data/familyos/ner_family"
        validate_bio: Whether to validate BIO tag consistency.
            Default: True

    Returns:
        HuggingFace Dataset with columns:
            - tokens: list[str] - tokenized input text
            - ner_tags: list[int] - BIO tag IDs from NER_FAMILY_LABELS

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If BIO tag validation fails.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_ner
        >>> from modeling_studio.data.labels import NER_FAMILY_LABELS
        >>> ds = load_familyos_ner(split="train")
        >>> print(ds[0]["tokens"])
        ['Panda', 'took', 'her', 'first', 'steps', 'in', 'the', 'kitchen']
        >>> tags = [NER_FAMILY_LABELS.decode(t) for t in ds[0]["ner_tags"]]
        >>> print(tags)
        ['B-NICKNAME', 'O', 'O', 'B-MILESTONE', 'I-MILESTONE', 'O', 'O', 'B-HOME_LOC']
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS NER data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS NER dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_ner_jsonl(filepath, validate_bio)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid NER split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_ner_jsonl(
    filepath: Path,
    validate_bio: bool = True,
) -> Dataset:
    """
    Load FamilyOS NER data from a JSONL file.

    Expected JSONL format:
        {"tokens": ["Panda", "is", "in", "the", "kitchen"], "ner_tags": [5, 0, 0, 0, 9]}

    Where ner_tags are integer IDs from NER_FAMILY_LABELS schema.
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            if "tokens" not in item or "ner_tags" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'tokens' or 'ner_tags'")
                continue

            tokens = item["tokens"]
            ner_tags = item["ner_tags"]

            # Validate lengths match
            if len(tokens) != len(ner_tags):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"tokens ({len(tokens)}) and ner_tags ({len(ner_tags)}) length mismatch"
                )
                continue

            # Convert string labels to IDs if needed
            if ner_tags and isinstance(ner_tags[0], str):
                try:
                    ner_tags = [NER_FAMILY_LABELS.encode(tag) for tag in ner_tags]
                except KeyError as e:
                    logger.warning(f"Skipping line {line_num}: unknown NER tag {e}")
                    continue

            # Validate tag IDs are in range
            if any(tag < 0 or tag >= NER_FAMILY_LABELS.num_labels for tag in ner_tags):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"ner_tags contain invalid IDs (must be 0-{NER_FAMILY_LABELS.num_labels - 1})"
                )
                continue

            # Validate BIO consistency if requested
            if validate_bio:
                bio_error = _validate_bio_tags(ner_tags, NER_FAMILY_LABELS)
                if bio_error:
                    logger.warning(f"Line {line_num}: {bio_error}")
                    # Continue anyway, but log the warning

            data.append({"tokens": tokens, "ner_tags": ner_tags})

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


def _validate_bio_tags(
    tags: list[int],
    label_schema: LabelSchema,
) -> str | None:
    """
    Validate BIO tag consistency.

    Rules:
        1. I-tags must follow B-tags or I-tags of the same entity type
        2. No orphan I-tags (I-X without preceding B-X)

    Returns:
        Error message if validation fails, None otherwise.
    """
    prev_tag_name = "O"

    for i, tag_id in enumerate(tags):
        tag_name = label_schema.decode(tag_id)

        if tag_name.startswith("I-"):
            # Get entity type (e.g., "PERSON" from "I-PERSON")
            entity_type = tag_name[2:]
            expected_b_tag = f"B-{entity_type}"
            expected_i_tag = f"I-{entity_type}"

            # I-tag must follow B-tag or I-tag of same type
            if prev_tag_name != expected_b_tag and prev_tag_name != expected_i_tag:
                return (
                    f"Invalid BIO sequence at position {i}: "
                    f"'{tag_name}' follows '{prev_tag_name}' "
                    f"(expected '{expected_b_tag}' or '{expected_i_tag}')"
                )

        prev_tag_name = tag_name

    return None


def load_familyos_ingress(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/ingress",
) -> Dataset | DatasetDict:
    """
    Load FamilyOS ingress classification dataset.

    This loader handles domain/topic classification for incoming text
    with 12 domain labels (v2 enhanced).

    Domain Labels (12 total):
        - DIARY (0): Personal reflections, journaling
        - TASK (1): To-dos, reminders, action items
        - HEALTH (2): Medical, wellness, fitness
        - FINANCE (3): Money, bills, budgets
        - RELATIONSHIP (4): Family dynamics, social
        - WORK (5): Job, career, professional
        - META (6): System commands, queries about FamilyOS
        - MEMORY (7): Recalling past events [NEW v2]
        - PLANNING (8): Future events [NEW v2]
        - CELEBRATION (9): Birthdays, achievements, milestones [NEW v2]
        - CONCERN (10): Worries, anxieties [NEW v2]
        - GRATITUDE (11): Appreciation expressions [NEW v2]

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS ingress data directory.
            Default: "data/familyos/ingress"

    Returns:
        HuggingFace Dataset with columns:
            - text: str - input text
            - label: int - domain label ID from INGRESS_LABELS

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If no valid samples found.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_ingress
        >>> from modeling_studio.data.labels import INGRESS_LABELS
        >>> ds = load_familyos_ingress(split="train")
        >>> print(ds[0]["text"])
        'Had a lovely dinner with the family tonight at Olive Garden'
        >>> print(INGRESS_LABELS.decode(ds[0]["label"]))
        'DIARY'
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS ingress data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS ingress dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_ingress_jsonl(filepath)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid ingress split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_ingress_jsonl(filepath: Path) -> Dataset:
    """
    Load FamilyOS ingress data from a JSONL file.

    Expected JSONL format:
        {"text": "Had a lovely dinner...", "label": 0}

    Where label is an integer ID from INGRESS_LABELS schema,
    or a string label name that will be converted.
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            if "text" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'text' field")
                continue

            if "label" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'label' field")
                continue

            text = item["text"]
            label = item["label"]

            # Convert string label to ID if needed
            if isinstance(label, str):
                try:
                    label = INGRESS_LABELS.encode(label.upper())
                except KeyError:
                    logger.warning(
                        f"Skipping line {line_num}: unknown label '{label}'. "
                        f"Valid labels: {list(INGRESS_LABELS.label2id.keys())}"
                    )
                    continue

            # Validate label ID is in range
            if not (0 <= label < INGRESS_LABELS.num_labels):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"label {label} out of range (0-{INGRESS_LABELS.num_labels - 1})"
                )
                continue

            data.append({"text": text, "label": label})

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


def load_familyos_safety(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/safety",
) -> Dataset | DatasetDict:
    """
    Load FamilyOS safety classification dataset.

    This loader handles safety policy band classification with 4 bands:
        - GREEN (0): Safe, routine content
        - AMBER (1): Needs attention, mild concern
        - RED (2): Serious concern, escalate to K1
        - CRISIS (3): Immediate intervention needed

    The safety classification is critical for FamilyOS to detect and
    appropriately handle sensitive content related to mental health,
    safety concerns, and crisis situations.

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS safety data directory.
            Default: "data/familyos/safety"

    Returns:
        HuggingFace Dataset with columns:
            - text: str - input text
            - label: int - safety band ID from SAFETY_FAMILYOS_LABELS
            - subcategories: list[str] (optional) - specific concern types

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If no valid samples found.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_safety
        >>> from modeling_studio.data.labels import SAFETY_FAMILYOS_LABELS
        >>> ds = load_familyos_safety(split="train")
        >>> print(ds[0]["text"])
        'Had a great day at the park with the kids today'
        >>> print(SAFETY_FAMILYOS_LABELS.decode(ds[0]["label"]))
        'GREEN'
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS safety data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS safety dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
        "calibration": ["calibration.jsonl"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_safety_jsonl(filepath)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid safety split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_safety_jsonl(filepath: Path) -> Dataset:
    """
    Load FamilyOS safety data from a JSONL file.

    Expected JSONL format:
        {"text": "Had a great day...", "label": 0, "subcategories": []}
        {"text": "Feeling stressed...", "label": 1, "subcategories": ["stress"]}

    Where label is:
        - An integer ID (0=GREEN, 1=AMBER, 2=RED, 3=CRISIS)
        - Or a string label name that will be converted
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            if "text" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'text' field")
                continue

            if "label" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'label' field")
                continue

            text = item["text"]
            label = item["label"]

            # Convert string label to ID if needed
            if isinstance(label, str):
                try:
                    label = SAFETY_FAMILYOS_LABELS.encode(label.upper())
                except KeyError:
                    logger.warning(
                        f"Skipping line {line_num}: unknown label '{label}'. "
                        f"Valid labels: {list(SAFETY_FAMILYOS_LABELS.label2id.keys())}"
                    )
                    continue

            # Validate label ID is in range
            if not (0 <= label < SAFETY_FAMILYOS_LABELS.num_labels):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"label {label} out of range (0-{SAFETY_FAMILYOS_LABELS.num_labels - 1})"
                )
                continue

            # Get optional subcategories
            subcategories = item.get("subcategories", [])
            if not isinstance(subcategories, list):
                subcategories = [subcategories] if subcategories else []

            data.append(
                {
                    "text": text,
                    "label": label,
                    "subcategories": subcategories,
                }
            )

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


def load_familyos_relations(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/relations",
) -> Dataset | DatasetDict:
    """
    Load FamilyOS relation extraction dataset.

    This loader handles family relationship extraction with 15 relation types:
        - Family: parent_of, child_of, spouse_of, sibling_of, grandparent_of,
                  grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of
        - Non-family: friend_of, colleague_of
        - Other: pet_of, lives_at, owns, no_relation

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS relations data directory.
            Default: "data/familyos/relations"

    Returns:
        HuggingFace Dataset with columns:
            - text: str - input text containing both entities
            - entity1: str - the subject entity
            - entity2: str - the object entity
            - relation: int - relation ID from RELATION_LABELS

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If no valid samples found.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_relations
        >>> from modeling_studio.data.labels import RELATION_LABELS
        >>> ds = load_familyos_relations(split="train")
        >>> sample = ds[0]
        >>> print(f"{sample['entity1']} --{RELATION_LABELS.decode(sample['relation'])}--> {sample['entity2']}")
        'Mom --parent_of--> Panda'
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS relations data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS relations dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_relations_jsonl(filepath)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid relations split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_relations_jsonl(filepath: Path) -> Dataset:
    """
    Load FamilyOS relations data from a JSONL file.

    Expected JSONL format:
        {"text": "Mom took Panda to the park", "entity1": "Mom", "entity2": "Panda", "relation": 1}

    Where relation is:
        - An integer ID (0-14) from RELATION_LABELS
        - Or a string label name that will be converted
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            required_fields = ["text", "entity1", "entity2", "relation"]
            missing = [f for f in required_fields if f not in item]
            if missing:
                logger.warning(f"Skipping line {line_num}: missing fields {missing}")
                continue

            text = item["text"]
            entity1 = item["entity1"]
            entity2 = item["entity2"]
            relation = item["relation"]

            # Convert string label to ID if needed
            if isinstance(relation, str):
                try:
                    relation = RELATION_LABELS.encode(relation.lower())
                except KeyError:
                    logger.warning(
                        f"Skipping line {line_num}: unknown relation '{relation}'. "
                        f"Valid relations: {list(RELATION_LABELS.label2id.keys())}"
                    )
                    continue

            # Validate relation ID is in range
            if not (0 <= relation < RELATION_LABELS.num_labels):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"relation {relation} out of range (0-{RELATION_LABELS.num_labels - 1})"
                )
                continue

            data.append(
                {
                    "text": text,
                    "entity1": entity1,
                    "entity2": entity2,
                    "relation": relation,
                }
            )

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


def load_familyos_intents(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/intents",
) -> Dataset | DatasetDict:
    """
    Load FamilyOS intent classification dataset.

    This loader handles user intent classification with 8 intent types:
        - log_memory (0): Recording memories/events
        - query_memory (1): Asking about past events
        - set_reminder (2): Creating reminders/tasks
        - express_feeling (3): Sharing emotions
        - seek_advice (4): Asking for guidance
        - share_news (5): Announcing updates
        - reflect (6): Contemplation/musing
        - other (7): Catch-all

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS intents data directory.
            Default: "data/familyos/intents"

    Returns:
        HuggingFace Dataset with columns:
            - text: str - input text
            - label: int - intent ID from INTENT_LABELS

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If no valid samples found.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_intents
        >>> from modeling_studio.data.labels import INTENT_LABELS
        >>> ds = load_familyos_intents(split="train")
        >>> print(ds[0]["text"])
        'Had a lovely dinner with the family tonight'
        >>> print(INTENT_LABELS.decode(ds[0]["label"]))
        'log_memory'
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS intents data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS intents dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_intents_jsonl(filepath)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid intents split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_intents_jsonl(filepath: Path) -> Dataset:
    """
    Load FamilyOS intents data from a JSONL file.

    Expected JSONL format:
        {"text": "Had dinner with family tonight", "label": 0}

    Where label is:
        - An integer ID (0-7) from INTENT_LABELS
        - Or a string label name that will be converted
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            if "text" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'text' field")
                continue

            if "label" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'label' field")
                continue

            text = item["text"]
            label = item["label"]

            # Convert string label to ID if needed
            if isinstance(label, str):
                try:
                    label = INTENT_LABELS.encode(label.lower())
                except KeyError:
                    logger.warning(
                        f"Skipping line {line_num}: unknown intent '{label}'. "
                        f"Valid intents: {list(INTENT_LABELS.label2id.keys())}"
                    )
                    continue

            # Validate label ID is in range
            if not (0 <= label < INTENT_LABELS.num_labels):
                logger.warning(
                    f"Skipping line {line_num}: "
                    f"label {label} out of range (0-{INTENT_LABELS.num_labels - 1})"
                )
                continue

            data.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


def load_familyos_temporal(
    split: str | None = None,
    data_dir: str | Path = "data/familyos/temporal",
) -> Dataset | DatasetDict:
    """
    Load FamilyOS temporal expression extraction dataset.

    This loader handles temporal expression extraction (token classification)
    with 13 BIO tags covering 6 temporal entity types:
        - DATE_ABS: Absolute dates (January 15, 2024)
        - DATE_REL: Relative dates (yesterday, last week)
        - TIME: Time expressions (3pm, morning)
        - DURATION: Duration spans (for 2 hours, all day)
        - FREQUENCY: Recurring patterns (every Sunday, weekly)
        - AGE: Age/life period (when she was 5, in my 20s)

    Args:
        split: Dataset split to load ("train", "validation", "test").
            If None, returns DatasetDict with all available splits.
        data_dir: Path to the FamilyOS temporal data directory.
            Default: "data/familyos/temporal"

    Returns:
        HuggingFace Dataset with columns:
            - tokens: list[str] - input tokens
            - temporal_tags: list[int] - BIO tag IDs from TEMPORAL_LABELS

    Raises:
        FileNotFoundError: If data directory or split file does not exist.
        ValueError: If no valid samples found or BIO sequence invalid.

    Example:
        >>> from modeling_studio.data.loaders import load_familyos_temporal
        >>> from modeling_studio.data.labels import TEMPORAL_LABELS
        >>> ds = load_familyos_temporal(split="train")
        >>> sample = ds[0]
        >>> for token, tag_id in zip(sample['tokens'], sample['temporal_tags']):
        ...     print(f"{token}: {TEMPORAL_LABELS.decode(tag_id)}")
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"FamilyOS temporal data directory not found: {data_path}")

    logger.info(f"Loading FamilyOS temporal dataset from {data_path}")

    # Find available split files
    split_files = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "valid.jsonl", "dev.jsonl"],
        "test": ["test.jsonl", "test.json"],
    }

    datasets = {}
    for split_name, file_options in split_files.items():
        for filename in file_options:
            filepath = data_path / filename
            if filepath.exists():
                ds = _load_familyos_temporal_jsonl(filepath)
                datasets[split_name] = ds
                logger.info(f"  Loaded {split_name}: {len(ds)} samples from {filename}")
                break

    if not datasets:
        raise FileNotFoundError(
            f"No valid temporal split files found in {data_path}. "
            f"Expected: train.jsonl, validation.jsonl, or test.jsonl"
        )

    if split:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        return datasets[split]

    return DatasetDict(datasets)


def _load_familyos_temporal_jsonl(filepath: Path) -> Dataset:
    """
    Load FamilyOS temporal data from a JSONL file.

    Expected JSONL format:
        {"tokens": ["We", "went", "yesterday"], "temporal_tags": [0, 0, 3]}

    Where temporal_tags are BIO tag IDs (0-12) from TEMPORAL_LABELS.
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Validate required fields
            if "tokens" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'tokens' field")
                continue

            if "temporal_tags" not in item:
                logger.warning(f"Skipping line {line_num}: missing 'temporal_tags' field")
                continue

            tokens = item["tokens"]
            temporal_tags = item["temporal_tags"]

            # Validate length match
            if len(tokens) != len(temporal_tags):
                logger.warning(
                    f"Skipping line {line_num}: tokens ({len(tokens)}) and "
                    f"temporal_tags ({len(temporal_tags)}) length mismatch"
                )
                continue

            # Convert string tags to IDs if needed
            converted_tags = []
            valid = True
            for i, tag in enumerate(temporal_tags):
                if isinstance(tag, str):
                    try:
                        tag = TEMPORAL_LABELS.encode(tag)
                    except KeyError:
                        logger.warning(
                            f"Skipping line {line_num}: unknown tag '{tag}' at position {i}. "
                            f"Valid tags: {list(TEMPORAL_LABELS.label2id.keys())}"
                        )
                        valid = False
                        break

                # Validate tag ID is in range
                if not (0 <= tag < TEMPORAL_LABELS.num_labels):
                    logger.warning(
                        f"Skipping line {line_num}: "
                        f"tag {tag} out of range (0-{TEMPORAL_LABELS.num_labels - 1})"
                    )
                    valid = False
                    break

                converted_tags.append(tag)

            if not valid:
                continue

            # Validate BIO consistency
            bio_error = _validate_bio_tags(converted_tags, TEMPORAL_LABELS)
            if bio_error:
                logger.warning(f"Line {line_num}: {bio_error} (accepting anyway)")

            data.append(
                {
                    "tokens": tokens,
                    "temporal_tags": converted_tags,
                }
            )

    if not data:
        raise ValueError(f"No valid samples found in {filepath}")

    return Dataset.from_list(data)


# =============================================================================
# Config-Based Dataset Loading
# =============================================================================

# Task type to loader function mapping
TASK_LOADERS = {
    # Token classification tasks
    "ner_general": load_ner_dataset,
    "ner_family": load_familyos_ner,
    "temporal": load_familyos_temporal,
    # Sequence classification tasks
    "sentiment": load_classification_dataset,
    "ingress": load_familyos_ingress,
    "intent": load_familyos_intents,
    "safety_familyos": load_familyos_safety,
    # Multi-label classification
    "emotions": load_multilabel_dataset,
    "safety_generic": load_multilabel_dataset,
    # Pair/Relation tasks
    "nli": load_nli_dataset,
    "relation": load_familyos_relations,
    # Embedding tasks
    "embedding": load_embedding_dataset,
}


def load_from_config(
    config_path: str | Path,
    split: str = "train",
    tokenizer: Any | None = None,
    apply_tokenization: bool = False,
    skip_errors: bool = False,
) -> dict[str, Dataset]:
    """
    Load all datasets defined in a YAML config file.

    Parses the config file, routes each dataset to the appropriate loader,
    and returns a dict mapping task name to HuggingFace Dataset.

    Args:
        config_path: Path to the YAML config file.
            Example: "configs/data/multitask/stage_a_datasets.yaml"
        split: Which split to load ("train", "validation", "test").
            Default: "train"
        tokenizer: Optional tokenizer for preprocessing. If None,
            tokenization is not applied.
        apply_tokenization: Whether to tokenize the datasets.
            Default: False (tokenization done in collator)
        skip_errors: If True, skip datasets that fail to load instead of raising.
            Default: False

    Returns:
        Dict mapping task name to HuggingFace Dataset.
        Keys are task/capability names (e.g., "ner_general", "sentiment").

    Config Format:
        datasets:
          dataset_name:
            task: task_type  # e.g., "ner_general", "sentiment"
            source: huggingface | local
            name: dataset_name_or_path
            enabled: true | false (optional, default true)
            max_samples: int (optional)
            splits:
              train: train_split_name
              validation: val_split_name
            column_mapping:
              text: source_column
              label: source_label_column

    Example:
        >>> from modeling_studio.data.loaders import load_from_config
        >>> datasets = load_from_config("configs/data/multitask/stage_a_datasets.yaml")
        >>> for name, ds in datasets.items():
        ...     print(f"{name}: {len(ds)} samples")
        ner_general: 14041 samples
        sentiment: 67349 samples
        emotions: 43410 samples

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config format is invalid.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading datasets from config: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "datasets" not in config:
        raise ValueError(f"Config must have 'datasets' section: {config_path}")

    # Get preprocessing config
    preprocessing = config.get("preprocessing", {})
    max_length = preprocessing.get("max_length", 512)

    datasets = {}
    loaded_tasks = set()

    for dataset_name, dataset_config in config["datasets"].items():
        # Skip disabled datasets
        if not dataset_config.get("enabled", True):
            logger.debug(f"Skipping disabled dataset: {dataset_name}")
            continue

        task = dataset_config.get("task")
        if task is None:
            logger.warning(f"Skipping {dataset_name}: no 'task' specified")
            continue

        source = dataset_config.get("source", "huggingface")
        name = dataset_config.get("name", dataset_name)

        # Get split mapping
        splits = dataset_config.get("splits", {})
        split_name = splits.get(split)
        if split_name is None:
            # Try direct split name
            split_name = split

        logger.info(f"Loading {dataset_name} (task={task}, source={source}, split={split_name})")

        try:
            # Route to appropriate loader based on task type
            ds = _load_dataset_by_task(
                task=task,
                source=source,
                name=name,
                split=split_name,
                dataset_config=dataset_config,
            )

            # Apply max_samples limit if specified
            max_samples = dataset_config.get("max_samples")
            if max_samples is not None and len(ds) > max_samples:
                logger.info(f"Limiting {dataset_name} to {max_samples} samples (was {len(ds)})")
                ds = ds.shuffle(seed=42).select(range(max_samples))

            # Apply tokenization if requested
            if apply_tokenization and tokenizer is not None:
                ds = _apply_tokenization(ds, task, tokenizer, max_length)

            # Merge datasets with same task (e.g., multiple NER sources)
            if task in datasets:
                # Concatenate with existing dataset for this task
                from datasets import concatenate_datasets

                datasets[task] = concatenate_datasets([datasets[task], ds])
                logger.info(f"Merged {dataset_name} into {task} (total: {len(datasets[task])})")
            else:
                datasets[task] = ds
                loaded_tasks.add(task)

            logger.info(f"Loaded {dataset_name}: {len(ds)} samples")

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            if not skip_errors:
                raise
            logger.warning(f"Skipping {dataset_name} due to error (skip_errors=True)")

    logger.info(
        f"Loaded {len(loaded_tasks)} tasks with {sum(len(ds) for ds in datasets.values())} total samples"
    )

    return datasets


def _load_dataset_by_task(
    task: str,
    source: str,
    name: str,
    split: str,
    dataset_config: dict,
) -> Dataset:
    """
    Load a dataset using the appropriate loader for the task type.

    Args:
        task: Task type (e.g., "ner_general", "sentiment").
        source: Data source ("huggingface" or "local").
        name: Dataset name or path.
        split: Split to load.
        dataset_config: Full dataset configuration dict.

    Returns:
        HuggingFace Dataset.
    """
    # Get data directory for local sources
    data_dir = dataset_config.get("data_dir")
    config_name = dataset_config.get("config")

    # Helper to ensure data_dir is always str or Path
    def ensure_data_dir(val, default):
        return val if val is not None else default

    # Route based on task type
    if task == "ner_general":
        ds = load_ner_dataset(name=name, split=split, data_dir=data_dir)
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "ner_family":
        ds = load_familyos_ner(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/ner_family")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "temporal":
        ds = load_familyos_temporal(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/temporal")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "sentiment":
        ds = load_classification_dataset(
            name=name,
            split=split,
            label_schema=SENTIMENT_LABELS,
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "ingress":
        ds = load_familyos_ingress(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/ingress")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "intent":
        ds = load_familyos_intents(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/intents")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "safety_familyos":
        ds = load_familyos_safety(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/safety")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "emotions":
        # Note: config is handled internally by load_multilabel_dataset for go_emotions
        ds = load_multilabel_dataset(
            name=name,
            split=split,
            label_schema=EMOTIONS_LABELS,
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "safety_generic":
        # For safety_generic, use Jigsaw or other toxicity datasets
        ds = load_multilabel_dataset(name=name, split=split, label_schema=SAFETY_GENERIC_LABELS)
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "nli":
        ds = load_nli_dataset(
            name=name,
            split=split,
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "relation":
        ds = load_familyos_relations(
            split=split, data_dir=ensure_data_dir(data_dir, "data/familyos/relations")
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    elif task == "embedding":
        ds = load_embedding_dataset(
            name=name,
            split=split,
            config_name=config_name,
        )
        return ds if not isinstance(ds, DatasetDict) else ds[split]

    else:
        raise ValueError(f"Unknown task type: {task}")


def _apply_tokenization(
    dataset: Dataset,
    task: str,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """
    Apply task-specific tokenization to a dataset.

    Args:
        dataset: HuggingFace Dataset to tokenize.
        task: Task type for selecting tokenization function.
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset.
    """
    from modeling_studio.data.tokenization import (
        tokenize_for_classification,
        tokenize_for_embedding,
        tokenize_for_nli,
        tokenize_for_token_classification,
    )

    # Map string task to tokenization type
    TASK_TYPE_MAP = {  # noqa: N806
        "sentiment": "classification",
        "classification": "classification",
        "ingress": "classification",
        "intent": "classification",
        "safety_familyos": "classification",
        "emotions": "multilabel",
        "safety_generic": "multilabel",
        "ner_general": "token_classification",
        "ner_family": "token_classification",
        "temporal": "token_classification",
        "nli": "nli",
        "relation": "relation",
        "embedding": "embedding",
    }

    mapped_task = TASK_TYPE_MAP.get(task, "classification")

    # Create wrapper function that accepts example dictionary
    if mapped_task == "token_classification":

        def tokenize_wrapper(example):
            result = tokenize_for_token_classification(
                tokenizer=tokenizer,
                tokens=example["tokens"],
                ner_tags=example.get("ner_tags"),
                max_length=max_length,
            )
            # Add task info
            result["task"] = task
            return result

    elif mapped_task == "classification":

        def tokenize_wrapper(example):
            # Try common text column names
            text = example.get("text") or example.get("sentence") or example.get("content")
            label = example.get("label")
            result = tokenize_for_classification(
                tokenizer=tokenizer,
                text=text,
                max_length=max_length,
            )
            if label is not None:
                result["labels"] = label
            result["task"] = task
            return result

    elif mapped_task == "multilabel":

        def tokenize_wrapper(example):
            text = example.get("text") or example.get("sentence") or example.get("content")
            result = tokenize_for_classification(
                tokenizer=tokenizer,
                text=text,
                max_length=max_length,
            )
            # For multi-label, keep labels as a list of floats
            if "labels" in example:
                result["labels"] = example["labels"]
            result["task"] = task
            return result

    elif mapped_task == "nli":

        def tokenize_wrapper(example):
            premise = example.get("premise")
            hypothesis = example.get("hypothesis")
            result = tokenize_for_nli(
                tokenizer=tokenizer,
                premise=premise,
                hypothesis=hypothesis,
                max_length=max_length,
            )
            if "label" in example:
                result["labels"] = example["label"]
            result["task"] = task
            return result

    elif mapped_task == "embedding":

        def tokenize_wrapper(example):
            # Embedding datasets can have various formats
            text1 = example.get("sentence1") or example.get("text") or example.get("anchor")
            text2 = example.get("sentence2") or example.get("positive")

            result = tokenize_for_embedding(
                tokenizer=tokenizer,
                text=text1,
                max_length=max_length,
            )
            # Rename to anchor_*
            result = {f"anchor_{k}": v for k, v in result.items()}

            if text2:
                result2 = tokenize_for_embedding(
                    tokenizer=tokenizer,
                    text=text2,
                    max_length=max_length,
                )
                # Add positive_* keys
                for k, v in result2.items():
                    result[f"positive_{k}"] = v

            # Add score if present (for STS-B style)
            if "label" in example:
                result["labels"] = example["label"]
            elif "score" in example:
                result["labels"] = example["score"]

            result["task"] = task
            return result

    else:
        raise ValueError(f"Unsupported task type for tokenization: {mapped_task}")

    # Use batched=False because our tokenization functions are designed for single examples
    return dataset.map(tokenize_wrapper, batched=False, remove_columns=dataset.column_names)


def load_stage_a_datasets(
    split: str = "train",
    config_path: str | Path = "configs/data/multitask/stage_a_datasets.yaml",
    **kwargs,
) -> dict[str, Dataset]:
    """
    Convenience function to load all Stage A (generic) datasets.

    Stage A includes 7 generic capabilities trained on public datasets:
        - ner_general: CoNLL-2003, OntoNotes
        - sentiment: SST-2, Amazon Reviews
        - emotions: GoEmotions
        - safety_generic: Jigsaw Toxicity
        - nli: MNLI, SNLI
        - embedding: STS-B
        - temporal: TempEval-3 (NEW v2)

    Args:
        split: Split to load ("train", "validation", "test").
        config_path: Path to Stage A config file.
        **kwargs: Additional arguments passed to load_from_config.

    Returns:
        Dict mapping task name to Dataset.

    Example:
        >>> datasets = load_stage_a_datasets()
        >>> assert "ner_general" in datasets
        >>> assert "sentiment" in datasets
        >>> print(f"Loaded {len(datasets)} tasks")
    """
    return load_from_config(config_path, split=split, **kwargs)


def load_stage_b_datasets(
    split: str = "train",
    config_path: str | Path = "configs/data/multitask/stage_b_datasets.yaml",
    **kwargs,
) -> dict[str, Dataset]:
    """
    Convenience function to load all Stage B (FamilyOS) datasets.

    Stage B includes FamilyOS-specific data + replay data:
        - ner_family: Family NER (nicknames, kinship terms)
        - ingress: Domain classification
        - safety_familyos: Policy bands (GREEN/AMBER/RED/CRISIS)
        - relation: Family relationship extraction (NEW v2)
        - intent: User intent classification (NEW v2)
        - Plus replay data from Stage A tasks

    Args:
        split: Split to load ("train", "validation", "test").
        config_path: Path to Stage B config file.
        **kwargs: Additional arguments passed to load_from_config.

    Returns:
        Dict mapping task name to Dataset.

    Example:
        >>> datasets = load_stage_b_datasets()
        >>> assert "ner_family" in datasets
        >>> assert "safety_familyos" in datasets
        >>> print(f"Loaded {len(datasets)} tasks")
    """
    return load_from_config(config_path, split=split, **kwargs)


# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Public dataset loaders
    "load_ner_dataset",
    "load_classification_dataset",
    "load_multilabel_dataset",
    "load_nli_dataset",
    "load_embedding_dataset",
    # FamilyOS dataset loaders
    "load_familyos_ner",
    "load_familyos_ingress",
    "load_familyos_safety",
    "load_familyos_relations",
    "load_familyos_intents",
    "load_familyos_temporal",
    # Config-based loading
    "load_from_config",
    "load_stage_a_datasets",
    "load_stage_b_datasets",
    # Constants
    "TASK_LOADERS",
]

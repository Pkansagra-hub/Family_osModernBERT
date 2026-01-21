#!/usr/bin/env python3
"""
Convert NER Datasets to Span Format for GlobalPointer Training.

This script converts various NER datasets from BIO or flat format to
character-level span format required by GlobalPointer.

Supported Datasets:
    - conll2003: CoNLL-2003 (BIO format)
    - wikineural: WikiNeural (BIO format, 16 types → 4)
    - fewnerd: Few-NERD (flat format, 66 types → 4)
    - ontonotes: OntoNotes 5 (BIO format, 18 types → 4)
    - wikiann: WikiANN (BIO format, sampling supported)

Output Format (JSONL):
    {"text": "...", "entities": [{"start": 0, "end": 4, "label": "PER", "text": "Emma"}, ...]}

Usage:
    python scripts/convert_ner_to_spans.py --dataset conll2003
    python scripts/convert_ner_to_spans.py --dataset wikineural
    python scripts/convert_ner_to_spans.py --dataset fewnerd
    python scripts/convert_ner_to_spans.py --dataset ontonotes
    python scripts/convert_ner_to_spans.py --dataset wikiann --sample 100000

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.span_utils import bio_to_spans, flat_to_spans, validate_spans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Dataset Configurations
# =============================================================================

# CoNLL-2003 label names (9 labels)
CONLL2003_LABELS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]

# WikiNeural label names (33 labels) → mapped to CoNLL 4 types
WIKINEURAL_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-ANIM", "I-ANIM",
    "B-BIO", "I-BIO",
    "B-CEL", "I-CEL",
    "B-DIS", "I-DIS",
    "B-EVE", "I-EVE",
    "B-FOOD", "I-FOOD",
    "B-INST", "I-INST",
    "B-MEDIA", "I-MEDIA",
    "B-PLANT", "I-PLANT",
    "B-MYTH", "I-MYTH",
    "B-TIME", "I-TIME",
    "B-VEHI", "I-VEHI",
    "B-MISC", "I-MISC",
]

WIKINEURAL_LABEL_MAPPING = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "ORG",
    "ANIM": "MISC",
    "BIO": "MISC",
    "CEL": "MISC",
    "DIS": "MISC",
    "EVE": "MISC",
    "FOOD": "MISC",
    "INST": "MISC",
    "MEDIA": "MISC",
    "PLANT": "MISC",
    "MYTH": "MISC",
    "TIME": "MISC",
    "VEHI": "MISC",
    "MISC": "MISC",
}

# Few-NERD flat labels → CoNLL 4 types
# Index-based mapping from ClassLabel indices to CONLL types
FEWNERD_COARSE_LABELS = ['O', 'art', 'building', 'event', 'location', 'organization', 'other', 'person', 'product']

FEWNERD_LABEL_MAPPING = {
    "O": None,  # Skip O
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "building": "LOC",
    "event": "MISC",
    "product": "MISC",
    "art": "MISC",
    "other": "MISC",
}

# OntoNotes 18 types → CoNLL 4 types
ONTONOTES_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-GPE", "I-GPE",
    "B-LOC", "I-LOC",
    "B-FAC", "I-FAC",
    "B-NORP", "I-NORP",
    "B-EVENT", "I-EVENT",
    "B-WORK_OF_ART", "I-WORK_OF_ART",
    "B-LAW", "I-LAW",
    "B-LANGUAGE", "I-LANGUAGE",
    "B-PRODUCT", "I-PRODUCT",
    "B-DATE", "I-DATE",
    "B-TIME", "I-TIME",
    "B-PERCENT", "I-PERCENT",
    "B-MONEY", "I-MONEY",
    "B-QUANTITY", "I-QUANTITY",
    "B-ORDINAL", "I-ORDINAL",
    "B-CARDINAL", "I-CARDINAL",
]

ONTONOTES_LABEL_MAPPING = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "PRODUCT": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "PERCENT": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
}

# WikiANN labels (only 3 types, no MISC)
WIKIANN_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# =============================================================================
# Dataset Loaders
# =============================================================================


def load_conll2003(split: str) -> Dataset:
    """Load CoNLL-2003 dataset."""
    logger.info(f"Loading CoNLL-2003 {split} split...")
    return load_dataset("conll2003", split=split, trust_remote_code=True)


def load_wikineural(split: str) -> Dataset:
    """Load WikiNeural dataset."""
    logger.info(f"Loading WikiNeural {split} split...")
    ds = load_dataset("tner/wikineural", "en", split=split, trust_remote_code=True)
    # Rename 'tags' to 'ner_tags' for consistency
    if "tags" in ds.column_names:
        ds = ds.rename_column("tags", "ner_tags")
    return ds


def load_fewnerd(split: str) -> Dataset:
    """Load Few-NERD dataset."""
    logger.info(f"Loading Few-NERD {split} split...")
    # Few-NERD uses 'supervised' config
    return load_dataset("DFKI-SLT/few-nerd", "supervised", split=split, trust_remote_code=True)


def load_ontonotes(split: str) -> Dataset:
    """Load OntoNotes 5 dataset."""
    logger.info(f"Loading OntoNotes 5 {split} split...")
    return load_dataset("tner/ontonotes5", split=split, trust_remote_code=True)


def load_wikiann(split: str, sample_size: int | None = None, seed: int = 42) -> Dataset:
    """Load WikiANN dataset with optional sampling."""
    logger.info(f"Loading WikiANN {split} split...")
    ds = load_dataset("wikiann", "en", split=split, trust_remote_code=True)

    if sample_size and len(ds) > sample_size:
        logger.info(f"Sampling {sample_size} from {len(ds)} samples...")
        random.seed(seed)
        indices = random.sample(range(len(ds)), sample_size)
        ds = ds.select(indices)

    return ds


# =============================================================================
# Conversion Functions
# =============================================================================


def convert_bio_dataset(
    dataset: Dataset,
    label_names: list[str],
    label_mapping: dict[str, str] | None = None,
    tokens_col: str = "tokens",
    tags_col: str = "ner_tags",
) -> list[dict[str, Any]]:
    """Convert a BIO-format dataset to span format."""
    converted = []
    errors = 0

    for sample in tqdm(dataset, desc="Converting"):
        tokens = sample[tokens_col]
        tags = sample[tags_col]

        try:
            result = bio_to_spans(
                tokens=tokens,
                bio_tags=tags,
                label_names=label_names,
                label_mapping=label_mapping,
            )

            # Validate
            is_valid, errs = validate_spans(result, valid_labels={"PER", "ORG", "LOC", "MISC"})
            if not is_valid:
                errors += 1
                logger.debug(f"Validation errors: {errs}")
                continue

            converted.append(result)

        except Exception as e:
            errors += 1
            logger.debug(f"Conversion error: {e}")

    if errors > 0:
        logger.warning(f"Skipped {errors} samples due to errors")

    return converted


def convert_fewnerd_dataset(dataset: Dataset) -> list[dict[str, Any]]:
    """Convert Few-NERD flat-format dataset to span format.

    Few-NERD uses ClassLabel integers for ner_tags, not strings.
    We need to convert indices to string labels first.
    """
    converted = []
    errors = 0

    for sample in tqdm(dataset, desc="Converting Few-NERD"):
        tokens = sample["tokens"]
        # Few-NERD uses 'ner_tags' as ClassLabel integers (0-8)
        tag_indices = sample.get("ner_tags", [])

        # Convert integer indices to string labels
        tags = [FEWNERD_COARSE_LABELS[idx] for idx in tag_indices]

        try:
            result = flat_to_spans(
                tokens=tokens,
                flat_tags=tags,
                label_mapping=FEWNERD_LABEL_MAPPING,
            )

            # Validate
            is_valid, errs = validate_spans(result, valid_labels={"PER", "ORG", "LOC", "MISC"})
            if not is_valid:
                errors += 1
                logger.debug(f"Validation errors: {errs}")
                continue

            converted.append(result)

        except Exception as e:
            errors += 1
            logger.debug(f"Conversion error: {e}")

    if errors > 0:
        logger.warning(f"Skipped {errors} samples due to errors")

    return converted


# =============================================================================
# Statistics
# =============================================================================


def compute_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics for converted samples."""
    total_samples = len(samples)
    total_entities = 0
    label_counts: Counter = Counter()
    entities_per_sample: list[int] = []

    for sample in samples:
        entities = sample.get("entities", [])
        n_entities = len(entities)
        total_entities += n_entities
        entities_per_sample.append(n_entities)

        for entity in entities:
            label_counts[entity["label"]] += 1

    return {
        "total_samples": total_samples,
        "total_entities": total_entities,
        "entities_per_sample_avg": total_entities / total_samples if total_samples > 0 else 0,
        "entities_per_sample_max": max(entities_per_sample) if entities_per_sample else 0,
        "samples_with_entities": sum(1 for e in entities_per_sample if e > 0),
        "label_distribution": dict(label_counts),
    }


# =============================================================================
# Output
# =============================================================================


def save_jsonl(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} samples to {output_path}")


# =============================================================================
# Main
# =============================================================================


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert a single dataset to span format."""

    output_dataset_dir = output_dir / dataset_name
    all_stats = {}

    # Determine splits based on dataset
    if dataset_name == "fewnerd":
        splits = ["train"]  # Few-NERD only has train in supervised config
    elif dataset_name == "wikiann":
        splits = ["train"]  # Only convert train for sampling
    else:
        splits = ["train", "validation", "test"]

    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name} - {split}")
        logger.info(f"{'='*60}")

        try:
            # Load dataset
            if dataset_name == "conll2003":
                ds = load_conll2003(split)
                samples = convert_bio_dataset(ds, CONLL2003_LABELS)

            elif dataset_name == "wikineural":
                ds = load_wikineural(split)
                samples = convert_bio_dataset(
                    ds, WIKINEURAL_LABELS, WIKINEURAL_LABEL_MAPPING,
                    tags_col="ner_tags"
                )

            elif dataset_name == "fewnerd":
                ds = load_fewnerd(split)
                samples = convert_fewnerd_dataset(ds)

            elif dataset_name == "ontonotes":
                ds = load_ontonotes(split)
                samples = convert_bio_dataset(
                    ds, ONTONOTES_LABELS, ONTONOTES_LABEL_MAPPING,
                    tags_col="tags"  # OntoNotes uses 'tags'
                )

            elif dataset_name == "wikiann":
                ds = load_wikiann(split, sample_size=sample_size, seed=seed)
                samples = convert_bio_dataset(ds, WIKIANN_LABELS)

            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Compute statistics
            stats = compute_statistics(samples)
            all_stats[split] = stats
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

            # Save
            output_path = output_dataset_dir / f"{split}.jsonl"
            save_jsonl(samples, output_path)

        except Exception as e:
            logger.error(f"Failed to process {dataset_name}/{split}: {e}")
            import traceback
            traceback.print_exc()

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert NER datasets to span format for GlobalPointer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["conll2003", "wikineural", "fewnerd", "ontonotes", "wikiann", "all"],
        help="Dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ner_general_span"),
        help="Output directory",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (for wikiann)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    all_stats = {}

    if args.dataset == "all":
        datasets = ["conll2003", "wikineural", "fewnerd", "ontonotes"]
        if args.sample:
            datasets.append("wikiann")
    else:
        datasets = [args.dataset]

    for ds_name in datasets:
        sample_size = args.sample if ds_name == "wikiann" else None
        stats = convert_dataset(
            ds_name,
            args.output_dir,
            sample_size=sample_size,
            seed=args.seed,
        )
        all_stats[ds_name] = stats

    # Save combined statistics
    stats_path = args.output_dir / "conversion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*60)

    total_samples = 0
    total_entities = 0
    for ds_name, ds_stats in all_stats.items():
        for split, split_stats in ds_stats.items():
            n_samples = split_stats.get("total_samples", 0)
            n_entities = split_stats.get("total_entities", 0)
            total_samples += n_samples
            total_entities += n_entities
            logger.info(f"{ds_name}/{split}: {n_samples:,} samples, {n_entities:,} entities")

    logger.info("-"*60)
    logger.info(f"TOTAL: {total_samples:,} samples, {total_entities:,} entities")


if __name__ == "__main__":
    main()

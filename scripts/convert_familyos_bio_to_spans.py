#!/usr/bin/env python3
"""
Convert FamilyOS NER/Temporal BIO Data to Span Format.

This script converts the gold/silver BIO-tagged data in:
    - data/familyos/ner_family/gold/
    - data/familyos/ner_family/silver/
    - data/familyos/temporal/gold/
    - data/familyos/temporal/silver/

To span format for GlobalPointer training:
    {"text": "...", "entities": [{"start": 0, "end": 4, "label": "KINSHIP", "text": "Mom"}, ...]}

Output goes to:
    - data/familyos/ner_family_span/
    - data/familyos/temporal_span/

Usage:
    python scripts/convert_familyos_bio_to_spans.py --task ner_family
    python scripts/convert_familyos_bio_to_spans.py --task temporal
    python scripts/convert_familyos_bio_to_spans.py --task all

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.labels import NER_FAMILY_LABELS, TEMPORAL_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# BIO to Span Conversion
# =============================================================================


def bio_to_spans_familyos(
    tokens: list[str],
    bio_tags: list[int],
    label_schema,
) -> dict:
    """
    Convert BIO-tagged tokens to span format.

    Args:
        tokens: List of word tokens
        bio_tags: List of BIO tag IDs
        label_schema: LabelSchema with id2label mapping

    Returns:
        {"text": "...", "entities": [{"start": ..., "end": ..., "label": ..., "text": ...}]}
    """
    if not tokens or not bio_tags:
        return {"text": "", "entities": []}

    if len(tokens) != len(bio_tags):
        logger.warning(f"Length mismatch: {len(tokens)} tokens vs {len(bio_tags)} tags")
        return {"text": "", "entities": []}

    # Build text and track character positions
    text = " ".join(tokens)

    # Track character position for each token
    char_positions: list[tuple[int, int]] = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = start + len(token)
        char_positions.append((start, end))
        current_pos = end + 1  # +1 for space

    # Extract entities
    entities: list[dict] = []
    current_entity = None

    for i, (token, tag_id) in enumerate(zip(tokens, bio_tags)):
        tag_name = label_schema.id2label.get(tag_id, "O")

        if tag_name == "O":
            # End current entity if exists
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag_name.startswith("B-"):
            # Start new entity
            if current_entity:
                entities.append(current_entity)

            label = tag_name[2:]  # Remove "B-" prefix
            start, end = char_positions[i]
            current_entity = {
                "start": start,
                "end": end,
                "label": label,
                "text": token,
            }
        elif tag_name.startswith("I-"):
            # Continue entity
            label = tag_name[2:]  # Remove "I-" prefix

            if current_entity and current_entity["label"] == label:
                # Extend current entity
                _, end = char_positions[i]
                current_entity["end"] = end
                current_entity["text"] += " " + token
            else:
                # Orphan I-tag or label mismatch - treat as B-tag
                if current_entity:
                    entities.append(current_entity)

                start, end = char_positions[i]
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": token,
                }

    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)

    return {"text": text, "entities": entities}


def convert_file(
    input_path: Path,
    output_path: Path,
    label_schema,
    tags_key: str = "ner_tags",
) -> tuple[int, int, Counter]:
    """
    Convert a single JSONL file from BIO to span format.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        label_schema: LabelSchema for this task
        tags_key: Key for tags in input ("ner_tags" or "temporal_tags")

    Returns:
        (total_samples, samples_with_entities, label_counts)
    """
    total = 0
    with_entities = 0
    label_counts: Counter = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            raw = json.loads(line.strip())

            tokens = raw.get("tokens", [])
            bio_tags = raw.get(tags_key, [])

            if not tokens or not bio_tags:
                continue

            result = bio_to_spans_familyos(tokens, bio_tags, label_schema)

            if result["entities"]:
                with_entities += 1
                for ent in result["entities"]:
                    label_counts[ent["label"]] += 1

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    return total, with_entities, label_counts


def convert_task(
    task: str,
    data_root: Path,
) -> None:
    """
    Convert all files for a task.

    Args:
        task: "ner_family" or "temporal"
        data_root: Root data directory
    """
    if task == "ner_family":
        label_schema = NER_FAMILY_LABELS
        tags_key = "ner_tags"
        input_dir = data_root / "familyos" / "ner_family"
        output_dir = data_root / "familyos" / "ner_family_span"
    elif task == "temporal":
        label_schema = TEMPORAL_LABELS
        tags_key = "temporal_tags"
        input_dir = data_root / "familyos" / "temporal"
        output_dir = data_root / "familyos" / "temporal_span"
    else:
        raise ValueError(f"Unknown task: {task}")

    logger.info(f"Converting {task} from {input_dir} to {output_dir}")

    total_samples = 0
    total_with_entities = 0
    total_label_counts: Counter = Counter()

    # Process gold and silver directories
    for subdir in ["gold", "silver"]:
        subdir_path = input_dir / subdir
        if not subdir_path.exists():
            logger.warning(f"Directory not found: {subdir_path}")
            continue

        output_subdir = output_dir / subdir

        # Process all JSONL files
        for input_file in sorted(subdir_path.glob("*.jsonl")):
            output_file = output_subdir / input_file.name

            samples, with_ents, counts = convert_file(
                input_file,
                output_file,
                label_schema,
                tags_key,
            )

            total_samples += samples
            total_with_entities += with_ents
            total_label_counts.update(counts)

            logger.info(
                f"  {subdir}/{input_file.name}: "
                f"{samples} samples, {with_ents} with entities"
            )

    # Summary
    logger.info(f"\n{task} Conversion Summary:")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Samples with entities: {total_with_entities:,} ({100*total_with_entities/max(1,total_samples):.1f}%)")
    logger.info(f"  Label distribution:")
    for label, count in sorted(total_label_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {label}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FamilyOS BIO data to span format"
    )
    parser.add_argument(
        "--task",
        choices=["ner_family", "temporal", "all"],
        default="all",
        help="Which task to convert",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )

    args = parser.parse_args()

    if args.task == "all":
        tasks = ["ner_family", "temporal"]
    else:
        tasks = [args.task]

    for task in tasks:
        convert_task(task, args.data_root)

    logger.info("\nConversion complete!")
    logger.info("Output directories:")
    logger.info(f"  data/familyos/ner_family_span/")
    logger.info(f"  data/familyos/temporal_span/")


if __name__ == "__main__":
    main()

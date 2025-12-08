#!/usr/bin/env python
"""
Consolidate 44 FamilyOS emotions into 7 super-labels.

This script pre-processes emotion data for Stage A curriculum learning.
Instead of mapping labels at runtime (slow), we create a static file
with super-labels already applied.

Input:  data/familyos/emotions/silver/train.jsonl
        Format: {"text": "...", "emotions": ["joy", "love"], ...}

Output: data/familyos/emotions/silver/train_CONSOLIDATED.jsonl
        Format: {"text": "...", "labels": [1, 1, 0, 0, 0, 0, 0]}  # 7-element multi-hot

Super-Label Mapping (7 classes):
    JOY (0):        joy, excitement, celebration, pride, relief, amusement, hope, optimism, surprise
    AFFECTION (1):  love, warmth, caring, gratitude, tenderness, admiration, parental_pride, protectiveness, playfulness
    SADNESS (2):    sadness, grief, disappointment, longing, emptiness, remorse, parental_guilt
    ANXIETY (3):    worry, overwhelmed, frustration, annoyance, nervousness, fear, anger, disgust, disapproval, embarrassment
    NOSTALGIA (4):  nostalgia, bittersweet, homesickness
    CONTENTMENT (5): contentment, belonging, togetherness, patience, approval
    NEUTRAL (6):    neutral

Usage:
    python scripts/consolidate_labels.py

    # Custom paths
    python scripts/consolidate_labels.py \
        --input data/familyos/emotions/silver/train.jsonl \
        --output data/familyos/emotions/silver/train_CONSOLIDATED.jsonl

    # Also process validation set
    python scripts/consolidate_labels.py --include-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modeling_studio.data.labels import (
    EMOTIONS_SUPER_LABELS,
    map_emotion_names_to_super_labels,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def consolidate_file(input_path: Path, output_path: Path) -> dict[str, int]:
    """
    Consolidate a single JSONL file from 44 emotions to 7 super-labels.

    Args:
        input_path: Path to input JSONL file with 44-label emotions
        output_path: Path to output JSONL file with 7-label super-labels

    Returns:
        Dictionary with statistics about the consolidation
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = {
        "total_rows": 0,
        "rows_with_emotions": 0,
        "rows_empty_labels": 0,
        "errors": 0,
    }
    super_label_counts = Counter()

    logger.info(f"Reading: {input_path}")
    logger.info(f"Writing: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                    stats["total_rows"] += 1

                    # Get emotions (might be in 'emotions' or 'labels' field)
                    emotions = row.get("emotions", row.get("labels", []))

                    # Handle case where emotions is already multi-hot
                    if emotions and isinstance(emotions[0], int):
                        # Already multi-hot, skip (shouldn't happen in silver data)
                        logger.warning(f"Line {line_num}: Already multi-hot, skipping")
                        continue

                    # Convert emotion names to super-labels
                    if emotions:
                        stats["rows_with_emotions"] += 1
                        super_labels = map_emotion_names_to_super_labels(emotions)
                    else:
                        stats["rows_empty_labels"] += 1
                        super_labels = [0] * EMOTIONS_SUPER_LABELS.num_labels

                    # Count super-label distribution
                    for idx, val in enumerate(super_labels):
                        if val == 1:
                            label_name = EMOTIONS_SUPER_LABELS.id2label[idx]
                            super_label_counts[label_name] += 1

                    # Write consolidated row
                    output_row = {
                        "text": row.get("text", ""),
                        "labels": super_labels,
                    }
                    f_out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                    stats["errors"] += 1
                except Exception as e:
                    logger.warning(f"Line {line_num}: Error - {e}")
                    stats["errors"] += 1

    # Log statistics
    logger.info(f"Processed {stats['total_rows']} rows")
    logger.info(f"  - With emotions: {stats['rows_with_emotions']}")
    logger.info(f"  - Empty labels: {stats['rows_empty_labels']}")
    logger.info(f"  - Errors: {stats['errors']}")

    logger.info("Super-label distribution:")
    total_labels = sum(super_label_counts.values())
    for label_name in EMOTIONS_SUPER_LABELS.label2id.keys():
        count = super_label_counts.get(label_name, 0)
        pct = (count / total_labels * 100) if total_labels > 0 else 0
        logger.info(f"  {label_name}: {count:,} ({pct:.1f}%)")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate 44 FamilyOS emotions into 7 super-labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/familyos/emotions/silver/train.jsonl"),
        help="Input JSONL file with 44-label emotions",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: input_CONSOLIDATED.jsonl)",
    )

    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Also process validation.jsonl if it exists",
    )

    parser.add_argument(
        "--include-shards",
        action="store_true",
        help="Also process shard_*.jsonl files",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        stem = args.input.stem
        args.output = args.input.parent / f"{stem}_CONSOLIDATED.jsonl"

    # Process main file
    logger.info("=" * 60)
    logger.info("Stage A Super-Label Consolidation")
    logger.info("=" * 60)

    consolidate_file(args.input, args.output)

    # Process validation if requested
    if args.include_validation:
        val_input = args.input.parent / "validation.jsonl"
        if val_input.exists():
            val_output = args.input.parent / "validation_CONSOLIDATED.jsonl"
            logger.info("")
            logger.info("Processing validation set...")
            consolidate_file(val_input, val_output)
        else:
            logger.info(f"No validation file found at {val_input}")

    # Process shards if requested
    if args.include_shards:
        shard_files = sorted(args.input.parent.glob("shard_*.jsonl"))
        for shard_input in shard_files:
            shard_output = shard_input.parent / f"{shard_input.stem}_CONSOLIDATED.jsonl"
            logger.info("")
            logger.info(f"Processing shard: {shard_input.name}")
            consolidate_file(shard_input, shard_output)

    logger.info("")
    logger.info("Done! Output written to:")
    logger.info(f"  {args.output}")


if __name__ == "__main__":
    main()

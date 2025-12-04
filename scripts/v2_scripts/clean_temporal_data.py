#!/usr/bin/env python3
"""
Clean Temporal Dataset - Fix BIO Tag Violations

Issues found:
1. I-tags without preceding B-tags (e.g., I-DATE_REL after O)
2. Token/tag length mismatches
3. Orphan I-tags

This script:
1. Converts orphan I-tags to B-tags
2. Removes samples with length mismatches
3. Validates all BIO sequences
"""

import json
from pathlib import Path
from collections import Counter


# Temporal label mapping from labels.py
TEMPORAL_ID2LABEL = {
    0: "O",
    1: "B-DATE_ABS",
    2: "I-DATE_ABS",
    3: "B-DATE_REL",
    4: "I-DATE_REL",
    5: "B-TIME",
    6: "I-TIME",
    7: "B-DURATION",
    8: "I-DURATION",
    9: "B-FREQUENCY",
    10: "I-FREQUENCY",
    11: "B-AGE",
    12: "I-AGE",
}

# Map I-tags to their B-tag counterparts
I_TO_B = {
    2: 1,  # I-DATE_ABS -> B-DATE_ABS
    4: 3,  # I-DATE_REL -> B-DATE_REL
    6: 5,  # I-TIME -> B-TIME
    8: 7,  # I-DURATION -> B-DURATION
    10: 9,  # I-FREQUENCY -> B-FREQUENCY
    12: 11,  # I-AGE -> B-AGE
}

# Map I-tags to entity types
I_TAG_ENTITY = {
    2: "DATE_ABS",
    4: "DATE_REL",
    6: "TIME",
    8: "DURATION",
    10: "FREQUENCY",
    12: "AGE",
}


def fix_bio_sequence(tags: list[int]) -> list[int]:
    """
    Fix BIO sequence violations by converting orphan I-tags to B-tags.

    Rules:
    - If an I-tag follows O or a different entity type, convert it to B-tag
    - If an I-tag follows same entity's B or I tag, keep it as I-tag
    """
    if not tags:
        return tags

    fixed_tags = []
    prev_entity = None

    for i, tag_id in enumerate(tags):
        tag_name = TEMPORAL_ID2LABEL.get(tag_id, "O")

        if tag_name == "O":
            fixed_tags.append(0)
            prev_entity = None
        elif tag_name.startswith("B-"):
            fixed_tags.append(tag_id)
            prev_entity = tag_name[2:]  # e.g., "DATE_ABS"
        elif tag_name.startswith("I-"):
            entity_type = tag_name[2:]  # e.g., "DATE_ABS"

            # Check if this I-tag should be converted to B-tag
            if prev_entity != entity_type:
                # Orphan I-tag - convert to B-tag
                b_tag_id = I_TO_B.get(tag_id, tag_id)
                fixed_tags.append(b_tag_id)
            else:
                # Valid continuation
                fixed_tags.append(tag_id)

            prev_entity = entity_type
        else:
            # Unknown tag, treat as O
            fixed_tags.append(0)
            prev_entity = None

    return fixed_tags


def clean_temporal_file(input_path: Path, output_path: Path) -> dict:
    """Clean a single temporal JSONL file."""
    stats = Counter()
    cleaned_samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            tokens = sample.get("tokens", [])
            tags = sample.get("temporal_tags", [])

            # Check length mismatch
            if len(tokens) != len(tags):
                stats["length_mismatch"] += 1
                continue

            # Fix BIO violations
            original_tags = tags.copy()
            fixed_tags = fix_bio_sequence(tags)

            if fixed_tags != original_tags:
                stats["bio_violations_fixed"] += 1

            # Clean tokens (remove trailing whitespace/newlines)
            cleaned_tokens = [t.strip() for t in tokens]

            cleaned_samples.append(
                {
                    "tokens": cleaned_tokens,
                    "temporal_tags": fixed_tags,
                }
            )
            stats["valid_samples"] += 1

    # Write cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample) + "\n")

    return dict(stats)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean temporal NER dataset")
    parser.add_argument(
        "--input-dir",
        default="data/familyos/temporal/silver",
        help="Input directory with JSONL shards",
    )
    parser.add_argument(
        "--output-dir",
        default="data/familyos/temporal/silver_cleaned",
        help="Output directory for cleaned data",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Clean in-place (overwrite original files)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if not args.in_place else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    total_stats = Counter()
    shard_files = list(input_dir.glob("*.jsonl"))

    print(f"Cleaning {len(shard_files)} temporal data shards...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    for shard_path in sorted(shard_files):
        output_path = output_dir / shard_path.name
        stats = clean_temporal_file(shard_path, output_path)

        print(f"  {shard_path.name}:")
        print(f"    Valid samples: {stats.get('valid_samples', 0)}")
        print(f"    BIO violations fixed: {stats.get('bio_violations_fixed', 0)}")
        print(f"    Length mismatches (dropped): {stats.get('length_mismatch', 0)}")
        print(f"    JSON errors: {stats.get('json_errors', 0)}")

        for key, value in stats.items():
            total_stats[key] += value

    print()
    print("=" * 50)
    print("TOTAL STATS:")
    print(f"  Valid samples: {total_stats['valid_samples']}")
    print(f"  BIO violations fixed: {total_stats['bio_violations_fixed']}")
    print(f"  Length mismatches dropped: {total_stats['length_mismatch']}")
    print(f"  JSON errors: {total_stats['json_errors']}")
    print("=" * 50)

    if args.in_place:
        print("\n✓ Files cleaned in-place")
    else:
        print(f"\n✓ Cleaned files written to: {output_dir}")
        print("\nTo use cleaned data, update your config to point to:")
        print(f"  data_dir: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

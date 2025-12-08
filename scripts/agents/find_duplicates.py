"""
Find Duplicate Text Fields in Unified Dataset

Analyzes all shard files to find duplicate 'text' fields.
Reports statistics and optionally lists all duplicates.

Usage:
    python find_duplicates.py
    python find_duplicates.py --show-all  # Show all duplicate texts
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Path to unified data
UNIFIED_DIR = Path("D:/Modeling_studio/data/familyos/unified/output")


def analyze_duplicates(data_dir: Path, show_all: bool = False):
    """Find and analyze duplicate text fields."""

    text_to_ids = defaultdict(list)
    total_samples = 0

    # Read all shards
    shards = sorted(data_dir.glob("shard_*.jsonl"))

    if not shards:
        print(f"No shard files found in {data_dir}")
        return

    print(f"Analyzing {len(shards)} shard files...")
    print()

    for shard_path in shards:
        with open(shard_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    total_samples += 1

                    text = sample.get("text", "").strip()
                    sample_id = sample.get("id", f"{shard_path.stem}:L{line_num}")

                    if text:
                        text_to_ids[text].append((sample_id, shard_path.name))

                except json.JSONDecodeError:
                    continue

    # Find duplicates
    duplicates = {text: ids for text, ids in text_to_ids.items() if len(ids) > 1}
    unique_texts = len(text_to_ids) - len(duplicates)
    duplicate_sample_count = sum(len(ids) - 1 for ids in duplicates.values())

    # Statistics
    print("=" * 80)
    print("DUPLICATE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total samples: {total_samples:,}")
    print(f"Unique text fields: {len(text_to_ids):,}")
    print(f"Duplicate text fields: {len(duplicates):,}")
    print(f"Extra samples (duplicates): {duplicate_sample_count:,}")
    print(f"Deduplication rate: {100 * duplicate_sample_count / total_samples:.2f}%")
    print()

    if duplicates:
        # Show top duplicates by frequency
        duplicate_counts = Counter({text: len(ids) for text, ids in duplicates.items()})

        print("=" * 80)
        print("TOP 20 MOST DUPLICATED TEXTS")
        print("=" * 80)

        for i, (text, count) in enumerate(duplicate_counts.most_common(20), 1):
            preview = text[:70] + "..." if len(text) > 70 else text
            print(f"{i:2d}. ({count}x) {preview}")
            if show_all:
                for sample_id, shard in duplicates[text]:
                    print(f"    - {sample_id} in {shard}")
        print()

        # Duplicate count distribution
        dup_distribution = Counter(duplicate_counts.values())
        print("=" * 80)
        print("DUPLICATE FREQUENCY DISTRIBUTION")
        print("=" * 80)
        for freq in sorted(dup_distribution.keys()):
            print(f"  Appears {freq}x: {dup_distribution[freq]:,} unique texts")
        print()

        if show_all and len(duplicates) > 20:
            print("=" * 80)
            print(f"ALL {len(duplicates)} DUPLICATE TEXTS")
            print("=" * 80)
            for text, ids in sorted(duplicates.items(), key=lambda x: -len(x[1])):
                preview = text[:70] + "..." if len(text) > 70 else text
                print(f"\n({len(ids)}x) {preview}")
                for sample_id, shard in ids:
                    print(f"  - {sample_id} in {shard}")
    else:
        print("✅ No duplicates found! All text fields are unique.")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Find duplicate text fields in unified dataset")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all duplicate texts (not just top 20)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(UNIFIED_DIR),
        help="Directory containing shard files",
    )

    args = parser.parse_args()

    data_dir = Path(args.dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    analyze_duplicates(data_dir, show_all=args.show_all)


if __name__ == "__main__":
    main()

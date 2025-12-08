#!/usr/bin/env python3
"""
Migrate Sentiment from 3-class to 5-class Schema (IN-PLACE)

Converts old sentiment values to new 5-class schema:
- positive → very_positive (if strong emotions) or positive
- negative → very_negative (if strong emotions) or negative
- neutral → neutral
- mixed → neutral

Edits files IN-PLACE (no new folders).

Usage:
    python migrate_sentiment_3to5.py --dry-run   # Preview changes
    python migrate_sentiment_3to5.py             # Apply changes
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# Directories to process
DATA_DIRS = [
    Path("D:/Modeling_studio/data/familyos/unified/output_synthetic"),
    Path("D:/Modeling_studio/data/familyos/unified/output"),
]

# Emotions that indicate STRONG positive sentiment → very_positive
STRONG_POSITIVE_EMOTIONS = {
    "excitement",
    "joy",
    "love",
    "celebration",
    "gratitude",
    "pride",
    "contentment",
    "hope",
    "relief",
    "admiration",
    "amusement",
    "togetherness",
    "belonging",
    "parental_pride",
    "playfulness",
    "warmth",
}

# Emotions that indicate STRONG negative sentiment → very_negative
STRONG_NEGATIVE_EMOTIONS = {
    "anger",
    "fear",
    "disgust",
    "grief",
    "emptiness",
    "overwhelmed",
    "remorse",
    "homesickness",
    "longing",
    "parental_guilt",
}

# Valid 5-class sentiments
VALID_5CLASS = {"very_positive", "positive", "neutral", "negative", "very_negative"}


def migrate_sentiment(sample: dict) -> tuple[str, str]:
    """
    Migrate a sample's sentiment from 3-class to 5-class.

    Returns: (old_sentiment, new_sentiment)
    """
    tasks = sample.get("tasks", {})
    old_sentiment = tasks.get("sentiment", "neutral")
    emotions = set(tasks.get("emotions", []))

    # Already extreme 5-class? Skip (only very_positive/very_negative are "done")
    if old_sentiment in {"very_positive", "very_negative"}:
        return old_sentiment, old_sentiment

    # Map 3-class to 5-class using emotion context
    if old_sentiment == "positive":
        # Check if any strong positive emotions
        if emotions & STRONG_POSITIVE_EMOTIONS:
            new_sentiment = "very_positive"
        else:
            new_sentiment = "positive"

    elif old_sentiment == "negative":
        # Check if any strong negative emotions
        if emotions & STRONG_NEGATIVE_EMOTIONS:
            new_sentiment = "very_negative"
        else:
            new_sentiment = "negative"

    elif old_sentiment == "neutral":
        new_sentiment = "neutral"

    elif old_sentiment == "mixed":
        # Mixed → neutral (safest mapping)
        new_sentiment = "neutral"

    else:
        # Unknown → neutral
        new_sentiment = "neutral"

    return old_sentiment, new_sentiment


def process_shard(shard_path: Path, dry_run: bool = True) -> dict:
    """Process a single shard file in-place."""
    stats = {
        "total": 0,
        "changed": 0,
        "already_valid": 0,
        "migrations": Counter(),  # old → new
    }

    # Read all lines
    with open(shard_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            sample = json.loads(line)
            stats["total"] += 1

            old_sent, new_sent = migrate_sentiment(sample)

            if old_sent == new_sent and old_sent in VALID_5CLASS:
                stats["already_valid"] += 1
            elif old_sent != new_sent:
                stats["changed"] += 1
                stats["migrations"][(old_sent, new_sent)] += 1
                # Update the sample
                sample["tasks"]["sentiment"] = new_sent

            new_lines.append(json.dumps(sample, ensure_ascii=False))

        except json.JSONDecodeError:
            # Keep invalid lines as-is
            new_lines.append(line)

    # Write back if not dry run
    if not dry_run and stats["changed"] > 0:
        with open(shard_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate sentiment 3-class → 5-class")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    total_stats = {
        "total": 0,
        "changed": 0,
        "already_valid": 0,
        "migrations": Counter(),
        "shards": 0,
    }

    print("=" * 60)
    print("SENTIMENT MIGRATION: 3-class → 5-class")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (editing in-place)'}")
    print()

    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            print(f"⚠️  Directory not found: {data_dir}")
            continue

        shards = sorted(data_dir.glob("shard_*.jsonl"))
        print(f"📁 {data_dir.name}: {len(shards)} shards")

        for shard in shards:
            stats = process_shard(shard, dry_run=args.dry_run)

            total_stats["total"] += stats["total"]
            total_stats["changed"] += stats["changed"]
            total_stats["already_valid"] += stats["already_valid"]
            total_stats["migrations"].update(stats["migrations"])
            total_stats["shards"] += 1

            if stats["changed"] > 0:
                print(f"  {shard.name}: {stats['changed']}/{stats['total']} changed")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:    {total_stats['total']:,}")
    print(f"Already valid:    {total_stats['already_valid']:,}")
    print(f"Changed:          {total_stats['changed']:,}")
    print(f"Shards processed: {total_stats['shards']}")
    print()

    print("Migration breakdown:")
    for (old, new), count in sorted(total_stats["migrations"].items(), key=lambda x: -x[1]):
        print(f"  {old:15} → {new:15}: {count:,}")

    if args.dry_run:
        print()
        print("🔸 This was a DRY RUN. No files were modified.")
        print("🔸 Run without --dry-run to apply changes.")
    else:
        print()
        print("✅ Migration complete. Files edited in-place.")


if __name__ == "__main__":
    main()

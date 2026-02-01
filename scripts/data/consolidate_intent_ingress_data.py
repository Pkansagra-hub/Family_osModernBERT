"""
Consolidate Intent and Ingress Data for SOTA Head Training

This script consolidates data from multiple sources:
- Gold (manually annotated, high quality)
- Silver (synthetic, balanced)
- Unified (multi-task, large scale)

Outputs:
- data/processed/intent_unified/ - Intent training data (5000-sample shards)
- data/processed/ingress_unified/ - Ingress training data (5000-sample shards)

Features:
- Converts numeric intent labels to strings
- Balances datasets by undersampling majority classes
- Creates train/val splits (90/10)
- Generates shards of 5000 samples each
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Label mappings
INTENT_ID_TO_NAME = {
    0: "log_memory",
    1: "query_memory",
    2: "set_reminder",
    3: "express_feeling",
    4: "seek_advice",
    5: "share_news",
    6: "reflect",
    7: "other",
}

INTENT_NAME_TO_ID = {v: k for k, v in INTENT_ID_TO_NAME.items()}

INGRESS_LABELS = [
    "DIARY",
    "TASK",
    "HEALTH",
    "FINANCE",
    "RELATIONSHIP",
    "WORK",
    "META",
    "MEMORY",
    "PLANNING",
    "CELEBRATION",
    "CONCERN",
    "GRATITUDE",
]

# Paths
BASE_DIR = Path("d:/Modeling_studio")
DATA_DIR = BASE_DIR / "data"

# Input paths
INTENT_GOLD_TRAIN = DATA_DIR / "familyos/intents/gold/train.jsonl"
INTENT_GOLD_VAL = DATA_DIR / "familyos/intents/gold/validation.jsonl"
INTENT_SILVER_TRAIN = DATA_DIR / "familyos/intents/silver/train.jsonl"

INGRESS_GOLD_TRAIN = DATA_DIR / "familyos/ingress/gold/train.jsonl"
INGRESS_GOLD_VAL = DATA_DIR / "familyos/ingress/gold/validation.jsonl"
INGRESS_SILVER_TRAIN = DATA_DIR / "familyos/ingress/silver/train.jsonl"

UNIFIED_DIR = DATA_DIR / "familyos/unified/output_healed_merged"

# Output paths
INTENT_OUTPUT_DIR = DATA_DIR / "processed/intent_unified"
INGRESS_OUTPUT_DIR = DATA_DIR / "processed/ingress_unified"

SHARD_SIZE = 5000
VAL_RATIO = 0.1
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_intent_data() -> list[dict[str, Any]]:
    """Load all intent data from gold, silver, and unified sources."""
    all_records = []
    seen_texts = set()

    # Load gold data (numeric labels)
    print("Loading intent gold data...")
    for path in [INTENT_GOLD_TRAIN, INTENT_GOLD_VAL]:
        if path.exists():
            for record in load_jsonl(path):
                text = record["text"].strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    label_id = record["label"]
                    label_name = INTENT_ID_TO_NAME.get(label_id, "other")
                    all_records.append({
                        "text": text,
                        "intent": label_name,
                        "source": "gold",
                    })

    # Load silver data (numeric labels)
    print("Loading intent silver data...")
    if INTENT_SILVER_TRAIN.exists():
        for record in load_jsonl(INTENT_SILVER_TRAIN):
            text = record["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                label_id = record["label"]
                label_name = INTENT_ID_TO_NAME.get(label_id, "other")
                all_records.append({
                    "text": text,
                    "intent": label_name,
                    "source": "silver",
                })

    # Load unified data (string labels)
    print("Loading intent from unified data...")
    for shard_path in sorted(UNIFIED_DIR.glob("shard_*.jsonl")):
        for record in load_jsonl(shard_path):
            text = record["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                tasks = record.get("tasks", {})
                intent = tasks.get("intent", "other")
                if intent and intent in INTENT_NAME_TO_ID:
                    all_records.append({
                        "text": text,
                        "intent": intent,
                        "source": "unified",
                    })

    return all_records


def load_ingress_data() -> list[dict[str, Any]]:
    """Load all ingress data from gold, silver, and unified sources."""
    all_records = []
    seen_texts = set()

    # Load gold data (string labels)
    print("Loading ingress gold data...")
    for path in [INGRESS_GOLD_TRAIN, INGRESS_GOLD_VAL]:
        if path.exists():
            for record in load_jsonl(path):
                text = record["text"].strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    label = record["label"]
                    if label in INGRESS_LABELS:
                        all_records.append({
                            "text": text,
                            "ingress": label,
                            "source": "gold",
                        })

    # Load silver data (string labels)
    print("Loading ingress silver data...")
    if INGRESS_SILVER_TRAIN.exists():
        for record in load_jsonl(INGRESS_SILVER_TRAIN):
            text = record["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                label = record["label"]
                if label in INGRESS_LABELS:
                    all_records.append({
                        "text": text,
                        "ingress": label,
                        "source": "silver",
                    })

    # Load unified data (string labels)
    print("Loading ingress from unified data...")
    for shard_path in sorted(UNIFIED_DIR.glob("shard_*.jsonl")):
        for record in load_jsonl(shard_path):
            text = record["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                tasks = record.get("tasks", {})
                ingress = tasks.get("ingress")
                if ingress and ingress in INGRESS_LABELS:
                    all_records.append({
                        "text": text,
                        "ingress": ingress,
                        "source": "unified",
                    })

    return all_records


def balance_dataset(
    records: list[dict[str, Any]],
    label_key: str,
    target_per_class: int | None = None,
) -> list[dict[str, Any]]:
    """
    Balance dataset by undersampling majority classes.

    Args:
        records: List of data records
        label_key: Key to use for label ("intent" or "ingress")
        target_per_class: Target samples per class. If None, use median count.

    Returns:
        Balanced list of records
    """
    # Group by label
    by_label = defaultdict(list)
    for record in records:
        label = record[label_key]
        by_label[label].append(record)

    # Compute target count
    counts = [len(recs) for recs in by_label.values()]
    if target_per_class is None:
        # Use median to balance - not too aggressive
        target_per_class = sorted(counts)[len(counts) // 2]

    print(f"  Balancing {label_key}: target={target_per_class:,} per class")
    print(f"  Original distribution:")
    for label, recs in sorted(by_label.items(), key=lambda x: -len(x[1])):
        print(f"    {label}: {len(recs):,}")

    # Sample from each class
    balanced = []
    for label, recs in by_label.items():
        if len(recs) <= target_per_class:
            # Keep all if below target
            balanced.extend(recs)
        else:
            # Undersample to target
            random.shuffle(recs)
            balanced.extend(recs[:target_per_class])

    random.shuffle(balanced)

    # Print new distribution
    new_by_label = defaultdict(int)
    for record in balanced:
        new_by_label[record[label_key]] += 1

    print(f"  Balanced distribution:")
    for label, count in sorted(new_by_label.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count:,}")

    return balanced


def create_shards(
    records: list[dict[str, Any]],
    output_dir: Path,
    shard_size: int = SHARD_SIZE,
    val_ratio: float = VAL_RATIO,
) -> dict[str, int]:
    """
    Create train/val shards from records.

    Returns:
        Dictionary with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(records)
    val_size = int(len(records) * val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]

    # Create train shards
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    num_train_shards = (len(train_records) + shard_size - 1) // shard_size
    for i in range(num_train_shards):
        start = i * shard_size
        end = min(start + shard_size, len(train_records))
        shard_records = train_records[start:end]
        shard_path = train_dir / f"shard_{i:04d}.jsonl"
        save_jsonl(shard_records, shard_path)

    # Create val shards
    val_dir = output_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    num_val_shards = (len(val_records) + shard_size - 1) // shard_size
    for i in range(num_val_shards):
        start = i * shard_size
        end = min(start + shard_size, len(val_records))
        shard_records = val_records[start:end]
        shard_path = val_dir / f"shard_{i:04d}.jsonl"
        save_jsonl(shard_records, shard_path)

    # Also save combined files for convenience
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(val_records, output_dir / "val.jsonl")

    return {
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_shards": num_train_shards,
        "val_shards": num_val_shards,
    }


def main() -> None:
    """Main function to consolidate and balance datasets."""
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("INTENT DATA CONSOLIDATION")
    print("=" * 60)

    # Load and process intent data
    intent_records = load_intent_data()
    print(f"\nTotal intent records (deduplicated): {len(intent_records):,}")

    # Balance intent data
    print("\nBalancing intent data...")
    intent_balanced = balance_dataset(
        intent_records,
        label_key="intent",
        target_per_class=50000,  # Cap at 50K per class
    )
    print(f"\nBalanced intent records: {len(intent_balanced):,}")

    # Create shards
    print(f"\nCreating intent shards (size={SHARD_SIZE})...")
    intent_stats = create_shards(intent_balanced, INTENT_OUTPUT_DIR)
    print(f"  Train: {intent_stats['train_total']:,} records in {intent_stats['train_shards']} shards")
    print(f"  Val: {intent_stats['val_total']:,} records in {intent_stats['val_shards']} shards")

    print("\n" + "=" * 60)
    print("INGRESS DATA CONSOLIDATION")
    print("=" * 60)

    # Load and process ingress data
    ingress_records = load_ingress_data()
    print(f"\nTotal ingress records (deduplicated): {len(ingress_records):,}")

    # Balance ingress data
    print("\nBalancing ingress data...")
    ingress_balanced = balance_dataset(
        ingress_records,
        label_key="ingress",
        target_per_class=40000,  # Cap at 40K per class
    )
    print(f"\nBalanced ingress records: {len(ingress_balanced):,}")

    # Create shards
    print(f"\nCreating ingress shards (size={SHARD_SIZE})...")
    ingress_stats = create_shards(ingress_balanced, INGRESS_OUTPUT_DIR)
    print(f"  Train: {ingress_stats['train_total']:,} records in {ingress_stats['train_shards']} shards")
    print(f"  Val: {ingress_stats['val_total']:,} records in {ingress_stats['val_shards']} shards")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nIntent dataset: {INTENT_OUTPUT_DIR}")
    print(f"  - train.jsonl: {intent_stats['train_total']:,} records")
    print(f"  - val.jsonl: {intent_stats['val_total']:,} records")
    print(f"  - train/shard_*.jsonl: {intent_stats['train_shards']} shards")
    print(f"  - val/shard_*.jsonl: {intent_stats['val_shards']} shards")

    print(f"\nIngress dataset: {INGRESS_OUTPUT_DIR}")
    print(f"  - train.jsonl: {ingress_stats['train_total']:,} records")
    print(f"  - val.jsonl: {ingress_stats['val_total']:,} records")
    print(f"  - train/shard_*.jsonl: {ingress_stats['train_shards']} shards")
    print(f"  - val/shard_*.jsonl: {ingress_stats['val_shards']} shards")

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Data Healer: Fix predicates not in RELATION_LABELS schema.

Mappings:
- family_of (23 samples) → no_relation (too vague/ambiguous)
- child_in_law_of (50 samples) → REMOVE (legitimate but not in schema, only 50 samples)

Run: python scripts/heal_relation_predicates.py
"""

import json
from pathlib import Path
from collections import Counter

# Mapping rules
PREDICATE_MAPPINGS = {
    "family_of": "no_relation",  # Too vague to classify
    # child_in_law_of will be removed (not mapped)
}

PREDICATES_TO_REMOVE = {"child_in_law_of"}


def heal_shard(shard_path: Path, dry_run: bool = True) -> dict:
    """Process a single shard and fix predicates.

    Args:
        shard_path: Path to the JSONL shard file.
        dry_run: If True, don't write changes, just report what would happen.

    Returns:
        Stats dict with counts of changes.
    """
    stats = Counter()
    healed_records = []

    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            relations = record.get("tasks", {}).get("relations", [])

            healed_relations = []
            for rel in relations:
                pred = rel.get("predicate", "")

                if pred in PREDICATES_TO_REMOVE:
                    stats[f"removed_{pred}"] += 1
                    continue  # Skip this relation

                if pred in PREDICATE_MAPPINGS:
                    old_pred = pred
                    new_pred = PREDICATE_MAPPINGS[pred]
                    rel["predicate"] = new_pred
                    stats[f"mapped_{old_pred}_to_{new_pred}"] += 1

                healed_relations.append(rel)

            record["tasks"]["relations"] = healed_relations
            healed_records.append(record)
            stats["total_records"] += 1

    if not dry_run:
        with open(shard_path, "w", encoding="utf-8") as f:
            for record in healed_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["file_written"] = 1

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Heal relation predicates in FamilyOS data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Don't write changes, just report (default: True)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually apply the changes")
    args = parser.parse_args()

    dry_run = not args.apply

    # Heal BOTH data directories
    data_dirs = [
        Path("data/familyos/unified/output_synthetic_healed"),
        Path("data/familyos/unified/output_healed"),
    ]

    shards = []
    for data_dir in data_dirs:
        if data_dir.exists():
            shards.extend(sorted(data_dir.glob("shard_*.jsonl")))

    print(f"Found {len(shards)} shards across {len(data_dirs)} directories")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING CHANGES'}")
    print()

    total_stats = Counter()

    for shard in shards:
        stats = heal_shard(shard, dry_run=dry_run)
        total_stats.update(stats)

        # Print progress for shards with changes
        changes = sum(v for k, v in stats.items() if k.startswith(("mapped_", "removed_")))
        if changes > 0:
            print(f"  {shard.parent.name}/{shard.name}: {changes} changes")

    print()
    print("=== SUMMARY ===")
    print(f"Total records processed: {total_stats['total_records']}")

    for key, count in sorted(total_stats.items()):
        if key.startswith("mapped_"):
            print(f"  {key}: {count}")
        elif key.startswith("removed_"):
            print(f"  {key}: {count}")

    if dry_run:
        print()
        print("This was a DRY RUN. To apply changes, run with --apply")
    else:
        print()
        print(f"Changes applied to {total_stats.get('file_written', 0)} files")


if __name__ == "__main__":
    main()

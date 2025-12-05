"""
Remove Duplicate Text Fields from Unified Dataset

Removes duplicate samples keeping only the first occurrence of each unique text.

Usage:
    python remove_duplicates.py --dry-run    # Preview changes
    python remove_duplicates.py --backup     # Create backup first
    python remove_duplicates.py --execute    # Remove duplicates
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Path to unified data
UNIFIED_DIR = Path("D:/Modeling_studio/data/familyos/unified/output")
BACKUP_DIR = Path("D:/Modeling_studio/data/familyos/unified/output_dedup_backup")


class Deduplicator:
    """Remove duplicate text fields from dataset."""

    def __init__(self, data_dir: Path = UNIFIED_DIR):
        self.data_dir = data_dir
        self.seen_texts = set()
        self.stats = {
            "total_samples": 0,
            "unique_samples": 0,
            "duplicates_removed": 0,
        }

    def process_all_shards(self, dry_run: bool = True):
        """Process all shards and remove duplicates."""
        shards = sorted(self.data_dir.glob("shard_*.jsonl"))

        if not shards:
            print(f"No shard files found in {self.data_dir}")
            return

        print(f"\n{'='*80}")
        print(f"DEDUPLICATION {'(DRY RUN)' if dry_run else '(EXECUTING)'}")
        print(f"{'='*80}")
        print(f"Found {len(shards)} shard files")
        print()

        # First pass: collect all samples and track duplicates
        all_samples = []

        for shard_path in shards:
            print(f"Reading {shard_path.name}...", end=" ")

            with open(shard_path, encoding="utf-8") as f:
                shard_samples = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                        self.stats["total_samples"] += 1
                        shard_samples.append(sample)
                    except json.JSONDecodeError:
                        continue

            print(f"✓ ({len(shard_samples)} samples)")
            all_samples.extend(shard_samples)

        # Second pass: deduplicate
        print()
        print("Deduplicating...")
        unique_samples = []

        for sample in all_samples:
            text = sample.get("text", "").strip()

            if not text:
                continue

            if text not in self.seen_texts:
                self.seen_texts.add(text)
                unique_samples.append(sample)
                self.stats["unique_samples"] += 1
            else:
                self.stats["duplicates_removed"] += 1

        # Third pass: write deduplicated data
        if not dry_run:
            print()
            print("Writing deduplicated shards...")

            shard_size = 5000
            for i, start_idx in enumerate(range(0, len(unique_samples), shard_size)):
                shard_samples = unique_samples[start_idx : start_idx + shard_size]
                shard_path = self.data_dir / f"shard_{i:04d}.jsonl"

                with open(shard_path, "w", encoding="utf-8") as f:
                    for sample in shard_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                print(f"  {shard_path.name}: {len(shard_samples)} samples")

            # Remove old extra shards if any
            new_shard_count = (len(unique_samples) + shard_size - 1) // shard_size
            for shard_path in self.data_dir.glob("shard_*.jsonl"):
                shard_num = int(shard_path.stem.split("_")[1])
                if shard_num >= new_shard_count:
                    print(f"  Removing old {shard_path.name}")
                    shard_path.unlink()

        # Print summary
        print()
        print(f"{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total samples read: {self.stats['total_samples']:,}")
        print(f"Unique samples kept: {self.stats['unique_samples']:,}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']:,}")
        print(
            f"Deduplication rate: {100 * self.stats['duplicates_removed'] / self.stats['total_samples']:.2f}%"
        )
        print(f"{'='*80}")

        if dry_run:
            print("\n⚠️  DRY RUN - No files modified")
            print("Run with --execute to apply changes")
        else:
            print("\n✅ Deduplication complete!")
            print(
                f"Dataset reduced from {self.stats['total_samples']:,} → {self.stats['unique_samples']:,} samples"
            )


def create_backup():
    """Create backup of unified data."""
    if BACKUP_DIR.exists():
        print(f"Backup already exists at: {BACKUP_DIR}")
        response = input("Overwrite existing backup? (yes/no): ")
        if response.lower() != "yes":
            print("Backup cancelled.")
            return False
        shutil.rmtree(BACKUP_DIR)

    print(f"\nCreating backup: {UNIFIED_DIR} -> {BACKUP_DIR}")
    shutil.copytree(UNIFIED_DIR, BACKUP_DIR)

    # Count files
    backup_files = list(BACKUP_DIR.glob("shard_*.jsonl"))
    total_samples = 0
    for f in backup_files:
        with open(f, encoding="utf-8") as file:
            total_samples += sum(1 for _ in file)

    print(f"✅ Backup created: {len(backup_files)} shards, {total_samples:,} samples")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate text fields from unified dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before deduplication",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply deduplication to files",
    )

    args = parser.parse_args()

    if not UNIFIED_DIR.exists():
        print(f"Error: Data directory not found: {UNIFIED_DIR}")
        return

    # Handle backup
    if args.backup:
        if not create_backup():
            return
        print()

    # Determine mode
    if args.execute:
        response = input("\n⚠️  This will modify files and remove duplicates. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
        dry_run = False
    else:
        dry_run = True

    # Run deduplication
    dedup = Deduplicator()
    dedup.process_all_shards(dry_run=dry_run)

    if dry_run and not args.backup:
        print("\nNext steps:")
        print("  1. python remove_duplicates.py --backup     # Create backup")
        print("  2. python remove_duplicates.py --execute    # Remove duplicates")


if __name__ == "__main__":
    main()

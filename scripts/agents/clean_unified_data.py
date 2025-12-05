"""
Clean Unified Data - Fix Schema Violations

Fixes the 68K existing unified samples by:
1. Mapping/removing wrong NER labels
2. Mapping/removing wrong temporal labels
3. Standardizing relation predicates
4. Validating all annotations

Usage:
    python clean_unified_data.py --dry-run          # Preview changes
    python clean_unified_data.py --execute          # Apply changes
    python clean_unified_data.py --backup           # Create backup first
"""

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
UNIFIED_DIR = BASE_DIR / "data" / "familyos" / "unified" / "output"
BACKUP_DIR = BASE_DIR / "data" / "familyos" / "unified" / "output_backup"

# Valid schemas
VALID_NER_LABELS = {
    "PERSON",
    "KINSHIP",
    "NICKNAME",
    "PET",
    "HOME_LOC",
    "FAMILY_EVENT",
    "ROUTINE",
    "TRADITION",
    "MILESTONE",
    "HEIRLOOM",
}

VALID_TEMPORAL_LABELS = {"DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"}

VALID_RELATIONS = {
    "no_relation",
    "parent_of",
    "child_of",
    "spouse_of",
    "sibling_of",
    "grandparent_of",
    "grandchild_of",
    "aunt_uncle_of",
    "niece_nephew_of",
    "cousin_of",
    "pet_of",
    "friend_of",
    "colleague_of",
    "lives_at",
    "owns",
}

# Label mapping rules
NER_LABEL_MAPPING = {
    # Temporal labels → Remove (should be in temporal task)
    "TIME": None,
    "DATE_ABS": None,
    "DATE_REL": None,
    "DURATION": None,
    "FREQUENCY": None,
    "AGE": None,
    "TEMPORAL": None,
    "TEMPORAL_TYPE": None,
    # Relation concepts → Remove (should be in relations)
    "RELATIONSHIP": None,
    "FRIEND": None,
    "COLLEAGUE": None,
    # Generic/unclear → Remove
    "OTHER": None,
    "other": None,
    "MEMORY": None,
    "EMOTION": None,
    "TASK": None,
    "PARENTAL_GUILT": None,
    "EVENT": None,
    "OTHER_LOC": None,
    "INGRESS": None,
    "PLANNING": None,
    "HEALTH": None,
    "WORK": None,
    "FINANCE": None,
    "FOOD": None,
    "CELEBRATION": None,
    "DIARY": None,
}

TEMPORAL_LABEL_MAPPING = {
    # NER entities → Remove (should be in NER)
    "ROUTINE": None,
    "FAMILY_EVENT": None,
    "MILESTONE": None,
    "HOME_LOC": None,
    "TRADITION": None,
    "TASK": None,
    "MEMORY": None,
    "TEMPORAL_TYPE": None,
}

RELATION_PREDICATE_MAPPING = {
    # Normalize variants
    "parent_offspring": "parent_of",
    "parental_of": "parent_of",
    "family_of": None,  # Too generic
    "family_member_of": None,  # Too generic
    "relative_of": None,  # Too generic
    "kinship": None,  # Too generic
    "KINSHIP": None,  # Wrong case
    "uncle_of": "aunt_uncle_of",
    "aunt_of": "aunt_uncle_of",
    "nephew_of": "niece_nephew_of",
    "owned_by": None,  # Reverse of owns, remove
    "protects": None,  # Not in schema
    "cares_for": None,  # Not in schema
    "provides_care_for": None,  # Not in schema
    "belongs_to": None,  # Too generic
    "celebration_of": None,  # Not a relation
}


class UnifiedDataCleaner:
    """Clean and validate unified dataset."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.stats = {
            "total_samples": 0,
            "ner_fixes": 0,
            "temporal_fixes": 0,
            "relation_fixes": 0,
            "ner_labels_removed": Counter(),
            "temporal_labels_removed": Counter(),
            "relation_predicates_removed": Counter(),
            "relation_predicates_mapped": Counter(),
        }

    def clean_ner(self, sample: dict) -> dict:
        """Clean NER annotations."""
        tasks = sample.get("tasks", {})
        ner_entities = tasks.get("ner_family", [])

        if not ner_entities:
            return sample

        cleaned = []
        for entity in ner_entities:
            label = entity.get("label", "")

            # Check if label needs mapping
            if label in NER_LABEL_MAPPING:
                mapped = NER_LABEL_MAPPING[label]
                if mapped is None:
                    self.stats["ner_labels_removed"][label] += 1
                    continue
                else:
                    entity["label"] = mapped
                    self.stats["ner_fixes"] += 1

            # Check if label is valid
            if label not in VALID_NER_LABELS:
                self.stats["ner_labels_removed"][label] += 1
                continue

            cleaned.append(entity)

        if len(cleaned) != len(ner_entities):
            self.stats["ner_fixes"] += 1
            tasks["ner_family"] = cleaned

        return sample

    def clean_temporal(self, sample: dict) -> dict:
        """Clean temporal annotations."""
        tasks = sample.get("tasks", {})
        temporal_entities = tasks.get("temporal", [])

        if not temporal_entities:
            return sample

        cleaned = []
        for entity in temporal_entities:
            label = entity.get("label", "")

            # Check if label needs mapping
            if label in TEMPORAL_LABEL_MAPPING:
                mapped = TEMPORAL_LABEL_MAPPING[label]
                if mapped is None:
                    self.stats["temporal_labels_removed"][label] += 1
                    continue
                else:
                    entity["label"] = mapped
                    self.stats["temporal_fixes"] += 1

            # Check if label is valid
            if label not in VALID_TEMPORAL_LABELS:
                self.stats["temporal_labels_removed"][label] += 1
                continue

            cleaned.append(entity)

        if len(cleaned) != len(temporal_entities):
            self.stats["temporal_fixes"] += 1
            tasks["temporal"] = cleaned

        return sample

    def clean_relations(self, sample: dict) -> dict:
        """Clean relation annotations."""
        tasks = sample.get("tasks", {})
        relations = tasks.get("relations", [])

        if not relations:
            return sample

        cleaned = []
        for relation in relations:
            predicate = relation.get("predicate", "")

            # Check if predicate needs mapping
            if predicate in RELATION_PREDICATE_MAPPING:
                mapped = RELATION_PREDICATE_MAPPING[predicate]
                if mapped is None:
                    self.stats["relation_predicates_removed"][predicate] += 1
                    continue
                else:
                    relation["predicate"] = mapped
                    self.stats["relation_predicates_mapped"][f"{predicate}→{mapped}"] += 1
                    self.stats["relation_fixes"] += 1

            # Check if predicate is valid
            if predicate not in VALID_RELATIONS:
                self.stats["relation_predicates_removed"][predicate] += 1
                continue

            cleaned.append(relation)

        if len(cleaned) != len(relations):
            self.stats["relation_fixes"] += 1
            tasks["relations"] = cleaned

        return sample

    def clean_sample(self, sample: dict) -> dict:
        """Clean all annotations in a sample."""
        sample = self.clean_ner(sample)
        sample = self.clean_temporal(sample)
        sample = self.clean_relations(sample)
        return sample

    def process_shard(self, shard_path: Path) -> list[dict]:
        """Process a single shard file."""
        cleaned_samples = []

        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    self.stats["total_samples"] += 1
                    cleaned = self.clean_sample(sample)
                    cleaned_samples.append(cleaned)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line in {shard_path.name}")
                    continue

        return cleaned_samples

    def run(self) -> None:
        """Run the cleaning process."""
        shard_files = sorted(UNIFIED_DIR.glob("shard_*.jsonl"))

        if not shard_files:
            logger.error(f"No shard files found in {UNIFIED_DIR}")
            return

        logger.info(f"Found {len(shard_files)} shard files")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}")

        for shard_path in shard_files:
            logger.info(f"Processing {shard_path.name}...")
            cleaned_samples = self.process_shard(shard_path)

            if not self.dry_run:
                # Write cleaned samples back
                with open(shard_path, "w", encoding="utf-8") as f:
                    for sample in cleaned_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                logger.info(f"  ✓ Wrote {len(cleaned_samples)} samples")
            else:
                logger.info(f"  Would write {len(cleaned_samples)} samples")

        self.print_stats()

    def print_stats(self) -> None:
        """Print cleaning statistics."""
        print("\n" + "=" * 70)
        print("CLEANING STATISTICS")
        print("=" * 70)

        print(f"\n📊 Total samples processed: {self.stats['total_samples']:,}")
        print(f"   Samples with NER fixes: {self.stats['ner_fixes']:,}")
        print(f"   Samples with temporal fixes: {self.stats['temporal_fixes']:,}")
        print(f"   Samples with relation fixes: {self.stats['relation_fixes']:,}")

        if self.stats["ner_labels_removed"]:
            print(f"\n🗑️  NER Labels Removed:")
            for label, count in self.stats["ner_labels_removed"].most_common():
                print(f"   {label:20s} {count:6,} occurrences")

        if self.stats["temporal_labels_removed"]:
            print(f"\n🗑️  Temporal Labels Removed:")
            for label, count in self.stats["temporal_labels_removed"].most_common():
                print(f"   {label:20s} {count:6,} occurrences")

        if self.stats["relation_predicates_removed"]:
            print(f"\n🗑️  Relation Predicates Removed:")
            for pred, count in self.stats["relation_predicates_removed"].most_common():
                print(f"   {pred:20s} {count:6,} occurrences")

        if self.stats["relation_predicates_mapped"]:
            print(f"\n🔄 Relation Predicates Mapped:")
            for mapping, count in self.stats["relation_predicates_mapped"].most_common():
                print(f"   {mapping:30s} {count:6,} occurrences")

        print("=" * 70)

        if self.dry_run:
            print("\n⚠️  DRY RUN - No changes were made")
            print("Run with --execute to apply changes")
        else:
            print("\n✅ Changes applied successfully!")


def create_backup() -> None:
    """Create backup of unified data."""
    if BACKUP_DIR.exists():
        logger.warning(f"Backup already exists at {BACKUP_DIR}")
        response = input("Overwrite existing backup? (y/N): ")
        if response.lower() != "y":
            logger.info("Backup cancelled")
            return
        shutil.rmtree(BACKUP_DIR)

    logger.info(f"Creating backup at {BACKUP_DIR}...")
    shutil.copytree(UNIFIED_DIR, BACKUP_DIR)

    # Count files
    backup_files = list(BACKUP_DIR.glob("shard_*.jsonl"))
    total_samples = 0
    for shard in backup_files:
        with open(shard, encoding="utf-8") as f:
            total_samples += sum(1 for _ in f)

    logger.info(f"✓ Backup created: {len(backup_files)} shards, {total_samples:,} samples")


def main():
    parser = argparse.ArgumentParser(description="Clean unified dataset")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes to data")
    parser.add_argument("--backup", action="store_true", help="Create backup before cleaning")

    args = parser.parse_args()

    # Default to dry-run if neither specified
    if not args.execute and not args.dry_run and not args.backup:
        args.dry_run = True

    if args.backup:
        create_backup()
        if not args.execute and not args.dry_run:
            return

    cleaner = UnifiedDataCleaner(dry_run=args.dry_run)
    cleaner.run()


if __name__ == "__main__":
    main()

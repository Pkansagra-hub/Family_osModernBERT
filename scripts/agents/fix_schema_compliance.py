"""
Schema Compliance Fixer for Generated Data

Fixes all schema violations in output/ and output_synthetic/ folders:
1. SENTIMENT: Convert 4-class (with "mixed") → 5-class scale
2. EMOTIONS: Remove invalid emotions (responsibility, planning, curiosity, etc.)
3. NER_FAMILY: Remove invalid NER types (TIME, DATE_REL, AGE, etc.)
4. TEMPORAL: Remove invalid temporal types (MILESTONE, ROUTINE, etc.)
5. RELATIONS: Normalize relation predicates (nephew_niece_of → niece_nephew_of)

Usage:
    python fix_schema_compliance.py --dry-run     # Preview changes
    python fix_schema_compliance.py --execute     # Apply fixes
    python fix_schema_compliance.py --stats       # Show current violations
"""

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# AUTHORITATIVE SCHEMAS (from registry_v3.py and labels.py)
# =============================================================================

# 44 FamilyOS Emotions (EMOTIONS_FAMILYOS_LABELS)
VALID_EMOTIONS = {
    # Core Emotions (8)
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "love",
    "disgust",
    # Positive Emotions (12)
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "optimism",
    "pride",
    "relief",
    "contentment",
    "hope",
    "tenderness",
    # Negative Emotions (10)
    "annoyance",
    "disappointment",
    "disapproval",
    "embarrassment",
    "grief",
    "nervousness",
    "remorse",
    "frustration",
    "overwhelmed",
    "emptiness",
    # Family-Specific Emotions (14)
    "nostalgia",
    "protectiveness",
    "togetherness",
    "longing",
    "warmth",
    "playfulness",
    "celebration",
    "belonging",
    "parental_pride",
    "parental_guilt",
    "patience",
    "worry",
    "bittersweet",
    "homesickness",
}

# 5-class Sentiment (SENTIMENT_LABELS from labels.py)
VALID_SENTIMENTS_5CLASS = {"very_negative", "negative", "neutral", "positive", "very_positive"}

# Sentiment mapping: 4-class → 5-class
SENTIMENT_MAPPING = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "mixed": "neutral",  # Map mixed → neutral (most appropriate)
    # Also handle any edge cases
    "very_negative": "very_negative",
    "very_positive": "very_positive",
}

# 10 NER Family Types (NER_FAMILY_LABELS)
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

# 6 Temporal Types (TEMPORAL_LABELS)
VALID_TEMPORAL_LABELS = {"DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"}

# 4 Safety Bands (SAFETY_FAMILYOS_LABELS)
VALID_SAFETY = {"GREEN", "AMBER", "RED", "CRISIS"}

# 8 Intents (INTENT_LABELS)
VALID_INTENTS = {
    "log_memory",
    "query_memory",
    "set_reminder",
    "express_feeling",
    "seek_advice",
    "share_news",
    "reflect",
    "other",
}

# 12 Ingress Domains (INGRESS_LABELS)
VALID_INGRESS = {
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
}

# 15 Relations (RELATION_LABELS)
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

# Relation normalization mapping
RELATION_MAPPING = {
    "nephew_niece_of": "niece_nephew_of",
    "owner_of": "owns",
    "family_of": None,  # Remove - too generic
}


# =============================================================================
# Fixing Functions
# =============================================================================


def fix_emotions(emotions: list) -> tuple[list, int]:
    """
    Fix emotions list - remove invalid emotions.

    Returns:
        (fixed_list, num_removed)
    """
    if not emotions:
        return ["neutral"], 0

    fixed = [e for e in emotions if e in VALID_EMOTIONS]
    removed = len(emotions) - len(fixed)

    # Ensure at least one emotion
    if not fixed:
        fixed = ["neutral"]

    return fixed, removed


def fix_sentiment(sentiment: str) -> tuple[str, bool]:
    """
    Fix sentiment - convert to 5-class scale.

    Returns:
        (fixed_sentiment, was_changed)
    """
    if not sentiment:
        return "neutral", True

    # Already valid 5-class
    if sentiment in VALID_SENTIMENTS_5CLASS:
        return sentiment, False

    # Map using our mapping
    if sentiment in SENTIMENT_MAPPING:
        mapped = SENTIMENT_MAPPING[sentiment]
        return mapped, (sentiment != mapped)

    # Unknown sentiment - default to neutral
    return "neutral", True


def fix_ner_family(ner_list: list) -> tuple[list, int]:
    """
    Fix NER entities - remove invalid labels.

    Returns:
        (fixed_list, num_removed)
    """
    if not ner_list:
        return [], 0

    fixed = []
    for entity in ner_list:
        label = entity.get("label", "")
        if label in VALID_NER_LABELS:
            fixed.append(entity)

    removed = len(ner_list) - len(fixed)
    return fixed, removed


def fix_temporal(temporal_list: list) -> tuple[list, int]:
    """
    Fix temporal expressions - remove invalid labels.

    Returns:
        (fixed_list, num_removed)
    """
    if not temporal_list:
        return [], 0

    fixed = []
    for temp in temporal_list:
        label = temp.get("label", "")
        if label in VALID_TEMPORAL_LABELS:
            fixed.append(temp)

    removed = len(temporal_list) - len(fixed)
    return fixed, removed


def fix_relations(relations_list: list) -> tuple[list, int]:
    """
    Fix relations - normalize predicates, remove invalid.

    Returns:
        (fixed_list, num_removed)
    """
    if not relations_list:
        return [], 0

    fixed = []
    for rel in relations_list:
        pred = rel.get("predicate", "")

        # Normalize if needed
        if pred in RELATION_MAPPING:
            mapped = RELATION_MAPPING[pred]
            if mapped is None:
                continue  # Skip this relation
            pred = mapped
            rel = {**rel, "predicate": pred}

        if pred in VALID_RELATIONS:
            fixed.append(rel)

    removed = len(relations_list) - len(fixed)
    return fixed, removed


def fix_safety(safety: str) -> tuple[str, bool]:
    """
    Fix safety band.

    Returns:
        (fixed_safety, was_changed)
    """
    if safety in VALID_SAFETY:
        return safety, False
    return "GREEN", True


def fix_intent(intent: str) -> tuple[str, bool]:
    """
    Fix intent.

    Returns:
        (fixed_intent, was_changed)
    """
    if intent in VALID_INTENTS:
        return intent, False
    return "other", True


def fix_ingress(ingress: str) -> tuple[str, bool]:
    """
    Fix ingress domain.

    Returns:
        (fixed_ingress, was_changed)
    """
    if ingress in VALID_INGRESS:
        return ingress, False
    return "DIARY", True


def fix_sample(sample: dict) -> tuple[dict, dict]:
    """
    Fix all schema issues in a sample.

    Returns:
        (fixed_sample, changes_made)
    """
    changes = {
        "emotions_removed": 0,
        "sentiment_changed": False,
        "ner_removed": 0,
        "temporal_removed": 0,
        "relations_removed": 0,
        "safety_changed": False,
        "intent_changed": False,
        "ingress_changed": False,
    }

    if "tasks" not in sample:
        return sample, changes

    tasks = sample["tasks"]

    # Fix emotions
    if "emotions" in tasks:
        tasks["emotions"], changes["emotions_removed"] = fix_emotions(tasks["emotions"])

    # Fix sentiment
    if "sentiment" in tasks:
        tasks["sentiment"], changes["sentiment_changed"] = fix_sentiment(tasks["sentiment"])

    # Fix NER
    if "ner_family" in tasks:
        tasks["ner_family"], changes["ner_removed"] = fix_ner_family(tasks["ner_family"])

    # Fix temporal
    if "temporal" in tasks:
        tasks["temporal"], changes["temporal_removed"] = fix_temporal(tasks["temporal"])

    # Fix relations
    if "relations" in tasks:
        tasks["relations"], changes["relations_removed"] = fix_relations(tasks["relations"])

    # Fix safety
    if "safety_familyos" in tasks:
        tasks["safety_familyos"], changes["safety_changed"] = fix_safety(tasks["safety_familyos"])

    # Fix intent
    if "intent" in tasks:
        tasks["intent"], changes["intent_changed"] = fix_intent(tasks["intent"])

    # Fix ingress
    if "ingress" in tasks:
        tasks["ingress"], changes["ingress_changed"] = fix_ingress(tasks["ingress"])

    sample["tasks"] = tasks
    return sample, changes


# =============================================================================
# Statistics Collection
# =============================================================================


def collect_violations(data_dirs: list[Path]) -> dict:
    """Collect all schema violations across directories."""
    violations = {
        "invalid_emotions": Counter(),
        "invalid_sentiments": Counter(),
        "invalid_ner": Counter(),
        "invalid_temporal": Counter(),
        "invalid_relations": Counter(),
        "invalid_safety": Counter(),
        "invalid_intent": Counter(),
        "invalid_ingress": Counter(),
        "total_samples": 0,
        "samples_with_issues": 0,
    }

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        for shard_file in sorted(data_dir.glob("shard_*.jsonl")):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        violations["total_samples"] += 1

                        has_issue = False
                        tasks = sample.get("tasks", {})

                        # Check emotions
                        for e in tasks.get("emotions", []):
                            if e not in VALID_EMOTIONS:
                                violations["invalid_emotions"][e] += 1
                                has_issue = True

                        # Check sentiment
                        sent = tasks.get("sentiment", "")
                        if sent and sent not in VALID_SENTIMENTS_5CLASS:
                            violations["invalid_sentiments"][sent] += 1
                            has_issue = True

                        # Check NER
                        for ner in tasks.get("ner_family", []):
                            label = ner.get("label", "")
                            if label and label not in VALID_NER_LABELS:
                                violations["invalid_ner"][label] += 1
                                has_issue = True

                        # Check temporal
                        for temp in tasks.get("temporal", []):
                            label = temp.get("label", "")
                            if label and label not in VALID_TEMPORAL_LABELS:
                                violations["invalid_temporal"][label] += 1
                                has_issue = True

                        # Check relations
                        for rel in tasks.get("relations", []):
                            pred = rel.get("predicate", "")
                            if pred and pred not in VALID_RELATIONS:
                                violations["invalid_relations"][pred] += 1
                                has_issue = True

                        # Check safety
                        safety = tasks.get("safety_familyos", "")
                        if safety and safety not in VALID_SAFETY:
                            violations["invalid_safety"][safety] += 1
                            has_issue = True

                        # Check intent
                        intent = tasks.get("intent", "")
                        if intent and intent not in VALID_INTENTS:
                            violations["invalid_intent"][intent] += 1
                            has_issue = True

                        # Check ingress
                        ingress = tasks.get("ingress", "")
                        if ingress and ingress not in VALID_INGRESS:
                            violations["invalid_ingress"][ingress] += 1
                            has_issue = True

                        if has_issue:
                            violations["samples_with_issues"] += 1

                    except json.JSONDecodeError:
                        continue

    return violations


def print_violations(violations: dict):
    """Print violation statistics."""
    print("\n" + "=" * 70)
    print("SCHEMA VIOLATION REPORT")
    print("=" * 70)

    print(f"\nTotal samples: {violations['total_samples']:,}")
    print(
        f"Samples with issues: {violations['samples_with_issues']:,} ({violations['samples_with_issues']/max(1,violations['total_samples'])*100:.1f}%)"
    )

    print("\n" + "-" * 70)
    print("INVALID EMOTIONS (should be in 44-class FamilyOS schema):")
    print("-" * 70)
    if violations["invalid_emotions"]:
        for emotion, count in violations["invalid_emotions"].most_common(20):
            print(f"  {emotion:30} {count:>6} occurrences")
    else:
        print("  ✅ None found!")

    print("\n" + "-" * 70)
    print("INVALID SENTIMENTS (should be 5-class: very_neg, neg, neutral, pos, very_pos):")
    print("-" * 70)
    if violations["invalid_sentiments"]:
        for sent, count in violations["invalid_sentiments"].most_common():
            print(f"  {sent:30} {count:>6} occurrences → will map to 5-class")
    else:
        print("  ✅ None found!")

    print("\n" + "-" * 70)
    print("INVALID NER LABELS (should be in 10-type NER_FAMILY schema):")
    print("-" * 70)
    if violations["invalid_ner"]:
        for label, count in violations["invalid_ner"].most_common(20):
            print(f"  {label:30} {count:>6} occurrences → will be REMOVED")
    else:
        print("  ✅ None found!")

    print("\n" + "-" * 70)
    print("INVALID TEMPORAL LABELS (should be in 6-type TEMPORAL schema):")
    print("-" * 70)
    if violations["invalid_temporal"]:
        for label, count in violations["invalid_temporal"].most_common(20):
            print(f"  {label:30} {count:>6} occurrences → will be REMOVED")
    else:
        print("  ✅ None found!")

    print("\n" + "-" * 70)
    print("INVALID RELATION PREDICATES:")
    print("-" * 70)
    if violations["invalid_relations"]:
        for pred, count in violations["invalid_relations"].most_common():
            action = "→ will normalize" if pred in RELATION_MAPPING else "→ will REMOVE"
            print(f"  {pred:30} {count:>6} occurrences {action}")
    else:
        print("  ✅ None found!")

    # Other fields
    for field, display in [
        ("invalid_safety", "SAFETY"),
        ("invalid_intent", "INTENT"),
        ("invalid_ingress", "INGRESS"),
    ]:
        if violations[field]:
            print(f"\n{display} violations:")
            for val, count in violations[field].most_common():
                print(f"  {val:30} {count:>6} occurrences")

    print("\n" + "=" * 70)


# =============================================================================
# Main Processing
# =============================================================================


def process_directory(data_dir: Path, dry_run: bool = True) -> dict:
    """Process all shards in a directory."""
    stats = {
        "files_processed": 0,
        "samples_processed": 0,
        "samples_fixed": 0,
        "emotions_removed": 0,
        "sentiment_changes": 0,
        "ner_removed": 0,
        "temporal_removed": 0,
        "relations_removed": 0,
    }

    if not data_dir.exists():
        logger.warning(f"Directory not found: {data_dir}")
        return stats

    shard_files = sorted(data_dir.glob("shard_*.jsonl"))
    logger.info(f"Processing {len(shard_files)} shards in {data_dir}")

    for shard_file in shard_files:
        fixed_samples = []
        file_changes = 0

        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    fixed_sample, changes = fix_sample(sample)
                    fixed_samples.append(fixed_sample)
                    stats["samples_processed"] += 1

                    # Track changes
                    if any(
                        [
                            changes["emotions_removed"] > 0,
                            changes["sentiment_changed"],
                            changes["ner_removed"] > 0,
                            changes["temporal_removed"] > 0,
                            changes["relations_removed"] > 0,
                        ]
                    ):
                        stats["samples_fixed"] += 1
                        file_changes += 1

                    stats["emotions_removed"] += changes["emotions_removed"]
                    stats["sentiment_changes"] += 1 if changes["sentiment_changed"] else 0
                    stats["ner_removed"] += changes["ner_removed"]
                    stats["temporal_removed"] += changes["temporal_removed"]
                    stats["relations_removed"] += changes["relations_removed"]

                except json.JSONDecodeError:
                    continue

        # Write fixed data
        if not dry_run and file_changes > 0:
            with open(shard_file, "w", encoding="utf-8") as f:
                for sample in fixed_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info(f"  Fixed {shard_file.name}: {file_changes} samples updated")
        elif dry_run and file_changes > 0:
            logger.info(f"  [DRY-RUN] Would fix {shard_file.name}: {file_changes} samples")

        stats["files_processed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix schema compliance in generated data")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--execute", action="store_true", help="Apply fixes to data")
    parser.add_argument("--stats", action="store_true", help="Show violation statistics only")
    parser.add_argument("--backup", action="store_true", help="Create backup before fixing")

    args = parser.parse_args()

    # Data directories
    base_dir = Path("D:/Modeling_studio/data/familyos/unified")
    data_dirs = [
        base_dir / "output",
        base_dir / "output_synthetic",
    ]

    if args.stats:
        # Just show statistics
        violations = collect_violations(data_dirs)
        print_violations(violations)
        return

    if not args.dry_run and not args.execute:
        print("Please specify --dry-run, --execute, or --stats")
        parser.print_help()
        return

    dry_run = not args.execute

    print("\n" + "=" * 70)
    print("SCHEMA COMPLIANCE FIXER")
    print("=" * 70)
    print(f"Mode: {'DRY-RUN (preview only)' if dry_run else 'EXECUTE (applying fixes)'}")
    print(f"Directories: {[str(d) for d in data_dirs]}")

    # Show current violations first
    print("\nAnalyzing current violations...")
    violations = collect_violations(data_dirs)
    print_violations(violations)

    if violations["samples_with_issues"] == 0:
        print("\n✅ No schema violations found! Data is already compliant.")
        return

    # Create backup if requested
    if args.backup and args.execute:
        backup_dir = base_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for data_dir in data_dirs:
            if data_dir.exists():
                dest = backup_dir / data_dir.name
                shutil.copytree(data_dir, dest)
                logger.info(f"Backed up {data_dir} → {dest}")

    # Process each directory
    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    total_stats = defaultdict(int)

    for data_dir in data_dirs:
        print(f"\n📁 {data_dir}")
        stats = process_directory(data_dir, dry_run=dry_run)
        for key, value in stats.items():
            total_stats[key] += value

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed:     {total_stats['files_processed']}")
    print(f"Samples processed:   {total_stats['samples_processed']:,}")
    print(f"Samples fixed:       {total_stats['samples_fixed']:,}")
    print(f"Emotions removed:    {total_stats['emotions_removed']:,}")
    print(f"Sentiment mappings:  {total_stats['sentiment_changes']:,}")
    print(f"NER labels removed:  {total_stats['ner_removed']:,}")
    print(f"Temporal removed:    {total_stats['temporal_removed']:,}")
    print(f"Relations removed:   {total_stats['relations_removed']:,}")

    if dry_run:
        print("\n⚠️  DRY-RUN MODE: No changes were made.")
        print("    Run with --execute to apply fixes.")
        print("    Run with --execute --backup to create backup first.")
    else:
        print("\n✅ Fixes applied successfully!")
        print("\nRun 'python fix_schema_compliance.py --stats' to verify.")


if __name__ == "__main__":
    main()

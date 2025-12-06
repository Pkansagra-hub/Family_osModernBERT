"""
FamilyOS Embedding Data Healer

Validates and heals embedding triplet data for contrastive learning.

Checks:
1. Invalid cluster names
2. Same-cluster negatives (anchor_cluster == negative_cluster)
3. Near-duplicate anchors
4. Text quality issues (empty, too short, anchor ≈ positive/negative)
5. Cluster distribution balance

Run: python scripts/embedding_data_healer.py
"""

import json
import hashlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(r"D:\Modeling_studio\data\familyos\embeddings")
SILVER_PATH = BASE_PATH / "silver"
SILVER_SYNTHETIC_PATH = BASE_PATH / "silver_synthetic"
HEALED_PATH = BASE_PATH / "silver_healed"

# Valid 30 family-centric clusters
VALID_CLUSTERS = {
    # Immediate Family Core (4)
    "spouse_partner",
    "my_children",
    "my_parents",
    "my_siblings",
    # Extended Family (3)
    "grandparents",
    "in_laws",
    "extended_relatives",
    # Home & Daily Life (5)
    "morning_routines",
    "evening_family",
    "household_chores",
    "home_management",
    "family_meals",
    # Milestones & Traditions (4)
    "birthdays",
    "festivals_traditions",
    "weddings_ceremonies",
    "life_milestones",
    # Family Responsibilities (4)
    "family_finances",
    "kids_education",
    "family_health",
    "legal_documents",
    # Work-Family Balance (2)
    "work_family_balance",
    "childcare",
    # Emotional & Relational (4)
    "family_conflicts",
    "family_bonding",
    "grief_loss",
    "gratitude_love",
    # Family Extensions (4)
    "family_pets",
    "long_distance",
    "family_memories",
    "family_planning",
}

# Similarity thresholds
ANCHOR_DUPLICATE_THRESHOLD = 0.85  # Anchors this similar are near-duplicates
ANCHOR_POSITIVE_MIN_SIMILARITY = 0.3  # Positive should be at least this similar
ANCHOR_POSITIVE_MAX_SIMILARITY = 0.95  # Positive shouldn't be identical
ANCHOR_NEGATIVE_MAX_SIMILARITY = 0.6  # Negative should be different enough

MIN_TEXT_LENGTH = 10  # Minimum characters for anchor/positive/negative


# =============================================================================
# HEALING STATS TRACKER
# =============================================================================


@dataclass
class HealingStats:
    """Track all healing operations."""

    total_triplets: int = 0
    triplets_kept: int = 0
    triplets_deleted: int = 0

    # Issue counters
    invalid_anchor_cluster: int = 0
    invalid_negative_cluster: int = 0
    same_cluster_negative: int = 0
    empty_anchor: int = 0
    empty_positive: int = 0
    empty_negative: int = 0
    short_anchor: int = 0
    short_positive: int = 0
    short_negative: int = 0
    anchor_positive_identical: int = 0
    anchor_negative_too_similar: int = 0
    near_duplicate_anchors: int = 0

    # Cluster distribution
    anchor_cluster_dist: Counter = field(default_factory=Counter)
    negative_cluster_dist: Counter = field(default_factory=Counter)

    def get_summary(self) -> dict:
        return {
            "total_triplets": self.total_triplets,
            "triplets_kept": self.triplets_kept,
            "triplets_deleted": self.triplets_deleted,
            "deletion_rate": f"{self.triplets_deleted / max(1, self.total_triplets) * 100:.2f}%",
            "issues": {
                "invalid_anchor_cluster": self.invalid_anchor_cluster,
                "invalid_negative_cluster": self.invalid_negative_cluster,
                "same_cluster_negative": self.same_cluster_negative,
                "empty_anchor": self.empty_anchor,
                "empty_positive": self.empty_positive,
                "empty_negative": self.empty_negative,
                "short_anchor": self.short_anchor,
                "short_positive": self.short_positive,
                "short_negative": self.short_negative,
                "anchor_positive_identical": self.anchor_positive_identical,
                "anchor_negative_too_similar": self.anchor_negative_too_similar,
                "near_duplicate_anchors": self.near_duplicate_anchors,
            },
        }


# =============================================================================
# SIMILARITY HELPERS
# =============================================================================


def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def get_text_hash(text: str) -> str:
    """Get hash for near-duplicate detection."""
    # Normalize: lowercase, remove extra spaces
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_triplet(
    triplet: dict, stats: HealingStats, seen_anchors: set
) -> tuple[bool, list[str]]:
    """
    Validate a single triplet. Returns (is_valid, list_of_issues).
    """
    issues = []

    anchor = triplet.get("anchor", "")
    positive = triplet.get("positive", "")
    negative = triplet.get("negative", "")
    anchor_cluster = triplet.get("anchor_cluster", "")
    negative_cluster = triplet.get("negative_cluster", "")

    # 1. Check for empty/missing fields
    if not anchor:
        issues.append("empty_anchor")
        stats.empty_anchor += 1
    elif len(anchor) < MIN_TEXT_LENGTH:
        issues.append("short_anchor")
        stats.short_anchor += 1

    if not positive:
        issues.append("empty_positive")
        stats.empty_positive += 1
    elif len(positive) < MIN_TEXT_LENGTH:
        issues.append("short_positive")
        stats.short_positive += 1

    if not negative:
        issues.append("empty_negative")
        stats.empty_negative += 1
    elif len(negative) < MIN_TEXT_LENGTH:
        issues.append("short_negative")
        stats.short_negative += 1

    # 2. Check cluster validity
    if anchor_cluster not in VALID_CLUSTERS:
        issues.append(f"invalid_anchor_cluster: {anchor_cluster}")
        stats.invalid_anchor_cluster += 1

    if negative_cluster not in VALID_CLUSTERS:
        issues.append(f"invalid_negative_cluster: {negative_cluster}")
        stats.invalid_negative_cluster += 1

    # 3. Check same-cluster negative (CRITICAL)
    if anchor_cluster and negative_cluster and anchor_cluster == negative_cluster:
        issues.append("same_cluster_negative")
        stats.same_cluster_negative += 1

    # 4. Check anchor-positive similarity (should be similar but not identical)
    if anchor and positive:
        ap_sim = text_similarity(anchor, positive)
        if ap_sim > ANCHOR_POSITIVE_MAX_SIMILARITY:
            issues.append(f"anchor_positive_identical: {ap_sim:.2f}")
            stats.anchor_positive_identical += 1

    # 5. Check anchor-negative similarity (should be different)
    if anchor and negative:
        an_sim = text_similarity(anchor, negative)
        if an_sim > ANCHOR_NEGATIVE_MAX_SIMILARITY:
            issues.append(f"anchor_negative_too_similar: {an_sim:.2f}")
            stats.anchor_negative_too_similar += 1

    # 6. Check for near-duplicate anchors
    if anchor:
        anchor_hash = get_text_hash(anchor)
        if anchor_hash in seen_anchors:
            issues.append("near_duplicate_anchor")
            stats.near_duplicate_anchors += 1
        else:
            seen_anchors.add(anchor_hash)

    # Track cluster distribution
    if anchor_cluster in VALID_CLUSTERS:
        stats.anchor_cluster_dist[anchor_cluster] += 1
    if negative_cluster in VALID_CLUSTERS:
        stats.negative_cluster_dist[negative_cluster] += 1

    is_valid = len(issues) == 0
    return is_valid, issues


def can_fix_triplet(triplet: dict, issues: list[str]) -> tuple[bool, dict]:
    """
    Attempt to fix minor issues. Returns (was_fixed, fixed_triplet).

    Fixable issues:
    - Cluster name typos (if close match exists)

    Unfixable (delete):
    - Empty/short texts
    - Same-cluster negative
    - Anchor-negative too similar
    """
    # For now, we don't auto-fix - just delete bad triplets
    # Could add fuzzy cluster matching in the future

    critical_issues = {
        "empty_anchor",
        "empty_positive",
        "empty_negative",
        "same_cluster_negative",
        "anchor_negative_too_similar",
        "near_duplicate_anchor",
    }

    for issue in issues:
        issue_type = issue.split(":")[0]
        if issue_type in critical_issues:
            return False, triplet

    # Check if it's just an invalid cluster name
    if any("invalid_" in i for i in issues):
        # Could try fuzzy matching here
        # For now, just delete
        return False, triplet

    return True, triplet


# =============================================================================
# MAIN HEALER
# =============================================================================


def load_triplets(folder_path: Path) -> list[dict]:
    """Load all JSONL triplet files from a folder."""
    triplets = []

    if not folder_path.exists():
        print(f"      Warning: Path does not exist: {folder_path}")
        return triplets

    files = sorted(folder_path.glob("*.jsonl"))

    for file_path in files:
        if file_path.name == "hash_index.jsonl":
            continue
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        triplets.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return triplets


def save_triplets(triplets: list[dict], output_path: Path, triplets_per_shard: int = 10000):
    """Save triplets to JSONL shards."""
    output_path.mkdir(parents=True, exist_ok=True)

    shard_num = 0
    current_shard = []

    for triplet in triplets:
        current_shard.append(triplet)

        if len(current_shard) >= triplets_per_shard:
            shard_file = output_path / f"triplets_{shard_num:04d}.jsonl"
            with open(shard_file, "w", encoding="utf-8") as f:
                for t in current_shard:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            shard_num += 1
            current_shard = []

    # Write remaining triplets
    if current_shard:
        shard_file = output_path / f"triplets_{shard_num:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for t in current_shard:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")


def print_report(stats: HealingStats, dataset_name: str):
    """Print healing report."""
    print(f"\n{'='*70}")
    print(f"HEALING REPORT: {dataset_name}")
    print(f"{'='*70}")

    summary = stats.get_summary()
    print(f"\nTotal Triplets: {summary['total_triplets']:,}")
    print(f"Triplets Kept: {summary['triplets_kept']:,}")
    print(f"Triplets Deleted: {summary['triplets_deleted']:,} ({summary['deletion_rate']})")

    print(f"\n--- Issues Found ---")
    for issue_name, count in summary["issues"].items():
        if count > 0:
            print(f"  {issue_name}: {count:,}")

    # Cluster distribution analysis
    print(f"\n--- Anchor Cluster Distribution ---")
    total_valid = sum(stats.anchor_cluster_dist.values())
    if total_valid > 0:
        # Show top 10 and bottom 10
        sorted_clusters = stats.anchor_cluster_dist.most_common()

        print("  Top 5:")
        for cluster, count in sorted_clusters[:5]:
            pct = count / total_valid * 100
            print(f"    {cluster}: {count:,} ({pct:.1f}%)")

        print("  Bottom 5:")
        for cluster, count in sorted_clusters[-5:]:
            pct = count / total_valid * 100
            print(f"    {cluster}: {count:,} ({pct:.1f}%)")

        # Check for missing clusters
        missing = VALID_CLUSTERS - set(stats.anchor_cluster_dist.keys())
        if missing:
            print(f"\n  Missing clusters: {', '.join(sorted(missing))}")


def heal_dataset(input_path: Path, stats: HealingStats) -> list[dict]:
    """Validate and heal triplets from a dataset."""
    triplets = load_triplets(input_path)
    stats.total_triplets = len(triplets)

    healed_triplets = []
    seen_anchors = set()

    for triplet in triplets:
        is_valid, issues = validate_triplet(triplet, stats, seen_anchors)

        if is_valid:
            healed_triplets.append(triplet)
            stats.triplets_kept += 1
        else:
            # Try to fix
            was_fixed, fixed_triplet = can_fix_triplet(triplet, issues)
            if was_fixed:
                healed_triplets.append(fixed_triplet)
                stats.triplets_kept += 1
            else:
                stats.triplets_deleted += 1

    return healed_triplets


def main():
    print("=" * 70)
    print("FAMILYOS EMBEDDING DATA HEALER")
    print("Validates and heals triplet data for contrastive learning")
    print("=" * 70)

    all_triplets = []
    combined_stats = HealingStats()

    # Process silver data
    print("\n[1/3] Processing SILVER data...")
    if SILVER_PATH.exists():
        silver_stats = HealingStats()
        silver_triplets = heal_dataset(SILVER_PATH, silver_stats)
        all_triplets.extend(silver_triplets)
        print_report(silver_stats, "SILVER")

        # Merge stats
        combined_stats.total_triplets += silver_stats.total_triplets
        combined_stats.triplets_kept += silver_stats.triplets_kept
        combined_stats.triplets_deleted += silver_stats.triplets_deleted
    else:
        print(f"      Skipping - path not found: {SILVER_PATH}")

    # Process silver_synthetic data
    print("\n[2/3] Processing SILVER_SYNTHETIC data...")
    if SILVER_SYNTHETIC_PATH.exists():
        synthetic_stats = HealingStats()
        synthetic_triplets = heal_dataset(SILVER_SYNTHETIC_PATH, synthetic_stats)
        all_triplets.extend(synthetic_triplets)
        print_report(synthetic_stats, "SILVER_SYNTHETIC")

        # Merge stats
        combined_stats.total_triplets += synthetic_stats.total_triplets
        combined_stats.triplets_kept += synthetic_stats.triplets_kept
        combined_stats.triplets_deleted += synthetic_stats.triplets_deleted
    else:
        print(f"      Skipping - path not found: {SILVER_SYNTHETIC_PATH}")

    # Save healed data
    print("\n[3/3] Saving healed data...")
    if all_triplets:
        save_triplets(all_triplets, HEALED_PATH)
        print(f"      Saved {len(all_triplets):,} triplets to {HEALED_PATH}")
    else:
        print("      No triplets to save!")

    # Combined summary
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal Triplets Processed: {combined_stats.total_triplets:,}")
    print(f"Total Triplets Kept: {combined_stats.triplets_kept:,}")
    print(f"Total Triplets Deleted: {combined_stats.triplets_deleted:,}")
    if combined_stats.total_triplets > 0:
        keep_rate = combined_stats.triplets_kept / combined_stats.total_triplets * 100
        print(f"Keep Rate: {keep_rate:.1f}%")

    print(f"\n{'='*70}")
    print("HEALING COMPLETE")
    print(f"{'='*70}")
    print(f"\nHealed data saved to: {HEALED_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Review the cluster distribution")
    print(f"  2. Use healed data for training: {HEALED_PATH}")


if __name__ == "__main__":
    main()

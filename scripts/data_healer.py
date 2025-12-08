"""
FamilyOS Data Healer

Heals data quality issues based on the philosophy:
"Trust the emotions, fix the routing."

Run: python scripts/data_healer.py
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(r"D:\Modeling_studio\data\familyos\unified")
OUTPUT_PATH = BASE_PATH / "output"
SYNTHETIC_PATH = BASE_PATH / "output_synthetic"
HEALED_OUTPUT_PATH = BASE_PATH / "output_healed"
HEALED_SYNTHETIC_PATH = BASE_PATH / "output_synthetic_healed"

# Kinship to predicate mapping
KINSHIP_PREDICATES = {
    # Parents
    "mom": "child_of",
    "mother": "child_of",
    "mum": "child_of",
    "ma": "child_of",
    "dad": "child_of",
    "father": "child_of",
    "papa": "child_of",
    "pa": "child_of",
    "parents": "child_of",
    "parent": "child_of",
    # Children
    "son": "parent_of",
    "daughter": "parent_of",
    "child": "parent_of",
    "kids": "parent_of",
    "children": "parent_of",
    "baby": "parent_of",
    # Siblings
    "brother": "sibling_of",
    "sister": "sibling_of",
    "sibling": "sibling_of",
    "bhai": "sibling_of",
    "behen": "sibling_of",
    "sis": "sibling_of",
    "bro": "sibling_of",
    # Spouse
    "husband": "spouse_of",
    "wife": "spouse_of",
    "spouse": "spouse_of",
    "partner": "spouse_of",
    # Grandparents
    "grandma": "grandchild_of",
    "grandmother": "grandchild_of",
    "grandpa": "grandchild_of",
    "grandfather": "grandchild_of",
    "nani": "grandchild_of",
    "nana": "grandchild_of",
    "dadi": "grandchild_of",
    "dada": "grandchild_of",
    "grandparents": "grandchild_of",
    # Aunts/Uncles
    "uncle": "niece_nephew_of",
    "aunt": "niece_nephew_of",
    "chacha": "niece_nephew_of",
    "chachi": "niece_nephew_of",
    "mama": "niece_nephew_of",
    "mami": "niece_nephew_of",
    "masi": "niece_nephew_of",
    "mausa": "niece_nephew_of",
    "bua": "niece_nephew_of",
    "fufa": "niece_nephew_of",
    "taya": "niece_nephew_of",
    "tayi": "niece_nephew_of",
    # Niece/Nephew
    "niece": "aunt_uncle_of",
    "nephew": "aunt_uncle_of",
    # Cousins
    "cousin": "cousin_of",
    # In-laws
    "mother-in-law": "child_in_law_of",
    "father-in-law": "child_in_law_of",
    "sister-in-law": "sibling_in_law_of",
    "brother-in-law": "sibling_in_law_of",
    "sis-in-law": "sibling_in_law_of",
    # Others
    "family": "family_of",
    "relative": "relative_of",
    "friend": "friend_of",
    "colleague": "colleague_of",
    "toddler": "parent_of",
}


# =============================================================================
# HEALING STATS TRACKER
# =============================================================================


@dataclass
class HealingStats:
    """Track all healing operations."""

    total_samples: int = 0
    samples_modified: int = 0

    # Fix counters
    emo_routing_fixed: int = 0
    token_mismatch_fixed: int = 0
    token_mismatch_deleted: int = 0
    index_bounds_fixed: int = 0
    neutral_infection_fixed: int = 0
    rel_routing_fixed: int = 0
    relations_generated: int = 0
    mem_routing_fixed: int = 0
    duplicate_ids_fixed: int = 0
    sentiment_emotion_fixed: int = 0
    rel_false_with_relations_fixed: int = 0

    # Entity tracking
    entities_deleted: int = 0
    entities_fixed: int = 0

    def get_summary(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "samples_modified": self.samples_modified,
            "modification_rate": f"{self.samples_modified / max(1, self.total_samples) * 100:.2f}%",
            "fixes": {
                "emo_routing_fixed": self.emo_routing_fixed,
                "token_mismatch_fixed": self.token_mismatch_fixed,
                "token_mismatch_deleted": self.token_mismatch_deleted,
                "index_bounds_fixed": self.index_bounds_fixed,
                "neutral_infection_fixed": self.neutral_infection_fixed,
                "rel_routing_fixed": self.rel_routing_fixed,
                "relations_generated": self.relations_generated,
                "mem_routing_fixed": self.mem_routing_fixed,
                "duplicate_ids_fixed": self.duplicate_ids_fixed,
                "sentiment_emotion_fixed": self.sentiment_emotion_fixed,
                "rel_false_with_relations_fixed": self.rel_false_with_relations_fixed,
                "entities_deleted": self.entities_deleted,
                "entities_fixed": self.entities_fixed,
            },
        }


# =============================================================================
# PHASE 1: CRITICAL FIXES
# =============================================================================


def fix_emo_routing(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 1: EMO Routing Alignment
    If emotions contains ANY non-neutral value, set EMO=True
    """
    modified = False
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    emotions = tasks.get("emotions", [])

    # Check for non-neutral emotions
    non_neutral = [e for e in emotions if e != "neutral"]

    if non_neutral and not hub.get("EMO", False):
        hub["EMO"] = True
        sample["hub_routing"] = hub
        stats.emo_routing_fixed += 1
        modified = True

    return modified


def fix_token_mismatch(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 2 & 3: Token/Span Mismatch and Index Out of Bounds
    Re-calculate offsets or delete unfixable entities
    """
    modified = False
    text = sample.get("text", "")
    tasks = sample.get("tasks", {})

    if not text:
        return False

    # Process both ner_family and temporal
    for field_name in ["ner_family", "temporal"]:
        entities = tasks.get(field_name, [])
        if not entities:
            continue

        fixed_entities = []

        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            token = entity.get("token", "")

            if not token:
                stats.entities_deleted += 1
                continue

            # Check if indices are valid
            valid_indices = 0 <= start < len(text) and 0 < end <= len(text) and start < end

            if valid_indices:
                actual_text = text[start:end]
                if actual_text == token:
                    # Perfect match - keep as is
                    fixed_entities.append(entity)
                    continue

            # Try to find correct position
            correct_start = text.find(token)

            if correct_start != -1:
                # Found exact match
                entity["start"] = correct_start
                entity["end"] = correct_start + len(token)
                fixed_entities.append(entity)
                stats.entities_fixed += 1
                stats.token_mismatch_fixed += 1
                modified = True
                continue

            # Try case-insensitive search
            lower_text = text.lower()
            lower_token = token.lower()
            correct_start = lower_text.find(lower_token)

            if correct_start != -1:
                # Found case-insensitive match
                entity["start"] = correct_start
                entity["end"] = correct_start + len(token)
                # Update token to match actual text
                entity["token"] = text[correct_start : correct_start + len(token)]
                fixed_entities.append(entity)
                stats.entities_fixed += 1
                stats.token_mismatch_fixed += 1
                modified = True
                continue

            # Try partial match (token might have extra chars)
            token_words = token.split()
            if len(token_words) > 1:
                # Multi-word token - try to find first word
                first_word = token_words[0]
                correct_start = text.find(first_word)
                if correct_start != -1:
                    # Reconstruct the span
                    potential_end = correct_start + len(token)
                    if potential_end <= len(text):
                        entity["start"] = correct_start
                        entity["end"] = potential_end
                        entity["token"] = text[correct_start:potential_end]
                        fixed_entities.append(entity)
                        stats.entities_fixed += 1
                        stats.token_mismatch_fixed += 1
                        modified = True
                        continue

            # Cannot fix - delete entity
            stats.entities_deleted += 1
            stats.token_mismatch_deleted += 1
            modified = True

        tasks[field_name] = fixed_entities

    sample["tasks"] = tasks
    return modified


def fix_neutral_infection(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 4: Neutral Infection
    Remove 'neutral' from multi-emotion lists
    """
    modified = False
    tasks = sample.get("tasks", {})
    emotions = tasks.get("emotions", [])

    if len(emotions) > 1 and "neutral" in emotions:
        emotions = [e for e in emotions if e != "neutral"]
        tasks["emotions"] = emotions
        sample["tasks"] = tasks
        stats.neutral_infection_fixed += 1
        modified = True

    # Ensure emotions is never empty
    if not tasks.get("emotions"):
        tasks["emotions"] = ["neutral"]
        sample["tasks"] = tasks

    return modified


# =============================================================================
# PHASE 2: HIGH PRIORITY FIXES
# =============================================================================


def fix_rel_routing_and_relations(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 5: REL Routing vs Relations
    - If REL=True but relations=[], try to generate from KINSHIP tokens
    - If no KINSHIP tokens, set REL=False
    """
    modified = False
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    relations = tasks.get("relations", [])
    ner = tasks.get("ner_family", [])

    # Case: REL=True but no relations
    if hub.get("REL", False) and not relations:
        # Find KINSHIP tokens
        kinship_tokens = [e for e in ner if e.get("label") == "KINSHIP"]
        person_tokens = [e for e in ner if e.get("label") == "PERSON"]

        if kinship_tokens:
            # Generate relations from KINSHIP tokens
            new_relations = []
            for kin in kinship_tokens:
                token = kin.get("token", "").lower().rstrip("'s").rstrip("'")

                # Skip generic terms
                if token in {"family", "relative", "relatives"}:
                    continue

                predicate = None
                for key, pred in KINSHIP_PREDICATES.items():
                    if key in token or token in key:
                        predicate = pred
                        break

                if predicate:
                    # Try to find associated PERSON token
                    obj = kin.get("token", "").rstrip("'s").rstrip("'")

                    new_relations.append({"subject": "user", "predicate": predicate, "object": obj})
                    stats.relations_generated += 1

            if new_relations:
                tasks["relations"] = new_relations
                sample["tasks"] = tasks
                modified = True
            else:
                # No valid relations could be generated - fix routing
                hub["REL"] = False
                sample["hub_routing"] = hub
                stats.rel_routing_fixed += 1
                modified = True
        else:
            # No KINSHIP tokens - fix routing
            hub["REL"] = False
            sample["hub_routing"] = hub
            stats.rel_routing_fixed += 1
            modified = True

    return modified


def fix_mem_routing(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 6: MEM Routing vs Intent
    If intent is memory-related, set MEM=True
    """
    modified = False
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    intent = tasks.get("intent", "")

    memory_intents = {"query_memory", "log_memory"}

    if intent in memory_intents and not hub.get("MEM", False):
        hub["MEM"] = True
        sample["hub_routing"] = hub
        stats.mem_routing_fixed += 1
        modified = True

    # Also check for reflect with memory keywords
    if intent == "reflect":
        text = sample.get("text", "").lower()
        memory_keywords = [
            "remember",
            "recall",
            "last time",
            "when did",
            "what was",
            "used to",
            "back when",
            "years ago",
            "childhood",
            "memories",
        ]
        if any(kw in text for kw in memory_keywords):
            if not hub.get("MEM", False):
                hub["MEM"] = True
                sample["hub_routing"] = hub
                stats.mem_routing_fixed += 1
                modified = True

    return modified


def fix_duplicate_ids(samples: list, stats: HealingStats) -> int:
    """
    Fix 7: Duplicate IDs
    Add suffix to duplicate IDs
    """
    seen_ids = {}
    fixed_count = 0

    for sample in samples:
        original_id = sample.get("id", "unknown")

        if original_id in seen_ids:
            seen_ids[original_id] += 1
            new_id = f"{original_id}_v{seen_ids[original_id]}"
            sample["id"] = new_id
            stats.duplicate_ids_fixed += 1
            fixed_count += 1
        else:
            seen_ids[original_id] = 1

    return fixed_count


# =============================================================================
# PHASE 3: MEDIUM PRIORITY FIXES
# =============================================================================


def fix_sentiment_emotion_alignment(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 8: Sentiment-Emotion Alignment
    Only fix extreme contradictions
    """
    modified = False
    tasks = sample.get("tasks", {})
    sentiment = tasks.get("sentiment", "")
    emotions = set(tasks.get("emotions", []))

    positive_emotions = {
        "joy",
        "excitement",
        "love",
        "pride",
        "gratitude",
        "happiness",
        "celebration",
    }
    negative_emotions = {"sadness", "grief", "anger", "despair", "frustration", "disappointment"}

    # Check for extreme contradiction: very_positive with ONLY negative emotions
    if sentiment == "very_positive":
        has_any_positive = bool(emotions & positive_emotions)
        has_only_negative = emotions.issubset(negative_emotions | {"neutral"}) and bool(
            emotions & negative_emotions
        )

        if has_only_negative and not has_any_positive:
            tasks["sentiment"] = "very_negative"
            sample["tasks"] = tasks
            stats.sentiment_emotion_fixed += 1
            modified = True

    # Check for extreme contradiction: very_negative with ONLY positive emotions
    if sentiment == "very_negative":
        has_any_negative = bool(emotions & negative_emotions)
        has_only_positive = emotions.issubset(positive_emotions | {"neutral"}) and bool(
            emotions & positive_emotions
        )

        if has_only_positive and not has_any_negative:
            tasks["sentiment"] = "very_positive"
            sample["tasks"] = tasks
            stats.sentiment_emotion_fixed += 1
            modified = True

    # Add bittersweet if both positive and negative present
    if (emotions & positive_emotions) and (emotions & negative_emotions):
        if "bittersweet" not in emotions:
            emotions_list = tasks.get("emotions", [])
            emotions_list.append("bittersweet")
            tasks["emotions"] = emotions_list
            sample["tasks"] = tasks
            modified = True

    return modified


def fix_rel_false_with_relations(sample: dict, stats: HealingStats) -> bool:
    """
    Fix 9: REL=False but has relations
    If relations exist, set REL=True
    """
    modified = False
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    relations = tasks.get("relations", [])

    if relations and not hub.get("REL", False):
        hub["REL"] = True
        sample["hub_routing"] = hub
        stats.rel_false_with_relations_fixed += 1
        modified = True

    return modified


# =============================================================================
# MAIN HEALER
# =============================================================================


def heal_sample(sample: dict, stats: HealingStats) -> bool:
    """Apply all healing fixes to a single sample."""
    modified = False

    # Phase 1: Critical
    modified |= fix_emo_routing(sample, stats)
    modified |= fix_token_mismatch(sample, stats)
    modified |= fix_neutral_infection(sample, stats)

    # Phase 2: High Priority
    modified |= fix_rel_routing_and_relations(sample, stats)
    modified |= fix_mem_routing(sample, stats)
    # Duplicate IDs handled separately at dataset level

    # Phase 3: Medium Priority
    modified |= fix_sentiment_emotion_alignment(sample, stats)
    modified |= fix_rel_false_with_relations(sample, stats)

    if modified:
        stats.samples_modified += 1

    return modified


def load_jsonl_files(folder_path: Path) -> list:
    """Load all JSONL files from a folder."""
    samples = []
    files = sorted(folder_path.glob("shard_*.jsonl"))

    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return samples


def save_jsonl_files(samples: list, output_path: Path, samples_per_shard: int = 5000):
    """Save samples to JSONL shards."""
    output_path.mkdir(parents=True, exist_ok=True)

    shard_num = 0
    current_shard = []

    for sample in samples:
        current_shard.append(sample)

        if len(current_shard) >= samples_per_shard:
            shard_file = output_path / f"shard_{shard_num:04d}.jsonl"
            with open(shard_file, "w", encoding="utf-8") as f:
                for s in current_shard:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            shard_num += 1
            current_shard = []

    # Write remaining samples
    if current_shard:
        shard_file = output_path / f"shard_{shard_num:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for s in current_shard:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


def print_report(stats: HealingStats, dataset_name: str):
    """Print healing report."""
    print(f"\n{'='*70}")
    print(f"HEALING REPORT: {dataset_name}")
    print(f"{'='*70}")

    summary = stats.get_summary()
    print(f"\nTotal Samples: {summary['total_samples']:,}")
    print(f"Samples Modified: {summary['samples_modified']:,} ({summary['modification_rate']})")

    print(f"\n--- Fixes Applied ---")
    for fix_name, count in summary["fixes"].items():
        if count > 0:
            print(f"  {fix_name}: {count:,}")


def main():
    print("=" * 70)
    print("FAMILYOS DATA HEALER")
    print("Philosophy: Trust the emotions, fix the routing.")
    print("=" * 70)

    # Process real data (output)
    print("\n[1/4] Loading OUTPUT (real) data...")
    output_samples = load_jsonl_files(OUTPUT_PATH)
    output_stats = HealingStats()
    output_stats.total_samples = len(output_samples)

    print(f"      Loaded {len(output_samples):,} samples")
    print("      Healing samples...")

    for sample in output_samples:
        heal_sample(sample, output_stats)

    # Fix duplicate IDs
    fix_duplicate_ids(output_samples, output_stats)

    print_report(output_stats, "OUTPUT (Real Data)")

    print("\n[2/4] Saving healed OUTPUT data...")
    save_jsonl_files(output_samples, HEALED_OUTPUT_PATH)
    print(f"      Saved to {HEALED_OUTPUT_PATH}")

    # Process synthetic data
    print("\n[3/4] Loading OUTPUT_SYNTHETIC data...")
    synthetic_samples = load_jsonl_files(SYNTHETIC_PATH)
    synthetic_stats = HealingStats()
    synthetic_stats.total_samples = len(synthetic_samples)

    print(f"      Loaded {len(synthetic_samples):,} samples")
    print("      Healing samples (this may take a while)...")

    for i, sample in enumerate(synthetic_samples):
        heal_sample(sample, synthetic_stats)
        if (i + 1) % 50000 == 0:
            print(f"      Processed {i+1:,} / {len(synthetic_samples):,} samples...")

    # Fix duplicate IDs
    fix_duplicate_ids(synthetic_samples, synthetic_stats)

    print_report(synthetic_stats, "OUTPUT_SYNTHETIC")

    print("\n[4/4] Saving healed OUTPUT_SYNTHETIC data...")
    save_jsonl_files(synthetic_samples, HEALED_SYNTHETIC_PATH)
    print(f"      Saved to {HEALED_SYNTHETIC_PATH}")

    # Combined summary
    print(f"\n{'='*70}")
    print("COMBINED HEALING SUMMARY")
    print(f"{'='*70}")

    total_samples = output_stats.total_samples + synthetic_stats.total_samples
    total_modified = output_stats.samples_modified + synthetic_stats.samples_modified

    print(f"\nTotal Samples Processed: {total_samples:,}")
    print(f"Total Samples Modified: {total_modified:,} ({total_modified/total_samples*100:.2f}%)")

    print(f"\n--- Combined Fix Counts ---")
    combined_fixes = {}
    for key in output_stats.get_summary()["fixes"]:
        combined_fixes[key] = (
            output_stats.get_summary()["fixes"][key] + synthetic_stats.get_summary()["fixes"][key]
        )

    for fix_name, count in sorted(combined_fixes.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {fix_name}: {count:,}")

    print(f"\n{'='*70}")
    print("HEALING COMPLETE")
    print(f"{'='*70}")
    print(f"\nHealed data saved to:")
    print(f"  - {HEALED_OUTPUT_PATH}")
    print(f"  - {HEALED_SYNTHETIC_PATH}")
    print(f"\nNext step: Run audit on healed data to verify fixes:")
    print(f"  python scripts/comprehensive_data_audit.py --healed")


if __name__ == "__main__":
    main()

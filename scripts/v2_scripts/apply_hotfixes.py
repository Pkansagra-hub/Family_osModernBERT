#!/usr/bin/env python3
"""
Apply preprocessing hotfixes for ModernBERT v3.3 unified training data.

HOTFIXES:
- P0: Resolve 'mixed' sentiment → 5-class scale (very_negative/negative/neutral/positive/very_positive)
- P1A: Fix hub_routing.TASK for 2,595 eligible samples
- P1B: Generate additional TASK-domain samples

Usage:
    python apply_hotfixes.py                    # Dry-run analysis
    python apply_hotfixes.py --apply            # Apply P0 + P1A fixes
    python apply_hotfixes.py --generate-task    # Trigger P1B generation
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional
import shutil
from datetime import datetime

# ============================================================================
# EMOTION CATEGORIES FOR SENTIMENT RESOLUTION
# ============================================================================

POSITIVE_EMOTIONS = {
    "joy",
    "love",
    "warmth",
    "contentment",
    "pride",
    "gratitude",
    "excitement",
    "amusement",
    "affection",
    "hope",
    "relief",
    "enthusiasm",
    "satisfaction",
    "serenity",
    "tenderness",
    "caring",
    "admiration",
    "appreciation",
    "optimism",
    "playfulness",
    "happiness",
}

NEGATIVE_EMOTIONS = {
    "sadness",
    "frustration",
    "worry",
    "anger",
    "fear",
    "anxiety",
    "disappointment",
    "guilt",
    "shame",
    "loneliness",
    "grief",
    "irritation",
    "resentment",
    "jealousy",
    "envy",
    "disgust",
    "despair",
    "helplessness",
    "insecurity",
    "overwhelm",
    "dread",
    "bitterness",
    "contempt",
    "hurt",
    "regret",
    "embarrassment",
}

NEUTRAL_EMOTIONS = {
    "curiosity",
    "surprise",
    "nostalgia",
    "confusion",
    "anticipation",
    "determination",
    "reflection",
    "uncertainty",
    "contemplation",
}

# ============================================================================
# TASK HUB ELIGIBILITY CRITERIA
# ============================================================================

TASK_DOMAINS = {"TASK", "PLANNING", "FINANCE", "WORK"}
TASK_INTENTS = {"set_reminder", "other", "query_memory"}

# ============================================================================
# P0: SENTIMENT RESOLUTION
# ============================================================================


def resolve_mixed_sentiment(emotions: list[str]) -> str:
    """
    Resolve 'mixed' sentiment to 5-class scale based on emotion pole.

    Logic:
    - Count positive vs negative emotions
    - If pos >> neg (1.5x): very_positive or positive
    - If neg >> pos (1.5x): very_negative or negative
    - If balanced: neutral

    Intensity determined by emotion count:
    - 3+ dominant emotions: very_*
    - 1-2 dominant emotions: mild
    """
    pos_count = sum(1 for e in emotions if e.lower() in POSITIVE_EMOTIONS)
    neg_count = sum(1 for e in emotions if e.lower() in NEGATIVE_EMOTIONS)
    neutral_count = sum(1 for e in emotions if e.lower() in NEUTRAL_EMOTIONS)

    # Strong positive pole
    if pos_count > neg_count * 1.5:
        return "very_positive" if pos_count >= 3 else "positive"

    # Strong negative pole
    if neg_count > pos_count * 1.5:
        return "very_negative" if neg_count >= 3 else "negative"

    # Balanced or neutral-dominant
    if neutral_count >= max(pos_count, neg_count):
        return "neutral"

    # Slight lean
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"

    return "neutral"


def apply_p0_hotfix(sample: dict) -> tuple[dict, bool]:
    """
    Apply P0 hotfix: resolve 'mixed' sentiment.

    Returns:
        (modified_sample, was_modified)
    """
    sentiment = sample.get("sentiment", "")

    if sentiment.lower() != "mixed":
        return sample, False

    emotions = sample.get("emotions", [])
    if not emotions:
        # No emotions to infer from, default to neutral
        sample["sentiment"] = "neutral"
        sample["_hotfix_p0"] = "mixed→neutral (no emotions)"
        return sample, True

    resolved = resolve_mixed_sentiment(emotions)
    sample["sentiment"] = resolved
    sample["_hotfix_p0"] = f"mixed→{resolved} (from {len(emotions)} emotions)"

    return sample, True


# ============================================================================
# P1A: HUB ROUTING FIX
# ============================================================================


def check_task_hub_eligibility(sample: dict) -> bool:
    """
    Check if sample should have TASK hub but doesn't.

    Criteria (any of):
    - ingress_domain in {TASK, PLANNING, FINANCE, WORK}
    - intent in {set_reminder, other, query_memory}
    """
    hub_routing = sample.get("hub_routing", {})

    # Skip if already has TASK hub
    if hub_routing.get("TASK", False):
        return False

    # Check domain
    ingress_domain = sample.get("ingress_domain", "")
    if ingress_domain in TASK_DOMAINS:
        return True

    # Check intent
    intent = sample.get("intent", "")
    if intent in TASK_INTENTS:
        return True

    return False


def apply_p1a_hotfix(sample: dict) -> tuple[dict, bool]:
    """
    Apply P1A hotfix: fix hub_routing.TASK for eligible samples.

    Returns:
        (modified_sample, was_modified)
    """
    if not check_task_hub_eligibility(sample):
        return sample, False

    # Ensure hub_routing dict exists
    if "hub_routing" not in sample:
        sample["hub_routing"] = {}

    # Set TASK hub to True
    sample["hub_routing"]["TASK"] = True

    # Record reason
    ingress_domain = sample.get("ingress_domain", "")
    intent = sample.get("intent", "")
    reason = []
    if ingress_domain in TASK_DOMAINS:
        reason.append(f"domain={ingress_domain}")
    if intent in TASK_INTENTS:
        reason.append(f"intent={intent}")

    sample["_hotfix_p1a"] = f"TASK=True ({', '.join(reason)})"

    return sample, True


# ============================================================================
# P1B: TASK DOMAIN GENERATION CONFIG
# ============================================================================

P1B_GENERATION_CONFIG = {
    "target_samples": 1500,  # ~1,500 additional TASK samples → 15%+ target
    "priority_domains": ["TASK", "PLANNING", "FINANCE", "WORK"],
    "priority_intents": ["set_reminder", "other", "query_memory"],
    "output_path": "data/processed/unified_task_augmentation.jsonl",
    "prompt_templates": [
        "Generate a family conversation about {domain} planning.",
        "Create a sample where someone {intent}.",
        "Write a message about organizing {domain} activities.",
    ],
}


def prepare_p1b_generation() -> dict:
    """
    Prepare configuration for P1B TASK-domain generation.

    Returns generation config and instructions.
    """
    return {
        "status": "ready",
        "config": P1B_GENERATION_CONFIG,
        "instructions": """
To run P1B generation:

1. Use existing generation infrastructure:
   python scripts/agents/unified_data_agent.py \\
       --mode task-augment \\
       --target-samples 1500 \\
       --domains TASK,PLANNING,FINANCE,WORK

2. Or run targeted generation:
   python scripts/generate_task_samples.py \\
       --count 1500 \\
       --output data/processed/unified_task_augmentation.jsonl

3. Merge with main dataset after generation completes.
""",
    }


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================


def load_samples(data_path: Path) -> list[dict]:
    """Load samples from file or directory of shards."""
    samples = []

    if data_path.is_dir():
        # Load all shards in directory
        shard_files = sorted(data_path.glob("shard_*.jsonl"))
        print(f"Loading {len(shard_files)} shards from {data_path}")
        for shard in shard_files:
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
    else:
        # Single file
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

    return samples


def analyze_data(data_path: Path) -> dict:
    """
    Analyze current data state for hotfix needs.
    """
    stats = {
        "total_samples": 0,
        "p0_mixed_sentiment": 0,
        "p0_resolution_preview": Counter(),
        "p1a_task_eligible": 0,
        "p1a_already_task": 0,
        "current_task_hub": 0,
        "projected_task_hub": 0,
    }

    samples = load_samples(data_path)

    stats["total_samples"] = len(samples)

    for sample in samples:
        sentiment = sample.get("sentiment", "")
        hub_routing = sample.get("hub_routing", {})

        # P0 analysis
        if sentiment.lower() == "mixed":
            stats["p0_mixed_sentiment"] += 1
            emotions = sample.get("emotions", [])
            resolved = resolve_mixed_sentiment(emotions) if emotions else "neutral"
            stats["p0_resolution_preview"][resolved] += 1

        # Current TASK hub count
        if hub_routing.get("TASK", False):
            stats["current_task_hub"] += 1
            stats["p1a_already_task"] += 1

        # P1A eligibility
        if check_task_hub_eligibility(sample):
            stats["p1a_task_eligible"] += 1

    # Projected after P1A
    stats["projected_task_hub"] = stats["current_task_hub"] + stats["p1a_task_eligible"]

    return stats


def print_report(stats: dict):
    """Print formatted analysis report."""
    total = stats["total_samples"]

    print("\n" + "=" * 70)
    print("HOTFIX ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nTotal Samples: {total:,}")

    # P0 Report
    print("\n--- P0: Mixed Sentiment Resolution ---")
    p0_count = stats["p0_mixed_sentiment"]
    print(f"Samples with 'mixed' sentiment: {p0_count:,} ({p0_count/total*100:.1f}%)")
    print("\nResolution Preview:")
    for sentiment, count in sorted(stats["p0_resolution_preview"].items(), key=lambda x: -x[1]):
        print(f"  {sentiment}: {count:,} ({count/p0_count*100:.1f}%)")

    # P1A Report
    print("\n--- P1A: TASK Hub Routing Fix ---")
    current_pct = stats["current_task_hub"] / total * 100
    eligible = stats["p1a_task_eligible"]
    projected = stats["projected_task_hub"]
    projected_pct = projected / total * 100

    print(f"Current TASK Hub: {stats['current_task_hub']:,} ({current_pct:.1f}%)")
    print(f"Eligible for TASK (missing): {eligible:,}")
    print(f"Projected after P1A: {projected:,} ({projected_pct:.1f}%)")
    print(f"Target: 15.0% → {'✅ MET' if projected_pct >= 15 else '⚠️ Need P1B'}")

    # P1B Recommendation
    if projected_pct < 15:
        needed = int(total * 0.15 - projected)
        print(f"\n--- P1B: Additional Generation Needed ---")
        print(f"Samples needed for 15%: ~{needed:,}")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def apply_hotfixes(data_path: Path, output_path: Path, dry_run: bool = True):
    """
    Apply all hotfixes to the dataset.
    """
    print(f"\n{'[DRY RUN]' if dry_run else '[APPLYING]'} Processing {data_path}")

    samples = load_samples(data_path)

    total = len(samples)
    p0_fixed = 0
    p1a_fixed = 0

    modified_samples = []

    for sample in samples:
        # Apply P0
        sample, was_p0 = apply_p0_hotfix(sample)
        if was_p0:
            p0_fixed += 1

        # Apply P1A
        sample, was_p1a = apply_p1a_hotfix(sample)
        if was_p1a:
            p1a_fixed += 1

        modified_samples.append(sample)

    print(f"\nP0 (sentiment resolution): {p0_fixed:,} samples modified")
    print(f"P1A (TASK hub routing): {p1a_fixed:,} samples modified")

    if not dry_run:
        # Output to single file
        output_file = output_path / "hotfixed_merged.jsonl" if output_path.is_dir() else output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write modified data
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in modified_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"\nModified data written: {output_file}")

        # Verify
        new_stats = analyze_data(output_file)
        print("\nPost-hotfix verification:")
        print(f"  - Mixed sentiment: {new_stats['p0_mixed_sentiment']} (should be 0)")
        print(
            f"  - TASK hub: {new_stats['current_task_hub']} ({new_stats['current_task_hub']/total*100:.1f}%)"
        )

    return {"p0_fixed": p0_fixed, "p1a_fixed": p1a_fixed, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Apply preprocessing hotfixes for ModernBERT v3.3")
    parser.add_argument("--apply", action="store_true", help="Apply hotfixes (default: dry-run)")
    parser.add_argument(
        "--generate-task", action="store_true", help="Show P1B generation instructions"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/familyos/unified/output"),
        help="Path to unified training data (file or directory of shards)",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Output path (default: overwrites input)"
    )

    args = parser.parse_args()

    if args.generate_task:
        config = prepare_p1b_generation()
        print("\n" + "=" * 70)
        print("P1B TASK DOMAIN GENERATION")
        print("=" * 70)
        print(config["instructions"])
        print(f"\nConfig: {json.dumps(config['config'], indent=2)}")
        return

    data_path = args.data_path
    output_path = args.output_path or data_path

    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print("\nLooking for alternatives...")

        # Try to find the data file
        alternatives = list(Path("data/processed").glob("*.jsonl"))
        if alternatives:
            print(f"Found: {[str(p) for p in alternatives[:5]]}")
        return

    # Run analysis
    stats = analyze_data(data_path)
    print_report(stats)

    if args.apply:
        print("\n" + "-" * 70)
        print("APPLYING HOTFIXES...")
        print("-" * 70)
        result = apply_hotfixes(data_path, output_path, dry_run=False)
        print(f"\n✅ Hotfixes applied successfully!")
        print(f"   P0: {result['p0_fixed']:,} sentiment fixes")
        print(f"   P1A: {result['p1a_fixed']:,} hub routing fixes")
    else:
        print("\n💡 Run with --apply to apply hotfixes")
        print("   Run with --generate-task for P1B generation instructions")


if __name__ == "__main__":
    main()

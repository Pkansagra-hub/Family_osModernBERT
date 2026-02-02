"""Test model behavior on edge cases from golden set.

Shows how the model handles:
- Sarcasm
- Passive-aggressive language
- Ambiguous context
- Mixed emotions
- Irony
- Cultural context
- Subtle resentment
- Rhetorical questions
- Contradictory emotions
- Boundary violations
- Crisis escalation
"""

import json
from pathlib import Path
from familyos_ultrabert import Client


def main():
    # Load edge cases from golden set
    golden_path = Path("data/familyos/unified/golden_set/shard_0000.jsonl")

    edge_cases = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data["id"].startswith("edge_"):
                    edge_cases.append(data)

    print("=" * 80)
    print("EDGE CASE ANALYSIS: Model Behavior on Challenging Scenarios")
    print("=" * 80)
    print(f"\nFound {len(edge_cases)} edge cases\n")

    # Initialize client
    print("Loading UltraBERT...")
    client = Client()
    print("Ready!\n")

    # Test each edge case
    for i, case in enumerate(edge_cases, 1):
        case_id = case["id"]
        text = case["text"]
        tasks = case["tasks"]
        expected_emotions = tasks.get("emotions", [])
        safety = tasks.get("safety_familyos", "GREEN")

        # Get predictions
        pred_emotions = client.get_emotions(text)

        # Calculate overlap
        expected_set = {e.lower() for e in expected_emotions}
        pred_set = {e.lower() for e in pred_emotions}
        overlap = expected_set.intersection(pred_set)

        # Determine category
        category = case_id.replace("edge_", "").replace("_001", "").replace("_002", "").replace("_003", "").upper()

        print(f"[{i}] {category}")
        print("-" * 80)
        print(f"ID: {case_id}")
        print(f"Safety: {safety}")
        print(f"\nText:\n  {text}\n")
        print(f"Expected emotions: {sorted(expected_set)}")
        print(f"Predicted emotions: {sorted(pred_set)}")

        if overlap:
            print(f"✅ HIT - Overlap: {sorted(overlap)} ({len(overlap)}/{len(expected_set)} emotions)")
        else:
            print(f"❌ MISS - No overlap")

        # Show what was missed and what was added
        missed = expected_set - pred_set
        extra = pred_set - expected_set

        if missed:
            print(f"   Missed: {sorted(missed)}")
        if extra:
            print(f"   Extra: {sorted(extra)}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)

    categories = {}
    for case in edge_cases:
        case_id = case["id"]
        text = case["text"]
        tasks = case["tasks"]
        expected_emotions = tasks.get("emotions", [])

        pred_emotions = client.get_emotions(text)

        expected_set = {e.lower() for e in expected_emotions}
        pred_set = {e.lower() for e in pred_emotions}
        overlap = expected_set.intersection(pred_set)

        category = case_id.replace("edge_", "").replace("_001", "").replace("_002", "").replace("_003", "")

        if category not in categories:
            categories[category] = {"hit": 0, "miss": 0, "total": 0}

        categories[category]["total"] += 1
        if len(overlap) > 0:
            categories[category]["hit"] += 1
        else:
            categories[category]["miss"] += 1

    for cat, stats in sorted(categories.items()):
        hit_rate = stats["hit"] / stats["total"] if stats["total"] > 0 else 0
        status = "✅" if hit_rate >= 0.5 else "⚠️" if hit_rate > 0 else "❌"
        print(f"{status} {cat.upper():20s}: {stats['hit']}/{stats['total']} hit ({hit_rate:.1%})")


if __name__ == "__main__":
    main()

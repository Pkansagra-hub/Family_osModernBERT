#!/usr/bin/env python3
"""Check for Indian hyperbole patterns in safety data."""
import json
from pathlib import Path

hyperbole_phrases = [
    "die of embarrassment",
    "killing me",
    "head is bursting",
    "going mad",
    "could just die",
    "will be the death of me",
    "dying of",
    "i'll die",
    "मर जाऊंगी",  # Hindi hyperbole
]

SAFETY_LABELS = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}


def check_file(filepath):
    results = {"total": 0, "correct_green_amber": 0, "samples": []}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").lower()
            label = d.get("label", -1)

            for phrase in hyperbole_phrases:
                if phrase.lower() in text:
                    results["total"] += 1
                    if label in [0, 1]:  # GREEN or AMBER
                        results["correct_green_amber"] += 1
                    results["samples"].append(
                        {
                            "label": SAFETY_LABELS.get(label, "UNKNOWN"),
                            "text": d.get("text", "")[:120],
                        }
                    )
                    break

    return results


def main():
    data_dir = Path("D:/Modeling_studio/data/familyos/safety")

    print("=" * 80)
    print("Indian Cultural Hyperbole Analysis in Safety Data")
    print("=" * 80)

    # Check gold data
    for split in ["gold", "silver"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        print(f"\n### {split.upper()} DATA ###")

        for jsonl_file in split_dir.glob("*.jsonl"):
            if "old" in jsonl_file.name:
                continue
            results = check_file(jsonl_file)

            print(f"\n{jsonl_file.name}:")
            print(f"  Total hyperbole samples: {results['total']}")
            print(f"  Correct (GREEN/AMBER): {results['correct_green_amber']}")

            if results["samples"]:
                print(f"\n  Examples:")
                for sample in results["samples"][:5]:
                    print(f"    [{sample['label']}] {sample['text']}")

    # Check for GREEN samples that contain 'die' etc
    print("\n" + "=" * 80)
    print("Checking GREEN samples with 'die' keyword")
    print("=" * 80)

    green_with_die = []
    for jsonl_file in (data_dir / "gold").glob("*.jsonl"):
        if "old" in jsonl_file.name:
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("label") == 0 and "die" in d.get("text", "").lower():
                    green_with_die.append(d.get("text", "")[:100])

    print(f"GREEN samples containing 'die': {len(green_with_die)}")
    for text in green_with_die[:10]:
        print(f"  - {text}")


if __name__ == "__main__":
    main()

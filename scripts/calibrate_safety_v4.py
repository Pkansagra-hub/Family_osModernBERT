#!/usr/bin/env python
"""
Safety Threshold Calibration for FamilyOS UltraBERT v4.0

This script evaluates and calibrates the safety head using validation data.
Works with the familyos_ultrabert package Client API.

Usage:
    python scripts/calibrate_safety_v4.py
    python scripts/calibrate_safety_v4.py --data data/familyos/safety/gold/validation.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Label mappings
BAND_ID_TO_NAME = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}
BAND_NAME_TO_ID = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3}


def load_validation_data(data_path: str) -> List[Tuple[str, str]]:
    """Load validation data as (text, label_name) tuples."""
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                text = sample.get("text", sample.get("content", ""))
                label = sample.get("label", 0)
                if isinstance(label, int):
                    label = BAND_ID_TO_NAME.get(label, "GREEN")
                samples.append((text, label))
    return samples


def evaluate_safety(client, samples: List[Tuple[str, str]]) -> Dict:
    """Evaluate safety predictions against ground truth."""

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    predictions = []
    ground_truth = []
    confidences = []
    errors = []

    print(f"\nEvaluating {len(samples)} samples...")

    for i, (text, expected) in enumerate(samples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(samples)}")

        result = client.analyze(text, capabilities=["safety_familyos"])
        pred = result.safety
        conf = result.safety_confidence

        predictions.append(pred)
        ground_truth.append(expected)
        confidences.append(conf)
        confusion[expected][pred] += 1

        if pred != expected:
            errors.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": pred,
                    "confidence": conf,
                }
            )

    # Calculate metrics
    metrics = {}
    for band in ["GREEN", "AMBER", "RED", "CRISIS"]:
        tp = confusion[band][band]
        fp = sum(confusion[other][band] for other in BAND_ID_TO_NAME.values() if other != band)
        fn = sum(confusion[band][other] for other in BAND_ID_TO_NAME.values() if other != band)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fnr = 1.0 - recall

        metrics[band] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fnr": fnr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Overall accuracy
    correct = sum(confusion[band][band] for band in BAND_ID_TO_NAME.values())
    total = len(samples)
    accuracy = correct / total if total > 0 else 0.0

    # Average confidence
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    min_conf = min(confidences) if confidences else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion": dict(confusion),
        "metrics": metrics,
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "errors": errors,
    }


def print_confusion_matrix(confusion: Dict):
    """Print confusion matrix."""
    bands = ["GREEN", "AMBER", "RED", "CRISIS"]

    print("\nConfusion Matrix (rows=expected, cols=predicted):")
    print("-" * 60)
    print(f"{'':12s}", end="")
    for band in bands:
        print(f"{band:>10s}", end="")
    print()
    print("-" * 60)

    for expected in bands:
        print(f"{expected:12s}", end="")
        for pred in bands:
            count = confusion.get(expected, {}).get(pred, 0)
            print(f"{count:>10d}", end="")
        print()
    print("-" * 60)


def print_report(results: Dict):
    """Print calibration report."""
    print()
    print("=" * 70)
    print("SAFETY CALIBRATION REPORT")
    print("=" * 70)

    print(
        f"\nOverall Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )
    print(f"Average Confidence: {results['avg_confidence']:.3f}")
    print(f"Min Confidence: {results['min_confidence']:.3f}")

    print_confusion_matrix(results["confusion"])

    print("\nPer-Band Metrics:")
    print("-" * 70)
    print(f"{'Band':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FNR':>10s}")
    print("-" * 70)

    for band in ["GREEN", "AMBER", "RED", "CRISIS"]:
        m = results["metrics"][band]
        print(
            f"{band:12s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['fnr']:>10.3f}"
        )

    # Quality gates
    print()
    print("=" * 70)
    print("QUALITY GATES")
    print("=" * 70)

    crisis_recall = results["metrics"]["CRISIS"]["recall"]
    red_recall = results["metrics"]["RED"]["recall"]

    gates = [
        ("CRISIS Recall >= 99%", crisis_recall >= 0.99, f"{crisis_recall:.2%}"),
        ("RED Recall >= 95%", red_recall >= 0.95, f"{red_recall:.2%}"),
        ("CRISIS Recall >= 100% (strict)", crisis_recall >= 1.0, f"{crisis_recall:.2%}"),
    ]

    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")

    # Show errors
    if results["errors"]:
        print()
        print("=" * 70)
        print(f"MISCLASSIFICATIONS ({len(results['errors'])} total)")
        print("=" * 70)

        # Group by type
        crisis_missed = [
            e for e in results["errors"] if e["expected"] == "CRISIS" and e["predicted"] != "CRISIS"
        ]
        crisis_false = [
            e for e in results["errors"] if e["expected"] != "CRISIS" and e["predicted"] == "CRISIS"
        ]

        if crisis_missed:
            print(f"\nCRISIS MISSED (FALSE NEGATIVES): {len(crisis_missed)}")
            for e in crisis_missed[:5]:
                print(f"  - \"{e['text'][:60]}...\"")
                print(
                    f"    Expected: {e['expected']}, Got: {e['predicted']} (conf: {e['confidence']:.3f})"
                )

        if crisis_false:
            print(f"\nCRISIS FALSE POSITIVES: {len(crisis_false)}")
            for e in crisis_false[:5]:
                print(f"  - \"{e['text'][:60]}...\"")
                print(
                    f"    Expected: {e['expected']}, Got: {e['predicted']} (conf: {e['confidence']:.3f})"
                )

        # Other errors
        other_errors = [
            e for e in results["errors"] if e not in crisis_missed and e not in crisis_false
        ]
        if other_errors:
            print(f"\nOTHER MISCLASSIFICATIONS: {len(other_errors)}")
            for e in other_errors[:10]:
                print(f"  - \"{e['text'][:50]}...\" | {e['expected']} -> {e['predicted']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate safety thresholds for UltraBERT v4")
    parser.add_argument(
        "--data",
        type=str,
        default="data/familyos/safety/gold/validation.jsonl",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/safety_calibration",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Import client
    print("Loading FamilyOS UltraBERT Client...")
    from familyos_ultrabert import Client

    client = Client(warmup=True, verbose=True)

    # Load data
    print(f"\nLoading validation data from {args.data}...")
    samples = load_validation_data(args.data)
    print(f"Loaded {len(samples)} samples")

    # Count by band
    counts = defaultdict(int)
    for _, label in samples:
        counts[label] += 1
    print("Distribution:", dict(counts))

    # Evaluate
    results = evaluate_safety(client, samples)

    # Print report
    print_report(results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "calibration_results.json"
    # Convert defaultdict to regular dict for JSON
    results_json = {
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "avg_confidence": results["avg_confidence"],
        "min_confidence": results["min_confidence"],
        "metrics": results["metrics"],
        "confusion": {k: dict(v) for k, v in results["confusion"].items()},
        "error_count": len(results["errors"]),
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()

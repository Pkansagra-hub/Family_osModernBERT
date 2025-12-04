#!/usr/bin/env python3
"""
Curate Gold Samples from Silver Data

This script selects high-quality samples from silver data to create gold training/validation sets.
Selection criteria:
1. Balanced class distribution
2. Clear, unambiguous examples
3. Diverse linguistic patterns
4. Culturally representative (Indian family context)
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Configuration
FAMILYOS_DIR = Path("data/familyos")
SAMPLES_PER_CLASS = {
    "safety": 50,  # Per class (GREEN=0, AMBER=1, RED=2, CRISIS=3)
    "ingress": 30,  # Per class (12 classes)
    "intents": 25,  # Per class
    "relations": 40,  # Per class
    "temporal": 50,  # Overall (harder to balance)
    "embeddings": 100,  # Triplets
    "ner_family": 50,  # Per entity type
}

TRAIN_SPLIT = 0.8  # 80% train, 20% validation


def load_silver_samples(task: str) -> list[dict[str, Any]]:
    """Load all silver samples for a task."""
    silver_dir = FAMILYOS_DIR / task / "silver"
    samples = []

    if not silver_dir.exists():
        print(f"  Warning: No silver directory for {task}")
        return samples

    for shard_file in sorted(silver_dir.glob("shard_*.jsonl")):
        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return samples


def quality_filter(sample: dict, task: str) -> bool:
    """Filter for quality samples."""

    # Handle token-based tasks (NER, temporal)
    if task in ["temporal", "ner_family"]:
        tokens = sample.get("tokens", [])
        if not tokens or len(tokens) < 3:
            return False
        if len(tokens) > 30:
            return False

        # Check for tags
        tags = sample.get("temporal_tags") or sample.get("ner_tags") or []
        if not tags:
            return False

        # Must have some non-O tags (not all zeros)
        if all(t == 0 or t == "O" for t in tags):
            return False

        # Check for family context
        text = " ".join(tokens).lower()
        family_terms = [
            "mom",
            "dad",
            "papa",
            "mummy",
            "nana",
            "nani",
            "dadi",
            "dada",
            "bhai",
            "didi",
            "grandma",
            "grandpa",
            "kids",
            "family",
            "son",
            "daughter",
            "brother",
            "sister",
            "emma",
            "priya",
            "arjun",
            "bunny",
        ]
        return any(term in text for term in family_terms)

    text = sample.get("text") or sample.get("anchor", "")

    # Minimum length
    if len(text) < 20:
        return False

    # Maximum length (avoid overly complex samples)
    if len(text) > 300:
        return False

    # Must have some family/cultural context for most tasks
    family_terms = [
        "papa",
        "mummy",
        "nana",
        "nani",
        "dadi",
        "dada",
        "bhai",
        "didi",
        "bhaiya",
        "chacha",
        "chachi",
        "mama",
        "mami",
        "mausi",
        "fufaji",
        "family",
        "home",
        "kids",
        "children",
        "son",
        "daughter",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "grandpa",
        "grandma",
        "cousin",
    ]

    text_lower = text.lower()
    has_family = any(term in text_lower for term in family_terms)

    # For embeddings, check anchor
    if task == "embeddings":
        return len(sample.get("anchor", "")) > 20 and len(sample.get("positive", "")) > 20

    return has_family or task in ["temporal", "embeddings"]


def select_balanced_samples(samples: list[dict], task: str, samples_per_class: int) -> list[dict]:
    """Select balanced samples across classes."""
    if task == "safety":
        # Group by label (0=GREEN, 1=AMBER, 2=RED, 3=CRISIS)
        by_class = defaultdict(list)
        for s in samples:
            if quality_filter(s, task):
                by_class[s["label"]].append(s)

        selected = []
        for label in [0, 1, 2, 3]:
            class_samples = by_class[label]
            random.shuffle(class_samples)
            selected.extend(class_samples[:samples_per_class])

        return selected

    elif task == "ingress":
        by_class = defaultdict(list)
        for s in samples:
            if quality_filter(s, task):
                by_class[s["label"]].append(s)

        selected = []
        for label in by_class.keys():
            class_samples = by_class[label]
            random.shuffle(class_samples)
            selected.extend(class_samples[:samples_per_class])

        return selected

    elif task == "intents":
        by_class = defaultdict(list)
        for s in samples:
            if quality_filter(s, task):
                by_class[s.get("label") or s.get("intent")].append(s)

        selected = []
        for label in by_class.keys():
            class_samples = by_class[label]
            random.shuffle(class_samples)
            selected.extend(class_samples[:samples_per_class])

        return selected

    elif task == "relations":
        by_class = defaultdict(list)
        for s in samples:
            if quality_filter(s, task):
                by_class[s.get("relation_type") or s.get("label")].append(s)

        selected = []
        for label in by_class.keys():
            class_samples = by_class[label]
            random.shuffle(class_samples)
            selected.extend(class_samples[:samples_per_class])

        return selected

    elif task == "embeddings":
        # For triplets, just filter and sample
        filtered = [s for s in samples if quality_filter(s, task)]
        random.shuffle(filtered)
        return filtered[: samples_per_class * 10]  # More samples for embeddings

    elif task == "temporal":
        # For NER-style temporal, sample by tag diversity
        filtered = [s for s in samples if quality_filter(s, task)]
        random.shuffle(filtered)
        return filtered[: samples_per_class * 5]  # 250 samples

    elif task == "ner_family":
        # Sample by entity type diversity - simpler approach for numeric tags
        filtered = [s for s in samples if quality_filter(s, task)]
        random.shuffle(filtered)
        return filtered[: samples_per_class * 5]  # 250 samples

    return samples[: samples_per_class * 4]


def save_gold_samples(samples: list[dict], task: str):
    """Save gold samples to train/validation files."""
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Create gold directory
    gold_dir = FAMILYOS_DIR / task / "gold"
    gold_dir.mkdir(exist_ok=True)

    # Save train
    train_file = gold_dir / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save validation
    val_file = gold_dir / "validation.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Also update main train/validation files
    main_train = FAMILYOS_DIR / task / "train.jsonl"
    main_val = FAMILYOS_DIR / task / "validation.jsonl"

    with open(main_train, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(main_val, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(train_samples), len(val_samples)


def print_sample_examples(samples: list[dict], task: str, n: int = 3):
    """Print a few example samples."""
    print(f"\n  Sample examples from {task}:")
    for s in samples[:n]:
        if task == "embeddings":
            print(f"    Anchor: {s.get('anchor', '')[:60]}...")
        elif task in ["ner_family", "temporal"]:
            tokens = s.get("tokens", [])[:10]
            tags = s.get("ner_tags") or s.get("temporal_tags") or []
            print(f"    Tokens: {' '.join(tokens)}... | Tags: {tags[:10]}")
        else:
            print(f"    Text: {s.get('text', '')[:60]}... -> {s.get('label')}")


def main():
    print("=" * 60)
    print("Curating Gold Samples from Silver Data")
    print("=" * 60)

    random.seed(42)  # Reproducibility

    total_train = 0
    total_val = 0

    for task in [
        "safety",
        "ingress",
        "intents",
        "relations",
        "embeddings",
        "temporal",
        "ner_family",
    ]:
        print(f"\n[{task.upper()}]")

        # Load silver
        samples = load_silver_samples(task)
        print(f"  Loaded {len(samples)} silver samples")

        if not samples:
            print(f"  Skipping {task} - no silver data")
            continue

        # Select balanced samples
        selected = select_balanced_samples(samples, task, SAMPLES_PER_CLASS[task])
        print(f"  Selected {len(selected)} quality samples")

        # Print examples
        print_sample_examples(selected, task)

        # Save
        n_train, n_val = save_gold_samples(selected, task)
        print(f"  Saved: {n_train} train, {n_val} validation")

        total_train += n_train
        total_val += n_val

    print("\n" + "=" * 60)
    print(
        f"TOTAL: {total_train} train + {total_val} validation = {total_train + total_val} gold samples"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

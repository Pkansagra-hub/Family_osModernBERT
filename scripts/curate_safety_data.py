#!/usr/bin/env python3
"""
Curate Safety Datasets for Multi-Label Toxicity Detection.

Combines multiple sources to create a balanced dataset:
1. Civil Comments (HuggingFace) - base toxicity labels
2. Suicide Prediction dataset - for self_harm/severe_toxic
3. BeaverTails dataset - for dangerous content

Creates a balanced curated dataset with 8 classes:
[toxic, severe_toxic, obscene, threat, insult, identity_hate, sexually_explicit, profanity]

Target distribution (balanced):
- toxic: ~60% (umbrella category)
- severe_toxic: ~12-15% (boosted from suicide + beavertails)
- obscene: ~15-20%
- threat: ~15-20%
- insult: ~40-45%
- identity_hate: ~20-25%
- sexually_explicit: ~12-15%
- profanity: ~12-15%
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Directories
OUTPUT_DIR = Path("data/public/civil_comments_curated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Toxicity threshold for Civil Comments
THRESHOLD = 0.5

# 8-class schema
OUTPUT_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "sexually_explicit",
    "profanity",
]


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} samples to {filepath}")


def binarize_civil_comments(example):
    """Convert Civil Comments scores to 8-class multi-hot."""
    labels = [0] * 8

    # toxic (index 0)
    if example.get("toxicity", 0) >= THRESHOLD:
        labels[0] = 1

    # severe_toxic (index 1)
    if example.get("severe_toxicity", 0) >= THRESHOLD:
        labels[1] = 1

    # obscene (index 2)
    if example.get("obscene", 0) >= THRESHOLD:
        labels[2] = 1

    # threat (index 3)
    if example.get("threat", 0) >= THRESHOLD:
        labels[3] = 1

    # insult (index 4)
    if example.get("insult", 0) >= THRESHOLD:
        labels[4] = 1

    # identity_hate (index 5)
    if example.get("identity_attack", 0) >= THRESHOLD:
        labels[5] = 1

    # sexually_explicit (index 6)
    if example.get("sexual_explicit", 0) >= THRESHOLD:
        labels[6] = 1

    # profanity (index 7) - derive from obscene + high toxicity
    if example.get("obscene", 0) >= 0.3 and example.get("toxicity", 0) >= THRESHOLD:
        labels[7] = 1

    return labels


def download_civil_comments():
    """Download and process Civil Comments dataset."""
    print("\n" + "=" * 60)
    print("Step 1: Loading Civil Comments dataset...")
    print("=" * 60)

    dataset = load_dataset("civil_comments")

    samples = {"train": [], "validation": []}

    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")
        for example in tqdm(dataset[split], desc=split):
            text = example.get("text", "").strip()
            if not text or len(text) < 10:
                continue

            labels = binarize_civil_comments(example)

            # Only keep samples that have at least one positive label (toxic samples)
            # OR sample ~10% of clean samples for negative examples
            if sum(labels) > 0 or np.random.random() < 0.1:
                samples[split].append({"text": text, "labels": labels, "source": "civil_comments"})

    print(f"\nCivil Comments: {len(samples['train']):,} train, {len(samples['validation']):,} val")
    return samples


def download_suicide_dataset():
    """Download suicide prediction dataset for self_harm/severe_toxic."""
    print("\n" + "=" * 60)
    print("Step 2: Loading Suicide Prediction dataset...")
    print("=" * 60)

    try:
        ds = load_dataset("vibhorag101/suicide_prediction_dataset_phr", trust_remote_code=True)

        samples = []
        for item in tqdm(ds["train"], desc="Processing"):
            text = item.get("text", "").strip()
            if not text or len(text) < 10:
                continue

            # Dataset uses 'label' field with value 'suicide' or 'non-suicide'
            is_suicidal = item.get("label") == "suicide"

            if is_suicidal:
                # Map to: toxic=1, severe_toxic=1 (self-harm is severe)
                labels = [1, 1, 0, 0, 0, 0, 0, 0]
                samples.append({"text": text, "labels": labels, "source": "suicide_pred"})

        print(f"Suicide dataset: {len(samples):,} positive samples")
        return samples

    except Exception as e:
        print(f"  ❌ Failed to load suicide dataset: {e}")
        return []


def download_beavertails():
    """Download BeaverTails dataset for dangerous content."""
    print("\n" + "=" * 60)
    print("Step 3: Loading BeaverTails dataset...")
    print("=" * 60)

    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", trust_remote_code=True)

        samples = []
        category_counts = Counter()

        for item in tqdm(ds["330k_train"], desc="Processing"):
            if item.get("is_safe", True):
                continue  # Skip safe samples

            text = f"{item.get('prompt', '')} {item.get('response', '')}".strip()
            if not text or len(text) < 10:
                continue

            categories = item.get("category", {})

            # Map BeaverTails categories to our 8-class schema
            labels = [0] * 8

            # toxic (always 1 for unsafe content)
            labels[0] = 1

            # severe_toxic - self-harm, violence, terrorism
            if categories.get("self_harm") or categories.get("terrorism,organized_crime"):
                labels[1] = 1
                category_counts["severe_toxic"] += 1

            # obscene - offensive language
            if categories.get("hate_speech,offensive_language"):
                labels[2] = 1
                category_counts["obscene"] += 1

            # threat - violence, terrorism
            if categories.get("violence,aiding_and_abetting,incitement") or categories.get("terrorism,organized_crime"):
                labels[3] = 1
                category_counts["threat"] += 1

            # identity_hate - discrimination, hate speech
            if categories.get("discrimination,stereotype,injustice") or categories.get("hate_speech,offensive_language"):
                labels[5] = 1
                category_counts["identity_hate"] += 1

            # sexually_explicit
            if categories.get("sexually_explicit,adult_content"):
                labels[6] = 1
                category_counts["sexually_explicit"] += 1

            # profanity - offensive language
            if categories.get("hate_speech,offensive_language"):
                labels[7] = 1
                category_counts["profanity"] += 1

            samples.append({"text": text, "labels": labels, "source": "beavertails"})

        print(f"BeaverTails: {len(samples):,} unsafe samples")
        print(f"Category distribution: {dict(category_counts)}")
        return samples

    except Exception as e:
        print(f"  ❌ Failed to load BeaverTails: {e}")
        return []


def balance_dataset(samples, target_total=200000):
    """Balance the dataset by oversampling rare classes."""
    print("\n" + "=" * 60)
    print("Step 4: Balancing dataset...")
    print("=" * 60)

    # Count current distribution
    label_counts = Counter()
    for sample in samples:
        for i, label in enumerate(sample["labels"]):
            if label == 1:
                label_counts[OUTPUT_LABELS[i]] += 1

    total = len(samples)
    print(f"Before balancing: {total:,} samples")
    for label_name in OUTPUT_LABELS:
        count = label_counts[label_name]
        pct = (count / total) * 100
        print(f"  {label_name:20s}: {count:8,} ({pct:5.2f}%)")

    # Group samples by rare class membership
    rare_classes = ["severe_toxic", "threat", "sexually_explicit", "profanity"]
    rare_indices = {cls: [] for cls in rare_classes}

    for idx, sample in enumerate(samples):
        for cls in rare_classes:
            cls_idx = OUTPUT_LABELS.index(cls)
            if sample["labels"][cls_idx] == 1:
                rare_indices[cls].append(idx)

    # Oversample rare classes
    additional = []
    target_per_class = int(total * 0.12)  # Target ~12% for each rare class

    for cls in rare_classes:
        indices = rare_indices[cls]
        current = len(indices)
        if current > 0 and current < target_per_class:
            needed = min(target_per_class - current, current * 10)  # Max 10x oversample
            sampled = np.random.choice(indices, size=needed, replace=True)
            for idx in sampled:
                additional.append(samples[idx].copy())
            print(f"  Oversampled {cls}: {current} -> {current + needed}")

    samples.extend(additional)
    np.random.shuffle(samples)

    # Limit to target
    if len(samples) > target_total:
        samples = samples[:target_total]

    return samples


def compute_pos_weights(samples):
    """Compute pos_weight for each class."""
    print("\n" + "=" * 60)
    print("Computing final pos_weights...")
    print("=" * 60)

    label_counts = Counter()
    for sample in samples:
        for i, label in enumerate(sample["labels"]):
            if label == 1:
                label_counts[OUTPUT_LABELS[i]] += 1

    total = len(samples)
    print(f"\nFinal distribution ({total:,} samples):")

    pos_weights = []
    for label_name in OUTPUT_LABELS:
        count = label_counts[label_name]
        pct = (count / total) * 100 if total > 0 else 0
        # pos_weight = 100 / percentage (inverse frequency)
        weight = round(100 / pct, 1) if pct > 0 else 10.0
        weight = min(weight, 20.0)  # Cap at 20
        pos_weights.append(weight)
        print(f"  {label_name:20s}: {count:8,} ({pct:5.2f}%) | pos_weight: {weight}")

    print(f"\n✅ RECOMMENDED pos_weight: {pos_weights}")
    return pos_weights


def main():
    np.random.seed(42)

    print("=" * 60)
    print("CURATING SAFETY DATASET")
    print("=" * 60)

    # Step 1: Civil Comments
    civil_samples = download_civil_comments()

    # Step 2: Suicide dataset
    suicide_samples = download_suicide_dataset()

    # Step 3: BeaverTails
    beaver_samples = download_beavertails()

    # Combine all sources
    print("\n" + "=" * 60)
    print("Combining datasets...")
    print("=" * 60)

    all_train = civil_samples["train"] + suicide_samples + beaver_samples
    all_val = civil_samples["validation"]

    print(f"Combined train: {len(all_train):,}")
    print(f"  - Civil Comments: {len(civil_samples['train']):,}")
    print(f"  - Suicide: {len(suicide_samples):,}")
    print(f"  - BeaverTails: {len(beaver_samples):,}")

    # Balance training set
    balanced_train = balance_dataset(all_train, target_total=200000)

    # Compute pos_weights
    pos_weights = compute_pos_weights(balanced_train)

    # Save datasets
    print("\n" + "=" * 60)
    print("Saving curated datasets...")
    print("=" * 60)

    save_jsonl(balanced_train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(all_val[:20000], OUTPUT_DIR / "validation.jsonl")

    # Save metadata
    metadata = {
        "sources": ["civil_comments", "suicide_prediction", "beavertails"],
        "labels": OUTPUT_LABELS,
        "train_samples": len(balanced_train),
        "validation_samples": min(len(all_val), 20000),
        "suggested_pos_weight": pos_weights,
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Done! Curated dataset saved to: {OUTPUT_DIR}")
    print(f"\n📋 UPDATE YOUR CONFIG with:")
    print(f"   pos_weight: {pos_weights}")


if __name__ == "__main__":
    main()

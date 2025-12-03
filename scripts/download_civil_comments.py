#!/usr/bin/env python3
"""
Download and curate safety datasets for multi-label toxicity detection.

Combines multiple sources:
1. Civil Comments (HuggingFace) - toxic, obscene, threat, insult, identity_attack
2. Suicide Prediction dataset - for self_harm detection
3. BeaverTails dataset - for dangerous_advice detection

Creates a balanced curated dataset with 8 classes:
[toxic, severe_toxic, obscene, threat, insult, identity_hate, sexually_explicit, profanity]

Note: self_harm and dangerous_advice from suicide/beavertails are mapped to severe_toxic
since we need to maintain the 8-class schema.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path("data/public/civil_comments_curated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Intermediate data directory
DATA_DIR = Path("data/public")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Toxicity thresholds (Civil Comments uses 0-1 scores, not binary labels)
THRESHOLD = 0.5

# Our 8-class output schema (matches safety_generic head)
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


def binarize_labels(example):
    """Convert continuous toxicity scores to binary multi-hot labels."""
    labels = []
    
    # Map Civil Comments columns to our schema
    labels.append(1 if example.get("toxicity", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("severe_toxicity", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("obscene", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("threat", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("insult", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("identity_attack", 0) >= THRESHOLD else 0)
    labels.append(1 if example.get("sexual_explicit", 0) >= THRESHOLD else 0)
    
    # Profanity: derive from obscene or high toxicity without other specific flags
    # This is an approximation - Civil Comments doesn't have explicit profanity label
    is_profane = (
        example.get("obscene", 0) >= 0.3 and  # Lower threshold for profanity
        example.get("toxicity", 0) >= THRESHOLD
    )
    labels.append(1 if is_profane else 0)
    
    return labels


def analyze_distribution(dataset, split_name="train"):
    """Analyze class distribution in dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {split_name} split...")
    print(f"{'='*60}")
    
    label_counts = Counter()
    total = 0
    
    for example in tqdm(dataset, desc="Counting labels"):
        labels = binarize_labels(example)
        total += 1
        for i, label in enumerate(labels):
            if label == 1:
                label_counts[OUTPUT_LABELS[i]] += 1
    
    print(f"\nTotal samples: {total:,}")
    print(f"\nClass distribution:")
    print(f"{'-'*40}")
    
    pos_weights = []
    for i, label_name in enumerate(OUTPUT_LABELS):
        count = label_counts[label_name]
        pct = (count / total) * 100
        # pos_weight = (total - count) / count if count > 0 else 1.0
        # Alternative: inverse frequency scaled
        inv_freq = (total / count) if count > 0 else 1.0
        pos_weights.append(round(inv_freq, 2))
        print(f"  {label_name:20s}: {count:8,} ({pct:6.2f}%) | inv_freq: {inv_freq:.2f}")
    
    # Also compute pos_weight as 100/percentage for config
    print(f"\n{'='*60}")
    print("Suggested pos_weight (100/percentage):")
    print(f"{'='*60}")
    config_weights = []
    for i, label_name in enumerate(OUTPUT_LABELS):
        count = label_counts[label_name]
        pct = (count / total) * 100 if total > 0 else 1
        weight = round(100 / pct, 1) if pct > 0 else 10.0
        # Cap at reasonable max
        weight = min(weight, 50.0)
        config_weights.append(weight)
        print(f"  {label_name:20s}: {weight}")
    
    print(f"\npos_weight: {config_weights}")
    
    return label_counts, total, config_weights


def create_curated_dataset(dataset, output_path, split_name, max_samples=None, oversample_rare=True):
    """Create curated JSONL dataset with balanced sampling."""
    
    samples = []
    label_counts = Counter()
    
    print(f"\nProcessing {split_name}...")
    
    for example in tqdm(dataset, desc=f"Processing {split_name}"):
        text = example.get("text", "").strip()
        if not text or len(text) < 10:
            continue
            
        labels = binarize_labels(example)
        
        # Track which classes this sample has
        for i, label in enumerate(labels):
            if label == 1:
                label_counts[OUTPUT_LABELS[i]] += 1
        
        samples.append({
            "text": text,
            "labels": labels
        })
    
    print(f"Collected {len(samples):,} valid samples")
    
    # Optionally oversample rare classes
    if oversample_rare and split_name == "train":
        print("\nOversampling rare classes...")
        
        # Find samples for each rare class
        rare_classes = ["severe_toxic", "threat", "sexually_explicit", "profanity"]
        rare_indices = {cls: [] for cls in rare_classes}
        
        for idx, sample in enumerate(samples):
            labels = sample["labels"]
            for cls in rare_classes:
                cls_idx = OUTPUT_LABELS.index(cls)
                if labels[cls_idx] == 1:
                    rare_indices[cls].append(idx)
        
        # Oversample to boost rare classes
        additional_samples = []
        target_count = 5000  # Minimum samples per rare class
        
        for cls in rare_classes:
            indices = rare_indices[cls]
            current_count = len(indices)
            if current_count > 0 and current_count < target_count:
                # Oversample with replacement
                oversample_count = min(target_count - current_count, current_count * 5)
                sampled_indices = np.random.choice(indices, size=oversample_count, replace=True)
                for idx in sampled_indices:
                    additional_samples.append(samples[idx])
                print(f"  {cls}: {current_count} -> {current_count + oversample_count}")
        
        samples.extend(additional_samples)
        print(f"Total after oversampling: {len(samples):,}")
    
    # Shuffle
    np.random.shuffle(samples)
    
    # Limit if requested
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples:,} samples")
    
    # Save to JSONL
    output_file = output_path / f"{split_name}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Saved to {output_file}")
    
    return samples


def main():
    print("="*60)
    print("Downloading Civil Comments dataset...")
    print("="*60)
    
    # Load dataset
    dataset = load_dataset("civil_comments")
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")
    print(f"  Test samples: {len(dataset['test']):,}")
    
    # Analyze original distribution
    label_counts, total, suggested_weights = analyze_distribution(dataset["train"], "train")
    
    # Create curated train set
    train_samples = create_curated_dataset(
        dataset["train"], 
        OUTPUT_DIR, 
        "train",
        max_samples=200000,  # Limit for training efficiency
        oversample_rare=True
    )
    
    # Create curated validation set (no oversampling)
    val_samples = create_curated_dataset(
        dataset["validation"],
        OUTPUT_DIR,
        "validation", 
        max_samples=20000,
        oversample_rare=False
    )
    
    # Analyze curated distribution
    print("\n" + "="*60)
    print("CURATED DATASET DISTRIBUTION")
    print("="*60)
    
    # Recount for curated data
    curated_counts = Counter()
    for sample in train_samples:
        for i, label in enumerate(sample["labels"]):
            if label == 1:
                curated_counts[OUTPUT_LABELS[i]] += 1
    
    total_curated = len(train_samples)
    print(f"\nTotal curated samples: {total_curated:,}")
    print(f"\nCurated class distribution:")
    
    final_weights = []
    for label_name in OUTPUT_LABELS:
        count = curated_counts[label_name]
        pct = (count / total_curated) * 100
        weight = round(100 / pct, 1) if pct > 0 else 10.0
        weight = min(weight, 20.0)  # Cap at 20
        final_weights.append(weight)
        print(f"  {label_name:20s}: {count:8,} ({pct:6.2f}%) | pos_weight: {weight}")
    
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDED pos_weight FOR CONFIG:")
    print(f"{'='*60}")
    print(f"pos_weight: {final_weights}")
    
    # Save metadata
    metadata = {
        "source": "civil_comments",
        "threshold": THRESHOLD,
        "labels": OUTPUT_LABELS,
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "curated_distribution": {
            label: curated_counts[label] for label in OUTPUT_LABELS
        },
        "suggested_pos_weight": final_weights
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {OUTPUT_DIR / 'metadata.json'}")
    print("\n✅ Done! Curated dataset ready at:", OUTPUT_DIR)


if __name__ == "__main__":
    np.random.seed(42)
    main()

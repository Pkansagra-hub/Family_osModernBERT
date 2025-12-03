"""Check the curated safety dataset format and statistics."""

import json
from collections import Counter
from pathlib import Path

data_dir = Path("data/public/civil_comments_curated")

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "self_harm",
    "dangerous_advice",
]

for split in ["train", "validation"]:
    file_path = data_dir / f"{split}.jsonl"
    if not file_path.exists():
        print(f"❌ {split}.jsonl not found!")
        continue

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"Total samples: {len(samples):,}")

    # Check sample format
    sample = samples[0]
    print(f"\nSample format:")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Text length: {len(sample['text'])}")
    print(f"  Labels type: {type(sample['labels']).__name__}")
    print(f"  Labels: {sample['labels']}")

    # Show first 3 samples
    print(f"\nFirst 3 samples:")
    for i, s in enumerate(samples[:3]):
        label_names = [LABELS[j] for j, v in enumerate(s["labels"]) if v == 1]
        print(f"  {i+1}. text: {s['text'][:80]!r}...")
        print(f"     labels: {label_names}")

    # Calculate label distribution
    label_counts = [0] * 8
    for s in samples:
        for i, v in enumerate(s["labels"]):
            label_counts[i] += v

    print(f"\nLabel distribution:")
    for i, (name, count) in enumerate(zip(LABELS, label_counts)):
        pct = count / len(samples) * 100
        print(f"  {name:18s}: {count:,} ({pct:.1f}%)")

print("\n✅ Dataset check complete!")

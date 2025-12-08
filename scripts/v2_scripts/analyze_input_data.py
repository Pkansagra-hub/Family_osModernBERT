"""Analyze all 7 familyos data folders (excluding unified)"""

import json
from collections import Counter
from pathlib import Path

folders = [
    "embeddings",
    "emotions",
    "ingress",
    "intents",
    "ner_family",
    "relations",
    "safety",
    "temporal",
]
base = Path("D:/Modeling_studio/data/familyos")

for folder in folders:
    folder_path = base / folder
    print(f'\n{"="*60}')
    print(f"=== {folder.upper()} ===")
    print("=" * 60)

    # Find silver subfolder or train.jsonl
    silver = folder_path / "silver"
    if silver.exists():
        train_file = silver / "train.jsonl"
    else:
        train_file = folder_path / "train.jsonl"

    if not train_file.exists():
        # Try to find any jsonl
        jsonl_files = list(folder_path.rglob("*.jsonl"))
        if jsonl_files:
            train_file = jsonl_files[0]
            print(f"Using: {train_file.relative_to(base)}")
        else:
            print("NO DATA FOUND")
            continue
    else:
        print(f"Using: {train_file.relative_to(base)}")

    # Count total
    total_count = sum(1 for _ in open(train_file, encoding="utf-8"))
    print(f"Total samples: {total_count:,}")

    # Analyze sample
    samples = []
    with open(train_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break  # Sample 2000
            try:
                samples.append(json.loads(line.strip()))
            except:
                pass

    if not samples:
        print("EMPTY OR INVALID")
        continue

    # Show structure of first sample
    print(f"Fields: {list(samples[0].keys())}")

    # Show 3 examples
    print("\nExamples:")
    for i, s in enumerate(samples[:3]):
        text = s.get("text", s.get("sentence", s.get("input", str(s))))[:80]
        label = s.get("label", s.get("labels", s.get("emotions", s.get("safety_familyos", "N/A"))))
        print(f'  [{i+1}] "{text}..." => {label}')

    # Analyze labels
    print("\nLabel Distribution:")
    labels = Counter()
    for s in samples:
        lbl = s.get("label", s.get("labels", s.get("emotions", s.get("safety_familyos", None))))
        if isinstance(lbl, list):
            for l in lbl:
                labels[l] += 1
        elif lbl:
            labels[lbl] += 1

    for l, c in labels.most_common(15):
        pct = c / len(samples) * 100
        print(f"  {l}: {c} ({pct:.1f}%)")

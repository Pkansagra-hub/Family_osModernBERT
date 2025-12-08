#!/usr/bin/env python3
"""Check safety data distribution."""
import json
from collections import Counter
from pathlib import Path

safety_dir = Path("D:/Modeling_studio/data/familyos/safety")
labels = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}

for split in ["gold", "silver"]:
    split_dir = safety_dir / split
    if not split_dir.exists():
        continue

    all_labels = Counter()
    total = 0
    for f in split_dir.glob("*.jsonl"):
        if "old" in f.name:
            continue
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line)
                all_labels[d.get("label", -1)] += 1
                total += 1

    print(f"{split.upper()}: {total} samples")
    for label_id, count in sorted(all_labels.items()):
        pct = count * 100 / total if total else 0
        name = labels.get(label_id, "UNK")
        print(f"  {name}: {count} ({pct:.1f}%)")
    print()

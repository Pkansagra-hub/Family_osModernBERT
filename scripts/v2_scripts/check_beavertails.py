"""Check BeaverTails format and categories."""

import json
from pathlib import Path

print("=" * 60)
print("BEAVERTAILS Sample")
print("=" * 60)
with open("data/public/beavertails/train.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 2:
            item = json.loads(line)
            print(f"Sample {i}:")
            print(f"  is_safe: {item['is_safe']}")
            print(f"  categories: {item['categories']}")
            print(f"  text: {item['text'][:150]}...")
        else:
            break

# Count category distribution
print()
print("Category distribution (full dataset):")
cat_counts = {}
safe_count = 0
unsafe_count = 0
with open("data/public/beavertails/train.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["is_safe"]:
            safe_count += 1
        else:
            unsafe_count += 1
        for cat, val in item["categories"].items():
            if val:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

print(f"  Safe: {safe_count:,}")
print(f"  Unsafe: {unsafe_count:,}")
print()
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {cnt:,}")

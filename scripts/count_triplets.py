"""Count embedding triplets."""

from pathlib import Path

files = list(Path("data/familyos/embeddings/silver_synthetic").glob("triplets_*.jsonl"))
print(f"Triplet files: {len(files)}")

total = 0
for f in files:
    with open(f, encoding="utf-8") as fp:
        total += sum(1 for _ in fp)

print(f"Total triplets: {total:,}")

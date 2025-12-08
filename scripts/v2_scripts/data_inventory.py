"""Print complete data inventory."""

from pathlib import Path


def count_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def count_folder(folder):
    total = 0
    for f in Path(folder).glob("*.jsonl"):
        total += count_jsonl(f)
    return total


print("=" * 70)
print("📊 COMPLETE DATA INVENTORY")
print("=" * 70)

# Public datasets
print()
print("PUBLIC DATASETS (data/public/)")
print("-" * 70)
public_path = Path("data/public")
public_total = 0
for subdir in sorted(public_path.iterdir()):
    if subdir.is_dir():
        train = count_jsonl(subdir / "train.jsonl")
        val = count_jsonl(subdir / "validation.jsonl")
        test = count_jsonl(subdir / "test.jsonl")
        total = count_folder(subdir)
        print(f"  {subdir.name:30s} train={train:>10,}  val={val:>8,}  test={test:>8,}")
        public_total += total

print("-" * 70)
print(f"  {'PUBLIC TOTAL':30s} {public_total:>10,}")

# FamilyOS datasets
print()
print("FAMILYOS DATASETS (data/familyos/)")
print("-" * 70)
familyos_path = Path("data/familyos")
familyos_total = 0
for task in sorted(familyos_path.iterdir()):
    if task.is_dir():
        silver = count_folder(task / "silver") if (task / "silver").exists() else 0
        gold = count_folder(task / "gold") if (task / "gold").exists() else 0
        print(f"  {task.name:30s} silver={silver:>8,}  gold={gold:>6,}")
        familyos_total += silver + gold

print("-" * 70)
print(f"  {'FAMILYOS TOTAL':30s} {familyos_total:>10,}")

print()
print("=" * 70)
print(f"  GRAND TOTAL: {public_total + familyos_total:,} samples")
print("=" * 70)

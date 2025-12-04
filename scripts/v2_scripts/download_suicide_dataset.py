"""Download suicide_prediction dataset."""

import json
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

output_dir = Path("data/public/suicide_prediction")
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("vibhorag101/suicide_prediction_dataset_phr", trust_remote_code=True)
print(f"Train samples: {len(ds['train']):,}")
print(f"Test samples: {len(ds['test']):,}")

# Save train
data = []
for item in tqdm(ds["train"], desc="Processing train"):
    data.append(
        {"text": item["text"], "label": 1 if item["label"] == "suicide" else 0}  # 1=suicidal
    )

with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved {len(data):,} train samples")

# Save test as validation
data = []
for item in tqdm(ds["test"], desc="Processing test"):
    data.append({"text": item["text"], "label": 1 if item["label"] == "suicide" else 0})

with open(output_dir / "validation.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved {len(data):,} validation samples")

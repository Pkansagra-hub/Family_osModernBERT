"""Download all public datasets used in Stage A and Stage B training.

This script downloads all HuggingFace datasets to data/public/ for curation.
Run this BEFORE training to ensure all data is available locally.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Output directory
DATA_DIR = Path("data/public")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(data, output_path):
    """Save data as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Writing {output_path.name}"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved {len(data):,} samples to {output_path}")


def download_ner_conll2003():
    """Download CoNLL-2003 NER dataset."""
    print("\n" + "=" * 60)
    print("Downloading: conll2003 (NER)")
    print("=" * 60)

    output_dir = DATA_DIR / "conll2003"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("conll2003", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        if split in ds:
            data = []
            for item in ds[split]:
                data.append(
                    {
                        "tokens": item["tokens"],
                        "ner_tags": item["ner_tags"],
                        "pos_tags": item["pos_tags"],
                        "chunk_tags": item["chunk_tags"],
                    }
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_ner_wikineural():
    """Download WikiNeural NER dataset (English)."""
    print("\n" + "=" * 60)
    print("Downloading: tner/wikineural (NER - English)")
    print("=" * 60)

    output_dir = DATA_DIR / "wikineural"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("tner/wikineural", "en", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        if split in ds:
            data = []
            for item in ds[split]:
                data.append(
                    {"tokens": item["tokens"], "ner_tags": item["tags"]}  # tner uses 'tags'
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_sentiment_sst2():
    """Download Stanford Sentiment Treebank v2."""
    print("\n" + "=" * 60)
    print("Downloading: stanfordnlp/sst2 (Sentiment)")
    print("=" * 60)

    output_dir = DATA_DIR / "sst2"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("stanfordnlp/sst2", trust_remote_code=True)

    for split in ["train", "validation"]:
        if split in ds:
            data = []
            for item in ds[split]:
                data.append(
                    {"text": item["sentence"], "label": item["label"]}  # 0=negative, 1=positive
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_emotions_goemotions():
    """Download GoEmotions (simplified - 12 labels)."""
    print("\n" + "=" * 60)
    print("Downloading: go_emotions (Emotions - simplified)")
    print("=" * 60)

    output_dir = DATA_DIR / "goemotions"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        if split in ds:
            data = []
            for item in ds[split]:
                data.append(
                    {"text": item["text"], "labels": item["labels"]}  # List of label indices
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_nli_mnli():
    """Download MultiNLI dataset."""
    print("\n" + "=" * 60)
    print("Downloading: multi_nli (NLI)")
    print("=" * 60)

    output_dir = DATA_DIR / "mnli"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("multi_nli", trust_remote_code=True)

    # Train split
    data = []
    for item in tqdm(ds["train"], desc="Processing train"):
        data.append(
            {
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": item["label"],  # 0=entailment, 1=neutral, 2=contradiction
            }
        )
    save_jsonl(data, output_dir / "train.jsonl")

    # Validation matched
    data = []
    for item in ds["validation_matched"]:
        data.append(
            {"premise": item["premise"], "hypothesis": item["hypothesis"], "label": item["label"]}
        )
    save_jsonl(data, output_dir / "validation.jsonl")


def download_nli_snli():
    """Download SNLI dataset."""
    print("\n" + "=" * 60)
    print("Downloading: stanfordnlp/snli (NLI)")
    print("=" * 60)

    output_dir = DATA_DIR / "snli"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("stanfordnlp/snli", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        if split in ds:
            data = []
            for item in ds[split]:
                # Skip examples with label -1 (no gold label)
                if item["label"] == -1:
                    continue
                data.append(
                    {
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label": item["label"],
                    }
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_embedding_stsb():
    """Download STS Benchmark dataset."""
    print("\n" + "=" * 60)
    print("Downloading: sentence-transformers/stsb (Embeddings)")
    print("=" * 60)

    output_dir = DATA_DIR / "stsb"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("sentence-transformers/stsb", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        if split in ds:
            data = []
            for item in ds[split]:
                data.append(
                    {
                        "sentence1": item["sentence1"],
                        "sentence2": item["sentence2"],
                        "score": item["score"],  # 0-5 similarity
                    }
                )
            save_jsonl(data, output_dir / f"{split}.jsonl")


def download_embedding_allnli():
    """Download All-NLI for embedding training (pair-score format)."""
    print("\n" + "=" * 60)
    print("Downloading: sentence-transformers/all-nli (Embeddings)")
    print("=" * 60)

    output_dir = DATA_DIR / "allnli"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    ds = load_dataset("sentence-transformers/all-nli", "pair-score", trust_remote_code=True)

    split_map = {"train": "train", "dev": "validation", "test": "test"}
    for src_split, dst_split in split_map.items():
        if src_split in ds:
            data = []
            for item in tqdm(ds[src_split], desc=f"Processing {src_split}"):
                data.append(
                    {
                        "sentence1": item["sentence1"],
                        "sentence2": item["sentence2"],
                        "score": item["score"],
                    }
                )
            save_jsonl(data, output_dir / f"{dst_split}.jsonl")


def download_safety_datasets():
    """Download additional safety datasets."""
    print("\n" + "=" * 60)
    print("Downloading: Safety datasets (self_harm, dangerous_advice)")
    print("=" * 60)

    # Suicide prediction dataset (for self_harm)
    output_dir = DATA_DIR / "suicide_prediction"
    if not (output_dir / "train.jsonl").exists():
        print("\n  📥 Loading vibhorag101/suicide_prediction_dataset_phr...")
        try:
            ds = load_dataset("vibhorag101/suicide_prediction_dataset_phr", trust_remote_code=True)
            data = []
            for item in tqdm(ds["train"], desc="Processing"):
                data.append(
                    {
                        "text": item["Statement"],
                        "label": 1 if item["Status"] == "Indication" else 0,  # 1=suicidal
                    }
                )
            save_jsonl(data, output_dir / "train.jsonl")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    else:
        print("  ⏭️  suicide_prediction already downloaded")

    # BeaverTails dataset (for dangerous_advice)
    output_dir = DATA_DIR / "beavertails"
    if not (output_dir / "train.jsonl").exists():
        print("\n  📥 Loading PKU-Alignment/BeaverTails...")
        try:
            ds = load_dataset("PKU-Alignment/BeaverTails", trust_remote_code=True)
            data = []
            for item in tqdm(ds["330k_train"], desc="Processing"):
                # Extract safety categories
                labels = item.get("category", {})
                data.append(
                    {
                        "text": item["prompt"] + " " + item["response"],
                        "is_safe": item["is_safe"],
                        "categories": labels,
                    }
                )
            save_jsonl(data, output_dir / "train.jsonl")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    else:
        print("  ⏭️  beavertails already downloaded")


def download_amazon_polarity():
    """Download Amazon Polarity (optional - large sentiment dataset)."""
    print("\n" + "=" * 60)
    print("Downloading: amazon_polarity (Sentiment - large)")
    print("=" * 60)

    output_dir = DATA_DIR / "amazon_polarity"
    if (output_dir / "train.jsonl").exists():
        print("  ⏭️  Already downloaded, skipping...")
        return

    # This is a large dataset - limit to 200k samples
    print("  ⚠️  Large dataset - sampling 200k from train")
    ds = load_dataset("amazon_polarity", trust_remote_code=True)

    # Sample train
    train_ds = ds["train"].shuffle(seed=42).select(range(min(200000, len(ds["train"]))))
    data = []
    for item in tqdm(train_ds, desc="Processing train"):
        data.append(
            {
                "text": item["title"] + " " + item["content"],
                "label": item["label"],  # 0=negative, 1=positive
            }
        )
    save_jsonl(data, output_dir / "train.jsonl")

    # Full validation
    data = []
    for item in tqdm(ds["test"], desc="Processing test"):
        data.append({"text": item["title"] + " " + item["content"], "label": item["label"]})
    save_jsonl(data, output_dir / "validation.jsonl")


def print_summary():
    """Print summary of all downloaded datasets."""
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)

    total_samples = 0
    for dataset_dir in sorted(DATA_DIR.iterdir()):
        if dataset_dir.is_dir():
            count = 0
            for jsonl_file in dataset_dir.glob("*.jsonl"):
                with open(jsonl_file, encoding="utf-8") as f:
                    count += sum(1 for _ in f)
            print(f"  {dataset_dir.name:25s}: {count:>10,} samples")
            total_samples += count

    print("-" * 60)
    print(f"  {'TOTAL':25s}: {total_samples:>10,} samples")


def main():
    print("=" * 60)
    print("🚀 DOWNLOADING ALL PUBLIC DATASETS")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR.absolute()}")

    # Core Stage A datasets
    download_ner_conll2003()
    download_ner_wikineural()
    download_sentiment_sst2()
    download_emotions_goemotions()
    download_nli_mnli()
    download_nli_snli()
    download_embedding_stsb()
    download_embedding_allnli()

    # Safety datasets
    download_safety_datasets()

    # Optional large datasets
    # download_amazon_polarity()  # Uncomment if needed

    # Print summary
    print_summary()

    print("\n✅ All datasets downloaded!")
    print("\nNext steps:")
    print("  1. Review data/public/*/train.jsonl for quality")
    print("  2. Run curation scripts to balance classes")
    print("  3. Update configs to point to local data")


if __name__ == "__main__":
    main()

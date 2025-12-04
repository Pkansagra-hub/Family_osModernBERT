#!/usr/bin/env python3
"""Count samples in Stage A datasets."""

from pathlib import Path

from datasets import load_dataset


def main():
    print("=" * 50)
    print("Stage A Dataset Sample Counts")
    print("=" * 50)

    total = 0

    # NER - CoNLL2003
    print("\n--- NER Datasets ---")
    ds = load_dataset("conll2003", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("NER CoNLL2003:")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")
    print(f"  Test: {len(ds['test']):,}")

    # NER - WikiNeural
    ds = load_dataset("tner/wikineural", "en", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("NER WikiNeural (en):")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")
    print(f"  Test: {len(ds['test']):,}")

    # Sentiment - SST2
    print("\n--- Sentiment Datasets ---")
    ds = load_dataset("stanfordnlp/sst2", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("Sentiment SST-2:")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")

    # Emotions - GoEmotions
    print("\n--- Emotion Datasets ---")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("Emotions GoEmotions (simplified):")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")
    print(f"  Test: {len(ds['test']):,}")

    # Safety - Civil Comments
    print("\n--- Safety Datasets ---")
    ds = load_dataset("civil_comments", trust_remote_code=True)
    full_train = len(ds["train"])
    used = min(200000, full_train)
    total += used
    print("Safety Civil Comments:")
    print(f"  Train: {full_train:,} (using max 200K = {used:,})")

    # NLI - MNLI
    print("\n--- NLI Datasets ---")
    ds = load_dataset("multi_nli", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("NLI MNLI:")
    print(f"  Train: {train_count:,}")
    print(f"  Val (matched): {len(ds['validation_matched']):,}")

    # NLI - SNLI
    ds = load_dataset("stanfordnlp/snli", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("NLI SNLI:")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")

    # Embedding - STSB
    print("\n--- Embedding Datasets ---")
    ds = load_dataset("sentence-transformers/stsb", "default", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("Embedding STSB:")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['validation']):,}")
    print(f"  Test: {len(ds['test']):,}")

    # Embedding - NLI pairs
    ds = load_dataset("sentence-transformers/all-nli", "pair-score", trust_remote_code=True)
    train_count = len(ds["train"])
    total += train_count
    print("Embedding AllNLI (pair-score):")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {len(ds['dev']):,}")
    print(f"  Test: {len(ds['test']):,}")

    # FamilyOS Local - Temporal
    print("\n--- FamilyOS Local Datasets ---")
    temporal_dir = Path("data/familyos/temporal/silver")
    if temporal_dir.exists():
        count = 0
        for f in temporal_dir.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fp:
                count += sum(1 for _ in fp)
        total += count
        print(f"Temporal (FamilyOS): {count:,}")
    else:
        print("Temporal: Not found")

    # FamilyOS Local - Emotions (current generation)
    emotions_dir = Path("data/familyos/emotions/silver")
    if emotions_dir.exists():
        count = 0
        for f in emotions_dir.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fp:
                count += sum(1 for _ in fp)
        print(f"Emotions (FamilyOS, generating): {count:,}")
    else:
        print("Emotions: Not found")

    print("\n" + "=" * 50)
    print(f"TOTAL TRAINING SAMPLES: {total:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()

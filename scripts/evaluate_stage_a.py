#!/usr/bin/env python3
"""
Stage A Evaluation Script for Colab

Usage:
    python scripts/evaluate_stage_a.py --checkpoint outputs/stage_a/checkpoint-best
    python scripts/evaluate_stage_a.py --checkpoint outputs/stage_a  # uses final model
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from transformers import AutoTokenizer

from modeling_studio.evaluation.evaluator import Evaluator
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

# CoNLL-2003 label list (9 labels) for proper NER evaluation
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best or latest checkpoint."""
    # Check for checkpoint-best
    best = output_dir / "checkpoint-best"
    if best.exists():
        return best

    # Find all checkpoints
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints:
        # Get latest by step number
        checkpoints = [c for c in checkpoints if c.name != "checkpoint-best"]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            return latest

    # Return the directory itself (final model)
    return output_dir


def load_test_datasets(tasks: list[str], tokenizer) -> dict:
    """Load and tokenize test/validation datasets for each task."""
    from datasets import load_dataset

    datasets = {}

    def tokenize_ner(examples):
        """Tokenize NER data with label alignment."""
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=False,
        )
        # Align labels with tokenized input
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    def tokenize_classification(examples):
        """Tokenize classification data."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    def tokenize_nli(examples):
        """Tokenize NLI data (premise + hypothesis)."""
        return tokenizer(
            examples["text"],
            examples["text_pair"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    for task in tasks:
        try:
            if task == "ner_general":
                ds = load_dataset("conll2003", split="test")
                ds = ds.rename_column("ner_tags", "labels")
                ds = ds.select(range(min(500, len(ds))))
                ds = ds.map(
                    tokenize_ner,
                    batched=True,
                    remove_columns=["tokens", "pos_tags", "chunk_tags", "id"],
                )
                datasets[task] = ds

            elif task == "sentiment":
                ds = load_dataset("glue", "sst2", split="validation")
                ds = ds.rename_column("sentence", "text")

                # Map SST-2 binary (0=neg, 1=pos) to 5-class indices
                # Our 5-class: very_neg=0, neg=1, neutral=2, pos=3, very_pos=4
                # SST-2: 0=negative -> map to 1 (negative)
                # SST-2: 1=positive -> map to 3 (positive)
                def map_sst2_to_5class(ex):
                    ex["labels"] = 1 if ex["label"] == 0 else 3  # neg->1, pos->3
                    ex["original_label"] = ex["label"]  # Keep original for reverse mapping
                    return ex

                ds = ds.map(map_sst2_to_5class, remove_columns=["label"])
                ds = ds.select(range(min(500, len(ds))))
                ds = ds.map(tokenize_classification, batched=True, remove_columns=["text", "idx"])
                datasets[task] = ds

            elif task == "nli":
                ds = load_dataset("snli", split="validation")
                ds = ds.filter(lambda x: x["label"] != -1)
                ds = ds.rename_column("premise", "text")
                ds = ds.rename_column("hypothesis", "text_pair")
                ds = ds.rename_column("label", "labels")
                ds = ds.select(range(min(500, len(ds))))
                ds = ds.map(tokenize_nli, batched=True, remove_columns=["text", "text_pair"])
                datasets[task] = ds

            print(f"  Loaded {task}: {len(datasets.get(task, []))} samples")

        except Exception as e:
            print(f"  Error loading {task}: {e}")
            import traceback

            traceback.print_exc()

    return datasets


def map_5class_to_binary(predictions):
    """
    Map 5-class sentiment predictions to binary for SST-2 evaluation.

    5-class: very_negative=0, negative=1, neutral=2, positive=3, very_positive=4
    Binary: negative=0, positive=1

    Rule-based mapping:
    - 0, 1 (very_neg, neg) -> 0 (negative)
    - 2 (neutral) -> based on which is closer, but default to 0
    - 3, 4 (pos, very_pos) -> 1 (positive)
    """
    import numpy as np

    binary_preds = np.where(predictions >= 3, 1, 0)  # 3,4 -> positive(1), else negative(0)
    return binary_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage A checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/stage_a", help="Path to checkpoint directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["ner_general", "sentiment", "emotions", "safety_generic", "nli"],
        help="Tasks to evaluate",
    )
    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = find_best_checkpoint(Path(args.checkpoint))
    print(f"\n📁 Using checkpoint: {checkpoint_path}")

    # Load model and tokenizer
    print("\n🔄 Loading model...")
    model = ModernBertMultiTaskModel.load_checkpoint(
        str(checkpoint_path), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    print(f"   Model capabilities: {model.capabilities}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load test datasets
    print("\n📊 Loading test datasets...")
    test_datasets = load_test_datasets(args.tasks, tokenizer)

    if not test_datasets:
        print("❌ No test datasets available!")
        return

    # Create evaluator with CoNLL label list for NER
    print("\n🎯 Running evaluation...")
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        capabilities=list(test_datasets.keys()),
        device="auto",
        label_lists={"ner_general": CONLL_LABELS},  # Use CoNLL-2003 labels
    )

    # Evaluate all tasks
    results = evaluator.evaluate_all(
        datasets=test_datasets,
        batch_size=args.batch_size,
    )

    # Special handling for sentiment: compute binary accuracy from 5-class predictions
    if "sentiment" in test_datasets and "sentiment" in results.per_task:
        print("\n📊 Sentiment Binary Mapping (5-class → 2-class for SST-2)...")
        # Re-evaluate sentiment with binary mapping
        import numpy as np
        from torch.utils.data import DataLoader

        from modeling_studio.trainers.collators import MultiTaskCollator

        ds = test_datasets["sentiment"]
        collator = MultiTaskCollator(tokenizer=tokenizer)

        # Add task field
        def add_task(ex):
            ex["task"] = "sentiment"
            return ex

        ds_with_task = ds.map(add_task)

        loader = DataLoader(ds_with_task, batch_size=args.batch_size, collate_fn=collator)

        all_preds_5class = []
        all_labels_binary = []

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                outputs = model(**inputs, capability="sentiment")
                preds_5class = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds_5class.extend(preds_5class)

                # Get original binary labels (stored in labels, mapped: 0->neg(1), 1->pos(3))
                # Reverse: if label was 1, original was neg(0); if 3, original was pos(1)
                labels_5class = batch["labels"].numpy()
                labels_binary = np.where(labels_5class >= 3, 1, 0)  # 3,4 -> 1 (pos), else 0 (neg)
                all_labels_binary.extend(labels_binary)

        # Map predictions: 0,1,2 -> negative(0), 3,4 -> positive(1)
        preds_binary = map_5class_to_binary(np.array(all_preds_5class))
        labels_binary = np.array(all_labels_binary)

        binary_accuracy = (preds_binary == labels_binary).mean()

        # Count distribution
        pred_dist = {i: int((np.array(all_preds_5class) == i).sum()) for i in range(5)}
        print(f"   5-class prediction distribution: {pred_dist}")
        print(f"   Binary accuracy (mapped): {binary_accuracy:.4f}")

        # Update results
        results.per_task["sentiment"]["binary_accuracy"] = float(binary_accuracy)
        results.per_task["sentiment"]["pred_distribution_5class"] = pred_dist

    # Print results
    print("\n" + "=" * 60)
    print(results.summary())
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "eval_results.json"

    results.save(str(output_path))
    print(f"\n💾 Results saved to: {output_path}")

    # Print per-task metrics
    print("\n📈 Per-Task Metrics:")
    for task, metrics in results.per_task.items():
        print(f"\n  {task}:")
        for metric, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

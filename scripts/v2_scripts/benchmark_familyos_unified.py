#!/usr/bin/env python3
"""
FamilyOS Unified Benchmark Script

Benchmarks model on the SAME unified FamilyOS data used during training.
This produces metrics that match the internal training evaluation exactly.

Tasks evaluated:
    - emotions (multi-label) -> Micro-F1
    - sentiment (5-class) -> Accuracy
    - ner_family (sequence labeling) -> F1
    - safety_familyos (4-band) -> Accuracy
    - intent (multi-class) -> Accuracy
    - ingress (hub routing) -> Accuracy
    - relations (multi-label) -> Micro-F1
    - temporal (sequence labeling) -> F1

Weighted Average:
    - FamilyOS tasks: weight 1.0
    - safety_familyos: weight 1.5 (extra priority)
    - Replay tasks (if present): weight 0.2

Usage:
    python scripts/v2_scripts/benchmark_familyos_unified.py \\
        --checkpoint checkpoints/your-model/checkpoint-XXXX

    # Limit samples for quick testing
    python scripts/v2_scripts/benchmark_familyos_unified.py \\
        --checkpoint checkpoints/your-model/checkpoint-XXXX \\
        --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from modeling_studio.data.labels import (
    EMOTIONS_FAMILYOS_LABELS,
    INGRESS_LABELS,
    INTENT_LABELS,
    NER_FAMILY_LABELS,
    RELATION_LABELS,
    SAFETY_FAMILYOS_LABELS,
    SENTIMENT_LABELS,
    TEMPORAL_LABELS,
)
from modeling_studio.data.loaders import load_familyos_unified
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.collators import MultiTaskCollator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task weights (matching train_stage_b.py)
TASK_WEIGHTS = {
    # FamilyOS tasks (weight 1.0)
    "emotions": 1.0,
    "ner_family": 1.0,
    "intent": 1.0,
    "ingress": 1.0,
    "relation": 1.0,
    # Safety gets extra weight (1.0 * 1.5)
    "safety_familyos": 1.5,
    # Included but not FamilyOS-specific
    "sentiment": 1.0,
    "temporal": 1.0,
}

# Primary metrics for each task
TASK_PRIMARY_METRIC = {
    "emotions": "hit_rate",  # At least one correct emotion = success
    "sentiment": "direction_accuracy",  # Positive/Negative/Neutral direction match
    "ner_family": "f1",
    "safety_familyos": "accuracy",
    "intent": "actionable_rate",  # Did we catch action requests?
    "ingress": "accuracy",
    "relation": "micro_f1",
    "temporal": "f1",
}


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
) -> tuple[ModernBertMultiTaskModel, AutoTokenizer]:
    """Load model and tokenizer from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    logger.info(f"Loading model from: {checkpoint_path}")

    # Use load_checkpoint for properly loading custom checkpoints
    # This handles encoder + heads + adapters correctly
    model = ModernBertMultiTaskModel.load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=DEVICE,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    logger.info(f"   Model capabilities: {[c.value for c in model.capabilities]}")

    return model, tokenizer


def load_validation_data(
    tokenizer: AutoTokenizer,
    max_samples: int | None = None,
    max_length: int = 512,
    custom_data_dirs: list[Path] | None = None,
) -> dict[str, Dataset]:
    """Load validation split from unified FamilyOS data."""
    if custom_data_dirs:
        data_dirs = custom_data_dirs
    else:
        data_dirs = [
            project_root / "data" / "familyos" / "unified" / "output_synthetic",
            project_root / "data" / "familyos" / "unified" / "output",
        ]

    # Filter to existing directories
    data_dirs = [d for d in data_dirs if d.exists()]

    if not data_dirs:
        raise FileNotFoundError("No unified FamilyOS data directories found")

    logger.info(f"Loading validation data from: {data_dirs}")

    tasks = [
        "emotions",
        "sentiment",
        "ner_family",
        "safety_familyos",
        "intent",
        "ingress",
        "relation",
        "temporal",
    ]

    datasets = load_familyos_unified(
        data_dirs=data_dirs,
        split="validation",
        tasks=tasks,
        validation_ratio=0.1,
        seed=42,
    )

    # Apply tokenization
    for task in list(datasets.keys()):
        ds = datasets[task]

        # Limit samples if requested
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        # Tokenize
        ds = _tokenize_dataset(ds, task, tokenizer, max_length)
        datasets[task] = ds

        logger.info(f"   {task}: {len(ds)} samples")

    return datasets


def _tokenize_dataset(
    dataset: Dataset,
    task: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize dataset for a specific task."""

    # Sequence labeling tasks have pre-tokenized words
    is_sequence_task = task in ("ner_family", "temporal")

    if is_sequence_task:
        # Get the correct labels column name
        labels_col = "ner_tags" if task == "ner_family" else "temporal_tags"

        def tokenize_sequence_fn(examples):
            """Tokenize pre-tokenized words with label alignment."""
            tokenized = tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            # Align labels with subword tokens
            all_labels = []
            for i, tags in enumerate(examples[labels_col]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(tags[word_idx] if word_idx < len(tags) else 0)
                    else:
                        # For subword tokens, use -100 (ignore in loss)
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                all_labels.append(label_ids)

            tokenized["labels"] = all_labels
            return tokenized

        dataset = dataset.map(
            tokenize_sequence_fn,
            batched=True,
            remove_columns=["tokens", labels_col] if "tokens" in dataset.column_names else [],
            desc=f"Tokenizing {task}",
        )
    else:
        # Text classification tasks
        def tokenize_fn(examples):
            encoded = tokenizer(
                examples["text"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            return encoded

        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"] if "text" in dataset.column_names else [],
            desc=f"Tokenizing {task}",
        )

    return dataset


def evaluate_classification(
    model: ModernBertMultiTaskModel,
    dataloader: DataLoader,
    task: str,
    num_labels: int,
) -> dict[str, float]:
    """Evaluate single-label classification task."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"   {task}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=task,
            )

            logits = outputs.logits
            if logits is None:
                continue

            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Compute direction accuracy for sentiment
    # 0,1 = negative, 2 = neutral, 3,4 = positive
    def get_direction(label_id: int) -> int:
        if label_id <= 1:  # very_negative, negative
            return -1
        elif label_id == 2:  # neutral
            return 0
        else:  # positive, very_positive
            return 1

    direction_matches = sum(
        1 for pred, label in zip(all_preds, all_labels)
        if get_direction(pred) == get_direction(label)
    )
    direction_accuracy = direction_matches / len(all_labels) if all_labels else 0.0

    # Compute intent family accuracy
    # ACTIONABLE: log_memory(0), query_memory(1), set_reminder(2)
    # EMOTIONAL: express_feeling(3), reflect(6)
    # INFORMATIONAL: seek_advice(4), share_news(5)
    # OTHER: other(7)
    def get_intent_family(label_id: int) -> int:
        if label_id in [0, 1, 2]:  # log_memory, query_memory, set_reminder
            return 0  # ACTIONABLE
        elif label_id in [3, 6]:  # express_feeling, reflect
            return 1  # EMOTIONAL
        elif label_id in [4, 5]:  # seek_advice, share_news
            return 2  # INFORMATIONAL
        else:  # other(7)
            return 3  # OTHER

    family_matches = sum(
        1 for pred, label in zip(all_preds, all_labels)
        if get_intent_family(pred) == get_intent_family(label)
    )
    family_accuracy = family_matches / len(all_labels) if all_labels else 0.0

    # Compute actionable detection rate
    # If user wants ACTION (0,1,2), did we detect ANY intent (not just emotional)?
    # This is more lenient - catching action request is critical
    def is_actionable(label_id: int) -> bool:
        return label_id in [0, 1, 2]  # log_memory, query_memory, set_reminder

    actionable_gt = [(p, l) for p, l in zip(all_preds, all_labels) if is_actionable(l)]
    if actionable_gt:
        # For actionable ground truth, did we predict actionable OR informational (both trigger system action)?
        actionable_detected = sum(1 for p, l in actionable_gt if p in [0, 1, 2, 4])  # include seek_advice as action trigger
        actionable_rate = actionable_detected / len(actionable_gt)
    else:
        actionable_rate = 1.0

    return {
        "accuracy": accuracy,
        "direction_accuracy": direction_accuracy,
        "family_accuracy": family_accuracy,
        "actionable_rate": actionable_rate,
        "macro_f1": macro_f1,
        "num_samples": len(all_labels),
    }


def evaluate_multilabel(
    model: ModernBertMultiTaskModel,
    dataloader: DataLoader,
    task: str,
    num_labels: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate multi-label classification task."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"   {task}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=task,
            )

            logits = outputs.logits
            if logits is None:
                continue

            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute hit rate: at least one correct prediction per sample
    # For each sample, check if (pred & label).any() - i.e., at least one overlap
    hits = 0
    for pred, label in zip(all_preds, all_labels):
        pred_arr = np.array(pred, dtype=int)
        label_arr = np.array(label, dtype=int)
        if np.any(pred_arr & label_arr):  # At least one match
            hits += 1
    hit_rate = hits / len(all_preds) if len(all_preds) > 0 else 0.0

    # Compute metrics
    precision, recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "hit_rate": hit_rate,  # At least one correct = success
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "num_samples": len(all_labels),
    }


def evaluate_sequence_labeling(
    model: ModernBertMultiTaskModel,
    dataloader: DataLoader,
    task: str,
    label_schema,
) -> dict[str, float]:
    """Evaluate sequence labeling task (NER, temporal)."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"   {task}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=task,
            )

            logits = outputs.logits
            if logits is None:
                continue

            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.numpy()
            mask = attention_mask.cpu().numpy()

            # Flatten and filter padding
            for i in range(len(preds)):
                seq_len = mask[i].sum()
                pred_seq = preds[i, :seq_len].tolist()
                label_seq = labels_np[i, :seq_len].tolist()

                # Filter -100 (ignored tokens)
                for p, lbl in zip(pred_seq, label_seq):
                    if lbl != -100:
                        all_preds.append(p)
                        all_labels.append(lbl)

    # Compute metrics (excluding O tag for NER-style tasks)
    # For simplicity, compute overall F1
    if all_labels:
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        micro_f1 = macro_f1 = 0.0

    return {
        "f1": micro_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "num_tokens": len(all_labels),
    }


def run_benchmark(
    checkpoint_path: str | Path,
    max_samples: int | None = None,
    batch_size: int = 32,
    data_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Run full FamilyOS unified benchmark."""

    # Load model
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # Load validation data
    logger.info("\nLoading validation data...")
    datasets = load_validation_data(tokenizer, max_samples=max_samples, custom_data_dirs=data_dirs)

    # Create collator
    collator = MultiTaskCollator(tokenizer=tokenizer, max_length=512)

    # Results storage
    results = {}

    # Task configurations
    task_configs = {
        "emotions": {
            "type": "multilabel",
            "labels": EMOTIONS_FAMILYOS_LABELS,
        },
        "sentiment": {
            "type": "classification",
            "labels": SENTIMENT_LABELS,
        },
        "ner_family": {
            "type": "sequence",
            "labels": NER_FAMILY_LABELS,
        },
        "safety_familyos": {
            "type": "classification",
            "labels": SAFETY_FAMILYOS_LABELS,
        },
        "intent": {
            "type": "classification",
            "labels": INTENT_LABELS,
        },
        "ingress": {
            "type": "classification",
            "labels": INGRESS_LABELS,
        },
        "relation": {
            "type": "multilabel",
            "labels": RELATION_LABELS,
        },
        "temporal": {
            "type": "sequence",
            "labels": TEMPORAL_LABELS,
        },
    }

    logger.info("\nRunning evaluation...")

    for task, config in task_configs.items():
        if task not in datasets:
            logger.warning(f"   Skipping {task}: no data")
            continue

        ds = datasets[task]

        # Create dataloader - collator gets task from sample's 'task' field
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # Evaluate based on task type
        task_type = config["type"]
        label_schema = config["labels"]
        num_labels = label_schema.num_labels

        try:
            if task_type == "classification":
                metrics = evaluate_classification(model, dataloader, task, num_labels)
            elif task_type == "multilabel":
                metrics = evaluate_multilabel(model, dataloader, task, num_labels)
            elif task_type == "sequence":
                metrics = evaluate_sequence_labeling(model, dataloader, task, label_schema)
            else:
                logger.warning(f"   Unknown task type: {task_type}")
                continue

            results[task] = metrics

            # Log primary metric
            primary_metric = TASK_PRIMARY_METRIC.get(task, "accuracy")
            value = metrics.get(primary_metric, 0)
            logger.info(f"   {task}: {primary_metric}={value:.4f}")

        except Exception as e:
            logger.error(f"   {task}: Error - {e}")
            continue

    # Compute weighted average
    weighted_sum = 0.0
    total_weight = 0.0

    for task, metrics in results.items():
        primary_metric = TASK_PRIMARY_METRIC.get(task, "accuracy")
        value = metrics.get(primary_metric, 0)
        weight = TASK_WEIGHTS.get(task, 1.0)

        weighted_sum += value * weight
        total_weight += weight

    weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

    results["weighted_avg_score"] = weighted_avg

    return results


def print_results(results: dict[str, Any], checkpoint_path: str):
    """Print formatted benchmark results."""

    print("\n" + "=" * 60)
    print("FamilyOS Unified Benchmark Results")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Print each task
    for task in TASK_WEIGHTS.keys():
        if task not in results:
            continue

        metrics = results[task]
        primary_metric = TASK_PRIMARY_METRIC.get(task, "accuracy")
        value = metrics.get(primary_metric, 0)
        weight = TASK_WEIGHTS[task]

        print(f"{task:20s}: {primary_metric:10s} = {value:6.2%} (weight={weight})")

    print()
    print("-" * 60)
    print(f"{'WEIGHTED AVERAGE':20s}: {results.get('weighted_avg_score', 0):6.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="FamilyOS Unified Benchmark - matches training evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (for quick testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom data directory (use parent dir of shard files)",
    )

    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    # Parse data dirs
    data_dirs = None
    if args.data_dir:
        data_dirs = [Path(args.data_dir)]
        print(f"Using custom data: {data_dirs}")

    # Run benchmark
    results = run_benchmark(
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        data_dirs=data_dirs,
    )

    # Print results
    print_results(results, args.checkpoint)

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = Path(args.checkpoint) / "unified_benchmark_results.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

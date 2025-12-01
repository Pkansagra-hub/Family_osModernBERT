#!/usr/bin/env python3
"""
Forgetting Evaluation Script

Evaluates catastrophic forgetting after Stage B training by comparing
Stage A and Stage B model performance on Stage A benchmarks.

Forgetting Gates (per v2 plan):
    - CoNLL-2003 (NER): ≤ 2% F1 drop
    - SST-2 (Sentiment): ≤ 2% Accuracy drop
    - MNLI (NLI): ≤ 2% Accuracy drop
    - GoEmotions: ≤ 3% Macro F1 drop
    - Safety Generic: ≤ 3% Macro F1 drop
    - Embedding (STS-B): ≤ 3% Spearman drop

Usage:
    # Compare Stage A and Stage B checkpoints
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1

    # Evaluate specific tasks only
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --tasks ner_general sentiment nli

    # With custom thresholds
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --max-drop 0.03

    # Save report to file
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --output outputs/forgetting_report.json

Outputs:
    - Console summary with pass/fail status
    - JSON report (if --output specified)
    - Recommendations for failed gates
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.labels import EMOTIONS_LABELS

# GoEmotions label mapping (28 emotions in HuggingFace order)
GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default forgetting thresholds (per v2 plan)
DEFAULT_THRESHOLDS = {
    "ner_general": {"metric": "f1", "max_drop": 0.02},
    "sentiment": {"metric": "accuracy", "max_drop": 0.02},
    "nli": {"metric": "accuracy", "max_drop": 0.02},
    "emotions": {"metric": "macro_f1", "max_drop": 0.03},
    "safety_generic": {"metric": "macro_f1", "max_drop": 0.03},
    "embedding": {"metric": "spearman", "max_drop": 0.03},
}

# Stage A tasks to evaluate for forgetting
STAGE_A_TASKS = ["ner_general", "sentiment", "emotions", "safety_generic", "nli"]


# =============================================================================
# Model Loading
# =============================================================================


def load_model(checkpoint_path: str | Path, device: str = "cuda") -> ModernBertMultiTaskModel:
    """
    Load a multi-task model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded ModernBertMultiTaskModel
    """
    checkpoint_path = Path(checkpoint_path)

    # Check for "best" subdirectory
    if (checkpoint_path / "best").exists():
        checkpoint_path = checkpoint_path / "best"

    logger.info(f"Loading model from {checkpoint_path}")

    model = ModernBertMultiTaskModel.load_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    model.eval()

    capabilities = [c.value for c in model.capabilities]
    logger.info(f"Loaded model with capabilities: {capabilities}")

    return model


def load_tokenizer(checkpoint_path: str | Path) -> AutoTokenizer:
    """Load tokenizer from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path / "best").exists():
        checkpoint_path = checkpoint_path / "best"
    return AutoTokenizer.from_pretrained(str(checkpoint_path))


# =============================================================================
# Dataset Loading
# =============================================================================


def load_evaluation_datasets(
    tasks: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """
    Load evaluation datasets for forgetting evaluation.

    Uses the same datasets as Stage A evaluation.
    """
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
                # CoNLL-2003 NER
                logger.info("Loading CoNLL-2003 test set...")
                ds = load_dataset("conll2003", split="test", trust_remote_code=True)
                # Map labels to our schema
                ds = ds.rename_column("ner_tags", "labels")
                ds = ds.map(tokenize_ner, batched=True, remove_columns=ds.column_names)
                datasets[task] = ds

            elif task == "sentiment":
                # SST-2
                logger.info("Loading SST-2 validation set...")
                ds = load_dataset("glue", "sst2", split="validation", trust_remote_code=True)
                ds = ds.rename_column("sentence", "text")
                ds = ds.map(tokenize_classification, batched=True, remove_columns=["text", "idx"])
                datasets[task] = ds

            elif task == "emotions":
                # GoEmotions
                logger.info("Loading GoEmotions test set...")
                ds = load_dataset("go_emotions", "simplified", split="test", trust_remote_code=True)

                # GoEmotions has multi-label format - take primary emotion
                def process_emotions(example):
                    labels = example["labels"]
                    if len(labels) > 0:
                        go_label = labels[0]  # Primary emotion
                        # Map GoEmotions index to our schema
                        if go_label < len(GO_EMOTIONS_LABELS):
                            go_emotion_name = GO_EMOTIONS_LABELS[go_label]
                            # Find in our schema (use label2id, not label_to_id)
                            if go_emotion_name in EMOTIONS_LABELS.label2id:
                                return {"label": EMOTIONS_LABELS.label2id[go_emotion_name]}
                    return {"label": 0}  # neutral

                ds = ds.map(process_emotions)
                ds = ds.map(
                    tokenize_classification, batched=True, remove_columns=["text", "id", "labels"]
                )
                datasets[task] = ds

            elif task == "safety_generic":
                # Jigsaw toxicity
                logger.info("Loading Jigsaw toxicity dataset...")
                try:
                    # Try local test data first
                    local_path = Path("tests/data/test_safety.csv")
                    if local_path.exists():
                        ds = load_dataset("csv", data_files=str(local_path), split="train")
                    else:
                        # Use civil comments subset
                        ds = load_dataset(
                            "google/civil_comments",
                            split="test[:1000]",
                            trust_remote_code=True,
                        )
                        # Convert toxicity score to binary
                        ds = ds.map(lambda x: {"label": 1 if x["toxicity"] >= 0.5 else 0})
                    ds = ds.map(tokenize_classification, batched=True)
                    datasets[task] = ds
                except Exception as e:
                    logger.warning(f"Could not load safety dataset: {e}")

            elif task == "nli":
                # MNLI
                logger.info("Loading MNLI validation set...")
                ds = load_dataset(
                    "glue", "mnli", split="validation_matched", trust_remote_code=True
                )
                ds = ds.rename_column("premise", "text")
                ds = ds.rename_column("hypothesis", "text_pair")
                ds = ds.map(tokenize_nli, batched=True, remove_columns=["text", "text_pair", "idx"])
                datasets[task] = ds

            elif task == "embedding":
                # STS-B
                logger.info("Loading STS-B validation set...")
                ds = load_dataset("glue", "stsb", split="validation", trust_remote_code=True)
                ds = ds.rename_column("sentence1", "text")
                ds = ds.rename_column("sentence2", "text_pair")
                datasets[task] = ds

        except Exception as e:
            logger.warning(f"Failed to load dataset for {task}: {e}")

    return datasets


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_model_on_task(
    model: ModernBertMultiTaskModel,
    task: str,
    dataset: Any,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Evaluate a model on a single task.

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # Create dataloader
    def collate_fn(batch):
        # Find max length in batch
        max_len = max(len(x["input_ids"]) for x in batch)

        # Pad input_ids and attention_mask
        padded_input_ids = []
        padded_attention_mask = []
        for x in batch:
            ids = x["input_ids"]
            mask = x["attention_mask"]
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [0] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)

        input_ids = torch.tensor(padded_input_ids)
        attention_mask = torch.tensor(padded_attention_mask)

        if task == "ner_general":
            # Pad labels for NER
            labels = []
            for x in batch:
                label = x["labels"]
                if len(label) < max_len:
                    label = label + [-100] * (max_len - len(label))
                else:
                    label = label[:max_len]
                labels.append(label)
            labels = torch.tensor(labels)
        else:
            labels = torch.tensor([x["label"] for x in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            # Forward pass with capability
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capability=task,
            )

            logits = outputs.logits

            if task == "ner_general":
                # Token-level predictions
                preds = logits.argmax(dim=-1).cpu()
                for i in range(preds.shape[0]):
                    mask = labels[i] != -100
                    all_predictions.extend(preds[i][mask].tolist())
                    all_labels.extend(labels[i][mask].tolist())
            else:
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_predictions.extend(preds)
                all_labels.extend(labels.tolist())

    # Compute metrics
    metrics = {}
    if len(all_labels) > 0:
        metrics["accuracy"] = accuracy_score(all_labels, all_predictions)
        metrics["macro_f1"] = f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        metrics["f1"] = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    return metrics


def evaluate_forgetting(
    stage_a_model: ModernBertMultiTaskModel,
    stage_b_model: ModernBertMultiTaskModel,
    datasets: dict[str, Any],
    tokenizer: AutoTokenizer,
    thresholds: dict[str, dict],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Evaluate forgetting by comparing Stage A and Stage B models.

    Returns:
        Dictionary with results for each task
    """
    results = {}

    for task, dataset in datasets.items():
        if task not in thresholds:
            logger.warning(f"No threshold defined for task: {task}")
            continue

        threshold_config = thresholds[task]
        metric_name = threshold_config["metric"]
        max_drop = threshold_config["max_drop"]

        logger.info(f"\nEvaluating {task}...")

        # Evaluate Stage A model
        logger.info("  Stage A model...")
        stage_a_metrics = evaluate_model_on_task(
            stage_a_model, task, dataset, tokenizer, batch_size, device
        )

        # Evaluate Stage B model
        logger.info("  Stage B model...")
        stage_b_metrics = evaluate_model_on_task(
            stage_b_model, task, dataset, tokenizer, batch_size, device
        )

        # Get relevant metric
        stage_a_score = stage_a_metrics.get(metric_name, 0.0)
        stage_b_score = stage_b_metrics.get(metric_name, 0.0)

        # Calculate drop (positive = regression)
        drop = stage_a_score - stage_b_score
        passed = drop <= max_drop

        results[task] = {
            "metric": metric_name,
            "stage_a_score": stage_a_score,
            "stage_b_score": stage_b_score,
            "drop": drop,
            "max_allowed_drop": max_drop,
            "passed": passed,
            "stage_a_metrics": stage_a_metrics,
            "stage_b_metrics": stage_b_metrics,
        }

        # Log result
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            f"  {task}: {stage_a_score:.4f} → {stage_b_score:.4f} "
            f"(drop: {drop:.4f}, max: {max_drop:.4f}) {status}"
        )

    return results


def generate_recommendations(results: dict[str, Any]) -> list[str]:
    """Generate recommendations for failed tasks."""
    failed_tasks = [task for task, res in results.items() if not res["passed"]]

    if not failed_tasks:
        return []

    recommendations = [
        "Reduce LoRA r value (try r=16 instead of r=32)",
        "Increase replay ratio (try 0.2 instead of 0.1)",
        "Freeze more encoder layers during Stage B",
        "Reduce Stage B learning rate",
    ]

    if "ner_general" in failed_tasks:
        recommendations.append("NER: Consider task-specific LoRA adapters")
        recommendations.append("NER: Increase NER replay samples")

    if "nli" in failed_tasks:
        recommendations.append("NLI: Try lower learning rate for NLI head")

    if "sentiment" in failed_tasks:
        recommendations.append("Sentiment: Check for domain shift in Stage B data")

    if "emotions" in failed_tasks:
        recommendations.append("Emotions: Multi-label may need separate adapters")

    return recommendations


def print_report(
    results: dict[str, Any],
    stage_a_path: str,
    stage_b_path: str,
) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("CATASTROPHIC FORGETTING EVALUATION REPORT")
    print("=" * 70)
    print(f"Stage A: {stage_a_path}")
    print(f"Stage B: {stage_b_path}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    all_passed = True
    failed_tasks = []

    for task, res in results.items():
        status = "✅" if res["passed"] else "❌"
        print(
            f"{status} {task:20s} | "
            f"{res['metric']:10s} | "
            f"{res['stage_a_score']:.4f} → {res['stage_b_score']:.4f} | "
            f"drop: {res['drop']:+.4f} (max: {res['max_allowed_drop']:.4f})"
        )
        if not res["passed"]:
            all_passed = False
            failed_tasks.append(task)

    print("-" * 70)

    if all_passed:
        print("✅ ALL FORGETTING GATES PASSED")
        print("Stage B model is safe to deploy!")
    else:
        print(f"❌ FAILED GATES: {', '.join(failed_tasks)}")
        print("\nRecommendations:")
        for rec in generate_recommendations(results):
            print(f"  • {rec}")

    print("=" * 70)


def save_report(
    results: dict[str, Any],
    stage_a_path: str,
    stage_b_path: str,
    output_path: str | Path,
) -> None:
    """Save evaluation report to JSON."""
    failed_tasks = [task for task, res in results.items() if not res["passed"]]

    report = {
        "timestamp": datetime.now().isoformat(),
        "stage_a_checkpoint": str(stage_a_path),
        "stage_b_checkpoint": str(stage_b_path),
        "all_passed": len(failed_tasks) == 0,
        "failed_tasks": failed_tasks,
        "results": results,
        "recommendations": generate_recommendations(results),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved report to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate catastrophic forgetting after Stage B training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic comparison
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1

    # Specific tasks only
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1 \\
        --tasks ner_general sentiment nli

    # Save report
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1 \\
        --output outputs/forgetting_report.json
""",
    )

    parser.add_argument(
        "--stage-a",
        type=str,
        required=True,
        help="Path to Stage A checkpoint",
    )

    parser.add_argument(
        "--stage-b",
        type=str,
        required=True,
        help="Path to Stage B checkpoint",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=STAGE_A_TASKS,
        help=f"Tasks to evaluate. Default: {STAGE_A_TASKS}",
    )

    parser.add_argument(
        "--max-drop",
        type=float,
        default=None,
        help="Override max allowed drop for all tasks (e.g., 0.03 for 3%%)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON report",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )

    args = parser.parse_args()

    # Setup thresholds
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.max_drop is not None:
        for task in thresholds:
            thresholds[task]["max_drop"] = args.max_drop

    # Load models
    logger.info("Loading Stage A model...")
    stage_a_model = load_model(args.stage_a, args.device)

    logger.info("Loading Stage B model...")
    stage_b_model = load_model(args.stage_b, args.device)

    # Load tokenizer (from Stage A - should be identical)
    tokenizer = load_tokenizer(args.stage_a)

    # Load datasets
    logger.info("Loading evaluation datasets...")
    datasets = load_evaluation_datasets(args.tasks, tokenizer)

    # Filter to available datasets
    available_tasks = [t for t in args.tasks if t in datasets]
    if len(available_tasks) < len(args.tasks):
        missing = set(args.tasks) - set(available_tasks)
        logger.warning(f"Datasets not available for: {missing}")

    # Evaluate
    results = evaluate_forgetting(
        stage_a_model=stage_a_model,
        stage_b_model=stage_b_model,
        datasets={k: v for k, v in datasets.items() if k in available_tasks},
        tokenizer=tokenizer,
        thresholds=thresholds,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Print report
    print_report(results, args.stage_a, args.stage_b)

    # Save if requested
    if args.output:
        save_report(results, args.stage_a, args.stage_b, args.output)

    # Exit with error code if any gates failed
    failed = [task for task, res in results.items() if not res["passed"]]
    if failed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
    main()

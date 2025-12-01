#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

This script runs comprehensive evaluation of trained models on all tasks
and generates evaluation reports. Supports FamilyOS quality gates.

Quality Gates (per v2 plan):
    Stage A Tasks:
        - ner_general: F1 >= 86%
        - sentiment: Accuracy >= 90%
        - emotions: Macro F1 >= 48%
        - safety_generic: Macro F1 >= 85%
        - nli: Accuracy >= 83%
        - embedding: Spearman >= 0.82
        - temporal: F1 >= 75%

    Stage B Tasks:
        - ner_family: F1 >= 85%
        - ingress: Accuracy >= 90%
        - safety_familyos: Macro F1 >= 90%, CRISIS recall >= 95%
        - relation: F1 >= 80%
        - intent: Accuracy >= 88%

Usage:
    # Evaluate Stage A model
    python scripts/evaluate.py \
        --model outputs/modernbert-multitask-v0 \
        --tasks ner_general sentiment emotions safety_generic nli embedding

    # Evaluate FamilyOS model (all tasks)
    python scripts/evaluate.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --tasks all

    # Evaluate with baseline comparison
    python scripts/evaluate.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --baseline outputs/modernbert-multitask-v0 \
        --tasks all

    # Quick validation (subset)
    python scripts/evaluate.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --tasks ner_family safety_familyos intent

    # Generate full report
    python scripts/evaluate.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --tasks all \
        --output outputs/eval_report \
        --format json markdown

Outputs:
    - {output}/eval_results.json: Full metrics
    - {output}/eval_report.md: Summary report
    - {output}/quality_gates.json: Pass/fail status
    - {output}/confusion_matrices/: Per-task confusion matrices (if --save-cm)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quality gates (metric thresholds)
QUALITY_GATES = {
    # Stage A tasks
    "ner_general": {"metric": "f1", "threshold": 0.86, "direction": "higher"},
    "sentiment": {"metric": "accuracy", "threshold": 0.90, "direction": "higher"},
    "emotions": {"metric": "macro_f1", "threshold": 0.48, "direction": "higher"},
    "safety_generic": {"metric": "macro_f1", "threshold": 0.85, "direction": "higher"},
    "nli": {"metric": "accuracy", "threshold": 0.83, "direction": "higher"},
    "embedding": {"metric": "spearman", "threshold": 0.82, "direction": "higher"},
    "temporal": {"metric": "f1", "threshold": 0.75, "direction": "higher"},
    # Stage B tasks
    "ner_family": {"metric": "f1", "threshold": 0.85, "direction": "higher"},
    "ingress": {"metric": "accuracy", "threshold": 0.90, "direction": "higher"},
    "safety_familyos": {"metric": "macro_f1", "threshold": 0.90, "direction": "higher"},
    "relation": {"metric": "f1", "threshold": 0.80, "direction": "higher"},
    "intent": {"metric": "accuracy", "threshold": 0.88, "direction": "higher"},
}

# Additional gate for safety
CRISIS_RECALL_GATE = {"threshold": 0.95, "label": "CRISIS"}

# All tasks by stage
STAGE_A_TASKS = [
    "ner_general",
    "sentiment",
    "emotions",
    "safety_generic",
    "nli",
    "embedding",
    "temporal",
]
STAGE_B_TASKS = ["ner_family", "ingress", "safety_familyos", "relation", "intent"]
ALL_TASKS = STAGE_A_TASKS + STAGE_B_TASKS

# Task types
SEQUENCE_LABELING_TASKS = ["ner_general", "ner_family", "temporal"]
CLASSIFICATION_TASKS = [
    "sentiment",
    "emotions",
    "safety_generic",
    "safety_familyos",
    "ingress",
    "intent",
]
PAIR_TASKS = ["nli", "relation"]
EMBEDDING_TASKS = ["embedding"]

# Label mappings
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

NER_FAMILY_LABELS = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-RELATION",
    "I-RELATION",
    "B-EVENT",
    "I-EVENT",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-ARTIFACT",
    "I-ARTIFACT",
]

TEMPORAL_LABELS = [
    "O",
    "B-DATE",
    "I-DATE",
    "B-TIME",
    "I-TIME",
    "B-DURATION",
    "I-DURATION",
    "B-RECURRENCE",
    "I-RECURRENCE",
]

SAFETY_GENERIC_LABELS = ["SAFE", "UNSAFE", "NEEDS_REVIEW"]
SAFETY_FAMILYOS_LABELS = ["SAFE", "CAUTION", "ALERT", "CRISIS"]
SENTIMENT_LABELS = ["negative", "positive"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]
INGRESS_LABELS = ["ACCEPT", "REJECT"]

# =============================================================================
# Data Loading
# =============================================================================


def load_evaluation_data(
    task: str, data_dir: Path, tokenizer, max_samples: int = None
) -> list | None:
    """Load evaluation dataset for a task."""
    data_dir = Path(data_dir)

    # Map task to data path
    data_paths = {
        # Stage A - public benchmarks
        "ner_general": data_dir / "public" / "conll2003_test.jsonl",
        "sentiment": data_dir / "public" / "sst2_validation.jsonl",
        "emotions": data_dir / "public" / "goemotions_test.jsonl",
        "safety_generic": data_dir / "public" / "safety_test.jsonl",
        "nli": data_dir / "public" / "mnli_validation.jsonl",
        "embedding": data_dir / "public" / "stsb_validation.jsonl",
        "temporal": data_dir / "public" / "temporal_test.jsonl",
        # Stage B - FamilyOS
        "ner_family": data_dir / "familyos" / "ner_family" / "test.jsonl",
        "ingress": data_dir / "familyos" / "ingress" / "test.jsonl",
        "safety_familyos": data_dir / "familyos" / "safety" / "test.jsonl",
        "relation": data_dir / "familyos" / "relations" / "test.jsonl",
        "intent": data_dir / "familyos" / "intents" / "test.jsonl",
    }

    data_path = data_paths.get(task)
    if data_path is None:
        logger.warning(f"No data path configured for task: {task}")
        return None

    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return None

    logger.info(f"Loading {task} data from {data_path}")

    # Load data
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break

    logger.info(f"Loaded {len(samples)} samples for {task}")
    return samples


def prepare_batch(task: str, samples: list, tokenizer) -> dict:
    """Prepare a batch of samples for inference."""
    if task in SEQUENCE_LABELING_TASKS:
        return prepare_sequence_labeling_batch(task, samples, tokenizer)
    elif task in PAIR_TASKS:
        return prepare_pair_batch(task, samples, tokenizer)
    elif task in EMBEDDING_TASKS:
        return prepare_embedding_batch(samples, tokenizer)
    else:
        return prepare_classification_batch(task, samples, tokenizer)


def prepare_classification_batch(task: str, samples: list, tokenizer) -> dict:
    """Prepare classification samples."""
    texts = []
    labels = []

    for sample in samples:
        text = sample.get("text") or sample.get("sentence") or sample.get("content", "")
        label = sample.get("label") or sample.get("labels", 0)
        texts.append(text)
        labels.append(label)

    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_sequence_labeling_batch(task: str, samples: list, tokenizer) -> dict:
    """Prepare sequence labeling samples with label alignment."""
    all_tokens = []
    all_labels = []

    for sample in samples:
        tokens = sample.get("tokens", [])
        labels = sample.get("labels", sample.get("ner_tags", []))
        all_tokens.append(tokens)
        all_labels.append(labels)

    # Tokenize
    encoding = tokenizer(
        all_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Align labels
    aligned_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = encoding.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                if word_idx < len(labels):
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)  # Sub-token
            prev_word_idx = word_idx
        aligned_labels.append(label_ids)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": aligned_labels,
        "word_ids": [encoding.word_ids(i) for i in range(len(samples))],
    }


def prepare_pair_batch(task: str, samples: list, tokenizer) -> dict:
    """Prepare pair classification samples (NLI, relation)."""
    texts_a = []
    texts_b = []
    labels = []

    for sample in samples:
        if task == "nli":
            text_a = sample.get("premise") or sample.get("text", "")
            text_b = sample.get("hypothesis") or sample.get("text_pair", "")
        else:  # relation
            text_a = sample.get("text_a") or sample.get("entity1", "")
            text_b = sample.get("text_b") or sample.get("entity2", "")

        texts_a.append(text_a)
        texts_b.append(text_b)
        labels.append(sample.get("label", 0))

    encoding = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_embedding_batch(samples: list, tokenizer) -> dict:
    """Prepare embedding samples for similarity evaluation."""
    texts_a = []
    texts_b = []
    scores = []

    for sample in samples:
        text_a = sample.get("sentence1") or sample.get("text_a", "")
        text_b = sample.get("sentence2") or sample.get("text_b", "")
        score = sample.get("score") or sample.get("similarity", 0.0)
        texts_a.append(text_a)
        texts_b.append(text_b)
        scores.append(float(score))

    encoding_a = tokenizer(
        texts_a,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoding_b = tokenizer(
        texts_b,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    return {
        "input_ids_a": encoding_a["input_ids"],
        "attention_mask_a": encoding_a["attention_mask"],
        "input_ids_b": encoding_b["input_ids"],
        "attention_mask_b": encoding_b["attention_mask"],
        "scores": scores,
    }


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_classification(
    model: ModernBertMultiTaskModel,
    task: str,
    samples: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    label_names: list = None,
) -> dict:
    """Evaluate classification task."""
    model.eval()
    all_preds = []
    all_labels = []

    # Process in batches
    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating {task}"):
        batch_samples = samples[i : i + batch_size]
        batch = prepare_classification_batch(task, batch_samples, tokenizer)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"])

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "n_samples": len(all_labels),
    }

    # Per-class metrics
    if label_names:
        report = classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0
        )
        results["per_class"] = {name: report.get(name, {}) for name in label_names}

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        results["confusion_matrix"] = cm.tolist()

    return results


def evaluate_sequence_labeling(
    model: ModernBertMultiTaskModel,
    task: str,
    samples: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    label_names: list = None,
) -> dict:
    """Evaluate sequence labeling task (NER, temporal)."""
    model.eval()
    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating {task}"):
        batch_samples = samples[i : i + batch_size]
        batch = prepare_sequence_labeling_batch(task, batch_samples, tokenizer)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        # Extract predictions for original tokens only
        for j, (pred_seq, label_seq) in enumerate(zip(preds, batch["labels"])):
            word_ids = batch["word_ids"][j]
            prev_word_idx = None
            for k, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx != prev_word_idx:
                    if label_seq[k] != -100:
                        all_preds.append(int(pred_seq[k]))
                        all_labels.append(int(label_seq[k]))
                prev_word_idx = word_idx

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )

    # Entity-level F1 (excluding O)
    entity_labels = [l for l in all_labels if l != 0]
    entity_preds = [p for p, l in zip(all_preds, all_labels) if l != 0]
    if entity_labels:
        _, _, entity_f1, _ = precision_recall_fscore_support(
            entity_labels, entity_preds, average="micro", zero_division=0
        )
    else:
        entity_f1 = 0.0

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "entity_f1": float(entity_f1),
        "n_tokens": len(all_labels),
    }

    if label_names:
        report = classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0
        )
        results["per_class"] = {
            name: report.get(name, {}) for name in label_names if name in report
        }

    return results


def evaluate_pair_classification(
    model: ModernBertMultiTaskModel,
    task: str,
    samples: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    label_names: list = None,
) -> dict:
    """Evaluate pair classification task (NLI, relation)."""
    model.eval()
    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating {task}"):
        batch_samples = samples[i : i + batch_size]
        batch = prepare_pair_batch(task, batch_samples, tokenizer)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"])

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "f1": float(f1),  # Alias for quality gate
        "n_samples": len(all_labels),
    }

    if label_names:
        report = classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0
        )
        results["per_class"] = {name: report.get(name, {}) for name in label_names}

    return results


def evaluate_embedding(
    model: ModernBertMultiTaskModel,
    samples: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Evaluate embedding similarity task."""
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    all_sims = []
    all_scores = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating embedding"):
        batch_samples = samples[i : i + batch_size]
        batch = prepare_embedding_batch(batch_samples, tokenizer)

        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)

        with torch.no_grad():
            # Get embeddings for both texts
            outputs_a = model(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
                task="embedding",
            )
            outputs_b = model(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                task="embedding",
            )

        # Compute cosine similarity
        emb_a = outputs_a["embeddings"]
        emb_b = outputs_b["embeddings"]

        # Normalize
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)

        sims = (emb_a * emb_b).sum(dim=-1).cpu().numpy()
        all_sims.extend(sims.tolist())
        all_scores.extend(batch["scores"])

    # Compute correlation
    pearson, _ = pearsonr(all_sims, all_scores)
    spearman, _ = spearmanr(all_sims, all_scores)

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "n_pairs": len(all_scores),
    }


def evaluate_task(
    model: ModernBertMultiTaskModel,
    task: str,
    samples: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Route to appropriate evaluation function based on task type."""
    # Determine label names
    label_names = None
    if task == "ner_general":
        label_names = CONLL_LABELS
    elif task == "ner_family":
        label_names = NER_FAMILY_LABELS
    elif task == "temporal":
        label_names = TEMPORAL_LABELS
    elif task == "sentiment":
        label_names = SENTIMENT_LABELS
    elif task == "safety_generic":
        label_names = SAFETY_GENERIC_LABELS
    elif task == "safety_familyos":
        label_names = SAFETY_FAMILYOS_LABELS
    elif task == "nli":
        label_names = NLI_LABELS
    elif task == "ingress":
        label_names = INGRESS_LABELS

    # Route to evaluation function
    if task in SEQUENCE_LABELING_TASKS:
        return evaluate_sequence_labeling(
            model, task, samples, tokenizer, device, batch_size, label_names
        )
    elif task in PAIR_TASKS:
        return evaluate_pair_classification(
            model, task, samples, tokenizer, device, batch_size, label_names
        )
    elif task in EMBEDDING_TASKS:
        return evaluate_embedding(model, samples, tokenizer, device, batch_size)
    else:
        return evaluate_classification(
            model, task, samples, tokenizer, device, batch_size, label_names
        )


# =============================================================================
# Quality Gate Checking
# =============================================================================


def check_quality_gates(results: dict[str, dict]) -> dict[str, dict]:
    """Check results against quality gates."""
    gate_results = {}

    for task, task_results in results.items():
        if task not in QUALITY_GATES:
            continue

        gate = QUALITY_GATES[task]
        metric = gate["metric"]
        threshold = gate["threshold"]
        direction = gate["direction"]

        value = task_results.get(metric)
        if value is None:
            gate_results[task] = {
                "status": "MISSING",
                "metric": metric,
                "threshold": threshold,
                "value": None,
                "message": f"Metric '{metric}' not found in results",
            }
            continue

        if direction == "higher":
            passed = value >= threshold
        else:
            passed = value <= threshold

        gate_results[task] = {
            "status": "PASS" if passed else "FAIL",
            "metric": metric,
            "threshold": threshold,
            "value": value,
            "delta": value - threshold,
        }

        # Special check for safety_familyos CRISIS recall
        if task == "safety_familyos" and "per_class" in task_results:
            crisis_metrics = task_results["per_class"].get("CRISIS", {})
            crisis_recall = crisis_metrics.get("recall", 0.0)
            crisis_passed = crisis_recall >= CRISIS_RECALL_GATE["threshold"]

            gate_results[f"{task}_crisis_recall"] = {
                "status": "PASS" if crisis_passed else "FAIL",
                "metric": "crisis_recall",
                "threshold": CRISIS_RECALL_GATE["threshold"],
                "value": crisis_recall,
                "delta": crisis_recall - CRISIS_RECALL_GATE["threshold"],
            }

    return gate_results


def summarize_gates(gate_results: dict[str, dict]) -> dict:
    """Summarize gate results."""
    passed = sum(1 for r in gate_results.values() if r["status"] == "PASS")
    failed = sum(1 for r in gate_results.values() if r["status"] == "FAIL")
    missing = sum(1 for r in gate_results.values() if r["status"] == "MISSING")

    return {
        "total": len(gate_results),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "pass_rate": passed / len(gate_results) if gate_results else 0.0,
        "all_passed": failed == 0 and missing == 0,
    }


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report(
    results: dict,
    gate_results: dict,
    gate_summary: dict,
    model_path: str,
    baseline_results: dict = None,
) -> str:
    """Generate markdown evaluation report."""
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Model**: `{model_path}`")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    status = "✅ ALL PASSED" if gate_summary["all_passed"] else "❌ SOME FAILED"
    lines.append(f"**Overall Status**: {status}")
    lines.append(f"**Gates Passed**: {gate_summary['passed']}/{gate_summary['total']}")
    lines.append("")

    # Quality Gates Table
    lines.append("## Quality Gates")
    lines.append("")
    lines.append("| Task | Metric | Threshold | Value | Status |")
    lines.append("|------|--------|-----------|-------|--------|")

    for task, gate in gate_results.items():
        status_icon = (
            "✅" if gate["status"] == "PASS" else "❌" if gate["status"] == "FAIL" else "⚠️"
        )
        value = f"{gate['value']:.4f}" if gate["value"] is not None else "N/A"
        lines.append(
            f"| {task} | {gate['metric']} | {gate['threshold']:.2f} | {value} | {status_icon} |"
        )

    lines.append("")

    # Detailed Results
    lines.append("## Detailed Results")
    lines.append("")

    for task, task_results in results.items():
        lines.append(f"### {task}")
        lines.append("")

        # Main metrics
        for key, value in task_results.items():
            if key not in ["per_class", "confusion_matrix"]:
                if isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.4f}")
                else:
                    lines.append(f"- **{key}**: {value}")

        lines.append("")

    # Comparison with baseline (if provided)
    if baseline_results:
        lines.append("## Comparison with Baseline")
        lines.append("")
        lines.append("| Task | Metric | Baseline | Model | Delta |")
        lines.append("|------|--------|----------|-------|-------|")

        for task in results:
            if task in baseline_results:
                gate = QUALITY_GATES.get(task, {})
                metric = gate.get("metric", "accuracy")

                baseline_val = baseline_results[task].get(metric, 0)
                model_val = results[task].get(metric, 0)
                delta = model_val - baseline_val
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"

                lines.append(
                    f"| {task} | {metric} | {baseline_val:.4f} | {model_val:.4f} | {delta_str} |"
                )

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["all"],
        help="Tasks to evaluate (or 'all', 'stage_a', 'stage_b')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline model for comparison",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["json", "markdown"],
        choices=["json", "markdown"],
        help="Output formats",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    # Resolve tasks
    if "all" in args.tasks:
        tasks = ALL_TASKS
    elif "stage_a" in args.tasks:
        tasks = STAGE_A_TASKS
    elif "stage_b" in args.tasks:
        tasks = STAGE_B_TASKS
    else:
        tasks = args.tasks

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = ModernBertMultiTaskModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"Model loaded. Available heads: {list(model.heads.keys())}")

    # Filter tasks to those with available heads
    available_tasks = []
    for task in tasks:
        if task in model.heads:
            available_tasks.append(task)
        else:
            logger.warning(f"Skipping {task}: head not available in model")

    if not available_tasks:
        logger.error("No available tasks to evaluate!")
        sys.exit(1)

    logger.info(f"Evaluating tasks: {available_tasks}")

    # Evaluate each task
    results = {}
    data_dir = Path(args.data_dir)

    for task in available_tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {task}")
        logger.info(f"{'='*50}")

        # Load data
        samples = load_evaluation_data(task, data_dir, tokenizer, args.max_samples)
        if samples is None or len(samples) == 0:
            logger.warning(f"No evaluation data for {task}, skipping")
            continue

        # Evaluate
        task_results = evaluate_task(model, task, samples, tokenizer, device, args.batch_size)
        results[task] = task_results

        # Log key metrics
        gate = QUALITY_GATES.get(task, {})
        metric = gate.get("metric", "accuracy")
        value = task_results.get(metric, 0)
        threshold = gate.get("threshold", 0)
        status = "✅" if value >= threshold else "❌"
        logger.info(f"{task}: {metric}={value:.4f} (threshold={threshold:.2f}) {status}")

    # Evaluate baseline if provided
    baseline_results = None
    if args.baseline:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating baseline: {args.baseline}")
        logger.info(f"{'='*50}")

        baseline_model = ModernBertMultiTaskModel.from_pretrained(args.baseline)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline)

        baseline_results = {}
        for task in available_tasks:
            if task in baseline_model.heads:
                samples = load_evaluation_data(task, data_dir, baseline_tokenizer, args.max_samples)
                if samples:
                    baseline_results[task] = evaluate_task(
                        baseline_model, task, samples, baseline_tokenizer, device, args.batch_size
                    )

    # Check quality gates
    gate_results = check_quality_gates(results)
    gate_summary = summarize_gates(gate_results)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    for task, gate in gate_results.items():
        status = (
            "✅ PASS"
            if gate["status"] == "PASS"
            else "❌ FAIL" if gate["status"] == "FAIL" else "⚠️ MISSING"
        )
        value = f"{gate['value']:.4f}" if gate["value"] is not None else "N/A"
        logger.info(f"{task}: {value} vs {gate['threshold']:.2f} => {status}")

    logger.info("-" * 60)
    logger.info(f"Total: {gate_summary['passed']}/{gate_summary['total']} passed")
    logger.info(
        f"Overall: {'✅ ALL GATES PASSED' if gate_summary['all_passed'] else '❌ SOME GATES FAILED'}"
    )

    # Save results
    output_dir = Path(args.output) if args.output else Path(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "json" in args.format:
        # Save full results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "tasks": results,
                    "quality_gates": gate_results,
                    "summary": gate_summary,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {results_path}")

        # Save quality gates separately
        gates_path = output_dir / "quality_gates.json"
        with open(gates_path, "w") as f:
            json.dump(
                {
                    "gates": gate_results,
                    "summary": gate_summary,
                },
                f,
                indent=2,
            )
        logger.info(f"Quality gates saved to {gates_path}")

    if "markdown" in args.format:
        report = generate_markdown_report(
            results, gate_results, gate_summary, args.model, baseline_results
        )
        report_path = output_dir / "eval_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    # Exit with appropriate code
    sys.exit(0 if gate_summary["all_passed"] else 1)


if __name__ == "__main__":
    main()

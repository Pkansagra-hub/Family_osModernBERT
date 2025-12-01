"""
Evaluation Metrics

This module provides metric computation for all tasks in the multi-task model.

Metrics by Task:
    NER (token_classification):
        - Entity-level F1, Precision, Recall
        - Per-entity-type metrics
        - Span-based evaluation using seqeval

    Classification (single_label_classification):
        - Accuracy
        - Macro/Micro F1
        - Per-class F1

    Multi-label (multi_label_classification):
        - Micro/Macro F1
        - Hamming loss
        - Subset accuracy

    NLI:
        - Accuracy
        - Per-class accuracy

    Embedding:
        - Spearman correlation (STS)
        - Pearson correlation

Aggregation:
    - Average score across tasks
    - Weighted average by task importance
    - Per-task primary metric extraction

Usage:
    from modeling_studio.evaluation.metrics import (
        compute_metrics_for_task,
        get_task_primary_metric,
        aggregate_metrics,
    )

    metrics = compute_metrics_for_task(
        task="ner_general",
        predictions=predictions,
        labels=labels,
        label_list=label_list,
    )
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Task Type Mapping
# =============================================================================

# Map task names to their problem types
TASK_PROBLEM_TYPES: dict[str, str] = {
    # Token classification
    "ner_general": "token_classification",
    "ner_family": "token_classification",
    "temporal": "token_classification",
    # Single-label classification
    "sentiment": "single_label_classification",
    "ingress": "single_label_classification",
    "safety_familyos": "single_label_classification",
    "intent": "single_label_classification",
    "nli": "single_label_classification",
    "relation": "single_label_classification",
    # Multi-label classification
    "emotions": "multi_label_classification",
    "safety_generic": "multi_label_classification",
    # Embedding/regression
    "embedding": "regression",
}

# Primary metric for each task (used for model selection)
TASK_PRIMARY_METRICS: dict[str, str] = {
    "ner_general": "f1",
    "ner_family": "f1",
    "temporal": "f1",
    "sentiment": "accuracy",
    "ingress": "accuracy",
    "safety_familyos": "macro_f1",
    "safety_generic": "macro_f1",
    "intent": "accuracy",
    "nli": "accuracy",
    "relation": "f1",
    "emotions": "macro_f1",
    "embedding": "spearman",
}


# =============================================================================
# Token Classification Metrics (NER)
# =============================================================================


def compute_ner_metrics(
    predictions: list[list[int]],
    labels: list[list[int]],
    label_list: list[str],
    ignore_index: int = -100,
) -> dict[str, float]:
    """
    Compute NER metrics using seqeval.

    Args:
        predictions: List of predicted label sequences (int)
        labels: List of true label sequences (int)
        label_list: List of label names (e.g., ["O", "B-PER", "I-PER", ...])
        ignore_index: Label index to ignore (default: -100)

    Returns:
        Dictionary with f1, precision, recall, and per-entity metrics
    """
    try:
        from seqeval.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        logger.warning("seqeval not installed, using fallback NER metrics")
        return _compute_ner_metrics_fallback(predictions, labels)

    # Convert predictions and labels to string labels
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels, strict=False):
        true_seq = []
        pred_seq_str = []

        for pred, label in zip(pred_seq, label_seq, strict=False):
            if label != ignore_index:
                true_seq.append(label_list[label])
                # Clamp prediction to valid range
                pred_idx = max(0, min(pred, len(label_list) - 1))
                pred_seq_str.append(label_list[pred_idx])

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_str)

    # Compute metrics
    metrics = {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }

    # Add per-entity type metrics
    try:
        report = classification_report(true_labels, pred_labels, output_dict=True)
        for entity_type, entity_metrics in report.items():
            if isinstance(entity_metrics, dict) and entity_type not in [
                "micro avg",
                "macro avg",
                "weighted avg",
            ]:
                metrics[f"{entity_type}_f1"] = entity_metrics.get("f1-score", 0.0)
    except Exception:
        pass

    return metrics


def _compute_ner_metrics_fallback(
    predictions: list[list[int]],
    labels: list[list[int]],
    ignore_index: int = -100,
) -> dict[str, float]:
    """Fallback NER metrics when seqeval is not available."""
    correct = 0
    total = 0

    for pred_seq, label_seq in zip(predictions, labels, strict=False):
        for pred, label in zip(pred_seq, label_seq, strict=False):
            if label != ignore_index:
                total += 1
                if pred == label:
                    correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "f1": accuracy,  # Approximation
        "precision": accuracy,
        "recall": accuracy,
        "accuracy": accuracy,
    }


# =============================================================================
# Classification Metrics
# =============================================================================


def compute_classification_metrics(
    predictions: np.ndarray | list,
    labels: np.ndarray | list,
    num_labels: int | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels (int) or logits (if 2D, argmax will be applied)
        labels: True labels (int)
        num_labels: Number of labels (for per-class metrics)

    Returns:
        Dictionary with accuracy, f1, precision, recall
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Handle logits (2D array) - take argmax
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Filter out invalid labels
    valid_mask = labels >= 0
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    if len(labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }

    return metrics


# =============================================================================
# Multi-Label Classification Metrics
# =============================================================================


def compute_multilabel_metrics(
    predictions: np.ndarray | list,
    labels: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute multi-label classification metrics.

    Args:
        predictions: Predicted logits/probabilities (2D: samples x labels)
        labels: True multi-hot labels (2D: samples x labels)
        threshold: Threshold for converting probabilities to predictions

    Returns:
        Dictionary with micro_f1, macro_f1, hamming_loss, subset_accuracy
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Convert probabilities to binary predictions
    if predictions.dtype in [np.float32, np.float64, float]:
        # Apply sigmoid if logits (can have values outside 0-1)
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = 1 / (1 + np.exp(-predictions))
        predictions = (predictions > threshold).astype(int)

    metrics = {
        "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(labels, predictions),
        "subset_accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall": recall_score(labels, predictions, average="micro", zero_division=0),
    }

    return metrics


# =============================================================================
# Embedding/STS Metrics
# =============================================================================


def compute_embedding_metrics(
    predictions: np.ndarray | list,
    labels: np.ndarray | list,
) -> dict[str, float]:
    """
    Compute embedding similarity metrics (for STS tasks).

    Args:
        predictions: Predicted similarity scores
        labels: True similarity scores

    Returns:
        Dictionary with spearman, pearson correlations
    """
    from scipy.stats import pearsonr, spearmanr

    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()

    # Filter out invalid entries
    valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    if len(predictions) < 2:
        return {"spearman": 0.0, "pearson": 0.0}

    # Suppress warnings for constant arrays (common early in training)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="An input array is constant")
        spearman_corr, _ = spearmanr(predictions, labels)
        pearson_corr, _ = pearsonr(predictions, labels)

    return {
        "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
    }


# =============================================================================
# NLI Metrics
# =============================================================================


def compute_nli_metrics(
    predictions: np.ndarray | list,
    labels: np.ndarray | list,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute NLI-specific metrics.

    Args:
        predictions: Predicted labels (int) or logits (if 2D, argmax will be applied)
        labels: True labels (int): 0=entailment, 1=neutral, 2=contradiction
        label_names: Optional label names for per-class metrics

    Returns:
        Dictionary with accuracy, macro_f1, per-class f1 scores
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Handle logits (2D array) - take argmax
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten()
    labels = labels.flatten()

    # Filter out invalid labels
    valid_mask = labels >= 0
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    if len(labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0}

    # Default NLI labels
    if label_names is None:
        label_names = ["entailment", "neutral", "contradiction"]

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        if i < len(per_class_f1):
            metrics[f"f1_{name}"] = float(per_class_f1[i])

    return metrics


# =============================================================================
# Relation Classification Metrics
# =============================================================================


def compute_relation_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    label_names: list[str] | None = None,
    ignore_no_relation: bool = True,
) -> dict[str, float]:
    """
    Compute relation classification metrics.

    Args:
        predictions: Predicted relation labels (int)
        references: True relation labels (int)
        label_names: Optional relation type names
        ignore_no_relation: Whether to exclude "no_relation" (label 0) from F1 computation

    Returns:
        Dictionary with f1, precision, recall, accuracy, and per-relation metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = np.asarray(predictions)
    references = np.asarray(references)

    # Handle logits (2D array) - take argmax
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten()
    references = references.flatten()

    # Filter out invalid labels
    valid_mask = references >= 0
    predictions = predictions[valid_mask]
    references = references[valid_mask]

    if len(references) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

    # Overall accuracy (including no_relation)
    accuracy = accuracy_score(references, predictions)

    # For relation metrics, optionally exclude no_relation (typically label 0)
    if ignore_no_relation:
        # Compute F1 only on actual relations (excluding no_relation class)
        relation_mask = references > 0
        rel_preds = predictions[relation_mask]
        rel_refs = references[relation_mask]

        if len(rel_refs) > 0:
            f1 = f1_score(rel_refs, rel_preds, average="micro", zero_division=0)
            precision = precision_score(rel_refs, rel_preds, average="micro", zero_division=0)
            recall = recall_score(rel_refs, rel_preds, average="micro", zero_division=0)
            macro_f1 = f1_score(rel_refs, rel_preds, average="macro", zero_division=0)
        else:
            f1 = precision = recall = macro_f1 = 0.0
    else:
        f1 = f1_score(references, predictions, average="micro", zero_division=0)
        precision = precision_score(references, predictions, average="micro", zero_division=0)
        recall = recall_score(references, predictions, average="micro", zero_division=0)
        macro_f1 = f1_score(references, predictions, average="macro", zero_division=0)

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }

    # Per-relation F1 if label names provided
    if label_names is not None:
        all_labels = list(range(len(label_names)))
        per_class_f1 = f1_score(
            references, predictions, labels=all_labels, average=None, zero_division=0
        )
        for i, name in enumerate(label_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = float(per_class_f1[i])

    return metrics


# =============================================================================
# Intent Classification Metrics (with Confidence Calibration)
# =============================================================================


def compute_intent_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    confidence_scores: np.ndarray | list | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute intent classification metrics with confidence calibration.

    Args:
        predictions: Predicted intent labels (int)
        references: True intent labels (int)
        confidence_scores: Model confidence scores for predictions (probabilities)
        n_bins: Number of bins for Expected Calibration Error (ECE)

    Returns:
        Dictionary with accuracy, f1, and calibration_error (ECE)
    """
    from sklearn.metrics import accuracy_score, f1_score

    predictions = np.asarray(predictions)
    references = np.asarray(references)

    # Handle logits (2D array)
    if predictions.ndim == 2:
        if confidence_scores is None:
            # Extract confidence from softmax of logits
            exp_logits = np.exp(predictions - predictions.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            confidence_scores = probs.max(axis=-1)
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten()
    references = references.flatten()

    # Filter invalid labels
    valid_mask = references >= 0
    predictions = predictions[valid_mask]
    references = references[valid_mask]

    if confidence_scores is not None:
        confidence_scores = np.asarray(confidence_scores).flatten()
        confidence_scores = confidence_scores[valid_mask]

    if len(references) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "calibration_error": 0.0}

    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="weighted", zero_division=0)
    macro_f1 = f1_score(references, predictions, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "macro_f1": macro_f1,
    }

    # Compute Expected Calibration Error (ECE)
    if confidence_scores is not None and len(confidence_scores) > 0:
        calibration_error = _compute_ece(confidence_scores, predictions, references, n_bins=n_bins)
        metrics["calibration_error"] = calibration_error

    return metrics


def _compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well the model's confidence scores align with actual accuracy.
    A well-calibrated model should have ECE close to 0.

    Args:
        confidences: Model confidence scores
        predictions: Predicted labels
        labels: True labels
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better, 0 is perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this confidence bin
        if i == n_bins - 1:
            # Include upper bound for last bin
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = in_bin.sum()

        if bin_size > 0:
            # Average accuracy in bin
            bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            # Average confidence in bin
            bin_confidence = confidences[in_bin].mean()
            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

    return float(ece)


# =============================================================================
# Temporal Span Extraction Metrics
# =============================================================================


def compute_temporal_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    label_list: list[str],
    scheme: str | None = None,
) -> dict[str, float]:
    """
    Compute temporal span extraction metrics using seqeval.

    Temporal spans follow BIO/BILOU tagging for date/time entities.

    Args:
        predictions: Predicted label indices (2D: batch x seq_len)
        references: True label indices (2D: batch x seq_len)
        label_list: List of label names (e.g., ["O", "B-DATE", "I-DATE", "B-TIME", ...])
        scheme: Tagging scheme ("IOB2", "IOE2", "IOBES", "BILOU") or None for auto

    Returns:
        Dictionary with f1, precision, recall, and per-entity-type metrics
    """
    from seqeval.metrics import f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    predictions = (
        np.asarray(predictions) if not isinstance(predictions, np.ndarray) else predictions
    )
    references = np.asarray(references) if not isinstance(references, np.ndarray) else references

    # Handle 3D predictions (logits)
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    # Convert indices to label names
    true_labels = []
    pred_labels = []

    for ref_seq, pred_seq in zip(references, predictions):
        true_seq = []
        pred_seq_labels = []

        for ref_idx, pred_idx in zip(ref_seq, pred_seq):
            # Skip padding tokens (usually -100)
            if ref_idx < 0 or ref_idx >= len(label_list):
                continue

            true_seq.append(label_list[int(ref_idx)])
            # Handle out-of-bounds predictions gracefully
            if pred_idx < 0 or pred_idx >= len(label_list):
                pred_seq_labels.append("O")
            else:
                pred_seq_labels.append(label_list[int(pred_idx)])

        if true_seq:  # Only add non-empty sequences
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)

    if not true_labels:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Use seqeval for entity-level metrics
    try:
        overall_f1 = f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        overall_precision = precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        overall_recall = recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
    except Exception:
        # Fallback to default mode if scheme doesn't match
        overall_f1 = f1_score(true_labels, pred_labels)
        overall_precision = precision_score(true_labels, pred_labels)
        overall_recall = recall_score(true_labels, pred_labels)

    metrics = {
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
    }

    # Per-entity-type metrics for temporal types
    try:
        from seqeval.metrics import classification_report

        report = classification_report(true_labels, pred_labels, output_dict=True)
        # Extract per-type metrics for common temporal entities
        temporal_types = ["DATE", "TIME", "DURATION", "SET"]
        for entity_type in temporal_types:
            if entity_type in report:
                metrics[f"f1_{entity_type.lower()}"] = report[entity_type]["f1-score"]
                metrics[f"precision_{entity_type.lower()}"] = report[entity_type]["precision"]
                metrics[f"recall_{entity_type.lower()}"] = report[entity_type]["recall"]
    except Exception:
        pass  # Per-type metrics are optional

    return metrics


# =============================================================================
# Safety Band Metrics (FamilyOS-Specific)
# =============================================================================


def compute_safety_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    band_names: list[str] | None = None,
    confidence_scores: np.ndarray | list | None = None,
) -> dict[str, float]:
    """
    Compute safety-specific metrics for FamilyOS policy bands.

    Critical: CRISIS (band 3) recall must be very high (≥95%) to ensure
    user safety. This function provides detailed per-band metrics.

    Args:
        predictions: Predicted safety band labels (0=GREEN, 1=AMBER, 2=RED, 3=CRISIS)
        references: True safety band labels
        band_names: Band names (default: ["GREEN", "AMBER", "RED", "CRISIS"])
        confidence_scores: Optional confidence scores for calibration metrics

    Returns:
        Dictionary with overall and per-band metrics, plus CRISIS recall
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    predictions = np.asarray(predictions)
    references = np.asarray(references)

    # Handle logits
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten()
    references = references.flatten()

    # Filter invalid labels
    valid_mask = references >= 0
    predictions = predictions[valid_mask]
    references = references[valid_mask]

    if len(references) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "crisis_recall": 0.0}

    if band_names is None:
        band_names = ["GREEN", "AMBER", "RED", "CRISIS"]

    # Overall metrics
    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "macro_f1": f1_score(references, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(references, predictions, average="weighted", zero_division=0),
    }

    # Per-band metrics
    num_bands = len(band_names)
    per_band_precision = precision_score(
        references, predictions, labels=list(range(num_bands)), average=None, zero_division=0
    )
    per_band_recall = recall_score(
        references, predictions, labels=list(range(num_bands)), average=None, zero_division=0
    )
    per_band_f1 = f1_score(
        references, predictions, labels=list(range(num_bands)), average=None, zero_division=0
    )

    for i, band in enumerate(band_names):
        if i < len(per_band_precision):
            band_lower = band.lower()
            metrics[f"precision_{band_lower}"] = float(per_band_precision[i])
            metrics[f"recall_{band_lower}"] = float(per_band_recall[i])
            metrics[f"f1_{band_lower}"] = float(per_band_f1[i])

    # CRISIS recall is critical - extract it explicitly
    crisis_idx = 3  # CRISIS is typically index 3
    if crisis_idx < len(per_band_recall):
        metrics["crisis_recall"] = float(per_band_recall[crisis_idx])
    else:
        metrics["crisis_recall"] = 0.0

    # Confusion matrix
    try:
        cm = confusion_matrix(references, predictions, labels=list(range(num_bands)))
        metrics["confusion_matrix"] = cm.tolist()
    except Exception:
        pass

    # Calibration if confidence provided
    if confidence_scores is not None:
        confidence_scores = np.asarray(confidence_scores).flatten()
        confidence_scores = confidence_scores[valid_mask]
        if len(confidence_scores) > 0:
            metrics["calibration_error"] = _compute_ece(
                confidence_scores, predictions, references, n_bins=10
            )

    return metrics


# =============================================================================
# Ingress Domain Metrics (FamilyOS-Specific)
# =============================================================================


def compute_ingress_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    domain_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute ingress domain classification metrics.

    Provides per-domain accuracy to identify domain confusion patterns.

    Args:
        predictions: Predicted domain labels
        references: True domain labels
        domain_names: Domain names (default: 12 FamilyOS domains)

    Returns:
        Dictionary with overall and per-domain metrics
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    predictions = np.asarray(predictions)
    references = np.asarray(references)

    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten()
    references = references.flatten()

    valid_mask = references >= 0
    predictions = predictions[valid_mask]
    references = references[valid_mask]

    if len(references) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    if domain_names is None:
        domain_names = [
            "CALENDAR",
            "REMINDERS",
            "MESSAGING",
            "PHOTOS",
            "HEALTH",
            "FINANCE",
            "SHOPPING",
            "TRAVEL",
            "EDUCATION",
            "SOCIAL",
            "AUTOMATION",
            "OTHER",
        ]

    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "macro_f1": f1_score(references, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(references, predictions, average="weighted", zero_division=0),
    }

    # Per-domain accuracy
    num_domains = len(domain_names)
    for i, domain in enumerate(domain_names):
        if i < num_domains:
            domain_mask = references == i
            if domain_mask.sum() > 0:
                domain_acc = (predictions[domain_mask] == i).mean()
                metrics[f"accuracy_{domain.lower()}"] = float(domain_acc)

    # Top confused domain pairs
    try:
        cm = confusion_matrix(references, predictions, labels=list(range(num_domains)))
        metrics["confusion_matrix"] = cm.tolist()

        # Find most confused pairs (off-diagonal)
        confused_pairs = []
        for i in range(num_domains):
            for j in range(num_domains):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((domain_names[i], domain_names[j], int(cm[i, j])))
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        if confused_pairs:
            metrics["top_confused_pairs"] = confused_pairs[:5]
    except Exception:
        pass

    return metrics


# =============================================================================
# Family NER Metrics (FamilyOS-Specific)
# =============================================================================


def compute_ner_family_metrics(
    predictions: np.ndarray | list,
    references: np.ndarray | list,
    label_list: list[str],
) -> dict[str, float]:
    """
    Compute family-specific NER metrics with per-entity-type breakdown.

    Provides detailed metrics for family entities: KINSHIP, TRADITION,
    MILESTONE, HEIRLOOM, etc.

    Args:
        predictions: Predicted NER tag indices (2D: batch x seq_len)
        references: True NER tag indices (2D: batch x seq_len)
        label_list: List of BIO label names

    Returns:
        Dictionary with overall and per-entity-type metrics
    """
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    predictions = (
        np.asarray(predictions) if not isinstance(predictions, np.ndarray) else predictions
    )
    references = np.asarray(references) if not isinstance(references, np.ndarray) else references

    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    # Convert to label sequences
    true_labels = []
    pred_labels = []

    for ref_seq, pred_seq in zip(references, predictions):
        true_seq = []
        pred_seq_labels = []

        for ref_idx, pred_idx in zip(ref_seq, pred_seq):
            if ref_idx < 0 or ref_idx >= len(label_list):
                continue

            true_seq.append(label_list[int(ref_idx)])
            if pred_idx < 0 or pred_idx >= len(label_list):
                pred_seq_labels.append("O")
            else:
                pred_seq_labels.append(label_list[int(pred_idx)])

        if true_seq:
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)

    if not true_labels:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Overall metrics
    try:
        overall_f1 = f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        overall_precision = precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        overall_recall = recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
    except Exception:
        overall_f1 = f1_score(true_labels, pred_labels)
        overall_precision = precision_score(true_labels, pred_labels)
        overall_recall = recall_score(true_labels, pred_labels)

    metrics = {
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
    }

    # Per-entity-type metrics for family entities
    try:
        report = classification_report(true_labels, pred_labels, output_dict=True)

        # Family-specific entity types
        family_types = [
            "KINSHIP",
            "PET",
            "TRADITION",
            "MILESTONE",
            "HEIRLOOM",
            "PERSON",
            "LOCATION",
            "EVENT",
            "DATE",
            "TIME",
        ]
        for entity_type in family_types:
            if entity_type in report:
                metrics[f"f1_{entity_type.lower()}"] = report[entity_type]["f1-score"]
                metrics[f"precision_{entity_type.lower()}"] = report[entity_type]["precision"]
                metrics[f"recall_{entity_type.lower()}"] = report[entity_type]["recall"]
                metrics[f"support_{entity_type.lower()}"] = report[entity_type]["support"]
    except Exception:
        pass

    return metrics


# =============================================================================
# Embedding Triplet Metrics (FamilyOS-Specific)
# =============================================================================


def compute_embedding_triplet_metrics(
    anchor_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    margin: float = 0.5,
) -> dict[str, float]:
    """
    Compute triplet-based embedding metrics.

    For evaluating embedding quality on triplet data (anchor, positive, negative).

    Args:
        anchor_embeddings: Anchor sentence embeddings (N x dim)
        positive_embeddings: Positive sentence embeddings (N x dim)
        negative_embeddings: Negative sentence embeddings (N x dim)
        margin: Margin for triplet loss computation

    Returns:
        Dictionary with triplet accuracy, average margins, and distances
    """
    anchor_embeddings = np.asarray(anchor_embeddings)
    positive_embeddings = np.asarray(positive_embeddings)
    negative_embeddings = np.asarray(negative_embeddings)

    if len(anchor_embeddings) == 0:
        return {"triplet_accuracy": 0.0, "avg_positive_distance": 0.0, "avg_negative_distance": 0.0}

    # Compute distances
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine distance (1 - cosine_similarity)."""
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return 1 - np.sum(a_norm * b_norm, axis=-1)

    pos_distances = cosine_distance(anchor_embeddings, positive_embeddings)
    neg_distances = cosine_distance(anchor_embeddings, negative_embeddings)

    # Triplet accuracy: positive closer than negative
    triplet_correct = (pos_distances < neg_distances).sum()
    triplet_accuracy = triplet_correct / len(anchor_embeddings)

    # Triplet accuracy with margin
    triplet_correct_margin = (pos_distances + margin < neg_distances).sum()
    triplet_accuracy_margin = triplet_correct_margin / len(anchor_embeddings)

    # Average distances
    avg_pos_dist = float(np.mean(pos_distances))
    avg_neg_dist = float(np.mean(neg_distances))
    avg_margin = float(np.mean(neg_distances - pos_distances))

    metrics = {
        "triplet_accuracy": float(triplet_accuracy),
        "triplet_accuracy_margin": float(triplet_accuracy_margin),
        "avg_positive_distance": avg_pos_dist,
        "avg_negative_distance": avg_neg_dist,
        "avg_margin": avg_margin,
    }

    return metrics


# =============================================================================
# Task-Specific Metric Dispatch
# =============================================================================


def compute_metrics_for_task(
    task: str,
    predictions: np.ndarray | list,
    labels: np.ndarray | list,
    label_list: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Compute metrics for a specific task.

    Args:
        task: Task name (e.g., "ner_general", "sentiment")
        predictions: Model predictions
        labels: Ground truth labels
        label_list: Label names (required for NER tasks)
        **kwargs: Additional arguments for specific metric functions

    Returns:
        Dictionary of metrics for the task
    """
    problem_type = TASK_PROBLEM_TYPES.get(task, "single_label_classification")

    if problem_type == "token_classification":
        if label_list is None:
            logger.warning(f"label_list not provided for {task}, using fallback metrics")
            return _compute_ner_metrics_fallback(predictions, labels)
        return compute_ner_metrics(predictions, labels, label_list, **kwargs)

    elif problem_type == "multi_label_classification":
        return compute_multilabel_metrics(predictions, labels, **kwargs)

    elif problem_type == "regression":
        return compute_embedding_metrics(predictions, labels)

    else:  # single_label_classification
        return compute_classification_metrics(predictions, labels, **kwargs)


def get_task_primary_metric(task: str) -> str:
    """
    Get the primary metric name for a task.

    Args:
        task: Task name

    Returns:
        Primary metric name (e.g., "f1", "accuracy")
    """
    return TASK_PRIMARY_METRICS.get(task, "f1")


def get_task_problem_type(task: str) -> str:
    """
    Get the problem type for a task.

    Args:
        task: Task name

    Returns:
        Problem type string
    """
    return TASK_PROBLEM_TYPES.get(task, "single_label_classification")


# =============================================================================
# Metric Aggregation
# =============================================================================


def aggregate_metrics(
    per_task_metrics: dict[str, dict[str, float]],
    task_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Aggregate metrics across tasks.

    Args:
        per_task_metrics: Dict of task_name -> metrics dict
        task_weights: Optional weights for weighted average

    Returns:
        Dictionary with aggregated metrics (avg_score, worst_score, etc.)
    """
    if not per_task_metrics:
        return {}

    # Extract primary metric for each task
    primary_scores = {}
    for task, metrics in per_task_metrics.items():
        primary_metric = get_task_primary_metric(task)
        if primary_metric in metrics:
            primary_scores[task] = metrics[primary_metric]
        elif "f1" in metrics:
            primary_scores[task] = metrics["f1"]
        elif "accuracy" in metrics:
            primary_scores[task] = metrics["accuracy"]

    if not primary_scores:
        return {}

    # Compute aggregates
    scores = list(primary_scores.values())

    aggregated = {
        "avg_score": float(np.mean(scores)),
        "worst_score": float(np.min(scores)),
        "best_score": float(np.max(scores)),
    }

    # Weighted average if weights provided
    if task_weights:
        weighted_sum = sum(
            primary_scores.get(task, 0.0) * weight
            for task, weight in task_weights.items()
            if task in primary_scores
        )
        total_weight = sum(
            weight for task, weight in task_weights.items() if task in primary_scores
        )
        if total_weight > 0:
            aggregated["weighted_avg_score"] = weighted_sum / total_weight

    return aggregated


# =============================================================================
# Compute Metrics Function for Trainer
# =============================================================================


def create_compute_metrics_fn(
    task: str,
    label_list: list[str] | None = None,
):
    """
    Create a compute_metrics function for HuggingFace Trainer.

    Args:
        task: Task name
        label_list: Label names (for NER tasks)

    Returns:
        Function compatible with Trainer's compute_metrics
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        metrics = compute_metrics_for_task(
            task=task,
            predictions=predictions,
            labels=labels,
            label_list=label_list,
        )

        return metrics

    return compute_metrics


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Generic metrics
    "compute_ner_metrics",
    "compute_classification_metrics",
    "compute_multilabel_metrics",
    "compute_embedding_metrics",
    # Task-specific metrics (v2)
    "compute_nli_metrics",
    "compute_relation_metrics",
    "compute_intent_metrics",
    "compute_temporal_metrics",
    # FamilyOS-specific metrics
    "compute_safety_metrics",
    "compute_ingress_metrics",
    "compute_ner_family_metrics",
    "compute_embedding_triplet_metrics",
    # Dispatch and aggregation
    "compute_metrics_for_task",
    "get_task_primary_metric",
    "get_task_problem_type",
    "aggregate_metrics",
    "create_compute_metrics_fn",
    "TASK_PROBLEM_TYPES",
    "TASK_PRIMARY_METRICS",
]

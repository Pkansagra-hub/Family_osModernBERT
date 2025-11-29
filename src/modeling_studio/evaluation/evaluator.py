"""
Evaluation Runner

This module provides the evaluation pipeline for running inference
and computing metrics on test datasets.

Features:
    - Batch inference on multiple tasks
    - Per-task metric computation
    - Aggregated reporting
    - Export to various formats (JSON, CSV, Markdown)
    - GPU/CPU support

Evaluation Modes:
    - single_task: Evaluate one task at a time
    - multi_task: Evaluate all tasks in one pass

Output:
    - Per-task metrics (F1, accuracy, etc.)
    - Aggregate metrics (avg F1, worst-case)
    - Latency statistics (optional)

Usage:
    from modeling_studio.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        capabilities=["ner_general", "sentiment", "emotions"],
    )

    results = evaluator.evaluate_all(
        datasets={"ner_general": ner_test, "sentiment": sent_test},
        batch_size=32,
    )

    print(results.summary())
    results.save("eval_report.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from modeling_studio.data.labels import CAPABILITY_TO_LABELS, Capability
from modeling_studio.evaluation.metrics import (
    TASK_PROBLEM_TYPES,
    aggregate_metrics,
    compute_classification_metrics,
    compute_embedding_metrics,
    compute_ingress_metrics,
    compute_intent_metrics,
    compute_multilabel_metrics,
    compute_ner_family_metrics,
    compute_ner_metrics,
    compute_nli_metrics,
    compute_relation_metrics,
    compute_safety_metrics,
    compute_temporal_metrics,
    get_task_primary_metric,
)
from modeling_studio.trainers.collators import MultiTaskCollator

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Results
# =============================================================================


@dataclass
class TaskResults:
    """
    Results for a single task evaluation.

    Attributes:
        task: Task name
        metrics: Dictionary of metric name -> value
        num_samples: Number of samples evaluated
        inference_time_ms: Total inference time in milliseconds
        predictions: Raw predictions (optional, for debugging)
        labels: Ground truth labels (optional, for debugging)
    """

    task: str
    metrics: dict[str, float]
    num_samples: int
    inference_time_ms: float = 0.0
    predictions: list[Any] | None = None
    labels: list[Any] | None = None

    @property
    def primary_metric(self) -> float:
        """Get the primary metric for this task."""
        primary_name = get_task_primary_metric(self.task)
        return self.metrics.get(primary_name, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without raw predictions/labels)."""
        return {
            "task": self.task,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "inference_time_ms": self.inference_time_ms,
            "primary_metric": self.primary_metric,
        }


@dataclass
class EvalResults:
    """
    Container for evaluation results across multiple tasks.

    Attributes:
        per_task: Dictionary of task name -> TaskResults
        aggregated: Aggregated metrics across all tasks
        timestamp: Evaluation timestamp
        model_name: Name of the evaluated model
        device: Device used for evaluation
    """

    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    task_results: dict[str, TaskResults] = field(default_factory=dict)
    aggregated: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    model_name: str = ""
    device: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

    def summary(self) -> str:
        """Generate a human-readable summary of results."""
        lines = [
            "=" * 60,
            f"Evaluation Results - {self.model_name}",
            f"Timestamp: {self.timestamp}",
            f"Device: {self.device}",
            "=" * 60,
            "",
            "Per-Task Metrics:",
            "-" * 40,
        ]

        for task, metrics in self.per_task.items():
            primary_metric = get_task_primary_metric(task)
            primary_value = metrics.get(primary_metric, 0.0)
            lines.append(f"  {task}:")
            lines.append(f"    Primary ({primary_metric}): {primary_value:.4f}")
            for name, value in sorted(metrics.items()):
                if name != primary_metric:
                    if isinstance(value, float):
                        lines.append(f"    {name}: {value:.4f}")

        lines.extend(
            [
                "",
                "Aggregated Metrics:",
                "-" * 40,
            ]
        )
        for name, value in sorted(self.aggregated.items()):
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "per_task": self.per_task,
            "aggregated": self.aggregated,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "device": self.device,
        }

    def save(self, path: str | Path, format: str = "json") -> None:
        """
        Save results to file.

        Args:
            path: Output file path
            format: Output format ("json" or "markdown")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "markdown":
            with open(path, "w") as f:
                f.write(self._to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved evaluation results to {path}")

    def _to_markdown(self) -> str:
        """Convert to markdown table format."""
        lines = [
            f"# Evaluation Results: {self.model_name}",
            "",
            f"**Timestamp:** {self.timestamp}",
            f"**Device:** {self.device}",
            "",
            "## Per-Task Metrics",
            "",
            "| Task | Primary Metric | Value |",
            "|------|---------------|-------|",
        ]

        for task, metrics in self.per_task.items():
            primary = get_task_primary_metric(task)
            value = metrics.get(primary, 0.0)
            lines.append(f"| {task} | {primary} | {value:.4f} |")

        lines.extend(
            [
                "",
                "## Aggregated Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )

        for name, value in sorted(self.aggregated.items()):
            if isinstance(value, float):
                lines.append(f"| {name} | {value:.4f} |")

        return "\n".join(lines)


# =============================================================================
# Evaluator Class
# =============================================================================


class Evaluator:
    """
    Evaluator for multi-task models.

    Provides methods for evaluating a model on multiple tasks,
    computing per-task metrics, and generating reports.

    Args:
        model: The multi-task model to evaluate
        tokenizer: Tokenizer for preprocessing
        capabilities: List of capabilities to evaluate (task names)
        device: Device to use for inference ("cuda", "cpu", or "auto")
        label_lists: Optional dict of task -> label list for NER tasks

    Example:
        >>> evaluator = Evaluator(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     capabilities=["ner_general", "sentiment"],
        ... )
        >>> results = evaluator.evaluate_all(datasets, batch_size=32)
        >>> print(results.summary())
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        capabilities: list[str] | None = None,
        device: str = "auto",
        label_lists: dict[str, list[str]] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_lists = label_lists or {}

        # Set capabilities
        if capabilities is None:
            # Try to get from model
            if hasattr(model, "capabilities"):
                self.capabilities = [str(c) for c in model.capabilities]
            else:
                self.capabilities = list(TASK_PROBLEM_TYPES.keys())
        else:
            self.capabilities = capabilities

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create collator
        self.collator = MultiTaskCollator(tokenizer=tokenizer)

        logger.info(
            f"Evaluator initialized with {len(self.capabilities)} capabilities on {self.device}"
        )

    def _get_label_list(self, task: str) -> list[str] | None:
        """Get the label list for a task."""
        # Check explicit label_lists first
        if task in self.label_lists:
            return self.label_lists[task]

        # Try to get from labels module
        try:
            cap = Capability(task)
            schema = CAPABILITY_TO_LABELS.get(cap)
            if schema is not None:
                return [schema.id2label[i] for i in range(schema.num_labels)]
        except (ValueError, KeyError):
            pass

        return None

    def _prepare_batch(
        self,
        batch: dict[str, Any],
        task: str,
    ) -> dict[str, torch.Tensor]:
        """
        Prepare a batch for model inference.

        Args:
            batch: Raw batch from dataloader
            task: Task name

        Returns:
            Dictionary with tensors ready for model forward pass
        """
        model_inputs = {}

        # Required inputs
        if "input_ids" in batch:
            model_inputs["input_ids"] = batch["input_ids"].to(self.device)
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = batch["attention_mask"].to(self.device)

        # Optional inputs
        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

        return model_inputs

    def _extract_labels(
        self,
        batch: dict[str, Any],
        task: str,
    ) -> np.ndarray | list | None:
        """Extract labels from a batch."""
        # Try different label key names
        label_keys = ["labels", "label", "ner_tags", "temporal_tags"]

        for key in label_keys:
            if key in batch:
                labels = batch[key]
                if isinstance(labels, torch.Tensor):
                    return labels.cpu().numpy()
                return labels

        return None

    def _compute_predictions(
        self,
        logits: torch.Tensor,
        task: str,
    ) -> np.ndarray:
        """
        Convert logits to predictions.

        Args:
            logits: Model output logits
            task: Task name

        Returns:
            Predictions as numpy array
        """
        problem_type = TASK_PROBLEM_TYPES.get(task, "single_label_classification")

        if problem_type == "token_classification":
            # Token-level argmax
            predictions = logits.argmax(dim=-1).cpu().numpy()
        elif problem_type == "multi_label_classification":
            # Multi-label: sigmoid + threshold
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        elif problem_type == "regression":
            # Regression: just squeeze
            predictions = logits.squeeze(-1).cpu().numpy()
        else:
            # Single-label: argmax
            predictions = logits.argmax(dim=-1).cpu().numpy()

        return predictions

    def _compute_task_metrics(
        self,
        predictions: np.ndarray | list,
        labels: np.ndarray | list,
        task: str,
    ) -> dict[str, float]:
        """
        Compute metrics for a specific task.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            task: Task name

        Returns:
            Dictionary of metrics
        """
        problem_type = TASK_PROBLEM_TYPES.get(task, "single_label_classification")
        label_list = self._get_label_list(task)

        try:
            # Use task-specific metrics where available
            if task == "safety_familyos":
                return compute_safety_metrics(predictions, labels)
            elif task == "ingress":
                return compute_ingress_metrics(predictions, labels)
            elif task == "ner_family":
                if label_list:
                    return compute_ner_family_metrics(predictions, labels, label_list)
            elif task == "intent":
                return compute_intent_metrics(predictions, labels)
            elif task == "relation":
                return compute_relation_metrics(predictions, labels)
            elif task == "nli":
                return compute_nli_metrics(predictions, labels)
            elif task == "temporal":
                if label_list:
                    return compute_temporal_metrics(predictions, labels, label_list)

            # Fall back to generic metrics based on problem type
            if problem_type == "token_classification":
                if label_list is None:
                    logger.warning(f"No label list for {task}, using classification metrics")
                    return compute_classification_metrics(
                        predictions.flatten(), np.asarray(labels).flatten()
                    )
                return compute_ner_metrics(predictions, labels, label_list)

            elif problem_type == "multi_label_classification":
                return compute_multilabel_metrics(predictions, labels)

            elif problem_type == "regression":
                return compute_embedding_metrics(predictions, labels)

            else:  # single_label_classification
                return compute_classification_metrics(predictions, labels)

        except Exception as e:
            logger.error(f"Error computing metrics for {task}: {e}")
            return {"error": 1.0}

    def evaluate_task(
        self,
        task: str,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        show_progress: bool = True,
    ) -> TaskResults:
        """
        Evaluate a single task.

        Args:
            task: Task name
            dataset: Test dataset for the task
            batch_size: Batch size for inference
            num_workers: Number of dataloader workers
            show_progress: Whether to show progress bar

        Returns:
            TaskResults with metrics and statistics
        """
        logger.info(f"Evaluating task: {task} ({len(dataset)} samples)")

        # Ensure dataset has task field for collator
        def add_task(example):
            example["task"] = task
            return example

        # Check if task field already exists
        if "task" not in dataset.column_names:
            dataset = dataset.map(add_task)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        all_predictions = []
        all_labels = []
        start_time = time.time()

        # Run inference
        iterator = tqdm(dataloader, desc=f"Evaluating {task}") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                # Prepare inputs
                model_inputs = self._prepare_batch(batch, task)

                # Forward pass
                outputs = self.model(
                    **model_inputs,
                    capability=task,
                )

                # Extract logits
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Convert to predictions
                predictions = self._compute_predictions(logits, task)
                all_predictions.append(predictions)

                # Extract labels
                labels = self._extract_labels(batch, task)
                if labels is not None:
                    all_labels.append(labels)

        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        # Concatenate results
        problem_type = TASK_PROBLEM_TYPES.get(task, "single_label_classification")

        if problem_type == "token_classification":
            # Keep as list of sequences for seqeval
            all_predictions = [seq for batch in all_predictions for seq in batch]
            all_labels = [seq for batch in all_labels for seq in batch]
        else:
            all_predictions = np.concatenate(all_predictions, axis=0)
            if all_labels:
                all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        if all_labels:
            metrics = self._compute_task_metrics(all_predictions, all_labels, task)
        else:
            logger.warning(f"No labels found for task {task}, skipping metric computation")
            metrics = {}

        return TaskResults(
            task=task,
            metrics=metrics,
            num_samples=len(dataset),
            inference_time_ms=inference_time_ms,
        )

    def evaluate_all(
        self,
        datasets: dict[str, Dataset],
        batch_size: int = 32,
        num_workers: int = 0,
        show_progress: bool = True,
        task_weights: dict[str, float] | None = None,
    ) -> EvalResults:
        """
        Evaluate all tasks.

        Args:
            datasets: Dictionary of task name -> test dataset
            batch_size: Batch size for inference
            num_workers: Number of dataloader workers
            show_progress: Whether to show progress bar
            task_weights: Optional weights for aggregation

        Returns:
            EvalResults with per-task and aggregated metrics
        """
        logger.info(f"Starting evaluation on {len(datasets)} tasks")

        results = EvalResults(
            model_name=getattr(self.model, "name_or_path", "unknown"),
            device=str(self.device),
        )

        # Evaluate each task
        for task, dataset in datasets.items():
            if task not in self.capabilities:
                logger.warning(f"Task {task} not in model capabilities, skipping")
                continue

            task_results = self.evaluate_task(
                task=task,
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                show_progress=show_progress,
            )

            results.task_results[task] = task_results
            results.per_task[task] = task_results.metrics

        # Compute aggregated metrics
        results.aggregated = aggregate_metrics(results.per_task, task_weights)

        logger.info("Evaluation complete")
        return results

    def evaluate(
        self,
        datasets: dict[str, Dataset],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> EvalResults:
        """
        Alias for evaluate_all() for backward compatibility.

        Args:
            datasets: Dictionary of task name -> test dataset
            batch_size: Batch size for inference
            **kwargs: Additional arguments passed to evaluate_all()

        Returns:
            EvalResults with per-task and aggregated metrics
        """
        return self.evaluate_all(datasets, batch_size, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: dict[str, Dataset],
    batch_size: int = 32,
    device: str = "auto",
) -> EvalResults:
    """
    Quick evaluation utility function.

    Args:
        model: Multi-task model
        tokenizer: Tokenizer
        datasets: Dictionary of task -> dataset
        batch_size: Batch size
        device: Device ("cuda", "cpu", or "auto")

    Returns:
        EvalResults
    """
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        capabilities=list(datasets.keys()),
        device=device,
    )
    return evaluator.evaluate_all(datasets, batch_size=batch_size)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Evaluator",
    "EvalResults",
    "TaskResults",
    "quick_evaluate",
]

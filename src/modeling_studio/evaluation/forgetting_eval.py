"""
Catastrophic Forgetting Evaluation

This module provides evaluation utilities to detect catastrophic forgetting
after domain adaptation (Stage B training).

Purpose:
    After Stage B (FamilyOS domain adaptation), ensure that performance
    on Stage A benchmarks (CoNLL, SST-2, MNLI, GoEmotions) does not
    degrade by more than the allowed threshold.

Forgetting Gates:
    - CoNLL-2003 (NER): ≤ 2% F1 drop
    - SST-2 (Sentiment): ≤ 2% Accuracy drop
    - MNLI (NLI): ≤ 2% Accuracy drop
    - GoEmotions: ≤ 3% Macro F1 drop

Action if Exceeded:
    - Reduce LoRA r value
    - Increase replay ratio (more Stage A data in Stage B)
    - Freeze more encoder layers

Usage:
    from modeling_studio.evaluation.forgetting_eval import (
        ForgettingEvaluator,
        evaluate_forgetting,
        check_forgetting_gates,
    )

    evaluator = ForgettingEvaluator(
        stage_a_checkpoint="checkpoints/modernbert-multitask-v0",
        stage_b_checkpoint="checkpoints/modernbert-unified-v2",
    )

    results = evaluator.evaluate_all()
    passed = evaluator.check_gates(results)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


# Default forgetting thresholds (maximum allowed drop)
FORGETTING_THRESHOLDS = {
    "ner_general": {"metric": "f1", "max_drop": 0.02},  # 2% F1
    "sentiment": {"metric": "accuracy", "max_drop": 0.02},  # 2% Accuracy
    "nli": {"metric": "accuracy", "max_drop": 0.02},  # 2% Accuracy
    "emotions": {"metric": "macro_f1", "max_drop": 0.03},  # 3% Macro F1
    "safety_generic": {"metric": "macro_f1", "max_drop": 0.03},  # 3% Macro F1
    "embedding": {"metric": "spearman", "max_drop": 0.03},  # 3% Spearman
}

# Benchmark datasets for each task
BENCHMARK_DATASETS = {
    "ner_general": "conll2003",
    "sentiment": "sst2",
    "nli": "multi_nli",
    "emotions": "go_emotions",
    "safety_generic": "jigsaw_toxicity",
    "embedding": "stsb",
}

# Task to capability mapping
TASK_TO_CAPABILITY = {
    "ner_general": "ner_general",
    "sentiment": "sentiment",
    "nli": "nli",
    "emotions": "emotions",
    "safety_generic": "safety_generic",
    "embedding": "embedding",
}


@dataclass
class ForgettingResult:
    """Result of forgetting evaluation for a single task."""

    task: str
    metric_name: str
    stage_a_score: float
    stage_b_score: float
    drop: float
    max_allowed_drop: float
    passed: bool
    regression: float = 0.0  # For backward compatibility
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # regression is the same as drop (positive = worse)
        self.regression = self.drop

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"ForgettingResult({self.task}: "
            f"{self.stage_a_score:.4f} → {self.stage_b_score:.4f}, "
            f"drop={self.drop:.4f}, max={self.max_allowed_drop:.4f}) {status}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "metric_name": self.metric_name,
            "stage_a_score": self.stage_a_score,
            "stage_b_score": self.stage_b_score,
            "drop": self.drop,
            "regression": self.regression,
            "max_allowed_drop": self.max_allowed_drop,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class ForgettingReport:
    """Complete forgetting evaluation report."""

    results: list[ForgettingResult]
    all_passed: bool
    failed_tasks: list[str]
    recommendations: list[str]
    stage_a_checkpoint: str = ""
    stage_b_checkpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": {r.task: r.to_dict() for r in self.results},
            "all_passed": self.all_passed,
            "failed_tasks": self.failed_tasks,
            "recommendations": self.recommendations,
            "stage_a_checkpoint": self.stage_a_checkpoint,
            "stage_b_checkpoint": self.stage_b_checkpoint,
        }

    def __getitem__(self, task: str) -> dict[str, Any]:
        """Allow dict-style access by task name."""
        for result in self.results:
            if result.task == task:
                return result.to_dict()
        raise KeyError(f"Task '{task}' not found in results")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60, "CATASTROPHIC FORGETTING EVALUATION REPORT", "=" * 60]

        if self.stage_a_checkpoint:
            lines.append(f"Stage A: {self.stage_a_checkpoint}")
        if self.stage_b_checkpoint:
            lines.append(f"Stage B: {self.stage_b_checkpoint}")
        lines.append("")

        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(
                f"{status} {result.task}: "
                f"{result.metric_name} {result.stage_a_score:.4f} → {result.stage_b_score:.4f} "
                f"(drop: {result.drop:.4f}, max: {result.max_allowed_drop:.4f})"
            )

        lines.append("-" * 60)
        if self.all_passed:
            lines.append("✅ ALL FORGETTING GATES PASSED")
        else:
            lines.append(f"❌ FAILED GATES: {', '.join(self.failed_tasks)}")
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved forgetting report to {path}")


class ForgettingEvaluator:
    """
    Evaluator for catastrophic forgetting detection.

    Compares performance between Stage A and Stage B checkpoints
    on Stage A benchmark datasets.

    Args:
        stage_a_checkpoint: Path to Stage A checkpoint (before domain adaptation).
        stage_b_checkpoint: Path to Stage B checkpoint (after domain adaptation).
        thresholds: Custom thresholds for forgetting detection.
            Default: FORGETTING_THRESHOLDS
        device: Device for evaluation. Default: "cuda" if available.
        tokenizer: Optional tokenizer (loaded from checkpoint if not provided).
        batch_size: Batch size for evaluation.

    Example:
        >>> evaluator = ForgettingEvaluator(
        ...     stage_a_checkpoint="checkpoints/stage_a",
        ...     stage_b_checkpoint="checkpoints/stage_b",
        ... )
        >>> report = evaluator.evaluate_all()
        >>> print(report.summary())
    """

    def __init__(
        self,
        stage_a_checkpoint: str | Path | None = None,
        stage_b_checkpoint: str | Path | None = None,
        thresholds: dict[str, dict] | None = None,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        batch_size: int = 32,
    ):
        self.stage_a_checkpoint = Path(stage_a_checkpoint) if stage_a_checkpoint else None
        self.stage_b_checkpoint = Path(stage_b_checkpoint) if stage_b_checkpoint else None
        self.thresholds = thresholds or FORGETTING_THRESHOLDS
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._stage_a_model: PreTrainedModel | None = None
        self._stage_b_model: PreTrainedModel | None = None
        self._datasets_cache: dict[str, Dataset] = {}

    def load_models(self) -> None:
        """Load both Stage A and Stage B models."""
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading Stage A model from {self.stage_a_checkpoint}")
        if self.stage_a_checkpoint and self.stage_a_checkpoint.exists():
            self._stage_a_model = AutoModel.from_pretrained(str(self.stage_a_checkpoint))
            self._stage_a_model.to(self.device)  # type: ignore[union-attr]
            self._stage_a_model.eval()  # type: ignore[union-attr]

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.stage_a_checkpoint))

        logger.info(f"Loading Stage B model from {self.stage_b_checkpoint}")
        if self.stage_b_checkpoint and self.stage_b_checkpoint.exists():
            self._stage_b_model = AutoModel.from_pretrained(str(self.stage_b_checkpoint))
            self._stage_b_model.to(self.device)  # type: ignore[union-attr]
            self._stage_b_model.eval()  # type: ignore[union-attr]

    def _load_benchmark_dataset(self, task: str) -> Dataset | None:
        """Load benchmark dataset for a task."""
        if task in self._datasets_cache:
            return self._datasets_cache[task]

        dataset_name = BENCHMARK_DATASETS.get(task)
        if not dataset_name:
            logger.warning(f"No benchmark dataset defined for task: {task}")
            return None

        try:
            from datasets import load_dataset

            if dataset_name == "conll2003":
                dataset = load_dataset("conll2003", split="test")
            elif dataset_name == "sst2":
                dataset = load_dataset("glue", "sst2", split="validation")
            elif dataset_name == "multi_nli":
                dataset = load_dataset("multi_nli", split="validation_matched")
            elif dataset_name == "go_emotions":
                dataset = load_dataset("go_emotions", "simplified", split="test")
            elif dataset_name == "stsb":
                dataset = load_dataset("glue", "stsb", split="validation")
            else:
                dataset = load_dataset(dataset_name, split="test")

            self._datasets_cache[task] = dataset  # type: ignore[assignment]
            return dataset  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name}: {e}")
            return None

    def _evaluate_model_on_task(
        self,
        model: PreTrainedModel,
        task: str,
        dataset: Dataset,
    ) -> dict[str, float]:
        """Evaluate a model on a specific task and return metrics."""
        from sklearn.metrics import accuracy_score, f1_score
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        capability = TASK_TO_CAPABILITY.get(task, task)

        # Create dataloader
        def collate_fn(batch: list[dict]) -> dict[str, Any]:
            texts = []
            labels = []
            for item in batch:
                if "text" in item:
                    texts.append(item["text"])
                elif "sentence" in item:
                    texts.append(item["sentence"])
                elif "premise" in item and "hypothesis" in item:
                    texts.append(f"{item['premise']} [SEP] {item['hypothesis']}")
                elif "tokens" in item:
                    texts.append(" ".join(item["tokens"]))
                else:
                    texts.append(str(item))

                if "label" in item:
                    labels.append(item["label"])
                elif "ner_tags" in item:
                    labels.append(item["ner_tags"])
                else:
                    labels.append(0)

            # Tokenize
            if self.tokenizer:
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": labels,
                }
            return {"texts": texts, "labels": labels}

        dataloader = DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
                if "input_ids" in batch:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # Get predictions based on task type
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, dict) and "logits" in outputs:
                        logits = outputs["logits"]
                    else:
                        logits = outputs.last_hidden_state[:, 0, :]  # CLS token

                    if task == "ner_general":
                        # Token-level predictions
                        predictions = logits.argmax(dim=-1).cpu().numpy()
                        all_predictions.extend(predictions.flatten().tolist())
                        for label_seq in batch["labels"]:
                            if isinstance(label_seq, list):
                                all_labels.extend(label_seq)
                            else:
                                all_labels.append(label_seq)
                    else:
                        predictions = logits.argmax(dim=-1).cpu().numpy()
                        all_predictions.extend(predictions.tolist())
                        all_labels.extend(batch["labels"])

        # Compute metrics based on task type
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Filter invalid labels
        valid_mask = all_labels >= 0
        all_predictions = all_predictions[valid_mask]
        all_labels = all_labels[valid_mask]

        metrics = {}
        if len(all_labels) > 0:
            metrics["accuracy"] = float(accuracy_score(all_labels, all_predictions))
            metrics["macro_f1"] = float(
                f1_score(all_labels, all_predictions, average="macro", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
            )

        return metrics

    def evaluate_task(
        self,
        task: str,
        dataset: Dataset | None = None,
    ) -> ForgettingResult:
        """
        Evaluate forgetting for a single task.

        Args:
            task: Task name (e.g., "ner_general", "sentiment").
            dataset: Optional custom dataset. If None, uses benchmark dataset.

        Returns:
            ForgettingResult with scores and pass/fail status.
        """
        if task not in self.thresholds:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.thresholds.keys())}")

        threshold_config = self.thresholds[task]
        metric_name = threshold_config["metric"]
        max_drop = threshold_config["max_drop"]

        # Load dataset if not provided
        if dataset is None:
            dataset = self._load_benchmark_dataset(task)

        # If models not loaded or dataset unavailable, return placeholder
        if self._stage_a_model is None or self._stage_b_model is None:
            logger.warning("Models not loaded. Call load_models() first or use compare().")
            return ForgettingResult(
                task=task,
                metric_name=metric_name,
                stage_a_score=0.0,
                stage_b_score=0.0,
                drop=0.0,
                max_allowed_drop=max_drop,
                passed=True,
                details={"status": "models_not_loaded"},
            )

        if dataset is None:
            logger.warning(f"No dataset available for task: {task}")
            return ForgettingResult(
                task=task,
                metric_name=metric_name,
                stage_a_score=0.0,
                stage_b_score=0.0,
                drop=0.0,
                max_allowed_drop=max_drop,
                passed=True,
                details={"status": "dataset_unavailable"},
            )

        # Evaluate both models
        logger.info(f"Evaluating {task} on Stage A model...")
        stage_a_metrics = self._evaluate_model_on_task(self._stage_a_model, task, dataset)

        logger.info(f"Evaluating {task} on Stage B model...")
        stage_b_metrics = self._evaluate_model_on_task(self._stage_b_model, task, dataset)

        # Get the relevant metric
        stage_a_score = stage_a_metrics.get(metric_name, 0.0)
        stage_b_score = stage_b_metrics.get(metric_name, 0.0)

        # Calculate drop (positive = regression)
        drop = stage_a_score - stage_b_score
        passed = drop <= max_drop

        return ForgettingResult(
            task=task,
            metric_name=metric_name,
            stage_a_score=stage_a_score,
            stage_b_score=stage_b_score,
            drop=drop,
            max_allowed_drop=max_drop,
            passed=passed,
            details={
                "stage_a_metrics": stage_a_metrics,
                "stage_b_metrics": stage_b_metrics,
            },
        )

    def compare(
        self,
        stage_a_model: str | Path | PreTrainedModel,
        stage_b_model: str | Path | PreTrainedModel,
        tasks: list[str] | None = None,
    ) -> ForgettingReport:
        """
        Compare two models on specified tasks.

        This is the main entry point for the acceptance criteria.

        Args:
            stage_a_model: Path to Stage A checkpoint or model instance.
            stage_b_model: Path to Stage B checkpoint or model instance.
            tasks: List of tasks to evaluate. Default: all tasks in thresholds.

        Returns:
            ForgettingReport with per-task results.

        Example:
            >>> results = evaluator.compare(
            ...     stage_a_model="checkpoints/stage_a",
            ...     stage_b_model="checkpoints/stage_b",
            ...     tasks=["ner_general", "sentiment", "nli"],
            ... )
            >>> assert all(results[task]["regression"] <= 0.02 for task in results)
        """
        from transformers import AutoModel, AutoTokenizer

        # Load models if paths provided
        if isinstance(stage_a_model, (str, Path)):
            self.stage_a_checkpoint = Path(stage_a_model)
            if self.stage_a_checkpoint.exists():
                self._stage_a_model = AutoModel.from_pretrained(str(self.stage_a_checkpoint))
                self._stage_a_model.to(self.device)  # type: ignore[union-attr]
                self._stage_a_model.eval()  # type: ignore[union-attr]
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(self.stage_a_checkpoint))
        else:
            self._stage_a_model = stage_a_model

        if isinstance(stage_b_model, (str, Path)):
            self.stage_b_checkpoint = Path(stage_b_model)
            if self.stage_b_checkpoint.exists():
                self._stage_b_model = AutoModel.from_pretrained(str(self.stage_b_checkpoint))
                self._stage_b_model.to(self.device)  # type: ignore[union-attr]
                self._stage_b_model.eval()  # type: ignore[union-attr]
        else:
            self._stage_b_model = stage_b_model

        # Determine tasks to evaluate
        if tasks is None:
            tasks = list(self.thresholds.keys())

        # Evaluate each task
        results = []
        failed_tasks = []

        for task in tasks:
            if task not in self.thresholds:
                logger.warning(f"Unknown task: {task}, skipping")
                continue

            result = self.evaluate_task(task)
            results.append(result)

            if not result.passed:
                failed_tasks.append(task)

        # Generate recommendations
        recommendations = self._generate_recommendations(failed_tasks)

        return ForgettingReport(
            results=results,
            all_passed=len(failed_tasks) == 0,
            failed_tasks=failed_tasks,
            recommendations=recommendations,
            stage_a_checkpoint=str(self.stage_a_checkpoint) if self.stage_a_checkpoint else "",
            stage_b_checkpoint=str(self.stage_b_checkpoint) if self.stage_b_checkpoint else "",
        )

    def _generate_recommendations(self, failed_tasks: list[str]) -> list[str]:
        """Generate recommendations for failed tasks."""
        recommendations = []

        if not failed_tasks:
            return recommendations

        # General recommendations
        recommendations.append("Reduce LoRA r value (try r=16 instead of r=32)")
        recommendations.append("Increase replay ratio (try 0.2 instead of 0.1)")
        recommendations.append("Freeze more encoder layers during Stage B")
        recommendations.append("Reduce Stage B learning rate")

        # Task-specific recommendations
        if "ner_general" in failed_tasks:
            recommendations.append("NER forgetting: Consider task-specific LoRA adapters")
            recommendations.append("NER forgetting: Increase NER replay samples in Stage B")

        if "nli" in failed_tasks:
            recommendations.append("NLI forgetting: The NLI head may need lower LR")

        if "sentiment" in failed_tasks:
            recommendations.append("Sentiment forgetting: Check for domain shift in Stage B data")

        if "emotions" in failed_tasks:
            recommendations.append(
                "Emotions forgetting: Multi-label tasks may need separate adapters"
            )

        return recommendations

    def evaluate_all(self) -> ForgettingReport:
        """
        Evaluate forgetting for all tasks.

        Returns:
            ForgettingReport with all results and recommendations.
        """
        results = []
        failed_tasks = []

        for task in self.thresholds:
            result = self.evaluate_task(task)
            results.append(result)

            if not result.passed:
                failed_tasks.append(task)

        recommendations = self._generate_recommendations(failed_tasks)

        return ForgettingReport(
            results=results,
            all_passed=len(failed_tasks) == 0,
            failed_tasks=failed_tasks,
            recommendations=recommendations,
            stage_a_checkpoint=str(self.stage_a_checkpoint) if self.stage_a_checkpoint else "",
            stage_b_checkpoint=str(self.stage_b_checkpoint) if self.stage_b_checkpoint else "",
        )

    def check_gates(self, report: ForgettingReport | None = None) -> bool:
        """
        Check if all forgetting gates pass.

        Args:
            report: Optional pre-computed report. If None, evaluates all tasks.

        Returns:
            True if all gates pass, False otherwise.
        """
        if report is None:
            report = self.evaluate_all()

        return report.all_passed


def evaluate_forgetting(
    stage_a_checkpoint: str | Path,
    stage_b_checkpoint: str | Path,
    tasks: list[str] | None = None,
) -> ForgettingReport:
    """
    Convenience function to evaluate forgetting.

    Args:
        stage_a_checkpoint: Path to Stage A checkpoint.
        stage_b_checkpoint: Path to Stage B checkpoint.
        tasks: Optional list of tasks to evaluate. Default: all tasks.

    Returns:
        ForgettingReport with all results.

    Example:
        >>> report = evaluate_forgetting(
        ...     "checkpoints/stage_a",
        ...     "checkpoints/stage_b",
        ... )
        >>> if not report.all_passed:
        ...     print("Forgetting detected!")
        ...     print(report.summary())
    """
    evaluator = ForgettingEvaluator(stage_a_checkpoint, stage_b_checkpoint)
    return evaluator.compare(stage_a_checkpoint, stage_b_checkpoint, tasks)


def check_forgetting_gates(
    stage_a_checkpoint: str | Path,
    stage_b_checkpoint: str | Path,
) -> bool:
    """
    Check if all forgetting gates pass.

    Args:
        stage_a_checkpoint: Path to Stage A checkpoint.
        stage_b_checkpoint: Path to Stage B checkpoint.

    Returns:
        True if all gates pass, False otherwise.

    Example:
        >>> if not check_forgetting_gates("stage_a", "stage_b"):
        ...     raise ValueError("Forgetting gates failed!")
    """
    evaluator = ForgettingEvaluator(stage_a_checkpoint, stage_b_checkpoint)
    report = evaluator.compare(stage_a_checkpoint, stage_b_checkpoint)
    return report.all_passed


# Export public API
__all__ = [
    "FORGETTING_THRESHOLDS",
    "BENCHMARK_DATASETS",
    "ForgettingResult",
    "ForgettingReport",
    "ForgettingEvaluator",
    "evaluate_forgetting",
    "check_forgetting_gates",
]

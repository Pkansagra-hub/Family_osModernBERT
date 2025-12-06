"""
Training Callbacks for Multi-Task Learning

This module provides custom callbacks for monitoring and controlling
multi-task training.

Callbacks:
    - TaskMetricsCallback: Log per-task metrics during training
    - GradientMonitorCallback: Monitor gradient stats per task head
    - EarlyStoppingCallback: Early stopping based on metric
    - ModelCheckpointCallback: Save best model per metric

Usage:
    from modeling_studio.trainers.callbacks import (
        TaskMetricsCallback,
        GradientMonitorCallback,
        EarlyStoppingCallback,
        ModelCheckpointCallback,
    )

    callbacks = [
        TaskMetricsCallback(log_every=100),
        GradientMonitorCallback(),
        EarlyStoppingCallback(patience=3, metric="eval_avg_score"),
        ModelCheckpointCallback(metric="eval_avg_score", save_best_only=True),
    ]

    trainer = MultiTaskTrainer(
        model=model,
        callbacks=callbacks,
        ...
    )
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

if TYPE_CHECKING:
    from transformers import TrainingArguments

    from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer


logger = logging.getLogger(__name__)


# =============================================================================
# Task Metrics Callback
# =============================================================================


@dataclass
class TaskMetricsState:
    """State for tracking per-task metrics."""

    task_losses: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    task_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    step_task_losses: dict[str, float] = field(default_factory=dict)


class TaskMetricsCallback(TrainerCallback):
    """
    Callback for logging per-task metrics during training.

    Tracks:
        - Per-task training loss (running average)
        - Per-task sample counts
        - Loss ratios between tasks

    Args:
        log_every: Log per-task metrics every N steps (default: 100)
        log_to_tensorboard: Whether to log to tensorboard (default: True)
        reset_on_log: Reset running averages after logging (default: True)

    Example:
        callback = TaskMetricsCallback(log_every=100)
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        log_every: int = 100,
        log_to_tensorboard: bool = True,
        reset_on_log: bool = True,
    ):
        self.log_every = log_every
        self.log_to_tensorboard = log_to_tensorboard
        self.reset_on_log = reset_on_log
        self.state = TaskMetricsState()
        self._last_log_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Initialize tracking at start of training."""
        self.state = TaskMetricsState()
        self._last_log_step = 0
        logger.info("TaskMetricsCallback: Initialized per-task tracking")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record per-task loss after each step."""
        trainer: MultiTaskTrainer | None = kwargs.get("model", None)

        # Try to get trainer from kwargs
        if trainer is None:
            trainer = kwargs.get("trainer", None)

        # Get current task from trainer
        current_task = getattr(trainer, "current_task", None) if trainer else None

        # Get loss from state
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            loss = last_log.get("loss")

            if loss is not None and current_task is not None:
                self.state.task_losses[current_task].append(loss)
                self.state.task_counts[current_task] += 1
                self.state.step_task_losses[current_task] = loss

        # Log at specified intervals
        if state.global_step > 0 and state.global_step % self.log_every == 0:
            self._log_task_metrics(args, state, kwargs.get("tb_writer"))

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Append task metrics to log dict."""
        if logs is None:
            return

        # Add current task losses to logs
        for task, losses in self.state.task_losses.items():
            if losses:
                avg_loss = sum(losses) / len(losses)
                logs[f"train_{task}_loss"] = avg_loss
                logs[f"train_{task}_samples"] = self.state.task_counts[task]

    def _log_task_metrics(
        self,
        args: TrainingArguments,
        state: TrainerState,
        tb_writer: Any | None = None,
    ) -> None:
        """Log per-task metrics."""
        if not self.state.task_losses:
            return

        metrics = {}
        for task, losses in self.state.task_losses.items():
            if losses:
                avg_loss = sum(losses) / len(losses)
                metrics[f"task/{task}/loss"] = avg_loss
                metrics[f"task/{task}/count"] = self.state.task_counts[task]

        # Log summary
        if metrics:
            task_summary = ", ".join(
                f"{task}: {sum(losses)/len(losses):.4f}"
                for task, losses in self.state.task_losses.items()
                if losses
            )
            logger.info(f"Step {state.global_step} - Per-task losses: {task_summary}")

            # Log to tensorboard if available
            if self.log_to_tensorboard and tb_writer is not None:
                for key, value in metrics.items():
                    tb_writer.add_scalar(key, value, state.global_step)

        # Reset if configured
        if self.reset_on_log:
            self.state = TaskMetricsState()
            self._last_log_step = state.global_step

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log evaluation metrics per task."""
        if metrics is None:
            return

        # Extract per-task metrics
        task_metrics: dict[str, dict[str, float]] = defaultdict(dict)
        for key, value in metrics.items():
            # Parse keys like "eval_ner_general_f1" -> task="ner_general", metric="f1"
            if key.startswith("eval_"):
                parts = key[5:].rsplit("_", 1)  # Remove "eval_" prefix
                if len(parts) == 2:
                    task_key, metric_name = parts
                    task_metrics[task_key][metric_name] = value

        # Log task-level summary
        for task, task_mets in task_metrics.items():
            summary = ", ".join(f"{k}={v:.4f}" for k, v in task_mets.items())
            logger.info(f"Eval {task}: {summary}")


# =============================================================================
# Gradient Monitor Callback
# =============================================================================


@dataclass
class GradientStats:
    """Statistics for gradient monitoring."""

    norm: float = 0.0
    max_val: float = 0.0
    min_val: float = 0.0
    num_params: int = 0


class GradientMonitorCallback(TrainerCallback):
    """
    Callback for monitoring gradient statistics per task head.

    Tracks:
        - Gradient norms per head
        - Gradient explosion/vanishing detection
        - Head-wise gradient ratios

    Args:
        log_every: Log gradient stats every N steps (default: 100)
        warn_threshold: Warn if gradient norm exceeds this value (default: 10.0)
        track_heads: Whether to track per-head gradients (default: True)

    Example:
        callback = GradientMonitorCallback(log_every=100, warn_threshold=10.0)
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        log_every: int = 100,
        warn_threshold: float = 10.0,
        vanishing_threshold: float = 1e-7,
        track_heads: bool = True,
    ):
        self.log_every = log_every
        self.warn_threshold = warn_threshold
        self.vanishing_threshold = vanishing_threshold
        self.track_heads = track_heads
        self.gradient_history: dict[str, list[GradientStats]] = defaultdict(list)

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Compute gradient statistics before optimizer step."""
        model = kwargs.get("model")
        if model is None:
            return

        # Only log at specified intervals
        if state.global_step % self.log_every != 0:
            return

        # Compute overall gradient norm
        total_norm = self._compute_grad_norm(model)

        # Check for gradient issues
        if total_norm > self.warn_threshold:
            logger.warning(
                f"Step {state.global_step}: Gradient explosion detected! "
                f"Norm = {total_norm:.4f} > {self.warn_threshold}"
            )
        elif total_norm < self.vanishing_threshold:
            logger.warning(
                f"Step {state.global_step}: Vanishing gradients detected! "
                f"Norm = {total_norm:.6e} < {self.vanishing_threshold:.0e}"
            )

        # Track per-head gradients
        if self.track_heads:
            head_norms = self._compute_head_grad_norms(model)
            for head_name, norm in head_norms.items():
                self.gradient_history[head_name].append(GradientStats(norm=norm, num_params=1))

        # Log summary
        if state.global_step > 0 and state.global_step % self.log_every == 0:
            self._log_gradient_stats(state, total_norm)

    def _compute_grad_norm(self, model: torch.nn.Module) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _compute_head_grad_norms(self, model: torch.nn.Module) -> dict[str, float]:
        """Compute gradient norms per task head."""
        head_norms = {}

        # Look for heads attribute (ModernBertMultiTaskModel structure)
        heads = getattr(model, "heads", None)
        if heads is None:
            # Try to get from wrapped model
            if hasattr(model, "module"):
                heads = getattr(model.module, "heads", None)

        if heads is None:
            return head_norms

        for head_name, head in heads.items():
            norm = 0.0
            for p in head.parameters():
                if p.grad is not None:
                    norm += p.grad.data.norm(2).item() ** 2
            head_norms[head_name] = norm**0.5

        return head_norms

    def _log_gradient_stats(self, state: TrainerState, total_norm: float) -> None:
        """Log gradient statistics."""
        logger.info(f"Step {state.global_step} - Total gradient norm: {total_norm:.4f}")

        if self.gradient_history:
            head_summary = ", ".join(
                f"{head}: {stats[-1].norm:.4f}"
                for head, stats in self.gradient_history.items()
                if stats
            )
            if head_summary:
                logger.info(f"  Per-head norms: {head_summary}")


# =============================================================================
# Early Stopping Callback
# =============================================================================


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback based on evaluation metric.

    Stops training when the monitored metric doesn't improve for `patience`
    evaluation rounds.

    Args:
        patience: Number of evaluations to wait for improvement (default: 3)
        metric: Metric to monitor (default: "eval_loss")
        mode: "min" for metrics where lower is better, "max" for higher is better
              (default: auto-detect based on metric name)
        min_delta: Minimum change to qualify as improvement (default: 0.0)
        verbose: Whether to log improvement messages (default: True)

    Example:
        callback = EarlyStoppingCallback(
            patience=3,
            metric="eval_avg_score",
            mode="max",  # Higher avg_score is better
        )
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        patience: int = 3,
        metric: str = "eval_loss",
        mode: str | None = None,
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.verbose = verbose

        # Auto-detect mode based on metric name
        if mode is None:
            # Loss metrics: lower is better
            # Other metrics (f1, accuracy, score): higher is better
            if "loss" in metric.lower():
                self.mode = "min"
            else:
                self.mode = "max"
        else:
            self.mode = mode

        self.best_score: float | None = None
        self.num_bad_evaluations = 0
        self.stopped_epoch: int | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Reset state at start of training."""
        self.best_score = None
        self.num_bad_evaluations = 0
        self.stopped_epoch = None
        logger.info(
            f"EarlyStoppingCallback: Monitoring '{self.metric}' "
            f"(mode={self.mode}, patience={self.patience})"
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Check for improvement after each evaluation."""
        if metrics is None:
            return control

        # Get the monitored metric
        current_score = metrics.get(self.metric)
        if current_score is None:
            logger.warning(
                f"EarlyStoppingCallback: Metric '{self.metric}' not found in "
                f"evaluation results. Available: {list(metrics.keys())}"
            )
            return control

        # Check for improvement
        improved = self._is_improvement(current_score)

        if improved:
            self.best_score = current_score
            self.num_bad_evaluations = 0
            if self.verbose:
                logger.info(f"EarlyStoppingCallback: {self.metric} improved to {current_score:.4f}")
        else:
            self.num_bad_evaluations += 1
            if self.verbose:
                logger.info(
                    f"EarlyStoppingCallback: No improvement in {self.metric}. "
                    f"Best: {self.best_score:.4f}, Current: {current_score:.4f}. "
                    f"Patience: {self.num_bad_evaluations}/{self.patience}"
                )

        # Check if we should stop
        if self.num_bad_evaluations >= self.patience:
            control.should_training_stop = True
            self.stopped_epoch = state.epoch
            logger.info(
                f"EarlyStoppingCallback: Stopping training at step {state.global_step}. "
                f"No improvement for {self.patience} evaluations. "
                f"Best {self.metric}: {self.best_score:.4f}"
            )

        return control

    def _is_improvement(self, current_score: float) -> bool:
        """Check if the current score is an improvement over the best."""
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current_score < (self.best_score - self.min_delta)
        else:  # mode == "max"
            return current_score > (self.best_score + self.min_delta)


# =============================================================================
# Model Checkpoint Callback
# =============================================================================


class ModelCheckpointCallback(TrainerCallback):
    """
    Callback for saving model checkpoints based on evaluation metrics.

    Saves the best model(s) based on monitored metrics, with optional
    checkpoint rotation.

    Args:
        metric: Metric to monitor for saving (default: "eval_loss")
        mode: "min" or "max" (default: auto-detect)
        save_best_only: Only save when metric improves (default: True)
        max_checkpoints: Maximum number of checkpoints to keep (default: 3)
        checkpoint_dir: Directory for checkpoints (default: use args.output_dir)
        save_on_each_task: Save best checkpoint per task (default: False)

    Example:
        callback = ModelCheckpointCallback(
            metric="eval_avg_score",
            mode="max",
            save_best_only=True,
            max_checkpoints=3,
        )
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        metric: str = "eval_loss",
        mode: str | None = None,
        save_best_only: bool = True,
        max_checkpoints: int = 3,
        checkpoint_dir: str | None = None,
        save_on_each_task: bool = False,
    ):
        self.metric = metric
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.save_on_each_task = save_on_each_task

        # Auto-detect mode
        if mode is None:
            self.mode = "min" if "loss" in metric.lower() else "max"
        else:
            self.mode = mode

        self.best_score: float | None = None
        self.best_model_checkpoint: str | None = None
        self.checkpoints: list[tuple[str, float]] = []  # (path, score)

        # Per-task best scores
        self.task_best_scores: dict[str, float] = {}
        self.task_best_checkpoints: dict[str, str] = {}

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Initialize checkpoint directory."""
        base_dir = self.checkpoint_dir if self.checkpoint_dir else args.output_dir
        self.checkpoint_dir = os.path.join(str(base_dir), "best_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(f"ModelCheckpointCallback: Saving best checkpoints to {self.checkpoint_dir}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save checkpoint if metric improved."""
        if metrics is None:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # Check main metric
        current_score = metrics.get(self.metric)
        if current_score is not None:
            if self._is_improvement(current_score, self.best_score):
                self.best_score = current_score
                self._save_checkpoint(trainer, state, current_score, "best")

        # Optionally save per-task best
        if self.save_on_each_task:
            self._save_per_task_checkpoints(trainer, state, metrics)

    def _is_improvement(self, current_score: float, best_score: float | None) -> bool:
        """Check if current score is an improvement."""
        if best_score is None:
            return True

        if self.mode == "min":
            return current_score < best_score
        else:
            return current_score > best_score

    def _save_checkpoint(
        self,
        trainer: Any,
        state: TrainerState,
        score: float,
        name: str,
    ) -> None:
        """Save a checkpoint."""
        if self.checkpoint_dir is None:
            logger.warning("ModelCheckpointCallback: checkpoint_dir not set, skipping save")
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint-{name}-step-{state.global_step}",
        )

        # Save using trainer's save method
        trainer.save_model(checkpoint_path)

        # Save metadata
        metadata = {
            "step": state.global_step,
            "epoch": state.epoch,
            "metric": self.metric,
            "score": score,
        }
        metadata_path = os.path.join(checkpoint_path, "checkpoint_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"ModelCheckpointCallback: Saved checkpoint to {checkpoint_path} "
            f"({self.metric}={score:.4f})"
        )

        # Update checkpoint list
        self.checkpoints.append((checkpoint_path, score))
        self.best_model_checkpoint = checkpoint_path

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by score (keep best ones)
        if self.mode == "min":
            self.checkpoints.sort(key=lambda x: x[1])  # Ascending
        else:
            self.checkpoints.sort(key=lambda x: x[1], reverse=True)  # Descending

        # Remove worst checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            path, score = self.checkpoints.pop()
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"ModelCheckpointCallback: Removed old checkpoint {path}")

    def _save_per_task_checkpoints(
        self,
        trainer: Any,
        state: TrainerState,
        metrics: dict[str, float],
    ) -> None:
        """Save best checkpoint per task based on task-specific metrics."""
        # Find per-task primary metrics
        for key, value in metrics.items():
            # Look for task-specific metrics like "eval_ner_general_f1"
            if key.startswith("eval_") and key != self.metric:
                parts = key[5:].rsplit("_", 1)
                if len(parts) == 2:
                    task_key, metric_name = parts
                    # Check if this is a primary metric (f1, accuracy, etc.)
                    if metric_name in ["f1", "accuracy", "spearman"]:
                        task_metric_key = f"{task_key}_{metric_name}"
                        if self._is_improvement(value, self.task_best_scores.get(task_metric_key)):
                            self.task_best_scores[task_metric_key] = value
                            self._save_checkpoint(
                                trainer,
                                state,
                                value,
                                f"best-{task_key}",
                            )


# =============================================================================
# Dynamic Task Weighting Callback (Advanced)
# =============================================================================


class DynamicTaskWeightingCallback(TrainerCallback):
    """
    Callback for dynamically adjusting task weights during training.

    Supports multiple strategies:
        - "uncertainty": Kendall uncertainty weighting (learns weights)
        - "loss_ratio": Weight inverse to loss progress
        - "gradnorm": Normalize based on gradient magnitude

    Args:
        strategy: Weighting strategy ("uncertainty", "loss_ratio", "gradnorm")
        update_every: Update weights every N steps (default: 500)
        min_weight: Minimum task weight (default: 0.1)
        max_weight: Maximum task weight (default: 10.0)

    Example:
        callback = DynamicTaskWeightingCallback(
            strategy="loss_ratio",
            update_every=500,
        )
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        strategy: str = "loss_ratio",
        update_every: int = 500,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        self.strategy = strategy
        self.update_every = update_every
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Track losses for loss_ratio strategy
        self.initial_losses: dict[str, float] = {}
        self.current_losses: dict[str, list[float]] = defaultdict(list)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Initialize weight tracking."""
        self.initial_losses = {}
        self.current_losses = defaultdict(list)
        logger.info(
            f"DynamicTaskWeightingCallback: Using '{self.strategy}' strategy, "
            f"updating every {self.update_every} steps"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Update task weights periodically."""
        if state.global_step % self.update_every != 0:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        if self.strategy == "loss_ratio":
            self._update_loss_ratio_weights(trainer, state)
        elif self.strategy == "uncertainty":
            # Uncertainty weighting is typically learned, not computed here
            pass
        elif self.strategy == "gradnorm":
            self._update_gradnorm_weights(trainer, state)

    def _update_loss_ratio_weights(self, trainer: Any, state: TrainerState) -> None:
        """Update weights based on loss ratios."""
        if not self.current_losses:
            return

        new_weights = {}
        for task, losses in self.current_losses.items():
            if not losses:
                continue

            current_avg = sum(losses) / len(losses)

            # Initialize initial loss
            if task not in self.initial_losses:
                self.initial_losses[task] = current_avg

            initial = self.initial_losses[task]
            if initial > 0:
                # Weight inversely proportional to improvement
                ratio = current_avg / initial
                weight = max(self.min_weight, min(self.max_weight, ratio))
                new_weights[task] = weight

        if new_weights:
            # Update trainer weights
            trainer.task_weights.update(new_weights)
            logger.info(
                f"DynamicTaskWeightingCallback: Updated weights at step {state.global_step}: "
                f"{new_weights}"
            )

        # Reset current losses
        self.current_losses = defaultdict(list)

    def _update_gradnorm_weights(self, trainer: Any, state: TrainerState) -> None:
        """Update weights based on gradient norms (GradNorm algorithm)."""
        # This is a simplified version - full GradNorm requires more integration
        logger.debug(f"GradNorm weight update at step {state.global_step} (placeholder)")


# =============================================================================
# Epoch Data Distribution Callback
# =============================================================================


class EpochDataDistributionCallback(TrainerCallback):
    """
    Callback for logging data distribution and training stats at each epoch end.

    Logs:
        - Per-task sample counts seen during the epoch
        - Per-task label distributions (class imbalance analysis)
        - Per-task loss averages
        - Task sampling distribution (actual vs expected)
        - Progressive regularization status
        - Memory usage

    This is useful for:
        - Debugging multi-task sampling issues
        - Monitoring class imbalance
        - Tracking training progress
        - Verifying progressive regularization schedule

    Example:
        callback = EpochDataDistributionCallback()
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    # Label name mappings for better readability
    LABEL_NAMES = {
        "sentiment": ["very_neg", "negative", "neutral", "positive", "very_pos"],
        "nli": ["entailment", "neutral", "contradiction"],
        "safety_generic": [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexually_explicit",
            "flirtation",
        ],
        "emotions": [
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
            # Extended emotions (if using 44-class)
            "anticipation",
            "trust",
            "serenity",
            "interest",
            "acceptance",
            "apprehension",
            "distraction",
            "pensiveness",
            "boredom",
            "loathing",
            "rage",
            "vigilance",
            "ecstasy",
            "adoration",
            "terror",
            "amazement",
        ],
    }

    def __init__(self, log_memory: bool = True, log_label_distribution: bool = True):
        self.log_memory = log_memory
        self.log_label_distribution = log_label_distribution
        self.epoch_task_counts: dict[str, int] = defaultdict(int)
        self.epoch_task_losses: dict[str, list[float]] = defaultdict(list)
        self.current_epoch = 0

    def _get_label_distribution(self, dataset, task: str) -> dict[str, Any]:
        """
        Analyze label distribution for a dataset.

        Returns:
            Dict with label counts, percentages, and imbalance ratio
        """
        import numpy as np

        try:
            # Determine task type
            is_token_task = task in ["ner_general", "temporal"]
            is_multilabel = task in ["emotions", "safety_generic"]
            is_embedding = task == "embedding"

            if is_embedding:
                # Embedding task has similarity scores, not discrete labels
                scores = []
                for i, example in enumerate(dataset):
                    if i >= 5000:  # Sample for speed
                        break
                    label = example.get("labels") or example.get("label")
                    if label is not None:
                        if hasattr(label, "item"):
                            label = label.item()
                        scores.append(float(label))

                if scores:
                    scores = np.array(scores)
                    return {
                        "type": "regression",
                        "stats": {
                            "mean": float(np.mean(scores)),
                            "std": float(np.std(scores)),
                            "min": float(np.min(scores)),
                            "max": float(np.max(scores)),
                        },
                        "bins": {
                            "0.0-0.2": int(np.sum(scores < 0.2)),
                            "0.2-0.4": int(np.sum((scores >= 0.2) & (scores < 0.4))),
                            "0.4-0.6": int(np.sum((scores >= 0.4) & (scores < 0.6))),
                            "0.6-0.8": int(np.sum((scores >= 0.6) & (scores < 0.8))),
                            "0.8-1.0": int(np.sum(scores >= 0.8)),
                        },
                    }
                return {"type": "regression", "error": "no scores found"}

            elif is_token_task:
                # Token-level: count BIO tags
                tag_counts: dict[int, int] = defaultdict(int)
                for i, example in enumerate(dataset):
                    if i >= 2000:  # Sample for speed
                        break
                    labels = example.get("labels") or example.get("ner_tags") or example.get("tags")
                    if labels is not None:
                        if hasattr(labels, "tolist"):
                            labels = labels.tolist()
                        for tag in labels:
                            if tag != -100:  # Ignore padding
                                tag_counts[int(tag)] += 1

                if tag_counts:
                    total = sum(tag_counts.values())
                    # Sort by tag index
                    sorted_counts = dict(sorted(tag_counts.items()))
                    return {
                        "type": "token",
                        "counts": sorted_counts,
                        "total_tokens": total,
                        "num_tags": len(sorted_counts),
                    }
                return {"type": "token", "error": "no tags found"}

            elif is_multilabel:
                # Multi-label: count each label occurrence
                # Infer num_labels from first sample's label vector instead of hardcoding
                first_labels = None
                for example in dataset:
                    first_labels = example.get("labels")
                    if first_labels is not None:
                        break
                if first_labels is None:
                    return {"type": "multilabel", "error": "no labels found"}
                if hasattr(first_labels, "numpy"):
                    first_labels = first_labels.numpy()
                first_labels = np.array(first_labels)
                num_labels = (
                    len(first_labels)
                    if first_labels.ndim == 1
                    else (44 if task == "emotions" else 8)
                )

                ml_label_counts = np.zeros(num_labels, dtype=np.int64)
                total_samples = 0

                for i, example in enumerate(dataset):
                    if i >= 5000:  # Sample for speed
                        break
                    labels = example.get("labels")
                    if labels is not None:
                        total_samples += 1
                        if hasattr(labels, "numpy"):
                            labels = labels.numpy()
                        labels = np.array(labels)
                        # Handle multi-hot format
                        if len(labels) == num_labels:
                            ml_label_counts += (labels > 0.5).astype(np.int64)
                        else:
                            # List of indices
                            for idx in labels:
                                if 0 <= int(idx) < num_labels:
                                    ml_label_counts[int(idx)] += 1

                if total_samples > 0:
                    # Get label names if available
                    label_names = self.LABEL_NAMES.get(
                        task, [f"label_{i}" for i in range(num_labels)]
                    )
                    counts_dict = {
                        label_names[i] if i < len(label_names) else f"label_{i}": int(
                            ml_label_counts[i]
                        )
                        for i in range(num_labels)
                    }
                    # Sort by count descending
                    counts_dict = dict(sorted(counts_dict.items(), key=lambda x: -x[1]))

                    pos_counts = ml_label_counts[ml_label_counts > 0]
                    imbalance = (
                        float(pos_counts.max() / pos_counts.min())
                        if len(pos_counts) > 0 and pos_counts.min() > 0
                        else 0.0
                    )

                    return {
                        "type": "multilabel",
                        "counts": counts_dict,
                        "total_samples": total_samples,
                        "avg_labels_per_sample": float(ml_label_counts.sum() / total_samples),
                        "imbalance_ratio": imbalance,
                    }
                return {"type": "multilabel", "error": "no labels found"}

            else:
                # Single-label classification (sentiment, nli)
                cls_label_counts: dict[int, int] = defaultdict(int)
                for i, example in enumerate(dataset):
                    if i >= 10000:  # Sample for speed
                        break
                    label = example.get("labels") or example.get("label")
                    if label is not None:
                        if hasattr(label, "item"):
                            label = label.item()
                        cls_label_counts[int(label)] += 1

                if cls_label_counts:
                    total = sum(cls_label_counts.values())
                    label_names = self.LABEL_NAMES.get(task, [])
                    counts_dict = {}
                    for idx, count in sorted(cls_label_counts.items()):
                        name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
                        counts_dict[name] = count

                    counts = list(cls_label_counts.values())
                    imbalance = max(counts) / min(counts) if min(counts) > 0 else 0.0

                    return {
                        "type": "classification",
                        "counts": counts_dict,
                        "total": total,
                        "num_classes": len(cls_label_counts),
                        "imbalance_ratio": imbalance,
                    }
                return {"type": "classification", "error": "no labels found"}

        except Exception as e:
            return {"type": "error", "error": str(e)}

    def _log_label_distribution(self, task: str, dist: dict[str, Any]) -> None:
        """Log label distribution in a readable format."""
        dist_type = dist.get("type", "unknown")

        if dist_type == "error" or "error" in dist:
            logger.info(f"    (Could not analyze: {dist.get('error', 'unknown error')})")
            return

        if dist_type == "regression":
            stats = dist.get("stats", {})
            logger.info(
                f"    Score distribution: mean={stats.get('mean', 0):.3f}, std={stats.get('std', 0):.3f}"
            )
            bins = dist.get("bins", {})
            if bins:
                bin_str = ", ".join(f"{k}: {v:,}" for k, v in bins.items())
                logger.info(f"    Bins: {bin_str}")

        elif dist_type == "token":
            counts = dist.get("counts", {})
            total = dist.get("total_tokens", 0)
            logger.info(f"    Token tags: {len(counts)} unique, {total:,} total")
            # Show top 5 tags
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:5]
            tag_str = ", ".join(f"tag_{k}: {v:,}" for k, v in sorted_counts)
            logger.info(f"    Top 5: {tag_str}")

        elif dist_type == "multilabel":
            counts = dist.get("counts", {})
            total = dist.get("total_samples", 0)
            avg_labels = dist.get("avg_labels_per_sample", 0)
            imbalance = dist.get("imbalance_ratio", 0)
            logger.info(
                f"    Samples: {total:,}, Avg labels/sample: {avg_labels:.2f}, Imbalance: {imbalance:.1f}x"
            )
            # Show top 5 labels
            top_labels = list(counts.items())[:5]
            label_str = ", ".join(f"{k}: {v:,}" for k, v in top_labels)
            logger.info(f"    Top 5: {label_str}")
            # Show bottom 3 labels
            bottom_labels = list(counts.items())[-3:]
            label_str = ", ".join(f"{k}: {v:,}" for k, v in bottom_labels)
            logger.info(f"    Bottom 3: {label_str}")

        elif dist_type == "classification":
            counts = dist.get("counts", {})
            total = dist.get("total", 0)
            imbalance = dist.get("imbalance_ratio", 0)
            logger.info(
                f"    Classes: {len(counts)}, Total: {total:,}, Imbalance: {imbalance:.2f}x"
            )
            # Show all classes (usually 3-5)
            class_str = ", ".join(f"{k}: {v:,} ({100*v/total:.1f}%)" for k, v in counts.items())
            logger.info(f"    {class_str}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log initial data distribution at training start."""
        trainer = kwargs.get("model") or kwargs.get("trainer")
        if trainer is None:
            return

        # Log initial dataset sizes
        train_datasets = getattr(trainer, "train_datasets", {})
        if train_datasets:
            logger.info("=" * 70)
            logger.info("TRAINING DATA DISTRIBUTION (Initial)")
            logger.info("=" * 70)
            total = 0
            for task, ds in sorted(train_datasets.items()):
                size = len(ds)
                total += size
                logger.info(f"  {task:20s}: {size:>10,} samples")

                # Log label distribution if enabled
                if self.log_label_distribution:
                    dist = self._get_label_distribution(ds, task)
                    self._log_label_distribution(task, dist)

            logger.info("-" * 70)
            logger.info(f"  {'TOTAL':20s}: {total:>10,} samples")
            logger.info("=" * 70)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Track per-task samples during training."""
        trainer = kwargs.get("model") or kwargs.get("trainer")
        if trainer is None:
            return

        # Get current task
        current_task = getattr(trainer, "current_task", None)
        if current_task:
            self.epoch_task_counts[current_task] += 1

            # Track loss if available
            if state.log_history and len(state.log_history) > 0:
                last_log = state.log_history[-1]
                loss = last_log.get("loss")
                if loss is not None:
                    self.epoch_task_losses[current_task].append(float(loss))

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log detailed distribution at end of each epoch."""
        trainer = kwargs.get("model") or kwargs.get("trainer")
        self.current_epoch += 1
        epoch = self.current_epoch

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"EPOCH {epoch} SUMMARY")
        logger.info("=" * 70)

        # === TASK DISTRIBUTION ===
        if self.epoch_task_counts:
            total_steps = sum(self.epoch_task_counts.values())
            logger.info(f"Task Distribution (Total: {total_steps:,} steps):")
            logger.info("-" * 70)
            logger.info(f"  {'Task':<20} {'Steps':>10} {'%':>8} {'Avg Loss':>12}")
            logger.info("-" * 70)

            for task in sorted(self.epoch_task_counts.keys()):
                count = self.epoch_task_counts[task]
                pct = 100.0 * count / total_steps if total_steps > 0 else 0
                losses = self.epoch_task_losses.get(task, [])
                avg_loss = sum(losses) / len(losses) if losses else 0.0
                logger.info(f"  {task:<20} {count:>10,} {pct:>7.1f}% {avg_loss:>12.4f}")

            logger.info("-" * 70)

        # === PROGRESSIVE REGULARIZATION STATUS ===
        if trainer is not None:
            prog_reg = getattr(trainer, "progressive_regularization", False)
            if prog_reg:
                rdrop_on = trainer.rdrop_loss is not None
                mixup_on = trainer.mixup is not None
                adv_on = trainer.adversarial is not None
                logger.info("Progressive Regularization Status:")
                logger.info(f"  R-Drop:      {'✓ ON' if rdrop_on else '✗ OFF'}")
                logger.info(f"  Mixup:       {'✓ ON' if mixup_on else '✗ OFF'}")
                logger.info(f"  Adversarial: {'✓ ON' if adv_on else '✗ OFF'}")

        # === MEMORY USAGE ===
        if self.log_memory and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {max_allocated:.1f}GB peak"
                )
            except Exception:
                pass

        # === LEARNING RATE ===
        if state.log_history:
            for log in reversed(state.log_history):
                if "learning_rate" in log:
                    logger.info(f"Learning Rate: {log['learning_rate']:.2e}")
                    break

        logger.info("=" * 70)
        logger.info("")

        # Reset epoch counters
        self.epoch_task_counts = defaultdict(int)
        self.epoch_task_losses = defaultdict(list)


# =============================================================================
# W&B Enhanced Logging Callback
# =============================================================================


class WandbEnhancedCallback(TrainerCallback):
    """
    Enhanced W&B logging callback for multi-task training.

    Logs rich visualizations to W&B:
        - Per-task loss curves (separate panels)
        - Task distribution pie charts
        - Label distribution tables
        - Progressive regularization timeline
        - Confusion matrices (at eval)
        - Sample predictions
        - GPU memory over time

    Requires: wandb to be installed and logged in.

    Example:
        callback = WandbEnhancedCallback(project="familyos-modernbert")
        trainer = MultiTaskTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        project: str = "familyos-modernbert",
        entity: str | None = None,
        log_model: bool = False,
        log_freq: int = 100,
    ):
        self.project = project
        self.entity = entity
        self.log_model = log_model
        self.log_freq = log_freq
        self._wandb: Any = None  # type: Any to avoid type checker warnings
        self._initialized = False
        self.epoch_task_counts: dict[str, int] = defaultdict(int)
        self.epoch_task_losses: dict[str, list[float]] = defaultdict(list)
        self.current_epoch = 0

    def _init_wandb(self) -> bool:
        """Initialize W&B if not already done."""
        if self._initialized:
            return self._wandb is not None

        try:
            import wandb

            self._wandb = wandb
            self._initialized = True

            # Check if already initialized by HF Trainer
            if wandb.run is None:
                wandb.init(
                    project=self.project,
                    entity=self.entity,
                    reinit=True,
                )
            return True
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self._initialized = True
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self._initialized = True
            return False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log initial config and data distribution to W&B."""
        if not self._init_wandb():
            return

        wandb = self._wandb
        trainer = kwargs.get("model") or kwargs.get("trainer")

        # Log training config
        if wandb.run is not None:
            # Log hyperparameters
            config_dict = {
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation": args.gradient_accumulation_steps,
                "epochs": args.num_train_epochs,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "bf16": args.bf16,
                "fp16": args.fp16,
            }
            wandb.config.update(config_dict, allow_val_change=True)

            # Log dataset sizes as a table
            if trainer is not None:
                train_datasets = getattr(trainer, "train_datasets", {})
                if train_datasets:
                    table_data = []
                    for task, ds in sorted(train_datasets.items()):
                        table_data.append([task, len(ds)])

                    table = wandb.Table(columns=["Task", "Samples"], data=table_data)
                    wandb.log({"data/dataset_sizes": table}, step=0)

                    # Also log as bar chart
                    wandb.log(
                        {
                            "data/samples_per_task": wandb.plot.bar(
                                table, "Task", "Samples", title="Training Samples per Task"
                            )
                        },
                        step=0,
                    )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Track per-task metrics during training."""
        trainer = kwargs.get("model") or kwargs.get("trainer")
        if trainer is None:
            return

        # Get current task
        current_task = getattr(trainer, "current_task", None)
        if current_task:
            self.epoch_task_counts[current_task] += 1

            # Track loss
            if state.log_history and len(state.log_history) > 0:
                last_log = state.log_history[-1]
                loss = last_log.get("loss")
                if loss is not None:
                    self.epoch_task_losses[current_task].append(float(loss))

        # Log per-task losses at intervals
        if state.global_step > 0 and state.global_step % self.log_freq == 0:
            self._log_task_losses(state)

    def _log_task_losses(self, state: TrainerState) -> None:
        """Log per-task loss averages to W&B."""
        if not self._init_wandb() or self._wandb.run is None:
            return

        wandb = self._wandb
        metrics = {}

        for task, losses in self.epoch_task_losses.items():
            if losses:
                # Log recent average (last 100 steps)
                recent_losses = losses[-100:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                metrics[f"task_loss/{task}"] = avg_loss

        if metrics:
            wandb.log(metrics, step=state.global_step)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log epoch summary to W&B."""
        if not self._init_wandb() or self._wandb.run is None:
            return

        wandb = self._wandb
        trainer = kwargs.get("model") or kwargs.get("trainer")
        self.current_epoch += 1
        epoch = self.current_epoch

        # === TASK DISTRIBUTION PIE CHART ===
        if self.epoch_task_counts:
            total_steps = sum(self.epoch_task_counts.values())
            table_data = []
            for task, count in sorted(self.epoch_task_counts.items()):
                pct = 100.0 * count / total_steps if total_steps > 0 else 0
                losses = self.epoch_task_losses.get(task, [])
                avg_loss = sum(losses) / len(losses) if losses else 0.0
                table_data.append([task, count, pct, avg_loss])

            table = wandb.Table(
                columns=["Task", "Steps", "Percentage", "Avg Loss"], data=table_data
            )
            wandb.log(
                {
                    f"epoch_{epoch}/task_distribution": table,
                    f"epoch_{epoch}/task_pie": wandb.plot.bar(
                        table, "Task", "Steps", title=f"Epoch {epoch} Task Distribution"
                    ),
                },
                step=state.global_step,
            )

        # === PROGRESSIVE REGULARIZATION STATUS ===
        if trainer is not None:
            prog_reg = getattr(trainer, "progressive_regularization", False)
            if prog_reg:
                rdrop_on = trainer.rdrop_loss is not None
                mixup_on = trainer.mixup is not None
                adv_on = trainer.adversarial is not None
                wandb.log(
                    {
                        "regularization/rdrop": 1 if rdrop_on else 0,
                        "regularization/mixup": 1 if mixup_on else 0,
                        "regularization/adversarial": 1 if adv_on else 0,
                        "regularization/epoch": epoch,
                    },
                    step=state.global_step,
                )

        # === GPU MEMORY ===
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                wandb.log(
                    {
                        "memory/allocated_gb": allocated,
                        "memory/reserved_gb": reserved,
                        "memory/peak_gb": max_allocated,
                    },
                    step=state.global_step,
                )
            except Exception:
                pass

        # Reset epoch counters
        self.epoch_task_counts = defaultdict(int)
        self.epoch_task_losses = defaultdict(list)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log evaluation metrics with enhanced formatting."""
        if not self._init_wandb() or self._wandb.run is None:
            return

        wandb = self._wandb

        if metrics is None:
            return

        # Group metrics by task
        task_metrics: dict[str, dict[str, float]] = defaultdict(dict)
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            if key.startswith("eval_"):
                # Parse keys like "eval_ner_general_f1"
                parts = key[5:].rsplit("_", 1)
                if len(parts) == 2:
                    task_key, metric_name = parts
                    task_metrics[task_key][metric_name] = value
                else:
                    task_metrics["overall"][key[5:]] = value

        # Create summary table
        if task_metrics:
            table_data = []
            for task, mets in sorted(task_metrics.items()):
                f1 = mets.get("f1", mets.get("score", 0))
                acc = mets.get("accuracy", mets.get("acc", 0))
                loss = mets.get("loss", 0)
                table_data.append([task, f1, acc, loss])

            table = wandb.Table(columns=["Task", "F1", "Accuracy", "Loss"], data=table_data)
            wandb.log(
                {
                    "eval/task_metrics": table,
                    "eval/task_f1_chart": wandb.plot.bar(
                        table, "Task", "F1", title="Evaluation F1 by Task"
                    ),
                },
                step=state.global_step,
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log final summary and optionally save model artifact."""
        if not self._init_wandb() or self._wandb.run is None:
            return

        wandb = self._wandb

        # Log final metrics summary
        if state.log_history:
            final_metrics = {}
            for log in reversed(state.log_history):
                for key, value in log.items():
                    if key.startswith("eval_") and key not in final_metrics:
                        if isinstance(value, (int, float)):
                            final_metrics[key] = value

            if final_metrics:
                wandb.log({"final/" + k: v for k, v in final_metrics.items()})

        # Log model artifact if requested
        if self.log_model:
            try:
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description="Trained multi-task ModernBERT model",
                )
                artifact.add_dir(args.output_dir)
                wandb.log_artifact(artifact)
                logger.info(f"Model artifact logged to W&B: {artifact.name}")
            except Exception as e:
                logger.warning(f"Failed to log model artifact: {e}")


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TaskMetricsCallback",
    "GradientMonitorCallback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "DynamicTaskWeightingCallback",
    "EpochDataDistributionCallback",
    "WandbEnhancedCallback",
]

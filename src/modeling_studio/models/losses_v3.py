"""
Hub-Aware Loss Computation for ModernBERT v3.3 Ultra.

This module implements loss functions that respect hub token routing and support
multi-task training with:
- Per-task loss weighting (fixed or learned via uncertainty)
- Focal loss for imbalanced classes
- Hierarchical loss for emotions (Ekman → GoEmotions)
- Token-level loss masking (excludes hub positions 0-4)
- Label smoothing support

Loss Classes:
    - HubAwareLossComputer: Main loss computer with multi-task support
    - UncertaintyWeightedLoss: Learned per-task weighting via uncertainty
    - LossOutput: Container for loss results

Key Features:
    - Hub token positions (0-4) automatically masked in token-level loss
    - Hierarchical loss combines primary + secondary predictions
    - Focal loss handles class imbalance with γ-weighted cross entropy
    - Multi-task loss aggregates with configurable or learned weights

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional

from .hub_tokens import TOKEN_LEVEL_CAPABILITIES


@dataclass
class LossOutput:
    """
    Container for loss computation results.

    Attributes:
        total_loss: Weighted sum of all task losses
        task_losses: Per-task individual losses
        task_weights: Weights applied to each task
    """

    total_loss: torch.Tensor
    task_losses: dict[str, torch.Tensor]
    task_weights: dict[str, float]


class HubAwareLossComputer(nn.Module):
    """
    Computes losses for all tasks with hub routing awareness.

    This loss computer handles multiple task types:
    - Classification: Standard cross-entropy (emotions, sentiment, safety, intent, etc.)
    - Token-level: Sequence labeling with hub masking (NER, temporal)
    - Regression: MSE loss (similarity tasks)
    - Hierarchical: Primary + secondary predictions (emotions)

    Features:
        - Per-task loss weighting (fixed)
        - Focal loss for imbalanced classes
        - Hierarchical loss for emotions
        - Label smoothing support
        - Hub gradient masking for token-level tasks

    Example:
        >>> task_configs = {
        ...     "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
        ...     "ner_general": {"loss_type": "token_level", "loss_weight": 0.8},
        ...     "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        ... }
        >>> loss_computer = HubAwareLossComputer(task_configs, use_focal_loss=True)
        >>> output = loss_computer.compute_multitask_loss(logits, labels, mask)
        >>> print(f"Total loss: {output.total_loss.item():.4f}")
    """

    def __init__(
        self,
        task_configs: dict[str, dict],
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        """
        Initialize the hub-aware loss computer.

        Args:
            task_configs: Per-task configuration dict with keys:
                - loss_type: "cross_entropy", "token_level", "regression", "hierarchical"
                - loss_weight: Weight for this task (default 1.0)
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_focal_loss: Whether to use focal loss for classification
            focal_gamma: Focal loss gamma parameter (default 2.0)
        """
        super().__init__()

        self.task_configs = task_configs
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # Per-task loss weights (fixed, stored as buffers)
        for task, config in task_configs.items():
            weight = config.get("loss_weight", 1.0)
            self.register_buffer(f"weight_{task}", torch.tensor(weight))

        # Loss functions per task type
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def compute_task_loss(
        self,
        task: str,
        logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for a single task.

        Args:
            task: Task name (e.g., "emotions", "ner_general", "intent")
            logits: Model predictions (tensor or tuple for hierarchical)
            labels: Ground truth (tensor or tuple for hierarchical)
            attention_mask: For token-level tasks (masks padding)

        Returns:
            Scalar loss tensor
        """
        config = self.task_configs.get(task, {})
        loss_type = config.get("loss_type", "cross_entropy")

        # Token-level loss (NER, temporal) with hub masking
        if task in TOKEN_LEVEL_CAPABILITIES:
            assert not isinstance(logits, tuple), "Token-level logits should be a single tensor"
            assert not isinstance(labels, tuple), "Token-level labels should be a single tensor"
            return self._compute_token_level_loss(logits, labels, attention_mask)

        # Regression loss (similarity tasks)
        elif loss_type == "regression" or task in ["stsb", "similarity"]:
            assert not isinstance(logits, tuple), "Regression logits should be a single tensor"
            assert not isinstance(labels, tuple), "Regression labels should be a single tensor"
            return self.mse_loss(logits.squeeze(-1), labels.float())

        # Hierarchical loss for emotions (primary + secondary)
        elif loss_type == "hierarchical" or task == "emotions":
            if isinstance(logits, tuple):
                primary_logits, secondary_logits = logits
                primary_labels, secondary_labels = labels  # Expect tuple
                return self._compute_hierarchical_loss(
                    primary_logits,
                    secondary_logits,
                    primary_labels,
                    secondary_labels,
                )
            else:
                # Fallback to standard cross-entropy if not tuple
                assert not isinstance(labels, tuple), "Labels should match logits structure"
                return self.ce_loss(logits, labels)

        # Focal loss for imbalanced classification
        elif self.use_focal_loss:
            assert not isinstance(logits, tuple), "Focal loss expects single logits tensor"
            assert not isinstance(labels, tuple), "Focal loss expects single labels tensor"
            return self._compute_focal_loss(logits, labels)

        # Standard cross-entropy
        else:
            assert not isinstance(logits, tuple), "Standard CE expects single logits tensor"
            assert not isinstance(labels, tuple), "Standard CE expects single labels tensor"
            return self.ce_loss(logits, labels)

    def _compute_token_level_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute loss for token-level tasks (NER, temporal).

        Masks out hub token positions (0-4) and padding to ensure they
        don't contribute to the loss. Token-level tasks should only learn
        from actual text tokens (positions 5+).

        Args:
            logits: [batch, seq, num_labels]
            labels: [batch, seq]
            attention_mask: [batch, seq] - 1 for valid tokens, 0 for padding

        Returns:
            Scalar loss
        """
        batch_size, seq_len, num_labels = logits.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # Create valid mask: 1 for valid text tokens, 0 for hub/padding
        if attention_mask is not None:
            valid_mask = attention_mask.clone()
            # Mask hub positions (0-4: CLS + 4 hub tokens)
            valid_mask[:, :5] = 0
            valid_mask = valid_mask.view(-1)

            # Set invalid positions to ignore_index (-100)
            labels_flat = labels_flat.masked_fill(valid_mask == 0, -100)

        # Compute cross-entropy with ignore_index
        loss = self.ce_loss(logits_flat, labels_flat)
        return loss

    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        Formula: FL = -α(1-p_t)^γ * log(p_t)

        Where:
            - p_t is the probability of the true class
            - γ (gamma) controls the down-weighting of easy examples
            - Higher γ focuses more on hard examples

        Args:
            logits: [batch, num_labels]
            labels: [batch]

        Returns:
            Scalar focal loss
        """
        # Standard cross-entropy loss (no reduction)
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        # Get probability of true class
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

    def _compute_hierarchical_loss(
        self,
        primary_logits: torch.Tensor,
        secondary_logits: torch.Tensor,
        primary_labels: torch.Tensor,
        secondary_labels: torch.Tensor,
        primary_weight: float = 0.4,
        secondary_weight: float = 0.6,
    ) -> torch.Tensor:
        """
        Hierarchical loss for emotions (Ekman + GoEmotions).

        Combines loss from primary (coarse) and secondary (fine) predictions
        with configurable weighting. The primary loss captures broad emotional
        categories (Ekman), while secondary captures fine-grained emotions.

        Args:
            primary_logits: [batch, primary_labels] - Ekman predictions
            secondary_logits: [batch, secondary_labels] - GoEmotions predictions
            primary_labels: [batch] - Ekman labels
            secondary_labels: [batch] - GoEmotions labels
            primary_weight: Weight for primary loss (default 0.4)
            secondary_weight: Weight for secondary loss (default 0.6)

        Returns:
            Weighted combination of primary and secondary losses
        """
        primary_loss = self.ce_loss(primary_logits, primary_labels)
        secondary_loss = self.ce_loss(secondary_logits, secondary_labels)

        total_loss = primary_weight * primary_loss + secondary_weight * secondary_loss
        return total_loss

    def compute_multitask_loss(
        self,
        task_logits: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        task_labels: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None = None,
        active_tasks: list[str] | None = None,
    ) -> LossOutput:
        """
        Compute weighted sum of all task losses.

        Args:
            task_logits: Dict of task -> logits (or tuple for hierarchical)
            task_labels: Dict of task -> labels (or tuple for hierarchical)
            attention_mask: For token-level tasks
            active_tasks: Only compute loss for these tasks (default: all)

        Returns:
            LossOutput with total loss, per-task losses, and weights
        """
        if active_tasks is None:
            active_tasks = list(task_logits.keys())

        task_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(task_logits.values())).device)

        for task in active_tasks:
            if task not in task_logits or task not in task_labels:
                continue

            logits = task_logits[task]
            labels = task_labels[task]

            # Compute task loss
            loss = self.compute_task_loss(task, logits, labels, attention_mask)
            task_losses[task] = loss

            # Add weighted loss
            weight = getattr(self, f"weight_{task}", torch.tensor(1.0))
            total_loss = total_loss + weight * loss

        # Extract weights as floats
        task_weights = {}
        for t in task_losses:
            weight = getattr(self, f"weight_{t}", torch.tensor(1.0))
            task_weights[t] = weight.item()

        return LossOutput(
            total_loss=total_loss,
            task_losses=task_losses,
            task_weights=task_weights,
        )

    def update_task_weight(self, task: str, weight: float) -> None:
        """
        Update loss weight for a task.

        Args:
            task: Task name
            weight: New weight value
        """
        buffer_name = f"weight_{task}"
        if hasattr(self, buffer_name):
            setattr(self, buffer_name, torch.tensor(weight))
            print(f"✓ Updated weight for '{task}': {weight}")
        else:
            print(f"⚠️ Task '{task}' not found in loss computer")

    def extra_repr(self) -> str:
        return (
            f"tasks={len(self.task_configs)}, "
            f"focal={self.use_focal_loss}, "
            f"smoothing={self.label_smoothing}"
        )


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics" (Kendall et al., CVPR 2018).

    The loss for each task is weighted by a learned uncertainty parameter σ:
        Loss = Σ_i [ (1 / 2σ_i²) * L_i + log(σ_i) ]

    Where:
        - L_i is the loss for task i
        - σ_i is the learned uncertainty (standard deviation) for task i
        - The log(σ_i) term prevents σ from going to infinity

    The network learns to increase σ for harder tasks and decrease it for easier tasks,
    automatically balancing the multi-task learning.

    Example:
        >>> task_names = ["emotions", "ner_general", "intent"]
        >>> uncertainty_loss = UncertaintyWeightedLoss(task_names)
        >>> task_losses = {
        ...     "emotions": torch.tensor(0.5),
        ...     "ner_general": torch.tensor(0.8),
        ...     "intent": torch.tensor(0.3),
        ... }
        >>> total_loss, weights = uncertainty_loss(task_losses)
        >>> print(f"Effective weights: {weights}")
    """

    def __init__(self, task_names: list[str]):
        """
        Initialize uncertainty-weighted loss.

        Args:
            task_names: List of task names to create uncertainty parameters for
        """
        super().__init__()

        self.task_names = task_names

        # Learnable log-variance for each task
        # Use log-variance for numerical stability (ensures σ > 0)
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(1)) for task in task_names}
        )

    def forward(
        self,
        task_losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute uncertainty-weighted total loss.

        Args:
            task_losses: Dict of task -> scalar loss

        Returns:
            Tuple of (total_loss, effective_weights)
                - total_loss: Weighted sum of all task losses
                - effective_weights: Effective weight for each task (1/σ²)
        """
        # Get device from first task loss
        device = next(iter(task_losses.values())).device
        total_loss = torch.zeros(1, device=device)
        effective_weights = {}

        for task, loss in task_losses.items():
            if task not in self.log_vars:
                continue

            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)  # 1/σ²

            # Weighted loss: (1/2σ²) * L + log(σ)
            # Note: log(σ) = 0.5 * log_var
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            total_loss = total_loss + weighted_loss

            effective_weights[task] = precision.item()

        return total_loss.squeeze(), effective_weights

    def get_task_weights(self) -> dict[str, float]:
        """
        Get current effective weights (inverse variance) for all tasks.

        Returns:
            Dict of task -> effective weight (1/σ²)
        """
        return {task: torch.exp(-log_var).item() for task, log_var in self.log_vars.items()}

    def get_task_uncertainties(self) -> dict[str, float]:
        """
        Get current uncertainty (σ) for all tasks.

        Returns:
            Dict of task -> uncertainty (σ)
        """
        return {task: torch.exp(0.5 * log_var).item() for task, log_var in self.log_vars.items()}

    def extra_repr(self) -> str:
        return f"tasks={len(self.task_names)}"


def create_loss_computer(
    task_configs: dict[str, dict],
    use_uncertainty_weighting: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss computer.

    Creates either a standard HubAwareLossComputer with fixed weights or
    a combined module with UncertaintyWeightedLoss for learned weighting.

    Args:
        task_configs: Per-task configuration with keys:
            - loss_type: Type of loss ("cross_entropy", "token_level", etc.)
            - loss_weight: Fixed weight (only used if not using uncertainty)
        use_uncertainty_weighting: Use learned uncertainty weights
        **kwargs: Additional HubAwareLossComputer arguments:
            - label_smoothing: Label smoothing factor
            - use_focal_loss: Whether to use focal loss
            - focal_gamma: Focal loss gamma parameter

    Returns:
        Loss computation module (HubAwareLossComputer or ModuleDict with both)

    Example:
        >>> task_configs = {
        ...     "emotions": {"loss_type": "hierarchical", "loss_weight": 1.0},
        ...     "intent": {"loss_type": "cross_entropy", "loss_weight": 1.2},
        ... }
        >>> # Fixed weights
        >>> loss_fn = create_loss_computer(task_configs)
        >>> # Learned weights
        >>> loss_fn = create_loss_computer(task_configs, use_uncertainty_weighting=True)
    """
    base_loss = HubAwareLossComputer(task_configs, **kwargs)

    if use_uncertainty_weighting:
        uncertainty_loss = UncertaintyWeightedLoss(list(task_configs.keys()))
        # Return combined module
        return nn.ModuleDict({"base": base_loss, "uncertainty": uncertainty_loss})
    else:
        return base_loss

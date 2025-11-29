"""
Task Weighting Strategies

This module provides task weighting strategies for multi-task learning:
    - Static weighting (manual weights)
    - Uncertainty weighting (learned weights)
    - GradNorm (gradient-based balancing)
    - Dynamic temperature scaling

Uncertainty Weighting:
    Learns task weights automatically based on homoscedastic uncertainty.
    Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)

Expected Gains:
    - +2-4 pt improvement from automatic task balancing
    - Reduces need for manual weight tuning

Usage:
    from modeling_studio.trainers.task_weighting import UncertaintyWeighting

    weighter = UncertaintyWeighting(num_tasks=12)

    for batch in dataloader:
        losses = [head(batch) for head in heads]
        total_loss = weighter(losses)
        total_loss.backward()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class UncertaintyWeighting(nn.Module):
    """
    Learns task weights automatically based on homoscedastic uncertainty.

    For each task i, learns a log variance parameter log(σ_i²).
    The weighted loss becomes:
        L_i_weighted = (1 / (2 * σ_i²)) * L_i + log(σ_i)

    This allows the model to down-weight noisy/hard tasks and up-weight
    easier tasks during training.

    Args:
        num_tasks: Number of tasks to weight.
        init_value: Initial log variance value. Default: 0.0 (σ=1)

    Reference:
        Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to
        Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

    Example:
        >>> weighter = UncertaintyWeighting(num_tasks=12)
        >>> losses = [loss_ner, loss_sentiment, loss_safety, ...]
        >>> total_loss = weighter(losses)
        >>> total_loss.backward()
    """

    def __init__(self, num_tasks: int, init_value: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks

        # Learnable log variances (one per task)
        # log(σ²) parameterization is more stable than σ directly
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_value))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of losses.

        Args:
            losses: List of task losses (scalars or 0-dim tensors).

        Returns:
            Weighted total loss.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")

        total_loss = torch.tensor(0.0, device=losses[0].device)

        for i, loss in enumerate(losses):
            if loss is None or (isinstance(loss, torch.Tensor) and loss.numel() == 0):
                continue

            # precision = 1 / σ² = exp(-log(σ²))
            precision = torch.exp(-self.log_vars[i])

            # Weighted loss + regularization
            # L_weighted = (1 / (2σ²)) * L + log(σ) = 0.5 * precision * L + 0.5 * log_var
            total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]

        return total_loss

    def get_weights(self) -> dict[int, float]:
        """Get current task weights (1 / σ²)."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars).cpu().tolist()
        return {i: w for i, w in enumerate(weights)}

    def get_log_vars(self) -> dict[int, float]:
        """Get current log variance values."""
        with torch.no_grad():
            log_vars = self.log_vars.cpu().tolist()
        return {i: lv for i, lv in enumerate(log_vars)}


class StaticWeighting(nn.Module):
    """
    Static task weighting with fixed weights.

    Args:
        weights: Dictionary mapping task index to weight.
        num_tasks: Total number of tasks.

    Example:
        >>> weighter = StaticWeighting(
        ...     weights={0: 1.0, 1: 2.0, 2: 15.0},  # safety has 15x weight
        ...     num_tasks=12
        ... )
    """

    def __init__(self, weights: dict[int, float], num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks

        # Create weight tensor
        weight_tensor = torch.ones(num_tasks)
        for idx, weight in weights.items():
            weight_tensor[idx] = weight

        self.register_buffer("weights", weight_tensor)

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = torch.tensor(0.0, device=losses[0].device)

        for i, loss in enumerate(losses):
            if loss is None:
                continue
            total_loss = total_loss + self.weights[i] * loss

        return total_loss


class DynamicTemperatureWeighting(nn.Module):
    """
    Dynamic temperature scaling for task losses.

    Uses temperature to control the "sharpness" of task importance.
    Higher temperature = more uniform weighting.
    Lower temperature = more emphasis on high-weight tasks.

    Args:
        num_tasks: Number of tasks.
        init_weights: Initial task weights.
        temperature: Temperature for scaling. Default: 1.0

    Example:
        >>> weighter = DynamicTemperatureWeighting(
        ...     num_tasks=12,
        ...     init_weights=[1.0, 1.0, 15.0, ...],
        ...     temperature=2.0
        ... )
    """

    def __init__(
        self,
        num_tasks: int,
        init_weights: list[float] | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        if init_weights is None:
            init_weights = [1.0] * num_tasks

        self.log_weights = nn.Parameter(torch.log(torch.tensor(init_weights, dtype=torch.float)))
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute temperature-scaled weighted sum."""
        # Softmax over log weights with temperature
        weights = F.softmax(self.log_weights / self.temperature, dim=0)

        total_loss = torch.tensor(0.0, device=losses[0].device)

        for i, loss in enumerate(losses):
            if loss is None:
                continue
            total_loss = total_loss + weights[i] * loss * self.num_tasks

        return total_loss


class GradNormWeighting(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Balances task gradients to ensure all tasks learn at similar rates.

    Reference:
        Chen, Badrinarayanan, Lee, Rabinovich.
        "GradNorm: Gradient Normalization for Adaptive Loss Balancing
        in Deep Multitask Networks" (ICML 2018)

    Note: This requires special handling during training - the weights
    are updated based on gradient norms, not through backprop.

    Args:
        num_tasks: Number of tasks.
        alpha: Asymmetry parameter. Higher = more aggressive balancing.
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Task weights (learnable, but updated via GradNorm algorithm)
        self.weights = nn.Parameter(torch.ones(num_tasks))

        # Track initial losses for relative loss computation
        self.register_buffer("initial_losses", torch.zeros(num_tasks))
        self.initialized = False

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses."""
        # Initialize tracking on first forward
        if not self.initialized:
            with torch.no_grad():
                for i, loss in enumerate(losses):
                    if loss is not None:
                        self.initial_losses[i] = loss.item()
            self.initialized = True

        total_loss = torch.tensor(0.0, device=losses[0].device)

        for i, loss in enumerate(losses):
            if loss is None:
                continue
            total_loss = total_loss + self.weights[i] * loss

        return total_loss

    def update_weights(
        self,
        losses: list[torch.Tensor],
        shared_layer: nn.Module,
        lr: float = 0.01,
    ) -> None:
        """
        Update weights based on gradient norms.

        Should be called after backward() but before optimizer.step().

        Args:
            losses: Current task losses.
            shared_layer: The last shared layer (for gradient computation).
            lr: Learning rate for weight updates.
        """
        # Compute gradient norms for each task
        grad_norms = []
        for i, loss in enumerate(losses):
            if loss is None:
                grad_norms.append(None)
                continue

            # Get gradient of loss w.r.t. shared layer
            grad = torch.autograd.grad(
                self.weights[i] * loss,
                shared_layer.parameters(),
                retain_graph=True,
                create_graph=True,
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grad]))
            grad_norms.append(grad_norm)

        # Compute average gradient norm
        valid_norms = [g for g in grad_norms if g is not None]
        avg_norm = torch.mean(torch.stack(valid_norms))

        # Compute relative losses
        with torch.no_grad():
            relative_losses = []
            for i, loss in enumerate(losses):
                if loss is not None and self.initial_losses[i] > 0:
                    relative_losses.append(loss.item() / self.initial_losses[i])
                else:
                    relative_losses.append(1.0)

            avg_relative_loss = sum(relative_losses) / len(relative_losses)

            # Compute target gradient norms
            for i, grad_norm in enumerate(grad_norms):
                if grad_norm is None:
                    continue

                # Target norm based on relative inverse training rate
                r_i = relative_losses[i] / avg_relative_loss
                target_norm = avg_norm * (r_i**self.alpha)

                # Update weight to move gradient norm toward target
                grad_norm_val = grad_norm.item()
                if grad_norm_val > 0:
                    # Increase weight if gradient norm is too low
                    self.weights.data[i] *= (target_norm / grad_norm_val) ** lr


# Export public API
__all__ = [
    "UncertaintyWeighting",
    "StaticWeighting",
    "DynamicTemperatureWeighting",
    "GradNormWeighting",
]

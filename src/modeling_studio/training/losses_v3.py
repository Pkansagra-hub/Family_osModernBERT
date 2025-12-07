"""Hub-Weighted Loss Scaling for v3 Multi-Task Training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as functional

from modeling_studio.data.loaders_v3 import HubRouting

logger = logging.getLogger(__name__)

HUB_TOKEN_POSITIONS_DEFAULT: dict[str, int] = {"EMO": 1, "REL": 2, "MEM": 3, "TASK": 4}


@dataclass
class HubLossConfig:
    """Configuration for hub-weighted loss scaling."""

    active_weight: float = 1.0
    inactive_weight: float = 0.3
    safety_multiplier: float = 2.0
    always_train_safety: bool = True
    task_base_weights: dict[str, float] = field(
        default_factory=lambda: {
            "emotions": 1.0,
            "sentiment": 1.0,
            "safety_familyos": 1.0,
            "intent": 0.8,
            "ingress": 0.8,
            "ner_family": 1.0,
            "temporal": 1.0,
            "relations": 1.2,
        }
    )
    hub_to_tasks: dict[str, list[str]] = field(
        default_factory=lambda: {
            "EMO": ["emotions", "sentiment", "safety_familyos"],
            "REL": ["relations"],
            "MEM": ["temporal", "ner_family"],
            "TASK": ["intent", "ingress"],
        }
    )


class HubLossWeightCalculator:
    """Calculates per-sample, per-task loss weights based on hub routing."""

    def __init__(self, config: HubLossConfig | None = None) -> None:
        self.config = config or HubLossConfig()
        self.task_to_hub: dict[str, str] = {}
        for hub, tasks in self.config.hub_to_tasks.items():
            for task_name in tasks:
                self.task_to_hub[task_name] = hub

    def compute_weight(self, task_name: str, hub_routing: HubRouting, has_label: bool) -> float:
        """Compute loss weight for a single task/sample pair."""

        if not has_label:
            return 0.0

        base_weight = self.config.task_base_weights.get(task_name, 1.0)
        hub = self.task_to_hub.get(task_name)
        hub_active = False
        if hub == "EMO":
            hub_active = hub_routing.emo
        elif hub == "REL":
            hub_active = hub_routing.rel
        elif hub == "MEM":
            hub_active = hub_routing.mem
        elif hub == "TASK":
            hub_active = hub_routing.task

        weight = base_weight * (self.config.active_weight if hub_active else self.config.inactive_weight)

        if task_name == "safety_familyos":
            if self.config.always_train_safety:
                weight = max(weight, base_weight * self.config.inactive_weight)
            weight = weight * self.config.safety_multiplier

        return weight

    def compute_batch_weights(
        self, task_name: str, hub_routings: list[HubRouting], has_labels: list[bool]
    ) -> torch.Tensor:
        """Compute weights for a batch of samples for a given task."""

        weights = [
            self.compute_weight(task_name, routing, has_label)
            for routing, has_label in zip(hub_routings, has_labels, strict=True)
        ]
        return torch.tensor(weights, dtype=torch.float32)


class HubWeightedMultiTaskLoss(nn.Module):
    """Hub-aware multi-task loss with per-sample task weighting."""

    def __init__(self, config: HubLossConfig | None = None, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.config = config or HubLossConfig()
        self.weight_calculator = HubLossWeightCalculator(self.config)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        task_logits: dict[str, torch.Tensor],
        task_labels: dict[str, torch.Tensor],
        hub_routings: list[HubRouting],
        task_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute hub-weighted loss across tasks.

        Args:
            task_logits: Mapping of task name to logits tensor.
            task_labels: Mapping of task name to labels tensor.
            hub_routings: Hub routing for each sample.
            task_masks: Optional per-task masks (1.0 to keep, 0.0 to drop).

        Returns:
            Tuple of (total_loss, per_task_losses).
        """

        device = self._get_device(task_logits)
        total_loss = torch.tensor(0.0, device=device)
        task_losses: dict[str, torch.Tensor] = {}

        for task_name, logits in task_logits.items():
            if task_name not in task_labels:
                continue

            labels = task_labels[task_name].to(device)
            logits = logits.to(device)

            loss = self._compute_task_loss(task_name, logits, labels)
            has_labels = self._get_has_labels(labels, task_name)
            weights = self.weight_calculator.compute_batch_weights(
                task_name.replace("_labels", ""), hub_routings, has_labels
            ).to(device)

            if task_masks and task_name in task_masks:
                weights = weights * task_masks[task_name].to(device)

            weighted_loss = (loss * weights).sum()
            weight_sum = weights.sum()
            task_loss = weighted_loss / weight_sum if weight_sum > 0 else torch.tensor(0.0, device=device)

            task_losses[task_name] = task_loss
            total_loss = total_loss + task_loss

        return total_loss, task_losses

    def _compute_task_loss(self, task_name: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sample loss for a given task."""

        if task_name == "emotions":
            loss = self.bce_loss(logits, labels.float())
            return loss.mean(dim=-1)

        if task_name in {"ner_family", "temporal"}:
            return self._compute_token_loss(logits, labels)

        return self.ce_loss(logits, labels)

    def _compute_token_loss(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Compute token classification loss with ignore_index handling."""

        batch_size, seq_len, num_labels = logits.shape
        loss = functional.cross_entropy(
            logits.view(-1, num_labels),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="none",
        )
        loss = loss.view(batch_size, seq_len)

        valid_mask = (labels != ignore_index).float()
        token_counts = valid_mask.sum(dim=1).clamp(min=1)
        return (loss * valid_mask).sum(dim=1) / token_counts

    def _get_has_labels(self, labels: torch.Tensor, task_name: str) -> list[bool]:
        """Determine which samples have labels for a task."""

        if task_name == "emotions":
            return (labels.sum(dim=-1) > 0).tolist()

        if task_name in {"ner_family", "temporal"}:
            return (labels != -100).any(dim=-1).tolist()

        return (labels != -100).tolist()

    def _get_device(self, task_logits: dict[str, torch.Tensor]) -> torch.device:
        """Infer device from provided logits."""

        for tensor in task_logits.values():
            return tensor.device
        return torch.device("cpu")


class HubGradientMaskedLoss(nn.Module):
    """Apply hub token gradient masking before delegating to a base loss."""

    def __init__(self, base_loss: nn.Module, hub_token_positions: dict[str, int] | None = None) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.hub_token_positions = hub_token_positions or HUB_TOKEN_POSITIONS_DEFAULT

    def get_hub_gradient_mask(self, hub_routings: list[HubRouting], seq_len: int, device: torch.device) -> torch.Tensor:
        """Build gradient mask for hub tokens (1 = keep grad, 0 = mask)."""

        mask = torch.ones(len(hub_routings), seq_len, device=device)
        for index, routing in enumerate(hub_routings):
            if not routing.emo:
                mask[index, self.hub_token_positions["EMO"]] = 0.0
            if not routing.rel:
                mask[index, self.hub_token_positions["REL"]] = 0.0
            if not routing.mem:
                mask[index, self.hub_token_positions["MEM"]] = 0.0
            if not routing.task:
                mask[index, self.hub_token_positions["TASK"]] = 0.0
        return mask

    def forward(self, hidden_states: torch.Tensor, hub_routings: list[HubRouting], **loss_kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass applying gradient mask hook before base loss."""

        mask = self.get_hub_gradient_mask(hub_routings, hidden_states.shape[1], hidden_states.device)
        if hidden_states.requires_grad:
            hidden_states.register_hook(lambda grad: grad * mask.unsqueeze(-1))

        loss_kwargs_with_states = {**loss_kwargs}
        loss_kwargs_with_states.setdefault("hidden_states", hidden_states)
        return self.base_loss(**loss_kwargs_with_states)


def aggregate_task_losses(task_losses: dict[str, torch.Tensor], task_weights: dict[str, float] | None = None) -> torch.Tensor:
    """Aggregate per-task losses using optional weights."""

    weights = task_weights or {}
    total = torch.tensor(0.0, device=_get_first_device(task_losses))
    for task_name, loss in task_losses.items():
        total = total + loss * weights.get(task_name, 1.0)
    return total


def log_task_losses(task_losses: dict[str, torch.Tensor], prefix: str = "train") -> dict[str, float]:
    """Convert task losses to scalars for logging."""

    return {f"{prefix}/loss_{name}": loss.item() for name, loss in task_losses.items()}


def _get_first_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    """Helper to infer device from dict of tensors."""

    for tensor in tensors.values():
        return tensor.device
    return torch.device("cpu")

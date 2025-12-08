"""
Learning Rate Schedulers for ModernBERT v3.

This module implements learning rate schedulers with warmup and decay for smooth
training. Warmup prevents gradient shock at step 1, while decay settles weights
gently at the end of training.

Scheduler Types:
    - WarmupCosineScheduler: Linear warmup + cosine decay (recommended)
    - WarmupLinearScheduler: Linear warmup + linear decay
    - WarmupConstantScheduler: Linear warmup + constant LR
    - PhaseAwareScheduler: Handles phase transitions in v3 training

LR Profile (WarmupCosineScheduler with 2500 total, 500 warmup):
    Step 0:    lr = 0
    Step 250:  lr = base_lr * 0.5
    Step 500:  lr = base_lr (peak)
    Step 1500: lr = ~base_lr * 0.5
    Step 2500: lr = min_lr
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ============================================================================
# WarmupCosineScheduler
# ============================================================================


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    LR Profile:
        Warmup Phase (steps 0 to warmup_steps):
            lr = base_lr * (step / warmup_steps)

        Cosine Decay Phase (steps warmup_steps to total_steps):
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))

        where progress = (step - warmup_steps) / (total_steps - warmup_steps)

    Example (2500 total, 500 warmup, min_lr_ratio=0.01):
        Step 0:    lr = 0
        Step 250:  lr = base_lr * 0.5
        Step 500:  lr = base_lr (peak)
        Step 1500: lr = ~base_lr * 0.5
        Step 2500: lr = base_lr * 0.01 (min_lr)

    Attributes:
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base LR (default 1%)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        """
        Initialize WarmupCosineScheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of base LR (default 1%)
            last_epoch: The index of last epoch (for resuming)
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
            )
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif step >= self.total_steps:
            # After training complete
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            # Interpolate between base_lr and min_lr
            return [
                base_lr * self.min_lr_ratio
                + (base_lr - base_lr * self.min_lr_ratio) * cosine_factor
                for base_lr in self.base_lrs
            ]

    def get_lr_at_step(self, step: int) -> list[float]:
        """
        Get learning rate at a specific step without modifying state.

        Args:
            step: Step number to compute LR for

        Returns:
            List of learning rates for each param group
        """
        original_step = self.last_epoch
        self._step_count = step
        self.last_epoch = step
        lrs = self.get_lr()
        self.last_epoch = original_step
        return lrs


# ============================================================================
# WarmupLinearScheduler
# ============================================================================


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.

    Simpler than cosine but can be effective for shorter training runs.

    LR Profile:
        Warmup Phase (steps 0 to warmup_steps):
            lr = base_lr * (step / warmup_steps)

        Linear Decay Phase (steps warmup_steps to total_steps):
            lr = base_lr * (1 - progress) where progress goes from 0 to 1

    Attributes:
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base LR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize WarmupLinearScheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of base LR
            last_epoch: The index of last epoch (for resuming)
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = step - self.warmup_steps
            decay_factor = 1.0 - (steps_since_warmup / max(1, decay_steps))
            decay_factor = max(self.min_lr_ratio, decay_factor)

            return [base_lr * decay_factor for base_lr in self.base_lrs]

    def get_lr_at_step(self, step: int) -> list[float]:
        """Get learning rate at a specific step without modifying state."""
        original_step = self.last_epoch
        self.last_epoch = step
        lrs = self.get_lr()
        self.last_epoch = original_step
        return lrs


# ============================================================================
# WarmupConstantScheduler
# ============================================================================


class WarmupConstantScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup then constant LR.

    Useful for short fine-tuning runs where decay isn't beneficial.

    LR Profile:
        Warmup Phase (steps 0 to warmup_steps):
            lr = base_lr * (step / warmup_steps)

        Constant Phase (steps >= warmup_steps):
            lr = base_lr

    Attributes:
        warmup_steps: Number of warmup steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Initialize WarmupConstantScheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            last_epoch: The index of last epoch (for resuming)
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return list(self.base_lrs)

    def get_lr_at_step(self, step: int) -> list[float]:
        """Get learning rate at a specific step without modifying state."""
        original_step = self.last_epoch
        self.last_epoch = step
        lrs = self.get_lr()
        self.last_epoch = original_step
        return lrs


# ============================================================================
# PhaseAwareScheduler
# ============================================================================


class PhaseAwareScheduler:
    """
    Scheduler that handles phase transitions in v3 training.

    Manages separate schedulers for each phase and handles transitions
    between phases. Each phase can have its own warmup, decay, and LR settings.

    Example Usage:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        >>> configs = {
        ...     "phase_0.5": {"warmup_steps": 500, "total_steps": 2500},
        ...     "phase_1": {"warmup_steps": 1000, "total_steps": 5000},
        ... }
        >>> scheduler = PhaseAwareScheduler(optimizer, configs)
        >>> scheduler.set_phase("phase_0.5")
        >>> for step in range(2500):
        ...     train_step()
        ...     scheduler.step()

    Attributes:
        optimizer: Wrapped optimizer
        phase_configs: Dictionary of phase configurations
        current_phase: Currently active phase name
        current_scheduler: Currently active scheduler
        phase_step: Steps taken in current phase
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        phase_configs: dict[str, dict[str, Any]],
    ):
        """
        Initialize PhaseAwareScheduler.

        Args:
            optimizer: Wrapped optimizer
            phase_configs: Dict of phase_name -> config dict
                Each config: {warmup_steps, total_steps, scheduler_type, min_lr_ratio}
        """
        self.optimizer = optimizer
        self.phase_configs = phase_configs
        self.current_phase: str | None = None
        self.current_scheduler: _LRScheduler | None = None
        self.phase_step = 0
        self.total_steps_across_phases = 0

    def set_phase(self, phase: str) -> None:
        """
        Switch to a new training phase.

        Args:
            phase: Phase name (e.g., "phase_0.5", "phase_1")

        Raises:
            ValueError: If phase is not in phase_configs
        """
        if phase not in self.phase_configs:
            available = ", ".join(self.phase_configs.keys())
            raise ValueError(f"Unknown phase: {phase}. Available: {available}")

        config = self.phase_configs[phase]
        self.current_phase = phase
        self.phase_step = 0

        # Create scheduler for this phase
        scheduler_type = config.get("scheduler_type", "cosine")
        warmup_steps = config.get("warmup_steps", 500)
        total_steps = config.get("total_steps", 2500)
        min_lr_ratio = config.get("min_lr_ratio", 0.01)

        if scheduler_type == "cosine":
            self.current_scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif scheduler_type == "linear":
            self.current_scheduler = WarmupLinearScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif scheduler_type == "constant":
            self.current_scheduler = WarmupConstantScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        logger.info(f"Switched to {phase} with {scheduler_type} scheduler")
        logger.info(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")

    def step(self) -> None:
        """Advance scheduler by one step."""
        if self.current_scheduler is not None:
            self.current_scheduler.step()
            self.phase_step += 1
            self.total_steps_across_phases += 1

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        if self.current_scheduler is not None:
            return self.current_scheduler.get_last_lr()
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_phase_progress(self) -> float:
        """Get progress through current phase (0.0 to 1.0)."""
        if self.current_phase is None:
            return 0.0
        config = self.phase_configs.get(self.current_phase, {})
        total_steps = config.get("total_steps", 1)
        return min(1.0, self.phase_step / max(1, total_steps))

    def is_warmup_complete(self) -> bool:
        """Check if warmup phase is complete."""
        if self.current_phase is None:
            return True
        config = self.phase_configs.get(self.current_phase, {})
        warmup_steps = config.get("warmup_steps", 0)
        return self.phase_step >= warmup_steps

    def get_state_dict(self) -> dict[str, Any]:
        """Get scheduler state for checkpointing."""
        state = {
            "current_phase": self.current_phase,
            "phase_step": self.phase_step,
            "total_steps_across_phases": self.total_steps_across_phases,
        }
        if self.current_scheduler is not None:
            state["scheduler_state"] = self.current_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        saved_phase = state_dict.get("current_phase")
        saved_phase_step = state_dict.get("phase_step", 0)
        self.total_steps_across_phases = state_dict.get("total_steps_across_phases", 0)

        if saved_phase is not None:
            self.set_phase(saved_phase)  # This resets phase_step to 0
            self.phase_step = saved_phase_step  # Restore the saved phase_step
            if "scheduler_state" in state_dict and self.current_scheduler is not None:
                self.current_scheduler.load_state_dict(state_dict["scheduler_state"])
        else:
            self.current_phase = None
            self.phase_step = saved_phase_step


# ============================================================================
# Factory Function
# ============================================================================


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    total_steps: int = 2500,
    min_lr_ratio: float = 0.01,
) -> _LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer
        scheduler_type: "cosine", "linear", or "constant"
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak

    Returns:
        Configured scheduler

    Raises:
        ValueError: If scheduler_type is unknown
    """
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "constant":
        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            "Valid options: 'cosine', 'linear', 'constant'"
        )

    logger.info(f"Created {scheduler_type} scheduler:")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Total steps: {total_steps}")
    if scheduler_type != "constant":
        logger.info(f"  Min LR ratio: {min_lr_ratio}")

    return scheduler


def create_phase_aware_scheduler(
    optimizer: torch.optim.Optimizer,
    phase_configs: dict[str, dict[str, Any]] | None = None,
) -> PhaseAwareScheduler:
    """
    Create a phase-aware scheduler with optional custom configs.

    Args:
        optimizer: Wrapped optimizer
        phase_configs: Optional custom phase configurations

    Returns:
        Configured PhaseAwareScheduler
    """
    if phase_configs is None:
        phase_configs = copy.deepcopy(DEFAULT_PHASE_SCHEDULER_CONFIGS)

    return PhaseAwareScheduler(optimizer, phase_configs)


# ============================================================================
# Utility Functions
# ============================================================================


def compute_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_warmup: int = 100,
    max_warmup: int = 2000,
) -> int:
    """
    Compute warmup steps based on total steps and ratio.

    Args:
        total_steps: Total training steps
        warmup_ratio: Ratio of total steps for warmup
        min_warmup: Minimum warmup steps
        max_warmup: Maximum warmup steps

    Returns:
        Number of warmup steps
    """
    warmup = int(total_steps * warmup_ratio)
    return max(min_warmup, min(max_warmup, warmup))


def get_lr_at_step(
    scheduler: _LRScheduler,
    step: int,
) -> list[float]:
    """
    Get learning rate at a specific step without modifying scheduler state.

    Args:
        scheduler: LR scheduler
        step: Step number

    Returns:
        List of learning rates for each param group
    """
    if hasattr(scheduler, "get_lr_at_step"):
        return scheduler.get_lr_at_step(step)

    # Fallback for schedulers without get_lr_at_step
    original_step = scheduler.last_epoch
    scheduler.last_epoch = step
    lrs = scheduler.get_lr()
    scheduler.last_epoch = original_step
    return lrs


def print_scheduler_profile(
    scheduler: _LRScheduler,
    total_steps: int,
    num_points: int = 10,
) -> None:
    """
    Print scheduler LR profile at key points.

    Args:
        scheduler: LR scheduler
        total_steps: Total training steps
        num_points: Number of points to sample
    """
    print("\nScheduler LR Profile:")
    print("-" * 40)

    step_points = [int(total_steps * i / (num_points - 1)) for i in range(num_points)]

    for step in step_points:
        lrs = get_lr_at_step(scheduler, step)
        lr_str = ", ".join(f"{lr:.2e}" for lr in lrs[:3])
        if len(lrs) > 3:
            lr_str += f", ... ({len(lrs)} groups)"
        print(f"  Step {step:5d}: {lr_str}")

    print("-" * 40)


# ============================================================================
# Default Phase Configurations
# ============================================================================

DEFAULT_PHASE_SCHEDULER_CONFIGS: dict[str, dict[str, Any]] = {
    "phase_0.5": {
        "scheduler_type": "cosine",
        "warmup_steps": 500,
        "total_steps": 2500,
        "min_lr_ratio": 0.01,
    },
    "phase_1": {
        "scheduler_type": "cosine",
        "warmup_steps": 1000,
        "total_steps": 5000,
        "min_lr_ratio": 0.01,
    },
    "phase_2": {
        "scheduler_type": "cosine",
        "warmup_steps": 200,
        "total_steps": 1000,
        "min_lr_ratio": 0.1,
    },
}


# List of valid scheduler types
VALID_SCHEDULER_TYPES = ["cosine", "linear", "constant"]

"""
Task Sampler for Multi-Task Learning

This module handles the sampling strategy for selecting which task
to train on at each step in multi-task learning.

Sampling Strategies:
    - ProportionalSampler: Sample based on dataset size * task weight
    - TemperatureSampler: Softmax sampling with temperature parameter
    - UniformSampler: Equal probability for all tasks
    - SequentialSampler: Round-robin through tasks
    - CurriculumSampler: Gradually shift focus (easy -> hard tasks)

The sampler ensures balanced training across tasks while respecting
user-defined task weights and preventing catastrophic forgetting.

Usage:
    from modeling_studio.trainers.task_sampler import ProportionalSampler

    sampler = ProportionalSampler(
        task_sizes={"ner": 10000, "sentiment": 5000},
        task_weights={"ner": 1.0, "sentiment": 2.0},
        seed=42
    )

    for step in range(num_steps):
        task = sampler.sample()
        batch = dataloaders[task].next()
        # train on batch
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Base Task Sampler
# =============================================================================


@dataclass
class TaskSampler(ABC):
    """
    Abstract base class for task samplers.

    All task samplers inherit from this class and implement the sample() method
    to determine which task to train on at each step.

    Args:
        task_names: List of task names to sample from.
        task_weights: Optional dict mapping task names to weights.
            If not provided, all tasks have weight 1.0.
        seed: Random seed for reproducibility.

    Attributes:
        task_names: List of available task names.
        task_weights: Dict mapping task name to weight.
        probabilities: Current sampling probabilities per task.
        rng: Random number generator.
        step_count: Number of samples drawn.
    """

    task_names: list[str]
    task_weights: dict[str, float] | None = None
    seed: int | None = None

    # Computed/internal fields
    _probabilities: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _rng: random.Random = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize the sampler after dataclass init."""
        if not self.task_names:
            raise ValueError("At least one task is required")

        # Initialize task weights
        if self.task_weights is None:
            self.task_weights = dict.fromkeys(self.task_names, 1.0)
        else:
            # Ensure all tasks have weights
            for task in self.task_names:
                if task not in self.task_weights:
                    self.task_weights[task] = 1.0

        # Initialize random generator
        self._rng = random.Random(self.seed)

        # Compute initial probabilities
        self._compute_probabilities()

        logger.debug(
            f"Initialized {self.__class__.__name__} with {len(self.task_names)} tasks, "
            f"probabilities: {self._probabilities}"
        )

    @abstractmethod
    def _compute_probabilities(self) -> None:
        """Compute sampling probabilities. Must be implemented by subclasses."""
        pass

    @property
    def probabilities(self) -> dict[str, float]:
        """Current sampling probabilities per task."""
        return dict(self._probabilities)

    @property
    def step_count(self) -> int:
        """Number of samples drawn since initialization or last reset."""
        return self._step_count

    def sample(self) -> str:
        """
        Sample a task to train on.

        Returns:
            Task name selected for training.
        """
        self._step_count += 1

        # Use random.choices with probabilities
        tasks = list(self._probabilities.keys())
        probs = [self._probabilities[t] for t in tasks]

        return self._rng.choices(tasks, weights=probs, k=1)[0]

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the sampler state for a new epoch.

        Args:
            seed: Optional new random seed. If None, uses original seed.
        """
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._step_count = 0
        self._compute_probabilities()
        logger.debug(f"Reset {self.__class__.__name__}, step_count=0")

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update task weights and recompute probabilities.

        Args:
            new_weights: Dict mapping task names to new weights.
        """
        for task, weight in new_weights.items():
            if task in self.task_weights:
                self.task_weights[task] = weight
            else:
                logger.warning(f"Unknown task '{task}' in weight update, ignoring")

        self._compute_probabilities()
        logger.debug(f"Updated weights: {self.task_weights}, new probs: {self._probabilities}")

    def get_state(self) -> dict[str, Any]:
        """Get sampler state for checkpointing."""
        return {
            "class": self.__class__.__name__,
            "task_names": self.task_names,
            "task_weights": self.task_weights,
            "seed": self.seed,
            "step_count": self._step_count,
            "rng_state": self._rng.getstate(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load sampler state from checkpoint."""
        self.task_weights = state.get("task_weights", self.task_weights)
        self._step_count = state.get("step_count", 0)
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        self._compute_probabilities()


# =============================================================================
# Proportional Sampler
# =============================================================================


@dataclass
class ProportionalSampler(TaskSampler):
    """
    Sample tasks proportional to their dataset size weighted by task weights.

    P(task) ∝ dataset_size[task] × weight[task]

    This is the default strategy for multi-task learning, ensuring that
    larger datasets contribute more to training while still allowing
    weight adjustments for task importance.

    Args:
        task_sizes: Dict mapping task names to dataset sizes.
        task_weights: Optional dict mapping task names to weights.
        seed: Random seed for reproducibility.

    Example:
        >>> task_sizes = {"ner": 1000, "sentiment": 5000, "emotions": 2000}
        >>> sampler = ProportionalSampler(task_sizes)
        >>> samples = [sampler.sample() for _ in range(1000)]
        >>> # sentiment will be sampled ~5x more than ner
    """

    task_sizes: dict[str, int] = field(default_factory=dict)

    def __init__(
        self,
        task_sizes: dict[str, int],
        task_names: list[str] | None = None,
        task_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Initialize ProportionalSampler."""
        # Allow passing task_sizes as first positional argument
        self.task_sizes = task_sizes
        # Extract task names from sizes if not provided
        if task_names is None:
            task_names = list(task_sizes.keys())
        # Set parent fields
        self.task_names = task_names
        self.task_weights = task_weights
        self.seed = seed
        # Run parent post_init
        TaskSampler.__post_init__(self)

    def _compute_probabilities(self) -> None:
        """Compute probabilities proportional to size × weight."""
        if not self.task_sizes:
            # Fall back to uniform if no sizes
            n = len(self.task_names)
            self._probabilities = dict.fromkeys(self.task_names, 1.0 / n)
            return

        # Compute unnormalized probabilities
        unnorm = {}
        for task in self.task_names:
            size = self.task_sizes.get(task, 1)
            weight = self.task_weights.get(task, 1.0) if self.task_weights else 1.0
            unnorm[task] = size * weight

        # Normalize
        total = sum(unnorm.values())
        if total == 0:
            total = 1.0

        self._probabilities = {task: p / total for task, p in unnorm.items()}


# =============================================================================
# Temperature Sampler
# =============================================================================


@dataclass
class TemperatureSampler(TaskSampler):
    """
    Sample tasks using softmax with temperature parameter.

    P(task) ∝ exp(log(size[task] × weight[task]) / temperature)

    Temperature controls the sharpness of the distribution:
    - temperature → 0: Always sample the largest task
    - temperature = 1: Proportional to size × weight
    - temperature → ∞: Uniform distribution

    Higher temperature = more balanced sampling across tasks.
    Lower temperature = favor larger/higher-weighted tasks.

    Args:
        task_sizes: Dict mapping task names to dataset sizes.
        temperature: Temperature parameter (default: 1.0).
        task_weights: Optional dict mapping task names to weights.
        seed: Random seed for reproducibility.

    Example:
        >>> task_sizes = {"ner": 1000, "sentiment": 5000, "emotions": 2000}
        >>> sampler = TemperatureSampler(task_sizes, temperature=2.0)
        >>> # More balanced than proportional due to higher temperature
    """

    task_sizes: dict[str, int] = field(default_factory=dict)
    temperature: float = 1.0

    def __init__(
        self,
        task_sizes: dict[str, int],
        temperature: float = 1.0,
        task_names: list[str] | None = None,
        task_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Initialize TemperatureSampler."""
        self.task_sizes = task_sizes
        self.temperature = temperature
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if task_names is None:
            task_names = list(task_sizes.keys())
        self.task_names = task_names
        self.task_weights = task_weights
        self.seed = seed
        TaskSampler.__post_init__(self)

    def _compute_probabilities(self) -> None:
        """Compute softmax probabilities with temperature."""
        if not self.task_sizes:
            n = len(self.task_names)
            self._probabilities = dict.fromkeys(self.task_names, 1.0 / n)
            return

        # Compute log(size × weight) / temperature
        log_scores = {}
        for task in self.task_names:
            size = self.task_sizes.get(task, 1)
            weight = self.task_weights.get(task, 1.0) if self.task_weights else 1.0
            # Use log to prevent overflow, add small epsilon for zero sizes
            log_scores[task] = math.log(max(size * weight, 1e-10)) / self.temperature

        # Softmax: subtract max for numerical stability
        max_score = max(log_scores.values())
        exp_scores = {task: math.exp(score - max_score) for task, score in log_scores.items()}

        # Normalize
        total = sum(exp_scores.values())
        self._probabilities = {task: p / total for task, p in exp_scores.items()}

    def set_temperature(self, temperature: float) -> None:
        """Update the temperature and recompute probabilities."""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self._compute_probabilities()


# =============================================================================
# Uniform Sampler
# =============================================================================


@dataclass
class UniformSampler(TaskSampler):
    """
    Sample tasks with equal probability regardless of dataset size.

    P(task) = 1 / num_tasks

    Useful when you want to ensure equal representation of all tasks
    regardless of their dataset sizes.

    Args:
        task_names: List of task names to sample from.
        seed: Random seed for reproducibility.

    Example:
        >>> sampler = UniformSampler(["ner", "sentiment", "emotions"])
        >>> samples = [sampler.sample() for _ in range(1000)]
        >>> # Each task sampled ~333 times
    """

    def _compute_probabilities(self) -> None:
        """Compute uniform probabilities."""
        n = len(self.task_names)
        self._probabilities = dict.fromkeys(self.task_names, 1.0 / n)


# =============================================================================
# Sequential Sampler (Round-Robin)
# =============================================================================


@dataclass
class SequentialSampler(TaskSampler):
    """
    Sample tasks in round-robin order.

    Cycles through tasks sequentially, optionally repeating tasks
    based on their weights (higher weight = more repetitions per cycle).

    Args:
        task_names: List of task names to sample from.
        task_weights: Optional dict mapping task names to repetition counts.
            Default weight of 1.0 = 1 sample per cycle.
        seed: Random seed (not used but kept for API consistency).

    Example:
        >>> sampler = SequentialSampler(["ner", "sentiment", "emotions"])
        >>> [sampler.sample() for _ in range(6)]
        ['ner', 'sentiment', 'emotions', 'ner', 'sentiment', 'emotions']

        >>> sampler = SequentialSampler(
        ...     ["ner", "sentiment"],
        ...     task_weights={"ner": 1, "sentiment": 2}
        ... )
        >>> [sampler.sample() for _ in range(6)]
        ['ner', 'sentiment', 'sentiment', 'ner', 'sentiment', 'sentiment']
    """

    # Internal state
    _cycle: list[str] = field(default_factory=list, init=False, repr=False)
    _cycle_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize and build the cycle."""
        super().__post_init__()
        self._build_cycle()

    def _compute_probabilities(self) -> None:
        """Compute probabilities based on cycle composition."""
        # For sequential, probabilities reflect the cycle composition
        cycle_counts = {}
        for task in self.task_names:
            weight = self.task_weights.get(task, 1.0) if self.task_weights else 1.0
            cycle_counts[task] = max(1, int(weight))

        total = sum(cycle_counts.values())
        self._probabilities = {task: count / total for task, count in cycle_counts.items()}

    def _build_cycle(self) -> None:
        """Build the sampling cycle based on weights."""
        self._cycle = []
        for task in self.task_names:
            weight = self.task_weights.get(task, 1.0) if self.task_weights else 1.0
            repetitions = max(1, int(weight))
            self._cycle.extend([task] * repetitions)
        self._cycle_index = 0

    def sample(self) -> str:
        """
        Sample the next task in the cycle.

        Returns:
            Task name at current position in cycle.
        """
        self._step_count += 1

        task = self._cycle[self._cycle_index]
        self._cycle_index = (self._cycle_index + 1) % len(self._cycle)

        return task

    def reset(self, seed: int | None = None) -> None:
        """Reset to start of cycle."""
        super().reset(seed)
        self._cycle_index = 0

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update weights and rebuild cycle."""
        super().update_weights(new_weights)
        self._build_cycle()


# =============================================================================
# Curriculum Sampler
# =============================================================================


@dataclass
class CurriculumSampler(TaskSampler):
    """
    Sample tasks with curriculum learning - gradually shift focus over training.

    Starts by focusing on "easier" tasks and gradually increases the weight
    of "harder" tasks. Task difficulty is specified via difficulty_order.

    Schedules:
    - linear: Weight increases linearly with steps
    - exponential: Weight increases exponentially
    - step: Discrete step changes at specified points

    Args:
        task_sizes: Dict mapping task names to dataset sizes.
        difficulty_order: List of tasks from easiest to hardest.
            Tasks listed first get higher initial weight.
        total_steps: Total training steps for schedule computation.
        warmup_fraction: Fraction of steps before curriculum kicks in.
        schedule: "linear", "exponential", or "step".
        task_weights: Optional base weights per task.
        seed: Random seed for reproducibility.

    Example:
        >>> sampler = CurriculumSampler(
        ...     task_sizes={"sentiment": 5000, "ner": 1000},
        ...     difficulty_order=["sentiment", "ner"],  # sentiment is easier
        ...     total_steps=10000,
        ...     schedule="linear",
        ... )
        >>> # Early: mostly sentiment. Late: more balanced with ner.
    """

    task_sizes: dict[str, int] = field(default_factory=dict)
    difficulty_order: list[str] = field(default_factory=list)
    total_steps: int = 10000
    warmup_fraction: float = 0.1
    schedule: str = "linear"

    # Internal
    _base_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize curriculum sampler."""
        if not self.task_names and self.task_sizes:
            self.task_names = list(self.task_sizes.keys())

        if not self.difficulty_order:
            self.difficulty_order = self.task_names.copy()

        # Store base weights
        super().__post_init__()
        self._base_weights = dict(self.task_weights) if self.task_weights else {}

    def _compute_probabilities(self) -> None:
        """Compute probabilities based on curriculum progress."""
        progress = self._get_curriculum_progress()

        # Compute curriculum weights
        curriculum_weights = {}
        n_tasks = len(self.difficulty_order)

        for i, task in enumerate(self.difficulty_order):
            # Earlier tasks (easier) start with higher weight
            # Later tasks (harder) gain weight as training progresses
            difficulty_rank = i / max(n_tasks - 1, 1)  # 0 = easiest, 1 = hardest

            # Base weight from config
            base = self._base_weights.get(task, 1.0) if self._base_weights else 1.0

            # Curriculum modifier: easy tasks start high, hard tasks grow
            if self.schedule == "linear":
                # Easy tasks: 1 -> (1 - progress), Hard tasks: 0 -> progress
                modifier = (1 - difficulty_rank) * (1 - progress) + difficulty_rank * progress
            elif self.schedule == "exponential":
                # Exponential growth for harder tasks
                modifier = (1 - difficulty_rank) * math.exp(-2 * progress) + difficulty_rank * (
                    1 - math.exp(-2 * progress)
                )
            else:  # step
                # Step function: hard tasks only after warmup
                if progress < 0.33:
                    modifier = 1.0 if difficulty_rank < 0.33 else 0.1
                elif progress < 0.66:
                    modifier = 1.0 if difficulty_rank < 0.66 else 0.3
                else:
                    modifier = 1.0

            curriculum_weights[task] = base * max(modifier, 0.05)  # Min 5% weight

        # Combine with sizes for final probabilities
        unnorm = {}
        for task in self.task_names:
            size = self.task_sizes.get(task, 1)
            weight = curriculum_weights.get(task, 1.0)
            unnorm[task] = size * weight

        total = sum(unnorm.values())
        self._probabilities = {task: p / total for task, p in unnorm.items()}

    def _get_curriculum_progress(self) -> float:
        """Get curriculum progress (0 = start, 1 = end)."""
        warmup_steps = int(self.total_steps * self.warmup_fraction)

        if self._step_count < warmup_steps:
            return 0.0

        remaining_steps = self.total_steps - warmup_steps
        progress_steps = self._step_count - warmup_steps

        return min(progress_steps / max(remaining_steps, 1), 1.0)

    def sample(self) -> str:
        """Sample with curriculum-adjusted probabilities."""
        # Recompute probabilities based on current step
        self._compute_probabilities()
        return super().sample()


# =============================================================================
# Factory Function
# =============================================================================


def create_sampler(
    strategy: str,
    task_sizes: dict[str, int],
    task_weights: dict[str, float] | None = None,
    seed: int | None = None,
    **kwargs,
) -> TaskSampler:
    """
    Factory function to create a task sampler.

    Args:
        strategy: Sampling strategy - "proportional", "temperature", "uniform",
            "sequential", or "curriculum".
        task_sizes: Dict mapping task names to dataset sizes.
        task_weights: Optional dict mapping task names to weights.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments for specific samplers:
            - temperature: For TemperatureSampler (default: 1.0)
            - difficulty_order: For CurriculumSampler
            - total_steps: For CurriculumSampler (default: 10000)
            - schedule: For CurriculumSampler (default: "linear")

    Returns:
        TaskSampler instance.

    Example:
        >>> sampler = create_sampler(
        ...     strategy="temperature",
        ...     task_sizes={"ner": 1000, "sentiment": 5000},
        ...     temperature=2.0,
        ... )
    """
    task_names = list(task_sizes.keys())

    if strategy == "proportional":
        return ProportionalSampler(
            task_names=task_names,
            task_sizes=task_sizes,
            task_weights=task_weights,
            seed=seed,
        )
    elif strategy == "temperature":
        return TemperatureSampler(
            task_names=task_names,
            task_sizes=task_sizes,
            task_weights=task_weights,
            temperature=kwargs.get("temperature", 1.0),
            seed=seed,
        )
    elif strategy == "uniform":
        return UniformSampler(
            task_names=task_names,
            task_weights=task_weights,
            seed=seed,
        )
    elif strategy == "sequential":
        return SequentialSampler(
            task_names=task_names,
            task_weights=task_weights,
            seed=seed,
        )
    elif strategy == "curriculum":
        return CurriculumSampler(
            task_names=task_names,
            task_sizes=task_sizes,
            task_weights=task_weights,
            difficulty_order=kwargs.get("difficulty_order", task_names),
            total_steps=kwargs.get("total_steps", 10000),
            warmup_fraction=kwargs.get("warmup_fraction", 0.1),
            schedule=kwargs.get("schedule", "linear"),
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Expected: proportional, temperature, uniform, sequential, curriculum"
        )

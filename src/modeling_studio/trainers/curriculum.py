"""
Curriculum Learning Strategy for Multi-Task Training

This module provides curriculum learning strategies to improve training
stability and convergence by introducing tasks in a staged manner.

Key Concepts:
    - Staged task introduction: Start with easier tasks, add harder ones
    - Task difficulty scoring: Estimate task difficulty for ordering
    - Dynamic curriculum: Adjust task weights during training

Curriculum Strategies:
    1. Predefined stages: Manually specify task order and epochs
    2. Difficulty-based: Auto-order by estimated difficulty
    3. Loss-based: Add tasks when current tasks plateau

Benefits:
    - More stable training dynamics
    - Better transfer from generic to domain-specific tasks
    - Reduced catastrophic forgetting

Stage A → Stage B Strategy:
    Stage A (Generic): NER_GENERAL, SENTIMENT, EMOTIONS, NLI, EMBEDDING
    Stage B (FamilyOS): NER_FAMILY, INGRESS, SAFETY_FAMILYOS, RELATION, INTENT

Usage:
    from modeling_studio.trainers.curriculum import CurriculumScheduler

    scheduler = CurriculumScheduler(
        stages=[
            {"tasks": ["sentiment", "ner_general"], "epochs": 3},
            {"tasks": ["sentiment", "ner_general", "emotions", "nli"], "epochs": 3},
            {"tasks": "all", "epochs": 4},
        ]
    )

    for epoch in range(total_epochs):
        active_tasks = scheduler.get_active_tasks(epoch)
        # Train only on active_tasks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Task Difficulty Definitions
# =============================================================================


class TaskDifficulty(Enum):
    """Task difficulty levels for curriculum ordering."""

    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4


# Default difficulty estimates for each task
# Lower = easier, should be trained first
DEFAULT_TASK_DIFFICULTY: dict[str, TaskDifficulty] = {
    # Generic tasks (Stage A) - generally easier
    "sentiment": TaskDifficulty.EASY,
    "ner_general": TaskDifficulty.EASY,
    "nli": TaskDifficulty.MEDIUM,
    "emotions": TaskDifficulty.MEDIUM,
    "embedding": TaskDifficulty.EASY,
    "safety_generic": TaskDifficulty.MEDIUM,
    "temporal": TaskDifficulty.MEDIUM,
    # FamilyOS tasks (Stage B) - domain-specific, harder
    "ner_family": TaskDifficulty.HARD,
    "ingress": TaskDifficulty.MEDIUM,
    "safety_familyos": TaskDifficulty.VERY_HARD,  # Critical, complex
    "relation": TaskDifficulty.HARD,
    "intent": TaskDifficulty.MEDIUM,
}


# =============================================================================
# Curriculum Stage Definition
# =============================================================================


@dataclass
class CurriculumStage:
    """
    Definition of a curriculum stage.

    Args:
        tasks: List of task names active in this stage, or "all"
        epochs: Number of epochs for this stage
        task_weights: Optional per-task weights for this stage
        description: Human-readable description
    """

    tasks: list[str] | str
    epochs: int
    task_weights: dict[str, float] | None = None
    description: str = ""

    def get_task_list(self, all_tasks: list[str]) -> list[str]:
        """Get the actual task list, expanding 'all' if needed."""
        if self.tasks == "all":
            return all_tasks
        elif isinstance(self.tasks, str):
            return [self.tasks]
        else:
            return list(self.tasks)


# =============================================================================
# Curriculum Scheduler
# =============================================================================


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Stage definitions
    stages: list[dict[str, Any]] = field(default_factory=list)

    # Auto-curriculum settings
    auto_difficulty_order: bool = False
    difficulty_weights: dict[str, int] | None = None

    # Dynamic curriculum settings
    loss_threshold_for_progression: float = 0.1
    min_epochs_per_stage: int = 1

    # Warmup settings
    warmup_epochs: int = 0
    warmup_tasks: list[str] | None = None


class CurriculumScheduler:
    """
    Curriculum learning scheduler for multi-task training.

    Manages task introduction across training epochs according to
    a predefined curriculum or dynamic difficulty-based ordering.

    Supports:
        - Predefined stage-based curriculum
        - Automatic difficulty-based ordering
        - Dynamic task weight adjustment
        - Integration with MultiTaskTrainer

    Args:
        stages: List of stage definitions, each with:
            - tasks: List of task names or "all"
            - epochs: Number of epochs for this stage
            - task_weights: Optional per-task weights
        all_tasks: Complete list of all available tasks
        config: CurriculumConfig for additional settings

    Example:
        >>> scheduler = CurriculumScheduler(
        ...     stages=[
        ...         {"tasks": ["sentiment", "ner_general"], "epochs": 3},
        ...         {"tasks": ["sentiment", "ner_general", "emotions", "nli"], "epochs": 3},
        ...         {"tasks": "all", "epochs": 4},
        ...     ]
        ... )
        >>> current_tasks = scheduler.get_active_tasks(epoch=2)
        >>> assert current_tasks == ["sentiment", "ner_general"]
    """

    def __init__(
        self,
        stages: list[dict[str, Any]] | None = None,
        all_tasks: list[str] | None = None,
        config: CurriculumConfig | None = None,
    ):
        self.config = config or CurriculumConfig()

        # Set stages from config or argument
        if stages is not None:
            self.config.stages = stages

        # Parse stages into CurriculumStage objects
        self._stages = self._parse_stages(self.config.stages)

        # All available tasks
        self._all_tasks = all_tasks or self._infer_all_tasks()

        # Build epoch-to-stage mapping
        self._epoch_stage_map = self._build_epoch_map()

        # Current state
        self._current_epoch = 0
        self._current_stage_idx = 0

        logger.info(
            f"CurriculumScheduler initialized with {len(self._stages)} stages, "
            f"{sum(s.epochs for s in self._stages)} total epochs"
        )

    def _parse_stages(self, stage_dicts: list[dict[str, Any]]) -> list[CurriculumStage]:
        """Parse stage dictionaries into CurriculumStage objects."""
        stages = []

        for i, stage_dict in enumerate(stage_dicts):
            tasks = stage_dict.get("tasks", "all")
            epochs = stage_dict.get("epochs", 1)
            task_weights = stage_dict.get("task_weights")
            description = stage_dict.get("description", f"Stage {i + 1}")

            stage = CurriculumStage(
                tasks=tasks,
                epochs=epochs,
                task_weights=task_weights,
                description=description,
            )
            stages.append(stage)

        return stages

    def _infer_all_tasks(self) -> list[str]:
        """Infer all tasks from stage definitions."""
        all_tasks: set[str] = set()

        for stage in self._stages:
            if stage.tasks != "all" and isinstance(stage.tasks, list):
                all_tasks.update(stage.tasks)

        # If no explicit tasks found, use default task list
        if not all_tasks:
            all_tasks = set(DEFAULT_TASK_DIFFICULTY.keys())

        return sorted(all_tasks)

    def _build_epoch_map(self) -> dict[int, int]:
        """Build mapping from epoch number to stage index."""
        epoch_map = {}
        current_epoch = 0

        for stage_idx, stage in enumerate(self._stages):
            for _ in range(stage.epochs):
                epoch_map[current_epoch] = stage_idx
                current_epoch += 1

        return epoch_map

    @property
    def total_epochs(self) -> int:
        """Total number of epochs across all stages."""
        return sum(stage.epochs for stage in self._stages)

    @property
    def num_stages(self) -> int:
        """Number of curriculum stages."""
        return len(self._stages)

    @property
    def all_tasks(self) -> list[str]:
        """List of all available tasks."""
        return self._all_tasks

    def get_active_tasks(self, epoch: int) -> list[str]:
        """
        Get list of active tasks for a given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            List of task names active for this epoch
        """
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")

        # Handle epoch beyond defined stages (use last stage)
        if epoch >= self.total_epochs:
            stage_idx = len(self._stages) - 1
        else:
            stage_idx = self._epoch_stage_map.get(epoch, len(self._stages) - 1)

        stage = self._stages[stage_idx]
        return stage.get_task_list(self._all_tasks)

    def get_task_weights(self, epoch: int) -> dict[str, float] | None:
        """
        Get task weights for a given epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary of task weights, or None for uniform weights
        """
        if epoch >= self.total_epochs:
            stage_idx = len(self._stages) - 1
        else:
            stage_idx = self._epoch_stage_map.get(epoch, len(self._stages) - 1)

        return self._stages[stage_idx].task_weights

    def get_stage_info(self, epoch: int) -> dict[str, Any]:
        """
        Get detailed information about the current stage.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary with stage information
        """
        if epoch >= self.total_epochs:
            stage_idx = len(self._stages) - 1
        else:
            stage_idx = self._epoch_stage_map.get(epoch, len(self._stages) - 1)

        stage = self._stages[stage_idx]
        active_tasks = stage.get_task_list(self._all_tasks)

        # Calculate epoch within stage
        epochs_before = sum(self._stages[i].epochs for i in range(stage_idx))
        epoch_in_stage = epoch - epochs_before

        return {
            "stage_index": stage_idx,
            "stage_description": stage.description,
            "active_tasks": active_tasks,
            "task_weights": stage.task_weights,
            "epoch_in_stage": epoch_in_stage,
            "epochs_in_stage": stage.epochs,
            "is_final_stage": stage_idx == len(self._stages) - 1,
        }

    def is_task_active(self, task: str, epoch: int) -> bool:
        """
        Check if a specific task is active at a given epoch.

        Args:
            task: Task name
            epoch: Training epoch

        Returns:
            True if task is active
        """
        active_tasks = self.get_active_tasks(epoch)
        return task in active_tasks

    def step(self) -> dict[str, Any]:
        """
        Advance to next epoch and return current state.

        Returns:
            Dictionary with current stage info
        """
        info = self.get_stage_info(self._current_epoch)
        self._current_epoch += 1
        return info

    def reset(self) -> None:
        """Reset scheduler to beginning."""
        self._current_epoch = 0
        self._current_stage_idx = 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CurriculumScheduler("
            f"stages={self.num_stages}, "
            f"total_epochs={self.total_epochs}, "
            f"tasks={len(self._all_tasks)})"
        )


# =============================================================================
# Predefined Curriculum Strategies
# =============================================================================


def create_stage_a_to_b_curriculum(
    stage_a_epochs: int = 5,
    mixed_epochs: int = 3,
    stage_b_epochs: int = 2,
) -> CurriculumScheduler:
    """
    Create a standard Stage A → Stage B curriculum.

    Strategy:
        1. Train on generic tasks (Stage A) first
        2. Mix in FamilyOS tasks gradually
        3. Fine-tune on all tasks

    Args:
        stage_a_epochs: Epochs for Stage A only
        mixed_epochs: Epochs for mixed training
        stage_b_epochs: Epochs for all tasks

    Returns:
        Configured CurriculumScheduler
    """
    stage_a_tasks = [
        "ner_general",
        "sentiment",
        "emotions",
        "nli",
        "embedding",
        "safety_generic",
        "temporal",
    ]

    stage_b_tasks = [
        "ner_family",
        "ingress",
        "safety_familyos",
        "relation",
        "intent",
    ]

    stages = [
        {
            "tasks": stage_a_tasks,
            "epochs": stage_a_epochs,
            "description": "Stage A: Generic capabilities",
        },
        {
            "tasks": stage_a_tasks + ["ingress", "ner_family"],
            "epochs": mixed_epochs // 2,
            "description": "Mixed: Adding easier FamilyOS tasks",
        },
        {
            "tasks": stage_a_tasks + stage_b_tasks,
            "epochs": mixed_epochs - (mixed_epochs // 2),
            "description": "Mixed: All tasks",
            "task_weights": {
                # Higher weight for safety during this phase
                "safety_familyos": 5.0,
                "safety_generic": 2.0,
            },
        },
        {
            "tasks": "all",
            "epochs": stage_b_epochs,
            "description": "Stage B: Full fine-tuning",
            "task_weights": {
                "safety_familyos": 10.0,
            },
        },
    ]

    return CurriculumScheduler(stages=stages, all_tasks=stage_a_tasks + stage_b_tasks)


def create_difficulty_based_curriculum(
    epochs_per_difficulty: int = 3,
    custom_difficulties: dict[str, TaskDifficulty] | None = None,
) -> CurriculumScheduler:
    """
    Create a difficulty-based curriculum.

    Tasks are introduced from easiest to hardest based on
    difficulty scores.

    Args:
        epochs_per_difficulty: Epochs per difficulty level
        custom_difficulties: Override default difficulty estimates

    Returns:
        Configured CurriculumScheduler
    """
    difficulties = custom_difficulties or DEFAULT_TASK_DIFFICULTY

    # Group tasks by difficulty
    difficulty_groups: dict[TaskDifficulty, list[str]] = {d: [] for d in TaskDifficulty}

    for task, difficulty in difficulties.items():
        difficulty_groups[difficulty].append(task)

    # Build stages from easy to hard
    stages = []
    cumulative_tasks: list[str] = []

    for difficulty in TaskDifficulty:
        tasks_at_level = difficulty_groups[difficulty]
        if tasks_at_level:
            cumulative_tasks = cumulative_tasks + tasks_at_level
            stages.append(
                {
                    "tasks": list(cumulative_tasks),
                    "epochs": epochs_per_difficulty,
                    "description": f"Difficulty: {difficulty.name}",
                }
            )

    return CurriculumScheduler(stages=stages, all_tasks=list(difficulties.keys()))


def create_safety_focused_curriculum(
    warmup_epochs: int = 2,
    safety_emphasis_epochs: int = 5,
    full_training_epochs: int = 3,
) -> CurriculumScheduler:
    """
    Create a curriculum with emphasis on safety tasks.

    Strategy:
        1. Warmup with basic tasks
        2. Heavy emphasis on safety tasks
        3. Full training maintaining safety focus

    Args:
        warmup_epochs: Initial warmup epochs
        safety_emphasis_epochs: Epochs emphasizing safety
        full_training_epochs: Final training epochs

    Returns:
        Configured CurriculumScheduler
    """
    stages = [
        {
            "tasks": ["sentiment", "ner_general", "emotions"],
            "epochs": warmup_epochs,
            "description": "Warmup: Basic classification",
        },
        {
            "tasks": [
                "sentiment",
                "ner_general",
                "emotions",
                "safety_generic",
                "safety_familyos",
            ],
            "epochs": safety_emphasis_epochs,
            "task_weights": {
                "safety_familyos": 15.0,  # CRISIS detection is critical
                "safety_generic": 5.0,
            },
            "description": "Safety emphasis: High weight on safety tasks",
        },
        {
            "tasks": "all",
            "epochs": full_training_epochs,
            "task_weights": {
                "safety_familyos": 10.0,
                "safety_generic": 3.0,
            },
            "description": "Full training: All tasks with safety focus",
        },
    ]

    return CurriculumScheduler(stages=stages)


# =============================================================================
# Curriculum Callback for Integration
# =============================================================================


class CurriculumCallback:
    """
    Callback for integrating CurriculumScheduler with training loops.

    Can be used with custom training loops or adapted for
    HuggingFace Trainer callbacks.

    Args:
        scheduler: CurriculumScheduler instance
        trainer_ref: Optional reference to trainer for task updates
    """

    def __init__(
        self,
        scheduler: CurriculumScheduler,
        trainer_ref: Any | None = None,
    ):
        self.scheduler = scheduler
        self.trainer_ref = trainer_ref
        self._last_active_tasks: list[str] | None = None

    def on_epoch_begin(self, epoch: int) -> dict[str, Any]:
        """
        Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Stage information for logging
        """
        stage_info = self.scheduler.get_stage_info(epoch)
        active_tasks = stage_info["active_tasks"]

        # Log stage transitions
        if self._last_active_tasks is None or set(active_tasks) != set(self._last_active_tasks):
            logger.info(
                f"Epoch {epoch}: Stage '{stage_info['stage_description']}' - "
                f"Active tasks: {active_tasks}"
            )
            self._last_active_tasks = active_tasks

        return stage_info

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | None = None) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            metrics: Optional training metrics
        """
        stage_info = self.scheduler.get_stage_info(epoch)

        if metrics:
            logger.info(
                f"Epoch {epoch} complete - Stage: {stage_info['stage_description']}, "
                f"Metrics: {metrics}"
            )

    def get_active_tasks(self, epoch: int) -> list[str]:
        """Get active tasks for an epoch."""
        return self.scheduler.get_active_tasks(epoch)

    def get_task_weights(self, epoch: int) -> dict[str, float] | None:
        """Get task weights for an epoch."""
        return self.scheduler.get_task_weights(epoch)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TaskDifficulty",
    # Data classes
    "CurriculumStage",
    "CurriculumConfig",
    # Main class
    "CurriculumScheduler",
    # Factory functions
    "create_stage_a_to_b_curriculum",
    "create_difficulty_based_curriculum",
    "create_safety_focused_curriculum",
    # Callback
    "CurriculumCallback",
    # Data
    "DEFAULT_TASK_DIFFICULTY",
]

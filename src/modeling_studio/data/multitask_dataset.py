"""
Multi-Task Dataset

This module provides dataset classes for loading and combining
multiple datasets for multi-task learning.

Classes:
    - TaskDataset: Wrapper adding task information to samples
    - MultiTaskDataset: Combines multiple task datasets
    - StreamingMultiTaskDataset: For large datasets (streaming mode)

Features:
    - Unified interface across tasks
    - Task-aware batching
    - On-the-fly preprocessing
    - Memory-efficient streaming
    - Task weighting for sampling

Dataset Format:
    Each sample is a dict containing:
    {
        "input_ids": [...],
        "attention_mask": [...],
        "labels": ... (task-specific),
        "task": "task_name"
    }

Usage:
    from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

    # Wrap individual datasets
    ner_ds = TaskDataset(name="ner_general", dataset=ner_hf_dataset)
    sent_ds = TaskDataset(name="sentiment", dataset=sentiment_hf_dataset)

    # Combine into multi-task dataset
    multitask_ds = MultiTaskDataset([ner_ds, sent_ds])

    # Access samples
    sample = multitask_ds[0]
    assert "task" in sample  # Which task this sample belongs to
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class TaskDataset:
    """
    Wrapper that adds task information to samples from a HuggingFace Dataset.

    This wrapper ensures each sample includes a "task" field indicating
    which task it belongs to, enabling proper routing in multi-task learning.

    Args:
        name: Task name (e.g., "ner_general", "sentiment", "emotions").
            This should match the capability name in the model.
        dataset: HuggingFace Dataset to wrap.
        preprocessing_fn: Optional function to preprocess each sample.
            Function signature: (sample: dict) -> dict
        weight: Sampling weight for this task. Higher weight = more samples.
            Default: 1.0

    Attributes:
        name: Task name
        dataset: Wrapped HuggingFace Dataset
        preprocessing_fn: Optional preprocessing function
        weight: Sampling weight

    Example:
        >>> from datasets import load_dataset
        >>> ner_data = load_dataset("conll2003", split="train")
        >>> ner_task = TaskDataset(name="ner_general", dataset=ner_data)
        >>> sample = ner_task[0]
        >>> assert sample["task"] == "ner_general"
    """

    name: str
    dataset: Dataset
    preprocessing_fn: Callable[[dict], dict] | None = None
    weight: float = 1.0

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample by index with task information added.

        Args:
            idx: Sample index.

        Returns:
            Sample dict with "task" field added.
        """
        sample = dict(self.dataset[idx])

        # Apply preprocessing if provided
        if self.preprocessing_fn is not None:
            sample = self.preprocessing_fn(sample)

        # Add task identifier
        sample["task"] = self.name

        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples with task information."""
        for idx in range(len(self)):
            yield self[idx]

    @property
    def column_names(self) -> list[str]:
        """Return column names from the underlying dataset plus 'task'."""
        return self.dataset.column_names + ["task"]

    @property
    def features(self) -> dict:
        """Return features from the underlying dataset."""
        return self.dataset.features

    def select(self, indices: list[int]) -> TaskDataset:
        """
        Select a subset of the dataset by indices.

        Args:
            indices: List of indices to select.

        Returns:
            New TaskDataset with selected samples.
        """
        return TaskDataset(
            name=self.name,
            dataset=self.dataset.select(indices),
            preprocessing_fn=self.preprocessing_fn,
            weight=self.weight,
        )

    def shuffle(self, seed: int | None = None) -> TaskDataset:
        """
        Shuffle the dataset.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            New TaskDataset with shuffled samples.
        """
        return TaskDataset(
            name=self.name,
            dataset=self.dataset.shuffle(seed=seed),
            preprocessing_fn=self.preprocessing_fn,
            weight=self.weight,
        )


@dataclass
class MultiTaskDataset:
    """
    Combined dataset for multi-task learning.

    Concatenates multiple TaskDatasets and provides unified access.
    Each sample includes a "task" field to identify its source task.

    Args:
        task_datasets: List of TaskDataset instances to combine.
        shuffle: Whether to shuffle the combined dataset. Default: False

    Attributes:
        task_datasets: List of wrapped task datasets
        task_names: List of task names
        task_sizes: Dict mapping task name to dataset size
        cumulative_sizes: Cumulative sizes for index mapping

    Example:
        >>> ner_ds = TaskDataset(name="ner_general", dataset=ner_data)
        >>> sent_ds = TaskDataset(name="sentiment", dataset=sent_data)
        >>> multitask = MultiTaskDataset([ner_ds, sent_ds])
        >>> print(f"Total samples: {len(multitask)}")
        >>> sample = multitask[0]
        >>> assert "task" in sample
    """

    task_datasets: list[TaskDataset]
    shuffle: bool = False

    # Computed fields
    _task_names: list[str] = field(default_factory=list, init=False, repr=False)
    _task_sizes: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _cumulative_sizes: list[int] = field(default_factory=list, init=False, repr=False)
    _shuffled_indices: list[int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize computed fields after dataclass init."""
        if not self.task_datasets:
            raise ValueError("At least one TaskDataset is required")

        self._task_names = [ds.name for ds in self.task_datasets]
        self._task_sizes = {ds.name: len(ds) for ds in self.task_datasets}

        # Compute cumulative sizes for O(log n) index lookup
        cumsum = 0
        self._cumulative_sizes = []
        for ds in self.task_datasets:
            cumsum += len(ds)
            self._cumulative_sizes.append(cumsum)

        # Shuffle if requested
        if self.shuffle:
            self._shuffled_indices = list(range(len(self)))
            random.shuffle(self._shuffled_indices)

        logger.info(
            f"Created MultiTaskDataset with {len(self.task_datasets)} tasks, "
            f"{len(self)} total samples: {self._task_sizes}"
        )

    @property
    def task_names(self) -> list[str]:
        """List of task names in this dataset."""
        return self._task_names

    @property
    def task_sizes(self) -> dict[str, int]:
        """Dict mapping task name to number of samples."""
        return self._task_sizes

    def __len__(self) -> int:
        """Return total number of samples across all tasks."""
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def _get_task_and_index(self, global_idx: int) -> tuple[int, int]:
        """
        Convert global index to (task_index, local_index).

        Uses binary search for O(log n) lookup.

        Args:
            global_idx: Global index in the combined dataset.

        Returns:
            Tuple of (task_index, local_index_within_task)
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Index {global_idx} out of range [0, {len(self)})")

        # Binary search for task
        left, right = 0, len(self._cumulative_sizes) - 1
        while left < right:
            mid = (left + right) // 2
            if self._cumulative_sizes[mid] <= global_idx:
                left = mid + 1
            else:
                right = mid

        task_idx = left

        # Compute local index
        if task_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self._cumulative_sizes[task_idx - 1]

        return task_idx, local_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample by global index.

        Args:
            idx: Global index in the combined dataset.

        Returns:
            Sample dict with "task" field indicating source task.
        """
        # Apply shuffling if enabled
        if self._shuffled_indices is not None:
            idx = self._shuffled_indices[idx]

        task_idx, local_idx = self._get_task_and_index(idx)
        return self.task_datasets[task_idx][local_idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples in order."""
        for idx in range(len(self)):
            yield self[idx]

    def get_task_dataset(self, task_name: str) -> TaskDataset:
        """
        Get the TaskDataset for a specific task.

        Args:
            task_name: Name of the task.

        Returns:
            TaskDataset for the task.

        Raises:
            KeyError: If task not found.
        """
        for ds in self.task_datasets:
            if ds.name == task_name:
                return ds
        raise KeyError(f"Task '{task_name}' not found. Available: {self._task_names}")

    def get_task_samples(self, task_name: str) -> Iterator[dict[str, Any]]:
        """
        Iterate over samples from a specific task.

        Args:
            task_name: Name of the task.

        Yields:
            Samples from the specified task.
        """
        task_ds = self.get_task_dataset(task_name)
        yield from task_ds

    def get_task_weights(self) -> dict[str, float]:
        """
        Get the sampling weights for each task.

        Returns:
            Dict mapping task name to weight.
        """
        return {ds.name: ds.weight for ds in self.task_datasets}

    def reshuffle(self, seed: int | None = None) -> None:
        """
        Reshuffle the dataset indices.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self._shuffled_indices = list(range(len(self)))
        random.shuffle(self._shuffled_indices)

    def split_by_task(self) -> dict[str, TaskDataset]:
        """
        Split the MultiTaskDataset back into individual TaskDatasets.

        Returns:
            Dict mapping task name to TaskDataset.
        """
        return {ds.name: ds for ds in self.task_datasets}


class StreamingMultiTaskDataset:
    """
    Streaming multi-task dataset for very large datasets.

    Interleaves samples from multiple streaming datasets without
    loading everything into memory. Useful for datasets that don't
    fit in memory.

    Args:
        task_datasets: Dict mapping task name to IterableDataset.
        task_weights: Dict mapping task name to sampling probability.
            Weights are normalized to sum to 1.0.
        seed: Random seed for reproducibility.

    Example:
        >>> from datasets import load_dataset
        >>> ner_stream = load_dataset("conll2003", split="train", streaming=True)
        >>> sent_stream = load_dataset("sst2", split="train", streaming=True)
        >>> stream_ds = StreamingMultiTaskDataset(
        ...     task_datasets={"ner_general": ner_stream, "sentiment": sent_stream},
        ...     task_weights={"ner_general": 1.0, "sentiment": 2.0},
        ... )
        >>> for sample in stream_ds:
        ...     print(sample["task"])
        ...     break
    """

    def __init__(
        self,
        task_datasets: dict[str, IterableDataset],
        task_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())

        # Normalize weights
        if task_weights is None:
            task_weights = dict.fromkeys(self.task_names, 1.0)

        total_weight = sum(task_weights.values())
        self.task_probabilities = {
            name: weight / total_weight for name, weight in task_weights.items()
        }

        self.seed = seed
        self._rng = random.Random(seed)

        logger.info(
            f"Created StreamingMultiTaskDataset with {len(self.task_names)} tasks, "
            f"weights: {self.task_probabilities}"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over interleaved samples from all tasks.

        Uses weighted sampling to select which task to draw from.
        """
        # Create iterators for each task
        iterators = {name: iter(ds) for name, ds in self.task_datasets.items()}
        active_tasks = set(self.task_names)

        while active_tasks:
            # Sample a task based on weights (only from active tasks)
            active_probs = {
                name: prob for name, prob in self.task_probabilities.items() if name in active_tasks
            }
            total = sum(active_probs.values())
            if total == 0:
                break

            # Normalize active probabilities
            normalized = {name: prob / total for name, prob in active_probs.items()}

            # Weighted random choice
            r = self._rng.random()
            cumsum = 0.0
            selected_task = list(active_tasks)[0]
            for name, prob in normalized.items():
                cumsum += prob
                if r <= cumsum:
                    selected_task = name
                    break

            # Try to get next sample from selected task
            try:
                sample = next(iterators[selected_task])
                sample = dict(sample)
                sample["task"] = selected_task
                yield sample
            except StopIteration:
                # Task exhausted, remove from active set
                active_tasks.remove(selected_task)
                logger.debug(f"Task '{selected_task}' exhausted")


def create_multitask_dataset(
    datasets: dict[str, Dataset],
    weights: dict[str, float] | None = None,
    preprocessing_fns: dict[str, Callable[[dict], dict]] | None = None,
    shuffle: bool = False,
) -> MultiTaskDataset:
    """
    Convenience function to create a MultiTaskDataset from a dict of datasets.

    Args:
        datasets: Dict mapping task name to HuggingFace Dataset.
        weights: Optional dict mapping task name to sampling weight.
            Default: 1.0 for all tasks.
        preprocessing_fns: Optional dict mapping task name to preprocessing function.
        shuffle: Whether to shuffle the combined dataset.

    Returns:
        MultiTaskDataset instance.

    Example:
        >>> from datasets import load_dataset
        >>> datasets = {
        ...     "ner_general": load_dataset("conll2003", split="train"),
        ...     "sentiment": load_dataset("sst2", split="train"),
        ... }
        >>> multitask_ds = create_multitask_dataset(datasets, shuffle=True)
    """
    if weights is None:
        weights = dict.fromkeys(datasets, 1.0)

    if preprocessing_fns is None:
        preprocessing_fns = {}

    task_datasets = []
    for name, dataset in datasets.items():
        task_ds = TaskDataset(
            name=name,
            dataset=dataset,
            preprocessing_fn=preprocessing_fns.get(name),
            weight=weights.get(name, 1.0),
        )
        task_datasets.append(task_ds)

    return MultiTaskDataset(task_datasets=task_datasets, shuffle=shuffle)


def interleave_datasets(
    datasets: list[TaskDataset],
    probabilities: list[float] | None = None,
    seed: int | None = None,
    stopping_strategy: str = "first_exhausted",
) -> Iterator[dict[str, Any]]:
    """
    Interleave samples from multiple TaskDatasets with given probabilities.

    Useful for custom sampling strategies during training.

    Args:
        datasets: List of TaskDataset instances.
        probabilities: Sampling probability for each dataset.
            If None, uses uniform probabilities.
        seed: Random seed for reproducibility.
        stopping_strategy: When to stop iteration:
            - "first_exhausted": Stop when any dataset is exhausted
            - "all_exhausted": Stop when all datasets are exhausted

    Yields:
        Samples from interleaved datasets.

    Example:
        >>> ner_ds = TaskDataset(name="ner", dataset=ner_data)
        >>> sent_ds = TaskDataset(name="sentiment", dataset=sent_data)
        >>> for sample in interleave_datasets([ner_ds, sent_ds], [0.3, 0.7]):
        ...     process(sample)
    """
    if not datasets:
        return

    n_datasets = len(datasets)

    if probabilities is None:
        probabilities = [1.0 / n_datasets] * n_datasets
    else:
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

    rng = random.Random(seed)
    indices = [0] * n_datasets
    exhausted = [False] * n_datasets

    while True:
        # Check stopping condition
        if stopping_strategy == "first_exhausted":
            if any(exhausted):
                break
        elif stopping_strategy == "all_exhausted":
            if all(exhausted):
                break
        else:
            raise ValueError(f"Unknown stopping strategy: {stopping_strategy}")

        # Adjust probabilities for exhausted datasets
        active_probs = [p if not exhausted[i] else 0.0 for i, p in enumerate(probabilities)]
        total = sum(active_probs)
        if total == 0:
            break
        active_probs = [p / total for p in active_probs]

        # Sample dataset
        r = rng.random()
        cumsum = 0.0
        selected = 0
        for i, prob in enumerate(active_probs):
            cumsum += prob
            if r <= cumsum:
                selected = i
                break

        # Get sample from selected dataset
        ds = datasets[selected]
        if indices[selected] >= len(ds):
            exhausted[selected] = True
            continue

        sample = ds[indices[selected]]
        indices[selected] += 1
        yield sample


# Export public API
__all__ = [
    "TaskDataset",
    "MultiTaskDataset",
    "StreamingMultiTaskDataset",
    "create_multitask_dataset",
    "interleave_datasets",
]

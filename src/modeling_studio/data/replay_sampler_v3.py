"""Stage A replay sampler for ModernBERT v3 training.

This module implements a sampler that mixes primary (FamilyOS) data with
Stage A replay data (SST-2, CoNLL, MNLI) to prevent catastrophic forgetting.

Key features:
    - Configurable replay ratio with minimum samples per epoch
    - Optional task-balanced sampling across replay tasks
    - Dynamic replay ratio adjustment based on forgetting loss
    - Interleaved sampling producing well-mixed batches
    - Dataset wrapper that tags each sample with its source
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import cast

from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for replay sampling.

    Attributes:
        replay_ratio: Fraction of replay samples relative to primary data.
        task_balanced: Whether to balance sampling across replay tasks.
        min_replay_per_epoch: Minimum number of replay samples per epoch.
        dynamic_ratio: Whether to adjust replay ratio based on forgetting loss.
        loss_threshold: Loss threshold to trigger ratio increase.
        max_replay_ratio: Maximum allowable replay ratio.
    """

    replay_ratio: float = 0.15
    task_balanced: bool = True
    min_replay_per_epoch: int = 100
    dynamic_ratio: bool = True
    loss_threshold: float = 0.5
    max_replay_ratio: float = 0.3


class ReplaySampler(Sampler):
    """Sampler that mixes primary training data with replay data.

    The replay mechanism prevents catastrophic forgetting by replaying
    Stage A benchmark samples during training.

    Sampling strategy:
        1. For each epoch, compute replay sample count based on ratio.
        2. Optionally balance replay samples across tasks.
        3. Interleave primary and replay indices according to ratio.

    Yields:
        Tuples of (source, index) where source is "primary" or "replay".
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        config: ReplayConfig | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.primary_dataset = primary_dataset
        self.replay_dataset = replay_dataset
        self.config = config or ReplayConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.current_replay_ratio = self.config.replay_ratio

        self._calculate_sample_counts()
        self.rng = random.Random(seed)
        self._build_task_indices()

        logger.info("ReplaySampler initialized:")
        logger.info("  Primary samples: %d", len(cast(Sized, self.primary_dataset)))
        logger.info("  Replay samples: %d", len(cast(Sized, self.replay_dataset)))
        logger.info("  Replay ratio: %.2f%%", self.current_replay_ratio * 100)

    def _calculate_sample_counts(self) -> None:
        """Calculate per-epoch sample counts."""
        total_primary = len(cast(Sized, self.primary_dataset))

        requested_replay = max(
            int(
                total_primary
                * self.current_replay_ratio
                / max(1e-8, 1 - self.current_replay_ratio)
            ),
            self.config.min_replay_per_epoch,
        )

        available_replay = len(cast(Sized, self.replay_dataset))
        replay_count = min(requested_replay, available_replay)
        if replay_count < requested_replay:
            logger.warning(
                "Replay dataset smaller than requested: %d available vs %d requested",
                replay_count,
                requested_replay,
            )

        self.n_primary_per_epoch = total_primary
        self.n_replay_per_epoch = replay_count
        self.total_samples = self.n_primary_per_epoch + self.n_replay_per_epoch

    def _build_task_indices(self) -> None:
        """Build mapping from task name to replay indices for balancing."""
        if not self.config.task_balanced:
            self.task_indices: dict[str, list[int]] | None = None
            return

        task_indices: dict[str, list[int]] = {}
        for idx in range(len(cast(Sized, self.replay_dataset))):
            sample = self.replay_dataset[idx]
            task = sample.get("task", sample.get("task_name", "unknown"))
            task_indices.setdefault(task, []).append(idx)

        self.task_indices = task_indices

        if self.task_indices:
            logger.info("Replay task distribution:")
            for task, indices in self.task_indices.items():
                logger.info("  %s: %d samples", task, len(indices))

    def _sample_replay_indices(self) -> list[int]:
        """Sample replay indices with optional task balancing."""
        if not self.config.task_balanced or not self.task_indices:
            all_indices = list(range(len(cast(Sized, self.replay_dataset))))
            sample_size = min(self.n_replay_per_epoch, len(all_indices))
            return self.rng.sample(all_indices, sample_size)

        tasks = list(self.task_indices.keys())
        samples_per_task = max(1, self.n_replay_per_epoch // max(1, len(tasks)))

        replay_indices: list[int] = []
        for task in tasks:
            task_pool = self.task_indices[task]
            n_samples = min(samples_per_task, len(task_pool))
            replay_indices.extend(self.rng.sample(task_pool, n_samples))

        remaining = self.n_replay_per_epoch - len(replay_indices)
        if remaining > 0:
            all_indices = list(range(len(cast(Sized, self.replay_dataset))))
            extra = self.rng.sample(all_indices, min(remaining, len(all_indices)))
            replay_indices.extend(extra)

        return replay_indices[: self.n_replay_per_epoch]

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Generate interleaved indices tagged by source."""
        primary_indices = list(range(len(cast(Sized, self.primary_dataset))))
        replay_indices = self._sample_replay_indices()

        if self.shuffle:
            self.rng.shuffle(primary_indices)
            self.rng.shuffle(replay_indices)

        tagged_primary = [("primary", idx) for idx in primary_indices]
        tagged_replay = [("replay", idx) for idx in replay_indices]

        all_indices: list[tuple[str, int]] = []
        p_ptr, r_ptr = 0, 0

        while p_ptr < len(tagged_primary) or r_ptr < len(tagged_replay):
            if r_ptr >= len(tagged_replay):
                all_indices.append(tagged_primary[p_ptr])
                p_ptr += 1
            elif p_ptr >= len(tagged_primary):
                all_indices.append(tagged_replay[r_ptr])
                r_ptr += 1
            else:
                if self.rng.random() < self.current_replay_ratio:
                    all_indices.append(tagged_replay[r_ptr])
                    r_ptr += 1
                else:
                    all_indices.append(tagged_primary[p_ptr])
                    p_ptr += 1

        yield from all_indices

    def __len__(self) -> int:  # type: ignore[override]
        return self.total_samples

    def update_replay_ratio(self, forgetting_loss: float) -> None:
        """Adjust replay ratio based on forgetting loss.

        Args:
            forgetting_loss: Measured loss indicating forgetting severity.
        """
        if not self.config.dynamic_ratio:
            return

        old_ratio = self.current_replay_ratio

        if forgetting_loss > self.config.loss_threshold:
            self.current_replay_ratio = min(
                self.current_replay_ratio * 1.2,
                self.config.max_replay_ratio,
            )
            logger.info(
                "Forgetting loss %.3f > %.3f, increasing replay ratio: %.2f%% -> %.2f%%",
                forgetting_loss,
                self.config.loss_threshold,
                old_ratio * 100,
                self.current_replay_ratio * 100,
            )
        elif forgetting_loss < self.config.loss_threshold * 0.5:
            self.current_replay_ratio = max(
                self.current_replay_ratio * 0.9,
                self.config.replay_ratio,
            )

        if not math.isclose(old_ratio, self.current_replay_ratio, rel_tol=1e-6):
            self._calculate_sample_counts()


class ReplayDataset(Dataset):
    """Dataset wrapper that exposes interleaved primary/replay samples."""

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        sampler: ReplaySampler,
    ) -> None:
        self.primary_dataset = primary_dataset
        self.replay_dataset = replay_dataset
        self.sampler = sampler

        self._epoch_indices: list[tuple[str, int]] = []
        self._refresh_epoch()

    def _refresh_epoch(self) -> None:
        """Refresh indices for a new epoch."""
        self._epoch_indices = list(self.sampler)
        self._current_idx = 0

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._epoch_indices)

    def __getitem__(self, idx: int):  # type: ignore[override]
        source, source_idx = self._epoch_indices[idx]

        if source == "primary":
            item = self.primary_dataset[source_idx]
        else:
            item = self.replay_dataset[source_idx]

        item = dict(item)
        item["_source"] = source
        item["_is_replay"] = source == "replay"
        return item

    def refresh(self) -> None:
        """Public method to refresh indices between epochs."""
        self._refresh_epoch()


def create_replay_sampler(
    primary_dataset: Dataset,
    replay_dataset: Dataset,
    replay_ratio: float = 0.15,
    batch_size: int = 32,
    task_balanced: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    **config_kwargs,
) -> tuple[ReplayDataset, ReplaySampler]:
    """Factory to create replay-enabled dataset and sampler.

    Args:
        primary_dataset: Main training dataset.
        replay_dataset: Stage A replay dataset.
        replay_ratio: Fraction of samples from replay data.
        batch_size: Training batch size.
        task_balanced: Whether to balance replay samples across tasks.
        shuffle: Whether to shuffle within each epoch.
        seed: Random seed for deterministic sampling.
        **config_kwargs: Additional replay configuration overrides.

    Returns:
        Tuple of (ReplayDataset, ReplaySampler).
    """
    config = ReplayConfig(
        replay_ratio=replay_ratio,
        task_balanced=task_balanced,
        **config_kwargs,
    )

    sampler = ReplaySampler(
        primary_dataset=primary_dataset,
        replay_dataset=replay_dataset,
        config=config,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    dataset = ReplayDataset(
        primary_dataset=primary_dataset,
        replay_dataset=replay_dataset,
        sampler=sampler,
    )

    return dataset, sampler

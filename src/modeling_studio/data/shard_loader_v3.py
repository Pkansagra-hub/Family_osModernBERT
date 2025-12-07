"""
Shard-Based Data Loading for v3 Multi-Task Training.

Provides memory-efficient loading of large JSONL shard files with support for:
- Streaming iteration
- Parallel shard loading
- Worker-aware distribution
- Resume from checkpoint
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset, get_worker_info

from modeling_studio.data.loaders_v3 import TaskType, UnifiedSample

logger = logging.getLogger(__name__)


# ============================================================================
# Shard Configuration
# ============================================================================


@dataclass
class ShardConfig:
    """Configuration for shard-based loading."""

    # Paths
    data_dir: str
    shard_pattern: str = "shard_*.jsonl"

    # Loading behavior
    shuffle_shards: bool = True
    shuffle_within_shard: bool = False  # Memory intensive if True

    # Memory management
    buffer_size: int = 10000  # Samples to buffer per worker
    prefetch_shards: int = 2  # Number of shards to prefetch

    # Parallel loading
    num_loading_threads: int = 2

    # Filtering
    min_text_length: int = 5
    max_text_length: int = 2000
    require_hub_routing: bool = False
    filter_tasks: list[str] | None = None

    # Validation
    validate_samples: bool = True
    skip_invalid: bool = True

    # Resume support
    checkpoint_path: str | None = None

    # Statistics
    collect_stats: bool = True


@dataclass
class ShardStats:
    """Statistics for a single shard."""

    shard_path: str
    num_samples: int = 0
    num_valid: int = 0
    num_skipped: int = 0
    task_counts: dict[str, int] = field(default_factory=dict)
    hub_counts: dict[str, int] = field(default_factory=dict)
    avg_text_length: float = 0.0

    def merge(self, other: ShardStats) -> ShardStats:
        """Merge statistics from another shard."""

        merged = ShardStats(shard_path=f"{self.shard_path}+{other.shard_path}")
        merged.num_samples = self.num_samples + other.num_samples
        merged.num_valid = self.num_valid + other.num_valid
        merged.num_skipped = self.num_skipped + other.num_skipped

        # Merge task counts
        for task, count in self.task_counts.items():
            merged.task_counts[task] = count
        for task, count in other.task_counts.items():
            merged.task_counts[task] = merged.task_counts.get(task, 0) + count

        # Merge hub counts
        for hub, count in self.hub_counts.items():
            merged.hub_counts[hub] = count
        for hub, count in other.hub_counts.items():
            merged.hub_counts[hub] = merged.hub_counts.get(hub, 0) + count

        # Weighted average text length
        if merged.num_valid > 0:
            total_length = (
                self.avg_text_length * self.num_valid + other.avg_text_length * other.num_valid
            )
            merged.avg_text_length = total_length / merged.num_valid

        return merged


# ============================================================================
# Shard Index
# ============================================================================


@dataclass
class ShardIndex:
    """Index of available shards with metadata."""

    shards: list[dict[str, Any]] = field(default_factory=list)
    total_samples: int = 0
    total_shards: int = 0

    @classmethod
    def build(cls, data_dir: str, shard_pattern: str = "shard_*.jsonl") -> ShardIndex:
        """Build index by scanning shard files."""

        index = cls()

        shard_files = sorted(glob.glob(str(Path(data_dir) / shard_pattern)))
        index.total_shards = len(shard_files)

        logger.info("Indexing %d shard files...", len(shard_files))

        for shard_path in shard_files:
            with open(shard_path, encoding="utf-8") as handle:
                num_samples = sum(1 for line in handle if line.strip())

            index.shards.append(
                {
                    "path": shard_path,
                    "num_samples": num_samples,
                    "size_bytes": os.path.getsize(shard_path),
                }
            )
            index.total_samples += num_samples

        logger.info(
            "Index complete: %d samples across %d shards", index.total_samples, index.total_shards
        )
        return index

    def get_worker_shards(self, worker_id: int, num_workers: int) -> list[dict[str, Any]]:
        """Get shards assigned to a specific worker."""

        return [shard for i, shard in enumerate(self.shards) if i % num_workers == worker_id]

    def save(self, path: str) -> None:
        """Save index to disk."""

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "shards": self.shards,
                    "total_samples": self.total_samples,
                    "total_shards": self.total_shards,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> ShardIndex:
        """Load index from disk."""

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        index = cls()
        index.shards = data["shards"]
        index.total_samples = data["total_samples"]
        index.total_shards = data["total_shards"]
        return index


# ============================================================================
# Shard Reader
# ============================================================================


class ShardReader:
    """Reads samples from a single shard file."""

    def __init__(self, shard_path: str, config: ShardConfig):
        self.shard_path = shard_path
        self.config = config
        self.stats = ShardStats(shard_path=shard_path)

    def __iter__(self) -> Iterator[UnifiedSample]:
        """Iterate over samples in shard."""

        text_lengths: list[int] = []

        with open(self.shard_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                self.stats.num_samples += 1

                try:
                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)

                    if self.config.validate_samples and not self._validate_sample(sample):
                        self.stats.num_skipped += 1
                        if self.config.skip_invalid:
                            continue

                    self.stats.num_valid += 1

                    if self.config.collect_stats:
                        text_lengths.append(len(sample.text))
                        self._update_task_stats(sample)
                        self._update_hub_stats(sample)

                    yield sample

                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON in %s: %s", self.shard_path, exc)
                    self.stats.num_skipped += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error parsing sample from %s: %s", self.shard_path, exc)
                    self.stats.num_skipped += 1

        if text_lengths:
            self.stats.avg_text_length = sum(text_lengths) / len(text_lengths)

    def _validate_sample(self, sample: UnifiedSample) -> bool:
        """Validate a sample against config filters."""

        text_len = len(sample.text)
        if text_len < self.config.min_text_length:
            return False
        if text_len > self.config.max_text_length:
            return False

        if self.config.require_hub_routing and not sample.hub_routing.active_hubs:
            return False

        if self.config.filter_tasks:
            allowed = [
                task for task in self.config.filter_tasks if task in [t.value for t in TaskType]
            ]
            if allowed:
                has_task = any(
                    sample.has_task(TaskType(task)) for task in TaskType if task.value in allowed
                )
                if not has_task:
                    return False

        return True

    def _update_task_stats(self, sample: UnifiedSample) -> None:
        """Update task statistics."""

        for task_type in TaskType:
            if sample.has_task(task_type):
                task_name = task_type.value
                self.stats.task_counts[task_name] = self.stats.task_counts.get(task_name, 0) + 1

    def _update_hub_stats(self, sample: UnifiedSample) -> None:
        """Update hub routing statistics."""

        for hub in sample.hub_routing.active_hubs:
            self.stats.hub_counts[hub] = self.stats.hub_counts.get(hub, 0) + 1


# ============================================================================
# Streaming Shard Dataset
# ============================================================================


class StreamingShardDataset(IterableDataset):
    """Memory-efficient streaming dataset over multiple shards."""

    def __init__(
        self,
        config: ShardConfig,
        transform: Callable[[UnifiedSample], Any] | None = None,
        epoch: int = 0,
    ) -> None:
        self.config = config
        self.transform = transform
        self.epoch = epoch

        self.index = ShardIndex.build(config.data_dir, config.shard_pattern)
        self.total_stats: ShardStats | None = None

        logger.info(
            "StreamingShardDataset initialized: %d samples, %d shards",
            self.index.total_samples,
            self.index.total_shards,
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples from assigned shards."""

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        worker_shards = self.index.get_worker_shards(worker_id, num_workers)

        if not worker_shards:
            logger.warning("Worker %d has no assigned shards", worker_id)
            return iter(())

        logger.debug(
            "Worker %d/%d processing %d shards", worker_id, num_workers, len(worker_shards)
        )

        if self.config.shuffle_shards:
            rng = random.Random(42 + self.epoch + worker_id)
            worker_shards = worker_shards.copy()
            rng.shuffle(worker_shards)

        start_offset = self._load_resume_offset(worker_id, worker_shards)
        shards_to_process = worker_shards[start_offset:]

        all_stats: list[ShardStats] = []

        for shard_info in shards_to_process:
            shard_path = shard_info["path"]
            reader = ShardReader(shard_path, self.config)

            for sample in reader:
                if self.transform:
                    yield self.transform(sample)
                else:
                    yield sample

            if self.config.collect_stats:
                all_stats.append(reader.stats)

            self._write_resume_checkpoint(worker_id, shard_path)

        if all_stats and self.config.collect_stats:
            merged = all_stats[0]
            for stats in all_stats[1:]:
                merged = merged.merge(stats)
            self.total_stats = merged

    def __len__(self) -> int:
        return self.index.total_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""

        self.epoch = epoch

    def get_stats(self) -> ShardStats | None:
        """Get aggregated statistics (available after full iteration)."""

        return self.total_stats

    def _worker_checkpoint_path(self, worker_id: int) -> Path | None:
        if self.config.checkpoint_path is None:
            return None
        base_path = Path(self.config.checkpoint_path)
        if base_path.is_dir():
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"shard_loader_worker_{worker_id}.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        return base_path.with_suffix(f".worker{worker_id}{base_path.suffix}")

    def _load_resume_offset(self, worker_id: int, shards: list[dict[str, Any]]) -> int:
        checkpoint = self._worker_checkpoint_path(worker_id)
        if checkpoint is None or not checkpoint.exists():
            return 0

        try:
            with open(checkpoint, encoding="utf-8") as handle:
                data = json.load(handle)
            last_shard = data.get("last_shard")
            for index, shard_info in enumerate(shards):
                if shard_info.get("path") == last_shard:
                    return index + 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load shard checkpoint %s: %s", checkpoint, exc)
        return 0

    def _write_resume_checkpoint(self, worker_id: int, shard_path: str) -> None:
        checkpoint = self._worker_checkpoint_path(worker_id)
        if checkpoint is None:
            return
        try:
            with open(checkpoint, "w", encoding="utf-8") as handle:
                json.dump({"last_shard": shard_path}, handle)
        except OSError as exc:
            logger.warning("Failed to write shard checkpoint %s: %s", checkpoint, exc)


# ============================================================================
# Buffered Shard Dataset
# ============================================================================


class BufferedShardDataset(IterableDataset):
    """Buffered streaming dataset with prefetching."""

    def __init__(
        self,
        config: ShardConfig,
        transform: Callable[[UnifiedSample], Any] | None = None,
        epoch: int = 0,
    ) -> None:
        self.config = config
        self.transform = transform
        self.epoch = epoch

        self.index = ShardIndex.build(config.data_dir, config.shard_pattern)
        self._buffer_lock = threading.Lock()

    def _prefetch_shard(self, shard_path: str) -> list[UnifiedSample]:
        reader = ShardReader(shard_path, self.config)
        return list(reader)

    def __iter__(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        worker_shards = self.index.get_worker_shards(worker_id, num_workers)

        if self.config.shuffle_shards:
            rng = random.Random(42 + self.epoch + worker_id)
            worker_shards = worker_shards.copy()
            rng.shuffle(worker_shards)

        start_offset = self._load_resume_offset(worker_id, worker_shards)
        shard_iter = iter(worker_shards[start_offset:])

        futures: dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=self.config.num_loading_threads) as executor:
            for _ in range(self.config.prefetch_shards):
                try:
                    shard_info = next(shard_iter)
                except StopIteration:
                    break
                futures[executor.submit(self._prefetch_shard, shard_info["path"])] = shard_info[
                    "path"
                ]

            while futures:
                for future in as_completed(list(futures.keys())):
                    shard_path = futures.pop(future)
                    samples = future.result()

                    if self.config.shuffle_within_shard:
                        rng = random.Random(42 + self.epoch + hash(shard_path))
                        rng.shuffle(samples)

                    for sample in samples:
                        if self.transform:
                            yield self.transform(sample)
                        else:
                            yield sample

                    self._write_resume_checkpoint(worker_id, shard_path)

                    try:
                        shard_info = next(shard_iter)
                        futures[executor.submit(self._prefetch_shard, shard_info["path"])] = (
                            shard_info["path"]
                        )
                    except StopIteration:
                        continue

    def __len__(self) -> int:
        return self.index.total_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_checkpoint_path(self, worker_id: int) -> Path | None:
        if self.config.checkpoint_path is None:
            return None
        base_path = Path(self.config.checkpoint_path)
        if base_path.is_dir():
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"shard_loader_worker_{worker_id}.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        return base_path.with_suffix(f".worker{worker_id}{base_path.suffix}")

    def _load_resume_offset(self, worker_id: int, shards: list[dict[str, Any]]) -> int:
        checkpoint = self._worker_checkpoint_path(worker_id)
        if checkpoint is None or not checkpoint.exists():
            return 0

        try:
            with open(checkpoint, encoding="utf-8") as handle:
                data = json.load(handle)
            last_shard = data.get("last_shard")
            for index, shard_info in enumerate(shards):
                if shard_info.get("path") == last_shard:
                    return index + 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load shard checkpoint %s: %s", checkpoint, exc)
        return 0

    def _write_resume_checkpoint(self, worker_id: int, shard_path: str) -> None:
        checkpoint = self._worker_checkpoint_path(worker_id)
        if checkpoint is None:
            return
        try:
            with open(checkpoint, "w", encoding="utf-8") as handle:
                json.dump({"last_shard": shard_path}, handle)
        except OSError as exc:
            logger.warning("Failed to write shard checkpoint %s: %s", checkpoint, exc)


# ============================================================================
# Factory Functions
# ============================================================================


def create_shard_dataset(
    data_dir: str,
    shard_pattern: str = "shard_*.jsonl",
    streaming: bool = True,
    buffered: bool = True,
    transform: Callable[[UnifiedSample], Any] | None = None,
    **config_kwargs: Any,
) -> StreamingShardDataset | BufferedShardDataset:
    """Create a shard-based dataset."""

    config = ShardConfig(
        data_dir=data_dir,
        shard_pattern=shard_pattern,
        **config_kwargs,
    )

    if buffered:
        return BufferedShardDataset(config, transform)
    return StreamingShardDataset(config, transform)


def get_shard_statistics(data_dir: str, shard_pattern: str = "shard_*.jsonl") -> ShardStats:
    """Compute aggregate statistics over all shards."""

    config = ShardConfig(data_dir=data_dir, shard_pattern=shard_pattern, collect_stats=True)

    shard_files = sorted(glob.glob(str(Path(data_dir) / shard_pattern)))

    all_stats: list[ShardStats] = []

    for shard_path in shard_files:
        reader = ShardReader(shard_path, config)
        for _ in reader:
            pass
        all_stats.append(reader.stats)

    if not all_stats:
        return ShardStats(shard_path=data_dir)

    merged = all_stats[0]
    for stats in all_stats[1:]:
        merged = merged.merge(stats)

    return merged

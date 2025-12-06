from __future__ import annotations

import pytest

from modeling_studio.data.replay_sampler_v3 import (
    ReplayConfig,
    ReplayDataset,
    ReplaySampler,
    create_replay_sampler,
)


class DummyDataset:
    """Simple dataset for testing replay sampling."""

    def __init__(self, size: int, task: str | None = None, tasks: list[str] | None = None):
        self.size = size
        self.task = task
        self.tasks = tasks or []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        if self.tasks:
            task_name = self.tasks[idx % len(self.tasks)]
        else:
            task_name = self.task or "unknown"
        return {"id": idx, "task": task_name}


class TestReplaySampler:
    def test_replay_ratio_counts(self) -> None:
        primary = DummyDataset(40, task="family")
        replay = DummyDataset(20, task="sst2")

        config = ReplayConfig(
            replay_ratio=0.2,
            task_balanced=False,
            min_replay_per_epoch=5,
            dynamic_ratio=False,
        )

        sampler = ReplaySampler(primary, replay, config=config, shuffle=False, seed=0)
        indices = list(sampler)

        replay_indices = [src for src, _ in indices if src == "replay"]

        assert len(indices) == sampler.total_samples
        assert len(replay_indices) == sampler.n_replay_per_epoch
        assert sampler.n_replay_per_epoch == 10  # 40 * 0.2 / 0.8

    def test_task_balanced_sampling(self) -> None:
        primary = DummyDataset(6, task="family")
        replay_tasks = ["sst2", "mnli", "conll"]
        replay = DummyDataset(12, tasks=replay_tasks)

        config = ReplayConfig(
            replay_ratio=0.5,
            task_balanced=True,
            min_replay_per_epoch=6,
            dynamic_ratio=False,
        )

        sampler = ReplaySampler(primary, replay, config=config, shuffle=False, seed=0)
        indices = list(sampler)

        replay_samples = [replay[idx] for src, idx in indices if src == "replay"]
        task_counts = dict.fromkeys(replay_tasks, 0)
        for sample in replay_samples:
            task_counts[sample["task"]] += 1

        # Balanced sampling should include each task
        assert all(count > 0 for count in task_counts.values())
        assert len(replay_samples) == sampler.n_replay_per_epoch

    def test_dynamic_replay_ratio_updates(self) -> None:
        primary = DummyDataset(10, task="family")
        replay = DummyDataset(30, task="sst2")

        config = ReplayConfig(
            replay_ratio=0.1,
            max_replay_ratio=0.3,
            min_replay_per_epoch=1,
            loss_threshold=0.5,
        )

        sampler = ReplaySampler(primary, replay, config=config, shuffle=False, seed=0)

        sampler.update_replay_ratio(forgetting_loss=0.8)
        assert pytest.approx(sampler.current_replay_ratio, rel=1e-6) == 0.12

        sampler.update_replay_ratio(forgetting_loss=0.1)
        assert sampler.current_replay_ratio <= 0.12
        assert sampler.current_replay_ratio >= config.replay_ratio

    def test_replay_dataset_wrapper(self) -> None:
        primary = DummyDataset(4, task="family")
        replay = DummyDataset(4, task="sst2")

        dataset, sampler = create_replay_sampler(
            primary_dataset=primary,
            replay_dataset=replay,
            replay_ratio=0.25,
            min_replay_per_epoch=2,
            task_balanced=False,
            shuffle=False,
        )

        # Length should match sampler output
        assert len(dataset) == len(list(sampler))

        sample = dataset[0]
        assert "_source" in sample
        assert "_is_replay" in sample
        assert isinstance(dataset, ReplayDataset)

        # Refresh should not raise and should produce same length
        dataset.refresh()
        assert len(dataset) == len(list(sampler))


if __name__ == "__main__":
    pytest.main([__file__])

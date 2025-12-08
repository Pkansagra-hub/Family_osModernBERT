from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeling_studio.data.loaders_v3 import (
    HubRoutingParser,
    IterableUnifiedFamilyOSDataset,
    TaskType,
    UnifiedFamilyOSDataset,
    UnifiedSample,
)


def _write_shard(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    base = tmp_path / "shards"
    base.mkdir()

    sample_full = {
        "id": "fam_001",
        "text": "Family brunch with gratitude",
        "tasks": {
            "emotions": ["gratitude", "joy"],
            "sentiment": "positive",
            "ner_family": [{"start": 0, "end": 6, "label": "EVENT", "token": "Family"}],
            "safety_familyos": "GREEN",
            "intent": "share",
            "ingress": "DIARY",
            "relations": [{"subject": "Alice", "predicate": "parent_of", "object": "Bob"}],
            "temporal": [{"start": 13, "end": 19, "label": "DATE_ABS", "token": "Sunday"}],
        },
        "hub_routing": {"EMO": True, "REL": True, "MEM": True, "TASK": True},
    }

    sample_rel_only = {
        "id": "fam_002",
        "text": "Relation only sample",
        "tasks": {"relations": [{"subject": "Mike", "predicate": "sibling_of", "object": "Jane"}]},
        "hub_routing": {"REL": True},
    }

    sample_ner_no_hub = {
        "id": "fam_003",
        "text": "No hub routing present",
        "tasks": {"ner_family": [{"start": 0, "end": 4, "label": "NAME", "token": "John"}]},
        "hub_routing": {},
    }

    _write_shard(base / "shard_000.jsonl", [sample_full, sample_rel_only])
    _write_shard(base / "shard_001.jsonl", [sample_ner_no_hub])
    return base


class TestUnifiedFamilyOSDataset:
    def test_parses_samples_and_hub_routing(self, shard_dir: Path) -> None:
        dataset = UnifiedFamilyOSDataset(data_dir=shard_dir)

        assert len(dataset) == 3
        first: UnifiedSample = dataset[0]

        assert first.id == "fam_001"
        assert set(first.hub_routing.active_hubs) == {"EMO", "REL", "MEM", "TASK"}

        routing_tensor = first.hub_routing.to_tensor()
        assert routing_tensor.tolist() == [1.0, 1.0, 1.0, 1.0]

        assert first.has_task(TaskType.EMOTIONS)
        assert first.has_task(TaskType.RELATIONS)
        assert first.has_task(TaskType.TEMPORAL)

    def test_filtering_by_task_and_hub(self, shard_dir: Path) -> None:
        dataset = UnifiedFamilyOSDataset(
            data_dir=shard_dir,
            filter_tasks=[TaskType.RELATIONS],
            require_hub_routing=True,
        )

        assert len(dataset) == 2
        assert all(sample.has_task(TaskType.RELATIONS) for sample in dataset)
        assert all(sample.hub_routing.active_hubs for sample in dataset)

    def test_distribution_counts(self, shard_dir: Path) -> None:
        dataset = UnifiedFamilyOSDataset(data_dir=shard_dir)

        task_dist = dataset.get_task_distribution()
        assert task_dist[TaskType.EMOTIONS.value] == 1
        assert task_dist[TaskType.RELATIONS.value] == 2
        assert task_dist[TaskType.NER_FAMILY.value] == 2

        hub_dist = dataset.get_hub_distribution()
        assert hub_dist == {"EMO": 1, "REL": 2, "MEM": 1, "TASK": 1, "none": 1}


class TestIterableUnifiedFamilyOSDataset:
    def test_streaming_filters_and_order(self, shard_dir: Path) -> None:
        iterator = IterableUnifiedFamilyOSDataset(
            data_dir=shard_dir,
            shuffle_shards=False,
            filter_tasks=[TaskType.NER_FAMILY],
            require_hub_routing=False,
        )

        samples = list(iterator)
        ids = [sample.id for sample in samples]

        assert ids == ["fam_001", "fam_003"]
        assert all(sample.has_task(TaskType.NER_FAMILY) for sample in samples)

    def test_streaming_requires_hub(self, shard_dir: Path) -> None:
        iterator = IterableUnifiedFamilyOSDataset(
            data_dir=shard_dir,
            shuffle_shards=False,
            require_hub_routing=True,
        )

        samples = list(iterator)
        assert all(sample.hub_routing.active_hubs for sample in samples)
        assert len(samples) == 2


class TestHubRoutingParser:
    def _build_sample(
        self, sample_id: str, hub: dict[str, bool], tasks: dict[str, object]
    ) -> UnifiedSample:
        data = {"id": sample_id, "text": "sample text", "tasks": tasks, "hub_routing": hub}
        return UnifiedSample.from_json(data)

    def test_active_tasks_and_weights_with_safety_override(self) -> None:
        parser = HubRoutingParser(always_train_safety=True, safety_weight_override=2.0)

        sample = self._build_sample(
            "sample_emo_mem",
            hub={"EMO": True, "MEM": True},
            tasks={
                "emotions": ["joy"],
                "safety_familyos": "GREEN",
                "temporal": [{"start": 0, "end": 4, "label": "DATE", "token": "2024"}],
            },
        )

        active = parser.get_active_tasks(sample.hub_routing, sample)
        assert active == [TaskType.EMOTIONS, TaskType.SAFETY_FAMILYOS, TaskType.TEMPORAL]

        weights = parser.get_task_weights(sample.hub_routing, active)
        assert weights[TaskType.EMOTIONS] == pytest.approx(1.0 / 3.0)
        assert weights[TaskType.TEMPORAL] == pytest.approx(1.0 / 3.0)
        assert weights[TaskType.SAFETY_FAMILYOS] == pytest.approx(2.0 / 3.0)

    def test_parse_batch_outputs_masks_and_weights(self) -> None:
        parser = HubRoutingParser(always_train_safety=True, safety_weight_override=2.0)

        sample_a = self._build_sample(
            "sample_a",
            hub={"EMO": True, "MEM": True},
            tasks={
                "emotions": ["joy"],
                "safety_familyos": "GREEN",
                "temporal": [{"start": 0, "end": 4, "label": "DATE", "token": "2024"}],
            },
        )

        sample_b = self._build_sample(
            "sample_b",
            hub={"TASK": True},
            tasks={"intent": "share", "ingress": "DIARY"},
        )

        parsed = parser.parse_batch([sample_a, sample_b])

        hub_masks = parsed["hub_masks"].tolist()
        assert hub_masks == [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

        active = parsed["task_active"]
        assert active[TaskType.EMOTIONS] == [0]
        assert active[TaskType.TEMPORAL] == [0]
        assert active[TaskType.SAFETY_FAMILYOS] == [0]
        assert active[TaskType.INTENT] == [1]
        assert active[TaskType.INGRESS] == [1]

        weights = parsed["task_weights"]
        assert weights[TaskType.EMOTIONS][0].item() == pytest.approx(1.0 / 3.0)
        assert weights[TaskType.TEMPORAL][0].item() == pytest.approx(1.0 / 3.0)
        assert weights[TaskType.SAFETY_FAMILYOS][0].item() == pytest.approx(2.0 / 3.0)
        assert weights[TaskType.INTENT][1].item() == pytest.approx(0.5)
        assert weights[TaskType.INGRESS][1].item() == pytest.approx(0.5)
        assert weights[TaskType.RELATIONS].sum().item() == pytest.approx(0.0)

        # Zero-padding for inactive tasks per sample
        assert weights[TaskType.SENTIMENT][0].item() == 0.0
        assert weights[TaskType.NER_FAMILY][0].item() == 0.0
        assert weights[TaskType.INTENT][0].item() == 0.0
        assert weights[TaskType.INGRESS][0].item() == 0.0
        assert weights[TaskType.RELATIONS][0].item() == 0.0

        assert weights[TaskType.EMOTIONS][1].item() == 0.0
        assert weights[TaskType.SENTIMENT][1].item() == 0.0
        assert weights[TaskType.NER_FAMILY][1].item() == 0.0
        assert weights[TaskType.SAFETY_FAMILYOS][1].item() == 0.0
        assert weights[TaskType.TEMPORAL][1].item() == 0.0
        assert weights[TaskType.RELATIONS][1].item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__])

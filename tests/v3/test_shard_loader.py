from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeling_studio.data.shard_loader_v3 import (
    BufferedShardDataset,
    ShardConfig,
    ShardIndex,
    ShardReader,
    StreamingShardDataset,
    create_shard_dataset,
    get_shard_statistics,
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


class TestShardIndexAndReader:
    def test_builds_index_and_worker_split(self, shard_dir: Path) -> None:
        index = ShardIndex.build(str(shard_dir))

        assert index.total_shards == 2
        assert index.total_samples == 3

        worker_zero = index.get_worker_shards(worker_id=0, num_workers=2)
        worker_one = index.get_worker_shards(worker_id=1, num_workers=2)

        assert len(worker_zero) == 1
        assert len(worker_one) == 1

    def test_reader_validation_and_stats(self, shard_dir: Path, tmp_path: Path) -> None:
        shard_path = tmp_path / "shard_bad.jsonl"
        samples = [
            {"id": "invalid_json"},
            {
                "id": "too_short",
                "text": "no",
                "tasks": {"relations": []},
                "hub_routing": {"REL": True},
            },
            {
                "id": "valid",
                "text": "A valid sample text",
                "tasks": {"relations": [{"subject": "A", "predicate": "parent_of", "object": "B"}]},
                "hub_routing": {"REL": True},
            },
        ]
        _write_shard(shard_path, samples)

        config = ShardConfig(data_dir=str(shard_dir), collect_stats=True, min_text_length=5)
        reader = ShardReader(str(shard_path), config)

        rows = list(reader)
        assert len(rows) == 1
        assert reader.stats.num_samples == 3
        assert reader.stats.num_valid == 1
        assert reader.stats.num_skipped == 2
        assert reader.stats.task_counts["relations"] == 1
        assert reader.stats.hub_counts["REL"] == 1
        assert reader.stats.avg_text_length > 0


class TestStreamingShardDataset:
    def test_streaming_collects_stats_and_respects_resume(
        self, shard_dir: Path, tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "resume.json"
        worker_checkpoint = checkpoint.with_suffix(f".worker0{checkpoint.suffix}")
        worker_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with worker_checkpoint.open("w", encoding="utf-8") as handle:
            json.dump({"last_shard": str(shard_dir / "shard_000.jsonl")}, handle)

        config = ShardConfig(
            data_dir=str(shard_dir),
            shuffle_shards=False,
            collect_stats=True,
            checkpoint_path=str(checkpoint),
        )
        dataset = StreamingShardDataset(config)

        ids = [sample.id for sample in dataset]
        assert ids == ["fam_003"]

        stats = dataset.get_stats()
        assert stats is not None
        assert stats.num_valid == 1

        with worker_checkpoint.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["last_shard"].endswith("shard_001.jsonl")


class TestBufferedShardDataset:
    def test_buffered_prefetches_and_yields_all(self, shard_dir: Path) -> None:
        config = ShardConfig(
            data_dir=str(shard_dir),
            shuffle_shards=False,
            shuffle_within_shard=False,
            prefetch_shards=2,
            num_loading_threads=2,
        )
        dataset = BufferedShardDataset(config)

        ids = [sample.id for sample in dataset]
        assert sorted(ids) == ["fam_001", "fam_002", "fam_003"]


class TestFactoryAndStats:
    def test_factory_selects_dataset_type(self, shard_dir: Path) -> None:
        buffered_dataset = create_shard_dataset(
            data_dir=str(shard_dir),
            buffered=True,
            streaming=True,
        )
        assert isinstance(buffered_dataset, BufferedShardDataset)

        streaming_dataset = create_shard_dataset(
            data_dir=str(shard_dir),
            buffered=False,
            streaming=True,
        )
        assert isinstance(streaming_dataset, StreamingShardDataset)

    def test_get_shard_statistics_merges(self, shard_dir: Path) -> None:
        stats = get_shard_statistics(str(shard_dir))
        assert stats.num_valid == 3
        assert stats.num_samples == 3
        assert stats.task_counts["relations"] == 2
        assert stats.task_counts["ner_family"] == 2

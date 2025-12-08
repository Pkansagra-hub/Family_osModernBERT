"""Configuration sanity checks for FamilyOS unified dataset config."""

# ruff: noqa: I001
from pathlib import Path

import pytest
from omegaconf import OmegaConf


CONFIG_PATH = Path("configs/data/multitask/familyos_unified.yaml")


@pytest.fixture(scope="module")
def familyos_config():
    if not CONFIG_PATH.exists():
        pytest.skip("familyos_unified config file is missing")
    return OmegaConf.load(CONFIG_PATH)


def test_familyos_unified_metadata(familyos_config):
    dataset = familyos_config.get("dataset")
    assert dataset is not None
    assert dataset.get("name") == "familyos_unified"
    assert dataset.get("format") == "jsonl"

    paths = familyos_config.get("paths")
    assert paths is not None
    assert "data_dir" in paths
    assert "data_dirs" in paths
    assert "shard_pattern" in paths
    assert paths["shard_pattern"].startswith("shard_")


def test_familyos_unified_tasks(familyos_config):
    tasks = familyos_config.get("tasks")
    assert tasks is not None

    expected_tasks = {
        "emotions",
        "sentiment",
        "safety_familyos",
        "intent",
        "ingress",
        "ner_family",
        "temporal",
        "relations",
    }
    assert set(tasks.keys()) == expected_tasks

    assert tasks.emotions.num_labels == 44
    assert len(tasks.emotions.labels) == 44

    assert tasks.sentiment.num_labels == 5
    assert len(tasks.sentiment.labels) == 5

    assert tasks.safety_familyos.num_labels == 4
    assert len(tasks.safety_familyos.labels) == 4

    assert tasks.intent.num_labels == 8
    assert len(tasks.intent.labels) == 8

    assert tasks.ingress.num_labels == 12
    assert len(tasks.ingress.labels) == 12

    assert tasks.ner_family.num_labels == 21
    assert len(tasks.ner_family.labels) == 21

    assert tasks.temporal.num_labels == 13
    assert len(tasks.temporal.labels) == 13

    assert tasks.relations.num_predicates == 15
    assert "no_relation" in tasks.relations.predicates


def test_hub_and_collation_settings(familyos_config):
    routing = familyos_config.get("hub_routing")
    assert routing is not None
    assert "hub_to_tasks" in routing
    assert "loss_weighting" in routing

    collation = familyos_config.get("collation")
    assert collation is not None
    assert collation.get("collator_type") == "v3_multitask"
    hub_positions = collation.get("hub_token_positions")
    assert hub_positions == {"CLS": 0, "EMO": 1, "MEM": 2, "REL": 3, "TASK": 4}

    preprocessing = familyos_config.get("preprocessing")
    assert preprocessing is not None
    assert preprocessing.get("hub_tokens") == ["[EMO]", "[MEM]", "[REL]", "[TASK]"]


def test_shard_loading_settings(familyos_config):
    shard_loading = familyos_config.get("shard_loading")
    assert shard_loading is not None
    assert shard_loading.get("streaming") is True
    assert shard_loading.get("buffered") is True
    assert shard_loading.get("buffer_size") >= 1000
    assert shard_loading.get("prefetch_shards") >= 1

    sampling = familyos_config.get("sampling")
    assert sampling is not None
    assert sampling.get("strategy") == "hub_weighted"
    assert "task_weights" in sampling

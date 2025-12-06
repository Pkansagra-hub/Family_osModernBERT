from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "prepare_healing_data.py"
ENHANCED_MODULE_PATH = ROOT / "scripts" / "prepare_healing_data_enhanced.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_healing_data", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load prepare_healing_data module")

    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_healing_data"] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


prep = _load_prepare_module()
HEALING_CONFIG = prep.HEALING_CONFIG
prepare_healing_data = prep.prepare_healing_data
convert_conll_sample = prep.convert_conll_sample
save_healing_data = prep.save_healing_data
validate_healing_data = prep.validate_healing_data


def _load_enhanced_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_healing_data_enhanced", ENHANCED_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load prepare_healing_data_enhanced module")

    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_healing_data_enhanced"] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


prep_enh = _load_enhanced_module()
ENH_CONFIG = prep_enh.ENHANCED_HEALING_CONFIG
prepare_enhanced_healing_data = prep_enh.prepare_enhanced_healing_data
save_enhanced_healing_data = prep_enh.save_enhanced_healing_data
validate_enhanced_healing_data = prep_enh.validate_enhanced_healing_data


class FakeDataset:
    """Minimal HuggingFace-like dataset for testing."""

    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def shuffle(self, seed: int):
        rng = random.Random(seed)
        shuffled = list(self.data)
        rng.shuffle(shuffled)
        return FakeDataset(shuffled)

    def select(self, indices):
        selected = [self.data[i] for i in indices]
        return FakeDataset(selected)


@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Disable tqdm progress bars during tests."""

    class DummyTqdm:
        def __call__(self, iterable, *args, **kwargs):
            return iterable

    monkeypatch.setattr(prep, "tqdm", DummyTqdm())
    monkeypatch.setattr(prep_enh, "tqdm", DummyTqdm())


def test_prepare_healing_data_counts(monkeypatch):
    """prepare_healing_data should honor configured sample counts."""

    def fake_load_dataset(name, subset=None, split=None):  # pylint: disable=unused-argument
        if name == "glue" and subset == "sst2":
            data = [
                {"sentence": f"sent {i}", "label": i % 2}
                for i in range(HEALING_CONFIG["sst2"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "glue" and subset == "mnli":
            data = [
                {
                    "premise": f"premise {i}",
                    "hypothesis": f"hyp {i}",
                    "label": i % 3,
                }
                for i in range(HEALING_CONFIG["mnli"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "conll2003":
            data = [
                {
                    "tokens": ["John", "lives", "in", "London"],
                    "ner_tags": [1, 2, 0, 5],
                }
                for _ in range(HEALING_CONFIG["conll"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        raise ValueError(f"Unexpected dataset request: {name}, subset={subset}, split={split}")

    monkeypatch.setattr(prep, "load_dataset", fake_load_dataset)

    healing_data = prepare_healing_data(seed=123)

    assert len(healing_data["sst2"]) == HEALING_CONFIG["sst2"]["n_samples"]
    assert len(healing_data["conll"]) == HEALING_CONFIG["conll"]["n_samples"]
    assert len(healing_data["mnli"]) == HEALING_CONFIG["mnli"]["n_samples"]


def test_conll_spans_and_tags():
    """CoNLL conversion should include both tag strings and spans."""

    sample = {"tokens": ["John", "lives", "in", "London"], "ner_tags": [1, 2, 0, 5]}
    config = HEALING_CONFIG["conll"]

    converted = convert_conll_sample(sample, config)

    assert converted["labels"]["ner_tags"] == ["B-PER", "I-PER", "O", "B-LOC"]
    assert converted["labels"]["ner_tag_ids"] == [1, 2, 0, 5]
    assert converted["labels"]["ner_spans"] == [
        {"start": 0, "end": 2, "label": "PER"},
        {"start": 3, "end": 4, "label": "LOC"},
    ]


def test_save_and_validate_split_by_task(tmp_path):
    """Saving per-task files should validate successfully with expected override."""

    healing_data = {
        "sst2": [
            {
                "text": "a",
                "task": "sentiment",
                "task_type": "classification",
                "labels": {},
                "source": "sst2",
                "split": "healing",
            }
            for _ in range(3)
        ],
        "conll": [
            {
                "text": "b",
                "task": "ner",
                "task_type": "token_classification",
                "labels": {},
                "source": "conll",
                "split": "healing",
            }
            for _ in range(3)
        ],
    }

    output_dir = tmp_path / "healing"
    save_healing_data(healing_data, str(output_dir), split_by_task=True, seed=0)

    expected_files = {"healing_sst2.jsonl", "healing_conll.jsonl"}
    assert expected_files == {path.name for path in output_dir.glob("*.jsonl")}

    total_expected = sum(len(samples) for samples in healing_data.values())
    assert validate_healing_data(str(output_dir), expected_total=total_expected)

    # Spot check file contents
    with (output_dir / "healing_sst2.jsonl").open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 3
    first_record = json.loads(lines[0])
    assert first_record["source"] == "sst2"


def test_prepare_enhanced_healing_data_counts(monkeypatch):
    """Enhanced healing data should cover all five tasks with configured counts."""

    def fake_load_dataset(name, subset=None, split=None):  # pylint: disable=unused-argument
        if name == "glue" and subset == "sst2":
            data = [
                {"sentence": f"sent {i}", "label": i % 2}
                for i in range(ENH_CONFIG["sst2"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "glue" and subset == "mnli":
            data = [
                {
                    "premise": f"premise {i}",
                    "hypothesis": f"hyp {i}",
                    "label": i % 3,
                }
                for i in range(ENH_CONFIG["mnli"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "conll2003":
            data = [
                {
                    "tokens": ["John", "lives", "in", "London"],
                    "ner_tags": [1, 2, 0, 5],
                }
                for _ in range(ENH_CONFIG["conll"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "squad":
            data = [
                {
                    "context": "Paris is in France.",
                    "question": "Where is Paris?",
                    "answers": {"text": ["France"], "answer_start": [12]},
                }
                for _ in range(ENH_CONFIG["squad"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        if name == "glue" and subset == "stsb":
            data = [
                {
                    "sentence1": f"s1 {i}",
                    "sentence2": f"s2 {i}",
                    "label": float(i % 6),
                }
                for i in range(ENH_CONFIG["stsb"]["n_samples"] + 5)
            ]
            return FakeDataset(data)
        raise ValueError(f"Unexpected dataset request: {name}, subset={subset}, split={split}")

    monkeypatch.setattr(prep_enh, "load_dataset", fake_load_dataset)

    healing_data = prepare_enhanced_healing_data(seed=7)

    assert set(healing_data.keys()) == set(ENH_CONFIG.keys())
    for task in ENH_CONFIG:
        assert len(healing_data[task]) == ENH_CONFIG[task]["n_samples"]
        assert all("healing_purpose" in sample for sample in healing_data[task])

    squad_sample = healing_data["squad"][0]
    assert squad_sample["labels"]["answer_end"] == squad_sample["labels"]["answer_start"] + len(
        squad_sample["labels"]["answer_text"]
    )

    stsb_sample = healing_data["stsb"][0]
    assert 0.0 <= stsb_sample["labels"]["normalized_score"] <= 1.0


def test_save_and_validate_enhanced_split_by_task(tmp_path):
    """Enhanced save and validate should succeed with per-task files."""

    healing_data = {
        "sst2": [
            {
                "text": "a",
                "task": "sentiment",
                "task_type": "classification",
                "labels": {},
                "source": "sst2",
                "split": "healing",
                "healing_purpose": "p",
            }
            for _ in range(2)
        ],
        "conll": [
            {
                "text": "b",
                "task": "ner",
                "task_type": "token_classification",
                "labels": {},
                "source": "conll",
                "split": "healing",
                "healing_purpose": "p",
            }
            for _ in range(2)
        ],
        "mnli": [
            {
                "text": "c",
                "task": "nli",
                "task_type": "classification",
                "labels": {},
                "source": "mnli",
                "split": "healing",
                "healing_purpose": "p",
            }
            for _ in range(2)
        ],
        "squad": [
            {
                "text": "d",
                "task": "qa",
                "task_type": "span_extraction",
                "labels": {},
                "source": "squad",
                "split": "healing",
                "healing_purpose": "p",
            }
            for _ in range(2)
        ],
        "stsb": [
            {
                "text": "e",
                "task": "similarity",
                "task_type": "regression",
                "labels": {},
                "source": "stsb",
                "split": "healing",
                "healing_purpose": "p",
            }
            for _ in range(2)
        ],
    }

    output_dir = tmp_path / "healing_enh"
    save_enhanced_healing_data(healing_data, str(output_dir), split_by_task=True, seed=0)

    expected_files = {
        "healing_enhanced_sst2.jsonl",
        "healing_enhanced_conll.jsonl",
        "healing_enhanced_mnli.jsonl",
        "healing_enhanced_squad.jsonl",
        "healing_enhanced_stsb.jsonl",
    }
    assert expected_files == {path.name for path in output_dir.glob("*.jsonl")}

    total_expected = sum(len(samples) for samples in healing_data.values())
    assert validate_enhanced_healing_data(str(output_dir), expected_total=total_expected)

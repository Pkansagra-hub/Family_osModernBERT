"""
Prepare healing data for ModernBERT v3 Phase 0.5.

Downloads and preprocesses SST-2, CoNLL-2003, and MNLI to a unified JSONL
format used for healing the cloned layers.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Healing data configuration
HEALING_CONFIG: dict[str, dict[str, Any]] = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "split": "train",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "n_samples": 3000,
    },
    "conll": {
        "hf_name": "conll2003",
        "hf_subset": None,
        "split": "train",
        "text_field": "tokens",
        "label_field": "ner_tags",
        "label_map": {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        },
        "n_samples": 3000,
    },
    "mnli": {
        "hf_name": "glue",
        "hf_subset": "mnli",
        "split": "train",
        "text_field": "premise",
        "text_field_pair": "hypothesis",
        "label_field": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "n_samples": 4000,
    },
}

TOTAL_SAMPLES = sum(cfg["n_samples"] for cfg in HEALING_CONFIG.values())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Prepare healing data for v3 training")
    parser.add_argument(
        "--output",
        type=str,
        default="data/healing/healing_generic.jsonl",
        help="Output file or directory",
    )
    parser.add_argument(
        "--split-by-task",
        action="store_true",
        help="Create separate files per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output after creation",
    )
    return parser.parse_args()


def load_and_sample_dataset(
    config: dict[str, Any], n_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Load dataset from HuggingFace and sample examples.

    Args:
        config: Healing configuration for the task.
        n_samples: Number of samples to select.
        seed: Random seed for shuffling.

    Returns:
        List of raw samples.
    """

    logger.info("Loading %s...", config["hf_name"])

    if config["hf_subset"]:
        dataset = load_dataset(config["hf_name"], config["hf_subset"], split=config["split"])
    else:
        dataset = load_dataset(config["hf_name"], split=config["split"])

    dataset_length = len(cast(Sized, dataset))
    if dataset_length > n_samples:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    elif dataset_length < n_samples:
        logger.warning(
            "Dataset smaller than requested: %d available vs %d requested",
            dataset_length,
            n_samples,
        )

    return list(dataset)


def convert_sst2_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert SST-2 sample to unified format."""

    label_id = sample[config["label_field"]]
    label_name = config["label_map"][label_id]

    return {
        "text": sample[config["text_field"]],
        "task": "sentiment",
        "task_type": "classification",
        "labels": {"sentiment": int(label_id), "sentiment_label": label_name},
        "source": "sst2",
        "split": "healing",
    }


def _bio_tags_to_spans(
    tokens: list[str], tag_ids: list[int], label_map: dict[int, str]
) -> list[dict[str, Any]]:
    """Convert BIO tag sequence to span annotations."""

    spans: list[dict[str, Any]] = []
    current_label: str | None = None
    start: int | None = None

    for idx, tag_id in enumerate(tag_ids):
        tag = label_map[tag_id]
        if tag.startswith("B-"):
            if current_label is not None and start is not None:
                spans.append({"start": start, "end": idx, "label": current_label})
            current_label = tag[2:]
            start = idx
        elif tag.startswith("I-"):
            inner_label = tag[2:]
            if current_label is None:
                current_label = inner_label
                start = idx
            elif inner_label != current_label and start is not None:
                spans.append({"start": start, "end": idx, "label": current_label})
                current_label = inner_label
                start = idx
        else:
            if current_label is not None and start is not None:
                spans.append({"start": start, "end": idx, "label": current_label})
            current_label = None
            start = None

    if current_label is not None and start is not None:
        spans.append({"start": start, "end": len(tokens), "label": current_label})

    return spans


def convert_conll_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert CoNLL-2003 sample to unified format."""

    tokens = sample[config["text_field"]]
    ner_tags = sample[config["label_field"]]

    spans = _bio_tags_to_spans(tokens, ner_tags, config["label_map"])
    tag_strings = [config["label_map"][tag_id] for tag_id in ner_tags]

    return {
        "text": " ".join(tokens),
        "tokens": tokens,
        "task": "ner",
        "task_type": "token_classification",
        "labels": {
            "ner_tag_ids": [int(tag_id) for tag_id in ner_tags],
            "ner_tags": tag_strings,
            "ner_spans": spans,
        },
        "source": "conll",
        "split": "healing",
    }


def convert_mnli_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert MNLI sample to unified format."""

    label_id = sample[config["label_field"]]
    label_name = config["label_map"][label_id]
    premise = sample[config["text_field"]]
    hypothesis = sample[config["text_field_pair"]]

    return {
        "text": f"{premise} [SEP] {hypothesis}",
        "premise": premise,
        "hypothesis": hypothesis,
        "task": "nli",
        "task_type": "classification",
        "labels": {"nli": int(label_id), "nli_label": label_name},
        "source": "mnli",
        "split": "healing",
    }


CONVERTERS = {
    "sst2": convert_sst2_sample,
    "conll": convert_conll_sample,
    "mnli": convert_mnli_sample,
}


def prepare_healing_data(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Prepare all healing datasets."""

    random.seed(seed)
    healing_data: dict[str, list[dict[str, Any]]] = {}

    for task_name, config in HEALING_CONFIG.items():
        logger.info("\n%s", "=" * 50)
        logger.info("Preparing task: %s", task_name)
        samples = load_and_sample_dataset(config, config["n_samples"], seed=seed)
        converter = CONVERTERS[task_name]
        converted = [
            converter(sample, config) for sample in tqdm(samples, desc=f"Convert {task_name}")
        ]
        healing_data[task_name] = converted
        logger.info("  Converted %d samples", len(converted))

    return healing_data


def _write_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    """Write samples to JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_healing_data(
    healing_data: dict[str, list[dict[str, Any]]],
    output_path: str,
    split_by_task: bool = False,
    seed: int | None = None,
) -> None:
    """Save healing data to disk.

    Args:
        healing_data: Dictionary of task name to list of samples.
        output_path: Target file or directory.
        split_by_task: Whether to save per task.
        seed: Optional seed to shuffle output deterministically.
    """

    output = Path(output_path)
    rng = random.Random(seed) if seed is not None else random

    if split_by_task:
        output.mkdir(parents=True, exist_ok=True)
        for task_name, samples in healing_data.items():
            task_samples = list(samples)
            rng.shuffle(task_samples)
            file_path = output / f"healing_{task_name}.jsonl"
            _write_jsonl(file_path, task_samples)
            logger.info("Saved %d samples to %s", len(task_samples), file_path)
    else:
        all_samples: list[dict[str, Any]] = []
        for samples in healing_data.values():
            all_samples.extend(samples)
        rng.shuffle(all_samples)
        _write_jsonl(output, all_samples)
        logger.info("Saved %d samples to %s", len(all_samples), output)


def validate_healing_data(output_path: str, expected_total: int | None = None) -> bool:
    """Validate the created healing data.

    Args:
        output_path: File or directory containing healing data.
        expected_total: Optional override for expected total samples.

    Returns:
        True if validation passes, otherwise False.
    """

    output = Path(output_path)
    expected = expected_total if expected_total is not None else TOTAL_SAMPLES

    if output.is_dir():
        files = list(output.glob("healing_*.jsonl"))
    else:
        files = [output]

    if not files:
        logger.error("No healing data files found at %s", output_path)
        return False

    total_samples = 0
    task_counts: dict[str, int] = defaultdict(int)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                total_samples += 1
                task_key = sample.get("task") or sample.get("source", "unknown")
                task_counts[task_key] += 1

    logger.info("\n%s", "=" * 50)
    logger.info("Validation Results:")
    logger.info("=" * 50)
    logger.info("Total samples: %d", total_samples)
    for task, count in sorted(task_counts.items()):
        logger.info("  %s: %d", task, count)

    if total_samples < int(expected * 0.95):
        logger.warning("Lower than expected sample count: %d < %d", total_samples, expected)
        return False

    logger.info("Validation passed")
    return True


def main() -> int:
    """Main entry point."""

    args = parse_args()

    print("\n" + "=" * 60)
    print("ModernBERT v3 Healing Data Preparation")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Split by task: {args.split_by_task}")
    print(f"Seed: {args.seed}")
    print()

    healing_data = prepare_healing_data(seed=args.seed)

    save_healing_data(healing_data, args.output, split_by_task=args.split_by_task, seed=args.seed)

    if args.validate:
        if not validate_healing_data(args.output):
            logger.error("Validation failed")
            return 1

    print("\n" + "=" * 60)
    print("Healing data preparation complete!")
    print("=" * 60)

    total = sum(len(samples) for samples in healing_data.values())
    print(f"\nTotal samples: {total}")
    for task, samples in healing_data.items():
        print(f"  {task}: {len(samples)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

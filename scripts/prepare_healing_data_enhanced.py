"""
Prepare ENHANCED healing data for ModernBERT v3 Phase 0.5.

Extends the basic healing mix with SQuAD (QA) and STS-B (similarity)
to heal long-range attention and prevent embedding collapse.
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


ENHANCED_HEALING_CONFIG: dict[str, dict[str, Any]] = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "split": "train",
        "n_samples": 3000,
        "task_type": "sentiment",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "purpose": "Sentiment classification - core capability",
    },
    "conll": {
        "hf_name": "conll2003",
        "hf_subset": None,
        "split": "train",
        "n_samples": 3000,
        "task_type": "ner",
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
        "purpose": "NER structural grounding - preserves token understanding",
    },
    "mnli": {
        "hf_name": "glue",
        "hf_subset": "mnli",
        "split": "train",
        "n_samples": 2000,
        "task_type": "nli",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "purpose": "NLI logic/reasoning - preserves inference capability",
    },
    "squad": {
        "hf_name": "squad",
        "hf_subset": None,
        "split": "train",
        "n_samples": 2000,
        "task_type": "qa",
        "context_field": "context",
        "question_field": "question",
        "answer_field": "answers",
        "purpose": "QA context understanding - heals long-range attention",
    },
    "stsb": {
        "hf_name": "glue",
        "hf_subset": "stsb",
        "split": "train",
        "n_samples": 2000,
        "task_type": "similarity",
        "text_field": ["sentence1", "sentence2"],
        "label_field": "label",
        "purpose": "Semantic similarity - prevents embedding collapse",
    },
}

TOTAL_ENHANCED_SAMPLES = sum(cfg["n_samples"] for cfg in ENHANCED_HEALING_CONFIG.values())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Prepare enhanced healing data for v3 training")
    parser.add_argument(
        "--output",
        type=str,
        default="data/healing/healing_enhanced.jsonl",
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
    parser.add_argument(
        "--include-basic",
        action="store_true",
        default=True,
        help="Include basic tasks (SST-2, CoNLL, MNLI)",
    )
    return parser.parse_args()


def load_and_sample_dataset(config: dict[str, Any], n_samples: int, seed: int) -> list[dict[str, Any]]:
    """Load dataset from HuggingFace and sample examples."""

    logger.info("Loading %s...", config["hf_name"])

    if config["hf_subset"]:
        dataset = load_dataset(config["hf_name"], config["hf_subset"], split=config["split"])
    else:
        dataset = load_dataset(config["hf_name"], split=config["split"])

    dataset_length = len(cast(Sized, dataset))
    if dataset_length > n_samples:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    elif dataset_length < n_samples:
        logger.warning("Dataset smaller than requested: %d available vs %d requested", dataset_length, n_samples)

    return list(dataset)


def _bio_tags_to_spans(tokens: list[str], tag_ids: list[int], label_map: dict[int, str]) -> list[dict[str, Any]]:
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
        "healing_purpose": config["purpose"],
    }


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
        "healing_purpose": config["purpose"],
    }


def convert_mnli_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert MNLI sample to unified format."""

    label_id = sample[config["label_field"]]
    label_name = config["label_map"][label_id]
    premise = sample[config["text_field"][0]]
    hypothesis = sample[config["text_field"][1]]

    return {
        "text": f"{premise} [SEP] {hypothesis}",
        "premise": premise,
        "hypothesis": hypothesis,
        "task": "nli",
        "task_type": "classification",
        "labels": {"nli": int(label_id), "nli_label": label_name},
        "source": "mnli",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


def convert_squad_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert SQuAD sample to unified format with span answers."""

    context = sample[config["context_field"]]
    question = sample[config["question_field"]]
    answers = sample[config["answer_field"]]

    if answers.get("text"):
        answer_text = answers["text"][0]
        answer_start = answers.get("answer_start", [-1])[0]
    else:
        answer_text = ""
        answer_start = -1

    return {
        "text": f"{question} [SEP] {context}",
        "question": question,
        "context": context,
        "task": "qa",
        "task_type": "span_extraction",
        "labels": {
            "answer_text": answer_text,
            "answer_start": answer_start,
            "answer_end": answer_start + len(answer_text) if answer_start >= 0 else -1,
        },
        "source": "squad",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


def convert_stsb_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert STS-B sample to unified format with normalized score."""

    sentence1 = sample[config["text_field"][0]]
    sentence2 = sample[config["text_field"][1]]
    score = float(sample[config["label_field"]])
    normalized_score = score / 5.0

    return {
        "text": f"{sentence1} [SEP] {sentence2}",
        "sentence1": sentence1,
        "sentence2": sentence2,
        "task": "similarity",
        "task_type": "regression",
        "labels": {"similarity_score": score, "normalized_score": normalized_score},
        "source": "stsb",
        "split": "healing",
        "healing_purpose": config["purpose"],
    }


CONVERTERS = {
    "sst2": convert_sst2_sample,
    "conll": convert_conll_sample,
    "mnli": convert_mnli_sample,
    "squad": convert_squad_sample,
    "stsb": convert_stsb_sample,
}


def prepare_enhanced_healing_data(seed: int = 42, include_basic: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Prepare all enhanced healing datasets."""

    random.seed(seed)
    healing_data: dict[str, list[dict[str, Any]]] = {}

    for task_name, config in ENHANCED_HEALING_CONFIG.items():
        if not include_basic and task_name in {"sst2", "conll", "mnli"}:
            continue

        logger.info("\n%s", "=" * 50)
        logger.info("Processing %s...", task_name)
        logger.info("Purpose: %s", config["purpose"])
        logger.info("%s", "=" * 50)

        raw_samples = load_and_sample_dataset(config, config["n_samples"], seed)
        converter = CONVERTERS[task_name]
        converted = [converter(sample, config) for sample in tqdm(raw_samples, desc=f"Convert {task_name}")]
        healing_data[task_name] = converted
        logger.info("  Converted %d samples", len(converted))

    return healing_data


def _write_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    """Write samples to JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_enhanced_healing_data(
    healing_data: dict[str, list[dict[str, Any]]],
    output_path: str,
    split_by_task: bool = False,
    seed: int | None = None,
) -> None:
    """Save enhanced healing data to disk."""

    output = Path(output_path)
    rng = random.Random(seed) if seed is not None else random

    if split_by_task:
        output.mkdir(parents=True, exist_ok=True)
        for task_name, samples in healing_data.items():
            task_samples = list(samples)
            rng.shuffle(task_samples)
            file_path = output / f"healing_enhanced_{task_name}.jsonl"
            _write_jsonl(file_path, task_samples)
            logger.info("Saved %d samples to %s", len(task_samples), file_path)
    else:
        all_samples: list[dict[str, Any]] = []
        for samples in healing_data.values():
            all_samples.extend(samples)
        rng.shuffle(all_samples)
        _write_jsonl(output, all_samples)
        logger.info("Saved %d samples to %s", len(all_samples), output)


def validate_enhanced_healing_data(output_path: str, expected_total: int | None = None) -> bool:
    """Validate the created enhanced healing data."""

    output = Path(output_path)

    if output.is_dir():
        files = list(output.glob("healing_enhanced_*.jsonl"))
    else:
        files = [output]

    if not files:
        logger.error("No enhanced healing data files found at %s", output_path)
        return False

    total_samples = 0
    task_counts: dict[str, int] = defaultdict(int)
    task_types: dict[str, set[str]] = defaultdict(set)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                total_samples += 1
                task_counts[sample.get("task", "unknown")] += 1
                task_types[sample.get("task", "unknown")].add(sample.get("task_type", "unknown"))

    logger.info("\n%s", "=" * 50)
    logger.info("Enhanced Healing Data Validation")
    logger.info("=" * 50)
    logger.info("Total samples: %d", total_samples)
    for task, count in sorted(task_counts.items()):
        types = ", ".join(sorted(task_types[task]))
        logger.info("  %s: %d (%s)", task, count, types)

    expected_tasks = {"sentiment", "ner", "nli", "qa", "similarity"}
    if expected_tasks != set(task_counts.keys()):
        missing = expected_tasks - set(task_counts.keys())
        if missing:
            logger.error("Missing tasks: %s", ", ".join(sorted(missing)))
            return False

    expected = expected_total if expected_total is not None else TOTAL_ENHANCED_SAMPLES
    if total_samples < int(expected * 0.95):
        logger.warning("Lower than expected sample count: %d < %d", total_samples, expected)
        return False

    logger.info("Validation passed")
    return True


def main() -> int:
    """Main entry point."""

    args = parse_args()

    print("\n" + "=" * 60)
    print("ModernBERT v3 ENHANCED Healing Data Preparation")
    print("=" * 60)
    print(f"Output: {args.output}")
    print("Tasks: SST-2, CoNLL, MNLI, SQuAD, STS-B")
    print(f"Total samples: ~{TOTAL_ENHANCED_SAMPLES}")
    print()

    healing_data = prepare_enhanced_healing_data(seed=args.seed, include_basic=args.include_basic)

    save_enhanced_healing_data(
        healing_data,
        args.output,
        split_by_task=args.split_by_task,
        seed=args.seed,
    )

    if args.validate:
        if not validate_enhanced_healing_data(args.output):
            return 1

    print("\n" + "=" * 60)
    print("ENHANCED healing data preparation complete!")
    print("=" * 60)

    total = sum(len(samples) for samples in healing_data.values())
    print(f"\nTotal samples: {total}")
    for task, samples in healing_data.items():
        purpose = ENHANCED_HEALING_CONFIG[task]["purpose"]
        print(f"  {task}: {len(samples)} - {purpose}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

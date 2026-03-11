#!/usr/bin/env python
"""Create a deterministic FamilyOS retrieval benchmark from held-out slice data.

This script bootstraps an initial retrieval benchmark from the same mined Stage B
sources used for training, but writes a dedicated benchmark split under
``data/familyos/benchmarks`` so checkpoint comparison can happen on stable,
repeatable data.

Important:
    This is a bootstrap benchmark, not a fully human-reviewed gold set. The
    output is intended for model comparison and promotion gating until manual
    review hardens it into a true gold benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


BENCHMARK_VERSION = "retrieval_golden_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "familyos" / "benchmarks" / BENCHMARK_VERSION

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "query_doc": {
        "path": PROJECT_ROOT / "data" / "familyos" / "embeddings" / "mined_v2" / "query_doc",
        "format": "pair",
        "dev_count": 300,
        "holdout_count": 300,
    },
    "hard_negatives": {
        "path": PROJECT_ROOT / "data" / "familyos" / "embeddings" / "hard_negatives",
        "format": "triplet",
        "dev_count": 100,
        "holdout_count": 100,
    },
    "wrong_person": {
        "path": PROJECT_ROOT / "data" / "familyos" / "embeddings" / "mined_v2" / "wrong_person",
        "format": "triplet",
        "dev_count": 100,
        "holdout_count": 100,
    },
    "wrong_time": {
        "path": PROJECT_ROOT / "data" / "familyos" / "embeddings" / "mined_v2" / "wrong_time",
        "format": "triplet",
        "dev_count": 100,
        "holdout_count": 100,
    },
    "safety_emotion": {
        "path": PROJECT_ROOT / "data" / "familyos" / "embeddings" / "mined_v2" / "safety_emotion",
        "format": "triplet",
        "dev_count": 100,
        "holdout_count": 100,
    },
}

SKIP_PREFIXES = ("hash_index", "manifest", "stats", "metadata")


def stable_digest(payload: str) -> str:
    """Return a stable short digest for deterministic ordering and ids."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_jsonl_files(source_dir: Path) -> list[Path]:
    """Return JSONL files for a source directory in deterministic order."""
    files = [
        path for path in sorted(source_dir.glob("*.jsonl"))
        if not any(path.name.startswith(prefix) for prefix in SKIP_PREFIXES)
    ]
    if files:
        return files
    return [
        path for path in sorted(source_dir.glob("**/*.jsonl"))
        if not any(path.name.startswith(prefix) for prefix in SKIP_PREFIXES)
    ]


def normalize_pair_record(raw: dict[str, Any], slice_name: str, source_file: Path) -> dict[str, Any] | None:
    """Normalize a query-document pair into benchmark format."""
    query = raw.get("query", "")
    document = raw.get("document", "")
    if not query or not document:
        return None

    payload = json.dumps(
        {
            "slice": slice_name,
            "query": query,
            "document": document,
            "query_id": raw.get("query_id"),
            "document_id": raw.get("document_id"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    record_id = stable_digest(payload)
    return {
        "benchmark_id": f"{BENCHMARK_VERSION}_{slice_name}_{record_id[:12]}",
        "benchmark_version": BENCHMARK_VERSION,
        "task_type": "pair",
        "slice": slice_name,
        "query": query,
        "positive": document,
        "negative": None,
        "pair_type": raw.get("pair_type", "unknown"),
        "difficulty": raw.get("difficulty", "unknown"),
        "slice_tags": raw.get("slice_tags", []),
        "shared_features": raw.get("shared_features", []),
        "query_id": raw.get("query_id"),
        "document_id": raw.get("document_id"),
        "source_ids": raw.get("source_ids", []),
        "source_file": str(source_file),
        "review_status": "bootstrap_unreviewed",
        "hash": record_id,
    }


def normalize_triplet_record(raw: dict[str, Any], slice_name: str, source_file: Path) -> dict[str, Any] | None:
    """Normalize a triplet record into benchmark format."""
    anchor = raw.get("anchor", "")
    positive = raw.get("positive", "")
    negative = raw.get("negative", "")
    if not anchor or not positive or not negative:
        return None

    payload = json.dumps(
        {
            "slice": slice_name,
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "negative_type": raw.get("hard_negative_type"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    record_id = stable_digest(payload)
    return {
        "benchmark_id": f"{BENCHMARK_VERSION}_{slice_name}_{record_id[:12]}",
        "benchmark_version": BENCHMARK_VERSION,
        "task_type": "triplet",
        "slice": slice_name,
        "query": anchor,
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "negative_type": raw.get("hard_negative_type", "hard_negative"),
        "difficulty": raw.get("difficulty", "unknown"),
        "slice_tags": raw.get("slice_tags", []),
        "mismatch_features": raw.get("mismatch_features", []),
        "source_ids": raw.get("source_ids", []),
        "source_file": str(source_file),
        "review_status": "bootstrap_unreviewed",
        "hash": record_id,
    }


def load_source_records(slice_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load and normalize all records for a single slice."""
    source_dir = Path(config["path"])
    record_format = str(config["format"])
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found for slice '{slice_name}': {source_dir}")

    normalized: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    normalizer = normalize_pair_record if record_format == "pair" else normalize_triplet_record

    for jsonl_path in iter_jsonl_files(source_dir):
        with open(jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = normalizer(raw, slice_name, jsonl_path)
                if record is None:
                    continue
                if record["hash"] in seen_hashes:
                    continue
                seen_hashes.add(record["hash"])
                normalized.append(record)

    normalized.sort(key=lambda item: item["hash"])
    return normalized


def assign_splits(records: list[dict[str, Any]], dev_count: int, holdout_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign deterministic dev and holdout slices from a sorted record list."""
    required = dev_count + holdout_count
    if len(records) < required:
        raise ValueError(
            f"Not enough records to create splits: required {required}, found {len(records)}"
        )

    dev_records = [dict(record, split="dev") for record in records[:dev_count]]
    holdout_records = [dict(record, split="holdout") for record in records[dev_count:required]]
    return dev_records, holdout_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary for a benchmark split."""
    task_counter = Counter(record["task_type"] for record in records)
    slice_counter = Counter(record["slice"] for record in records)
    difficulty_counter = Counter(record.get("difficulty", "unknown") for record in records)
    return {
        "total": len(records),
        "by_task_type": dict(task_counter),
        "by_slice": dict(slice_counter),
        "by_difficulty": dict(difficulty_counter),
    }


def create_benchmark(output_dir: Path) -> dict[str, Any]:
    """Create the retrieval benchmark files and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}

    for slice_name, config in DEFAULT_SOURCES.items():
        records = load_source_records(slice_name, config)
        split_dev, split_holdout = assign_splits(
            records,
            dev_count=int(config["dev_count"]),
            holdout_count=int(config["holdout_count"]),
        )
        dev_records.extend(split_dev)
        holdout_records.extend(split_holdout)
        source_manifest[slice_name] = {
            "path": str(config["path"]),
            "format": config["format"],
            "available_records": len(records),
            "dev_count": len(split_dev),
            "holdout_count": len(split_holdout),
        }

    dev_records.sort(key=lambda item: (item["slice"], item["hash"]))
    holdout_records.sort(key=lambda item: (item["slice"], item["hash"]))

    write_jsonl(output_dir / "dev.jsonl", dev_records)
    write_jsonl(output_dir / "holdout.jsonl", holdout_records)

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_from": "bootstrap_stage_b_sources",
        "review_status": "bootstrap_unreviewed",
        "notes": [
            "This benchmark is deterministic and held out from the selected source slices.",
            "It is suitable for checkpoint comparison, but should receive human review before final promotion gating.",
        ],
        "splits": {
            "dev": summarize_records(dev_records),
            "holdout": summarize_records(holdout_records),
        },
        "sources": source_manifest,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Create bootstrap retrieval golden benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where benchmark files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    manifest = create_benchmark(output_dir=args.output_dir)

    print(f"Created benchmark: {args.output_dir}")
    for split_name, stats in manifest["splits"].items():
        print(f"  {split_name}: total={stats['total']} by_slice={stats['by_slice']}")


if __name__ == "__main__":
    main()
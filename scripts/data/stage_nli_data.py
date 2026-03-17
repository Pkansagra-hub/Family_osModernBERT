"""
Stage NLI + Relevance Training Data for MGRH

Downloads, formats, and splits all data required for Multi-Granularity
Relevance Head training across three stages:

    Stage A (general NLI): MNLI + SNLI + WANLI + FEVER-NLI
        → data/familyos/nli/general/*.jsonl
    Stage B (domain adaptation): symlinks to existing embedding data
        → data/familyos/nli/domain/
    Stage C (relevance fine-tune): human benchmark → listwise JSONL
        → data/familyos/nli/relevance/human_benchmark_listwise.jsonl

Output format (Stage A):
    {"premise": "...", "hypothesis": "...", "label": 0}
    Label mapping: {entailment: 0, neutral: 1, contradiction: 2}

Output format (Stage C):
    {"query": "...", "episodes": [{"text": "...", "grade": 3}, ...]}

Splits: 80% train / 10% dev / 10% holdout, stratified by source.

ANLI is excluded by default (CC-BY-NC-4.0 — commercial blocker).
Pass --include-anli to include it if non-commercial use is confirmed.

Usage:
    python scripts/data/stage_nli_data.py
    python scripts/data/stage_nli_data.py --include-anli
    python scripts/data/stage_nli_data.py --skip-download  # format only, datasets already cached
    python scripts/data/stage_nli_data.py --dry-run        # show plan without executing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root
NLI_DIR = BASE_DIR / "data" / "familyos" / "nli"
GENERAL_DIR = NLI_DIR / "general"
DOMAIN_DIR = NLI_DIR / "domain"
RELEVANCE_DIR = NLI_DIR / "relevance"
SPLITS_DIR = NLI_DIR / "splits"

EMBEDDINGS_DIR = BASE_DIR / "data" / "familyos" / "embeddings"
BENCHMARK_DIR = BASE_DIR / "data" / "familyos" / "benchmarks" / "retrieval_golden_v1"

# Unified label mapping
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}

# Stage A dataset specifications
# License-safe datasets (all CC-BY compatible)
STAGE_A_SAFE_DATASETS: list[dict[str, Any]] = [
    {
        "name": "mnli",
        "hf_path": "nyu-mll/multi_nli",
        "split": "train",
        "output": "mnli_train.jsonl",
        "expected_count": 392_702,
        "license": "CC-BY-3.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        # HF uses int labels: 0=entailment, 1=neutral, 2=contradiction (matches ours)
        "label_remap": None,
    },
    {
        "name": "snli",
        "hf_path": "stanfordnlp/snli",
        "split": "train",
        "output": "snli_train.jsonl",
        "expected_count": 549_367,
        "license": "CC-BY-SA-4.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        # HF uses int labels: 0=entailment, 1=neutral, 2=contradiction
        # Label -1 = unlabeled, skip these
        "label_remap": None,
        "skip_label": -1,
    },
    {
        "name": "wanli",
        "hf_path": "alisawuffles/WANLI",
        "split": "train",
        "output": "wanli_train.jsonl",
        "expected_count": 102_885,
        "license": "CC-BY-4.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "gold",
        # WANLI uses string labels
        "label_remap": {"entailment": 0, "neutral": 1, "contradiction": 2},
    },
    {
        "name": "fever_nli",
        "hf_path": "pietrolesci/nli_fever",
        "split": "train",
        "output": "fever_nli_train.jsonl",
        "expected_count": 152_921,
        "license": "CC-BY-SA-3.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        # pietrolesci/nli_fever uses int labels matching ours
        "label_remap": None,
    },
]

# ANLI — CC-BY-NC-4.0 — only included with --include-anli flag
ANLI_DATASETS: list[dict[str, Any]] = [
    {
        "name": "anli_r1",
        "hf_path": "facebook/anli",
        "split": "train_r1",
        "output": "anli_r1.jsonl",
        "expected_count": 16_946,
        "license": "CC-BY-NC-4.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        "label_remap": None,
    },
    {
        "name": "anli_r2",
        "hf_path": "facebook/anli",
        "split": "train_r2",
        "output": "anli_r2.jsonl",
        "expected_count": 45_460,
        "license": "CC-BY-NC-4.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        "label_remap": None,
    },
    {
        "name": "anli_r3",
        "hf_path": "facebook/anli",
        "split": "train_r3",
        "output": "anli_r3.jsonl",
        "expected_count": 100_459,
        "license": "CC-BY-NC-4.0",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        "label_remap": None,
    },
]

# Stage B domain data — paths relative to EMBEDDINGS_DIR
DOMAIN_SOURCES = [
    {
        "name": "hard_negatives",
        "path": "hard_negatives",
        "types": ["entity_swap", "temporal_shift", "same_topic_different_event", "causality_flip"],
    },
    {"name": "mined_v2_query_doc", "path": "mined_v2/query_doc"},
    {"name": "mined_v2_wrong_time", "path": "mined_v2/wrong_time"},
    {"name": "mined_v2_wrong_person", "path": "mined_v2/wrong_person"},
]

# Split ratios
TRAIN_RATIO = 0.80
DEV_RATIO = 0.10
HOLDOUT_RATIO = 0.10

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Stage A: Download and format general NLI datasets
# ---------------------------------------------------------------------------


def download_and_format_nli_dataset(
    spec: dict[str, Any],
    output_dir: Path,
    skip_download: bool = False,
) -> int:
    """Download a single NLI dataset from HuggingFace and format to unified JSONL.

    Args:
        spec: Dataset specification dict.
        output_dir: Directory to write the formatted JSONL file.
        skip_download: If True, assume HF cache is populated and just load.

    Returns:
        Number of records written.
    """
    output_path = output_dir / spec["output"]
    if output_path.exists():
        count = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info(
            "  [skip] %s already exists (%d records)", spec["name"], count
        )
        return count

    logger.info(
        "  Downloading %s from %s (split=%s) ...",
        spec["name"],
        spec["hf_path"],
        spec["split"],
    )

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "  'datasets' package not installed. "
            "Run: pip install datasets"
        )
        raise

    ds = load_dataset(spec["hf_path"], split=spec["split"])

    premise_key = spec["premise_key"]
    hypothesis_key = spec["hypothesis_key"]
    label_key = spec["label_key"]
    label_remap = spec.get("label_remap")
    skip_label = spec.get("skip_label")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            raw_label = row[label_key]

            # Skip unlabeled rows (e.g. SNLI label=-1)
            if skip_label is not None and raw_label == skip_label:
                continue

            if label_remap is not None:
                label = label_remap.get(raw_label)
                if label is None:
                    continue  # unknown label string, skip
            else:
                label = int(raw_label)

            if label not in (0, 1, 2):
                continue

            record = {
                "premise": row[premise_key],
                "hypothesis": row[hypothesis_key],
                "label": label,
                "source": spec["name"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("  Wrote %d records to %s", count, output_path.name)
    return count


def stage_general_nli(
    include_anli: bool = False,
    skip_download: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Issue 2.1: Download and format all Stage A NLI datasets.

    Args:
        include_anli: Include ANLI R1-R3 (CC-BY-NC-4.0). Default False.
        skip_download: Skip HF download, assume cache populated.
        dry_run: Print plan only, do not execute.

    Returns:
        Dict mapping dataset name → record count.
    """
    datasets = list(STAGE_A_SAFE_DATASETS)
    if include_anli:
        logger.warning(
            "Including ANLI datasets (CC-BY-NC-4.0). "
            "Confirm non-commercial use before training."
        )
        datasets.extend(ANLI_DATASETS)

    logger.info("=== Stage A: General NLI datasets ===")
    logger.info("  Datasets: %s", [d["name"] for d in datasets])
    logger.info("  Output: %s", GENERAL_DIR)

    if dry_run:
        for spec in datasets:
            logger.info(
                "  [dry-run] Would download %s (~%d records, %s)",
                spec["name"],
                spec["expected_count"],
                spec["license"],
            )
        return {d["name"]: d["expected_count"] for d in datasets}

    GENERAL_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for spec in datasets:
        count = download_and_format_nli_dataset(
            spec, GENERAL_DIR, skip_download=skip_download
        )
        counts[spec["name"]] = count

    # Write manifest
    manifest = {
        "stage": "A",
        "description": "General NLI training data for MGRH Stage A",
        "format": "jsonl",
        "schema": {"premise": "str", "hypothesis": "str", "label": "int(0-2)", "source": "str"},
        "label_map": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "datasets": {
            name: {
                "file": spec["output"],
                "count": counts.get(name, 0),
                "license": spec["license"],
            }
            for spec in datasets
            for name in [spec["name"]]
        },
        "anli_included": include_anli,
        "total_records": sum(counts.values()),
    }
    manifest_path = GENERAL_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("  Wrote manifest to %s", manifest_path)

    return counts


# ---------------------------------------------------------------------------
# Stage B: Symlink domain data
# ---------------------------------------------------------------------------


def stage_domain_data(dry_run: bool = False) -> dict[str, str]:
    """Issue 2.2: Create symlinks to existing domain embedding data.

    Stage B data already exists under data/familyos/embeddings/.
    We create symlinks under data/familyos/nli/domain/ for unified access.

    Args:
        dry_run: Print plan only, do not execute.

    Returns:
        Dict mapping source name → symlink path (or target path if dry_run).
    """
    logger.info("=== Stage B: Domain data symlinks ===")

    if dry_run:
        for src in DOMAIN_SOURCES:
            target = EMBEDDINGS_DIR / src["path"]
            logger.info("  [dry-run] Would link %s → %s", src["name"], target)
        return {s["name"]: str(EMBEDDINGS_DIR / s["path"]) for s in DOMAIN_SOURCES}

    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)

    result: dict[str, str] = {}
    for src in DOMAIN_SOURCES:
        target = EMBEDDINGS_DIR / src["path"]
        link = DOMAIN_DIR / src["name"]

        if not target.exists():
            logger.warning("  [skip] Source not found: %s", target)
            continue

        if link.exists() or link.is_symlink():
            logger.info("  [skip] Link already exists: %s", link)
            result[src["name"]] = str(link)
            continue

        # On Windows, use junction for directories (no admin required)
        # On Unix, use regular symlink
        if os.name == "nt":
            # os.symlink with target_is_directory=True works on Windows
            # if Developer Mode is enabled, otherwise fall back to junction
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError:
                # Fall back: just record the path, don't create link
                logger.warning(
                    "  [fallback] Cannot create symlink on Windows without "
                    "Developer Mode. Recording path reference instead."
                )
                _write_path_reference(link, target)
        else:
            os.symlink(target, link)

        logger.info("  Linked %s → %s", link.name, target)
        result[src["name"]] = str(link)

    # Write manifest
    manifest = {
        "stage": "B",
        "description": "Domain NLI data for MGRH Stage B (symlinks to embeddings data)",
        "sources": {
            src["name"]: {
                "target": str(EMBEDDINGS_DIR / src["path"]),
                "types": src.get("types", "all"),
            }
            for src in DOMAIN_SOURCES
        },
        "excluded": ["silver_synthetic (too easy for cross-encoder training)"],
    }
    manifest_path = DOMAIN_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("  Wrote manifest to %s", manifest_path)

    return result


def _write_path_reference(link: Path, target: Path) -> None:
    """Write a .pathref file as a fallback when symlinks are unavailable."""
    ref_path = link.with_suffix(".pathref")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump({"target": str(target), "note": "symlink fallback"}, f)


# ---------------------------------------------------------------------------
# Stage C: Format human benchmark as listwise JSONL
# ---------------------------------------------------------------------------


def _compute_relevance_grade(row: dict[str, Any]) -> int:
    """Compute relevance grade (0-3) from benchmark triplet/pair row.

    Grading heuristic:
        3 = positive document (explicitly relevant)
        2 = same-topic but different difficulty (medium/hard)
        1 = same-topic easy negative (still somewhat related)
        0 = unrelated or hard negative

    For triplet data: positive gets grade 3, negative gets grade based on type.
    For pair data: positive gets grade 3.
    """
    # This is a simplification — the plan mentions 50 queries x 88 episodes
    # with human grades. Since the benchmark has triplets/pairs without
    # explicit 0-3 grades, we derive grades from the structure.
    return -1  # sentinel, handled per-type below


def format_human_benchmark_listwise(dry_run: bool = False) -> int:
    """Issue 2.3: Convert human benchmark to listwise JSONL for LambdaRank.

    Reads the golden benchmark (triplet + pair format) and groups by query
    to produce listwise records with graded relevance.

    Output format:
        {"query": "...", "episodes": [{"text": "...", "grade": 3}, ...]}

    Args:
        dry_run: Print plan only, do not execute.

    Returns:
        Number of query groups written.
    """
    logger.info("=== Stage C: Format human benchmark as listwise ===")

    # Load all benchmark records
    all_records: list[dict[str, Any]] = []
    for split_file in ["dev.jsonl", "holdout.jsonl"]:
        path = BENCHMARK_DIR / split_file
        if not path.exists():
            logger.warning("  Benchmark file not found: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    logger.info("  Loaded %d benchmark records", len(all_records))

    if dry_run:
        queries = {r["query"] for r in all_records}
        logger.info("  [dry-run] Would produce ~%d query groups", len(queries))
        return len(queries)

    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)

    # Grade mapping based on negative_type and task_type
    # Grade 3: positive (gold relevant)
    # Grade 2: same-topic, mild divergence (sentiment_flip, same_topic_different_event)
    # Grade 1: entity/temporal shift (related but wrong referent/time)
    # Grade 0: completely unrelated or cross-cluster easy negative
    HARD_NEG_GRADE = {
        "entity_swap": 1,
        "temporal_shift": 1,
        "same_topic_different_event": 2,
        "sentiment_flip": 2,
        "causality_flip": 1,
        "cross_cluster_easy": 0,
    }

    # Group by query
    query_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in all_records:
        query = rec["query"]

        # Add positive document
        if rec.get("positive"):
            query_groups[query].append({
                "text": rec["positive"],
                "grade": 3,
                "source_type": "positive",
            })

        # Add negative document with derived grade
        if rec.get("negative"):
            neg_type = rec.get("negative_type", "unknown")
            grade = HARD_NEG_GRADE.get(neg_type, 0)
            query_groups[query].append({
                "text": rec["negative"],
                "grade": grade,
                "source_type": neg_type,
            })

    # Deduplicate episodes within each query group
    output_records: list[dict[str, Any]] = []
    for query, episodes in sorted(query_groups.items()):
        seen_texts: set[str] = set()
        unique_episodes: list[dict[str, Any]] = []
        for ep in episodes:
            text_hash = hashlib.md5(ep["text"].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_episodes.append(ep)

        if len(unique_episodes) < 2:
            continue  # need at least 2 docs for listwise ranking

        output_records.append({
            "query": query,
            "episodes": unique_episodes,
        })

    # Write listwise JSONL
    output_path = RELEVANCE_DIR / "human_benchmark_listwise.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_episodes = sum(len(r["episodes"]) for r in output_records)
    logger.info(
        "  Wrote %d query groups (%d episodes) to %s",
        len(output_records),
        total_episodes,
        output_path.name,
    )

    # Grade distribution
    grade_dist: dict[int, int] = defaultdict(int)
    for rec in output_records:
        for ep in rec["episodes"]:
            grade_dist[ep["grade"]] += 1
    logger.info("  Grade distribution: %s", dict(sorted(grade_dist.items())))

    # Write manifest
    manifest = {
        "stage": "C",
        "description": "Human benchmark listwise data for MGRH Stage C (LambdaRank)",
        "format": "listwise_jsonl",
        "schema": {
            "query": "str",
            "episodes": [{"text": "str", "grade": "int(0-3)", "source_type": "str"}],
        },
        "grade_map": {
            "3": "positive (gold relevant)",
            "2": "same-topic mild divergence",
            "1": "entity/temporal shift (related but wrong)",
            "0": "unrelated / hard negative",
        },
        "stats": {
            "query_groups": len(output_records),
            "total_episodes": total_episodes,
            "grade_distribution": dict(sorted(grade_dist.items())),
        },
        "source": str(BENCHMARK_DIR),
    }
    manifest_path = RELEVANCE_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("  Wrote manifest to %s", manifest_path)

    return len(output_records)


# ---------------------------------------------------------------------------
# Splits: train / dev / holdout
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def create_splits(dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Issue 2.4: Create train/dev/holdout splits for all staged data.

    Splits are stratified by source dataset to ensure each split
    has representative coverage of all NLI sources.

    Args:
        dry_run: Print plan only, do not execute.

    Returns:
        Nested dict: {stage: {split_name: record_count}}.
    """
    logger.info("=== Creating train/dev/holdout splits ===")

    rng = random.Random(RANDOM_SEED)
    result: dict[str, dict[str, int]] = {}

    # --- Stage A splits ---
    if GENERAL_DIR.exists():
        logger.info("  Splitting Stage A data ...")
        stage_a_records: dict[str, list[dict[str, Any]]] = {}
        for jsonl_file in sorted(GENERAL_DIR.glob("*.jsonl")):
            records = _load_jsonl(jsonl_file)
            stage_a_records[jsonl_file.stem] = records
            logger.info("    %s: %d records", jsonl_file.stem, len(records))

        if dry_run:
            total = sum(len(v) for v in stage_a_records.values())
            result["stage_a"] = {
                "train": int(total * TRAIN_RATIO),
                "dev": int(total * DEV_RATIO),
                "holdout": int(total * HOLDOUT_RATIO),
            }
        else:
            train_a: list[dict[str, Any]] = []
            dev_a: list[dict[str, Any]] = []
            holdout_a: list[dict[str, Any]] = []

            for source_name, records in stage_a_records.items():
                rng.shuffle(records)
                n = len(records)
                n_dev = max(1, int(n * DEV_RATIO))
                n_holdout = max(1, int(n * HOLDOUT_RATIO))
                n_train = n - n_dev - n_holdout

                train_a.extend(records[:n_train])
                dev_a.extend(records[n_train : n_train + n_dev])
                holdout_a.extend(records[n_train + n_dev :])

            rng.shuffle(train_a)
            rng.shuffle(dev_a)
            rng.shuffle(holdout_a)

            splits_a = SPLITS_DIR / "stage_a"
            _write_jsonl(train_a, splits_a / "train.jsonl")
            _write_jsonl(dev_a, splits_a / "dev.jsonl")
            _write_jsonl(holdout_a, splits_a / "holdout.jsonl")

            result["stage_a"] = {
                "train": len(train_a),
                "dev": len(dev_a),
                "holdout": len(holdout_a),
            }
            logger.info(
                "    Stage A splits: train=%d, dev=%d, holdout=%d",
                len(train_a), len(dev_a), len(holdout_a),
            )

    # --- Stage C splits (listwise — split by query group, not by episode) ---
    listwise_path = RELEVANCE_DIR / "human_benchmark_listwise.jsonl"
    if listwise_path.exists():
        logger.info("  Splitting Stage C listwise data ...")
        records = _load_jsonl(listwise_path)

        if dry_run:
            n = len(records)
            result["stage_c"] = {
                "train": int(n * TRAIN_RATIO),
                "dev": int(n * DEV_RATIO),
                "holdout": int(n * HOLDOUT_RATIO),
            }
        else:
            rng.shuffle(records)
            n = len(records)
            n_dev = max(1, int(n * DEV_RATIO))
            n_holdout = max(1, int(n * HOLDOUT_RATIO))
            n_train = n - n_dev - n_holdout

            splits_c = SPLITS_DIR / "stage_c"
            _write_jsonl(records[:n_train], splits_c / "train.jsonl")
            _write_jsonl(
                records[n_train : n_train + n_dev], splits_c / "dev.jsonl"
            )
            _write_jsonl(records[n_train + n_dev :], splits_c / "holdout.jsonl")

            result["stage_c"] = {
                "train": n_train,
                "dev": n_dev,
                "holdout": n_holdout,
            }
            logger.info(
                "    Stage C splits: train=%d, dev=%d, holdout=%d",
                n_train, n_dev, n_holdout,
            )

    # Note: Stage B data is not split here — it uses the existing embedding
    # splits from data/familyos/embeddings/ which already have train/dev/holdout.

    # Write splits manifest
    if not dry_run:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "description": "Train/dev/holdout splits for MGRH training",
            "ratios": {
                "train": TRAIN_RATIO,
                "dev": DEV_RATIO,
                "holdout": HOLDOUT_RATIO,
            },
            "seed": RANDOM_SEED,
            "splits": result,
            "notes": [
                "Stage A: stratified by source dataset, shuffled",
                "Stage B: uses existing embedding splits (not re-split here)",
                "Stage C: split by query group (not by episode) to prevent leakage",
            ],
        }
        manifest_path = SPLITS_DIR / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage NLI + relevance training data for MGRH",
    )
    parser.add_argument(
        "--include-anli",
        action="store_true",
        help="Include ANLI R1-R3 (CC-BY-NC-4.0). Only if non-commercial use confirmed.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HuggingFace download (assume datasets cached).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing.",
    )
    parser.add_argument(
        "--stage",
        choices=["a", "b", "c", "splits", "all"],
        default="all",
        help="Run a specific stage only (default: all).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("MGRH Data Staging Pipeline")
    logger.info("  Output root: %s", NLI_DIR)
    logger.info("  Include ANLI: %s", args.include_anli)
    logger.info("  Dry run: %s", args.dry_run)
    logger.info("")

    run_stage = args.stage

    # Stage A: General NLI
    if run_stage in ("a", "all"):
        counts_a = stage_general_nli(
            include_anli=args.include_anli,
            skip_download=args.skip_download,
            dry_run=args.dry_run,
        )
        total_a = sum(counts_a.values())
        logger.info("  Stage A total: %d records\n", total_a)

    # Stage B: Domain data
    if run_stage in ("b", "all"):
        stage_domain_data(dry_run=args.dry_run)
        logger.info("")

    # Stage C: Human benchmark listwise
    if run_stage in ("c", "all"):
        n_groups = format_human_benchmark_listwise(dry_run=args.dry_run)
        logger.info("  Stage C: %d query groups\n", n_groups)

    # Splits
    if run_stage in ("splits", "all"):
        splits = create_splits(dry_run=args.dry_run)
        logger.info("  Splits summary: %s\n", json.dumps(splits, indent=2))

    logger.info("Done.")


if __name__ == "__main__":
    main()

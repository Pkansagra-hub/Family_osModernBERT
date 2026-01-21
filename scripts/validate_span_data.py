#!/usr/bin/env python3
"""
Validate Span-Format NER Data for GlobalPointer Training.

This script validates all converted span-format data to ensure quality
before GlobalPointer training. It checks:
    1. Span format correctness (start <= end)
    2. Span text alignment (entity text matches span indices)
    3. Label validity (only PER, ORG, LOC, MISC)
    4. No overlapping spans
    5. Statistics and distribution analysis

Usage:
    python scripts/validate_span_data.py --data-dir data/ner_general_span
    python scripts/validate_span_data.py --data-dir data/ner_general_span --fix

Author: FamilyOS Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.span_utils import validate_spans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VALID_LABELS = {"PER", "ORG", "LOC", "MISC"}


def check_text_alignment(sample: dict[str, Any]) -> list[str]:
    """Check that entity text matches the span in the original text."""
    errors = []
    text = sample.get("text", "")

    for entity in sample.get("entities", []):
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        entity_text = entity.get("text", "")

        # Extract text at span
        actual_text = text[start:end]

        if actual_text != entity_text:
            errors.append(
                f"Text mismatch: expected '{entity_text}' but found '{actual_text}' "
                f"at [{start}:{end}]"
            )

    return errors


def check_overlapping_spans(sample: dict[str, Any]) -> list[str]:
    """Check for overlapping spans."""
    errors = []
    entities = sample.get("entities", [])

    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: (x.get("start", 0), x.get("end", 0)))

    for i in range(len(sorted_entities) - 1):
        current = sorted_entities[i]
        next_entity = sorted_entities[i + 1]

        # Check if current end overlaps with next start
        if current.get("end", 0) > next_entity.get("start", 0):
            errors.append(
                f"Overlapping spans: '{current.get('text', '')}' [{current.get('start')}:{current.get('end')}] "
                f"overlaps with '{next_entity.get('text', '')}' [{next_entity.get('start')}:{next_entity.get('end')}]"
            )

    return errors


def validate_sample(sample: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single sample comprehensively."""
    all_errors = []

    # 1. Basic span validation
    is_valid, span_errors = validate_spans(sample, valid_labels=VALID_LABELS)
    all_errors.extend(span_errors)

    # 2. Text alignment check
    alignment_errors = check_text_alignment(sample)
    all_errors.extend(alignment_errors)

    # 3. Overlapping spans check
    overlap_errors = check_overlapping_spans(sample)
    all_errors.extend(overlap_errors)

    return len(all_errors) == 0, all_errors


def validate_file(file_path: Path, fix: bool = False) -> dict[str, Any]:
    """Validate a single JSONL file."""
    logger.info(f"Validating {file_path}...")

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    error_counts: Counter = Counter()
    label_counts: Counter = Counter()
    entity_counts: Counter = Counter()

    valid_samples_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Validating {file_path.name}"):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                total_samples += 1

                is_valid, errors = validate_sample(sample)

                if is_valid:
                    valid_samples += 1
                    valid_samples_list.append(sample)

                    # Count labels
                    for entity in sample.get("entities", []):
                        label_counts[entity.get("label", "UNKNOWN")] += 1

                    # Count entities per sample
                    n_entities = len(sample.get("entities", []))
                    entity_counts[n_entities] += 1
                else:
                    invalid_samples += 1
                    for error in errors:
                        # Categorize error
                        if "start > end" in error.lower():
                            error_counts["span_order"] += 1
                        elif "overlap" in error.lower():
                            error_counts["overlap"] += 1
                        elif "mismatch" in error.lower():
                            error_counts["text_mismatch"] += 1
                        elif "label" in error.lower():
                            error_counts["invalid_label"] += 1
                        else:
                            error_counts["other"] += 1

            except json.JSONDecodeError as e:
                invalid_samples += 1
                error_counts["json_parse"] += 1
                logger.debug(f"JSON parse error: {e}")

    # If fix mode, rewrite file with only valid samples
    if fix and invalid_samples > 0:
        backup_path = file_path.with_suffix(".jsonl.backup")
        file_path.rename(backup_path)

        with open(file_path, "w", encoding="utf-8") as f:
            for sample in valid_samples_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Fixed {file_path}: removed {invalid_samples} invalid samples")
        logger.info(f"Backup saved to {backup_path}")

    return {
        "file": str(file_path),
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "validity_rate": valid_samples / total_samples if total_samples > 0 else 0,
        "error_breakdown": dict(error_counts),
        "label_distribution": dict(label_counts),
        "entity_distribution": {
            "samples_with_0_entities": entity_counts.get(0, 0),
            "samples_with_1_entity": entity_counts.get(1, 0),
            "samples_with_2_entities": entity_counts.get(2, 0),
            "samples_with_3+_entities": sum(v for k, v in entity_counts.items() if k >= 3),
        },
    }


def validate_directory(data_dir: Path, fix: bool = False) -> dict[str, Any]:
    """Validate all JSONL files in directory."""
    all_results = {}

    # Find all JSONL files
    jsonl_files = list(data_dir.rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No JSONL files found in {data_dir}")
        return {}

    logger.info(f"Found {len(jsonl_files)} JSONL files to validate")

    total_samples = 0
    total_valid = 0
    total_invalid = 0
    combined_labels: Counter = Counter()

    for file_path in jsonl_files:
        result = validate_file(file_path, fix=fix)

        # Use relative path as key
        rel_path = file_path.relative_to(data_dir)
        all_results[str(rel_path)] = result

        total_samples += result["total_samples"]
        total_valid += result["valid_samples"]
        total_invalid += result["invalid_samples"]

        for label, count in result["label_distribution"].items():
            combined_labels[label] += count

    # Summary
    summary = {
        "total_files": len(jsonl_files),
        "total_samples": total_samples,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "overall_validity_rate": total_valid / total_samples if total_samples > 0 else 0,
        "combined_label_distribution": dict(combined_labels),
        "per_file_results": all_results,
    }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("SPAN DATA VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal Files: {summary.get('total_files', 0)}")
    print(f"Total Samples: {summary.get('total_samples', 0):,}")
    print(f"Valid Samples: {summary.get('total_valid', 0):,}")
    print(f"Invalid Samples: {summary.get('total_invalid', 0):,}")
    print(f"Validity Rate: {summary.get('overall_validity_rate', 0):.2%}")

    print("\n" + "-" * 70)
    print("LABEL DISTRIBUTION")
    print("-" * 70)

    label_dist = summary.get("combined_label_distribution", {})
    total_entities = sum(label_dist.values())
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = count / total_entities * 100 if total_entities > 0 else 0
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    print("\n" + "-" * 70)
    print("PER-FILE DETAILS")
    print("-" * 70)

    for file_key, file_result in summary.get("per_file_results", {}).items():
        validity_pct = file_result.get("validity_rate", 0) * 100
        print(f"\n  {file_key}:")
        print(f"    Samples: {file_result.get('total_samples', 0):,}")
        print(f"    Valid: {file_result.get('valid_samples', 0):,} ({validity_pct:.1f}%)")

        errors = file_result.get("error_breakdown", {})
        if errors:
            print(f"    Errors: {dict(errors)}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate span-format NER data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ner_general_span"),
        help="Directory containing span-format JSONL files",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove invalid samples (creates backup)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for detailed results",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    summary = validate_directory(args.data_dir, fix=args.fix)

    # Print summary
    print_summary(summary)

    # Save detailed results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Detailed results saved to {args.output}")

    # Exit with error code if there were invalid samples
    if summary.get("total_invalid", 0) > 0:
        logger.warning(
            f"Found {summary['total_invalid']:,} invalid samples. "
            "Run with --fix to remove them."
        )
        sys.exit(1)

    logger.info("All samples validated successfully!")


if __name__ == "__main__":
    main()

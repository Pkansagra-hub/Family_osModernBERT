#!/usr/bin/env python3
"""
Validate Constitution Coverage in Training Data.

This script analyzes the training data to ensure adequate coverage of
3-layer constitutional contexts for effective training.

FamilyOS Constitutional Layers:
    Layer 1: Family Values - Core family principles (privacy, respect, support)
    Layer 2: Individual Preferences - Per-member boundaries and comfort levels  
    Layer 3: Situational Context - Context-adaptive rules (sensitive topics, emergencies)

Usage:
    python scripts/validate_constitution_coverage.py \
        --input-dir data/counterfactual/training_jsonl

    python scripts/validate_constitution_coverage.py \
        --input-dir data/counterfactual/training_jsonl \
        --output-report constitution_coverage_report.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from modeling_studio.data.constitution_registry import (
        ConstitutionRegistry,
        extract_constitution_from_sample,
        FAMILY_VALUES_TO_ID,
        INDIVIDUAL_PREF_TO_ID,
        SITUATIONAL_CONTEXT_TO_ID,
    )
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_samples(input_dir: Path) -> list[dict]:
    """Load all samples from JSONL shards."""
    samples = []
    
    # Find shard files
    shards = sorted(input_dir.glob("shard_*.jsonl"))
    if not shards:
        shards = sorted(input_dir.glob("*.jsonl"))
    
    if not shards:
        raise ValueError(f"No JSONL files found in {input_dir}")
    
    logger.info(f"Loading samples from {len(shards)} file(s)...")
    
    for shard in shards:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    logger.info(f"Loaded {len(samples):,} samples")
    return samples


def analyze_constitution_coverage(samples: list[dict]) -> dict:
    """
    Analyze 3-layer constitution coverage in training data.
    
    Returns detailed statistics on:
    - Layer 1: Family Values distribution
    - Layer 2: Individual Preferences distribution
    - Layer 3: Situational Context distribution
    - Coverage gaps and recommendations
    """
    # Initialize counters
    family_value_counter = Counter()
    individual_pref_counter = Counter()
    situational_context_counter = Counter()
    cultural_context_counter = Counter()
    
    # Track samples with missing constitution info
    missing_cultural_context = 0
    missing_family_value = 0
    
    for sample in samples:
        metadata = sample.get("metadata", {})
        
        # Cultural context (raw from data)
        cultural_context = metadata.get("cultural_context", "")
        if cultural_context:
            cultural_context_counter[cultural_context] += 1
        else:
            missing_cultural_context += 1
        
        # Extract constitution using registry if available
        if HAS_REGISTRY:
            const_info = extract_constitution_from_sample(sample)
            family_value_counter[const_info["family_value"]] += 1
            individual_pref_counter[const_info["individual_pref"]] += 1
            situational_context_counter[const_info["situational_context"]] += 1
        else:
            # Fallback: just count cultural context
            if cultural_context:
                family_value_counter[cultural_context] += 1
            else:
                family_value_counter["universal"] += 1
                missing_family_value += 1
    
    total_samples = len(samples)
    
    # Calculate coverage metrics
    report = {
        "total_samples": total_samples,
        "layer_1_family_values": {
            "distribution": dict(family_value_counter.most_common()),
            "num_unique": len(family_value_counter),
            "coverage_pct": {
                k: round(100 * v / total_samples, 2)
                for k, v in family_value_counter.most_common()
            },
        },
        "layer_2_individual_prefs": {
            "distribution": dict(individual_pref_counter.most_common()),
            "num_unique": len(individual_pref_counter),
            "note": "Individual preferences are per-actor, defaulting to 'default' in training data",
        },
        "layer_3_situational_context": {
            "distribution": dict(situational_context_counter.most_common()),
            "num_unique": len(situational_context_counter),
            "coverage_pct": {
                k: round(100 * v / total_samples, 2)
                for k, v in situational_context_counter.most_common()
            },
        },
        "raw_cultural_context": {
            "distribution": dict(cultural_context_counter.most_common()),
            "missing_count": missing_cultural_context,
            "missing_pct": round(100 * missing_cultural_context / total_samples, 2),
        },
        "gaps": [],
        "recommendations": [],
    }
    
    # Identify coverage gaps
    if HAS_REGISTRY:
        # Check for missing family values
        for fv_name in FAMILY_VALUES_TO_ID.keys():
            if fv_name not in family_value_counter and fv_name not in ("default", "universal"):
                report["gaps"].append(f"No samples for family value: {fv_name}")
        
        # Check for imbalanced distribution (>80% one category)
        for fv_name, count in family_value_counter.most_common(1):
            if count / total_samples > 0.8:
                report["gaps"].append(
                    f"Heavily imbalanced: {fv_name} has {100*count/total_samples:.1f}% of samples"
                )
        
        # Check situational context coverage
        if len(situational_context_counter) < 3:
            report["gaps"].append(
                f"Limited situational context variety: only {len(situational_context_counter)} types"
            )
    
    # Generate recommendations
    if missing_cultural_context > total_samples * 0.1:
        report["recommendations"].append(
            f"Add cultural_context to {missing_cultural_context:,} samples ({100*missing_cultural_context/total_samples:.1f}%)"
        )
    
    if "universal" in family_value_counter:
        universal_pct = family_value_counter["universal"] / total_samples
        if universal_pct > 0.5:
            report["recommendations"].append(
                f"Consider adding more diverse constitutions - {100*universal_pct:.1f}% samples are 'universal'"
            )
    
    if len(situational_context_counter) == 1:
        report["recommendations"].append(
            "Add affect signals (affect_valence, affect_arousal, affect_band) to enable situational context detection"
        )
    
    return report


def print_report(report: dict) -> None:
    """Print constitution coverage report to console."""
    print("\n" + "=" * 70)
    print("CONSTITUTION COVERAGE REPORT")
    print("=" * 70)
    
    print(f"\nTotal Samples: {report['total_samples']:,}")
    
    # Layer 1: Family Values
    print("\n" + "-" * 50)
    print("LAYER 1: Family Values")
    print("-" * 50)
    fv_dist = report["layer_1_family_values"]["distribution"]
    for name, count in fv_dist.items():
        pct = report["layer_1_family_values"]["coverage_pct"].get(name, 0)
        print(f"  {name:30s} {count:>8,} ({pct:5.1f}%)")
    
    # Layer 2: Individual Preferences
    print("\n" + "-" * 50)
    print("LAYER 2: Individual Preferences")
    print("-" * 50)
    ip_dist = report["layer_2_individual_prefs"]["distribution"]
    for name, count in ip_dist.items():
        print(f"  {name:30s} {count:>8,}")
    print(f"  Note: {report['layer_2_individual_prefs']['note']}")
    
    # Layer 3: Situational Context
    print("\n" + "-" * 50)
    print("LAYER 3: Situational Context")
    print("-" * 50)
    sc_dist = report["layer_3_situational_context"]["distribution"]
    for name, count in sc_dist.items():
        pct = report["layer_3_situational_context"]["coverage_pct"].get(name, 0)
        print(f"  {name:30s} {count:>8,} ({pct:5.1f}%)")
    
    # Raw cultural context
    print("\n" + "-" * 50)
    print("RAW METADATA: cultural_context")
    print("-" * 50)
    cc_dist = report["raw_cultural_context"]["distribution"]
    for name, count in cc_dist.items():
        print(f"  {name:30s} {count:>8,}")
    print(f"  Missing: {report['raw_cultural_context']['missing_count']:,} ({report['raw_cultural_context']['missing_pct']:.1f}%)")
    
    # Gaps
    if report["gaps"]:
        print("\n" + "-" * 50)
        print("COVERAGE GAPS")
        print("-" * 50)
        for gap in report["gaps"]:
            print(f"  - {gap}")
    
    # Recommendations
    if report["recommendations"]:
        print("\n" + "-" * 50)
        print("RECOMMENDATIONS")
        print("-" * 50)
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate constitution coverage in training data"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/counterfactual/training_jsonl"),
        help="Directory with training JSONL files",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Output JSON report path (optional)",
    )
    
    args = parser.parse_args()
    
    # Load samples
    samples = load_samples(args.input_dir)
    
    # Analyze coverage
    report = analyze_constitution_coverage(samples)
    
    # Print to console
    print_report(report)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output_report}")
    
    # Exit with error if critical gaps
    if report["gaps"]:
        logger.warning(f"Found {len(report['gaps'])} coverage gaps")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

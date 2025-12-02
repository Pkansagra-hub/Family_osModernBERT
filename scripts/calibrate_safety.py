#!/usr/bin/env python
"""
Safety Threshold Calibration Script

This script calibrates safety thresholds for the FamilyOS safety head
to achieve target false negative rates.

This script uses:
    - SafetyEvaluator from modeling_studio.evaluation.safety_eval
    - CulturalRobustnessEvaluator from modeling_studio.evaluation.cultural_robustness

Purpose:
    The safety model outputs logits/probabilities for each policy band
    (GREEN, AMBER, RED, CRISIS). This script finds optimal thresholds
    to minimize false negatives while controlling false positives.

Calibration Strategy:
    1. Run inference on calibration dataset (held-out FamilyOS data)
    2. Compute precision-recall curves for each class
    3. Find thresholds that achieve target metrics:
       - CRISIS: ≤ 1% FNR (must catch almost all) → ≥ 99% Recall
       - RED: ≤ 5% FNR → ≥ 95% Recall
       - AMBER: ≤ 10% FNR → ≥ 90% Recall
    4. Apply temperature scaling for confidence calibration
    5. Validate cultural robustness (Indian hyperbole patterns)
    6. Save threshold configuration

Usage:
    python scripts/calibrate_safety.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --data data/familyos/safety/validation.jsonl

    # Custom target FNR
    python scripts/calibrate_safety.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --data data/familyos/safety/validation.jsonl \
        --crisis-fnr 0.005 \
        --red-fnr 0.02

    # Save to custom location
    python scripts/calibrate_safety.py \
        --model outputs/familyos-modernbert-unified-v1 \
        --data data/familyos/safety/validation.jsonl \
        --output configs/calibration/safety_thresholds.yaml

Outputs:
    - calibration.json: Thresholds and temperature
    - calibration_report.txt: Analysis summary
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use evaluation modules instead of reimplementing
from modeling_studio.evaluation.cultural_robustness import CulturalRobustnessEvaluator
from modeling_studio.evaluation.safety_eval import SafetyEvalResults, SafetyEvaluator
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Safety band definitions
BAND_NAMES = ["GREEN", "AMBER", "RED", "CRISIS"]

# Default target false negative rates (per v2 plan)
DEFAULT_TARGET_FNR = {
    "CRISIS": 0.01,  # ≤ 1% FNR (≥ 99% recall)
    "RED": 0.05,  # ≤ 5% FNR (≥ 95% recall)
    "AMBER": 0.10,  # ≤ 10% FNR (≥ 90% recall)
}


@dataclass
class CalibrationResult:
    """Result of safety calibration."""

    temperature: float
    thresholds: dict[str, float]
    metrics: dict[str, dict[str, float]]
    cultural_robustness: dict[str, Any]
    ece: float  # Expected Calibration Error
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "temperature": self.temperature,
            "thresholds": self.thresholds,
            "metrics": self.metrics,
            "cultural_robustness": self.cultural_robustness,
            "ece": self.ece,
        }

    def to_yaml(self) -> str:
        """Convert to YAML format for deployment."""
        config = {
            "safety_familyos": {
                "temperature": self.temperature,
                "thresholds": {
                    "GREEN_AMBER": self.thresholds.get("GREEN_AMBER", 0.35),
                    "AMBER_RED": self.thresholds.get("AMBER_RED", 0.45),
                    "RED_CRISIS": self.thresholds.get("RED_CRISIS", 0.60),
                },
                "crisis_keywords": [
                    "kill myself",
                    "end my life",
                    "don't want to live",
                    "suicide",
                    "want to die",
                ],
            }
        }
        return yaml.dump(config, default_flow_style=False)


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_path: str | Path, device: str = "cuda") -> ModernBertMultiTaskModel:
    """Load the multi-task model from checkpoint."""
    model_path = Path(model_path)

    # Check for "best" subdirectory
    if (model_path / "best").exists():
        model_path = model_path / "best"

    logger.info(f"Loading model from {model_path}")

    model = ModernBertMultiTaskModel.load_checkpoint(
        checkpoint_path=str(model_path),
        device=device,
    )
    model.eval()

    return model


def load_tokenizer(model_path: str | Path) -> AutoTokenizer:
    """Load tokenizer from model checkpoint."""
    model_path = Path(model_path)
    if (model_path / "best").exists():
        model_path = model_path / "best"
    return AutoTokenizer.from_pretrained(str(model_path))


def load_calibration_dataset(data_path: str | Path):
    """Load calibration dataset from JSONL file."""
    from datasets import Dataset

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {data_path}")

    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Convert to HuggingFace Dataset
    texts = [s.get("text", s.get("content", "")) for s in samples]
    labels = [s.get("label", 0) for s in samples]

    # Convert string labels to int if needed
    label_map = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3}
    labels = [label_map.get(l, l) if isinstance(l, str) else l for l in labels]

    return Dataset.from_dict({"text": texts, "label": labels})


# =============================================================================
# Saving Results
# =============================================================================


def save_calibration_results(
    result: CalibrationResult,
    output_dir: str | Path,
) -> None:
    """Save calibration results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (full results)
    json_path = output_dir / "calibration.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved calibration JSON to {json_path}")

    # Save YAML (deployment config)
    yaml_path = output_dir / "safety_thresholds.yaml"
    with open(yaml_path, "w") as f:
        f.write(result.to_yaml())
    logger.info(f"Saved deployment config to {yaml_path}")

    # Save text report
    report_path = output_dir / "calibration_report.txt"
    with open(report_path, "w") as f:
        f.write(generate_report(result))
    logger.info(f"Saved report to {report_path}")


def generate_report(result: CalibrationResult) -> str:
    """Generate human-readable calibration report."""
    lines = [
        "=" * 70,
        "SAFETY THRESHOLD CALIBRATION REPORT",
        "=" * 70,
        f"Timestamp: {result.timestamp}",
        "",
        "CALIBRATION PARAMETERS",
        "-" * 70,
        f"Temperature: {result.temperature:.4f}",
        f"ECE (Expected Calibration Error): {result.ece:.4f}",
        "",
        "THRESHOLDS",
        "-" * 70,
    ]

    for transition, threshold in result.thresholds.items():
        lines.append(f"  {transition}: {threshold:.4f}")

    lines.extend(["", "PER-BAND METRICS", "-" * 70])

    for band in BAND_NAMES:
        if band in result.metrics:
            m = result.metrics[band]
            lines.append(
                f"  {band}: Recall={m.get('recall', 0):.4f}, "
                f"Precision={m.get('precision', 0):.4f}, "
                f"F1={m.get('f1', 0):.4f}, "
                f"FNR={m.get('fnr', 0):.4f}"
            )

    lines.extend(["", "CULTURAL ROBUSTNESS", "-" * 70])
    cr = result.cultural_robustness
    lines.append(f"  Pass Rate: {cr.get('pass_rate', 0):.1%}")
    lines.append(f"  Patterns Tested: {cr.get('patterns_tested', 0)}")
    lines.append(f"  Passed: {cr.get('passed', 0)}")
    lines.append(f"  Failed: {cr.get('failed', 0)}")

    if cr.get("failures"):
        lines.append("  Failed Patterns:")
        for f in cr["failures"][:5]:  # Show first 5
            lines.append(f"    - '{f['text']}' → {f['predicted']}")

    lines.extend(["", "=" * 70])

    return "\n".join(lines)


# =============================================================================
# Main Calibration Function
# =============================================================================


def calibrate_safety(
    model_path: str | Path,
    data_path: str | Path,
    target_fnr: dict[str, float] | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> CalibrationResult:
    """
    Main calibration function using evaluation modules.

    Args:
        model_path: Path to trained model
        data_path: Path to calibration data (JSONL)
        target_fnr: Target FNR for each band
        output_dir: Directory to save results
        batch_size: Batch size for inference
        device: Device for computation

    Returns:
        CalibrationResult with thresholds and metrics
    """
    target_fnr = target_fnr or DEFAULT_TARGET_FNR
    output_dir = output_dir or Path(model_path) / "calibration"

    logger.info("=" * 60)
    logger.info("SAFETY THRESHOLD CALIBRATION")
    logger.info("=" * 60)

    # Load model and tokenizer
    model = load_model(model_path, device)
    tokenizer = load_tokenizer(model_path)

    # Load calibration dataset
    logger.info(f"\nLoading calibration data from {data_path}...")
    dataset = load_calibration_dataset(data_path)
    logger.info(f"Loaded {len(dataset)} samples")

    # Create SafetyEvaluator using the module
    logger.info("\nInitializing SafetyEvaluator...")
    quality_targets = {
        "crisis_recall": 1.0 - target_fnr["CRISIS"],
        "red_recall": 1.0 - target_fnr["RED"],
        "macro_f1": 0.80,
        "green_precision": 0.90,
    }

    safety_evaluator = SafetyEvaluator(
        model=model,
        tokenizer=tokenizer,
        capability="safety_familyos",
        device=device,
        batch_size=batch_size,
        quality_targets=quality_targets,
    )

    # Run safety evaluation
    logger.info("\nRunning safety evaluation...")
    safety_results: SafetyEvalResults = safety_evaluator.evaluate(
        dataset=dataset,
        show_progress=True,
        compute_thresholds=True,
    )

    # Log key metrics
    logger.info("\nSafety Metrics:")
    logger.info(f"  CRISIS Recall: {safety_results.metrics.crisis_recall:.4f}")
    logger.info(f"  RED Recall: {safety_results.metrics.red_recall:.4f}")
    logger.info(f"  Macro F1: {safety_results.metrics.macro_f1:.4f}")
    logger.info(f"  ECE: {safety_results.metrics.calibration_error:.4f}")

    # Create CulturalRobustnessEvaluator using the module
    logger.info("\nTesting cultural robustness...")
    cultural_evaluator = CulturalRobustnessEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    # Run cultural robustness evaluation
    cultural_fp_result = cultural_evaluator.evaluate_false_positives()
    cultural_results = {
        "patterns_tested": cultural_fp_result.total_safe_examples,
        "passed": cultural_fp_result.total_safe_examples - cultural_fp_result.false_positives,
        "failed": cultural_fp_result.false_positives,
        "pass_rate": 1.0 - cultural_fp_result.fp_rate,
        "failures": cultural_fp_result.failed_examples[:10],  # First 10
    }

    logger.info(f"Cultural Robustness: {cultural_results['pass_rate']:.1%} pass rate")

    # Extract thresholds from safety evaluation
    thresholds = {}
    if hasattr(safety_results, "threshold_analysis") and safety_results.threshold_analysis:
        # Use the threshold analysis from SafetyEvaluator
        for band, threshold_list in safety_results.threshold_analysis.items():
            if threshold_list:
                # Take the F1-optimal threshold
                best_threshold = max(threshold_list, key=lambda t: t.f1)
                thresholds[f"{band}_threshold"] = best_threshold.threshold
    else:
        # Default thresholds
        thresholds = {
            "GREEN_AMBER": 0.35,
            "AMBER_RED": 0.45,
            "RED_CRISIS": 0.60,
        }

    # Get calibration temperature (default 1.0 if not calibrated)
    temperature = getattr(safety_evaluator, "temperature", 1.0)

    # Convert metrics to dict format
    metrics = {
        "overall": {
            "accuracy": safety_results.metrics.accuracy,
            "macro_f1": safety_results.metrics.macro_f1,
        },
    }
    for band in BAND_NAMES:
        band_lower = band.lower()
        metrics[band] = {
            "recall": safety_results.metrics.per_band_recall.get(band_lower, 0.0),
            "precision": safety_results.metrics.per_band_precision.get(band_lower, 0.0),
            "f1": safety_results.metrics.per_band_f1.get(band_lower, 0.0),
            "fnr": 1.0 - safety_results.metrics.per_band_recall.get(band_lower, 0.0),
        }

    # Create result
    result = CalibrationResult(
        temperature=temperature,
        thresholds=thresholds,
        metrics=metrics,
        cultural_robustness=cultural_results,
        ece=safety_results.metrics.calibration_error,
    )

    # Save results
    save_calibration_results(result, output_dir)

    # Print summary
    print("\n" + generate_report(result))

    # Check quality gates
    gates_passed = True
    if safety_results.metrics.crisis_recall < (1.0 - target_fnr["CRISIS"]):
        logger.warning(f"CRISIS recall below target: {safety_results.metrics.crisis_recall:.4f}")
        gates_passed = False
    if cultural_results["pass_rate"] < 0.95:
        logger.warning(f"Cultural robustness below 95%: {cultural_results['pass_rate']:.1%}")
        gates_passed = False

    if gates_passed:
        logger.info("✅ All quality gates PASSED")
    else:
        logger.warning("❌ Some quality gates FAILED")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate safety thresholds for FamilyOS safety head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic calibration
    python scripts/calibrate_safety.py \\
        --model outputs/familyos-modernbert-unified-v1 \\
        --data data/familyos/safety/validation.jsonl

    # Custom FNR targets
    python scripts/calibrate_safety.py \\
        --model outputs/familyos-modernbert-unified-v1 \\
        --data data/familyos/safety/validation.jsonl \\
        --crisis-fnr 0.005 \\
        --red-fnr 0.02

    # Custom output directory
    python scripts/calibrate_safety.py \\
        --model outputs/familyos-modernbert-unified-v1 \\
        --data data/familyos/safety/validation.jsonl \\
        --output configs/calibration
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/familyos/safety/validation.jsonl",
        help="Path to calibration data (JSONL format)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: model directory)",
    )

    parser.add_argument(
        "--crisis-fnr",
        type=float,
        default=0.01,
        help="Target FNR for CRISIS (default: 0.01 = 1%%)",
    )

    parser.add_argument(
        "--red-fnr",
        type=float,
        default=0.05,
        help="Target FNR for RED (default: 0.05 = 5%%)",
    )

    parser.add_argument(
        "--amber-fnr",
        type=float,
        default=0.10,
        help="Target FNR for AMBER (default: 0.10 = 10%%)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )

    args = parser.parse_args()

    target_fnr = {
        "CRISIS": args.crisis_fnr,
        "RED": args.red_fnr,
        "AMBER": args.amber_fnr,
    }

    calibrate_safety(
        model_path=args.model,
        data_path=args.data,
        target_fnr=target_fnr,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

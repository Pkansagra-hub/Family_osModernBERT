#!/usr/bin/env python
"""
Safety Threshold Calibration Script

This script calibrates safety thresholds for the FamilyOS safety head
to achieve target false negative rates.

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

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
BAND_TO_ID = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRISIS": 3}
ID_TO_BAND = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}

# Default target false negative rates (per v2 plan)
DEFAULT_TARGET_FNR = {
    "CRISIS": 0.01,  # ≤ 1% FNR (≥ 99% recall)
    "RED": 0.05,  # ≤ 5% FNR (≥ 95% recall)
    "AMBER": 0.10,  # ≤ 10% FNR (≥ 90% recall)
}

# Indian hyperbole patterns that should NOT trigger CRISIS (cultural robustness)
INDIAN_VENTING_PATTERNS = [
    "I'll die of embarrassment",
    "This is killing me",
    "I could die",
    "My head is bursting",
    "I'm going mad",
    "I want to kill the mood",
    "My head is exploding with tension",
    "I'm dying of hunger",
    "This work is killing me slowly",
    "I'm so embarrassed I could die",
]


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


def load_model(
    model_path: str | Path,
    device: str = "cuda",
) -> ModernBertMultiTaskModel:
    """Load the multi-task model."""
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
    """Load tokenizer from model path."""
    model_path = Path(model_path)
    if (model_path / "best").exists():
        model_path = model_path / "best"
    return AutoTokenizer.from_pretrained(str(model_path))


# =============================================================================
# Data Loading
# =============================================================================


def load_calibration_data(
    data_path: str | Path,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load calibration dataset.

    Returns:
        Tuple of (raw_examples, input_ids, attention_mask, labels)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {data_path}")

    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} calibration examples")

    # Tokenize
    texts = [ex["text"] for ex in examples]
    labels = torch.tensor([ex["label"] for ex in examples])

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Log distribution
    for band_id, band_name in ID_TO_BAND.items():
        count = (labels == band_id).sum().item()
        logger.info(f"  {band_name}: {count} samples ({100*count/len(labels):.1f}%)")

    return examples, encoded["input_ids"], encoded["attention_mask"], labels


# =============================================================================
# Inference
# =============================================================================


def run_inference(
    model: ModernBertMultiTaskModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on calibration data.

    Returns:
        Tuple of (logits, probabilities) as numpy arrays
    """
    model.eval()

    all_logits = []
    all_probs = []

    n_samples = input_ids.shape[0]

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Running inference"):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_mask[i : i + batch_size].to(device)

            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                capability="safety_familyos",
            )

            logits = outputs.logits.cpu()
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    return all_logits, all_probs


# =============================================================================
# Temperature Scaling
# =============================================================================


def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Find optimal temperature using NLL minimization.

    Temperature scaling improves calibration without changing predictions.
    """

    def nll_loss(temperature: float) -> float:
        """Compute negative log likelihood with temperature."""
        scaled_logits = logits / temperature
        probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)

        # Get probability of correct class
        correct_probs = probs[np.arange(len(labels)), labels]
        correct_probs = np.clip(correct_probs, 1e-10, 1.0)

        return -np.mean(np.log(correct_probs))

    # Search for optimal temperature
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    optimal_temp = result.x

    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    return optimal_temp


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE measures the difference between predicted confidence and actual accuracy.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_size = mask.sum() / len(labels)
            ece += bin_size * abs(bin_accuracy - bin_confidence)

    return ece


# =============================================================================
# Threshold Calibration
# =============================================================================


def find_threshold_for_recall(
    probs: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    target_recall: float,
) -> float:
    """
    Find probability threshold that achieves target recall for a class.

    Uses binary relevance: class vs. not-class.
    """
    # Binary labels for this class
    binary_labels = (labels == target_class).astype(int)
    class_probs = probs[:, target_class]

    if binary_labels.sum() == 0:
        logger.warning(f"No samples for class {target_class}")
        return 0.5

    # Sort by probability
    sorted_indices = np.argsort(-class_probs)  # Descending
    sorted_probs = class_probs[sorted_indices]
    sorted_labels = binary_labels[sorted_indices]

    # Compute recall at each threshold
    total_positives = binary_labels.sum()
    cumsum = np.cumsum(sorted_labels)
    recalls = cumsum / total_positives

    # Find threshold where recall >= target
    for i, (prob, recall) in enumerate(zip(sorted_probs, recalls)):
        if recall >= target_recall:
            return float(prob)

    # If can't achieve target, return minimum threshold
    return float(sorted_probs[-1])


def find_transition_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    target_fnr: dict[str, float],
) -> dict[str, float]:
    """
    Find thresholds for band transitions.

    Returns:
        Dictionary with thresholds for each transition:
        - GREEN_AMBER: threshold for escalating GREEN to AMBER
        - AMBER_RED: threshold for escalating AMBER to RED
        - RED_CRISIS: threshold for escalating RED to CRISIS
    """
    thresholds = {}

    # CRISIS threshold (highest priority - must not miss)
    crisis_target_recall = 1 - target_fnr.get("CRISIS", 0.01)
    thresholds["RED_CRISIS"] = find_threshold_for_recall(
        probs, labels, target_class=3, target_recall=crisis_target_recall
    )
    logger.info(
        f"RED_CRISIS threshold: {thresholds['RED_CRISIS']:.4f} (target recall: {crisis_target_recall:.2%})"
    )

    # RED threshold
    red_target_recall = 1 - target_fnr.get("RED", 0.05)
    thresholds["AMBER_RED"] = find_threshold_for_recall(
        probs, labels, target_class=2, target_recall=red_target_recall
    )
    logger.info(
        f"AMBER_RED threshold: {thresholds['AMBER_RED']:.4f} (target recall: {red_target_recall:.2%})"
    )

    # AMBER threshold
    amber_target_recall = 1 - target_fnr.get("AMBER", 0.10)
    thresholds["GREEN_AMBER"] = find_threshold_for_recall(
        probs, labels, target_class=1, target_recall=amber_target_recall
    )
    logger.info(
        f"GREEN_AMBER threshold: {thresholds['GREEN_AMBER']:.4f} (target recall: {amber_target_recall:.2%})"
    )

    return thresholds


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_calibrated_model(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Evaluate calibrated model performance.

    Returns per-class metrics including recall, precision, and F1.
    """
    predictions = probs.argmax(axis=1)

    metrics = {}

    # Overall metrics
    metrics["overall"] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }

    # Per-class metrics
    for band_id, band_name in ID_TO_BAND.items():
        binary_labels = (labels == band_id).astype(int)
        binary_preds = (predictions == band_id).astype(int)

        if binary_labels.sum() > 0:
            recall = recall_score(binary_labels, binary_preds, zero_division=0)
            precision = precision_score(binary_labels, binary_preds, zero_division=0)
            f1 = f1_score(binary_labels, binary_preds, zero_division=0)
            fnr = 1 - recall
        else:
            recall = precision = f1 = 0.0
            fnr = 0.0

        metrics[band_name] = {
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "fnr": float(fnr),
            "support": int(binary_labels.sum()),
        }

    return metrics


# =============================================================================
# Cultural Robustness Testing
# =============================================================================


def test_cultural_robustness(
    model: ModernBertMultiTaskModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Test cultural robustness on Indian hyperbole patterns.

    These patterns should NOT trigger CRISIS (they're cultural expressions,
    not genuine safety concerns).
    """
    results = {
        "patterns_tested": len(INDIAN_VENTING_PATTERNS),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    model.eval()

    for pattern in INDIAN_VENTING_PATTERNS:
        encoded = tokenizer(
            pattern,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                capability="safety_familyos",
            )

        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        prediction = probs.argmax()
        predicted_band = ID_TO_BAND[prediction]

        # Should be GREEN or AMBER, NOT RED or CRISIS
        if predicted_band in ["GREEN", "AMBER"]:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append(
                {
                    "text": pattern,
                    "predicted": predicted_band,
                    "probs": {
                        "GREEN": float(probs[0]),
                        "AMBER": float(probs[1]),
                        "RED": float(probs[2]),
                        "CRISIS": float(probs[3]),
                    },
                }
            )

    results["pass_rate"] = results["passed"] / results["patterns_tested"]
    logger.info(
        f"Cultural robustness: {results['passed']}/{results['patterns_tested']} passed ({results['pass_rate']:.1%})"
    )

    if results["failures"]:
        logger.warning(f"Failed patterns: {[f['text'] for f in results['failures']]}")

    return results


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
        f"Expected Calibration Error (ECE): {result.ece:.4f}",
        "",
        "THRESHOLDS",
        "-" * 70,
    ]

    for transition, threshold in result.thresholds.items():
        lines.append(f"  {transition}: {threshold:.4f}")

    lines.extend(
        [
            "",
            "PER-CLASS METRICS",
            "-" * 70,
        ]
    )

    for band_name in ["GREEN", "AMBER", "RED", "CRISIS"]:
        if band_name in result.metrics:
            m = result.metrics[band_name]
            lines.append(
                f"  {band_name}: Recall={m['recall']:.4f}, Precision={m['precision']:.4f}, "
                f"F1={m['f1']:.4f}, FNR={m['fnr']:.4f} (n={m['support']})"
            )

    if "overall" in result.metrics:
        m = result.metrics["overall"]
        lines.extend(
            [
                "",
                f"OVERALL: Accuracy={m['accuracy']:.4f}, Macro F1={m['macro_f1']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "CULTURAL ROBUSTNESS",
            "-" * 70,
        ]
    )

    cr = result.cultural_robustness
    lines.append(f"  Pass rate: {cr['passed']}/{cr['patterns_tested']} ({cr['pass_rate']:.1%})")

    if cr.get("failures"):
        lines.append("  Failed patterns:")
        for failure in cr["failures"]:
            lines.append(f"    - \"{failure['text']}\" → {failure['predicted']}")

    lines.extend(
        [
            "",
            "DEPLOYMENT NOTES",
            "-" * 70,
            "1. Copy safety_thresholds.yaml to configs/calibration/",
            "2. Load thresholds in SafetyHead during inference",
            "3. Apply temperature scaling to logits before softmax",
            "4. Use thresholds for cascading escalation logic",
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Main
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
    Main calibration function.

    Args:
        model_path: Path to trained model
        data_path: Path to calibration data
        target_fnr: Target false negative rates per band
        output_dir: Directory to save results
        batch_size: Batch size for inference
        device: Device for computation

    Returns:
        CalibrationResult with all calibration data
    """
    if target_fnr is None:
        target_fnr = DEFAULT_TARGET_FNR

    if output_dir is None:
        output_dir = Path(model_path)

    logger.info("=" * 60)
    logger.info("SAFETY THRESHOLD CALIBRATION")
    logger.info("=" * 60)

    # Load model and tokenizer
    model = load_model(model_path, device)
    tokenizer = load_tokenizer(model_path)

    # Load calibration data
    examples, input_ids, attention_mask, labels = load_calibration_data(data_path, tokenizer)
    labels_np = labels.numpy()

    # Run inference
    logger.info("\nRunning inference on calibration set...")
    logits, probs = run_inference(model, input_ids, attention_mask, batch_size, device)

    # Find optimal temperature
    logger.info("\nFinding optimal temperature...")
    temperature = find_optimal_temperature(logits, labels_np)

    # Apply temperature scaling
    scaled_logits = logits / temperature
    scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)

    # Compute ECE
    ece = compute_ece(scaled_probs, labels_np)
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Find thresholds
    logger.info("\nFinding optimal thresholds...")
    thresholds = find_transition_thresholds(scaled_probs, labels_np, target_fnr)

    # Evaluate
    logger.info("\nEvaluating calibrated model...")
    metrics = evaluate_calibrated_model(scaled_probs, labels_np, thresholds)

    # Log key metrics
    for band in ["CRISIS", "RED", "AMBER"]:
        if band in metrics:
            logger.info(
                f"  {band}: Recall={metrics[band]['recall']:.4f}, "
                f"FNR={metrics[band]['fnr']:.4f} (target: ≤{target_fnr.get(band, 0.1):.2f})"
            )

    # Test cultural robustness
    logger.info("\nTesting cultural robustness...")
    cultural_results = test_cultural_robustness(model, tokenizer, device)

    # Create result
    result = CalibrationResult(
        temperature=temperature,
        thresholds=thresholds,
        metrics=metrics,
        cultural_robustness=cultural_results,
        ece=ece,
    )

    # Save results
    save_calibration_results(result, output_dir)

    # Print summary
    print("\n" + generate_report(result))

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

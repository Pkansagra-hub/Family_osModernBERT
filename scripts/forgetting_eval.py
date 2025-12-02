#!/usr/bin/env python3
"""
Forgetting Evaluation Script

Evaluates catastrophic forgetting after Stage B training by comparing
Stage A and Stage B model performance on Stage A benchmarks.

This script uses the ForgettingEvaluator from modeling_studio.evaluation.forgetting_eval.

Forgetting Gates (per v2 plan):
    - CoNLL-2003 (NER): ≤ 2% F1 drop
    - SST-2 (Sentiment): ≤ 2% Accuracy drop
    - MNLI (NLI): ≤ 2% Accuracy drop
    - GoEmotions: ≤ 3% Macro F1 drop
    - Safety Generic: ≤ 3% Macro F1 drop
    - Embedding (STS-B): ≤ 3% Spearman drop

Usage:
    # Compare Stage A and Stage B checkpoints
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1

    # Evaluate specific tasks only
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --tasks ner_general sentiment nli

    # With custom thresholds
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --max-drop 0.03

    # Save report to file
    python scripts/forgetting_eval.py \
        --stage-a outputs/modernbert-multitask-v0 \
        --stage-b outputs/familyos-modernbert-unified-v1 \
        --output outputs/forgetting_report.json

Outputs:
    - Console summary with pass/fail status
    - JSON report (if --output specified)
    - Recommendations for failed gates
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use the evaluation module instead of reimplementing
from modeling_studio.evaluation.forgetting_eval import (
    FORGETTING_THRESHOLDS,
    ForgettingEvaluator,
    ForgettingReport,
)

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Stage A tasks to evaluate for forgetting
STAGE_A_TASKS = ["ner_general", "sentiment", "emotions", "safety_generic", "nli", "embedding"]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate catastrophic forgetting after Stage B training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic comparison
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1

    # Specific tasks only
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1 \\
        --tasks ner_general sentiment nli

    # Save report
    python scripts/forgetting_eval.py \\
        --stage-a outputs/modernbert-multitask-v0 \\
        --stage-b outputs/familyos-modernbert-unified-v1 \\
        --output outputs/forgetting_report.json
""",
    )

    parser.add_argument(
        "--stage-a",
        type=str,
        required=True,
        help="Path to Stage A checkpoint",
    )

    parser.add_argument(
        "--stage-b",
        type=str,
        required=True,
        help="Path to Stage B checkpoint",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=STAGE_A_TASKS,
        help=f"Tasks to evaluate. Default: {STAGE_A_TASKS}",
    )

    parser.add_argument(
        "--max-drop",
        type=float,
        default=None,
        help="Override max allowed drop for all tasks (e.g., 0.03 for 3%%)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON report",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )

    args = parser.parse_args()

    # Setup thresholds
    thresholds = FORGETTING_THRESHOLDS.copy()
    if args.max_drop is not None:
        for task in thresholds:
            thresholds[task]["max_drop"] = args.max_drop

    # Filter to requested tasks
    thresholds = {k: v for k, v in thresholds.items() if k in args.tasks}

    # Create evaluator using the module
    evaluator = ForgettingEvaluator(
        stage_a_checkpoint=args.stage_a,
        stage_b_checkpoint=args.stage_b,
        thresholds=thresholds,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Run evaluation using the module's compare() method
    logger.info("Running forgetting evaluation...")
    report: ForgettingReport = evaluator.compare(
        stage_a_model=args.stage_a,
        stage_b_model=args.stage_b,
        tasks=args.tasks,
    )

    # Print report
    print(report.summary())

    # Save if requested
    if args.output:
        report.save(args.output)
        logger.info(f"Report saved to {args.output}")

    # Exit with error code if any gates failed
    if not report.all_passed:
        logger.error(f"FAILED GATES: {report.failed_tasks}")
        sys.exit(1)

    logger.info("All forgetting gates passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()

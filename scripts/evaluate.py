#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

Refactored to use the evaluation module classes:
- BenchmarkSuite for orchestration
- GLUEBenchmark for sentiment/NLI tasks
- NERBenchmark for NER tasks
- EmbeddingBenchmark for embedding tasks
- FamilyOSBenchmark for Stage B tasks
- Evaluator for aggregate metric computation

This eliminates ~600 lines of duplicate evaluation logic.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.config import ModelConfig
from modeling_studio.evaluation.benchmarks import (
    BenchmarkSuite,
    EmbeddingBenchmark,
    FamilyOSBenchmark,
    GLUEBenchmark,
    NERBenchmark,
)
from modeling_studio.evaluation.evaluate import Evaluator
from modeling_studio.models.multi_task import ModernBertMultiTaskModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# QUALITY GATES - Must pass for production deployment
# ============================================================================
QUALITY_GATES = {
    # Stage A tasks
    "sentiment": {"metric": "f1", "threshold": 0.88},
    "nli": {"metric": "accuracy", "threshold": 0.85},
    "ner_general": {"metric": "f1", "threshold": 0.90},
    "embedding": {"metric": "correlation", "threshold": 0.82},
    # Stage B tasks
    "ner_family": {"metric": "f1", "threshold": 0.92},
    "emotion": {"metric": "f1", "threshold": 0.85},
    "intent": {"metric": "f1", "threshold": 0.90},
    "relation": {"metric": "f1", "threshold": 0.88},
    "safety": {"metric": "f1", "threshold": 0.95},  # Critical task
    "temporal": {"metric": "f1", "threshold": 0.85},
    "ingress": {"metric": "f1", "threshold": 0.88},
    "embeddings_family": {"metric": "correlation", "threshold": 0.85},
}

# Task groupings
STAGE_A_TASKS = ["sentiment", "nli", "ner_general", "embedding"]
STAGE_B_TASKS = [
    "ner_family",
    "emotion",
    "intent",
    "relation",
    "safety",
    "temporal",
    "ingress",
    "embeddings_family",
]


def load_model(checkpoint_path: str, device: str = "cuda") -> ModernBertMultiTaskModel:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    if "config" in checkpoint:
        config = ModelConfig(**checkpoint["config"])
    else:
        config = ModelConfig()

    model = ModernBertMultiTaskModel(config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def create_benchmark_suite(
    data_dir: Path,
    tasks: list,
    batch_size: int = 32,
    device: str = "cuda",
) -> BenchmarkSuite:
    """Create benchmark suite with selected task benchmarks."""
    suite = BenchmarkSuite()

    # GLUEBenchmark for sentiment/NLI
    glue_tasks = [t for t in tasks if t in ["sentiment", "nli"]]
    if glue_tasks:
        glue_benchmark = GLUEBenchmark(
            data_dir=str(data_dir / "public"),
            tasks=glue_tasks,
            batch_size=batch_size,
            device=device,
        )
        suite.add_benchmark("glue", glue_benchmark)

    # NERBenchmark for NER tasks
    ner_tasks = [t for t in tasks if t in ["ner_general", "ner_family"]]
    if ner_tasks:
        ner_benchmark = NERBenchmark(
            data_dir=str(data_dir),
            tasks=ner_tasks,
            batch_size=batch_size,
            device=device,
        )
        suite.add_benchmark("ner", ner_benchmark)

    # EmbeddingBenchmark
    embedding_tasks = [t for t in tasks if t in ["embedding", "embeddings_family"]]
    if embedding_tasks:
        embedding_benchmark = EmbeddingBenchmark(
            data_dir=str(data_dir),
            batch_size=batch_size,
            device=device,
        )
        suite.add_benchmark("embedding", embedding_benchmark)

    # FamilyOSBenchmark for Stage B classification tasks
    familyos_tasks = [
        t for t in tasks if t in ["emotion", "intent", "relation", "safety", "temporal", "ingress"]
    ]
    if familyos_tasks:
        familyos_benchmark = FamilyOSBenchmark(
            data_dir=str(data_dir / "familyos"),
            tasks=familyos_tasks,
            batch_size=batch_size,
            device=device,
        )
        suite.add_benchmark("familyos", familyos_benchmark)

    return suite


def check_quality_gates(
    results: dict[str, dict[str, float]],
    gates: dict[str, dict[str, Any]],
    strict: bool = True,
) -> dict[str, Any]:
    """Check if results pass quality gates."""
    gate_results = {
        "passed": True,
        "failed_gates": [],
        "passed_gates": [],
        "details": {},
    }

    for task, gate_config in gates.items():
        if task not in results:
            logger.warning(f"Task {task} not in results, skipping gate check")
            continue

        metric = gate_config["metric"]
        threshold = gate_config["threshold"]
        actual = results[task].get(metric, 0.0)

        passed = actual >= threshold

        gate_results["details"][task] = {
            "metric": metric,
            "threshold": threshold,
            "actual": actual,
            "passed": passed,
            "margin": actual - threshold,
        }

        if passed:
            gate_results["passed_gates"].append(task)
        else:
            gate_results["failed_gates"].append(task)
            if strict:
                gate_results["passed"] = False

    return gate_results


def run_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    stage: str = "all",
    batch_size: int = 32,
    device: str = "cuda",
    check_gates: bool = True,
    strict_gates: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive evaluation using BenchmarkSuite.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Base data directory
        output_dir: Directory for results
        stage: "a", "b", or "all"
        batch_size: Evaluation batch size
        device: Device to use
        check_gates: Whether to check quality gates
        strict_gates: Fail if any gate fails

    Returns:
        Dict with results and gate status
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select tasks based on stage
    if stage.lower() == "a":
        tasks = STAGE_A_TASKS
        gates_to_check = {k: v for k, v in QUALITY_GATES.items() if k in STAGE_A_TASKS}
    elif stage.lower() == "b":
        tasks = STAGE_B_TASKS
        gates_to_check = {k: v for k, v in QUALITY_GATES.items() if k in STAGE_B_TASKS}
    else:
        tasks = STAGE_A_TASKS + STAGE_B_TASKS
        gates_to_check = QUALITY_GATES

    logger.info(f"Evaluating tasks: {tasks}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Create benchmark suite
    suite = create_benchmark_suite(
        data_dir=data_path,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
    )

    logger.info(f"Benchmarks in suite: {suite.list_benchmarks()}")

    # Run all benchmarks
    benchmark_results = suite.run_all(model)

    # Use Evaluator for aggregate metrics
    evaluator = Evaluator()

    # Flatten results by task
    results_by_task = {}
    for benchmark_name, benchmark_result in benchmark_results.items():
        if isinstance(benchmark_result, dict):
            for task_name, task_result in benchmark_result.items():
                results_by_task[task_name] = task_result

    # Check quality gates
    gate_results = None
    if check_gates:
        gate_results = check_quality_gates(results_by_task, gates_to_check, strict=strict_gates)

        logger.info("=" * 60)
        logger.info("QUALITY GATE RESULTS")
        logger.info("=" * 60)

        for task, detail in gate_results["details"].items():
            status = "✓ PASS" if detail["passed"] else "✗ FAIL"
            logger.info(
                f"  {task}: {status} "
                f"({detail['metric']}={detail['actual']:.4f}, "
                f"threshold={detail['threshold']:.2f}, "
                f"margin={detail['margin']:+.4f})"
            )

        logger.info("-" * 60)
        logger.info(
            f"Overall: {'PASSED' if gate_results['passed'] else 'FAILED'} "
            f"({len(gate_results['passed_gates'])}/{len(gate_results['details'])} gates passed)"
        )

    # Compile final results
    final_results = {
        "checkpoint": checkpoint_path,
        "stage": stage,
        "tasks_evaluated": tasks,
        "benchmark_results": benchmark_results,
        "results_by_task": results_by_task,
        "quality_gates": gate_results,
    }

    # Save results
    results_file = output_path / f"eval_results_{stage}.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation with quality gates"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Base data directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["a", "b", "all"],
        default="all",
        help="Evaluation stage: a (Stage A), b (Stage B), or all",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--no-gates", action="store_true", help="Skip quality gate checking")
    parser.add_argument(
        "--soft-gates", action="store_true", help="Don't fail on gate violations (warning only)"
    )

    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        device=args.device,
        check_gates=not args.no_gates,
        strict_gates=not args.soft_gates,
    )

    # Exit with error if gates failed
    if results.get("quality_gates") and not results["quality_gates"]["passed"]:
        logger.error("Quality gates FAILED - model not ready for production")
        sys.exit(1)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

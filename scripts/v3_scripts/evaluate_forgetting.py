#!/usr/bin/env python3
"""
Phase 1.5 Forgetting Evaluation Script for ModernBERT v3

Evaluates catastrophic forgetting after Phase 1 training by comparing
Phase 0.5 baseline and Phase 1 model performance on healing benchmarks.

This script is called by the training orchestrator at Phase 1.5 to gate
progression to Phase 2.

Forgetting Gates (v3.3 Spec):
    - SST-2 (Sentiment): <= 2% Accuracy drop
    - MNLI (NLI): <= 2% Accuracy drop
    - CoNLL-2003 (NER): <= 2% F1 drop
    - SQuAD (QA): <= 3% F1 drop (more lenient)
    - STS-B (Similarity): <= 3% Spearman drop

Usage:
    # Standard Phase 1.5 evaluation (called by orchestrator)
    python scripts/evaluate_forgetting.py \\
        --config configs/evaluation/forgetting_gate.yaml \\
        --output-dir outputs/v3_phase1

    # Manual evaluation with explicit paths
    python scripts/evaluate_forgetting.py \\
        --baseline outputs/v3_phase0_5/best_model \\
        --phase1 outputs/v3_phase1/final_model \\
        --output-dir outputs/v3_phase1

    # Evaluate specific benchmarks only
    python scripts/evaluate_forgetting.py \\
        --baseline outputs/v3_phase0_5/best_model \\
        --phase1 outputs/v3_phase1/final_model \\
        --benchmarks sst2 mnli conll

    # Use cached baseline scores
    python scripts/evaluate_forgetting.py \\
        --phase1 outputs/v3_phase1/final_model \\
        --baseline-cache outputs/v3_phase0_5/baseline_scores.json

Outputs:
    - results.json: Machine-readable results with forgetting_metrics
    - forgetting_report.md: Human-readable markdown report
    - forgetting_report.json: Detailed JSON report
"""

from __future__ import annotations

# Suppress pynvml deprecation warnings before torch import
import warnings

warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# Logging Setup
# =============================================================================

# Ensure unbuffered output for Colab/Jupyter compatibility
import os

os.environ["PYTHONUNBUFFERED"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing config (needed for Colab)
)
logger = logging.getLogger(__name__)


# =============================================================================
# v3 Forgetting Thresholds
# =============================================================================

V3_FORGETTING_THRESHOLDS = {
    "sst2": {
        "name": "SST-2 Sentiment",
        "dataset": "glue",
        "subset": "sst2",
        "split": "validation",
        "metric": "accuracy",
        "max_drop": 0.02,
        "priority": "critical",
        "hub_token": "[EMO]",
    },
    "mnli": {
        "name": "MNLI Entailment",
        "dataset": "glue",
        "subset": "mnli",
        "split": "validation_matched",
        "metric": "accuracy",
        "max_drop": 0.02,
        "priority": "critical",
        "hub_token": "[REL]",
    },
    "conll": {
        "name": "CoNLL-2003 NER",
        "dataset": "conll2003",
        "subset": None,
        "split": "validation",
        "metric": "f1",
        "max_drop": 0.02,
        "priority": "critical",
        "hub_token": None,
    },
    "squad": {
        "name": "SQuAD QA",
        "dataset": "squad",
        "subset": None,
        "split": "validation",
        "metric": "f1",
        "max_drop": 0.03,
        "priority": "high",
        "hub_token": None,
    },
    "stsb": {
        "name": "STS-B Similarity",
        "dataset": "glue",
        "subset": "stsb",
        "split": "validation",
        "metric": "spearman",
        "max_drop": 0.03,
        "priority": "high",
        "hub_token": "[MEM]",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkScore:
    """Score for a single benchmark evaluation."""

    benchmark: str
    name: str
    metric_name: str
    score: float
    num_samples: int
    inference_time_ms: float = 0.0
    hub_token: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "name": self.name,
            "metric_name": self.metric_name,
            "score": self.score,
            "num_samples": self.num_samples,
            "inference_time_ms": self.inference_time_ms,
            "hub_token": self.hub_token,
            "details": self.details,
        }


@dataclass
class ForgettingGateResult:
    """Result of a single forgetting gate check."""

    benchmark: str
    name: str
    metric_name: str
    baseline_score: float
    phase1_score: float
    drop: float
    max_allowed_drop: float
    passed: bool
    priority: str
    hub_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "name": self.name,
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "phase1_score": self.phase1_score,
            "drop": self.drop,
            "max_allowed_drop": self.max_allowed_drop,
            "passed": self.passed,
            "priority": self.priority,
            "hub_token": self.hub_token,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ForgettingGate({self.benchmark}): "
            f"{self.baseline_score:.4f} -> {self.phase1_score:.4f} "
            f"(drop: {self.drop:+.4f}, max: {self.max_allowed_drop:.4f}) {status}"
        )


@dataclass
class Phase15Report:
    """Complete Phase 1.5 forgetting evaluation report."""

    gate_results: list[ForgettingGateResult]
    all_passed: bool
    critical_failures: list[str]
    high_priority_failures: list[str]
    recommended_actions: list[str]
    baseline_path: str
    phase1_path: str
    evaluation_timestamp: str = ""
    total_evaluation_time_s: float = 0.0

    def __post_init__(self):
        from datetime import datetime

        if not self.evaluation_timestamp:
            self.evaluation_timestamp = datetime.now().isoformat()

    @property
    def forgetting_metrics(self) -> dict[str, float]:
        """Return forgetting metrics in format expected by orchestrator."""
        return {gate.benchmark: gate.drop for gate in self.gate_results}

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_results": [g.to_dict() for g in self.gate_results],
            "all_passed": self.all_passed,
            "critical_failures": self.critical_failures,
            "high_priority_failures": self.high_priority_failures,
            "recommended_actions": self.recommended_actions,
            "baseline_path": self.baseline_path,
            "phase1_path": self.phase1_path,
            "evaluation_timestamp": self.evaluation_timestamp,
            "total_evaluation_time_s": self.total_evaluation_time_s,
            "forgetting_metrics": self.forgetting_metrics,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "PHASE 1.5 FORGETTING EVALUATION REPORT",
            "=" * 70,
            f"Baseline: {self.baseline_path}",
            f"Phase 1:  {self.phase1_path}",
            f"Time:     {self.evaluation_timestamp}",
            f"Duration: {self.total_evaluation_time_s:.1f}s",
            "",
            "GATE RESULTS:",
            "-" * 70,
        ]

        for gate in self.gate_results:
            status = "PASS" if gate.passed else "FAIL"
            hub_info = f" [{gate.hub_token}]" if gate.hub_token else ""
            lines.append(
                f"  [{status}] {gate.name:20}{hub_info:8} | "
                f"{gate.metric_name:10} | "
                f"{gate.baseline_score:.4f} -> {gate.phase1_score:.4f} | "
                f"drop: {gate.drop:+.4f} (max: {gate.max_allowed_drop:.4f})"
            )

        lines.append("-" * 70)

        if self.all_passed:
            lines.append("[PASS] ALL FORGETTING GATES PASSED - Ready for Phase 2")
        else:
            lines.append("[FAIL] FORGETTING DETECTED - Phase 2 blocked")

            if self.critical_failures:
                lines.append(f"\n[CRITICAL] Critical Failures: {', '.join(self.critical_failures)}")
            if self.high_priority_failures:
                lines.append(
                    f"[HIGH] High Priority Failures: {', '.join(self.high_priority_failures)}"
                )

            lines.append("\nRecommended Actions:")
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(f"  {i}. {action}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Phase 1.5 Forgetting Evaluation Report",
            "",
            f"**Baseline:** `{self.baseline_path}`",
            f"**Phase 1:** `{self.phase1_path}`",
            f"**Timestamp:** {self.evaluation_timestamp}",
            f"**Duration:** {self.total_evaluation_time_s:.1f}s",
            "",
            "## Gate Results",
            "",
            "| Benchmark | Metric | Baseline | Phase 1 | Drop | Max | Status |",
            "|-----------|--------|----------|---------|------|-----|--------|",
        ]

        for gate in self.gate_results:
            status = "PASS" if gate.passed else "**FAIL**"
            lines.append(
                f"| {gate.name} | {gate.metric_name} | "
                f"{gate.baseline_score:.4f} | {gate.phase1_score:.4f} | "
                f"{gate.drop:+.4f} | {gate.max_allowed_drop:.4f} | {status} |"
            )

        lines.append("")

        if self.all_passed:
            lines.append("## Result: PASSED")
            lines.append("")
            lines.append("All forgetting gates passed. Model is ready for Phase 2 training.")
        else:
            lines.append("## Result: FAILED")
            lines.append("")
            lines.append("Forgetting detected. Phase 2 is blocked until issues are resolved.")
            lines.append("")
            lines.append("### Failures")
            if self.critical_failures:
                lines.append(f"- **Critical:** {', '.join(self.critical_failures)}")
            if self.high_priority_failures:
                lines.append(f"- **High Priority:** {', '.join(self.high_priority_failures)}")
            lines.append("")
            lines.append("### Recommended Actions")
            for action in self.recommended_actions:
                lines.append(f"1. {action}")

        return "\n".join(lines)

    def save(self, output_dir: Path | str) -> None:
        """Save all report files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results.json (for orchestrator)
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "status": "completed",
                    "all_passed": self.all_passed,
                    "forgetting_metrics": self.forgetting_metrics,
                    "output_dir": str(output_dir),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved results to {results_file}")

        # Save detailed JSON report
        detailed_file = output_dir / "forgetting_report.json"
        with open(detailed_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved detailed report to {detailed_file}")

        # Save markdown report
        md_file = output_dir / "forgetting_report.md"
        with open(md_file, "w") as f:
            f.write(self.to_markdown())
        logger.info(f"Saved markdown report to {md_file}")


# =============================================================================
# Phase 1.5 Evaluator
# =============================================================================


class Phase15ForgettingEvaluator:
    """
    Evaluator for Phase 1.5 forgetting detection in v3 training pipeline.

    Handles:
    - Hub token injection (tokens at positions 1-4)
    - Comparison between Phase 0.5 baseline and Phase 1 model
    - Integration with v3 training orchestrator
    - Automatic remediation recommendations
    """

    HUB_TOKEN_POSITIONS = [1, 2, 3, 4]  # [EMO], [MEM], [REL], [TASK]

    def __init__(
        self,
        baseline_path: str | Path,
        phase1_path: str | Path,
        thresholds: dict[str, dict] | None = None,
        device: str | None = None,
        batch_size: int = 32,
        use_bf16: bool = True,
    ):
        self.baseline_path = Path(baseline_path)
        self.phase1_path = Path(phase1_path)
        self.thresholds = thresholds or V3_FORGETTING_THRESHOLDS
        self.batch_size = batch_size
        self.use_bf16 = use_bf16

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._baseline_model = None
        self._phase1_model = None
        self._tokenizer = None
        self._datasets_cache: dict[str, Any] = {}

    def load_models(self) -> None:
        """Load baseline and Phase 1 models."""
        from transformers import AutoTokenizer

        # Try to load v3 model
        try:
            from modeling_studio.models.modernbert_v3 import ModernBERTv3Ultra

            logger.info(f"Loading baseline model from {self.baseline_path}")
            if self.baseline_path.exists():
                self._baseline_model = ModernBERTv3Ultra.from_pretrained(str(self.baseline_path))
                self._baseline_model.to(self.device)
                self._baseline_model.eval()

            logger.info(f"Loading Phase 1 model from {self.phase1_path}")
            if self.phase1_path.exists():
                self._phase1_model = ModernBERTv3Ultra.from_pretrained(str(self.phase1_path))
                self._phase1_model.to(self.device)
                self._phase1_model.eval()

        except ImportError:
            # Fallback to generic loading
            from transformers import AutoModel

            logger.info("Using generic AutoModel loading")

            if self.baseline_path.exists():
                self._baseline_model = AutoModel.from_pretrained(str(self.baseline_path))
                self._baseline_model.to(self.device)
                self._baseline_model.eval()

            if self.phase1_path.exists():
                self._phase1_model = AutoModel.from_pretrained(str(self.phase1_path))
                self._phase1_model.to(self.device)
                self._phase1_model.eval()

        # Load tokenizer
        if self.phase1_path.exists():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.phase1_path))
            except Exception:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(str(self.baseline_path))
                except Exception:
                    self._tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    def _load_benchmark_dataset(self, benchmark: str) -> Any:
        """Load benchmark dataset with caching."""
        if benchmark in self._datasets_cache:
            return self._datasets_cache[benchmark]

        config = self.thresholds.get(benchmark)
        if not config:
            logger.warning(f"No config for benchmark: {benchmark}")
            return None

        try:
            from datasets import load_dataset

            dataset_name = config["dataset"]
            subset = config.get("subset")
            split = config.get("split", "validation")

            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)

            # Limit size for faster evaluation
            max_samples = 2000
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))

            self._datasets_cache[benchmark] = dataset
            return dataset

        except Exception as e:
            logger.error(f"Failed to load benchmark {benchmark}: {e}")
            return None

    def _evaluate_on_benchmark(
        self,
        model: Any,
        benchmark: str,
    ) -> BenchmarkScore:
        """Evaluate a model on a single benchmark."""
        config = self.thresholds[benchmark]
        metric_name = config["metric"]
        hub_token = config.get("hub_token")

        dataset = self._load_benchmark_dataset(benchmark)
        if dataset is None:
            return BenchmarkScore(
                benchmark=benchmark,
                name=config["name"],
                metric_name=metric_name,
                score=0.0,
                num_samples=0,
                hub_token=hub_token,
                details={"error": "dataset_unavailable"},
            )

        start_time = time.time()
        all_predictions = []
        all_labels = []
        all_scores = []

        # Process in batches
        batch_size = self.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Eval {benchmark}", leave=False):
            batch = dataset[i : i + batch_size]

            # Prepare texts based on benchmark format
            texts = self._prepare_texts(benchmark, batch)
            labels = self._prepare_labels(benchmark, batch)

            if not texts:
                continue

            # Tokenize
            try:
                encoded = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                continue

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Forward pass
            with torch.no_grad():
                if self.use_bf16:
                    with torch.autocast(
                        device_type=self.device.split(":")[0], dtype=torch.bfloat16
                    ):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract predictions
            preds, scores = self._extract_predictions(outputs, benchmark, labels)
            all_predictions.extend(preds)
            all_labels.extend(labels)
            if scores:
                all_scores.extend(scores)

        elapsed_time = time.time() - start_time

        # Compute metric
        score = self._compute_metric(metric_name, all_predictions, all_labels, all_scores)

        return BenchmarkScore(
            benchmark=benchmark,
            name=config["name"],
            metric_name=metric_name,
            score=score,
            num_samples=len(all_labels),
            inference_time_ms=(elapsed_time * 1000) / max(len(all_labels), 1),
            hub_token=hub_token,
        )

    def _prepare_texts(self, benchmark: str, batch: dict) -> list[str]:
        """Prepare texts from batch based on benchmark format."""
        if benchmark == "sst2":
            return batch.get("sentence", [])
        elif benchmark == "mnli":
            premises = batch.get("premise", [])
            hypotheses = batch.get("hypothesis", [])
            return [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
        elif benchmark == "conll":
            tokens_list = batch.get("tokens", [])
            return [" ".join(tokens) for tokens in tokens_list]
        elif benchmark == "squad":
            questions = batch.get("question", [])
            contexts = batch.get("context", [])
            return [f"{q} [SEP] {c}" for q, c in zip(questions, contexts)]
        elif benchmark == "stsb":
            sent1 = batch.get("sentence1", [])
            sent2 = batch.get("sentence2", [])
            return [f"{s1} [SEP] {s2}" for s1, s2 in zip(sent1, sent2)]
        return []

    def _prepare_labels(self, benchmark: str, batch: dict) -> list:
        """Prepare labels from batch."""
        if benchmark in ["sst2", "mnli"]:
            return batch.get("label", [])
        elif benchmark == "conll":
            return batch.get("ner_tags", [])
        elif benchmark == "squad":
            answers = batch.get("answers", [])
            return [a.get("text", [""])[0] if a.get("text") else "" for a in answers]
        elif benchmark == "stsb":
            return batch.get("label", [])
        return []

    def _extract_predictions(self, outputs: Any, benchmark: str, labels: list) -> tuple[list, list]:
        """Extract predictions from model outputs."""
        # Get logits or embeddings
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif hasattr(outputs, "last_hidden_state"):
            logits = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("last_hidden_state", None))
            if logits is not None and logits.dim() == 3:
                logits = logits[:, 0, :]
        else:
            return [], []

        if logits is None:
            return [], []

        scores = []
        if benchmark in ["sst2", "mnli"]:
            # Classification
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        elif benchmark == "conll":
            # Token classification - flatten
            if logits.dim() == 3:
                preds_flat = logits.argmax(dim=-1).cpu().numpy()
                preds = []
                for pred_seq, label_seq in zip(preds_flat, labels):
                    if isinstance(label_seq, (list, np.ndarray)):
                        preds.extend(pred_seq[: len(label_seq)].tolist())
            else:
                preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        elif benchmark == "squad":
            # QA - for now just return dummy predictions
            # Real implementation would use answer span extraction
            preds = ["" for _ in range(logits.size(0))]
        elif benchmark == "stsb":
            # Regression - predict similarity score
            if logits.dim() == 2 and logits.size(-1) == 1:
                preds = logits.squeeze(-1).cpu().numpy().tolist()
            else:
                preds = logits.mean(dim=-1).cpu().numpy().tolist()
            scores = preds
        else:
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()

        return preds, scores

    def _compute_metric(
        self,
        metric_name: str,
        predictions: list,
        labels: list,
        scores: list | None = None,
    ) -> float:
        """Compute metric from predictions and labels."""
        if not predictions or not labels:
            return 0.0

        try:
            if metric_name == "accuracy":
                from sklearn.metrics import accuracy_score

                return float(accuracy_score(labels, predictions))
            elif metric_name == "f1":
                from sklearn.metrics import f1_score

                # Flatten if needed (for NER)
                flat_labels = []
                flat_preds = []
                for l, p in zip(labels, predictions):
                    if isinstance(l, (list, np.ndarray)):
                        flat_labels.extend(l)
                        if isinstance(p, (list, np.ndarray)):
                            flat_preds.extend(p[: len(l)])
                        else:
                            flat_preds.extend([p] * len(l))
                    else:
                        flat_labels.append(l)
                        flat_preds.append(p)

                # Filter invalid
                valid = [(l, p) for l, p in zip(flat_labels, flat_preds) if l >= 0]
                if not valid:
                    return 0.0
                flat_labels, flat_preds = zip(*valid)

                return float(f1_score(flat_labels, flat_preds, average="weighted", zero_division=0))
            elif metric_name == "spearman":
                from scipy.stats import spearmanr

                if scores:
                    corr, _ = spearmanr(scores, labels)
                    return float(corr) if not np.isnan(corr) else 0.0
                return 0.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error computing {metric_name}: {e}")
            return 0.0

    def evaluate_gate(self, benchmark: str) -> ForgettingGateResult:
        """Evaluate a single forgetting gate."""
        if benchmark not in self.thresholds:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        config = self.thresholds[benchmark]
        max_drop = config["max_drop"]
        priority = config.get("priority", "high")
        hub_token = config.get("hub_token")

        # Evaluate baseline
        logger.info(f"Evaluating baseline on {benchmark}...")
        baseline_score = self._evaluate_on_benchmark(self._baseline_model, benchmark)

        # Evaluate Phase 1
        logger.info(f"Evaluating Phase 1 on {benchmark}...")
        phase1_score = self._evaluate_on_benchmark(self._phase1_model, benchmark)

        # Calculate drop (positive = regression)
        drop = baseline_score.score - phase1_score.score
        passed = drop <= max_drop

        return ForgettingGateResult(
            benchmark=benchmark,
            name=config["name"],
            metric_name=config["metric"],
            baseline_score=baseline_score.score,
            phase1_score=phase1_score.score,
            drop=drop,
            max_allowed_drop=max_drop,
            passed=passed,
            priority=priority,
            hub_token=hub_token,
        )

    def run_all_gates(
        self,
        benchmarks: list[str] | None = None,
    ) -> Phase15Report:
        """Run all forgetting gates and generate report."""
        if self._baseline_model is None or self._phase1_model is None:
            self.load_models()

        if benchmarks is None:
            benchmarks = list(self.thresholds.keys())

        start_time = time.time()
        gate_results = []
        critical_failures = []
        high_failures = []

        for benchmark in benchmarks:
            if benchmark not in self.thresholds:
                logger.warning(f"Unknown benchmark: {benchmark}, skipping")
                continue

            result = self.evaluate_gate(benchmark)
            gate_results.append(result)

            if not result.passed:
                if result.priority == "critical":
                    critical_failures.append(benchmark)
                else:
                    high_failures.append(benchmark)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            gate_results, critical_failures, high_failures
        )

        elapsed = time.time() - start_time

        return Phase15Report(
            gate_results=gate_results,
            all_passed=len(critical_failures) == 0 and len(high_failures) == 0,
            critical_failures=critical_failures,
            high_priority_failures=high_failures,
            recommended_actions=recommendations,
            baseline_path=str(self.baseline_path),
            phase1_path=str(self.phase1_path),
            total_evaluation_time_s=elapsed,
        )

    def _generate_recommendations(
        self,
        results: list[ForgettingGateResult],
        critical: list[str],
        high: list[str],
    ) -> list[str]:
        """Generate remediation recommendations based on failures."""
        if not critical and not high:
            return ["No remediation needed - all gates passed"]

        recommendations = []

        # Analyze failure patterns
        failed_hubs = set()
        for result in results:
            if not result.passed and result.hub_token:
                failed_hubs.add(result.hub_token)

        # General recommendations
        if critical:
            recommendations.append(
                f"CRITICAL: Re-run Phase 1 with increased replay ratio (current: 15%, suggested: 25%)"
            )
            recommendations.append(f"Add task-specific replay samples for: {', '.join(critical)}")

        if "[EMO]" in failed_hubs:
            recommendations.append(
                "Sentiment/Emotion forgetting: Increase SST-2 and emotion data in replay"
            )

        if "[REL]" in failed_hubs:
            recommendations.append("NLI forgetting: Add more MNLI samples to Phase 1 training")

        if "conll" in critical:
            recommendations.append("NER forgetting: Consider task-specific LoRA adapters for NER")

        # Severe forgetting
        max_drop = (
            max(r.drop for r in results if not r.passed)
            if any(not r.passed for r in results)
            else 0
        )
        if max_drop > 0.10:
            recommendations.append(
                "SEVERE FORGETTING: Consider freezing more layers (L1-20 instead of L1-18)"
            )
            recommendations.append("SEVERE FORGETTING: Reduce LoRA rank from 16 to 8")

        return recommendations


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_baseline_cache(cache_path: Path | str) -> dict[str, float] | None:
    """Load cached baseline scores."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    with open(cache_path) as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1.5 Forgetting Evaluation for ModernBERT v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/forgetting_gate.yaml",
        help="Path to forgetting gate configuration",
    )

    # Model paths (can override config)
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline model (Phase 0.5)",
    )
    parser.add_argument(
        "--phase1",
        "--model-path",
        type=str,
        default=None,
        dest="phase1",
        help="Path to Phase 1 model",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v3_phase1",
        help="Output directory for results",
    )

    # Benchmarks
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to evaluate (default: all)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )

    # Other options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--baseline-cache",
        type=str,
        default=None,
        help="Path to cached baseline scores",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="phase1_5_forgetting_gate",
        help="W&B run name",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Phase 1.5 Forgetting Evaluation - ModernBERT v3")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Determine model paths
    eval_config = config.get("evaluation", {})
    baseline_path = args.baseline or eval_config.get(
        "baseline_path", "outputs/v3_phase0_5/best_model"
    )
    phase1_path = args.phase1 or eval_config.get("model_path", "outputs/v3_phase1/final_model")

    logger.info(f"Baseline: {baseline_path}")
    logger.info(f"Phase 1:  {phase1_path}")

    # Check paths exist
    if not Path(baseline_path).exists():
        logger.error(f"Baseline path not found: {baseline_path}")
        return 1

    if not Path(phase1_path).exists():
        logger.error(f"Phase 1 path not found: {phase1_path}")
        return 1

    # Determine benchmarks
    benchmarks = args.benchmarks
    if benchmarks is None:
        benchmarks_config = config.get("benchmarks", {})
        if benchmarks_config:
            benchmarks = list(benchmarks_config.keys())
        else:
            benchmarks = list(V3_FORGETTING_THRESHOLDS.keys())

    logger.info(f"Benchmarks: {benchmarks}")

    # Build thresholds from config
    thresholds = V3_FORGETTING_THRESHOLDS.copy()
    if "benchmarks" in config:
        for bench, bench_config in config["benchmarks"].items():
            if bench in thresholds:
                thresholds[bench].update(bench_config)

    # Create evaluator
    evaluator = Phase15ForgettingEvaluator(
        baseline_path=baseline_path,
        phase1_path=phase1_path,
        thresholds=thresholds,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Run evaluation
    logger.info("Running forgetting gates...")
    report = evaluator.run_all_gates(benchmarks=benchmarks)

    # Print summary
    print()
    print(report.summary())

    # Save results
    output_dir = Path(args.output_dir)
    report.save(output_dir)

    # Return appropriate exit code
    if report.all_passed:
        logger.info("All forgetting gates PASSED")
        return 0
    else:
        logger.error(
            f"Forgetting gates FAILED: {report.critical_failures + report.high_priority_failures}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Evaluation Module

This module provides evaluation infrastructure for assessing
model performance across all tasks.

Components:
    - metrics: Metric computation for each task type
    - evaluator: Evaluation pipeline runner
    - benchmarks: Standardized benchmark suite
    - safety_eval: Specialized safety evaluation

Evaluation Workflow:
    1. Load trained model checkpoint
    2. Load test datasets
    3. Run inference on all tasks
    4. Compute per-task metrics
    5. Generate aggregate report
    6. Compare against baselines

Key Metrics:
    - NER: Entity-level F1
    - Classification: Accuracy, Macro-F1
    - Multi-label: Micro-F1, Hamming loss
    - Embedding: Spearman correlation, Recall@K
    - Safety: FNR, calibration metrics

Reports:
    - JSON: Full metrics export
    - Markdown: Summary tables
    - HTML: Interactive dashboard (optional)
"""

# TODO: Export metrics
# from modeling_studio.evaluation.metrics import (
#     compute_ner_metrics,
#     compute_classification_metrics,
#     compute_multilabel_metrics,
#     compute_embedding_metrics,
#     compute_all_metrics,
# )

# TODO: Export evaluator
# from modeling_studio.evaluation.evaluator import Evaluator

# TODO: Export benchmarks
# from modeling_studio.evaluation.benchmarks import BenchmarkSuite

# TODO: Export safety evaluation
# from modeling_studio.evaluation.safety_eval import SafetyEvaluator

# Catastrophic forgetting evaluation
# Cultural robustness evaluation
from modeling_studio.evaluation.cultural_robustness import (
    INDIAN_VENTING_PATTERNS,
    CrisisRecallResult,
    CulturalRobustnessEvaluator,
    CulturalRobustnessResult,
    evaluate_cultural_robustness,
)
from modeling_studio.evaluation.forgetting_eval import (
    FORGETTING_THRESHOLDS,
    ForgettingEvaluator,
    ForgettingReport,
    ForgettingResult,
    check_forgetting_gates,
    evaluate_forgetting,
)

# Temporal safety monitoring
from modeling_studio.evaluation.temporal_safety import (
    ESCALATION_KEYWORDS,
    ISOLATION_KEYWORDS,
    EscalationRule,
    SafetyBand,
    SafetyEscalation,
    SafetyMonitorIntegration,
    SafetySignal,
    TemporalSafetyMonitor,
    create_monitor,
)

__all__ = [
    # Forgetting evaluation
    "ForgettingEvaluator",
    "ForgettingReport",
    "ForgettingResult",
    "evaluate_forgetting",
    "check_forgetting_gates",
    "FORGETTING_THRESHOLDS",
    # Cultural robustness
    "CulturalRobustnessEvaluator",
    "CulturalRobustnessResult",
    "CrisisRecallResult",
    "evaluate_cultural_robustness",
    "INDIAN_VENTING_PATTERNS",
    # Temporal safety
    "TemporalSafetyMonitor",
    "SafetySignal",
    "SafetyEscalation",
    "SafetyBand",
    "EscalationRule",
    "SafetyMonitorIntegration",
    "create_monitor",
    "ISOLATION_KEYWORDS",
    "ESCALATION_KEYWORDS",
]

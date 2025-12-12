"""Golden outputs for regression benchmarks.

Milestone 1: placeholder module.
Later milestones will populate this with minimal, stable expectations.

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict


# Minimal set of stable expectations.
# Keep these conservative to reduce flakiness across backends.
GOLDEN_OUTPUTS: Dict[str, Dict[str, Any]] = {
	"Mom picked up the kids from school": {
		"sentiment": "positive",
		"safety": "GREEN",
		"emotions_contain": ["joy"],
		"entities_contain": ["Mom"],
	},
	"I love my family so much": {
		"sentiment": "very_positive",
		"safety": "GREEN",
		"emotions_contain": ["love"],
	},
}


# Optional numeric baselines for embedding/retrieval behavior.
# These are used by RegressionSuite to detect regressions in embedding quality.
GOLDEN_EMBEDDING_METRICS: Dict[str, float] = {
	"recall_at_1_10_distractors_min": 0.90,
	"recall_at_1_100_distractors_min": 0.80,
	"recall_at_10_100_distractors_min": 0.95,
}

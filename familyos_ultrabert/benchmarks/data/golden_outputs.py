"""Golden outputs for regression benchmarks.

These expectations are intentionally conservative to avoid flakiness across:
- backends (PyTorch vs ONNX)
- devices (CPU vs GPU)
- minor model/version drift

This module is standard-library only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from familyos_ultrabert.benchmarks.data.test_cases import (
	CRISIS_CASES,
	EMOTION_CASES,
	SAFETY_BAND_CASES,
	SENTIMENT_CASES,
	SENTIMENT_CASES_EXPANDED,
)


# Minimal set of stable expectations.
# Keep these conservative to reduce flakiness across backends.
# -----------------------------------------------------------------------------
# Legacy minimal dict format (kept for backwards compatibility / determinism)
# -----------------------------------------------------------------------------

GOLDEN_OUTPUTS: Dict[str, Dict[str, Any]] = {
	"Mom picked up the kids from school": {
		"sentiment_direction": "positive",
		"safety": "GREEN",
		"emotions_super_any_of": ["AFFECTION", "JOY"],
		"entities_contain": ["Mom"],
	},
}


# -----------------------------------------------------------------------------
# World-class regression gates: larger, stratified labeled sets
# -----------------------------------------------------------------------------

# Sentiment: use valence/direction for stability.
GOLDEN_SENTIMENT_CASES: List[Tuple[str, str]] = list(SENTIMENT_CASES) + list(SENTIMENT_CASES_EXPANDED)

# Additional direction-only cases to boost statistical power without relying on
# brittle 5-class boundaries.
_SENTIMENT_POSITIVE: List[str] = [
	"I am thrilled we had dinner together",
	"What a wonderful family weekend",
	"I feel grateful for your help",
	"I am proud of how the kids handled today",
	"That was a great conversation with Mom",
	"I love spending time with the family",
	"I feel relieved and happy now",
	"This made my day",
	"Everything went really well",
	"I feel supported and cared for",
	"We had such a fun time at the park",
	"I am so excited about the holiday plans",
	"Thank you, that means a lot",
	"I feel calm and content",
	"This is fantastic news",
	"I am feeling optimistic",
	"That was the best part of my day",
	"I feel appreciated",
	"I am really enjoying this",
	"I am genuinely happy",
]

_SENTIMENT_NEUTRAL: List[str] = [
	"The appointment went fine",
	"We have a meeting on Tuesday",
	"I took the kids to school",
	"Dinner is at 6pm",
	"I will call you tomorrow",
	"The package arrived today",
	"We went to the store",
	"The report is due on Friday",
	"I have a doctor visit next week",
	"The car is parked outside",
	"The kids are at practice",
	"I updated the calendar",
	"We need to buy groceries",
	"I sent the email",
	"We are leaving at 5pm",
	"The meeting ended at 3pm",
	"We watched a movie",
	"I completed the form",
	"I have work tomorrow",
	"The weather is cloudy",
]

_SENTIMENT_NEGATIVE: List[str] = [
	"I am upset about what happened",
	"I feel hopeless today",
	"This is awful and frustrating",
	"I am disappointed in the outcome",
	"I feel overwhelmed right now",
	"I am stressed about work",
	"I am worried about the kids",
	"I feel anxious",
	"This is really hard",
	"I am angry about this",
	"I feel sad today",
	"I feel exhausted and burned out",
	"This situation is miserable",
	"I regret what I said",
	"I feel embarrassed",
	"I am scared",
	"I am frustrated",
	"I feel lonely",
	"I feel like nothing is working",
	"I am not okay today",
]

GOLDEN_SENTIMENT_DIRECTION_CASES: List[Tuple[str, str]] = (
	[(t, "positive") for t in _SENTIMENT_POSITIVE]
	+ [(t, "neutral") for t in _SENTIMENT_NEUTRAL]
	+ [(t, "negative") for t in _SENTIMENT_NEGATIVE]
)

# Safety: hard-gate crisis recall, plus broader band coverage.
GOLDEN_SAFETY_BAND_CASES: List[Tuple[str, str]] = list(SAFETY_BAND_CASES)
GOLDEN_SAFETY_CRISIS_CASES: List[Tuple[str, str]] = list(CRISIS_CASES)

# Emotions: multi-label expectations.
GOLDEN_EMOTION_CASES: List[Tuple[str, List[str]]] = list(EMOTION_CASES)


# Optional numeric baselines for embedding/retrieval behavior.
# These are used by RegressionSuite to detect regressions in embedding quality.
GOLDEN_EMBEDDING_METRICS: Dict[str, float] = {
	"recall_at_1_10_distractors_min": 0.90,
	"recall_at_1_100_distractors_min": 0.80,
	"recall_at_10_100_distractors_min": 0.95,
}

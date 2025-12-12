"""Classification benchmark suite (sentiment, emotions, intent).

Implements:
- Issue #10: 5-class sentiment checks

Constraint: standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.test_cases import EMOTION_CASES, INTENT_CASES, SENTIMENT_CASES
from familyos_ultrabert.benchmarks.suite import register_suite
from familyos_ultrabert.labels import INTENT_LABELS


def _direction(label: str) -> str:
	if label in ("very_positive", "positive"):
		return "positive"
	if label in ("very_negative", "negative"):
		return "negative"
	return "neutral"


def _normalize_emotion_labels(values: Any) -> List[str]:
	"""Normalize a predicted emotions payload into lowercase label strings.

	Args:
		values: Predicted emotions payload.

	Returns:
		List of normalized emotion labels.
	"""
	if values is None:
		return []
	if isinstance(values, str):
		return [values.strip().lower()] if values.strip() else []
	if isinstance(values, dict):
		# Some APIs might return a score dict; keep keys as labels.
		return [str(k).strip().lower() for k in values.keys() if str(k).strip()]
	if isinstance(values, Iterable):
		out: List[str] = []
		for v in values:
			if v is None:
				continue
			label = str(v).strip().lower()
			if label:
				out.append(label)
		return out
	return [str(values).strip().lower()] if str(values).strip() else []


@register_suite
class ClassificationSuite(BenchmarkSuite):
	"""Classification suite (Milestone 3 starts with sentiment)."""

	name: str = "classification"
	description: str = "Sentiment (and later emotions/intent) benchmarks"

	_DIRECTION_THRESHOLD: float = 0.80
	_EMOTION_HIT_RATE_THRESHOLD: float = 0.85
	_INTENT_VALID_RATE_THRESHOLD: float = 1.00

	def run(self) -> List[Any]:
		# ---------------------------------------------------------------------
		# Sentiment
		# ---------------------------------------------------------------------
		total = len(SENTIMENT_CASES)
		correct_5 = 0
		correct_dir = 0
		mismatches: List[Dict[str, Any]] = []

		for text, expected in SENTIMENT_CASES:
			try:
				pred = str(self.client.get_sentiment(text))
				if pred == expected:
					correct_5 += 1
				else:
					mismatches.append({"text": text, "expected": expected, "predicted": pred})

				if _direction(pred) == _direction(expected):
					correct_dir += 1
			except Exception as exc:  # noqa: BLE001
				mismatches.append({"text": text, "expected": expected, "error": f"{type(exc).__name__}: {exc}"})

		acc_5 = (float(correct_5) / float(total)) if total > 0 else 0.0
		acc_dir = (float(correct_dir) / float(total)) if total > 0 else 0.0

		# 5-class accuracy is reported (non-gating).
		self.add_result(
			name="sentiment_5class_accuracy",
			passed=True,
			score=acc_5,
			details={"total": total, "correct": correct_5, "mismatches": mismatches},
		)

		# Direction accuracy gating per plan (>80%).
		self.add_result(
			name="sentiment_direction_accuracy",
			passed=(acc_dir >= self._DIRECTION_THRESHOLD and total > 0),
			score=acc_dir,
			threshold=self._DIRECTION_THRESHOLD,
			details={"total": total, "correct": correct_dir},
		)

		# ---------------------------------------------------------------------
		# Emotions (multi-label)
		# ---------------------------------------------------------------------
		emotion_total = len(EMOTION_CASES)
		emotion_hits = 0
		emotion_details: List[Dict[str, Any]] = []

		for text, expected_emotions in EMOTION_CASES:
			try:
				pred_raw = self.client.get_emotions(text)
				pred = set(_normalize_emotion_labels(pred_raw))
				expected = set(_normalize_emotion_labels(expected_emotions))
				hit = len(pred.intersection(expected)) > 0
				if hit:
					emotion_hits += 1
				emotion_details.append(
					{
						"text": text,
						"expected": sorted(expected),
						"predicted": sorted(pred),
						"hit": hit,
					}
				)
			except Exception as exc:  # noqa: BLE001
				emotion_details.append(
					{
						"text": text,
						"expected": list(expected_emotions),
						"error": f"{type(exc).__name__}: {exc}",
					}
				)

		emotion_hit_rate = (
			(float(emotion_hits) / float(emotion_total)) if emotion_total > 0 else 0.0
		)
		self.add_result(
			name="emotions_hit_rate",
			passed=(emotion_hit_rate >= self._EMOTION_HIT_RATE_THRESHOLD and emotion_total > 0),
			score=emotion_hit_rate,
			threshold=self._EMOTION_HIT_RATE_THRESHOLD,
			details={"total": emotion_total, "hits": emotion_hits, "cases": emotion_details},
		)

		# ---------------------------------------------------------------------
		# Intent (single-label)
		# ---------------------------------------------------------------------
		intent_total = len(INTENT_CASES)
		intent_valid = 0
		intent_correct = 0
		intent_cases: List[Dict[str, Any]] = []
		valid_intents = set(INTENT_LABELS.label2id.keys())

		for text, expected in INTENT_CASES:
			try:
				pred = str(self.client.get_intent(text))
				is_valid = pred in valid_intents and pred != "unknown"
				if is_valid:
					intent_valid += 1
				if pred == expected:
					intent_correct += 1
				intent_cases.append(
					{
						"text": text,
						"expected": expected,
						"predicted": pred,
						"valid": is_valid,
					},
				)
			except Exception as exc:  # noqa: BLE001
				intent_cases.append(
					{
						"text": text,
						"expected": expected,
						"error": f"{type(exc).__name__}: {exc}",
					},
				)

		intent_valid_rate = (float(intent_valid) / float(intent_total)) if intent_total > 0 else 0.0
		intent_accuracy = (float(intent_correct) / float(intent_total)) if intent_total > 0 else 0.0

		# Gating: the intent head must be functional (valid label for all cases).
		self.add_result(
			name="intent_valid_label_rate",
			passed=(intent_valid_rate >= self._INTENT_VALID_RATE_THRESHOLD and intent_total > 0),
			score=intent_valid_rate,
			threshold=self._INTENT_VALID_RATE_THRESHOLD,
			details={
				"total": intent_total,
				"valid": intent_valid,
				"valid_intents": sorted(valid_intents),
				"cases": intent_cases,
			},
		)

		# Report-only: accuracy can fluctuate across model versions and domains.
		self.add_result(
			name="intent_accuracy_report",
			passed=True,
			score=intent_accuracy,
			details={"total": intent_total, "correct": intent_correct},
		)

		return self.results

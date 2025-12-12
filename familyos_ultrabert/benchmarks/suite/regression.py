"""Regression benchmark suite (golden outputs, determinism).

Implements:
- Issue #20: Golden outputs / determinism checks

Notes:
- This suite aims to be standard-library only for its own logic.
- If optional dependencies (e.g., torch) are available at runtime, we may use
	them for best-effort seeding to reduce flakiness.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import math

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.data.golden_outputs import (
	GOLDEN_EMBEDDING_METRICS,
	GOLDEN_EMOTION_CASES,
	GOLDEN_OUTPUTS,
	GOLDEN_SAFETY_BAND_CASES,
	GOLDEN_SAFETY_CRISIS_CASES,
	GOLDEN_SENTIMENT_CASES,
	GOLDEN_SENTIMENT_DIRECTION_CASES,
)
from familyos_ultrabert.benchmarks.data.test_cases import RETRIEVAL_CASES_10, RETRIEVAL_CASES_100
from familyos_ultrabert.benchmarks.suite import register_suite
from familyos_ultrabert.benchmarks.types import BenchmarkSeverity


def _set_best_effort_seed(seed: int) -> Dict[str, Any]:
	"""Set seeds for common RNGs (best-effort).

	This suite is stdlib-only, but the runtime may have torch installed.
	We attempt to seed it to improve determinism for regression checks.

	Args:
		seed: Seed integer.

	Returns:
		A dict describing which libraries were seeded.
	"""
	random.seed(seed)
	info: Dict[str, Any] = {"seed": seed, "random": True}

	# torch is optional
	try:
		import torch  # type: ignore

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
		# Keep it conservative: only set deterministic algorithms when available.
		try:
			torch.use_deterministic_algorithms(True)
			info["torch_deterministic_algorithms"] = True
		except Exception:  # noqa: BLE001
			info["torch_deterministic_algorithms"] = False
		info["torch"] = True
	except Exception:  # noqa: BLE001
		info["torch"] = False

	# Note: PYTHONHASHSEED must be set before interpreter start.
	info["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
	return info


def _entity_texts(entities: Any) -> List[str]:
	"""Extract entity text spans from a list-like structure."""
	if not isinstance(entities, list):
		return []
	out: List[str] = []
	for e in entities:
		if isinstance(e, dict):
			val = str(e.get("text", e.get("entity", ""))).strip()
			if val:
				out.append(val)
		else:
			val = str(e).strip()
			if val:
				out.append(val)
	return out


def _cosine_similarity(a: List[float], b: List[float]) -> float:
	"""Compute cosine similarity; returns 0.0 if a or b has zero norm."""
	norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
	norm_b = math.sqrt(sum(float(x) * float(x) for x in b))
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	dot = sum(float(x) * float(y) for x, y in zip(a, b))
	return float(dot / (norm_a * norm_b))


def _recall_at_k(client: Any, cases: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
	"""Compute Recall@K for retrieval cases using cosine similarity."""
	total = 0
	hits = 0
	failures: List[Dict[str, Any]] = []

	# Simple memoization to avoid recomputing embeddings.
	cache: Dict[str, List[float]] = {}

	def embed(text: str) -> List[float]:
		if text not in cache:
			cache[text] = list(client.get_embedding(text))
		return cache[text]

	for rc in cases:
		query = str(rc.get("query", ""))
		relevant = str(rc.get("relevant", ""))
		distractors = list(rc.get("distractors", []))
		if not query or not relevant:
			continue
		total += 1

		q = embed(query)
		candidates = [relevant] + [str(d) for d in distractors]
		scored = [(t, _cosine_similarity(q, embed(t))) for t in candidates]
		scored.sort(key=lambda x: x[1], reverse=True)
		topk = [t for (t, _s) in scored[: max(1, int(k))]]
		if relevant in topk:
			hits += 1
		else:
			failures.append({"query": query, "relevant": relevant, "topk": topk})

	acc = (float(hits) / float(total)) if total > 0 else 0.0
	return {"recall": acc, "total": total, "hits": hits, "failures": failures}


def _sentiment_direction(label: str) -> str:
	if label in ("very_positive", "positive"):
		return "positive"
	if label in ("very_negative", "negative"):
		return "negative"
	return "neutral"


def _accuracy_from_boolean_cases(cases: List[Tuple[str, bool]]) -> Dict[str, Any]:
	"""Compute accuracy details from (id, ok) cases."""
	total = len(cases)
	ok_count = sum(1 for _id, ok in cases if ok)
	acc = (float(ok_count) / float(total)) if total > 0 else 0.0
	return {"total": total, "passed": ok_count, "accuracy": acc}


def _emotion_super_labels(emotion_names: List[str]) -> List[str]:
	"""Map granular emotion names to stable super-label categories.

	Args:
		emotion_names: List of granular emotion labels.

	Returns:
		Sorted list of unique super-label names (e.g., ["AFFECTION", "JOY"]).
	"""
	try:
		from familyos_ultrabert.data.labels import EMOTION_TO_SUPER_LABEL  # type: ignore
	except Exception:  # noqa: BLE001
		return []

	supers: set[str] = set()
	for emo in emotion_names:
		key = str(emo).strip().lower()
		if not key:
			continue
		sl = EMOTION_TO_SUPER_LABEL.get(key)
		if sl:
			supers.add(str(sl))
	return sorted(supers)


@register_suite
class RegressionSuite(BenchmarkSuite):
	"""Golden output regression suite."""

	name: str = "regression"
	description: str = "Golden output checks for sentiment/safety/emotions/entities"

	def run(self) -> List["BenchmarkResult"]:
		quick_mode = bool(os.environ.get("FAMILYOS_ULTRABERT_BENCH_QUICK", "").strip())
		quick_gate_severity = BenchmarkSeverity.WARN if quick_mode else BenchmarkSeverity.FAIL

		seed_str: Optional[str] = os.environ.get("ULTRABERT_BENCHMARK_SEED")
		seed = int(seed_str) if seed_str is not None and seed_str.strip() else 1234
		self.add_result(
			name="determinism_seed_applied",
			passed=True,
			details=_set_best_effort_seed(seed),
		)

		# World-class labeled sets should be present; legacy GOLDEN_OUTPUTS is optional.
		if not (GOLDEN_SENTIMENT_DIRECTION_CASES or GOLDEN_SENTIMENT_CASES or GOLDEN_SAFETY_BAND_CASES or GOLDEN_EMOTION_CASES):
			self.add_skipped(name="golden_sets_present", reason="no golden labeled sets configured")
			return self.results

		# ------------------------------------------------------------------
		# Sentiment direction (hard gate; large set)
		# ------------------------------------------------------------------
		try:
			failures: List[Dict[str, Any]] = []
			checks: List[Tuple[str, bool]] = []
			for text, expected_dir in GOLDEN_SENTIMENT_DIRECTION_CASES:
				res = self.client.analyze(text, capabilities=["sentiment"])
				obs = str(getattr(res, "sentiment", ""))
				obs_dir = _sentiment_direction(obs)
				ok = (obs_dir == str(expected_dir))
				checks.append((text, ok))
				if not ok:
					failures.append({"text": text, "expected": expected_dir, "observed": obs, "observed_direction": obs_dir})

			details = _accuracy_from_boolean_cases(checks)
			details["failures"] = failures
			min_acc = 0.85
			self.add_result(
				name="golden_sentiment_direction_accuracy",
				passed=(float(details["accuracy"]) >= min_acc and int(details["total"]) > 0),
				score=float(details["accuracy"]),
				threshold=min_acc,
				details=details,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="golden_sentiment_direction_accuracy", error=f"{type(exc).__name__}: {exc}")

		# ------------------------------------------------------------------
		# Sentiment (5-class) exact label (signal-only; small set)
		# ------------------------------------------------------------------
		try:
			failures = []
			checks = []
			for text, expected_label in GOLDEN_SENTIMENT_CASES:
				res = self.client.analyze(text, capabilities=["sentiment"])
				obs = str(getattr(res, "sentiment", ""))
				ok = (obs == str(expected_label))
				checks.append((text, ok))
				if not ok:
					failures.append({"text": text, "expected": expected_label, "observed": obs})

			details = _accuracy_from_boolean_cases(checks)
			details["failures"] = failures
			min_acc = 0.60
			self.add_result(
				name="golden_sentiment_5class_accuracy",
				passed=(float(details["accuracy"]) >= min_acc and int(details["total"]) > 0),
				score=float(details["accuracy"]),
				threshold=min_acc,
				details=details,
				severity=BenchmarkSeverity.WARN,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="golden_sentiment_5class_accuracy", error=f"{type(exc).__name__}: {exc}")

		# ------------------------------------------------------------------
		# Safety band accuracy (hard gate) + crisis recall (hard gate)
		# ------------------------------------------------------------------
		try:
			failures = []
			checks = []
			for text, expected_band in GOLDEN_SAFETY_BAND_CASES:
				res = self.client.analyze(text, capabilities=["safety_familyos"])
				obs = str(getattr(res, "safety", ""))
				ok = (obs == str(expected_band))
				checks.append((text, ok))
				if not ok:
					failures.append({"text": text, "expected": expected_band, "observed": obs})
			details = _accuracy_from_boolean_cases(checks)
			details["failures"] = failures
			min_acc = 0.80
			self.add_result(
				name="golden_safety_band_accuracy",
				passed=(float(details["accuracy"]) >= min_acc and int(details["total"]) > 0),
				score=float(details["accuracy"]),
				threshold=min_acc,
				details=details,
				severity=quick_gate_severity,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="golden_safety_band_accuracy", error=f"{type(exc).__name__}: {exc}")

		try:
			failures = []
			checks = []
			crisis_cases = [(t, y) for (t, y) in GOLDEN_SAFETY_CRISIS_CASES if str(y) == "CRISIS"]
			for text, _expected in crisis_cases:
				res = self.client.analyze(text, capabilities=["safety_familyos"])
				obs = str(getattr(res, "safety", ""))
				ok = (obs == "CRISIS")
				checks.append((text, ok))
				if not ok:
					failures.append({"text": text, "expected": "CRISIS", "observed": obs})
			details = _accuracy_from_boolean_cases(checks)
			details["failures"] = failures
			details["note"] = "Recall on CRISIS-only cases (GREEN examples are excluded)."
			min_acc = 1.00
			self.add_result(
				name="golden_safety_crisis_recall",
				passed=(float(details["accuracy"]) >= min_acc and int(details["total"]) > 0),
				score=float(details["accuracy"]),
				threshold=min_acc,
				details=details,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="golden_safety_crisis_recall", error=f"{type(exc).__name__}: {exc}")

		# ------------------------------------------------------------------
		# Emotions (super-label hit-rate; signal-only)
		# ------------------------------------------------------------------
		try:
			failures = []
			checks = []
			for text, expected_emotions in GOLDEN_EMOTION_CASES:
				res = self.client.analyze(text, capabilities=["emotions"])
				obs_emos = [str(e).strip().lower() for e in list(getattr(res, "emotions", []))]
				obs_supers = _emotion_super_labels(obs_emos)
				exp_supers = _emotion_super_labels([str(e).strip().lower() for e in list(expected_emotions)])
				ok = bool(exp_supers) and any(s in obs_supers for s in exp_supers)
				checks.append((text, ok))
				if not ok:
					failures.append({"text": text, "expected_super_any_of": exp_supers, "observed_super": obs_supers, "observed": obs_emos})
			details = _accuracy_from_boolean_cases(checks)
			details["failures"] = failures
			min_acc = 0.70
			self.add_result(
				name="golden_emotions_superlabel_hit_rate",
				passed=(float(details["accuracy"]) >= min_acc and int(details["total"]) > 0),
				score=float(details["accuracy"]),
				threshold=min_acc,
				details=details,
				severity=BenchmarkSeverity.WARN,
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="golden_emotions_superlabel_hit_rate", error=f"{type(exc).__name__}: {exc}")

		# ------------------------------------------------------------------
		# Legacy exact-match expectations (kept for determinism debugging)
		# ------------------------------------------------------------------
		if GOLDEN_OUTPUTS:
			failures = []
			passed = 0
			total = 0

			for text, expected in GOLDEN_OUTPUTS.items():
				total += 1
				try:
					capabilities = ["sentiment", "safety_familyos", "emotions", "ner_family"]
					res = self.client.analyze(text, capabilities=capabilities)

					exp_sent = expected.get("sentiment")
					exp_sent_in = list(expected.get("sentiment_in", []))
					exp_sent_dir = expected.get("sentiment_direction")
					exp_safe = expected.get("safety")
					exp_emos = list(expected.get("emotions_contain", []))
					exp_emos_any = list(expected.get("emotions_any_of", []))
					exp_emos_super_any = list(expected.get("emotions_super_any_of", []))
					exp_entities = list(expected.get("entities_contain", []))

					obs_sent = str(getattr(res, "sentiment", ""))
					obs_safe = str(getattr(res, "safety", ""))
					obs_emos = [str(e).strip().lower() for e in list(getattr(res, "emotions", []))]
					obs_emo_supers = _emotion_super_labels(obs_emos)
					obs_entities = _entity_texts(getattr(res, "entities", []))

					ok = True
					mismatch: Dict[str, Any] = {"text": text}

					if exp_sent is not None and obs_sent != str(exp_sent):
						ok = False
						mismatch["sentiment"] = {"expected": exp_sent, "observed": obs_sent}
					if exp_sent_in and obs_sent not in [str(x) for x in exp_sent_in]:
						ok = False
						mismatch["sentiment_in"] = {"expected": exp_sent_in, "observed": obs_sent}
					if exp_sent_dir is not None and _sentiment_direction(obs_sent) != str(exp_sent_dir):
						ok = False
						mismatch["sentiment_direction"] = {
							"expected": exp_sent_dir,
							"observed": _sentiment_direction(obs_sent),
							"observed_label": obs_sent,
						}
					if exp_safe is not None and obs_safe != str(exp_safe):
						ok = False
						mismatch["safety"] = {"expected": exp_safe, "observed": obs_safe}

					missing_emos = [e for e in exp_emos if str(e).strip().lower() not in obs_emos]
					if missing_emos:
						ok = False
						mismatch["emotions_missing"] = missing_emos
						mismatch["emotions_observed"] = obs_emos

					if exp_emos_any:
						norm_any = [str(e).strip().lower() for e in exp_emos_any if str(e).strip()]
						if norm_any and not any(e in obs_emos for e in norm_any):
							ok = False
							mismatch["emotions_any_of_missing"] = norm_any
							mismatch["emotions_observed"] = obs_emos

					if exp_emos_super_any:
						norm_sup = [str(e).strip() for e in exp_emos_super_any if str(e).strip()]
						if norm_sup and not any(s in obs_emo_supers for s in norm_sup):
							ok = False
							mismatch["emotions_super_any_of_missing"] = norm_sup
							mismatch["emotions_super_observed"] = obs_emo_supers

					missing_entities: List[str] = []
					for ent in exp_entities:
						needle = str(ent).strip()
						if not needle:
							continue
						if not any(needle.lower() in str(o).lower() for o in obs_entities):
							missing_entities.append(needle)
					if missing_entities:
						ok = False
						mismatch["entities_missing"] = missing_entities
						mismatch["entities_observed"] = obs_entities

					if ok:
						passed += 1
					else:
						failures.append(mismatch)
				except Exception as exc:  # noqa: BLE001
					failures.append({"text": text, "error": f"{type(exc).__name__}: {exc}"})

			acc = (float(passed) / float(total)) if total > 0 else 0.0
			self.add_result(
				name="golden_outputs_match",
				passed=(passed == total and total > 0),
				score=acc,
				details={"total": total, "passed": passed, "failures": failures},
				severity=BenchmarkSeverity.INFO,
			)
		else:
			self.add_skipped(name="golden_outputs_match", reason="legacy GOLDEN_OUTPUTS empty")

		# Embedding/retrieval metric minimums (ported from verify_embedding_benchmarks style checks)
		if GOLDEN_EMBEDDING_METRICS:
			try:
				r10 = _recall_at_k(self.client, RETRIEVAL_CASES_10, k=1)
				r100_1 = _recall_at_k(self.client, RETRIEVAL_CASES_100, k=1)
				r100_10 = _recall_at_k(self.client, RETRIEVAL_CASES_100, k=10)

				min_1_10 = float(GOLDEN_EMBEDDING_METRICS.get("recall_at_1_10_distractors_min", 0.0))
				min_1_100 = float(GOLDEN_EMBEDDING_METRICS.get("recall_at_1_100_distractors_min", 0.0))
				min_10_100 = float(GOLDEN_EMBEDDING_METRICS.get("recall_at_10_100_distractors_min", 0.0))

				self.add_result(
					name="golden_embedding_recall_at1_10d",
					passed=(r10["recall"] >= min_1_10 and r10["total"] > 0),
					score=float(r10["recall"]),
					threshold=min_1_10,
					details=r10,
					severity=quick_gate_severity,
				)
				self.add_result(
					name="golden_embedding_recall_at1_100d",
					passed=(r100_1["recall"] >= min_1_100 and r100_1["total"] > 0),
					score=float(r100_1["recall"]),
					threshold=min_1_100,
					details=r100_1,
					severity=quick_gate_severity,
				)
				self.add_result(
					name="golden_embedding_recall_at10_100d",
					passed=(r100_10["recall"] >= min_10_100 and r100_10["total"] > 0),
					score=float(r100_10["recall"]),
					threshold=min_10_100,
					details=r100_10,
					severity=quick_gate_severity,
				)
			except Exception as exc:  # noqa: BLE001
				self.add_error(name="golden_embedding_metrics", error=f"{type(exc).__name__}: {exc}")
		else:
			self.add_skipped(name="golden_embedding_metrics", reason="no golden embedding metrics configured")

		# Basic determinism check: same text, same outputs (best-effort).
		try:
			sample_text = GOLDEN_SENTIMENT_DIRECTION_CASES[0][0] if GOLDEN_SENTIMENT_DIRECTION_CASES else next(iter(GOLDEN_OUTPUTS.keys()))
			a = self.client.analyze(sample_text, capabilities=["sentiment", "safety_familyos", "emotions"])
			b = self.client.analyze(sample_text, capabilities=["sentiment", "safety_familyos", "emotions"])

			def _norm_emotions(val: Any) -> List[str]:
				if not isinstance(val, list):
					return []
				return sorted({str(e).strip().lower() for e in val if str(e).strip()})

			emo_a = _norm_emotions(getattr(a, "emotions", []))
			emo_b = _norm_emotions(getattr(b, "emotions", []))
			same = (
				str(getattr(a, "sentiment", "")) == str(getattr(b, "sentiment", ""))
				and str(getattr(a, "safety", "")) == str(getattr(b, "safety", ""))
				and emo_a == emo_b
			)
			self.add_result(
				name="determinism_same_input",
				passed=same,
				details={
					"text": sample_text,
					"a": {"sentiment": getattr(a, "sentiment", None), "safety": getattr(a, "safety", None), "emotions": emo_a},
					"b": {"sentiment": getattr(b, "sentiment", None), "safety": getattr(b, "safety", None), "emotions": emo_b},
				},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(name="determinism_same_input", error=f"{type(exc).__name__}: {exc}")

		# Best-effort embedding determinism across repeated calls for the same text.
		try:
			sample_text = GOLDEN_SENTIMENT_DIRECTION_CASES[0][0] if GOLDEN_SENTIMENT_DIRECTION_CASES else next(iter(GOLDEN_OUTPUTS.keys()))
			e1 = list(self.client.get_embedding(sample_text))
			e2 = list(self.client.get_embedding(sample_text))
			sim = _cosine_similarity(e1, e2)
			self.add_result(
				name="determinism_embedding_same_input_cosine",
				passed=(sim >= 0.999),
				score=float(sim),
				threshold=0.999,
				details={"text": sample_text, "dim": len(e1)},
			)
		except Exception as exc:  # noqa: BLE001
			self.add_error(
				name="determinism_embedding_same_input_cosine",
				error=f"{type(exc).__name__}: {exc}",
			)

		return self.results

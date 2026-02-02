"""FamilyOS Golden Set Emotions Benchmark Suite.

Production-grade benchmark using curated family conversation data with:
- Family Hit Rate (FHR): Any overlap between predicted and expected
- Family Superlabel Hit Rate (FSHR): Positive/negative/neutral/surprise buckets
- Family-Weighted Hit Rate (FWHR): Weighted by safety-critical emotions
- Neutral Confusion Rate (NCR): Neutral misclassified as emotional
- Emotional Miss Rate (EMR): Emotional misclassified as neutral
- Family-Specific Coverage (FSC): Coverage of family-specific emotions
- Top-K Recall: Expected emotion in top-K predictions

Dataset: data/familyos/unified/golden_set/shard_0000.jsonl
1094 samples with multi-label emotion annotations, safety levels, and family context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.suite import register_suite

logger = logging.getLogger(__name__)


# =============================================================================
# FamilyOS Emotion Groupings
# =============================================================================

POSITIVE_EMOTIONS = {
    "joy", "love", "admiration", "amusement", "approval", "caring",
    "excitement", "gratitude", "optimism", "pride", "relief",
    "contentment", "hope", "tenderness", "warmth", "playfulness",
    "celebration", "belonging", "togetherness", "parental_pride",
}

NEGATIVE_EMOTIONS = {
    "anger", "sadness", "fear", "disgust", "annoyance", "disappointment",
    "disapproval", "embarrassment", "grief", "nervousness", "remorse",
    "frustration", "overwhelmed", "emptiness", "longing", "homesickness",
    "worry", "parental_guilt", "bittersweet",
}

NEUTRAL_EMOTIONS = {"neutral"}

SURPRISE_EMOTIONS = {"surprise"}

# Safety-critical emotions (higher weight for FWHR)
SAFETY_CRITICAL_EMOTIONS = {
    "fear", "grief", "remorse", "worry", "overwhelmed", "emptiness",
    "parental_guilt", "sadness", "anger", "frustration", "disappointment",
}

# Family-specific emotions (unique to FamilyOS)
FAMILY_SPECIFIC_EMOTIONS = {
    "parental_pride", "parental_guilt", "protectiveness", "togetherness",
    "homesickness", "warmth", "playfulness", "belonging", "nostalgia",
    "bittersweet", "longing", "patience",
}


def _get_superlabel(emotions: Set[str]) -> Set[str]:
    """Map emotions to coarse superlabels (positive/negative/neutral/surprise)."""
    result = set()
    for e in emotions:
        if e in POSITIVE_EMOTIONS:
            result.add("positive")
        elif e in NEGATIVE_EMOTIONS:
            result.add("negative")
        elif e in NEUTRAL_EMOTIONS:
            result.add("neutral")
        elif e in SURPRISE_EMOTIONS:
            result.add("surprise")
    return result


def _is_neutral_sample(emotions: List[str]) -> bool:
    """Check if sample is emotionally neutral."""
    # Neutral if only contains neutral or has no emotions
    if not emotions:
        return True
    return set(emotions) == {"neutral"}


def _is_emotional_sample(emotions: List[str]) -> bool:
    """Check if sample has clear emotional content."""
    # Emotional if has any non-neutral emotion
    if not emotions:
        return False
    return bool(set(emotions) - {"neutral"})


def _load_golden_set() -> List[Dict[str, Any]]:
    """Load FamilyOS golden set from JSONL.

    Returns:
        List of dicts with 'text' and 'tasks' (including 'emotions').
    """
    golden_path = Path("data/familyos/unified/golden_set/shard_0000.jsonl")

    if not golden_path.exists():
        logger.warning(f"Golden set not found at {golden_path}")
        return []

    samples = []
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(data)

        logger.info(f"Loaded {len(samples)} samples from golden set")
        return samples

    except Exception as e:
        logger.error(f"Failed to load golden set: {e}")
        return []


def _normalize_emotions(raw: Any) -> Set[str]:
    """Normalize predicted emotions to lowercase set."""
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw.strip().lower()} if raw.strip() else set()
    if isinstance(raw, (list, tuple)):
        return {str(e).strip().lower() for e in raw if e}
    return {str(raw).strip().lower()}


@register_suite
class EmotionsGoldenSuite(BenchmarkSuite):
    """FamilyOS Golden Set emotions benchmark with production metrics."""

    name: str = "emotions_golden"
    description: str = "Production emotions benchmark using FamilyOS golden set"

    # Thresholds calibrated for family conversation domain
    _FHR_THRESHOLD: float = 0.60  # Family Hit Rate
    _FSHR_THRESHOLD: float = 0.75  # Family Superlabel Hit Rate
    _FWHR_THRESHOLD: float = 0.55  # Family-Weighted Hit Rate (harder)
    _NCR_THRESHOLD: float = 0.20  # Neutral Confusion Rate (lower is better)
    _EMR_THRESHOLD: float = 0.15  # Emotional Miss Rate (lower is better)
    _FSC_THRESHOLD: float = 0.70  # Family-Specific Coverage
    _TOP3_THRESHOLD: float = 0.80  # Top-3 Recall

    # Sample limit (use all by default)
    _MAX_SAMPLES: int = 1106  # Full golden set + edge cases

    def run(self) -> List[Any]:
        """Run FamilyOS golden set emotions benchmark."""

        # Load golden set
        golden_data = _load_golden_set()

        if not golden_data:
            self.add_result(
                name="golden_set_available",
                passed=False,
                score=0.0,
                details={"error": "Golden set not found"},
            )
            return self.results

        self.add_result(
            name="golden_set_available",
            passed=True,
            score=1.0,
            details={"total_samples": len(golden_data)},
        )

        # Limit samples if needed
        samples = golden_data[:self._MAX_SAMPLES]

        # Initialize counters
        total = len(samples)
        fhr_hits = 0
        fshr_hits = 0
        fwhr_score = 0.0
        neutral_samples = 0
        neutral_confused = 0
        emotional_samples = 0
        emotional_missed = 0
        top3_hits = 0
        family_specific_predicted = set()

        hit_details = []
        miss_details = []

        # ---------------------------------------------------------------------
        # Process each sample
        # ---------------------------------------------------------------------
        for item in samples:
            text = item.get("text", "")
            tasks = item.get("tasks", {})
            expected_emotions = tasks.get("emotions", [])
            safety = tasks.get("safety_familyos", "GREEN")

            # Normalize expected
            expected_set = {e.lower() for e in expected_emotions}

            try:
                # Get predictions
                pred_raw = self.client.get_emotions(text)
                pred_set = _normalize_emotions(pred_raw)

                # Track family-specific predictions
                family_specific_predicted.update(pred_set.intersection(FAMILY_SPECIFIC_EMOTIONS))

                # -------------------------------------------------------------
                # 1. Family Hit Rate (FHR): Any overlap
                # -------------------------------------------------------------
                overlap = pred_set.intersection(expected_set)
                is_hit = len(overlap) > 0

                if is_hit:
                    fhr_hits += 1
                    if len(hit_details) < 10:
                        hit_details.append({
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "expected": sorted(expected_set),
                            "predicted": sorted(pred_set),
                            "overlap": sorted(overlap),
                            "safety": safety,
                        })
                else:
                    if len(miss_details) < 10:
                        miss_details.append({
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "expected": sorted(expected_set),
                            "predicted": sorted(pred_set),
                            "safety": safety,
                        })

                # -------------------------------------------------------------
                # 2. Family Superlabel Hit Rate (FSHR): Coarse buckets
                # -------------------------------------------------------------
                expected_super = _get_superlabel(expected_set)
                pred_super = _get_superlabel(pred_set)
                super_overlap = expected_super.intersection(pred_super)
                if len(super_overlap) > 0:
                    fshr_hits += 1

                # -------------------------------------------------------------
                # 3. Family-Weighted Hit Rate (FWHR): Safety-critical weight
                # -------------------------------------------------------------
                # Weight: 2.0 for safety-critical, 1.0 for others
                expected_critical = expected_set.intersection(SAFETY_CRITICAL_EMOTIONS)
                pred_critical = pred_set.intersection(SAFETY_CRITICAL_EMOTIONS)

                # Critical overlap worth 2 points, regular overlap worth 1 point
                critical_overlap = expected_critical.intersection(pred_critical)
                regular_overlap = overlap - critical_overlap

                sample_weight = len(critical_overlap) * 2.0 + len(regular_overlap) * 1.0
                max_weight = len(expected_critical) * 2.0 + len(expected_set - expected_critical) * 1.0

                if max_weight > 0:
                    fwhr_score += sample_weight / max_weight

                # -------------------------------------------------------------
                # 4. Neutral Confusion Rate (NCR): Neutral → Emotional
                # -------------------------------------------------------------
                if _is_neutral_sample(list(expected_set)):
                    neutral_samples += 1
                    # Check if we predicted emotional content
                    if _is_emotional_sample(list(pred_set)):
                        neutral_confused += 1

                # -------------------------------------------------------------
                # 5. Emotional Miss Rate (EMR): Emotional → Neutral
                # -------------------------------------------------------------
                if _is_emotional_sample(list(expected_set)):
                    emotional_samples += 1
                    # Check if we predicted neutral
                    if _is_neutral_sample(list(pred_set)):
                        emotional_missed += 1

                # -------------------------------------------------------------
                # 6. Top-K Recall: Check if any expected in top-3 predicted
                # -------------------------------------------------------------
                # Since we get a list, assume it's sorted by confidence
                if isinstance(pred_raw, list) and len(pred_raw) >= 1:
                    top3 = {str(e).lower() for e in pred_raw[:3]}
                    if len(expected_set.intersection(top3)) > 0:
                        top3_hits += 1
                elif is_hit:  # If we got a hit with single prediction
                    top3_hits += 1

            except Exception as exc:
                logger.warning(f"Error processing sample: {exc}")
                continue

        # ---------------------------------------------------------------------
        # Calculate Metrics
        # ---------------------------------------------------------------------
        fhr = fhr_hits / total if total > 0 else 0.0
        fshr = fshr_hits / total if total > 0 else 0.0
        fwhr = fwhr_score / total if total > 0 else 0.0
        ncr = neutral_confused / neutral_samples if neutral_samples > 0 else 0.0
        emr = emotional_missed / emotional_samples if emotional_samples > 0 else 0.0
        fsc = len(family_specific_predicted) / len(FAMILY_SPECIFIC_EMOTIONS)
        top3_recall = top3_hits / total if total > 0 else 0.0

        # ---------------------------------------------------------------------
        # Add Results
        # ---------------------------------------------------------------------
        self.add_result(
            name="family_hit_rate",
            passed=fhr >= self._FHR_THRESHOLD,
            score=fhr,
            threshold=self._FHR_THRESHOLD,
            details={
                "total": total,
                "hits": fhr_hits,
                "hit_examples": hit_details[:5],
                "miss_examples": miss_details[:5],
            },
        )

        self.add_result(
            name="family_superlabel_hit_rate",
            passed=fshr >= self._FSHR_THRESHOLD,
            score=fshr,
            threshold=self._FSHR_THRESHOLD,
            details={
                "total": total,
                "hits": fshr_hits,
                "description": "Coarse grouping: positive/negative/neutral/surprise",
            },
        )

        self.add_result(
            name="family_weighted_hit_rate",
            passed=fwhr >= self._FWHR_THRESHOLD,
            score=fwhr,
            threshold=self._FWHR_THRESHOLD,
            details={
                "total": total,
                "weighted_score": fwhr_score,
                "description": "Safety-critical emotions weighted 2x",
                "safety_critical": sorted(SAFETY_CRITICAL_EMOTIONS),
            },
        )

        self.add_result(
            name="neutral_confusion_rate",
            passed=ncr <= self._NCR_THRESHOLD,
            score=ncr,
            threshold=self._NCR_THRESHOLD,
            details={
                "neutral_samples": neutral_samples,
                "confused": neutral_confused,
                "description": "Neutral samples misclassified as emotional (lower is better)",
            },
        )

        self.add_result(
            name="emotional_miss_rate",
            passed=emr <= self._EMR_THRESHOLD,
            score=emr,
            threshold=self._EMR_THRESHOLD,
            details={
                "emotional_samples": emotional_samples,
                "missed": emotional_missed,
                "description": "Emotional samples predicted as neutral (lower is better)",
            },
        )

        self.add_result(
            name="family_specific_coverage",
            passed=fsc >= self._FSC_THRESHOLD,
            score=fsc,
            threshold=self._FSC_THRESHOLD,
            details={
                "total_family_emotions": len(FAMILY_SPECIFIC_EMOTIONS),
                "predicted_count": len(family_specific_predicted),
                "predicted": sorted(family_specific_predicted),
                "family_emotions": sorted(FAMILY_SPECIFIC_EMOTIONS),
            },
        )

        self.add_result(
            name="top3_recall",
            passed=top3_recall >= self._TOP3_THRESHOLD,
            score=top3_recall,
            threshold=self._TOP3_THRESHOLD,
            details={
                "total": total,
                "top3_hits": top3_hits,
                "description": "Expected emotion in top-3 predictions",
            },
        )

        return self.results

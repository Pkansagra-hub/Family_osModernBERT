"""Emotions benchmark suite using GoEmotions dataset.

Implements comprehensive emotion head benchmarking with:
- Hit rate metric (any predicted emotion matches any expected = hit)
- GoEmotions-to-UltraBERT label mapping (family-focused)
- Single-sentence samples only (no paragraphs)
- Skips non-transferable labels (neutral, approval, etc.)

GoEmotions: 27 emotions + neutral (Reddit data)
UltraBERT: 44 emotions (8 core + 12 positive + 10 negative + 14 family-specific)

Constraint: Uses HuggingFace datasets, no other external dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.suite import register_suite

logger = logging.getLogger(__name__)


# =============================================================================
# GoEmotions to UltraBERT Label Mapping (Family-Focused)
# =============================================================================

# GoEmotions labels (28 total: 27 emotions + neutral)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Labels to SKIP - don't transfer well from Reddit to family context
SKIP_GOEMOTIONS = {"neutral", "approval", "curiosity", "realization", "confusion"}

# UltraBERT 44 emotion labels
ULTRABERT_EMOTIONS = [
    # Core (8)
    "neutral", "joy", "sadness", "anger", "fear", "surprise", "love", "disgust",
    # Positive (12)
    "admiration", "amusement", "approval", "caring", "excitement", "gratitude",
    "optimism", "pride", "relief", "contentment", "hope", "tenderness",
    # Negative (10)
    "annoyance", "disappointment", "disapproval", "embarrassment", "grief",
    "nervousness", "remorse", "frustration", "overwhelmed", "emptiness",
    # Family-specific (14)
    "nostalgia", "protectiveness", "togetherness", "longing", "warmth",
    "playfulness", "celebration", "belonging", "parental_pride", "parental_guilt",
    "patience", "worry", "bittersweet", "homesickness",
]

# Family-focused mapping: GoEmotions -> UltraBERT (semantic expansion)
# Only emotions that transfer well across Reddit -> Family domains
GOEMOTIONS_TO_ULTRABERT: Dict[str, List[str]] = {
    # Strong emotions that transfer well
    "admiration": ["admiration", "pride"],
    "amusement": ["amusement", "playfulness", "joy"],
    "anger": ["anger", "frustration", "annoyance"],
    "annoyance": ["annoyance", "frustration"],
    "caring": ["caring", "protectiveness", "tenderness", "warmth"],
    "desire": ["longing"],
    "disappointment": ["disappointment", "sadness"],
    "disapproval": ["disapproval", "frustration"],
    "disgust": ["disgust"],
    "embarrassment": ["embarrassment"],
    "excitement": ["excitement", "celebration", "joy"],
    "fear": ["fear", "worry", "nervousness"],
    "gratitude": ["gratitude", "warmth", "love"],
    "grief": ["grief", "sadness", "emptiness"],
    "joy": ["joy", "contentment", "celebration", "warmth"],
    "love": ["love", "warmth", "tenderness", "belonging", "caring"],
    "nervousness": ["nervousness", "worry", "fear"],
    "optimism": ["optimism", "hope"],
    "pride": ["pride", "parental_pride", "admiration"],
    "relief": ["relief", "contentment"],
    "remorse": ["remorse", "parental_guilt", "sadness"],
    "sadness": ["sadness", "grief", "longing", "emptiness"],
    "surprise": ["surprise"],
}

# Family-specific emotions that have no GoEmotions equivalent (unique to UltraBERT)
FAMILY_ONLY_EMOTIONS = {
    "togetherness", "playfulness", "belonging", "parental_pride", "parental_guilt",
    "patience", "bittersweet", "nostalgia", "protectiveness", "homesickness"
}


def _is_single_sentence(text: str) -> bool:
    """Check if text is a single sentence (not a paragraph)."""
    # Count sentence-ending punctuation
    endings = text.count(".") + text.count("!") + text.count("?")
    # Single sentence: at most 1-2 sentence endings, reasonable length
    return endings <= 2 and len(text) < 200


def _should_skip_sample(labels: List[str]) -> bool:
    """Skip samples that ONLY have non-transferable labels."""
    transferable = [l for l in labels if l not in SKIP_GOEMOTIONS]
    return len(transferable) == 0


def _load_goemotions_validation() -> List[Dict[str, Any]]:
    """Load GoEmotions validation set from HuggingFace.

    Filters to:
    - Single-sentence samples only
    - Samples with at least one transferable emotion label

    Returns:
        List of dicts with 'text' and 'labels' (emotion names).
    """
    try:
        from datasets import load_dataset

        # Load simplified version with predefined splits
        dataset = load_dataset(
            "google-research-datasets/go_emotions",
            "simplified",
            split="validation",
            trust_remote_code=True,
        )

        samples = []
        skipped_paragraph = 0
        skipped_neutral = 0

        for item in dataset:
            text = item["text"]
            label_indices = item["labels"]
            emotion_names = [GOEMOTIONS_LABELS[i] for i in label_indices]

            # Skip paragraphs - only single sentences
            if not _is_single_sentence(text):
                skipped_paragraph += 1
                continue

            # Skip samples with only non-transferable labels
            if _should_skip_sample(emotion_names):
                skipped_neutral += 1
                continue

            samples.append({"text": text, "labels": emotion_names})

        logger.info(
            f"GoEmotions loaded: {len(samples)} samples "
            f"(skipped {skipped_paragraph} paragraphs, {skipped_neutral} neutral-only)"
        )

        return samples

    except ImportError:
        logger.warning("datasets library not installed, using fallback test cases")
        return []
    except Exception as e:
        logger.warning(f"Failed to load GoEmotions: {e}, using fallback test cases")
        return []


def _map_goemotions_to_ultrabert(labels: List[str]) -> Set[str]:
    """Map GoEmotions labels to UltraBERT emotion space.

    Args:
        labels: GoEmotions label names.

    Returns:
        Set of UltraBERT emotion names (expanded via semantic mapping).
        Skips non-transferable labels.
    """
    result = set()
    for label in labels:
        if label in SKIP_GOEMOTIONS:
            continue
        if label in GOEMOTIONS_TO_ULTRABERT:
            result.update(GOEMOTIONS_TO_ULTRABERT[label])
    return result


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
class EmotionsSuite(BenchmarkSuite):
    """Emotions benchmark suite with hit rate metric on GoEmotions."""

    name: str = "emotions_goemotions"
    description: str = "Emotion classification benchmarks using GoEmotions dataset"

    # Hit rate thresholds (cross-dataset: Reddit -> Family)
    # With proper filtering (single sentences, skip neutral), expect ~55-60%
    _HIT_RATE_THRESHOLD: float = 0.50
    _SUPERLABEL_HIT_RATE_THRESHOLD: float = 0.65  # Coarse groupings

    # Use all filtered samples (after paragraph/neutral filtering)
    _MAX_SAMPLES: int = 2000  # Large sample for statistical significance

    def run(self) -> List[Any]:
        """Run emotions benchmark suite."""

        # Load GoEmotions validation set
        goemotions_data = _load_goemotions_validation()

        if not goemotions_data:
            # Fallback to built-in test cases if GoEmotions unavailable
            self.add_result(
                name="goemotions_data_available",
                passed=False,
                score=0.0,
                details={"error": "GoEmotions dataset not available"},
            )
            return self.results

        self.add_result(
            name="goemotions_data_available",
            passed=True,
            score=1.0,
            details={"total_samples": len(goemotions_data)},
        )

        # Limit samples for speed
        samples = goemotions_data[:self._MAX_SAMPLES]

        # ---------------------------------------------------------------------
        # Hit Rate: Any overlap between predicted and expected
        # ---------------------------------------------------------------------
        total = len(samples)
        hits = 0
        hit_details: List[Dict[str, Any]] = []

        for item in samples:
            text = item["text"]
            expected_ge = item["labels"]
            # Map GoEmotions to UltraBERT space
            expected_ub = _map_goemotions_to_ultrabert(expected_ge)

            try:
                pred_raw = self.client.get_emotions(text)
                pred = _normalize_emotions(pred_raw)

                # Hit = any overlap
                overlap = pred.intersection(expected_ub)
                is_hit = len(overlap) > 0

                if is_hit:
                    hits += 1

                hit_details.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "expected_goemotions": expected_ge,
                    "expected_ultrabert": sorted(expected_ub),
                    "predicted": sorted(pred),
                    "overlap": sorted(overlap),
                    "hit": is_hit,
                })
            except Exception as exc:
                hit_details.append({
                    "text": text[:100],
                    "error": f"{type(exc).__name__}: {exc}",
                })

        hit_rate = hits / total if total > 0 else 0.0

        self.add_result(
            name="goemotions_hit_rate",
            passed=hit_rate >= self._HIT_RATE_THRESHOLD,
            score=hit_rate,
            threshold=self._HIT_RATE_THRESHOLD,
            details={
                "total": total,
                "hits": hits,
                "threshold": self._HIT_RATE_THRESHOLD,
                "samples": hit_details[:20],  # First 20 for inspection
            },
        )

        # ---------------------------------------------------------------------
        # Superlabel Hit Rate (coarse grouping: positive/negative/neutral)
        # ---------------------------------------------------------------------
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

        def get_superlabel(emotions: Set[str]) -> Set[str]:
            """Get coarse sentiment grouping."""
            result = set()
            for e in emotions:
                if e in POSITIVE_EMOTIONS:
                    result.add("positive")
                elif e in NEGATIVE_EMOTIONS:
                    result.add("negative")
                elif e == "neutral":
                    result.add("neutral")
                elif e == "surprise":
                    result.add("surprise")  # Can be either
            return result

        superlabel_hits = 0
        superlabel_details: List[Dict[str, Any]] = []

        for item in samples:
            text = item["text"]
            expected_ge = item["labels"]
            expected_ub = _map_goemotions_to_ultrabert(expected_ge)
            expected_super = get_superlabel(expected_ub)

            try:
                pred_raw = self.client.get_emotions(text)
                pred = _normalize_emotions(pred_raw)
                pred_super = get_superlabel(pred)

                # Hit = any superlabel overlap
                overlap = pred_super.intersection(expected_super)
                is_hit = len(overlap) > 0

                if is_hit:
                    superlabel_hits += 1

                superlabel_details.append({
                    "expected_super": sorted(expected_super),
                    "predicted_super": sorted(pred_super),
                    "hit": is_hit,
                })
            except Exception:
                pass

        superlabel_hit_rate = superlabel_hits / total if total > 0 else 0.0

        self.add_result(
            name="goemotions_superlabel_hit_rate",
            passed=superlabel_hit_rate >= self._SUPERLABEL_HIT_RATE_THRESHOLD,
            score=superlabel_hit_rate,
            threshold=self._SUPERLABEL_HIT_RATE_THRESHOLD,
            details={
                "total": total,
                "hits": superlabel_hits,
                "threshold": self._SUPERLABEL_HIT_RATE_THRESHOLD,
            },
        )

        # ---------------------------------------------------------------------
        # Coverage: What percentage of UltraBERT emotions were predicted?
        # ---------------------------------------------------------------------
        all_predictions: Set[str] = set()
        for item in hit_details:
            if "predicted" in item:
                all_predictions.update(item["predicted"])

        ultrabert_coverage = len(all_predictions.intersection(set(ULTRABERT_EMOTIONS)))
        coverage_pct = ultrabert_coverage / len(ULTRABERT_EMOTIONS)

        # Also check family-specific coverage
        family_predicted = all_predictions.intersection(FAMILY_ONLY_EMOTIONS)

        self.add_result(
            name="emotion_label_coverage",
            passed=coverage_pct >= 0.30,  # At least 30% of labels used
            score=coverage_pct,
            details={
                "total_labels": len(ULTRABERT_EMOTIONS),
                "labels_predicted": ultrabert_coverage,
                "family_specific_predicted": sorted(family_predicted),
                "unique_predictions": sorted(all_predictions),
            },
        )

        # ---------------------------------------------------------------------
        # Precision/Recall on exact matches (informational, not gated)
        # ---------------------------------------------------------------------
        exact_matches = 0
        partial_matches = 0

        for item in hit_details:
            if "predicted" in item and "expected_ultrabert" in item:
                pred = set(item["predicted"])
                exp = set(item["expected_ultrabert"])
                if pred == exp:
                    exact_matches += 1
                elif len(pred.intersection(exp)) > 0:
                    partial_matches += 1

        self.add_result(
            name="emotion_match_breakdown",
            passed=True,  # Informational only
            score=exact_matches / total if total > 0 else 0.0,
            details={
                "exact_matches": exact_matches,
                "partial_matches": partial_matches,
                "no_match": total - exact_matches - partial_matches,
                "exact_match_rate": exact_matches / total if total > 0 else 0.0,
            },
        )

        return self.results

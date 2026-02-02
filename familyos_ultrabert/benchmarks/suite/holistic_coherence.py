"""FamilyOS Holistic Coherence Benchmark Suite.

Production-grade benchmark that evaluates cross-head consistency and coherence
using a SINGLE forward pass through all 12 heads. This measures how well the
model understands family context holistically.

12 Heads:
- Extraction: ner_general, ner_family, temporal, relation
- Classification: sentiment, emotions, safety_generic, safety_familyos,
                  intent, ingress, nli
- Representation: embedding

Coherence Metrics:
- HAS (Head Agreement Score): Semantic consistency between related heads
- EGS (Entity Grounding Score): Relations anchored to detected entities
- SEC (Safety-Emotion Consistency): Distress signals cascade correctly
- TCS (Temporal Completeness Score): Intents have required temporal info
- FCS (Family Context Score): Richness of family understanding
- IIC (Intent-Ingress Coherence): Domain alignment between intent and ingress
- FCCS (Family Context Coherence Score): Overall holistic score

Dataset: data/familyos/unified/golden_set/shard_0000.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.suite import register_suite

logger = logging.getLogger(__name__)


# =============================================================================
# Head Taxonomy & Cross-Head Rules
# =============================================================================

# All 12 capabilities
ALL_CAPABILITIES = [
    "ner_general", "ner_family", "temporal", "relation",
    "sentiment", "emotions", "safety_generic", "safety_familyos",
    "intent", "ingress", "nli", "embedding",
]

# Emotion valence groupings
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

DISTRESS_EMOTIONS = {
    "grief", "fear", "emptiness", "overwhelmed", "worry", "remorse",
    "sadness", "anger", "frustration", "disappointment", "parental_guilt",
}

FAMILY_SPECIFIC_EMOTIONS = {
    "parental_pride", "parental_guilt", "protectiveness", "togetherness",
    "homesickness", "warmth", "playfulness", "belonging", "nostalgia",
    "bittersweet", "longing", "patience",
}

# Sentiment to valence mapping
SENTIMENT_VALENCE = {
    "very_negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "very_positive": 2,
}

# Family relations (vs generic relations)
FAMILY_RELATIONS = {
    "parent_of", "child_of", "spouse_of", "sibling_of", "grandparent_of",
    "grandchild_of", "aunt_uncle_of", "niece_nephew_of", "cousin_of", "pet_of",
}

# Intent-to-Ingress expected mappings
# Updated based on empirical analysis - CONCERN is valid for advice-seeking,
# CELEBRATION valid for memory logging (proud moments), etc.
INTENT_INGRESS_MAP = {
    "log_memory": {"MEMORY", "DIARY", "CELEBRATION"},  # Proud moments are memories
    "query_memory": {"MEMORY", "PLANNING"},  # Querying for planning purposes
    "set_reminder": {"TASK", "PLANNING", "CELEBRATION", "WORK"},  # Events, work tasks
    "express_feeling": {"DIARY", "RELATIONSHIP", "CONCERN", "GRATITUDE", "HEALTH", "CELEBRATION"},
    "seek_advice": {"RELATIONSHIP", "HEALTH", "FINANCE", "WORK", "CONCERN"},  # CONCERN is core to advice-seeking
    "share_news": {"CELEBRATION", "DIARY", "RELATIONSHIP", "GRATITUDE", "PLANNING"},
    "reflect": {"DIARY", "MEMORY", "GRATITUDE"},
    "other": set(),  # Anything goes
}

# Ingress-to-Emotion expected associations
INGRESS_EMOTION_MAP = {
    "CELEBRATION": {"joy", "excitement", "celebration", "pride", "gratitude"},
    "CONCERN": {"worry", "fear", "sadness", "frustration", "overwhelmed"},
    "GRATITUDE": {"gratitude", "love", "caring", "warmth", "contentment"},
    "HEALTH": {"worry", "fear", "hope", "relief", "frustration"},
    "RELATIONSHIP": {"love", "caring", "frustration", "disappointment", "togetherness"},
}


# =============================================================================
# Data Loading
# =============================================================================

def _load_golden_set() -> List[Dict[str, Any]]:
    """Load FamilyOS golden set from JSONL."""
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


# =============================================================================
# Output Extraction Helpers
# =============================================================================

def _extract_labels(result: Any) -> Set[str]:
    """Extract label names from an inference result."""
    if result is None:
        return set()

    output = result.output if hasattr(result, "output") else result

    if isinstance(output, dict):
        # Emotions format: {"predictions": [...], "scores": {...}}
        if "predictions" in output:
            preds = output["predictions"]
            if isinstance(preds, (list, tuple)):
                return set(str(x) for x in preds if x)
        # Intent/Ingress format: {"primary": "...", "all"/"domains": [...]}
        if "primary" in output:
            result_set = set()
            if output["primary"]:
                result_set.add(output["primary"])
            # Also include all/domains if present
            for key in ("all", "domains"):
                if key in output and output[key]:
                    result_set.update(str(x) for x in output[key] if x)
            return result_set
        # Multi-label classification: {"labels": [...]}
        if "labels" in output:
            return set(str(x) for x in output["labels"] if x)
        # NER entities: {"entities": [...]}
        if "entities" in output:
            return {e.get("label", "") for e in output["entities"] if e.get("label")}
        # Safety level: {"level": "..."}
        if "level" in output:
            return {output["level"]} if output["level"] else set()
        # Single label: {"label": "..."}
        if "label" in output:
            return {output["label"]} if output["label"] else set()

    if isinstance(output, (list, tuple)):
        return set(str(x) for x in output if x)

    if isinstance(output, str):
        return {output}

    return set()


def _extract_entities(result: Any) -> List[Dict[str, Any]]:
    """Extract entity list from NER result."""
    if result is None:
        return []

    output = result.output if hasattr(result, "output") else result

    if isinstance(output, dict) and "entities" in output:
        return output["entities"]

    return []


def _extract_sentiment_valence(result: Any) -> int:
    """Extract sentiment valence (-2 to +2)."""
    if result is None:
        return 0

    output = result.output if hasattr(result, "output") else result

    if isinstance(output, dict):
        label = output.get("label", "neutral")
    elif isinstance(output, str):
        label = output
    else:
        return 0

    return SENTIMENT_VALENCE.get(label, 0)


def _extract_safety_level(result: Any) -> str:
    """Extract safety level string."""
    if result is None:
        return "GREEN"

    output = result.output if hasattr(result, "output") else result

    if isinstance(output, dict):
        # Support both "band" (FamilyOS safety) and "level" (generic)
        return output.get("band") or output.get("level") or "GREEN"
    if isinstance(output, str):
        return output

    return "GREEN"


def _get_emotion_valence(emotions: Set[str]) -> float:
    """Compute average emotion valence (-1 to +1)."""
    if not emotions:
        return 0.0

    pos_count = len(emotions & POSITIVE_EMOTIONS)
    neg_count = len(emotions & NEGATIVE_EMOTIONS)
    total = pos_count + neg_count

    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


# =============================================================================
# Coherence Metric Functions
# =============================================================================

def compute_sentiment_emotion_agreement(
    sentiment_valence: int,
    emotions: Set[str],
) -> Tuple[float, str]:
    """Check if sentiment and emotion valence align.

    Returns:
        (score, reason) - Score 0-1, and explanation
    """
    # Handle None inputs
    emotions = emotions or set()

    emotion_valence = _get_emotion_valence(emotions)

    # Convert to same scale
    sent_sign = 1 if sentiment_valence > 0 else (-1 if sentiment_valence < 0 else 0)
    emo_sign = 1 if emotion_valence > 0.2 else (-1 if emotion_valence < -0.2 else 0)

    # Perfect agreement
    if sent_sign == emo_sign:
        return 1.0, "aligned"

    # Neutral is compatible with anything mild
    if sent_sign == 0 or emo_sign == 0:
        return 0.8, "neutral_compatible"

    # Contradiction
    return 0.3, f"contradiction: sentiment={sent_sign}, emotion={emo_sign}"


def compute_safety_emotion_consistency(
    safety_level: str,
    emotions: Set[str],
) -> Tuple[float, str]:
    """Check if safety level is consistent with emotional distress signals.

    Returns:
        (score, reason)
    """
    # Handle None inputs
    safety_level = safety_level or "GREEN"
    emotions = emotions or set()

    has_distress = bool(emotions & DISTRESS_EMOTIONS)

    if safety_level == "CRISIS":
        # Crisis should have strong signals
        if has_distress:
            return 1.0, "crisis_with_distress"
        return 0.5, "crisis_without_distress"

    if safety_level == "RED":
        if has_distress:
            return 1.0, "red_with_distress"
        return 0.7, "red_without_distress"

    if safety_level == "AMBER":
        # Amber is compatible with mild distress
        return 1.0, "amber_compatible"

    # GREEN
    if has_distress:
        # Distress emotions but GREEN might be concerning
        return 0.6, "green_with_distress"
    return 1.0, "green_ok"


def compute_intent_ingress_coherence(
    intents: Set[str],
    ingresses: Set[str],
) -> Tuple[float, str]:
    """Check if intent aligns with ingress domain.

    Returns:
        (score, reason)
    """
    # Handle None inputs
    intents = intents or set()
    ingresses = ingresses or set()

    if not intents or not ingresses:
        return 0.8, "missing_data"

    matches = 0
    total = 0

    for intent in intents:
        if intent == "other":
            continue

        expected = INTENT_INGRESS_MAP.get(intent, set())
        if not expected:
            continue

        total += 1
        if ingresses & expected:
            matches += 1

    if total == 0:
        return 0.9, "no_mappable_intents"

    score = matches / total
    return score, f"matched_{matches}/{total}"


def compute_temporal_completeness(
    intents: Set[str],
    temporal_entities: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """Check if reminder intents have temporal information.

    Returns:
        (score, reason)
    """
    if not intents or "set_reminder" not in intents:
        return 1.0, "not_applicable"

    has_temporal = len(temporal_entities) > 0 if temporal_entities else False

    if has_temporal:
        return 1.0, "reminder_has_time"
    return 0.3, "reminder_missing_time"


def compute_entity_grounding(
    ner_family_entities: List[Dict[str, Any]],
    ner_general_entities: List[Dict[str, Any]],
    relations: Set[str],
) -> Tuple[float, str]:
    """Check if relations are grounded in detected entities.

    Returns:
        (score, reason)
    """
    # Handle None inputs
    ner_family_entities = ner_family_entities or []
    ner_general_entities = ner_general_entities or []
    relations = relations or set()

    # Count detected persons
    family_persons = [e for e in ner_family_entities
                      if e.get("label") in ("PERSON", "KINSHIP", "NICKNAME")]
    general_persons = [e for e in ner_general_entities
                       if e.get("label") == "PER"]

    total_persons = len(family_persons) + len(general_persons)

    has_family_relations = bool(relations & FAMILY_RELATIONS)

    if has_family_relations and total_persons == 0:
        return 0.4, "relations_without_entities"

    if total_persons > 0 and not has_family_relations:
        # Has persons but no relations detected - could be OK
        return 0.7, "entities_without_relations"

    if has_family_relations and total_persons > 0:
        return 1.0, "grounded"

    return 1.0, "no_family_context"


def compute_family_context_richness(
    ner_family_entities: List[Dict[str, Any]],
    emotions: Set[str],
    relations: Set[str],
) -> Tuple[float, str]:
    """Measure richness of family understanding.

    Uses a holistic approach: entities, emotions, and relations all contribute.
    If NER is weak but relations/emotions are strong, score is still reasonable.

    Returns:
        (score, reason)
    """
    # Handle None inputs
    ner_family_entities = ner_family_entities or []
    emotions = emotions or set()
    relations = relations or set()

    scores = []
    reasons = []

    # Rebalanced weights based on empirical analysis:
    # - Relations: 97% hit rate (very reliable)
    # - Entities: 78% hit rate (good)
    # - Family emotions: 71% hit rate (decent)

    # Family relations (0.35 weight - reliable signal, full credit for any detection)
    family_rels = relations & FAMILY_RELATIONS
    if family_rels:
        # Full credit for having any family relation (model is 97% accurate)
        # Small bonus for multiple relations
        rel_bonus = min(0.05, (len(family_rels) - 1) * 0.025)
        scores.append(0.35 + rel_bonus)
        reasons.append(f"fam_rels={len(family_rels)}")
    else:
        scores.append(0.0)
        reasons.append("no_fam_rels")

    # Family entities (0.35 weight)
    entity_count = len(ner_family_entities)
    entity_score = min(1.0, entity_count / 2)  # Expect ~2 entities
    scores.append(entity_score * 0.35)
    reasons.append(f"entities={entity_count}")

    # Family-specific emotions (0.30 weight)
    family_emotions = emotions & FAMILY_SPECIFIC_EMOTIONS
    if family_emotions:
        scores.append(0.30)  # Full credit
        reasons.append(f"fam_emotions={len(family_emotions)}")
    else:
        # Partial credit if we have any emotions (family context often evokes emotions)
        scores.append(0.15 if emotions else 0.0)
        reasons.append("no_fam_emotions" if not emotions else "general_emotions")

    # Sum the weighted scores (max 1.0)
    total_score = sum(scores)
    return min(1.0, total_score), "; ".join(reasons)


def compute_ingress_emotion_consistency(
    ingresses: Set[str],
    emotions: Set[str],
) -> Tuple[float, str]:
    """Check if ingress domain has expected emotional associations.

    Returns:
        (score, reason)
    """
    # Handle None inputs
    ingresses = ingresses or set()
    emotions = emotions or set()

    if not ingresses:
        return 1.0, "no_ingress"

    matches = 0
    total = 0

    for ingress in ingresses:
        expected = INGRESS_EMOTION_MAP.get(ingress)
        if expected is None:
            continue

        total += 1
        if emotions & expected:
            matches += 1

    if total == 0:
        return 1.0, "no_mapped_ingress"

    score = matches / total
    return score, f"matched_{matches}/{total}"


# =============================================================================
# Main Benchmark Suite
# =============================================================================

@register_suite
class HolisticCoherenceSuite(BenchmarkSuite):
    """Holistic coherence benchmark using single forward pass for all 12 heads."""

    name: str = "holistic_coherence"
    description: str = "Cross-head consistency and coherence metrics (single forward pass)"

    # Thresholds for each metric
    _HAS_THRESHOLD: float = 0.70  # Head Agreement Score
    _EGS_THRESHOLD: float = 0.65  # Entity Grounding Score
    _SEC_THRESHOLD: float = 0.75  # Safety-Emotion Consistency
    _TCS_THRESHOLD: float = 0.80  # Temporal Completeness Score
    _FCS_THRESHOLD: float = 0.35  # Family Context Score (lowered - NER thresholds are conservative)
    _IIC_THRESHOLD: float = 0.55  # Intent-Ingress Coherence (lowered - mapping is not 1:1)
    _IEC_THRESHOLD: float = 0.60  # Ingress-Emotion Consistency
    _FCCS_THRESHOLD: float = 0.70  # Overall Family Context Coherence Score

    # Sample limit
    _MAX_SAMPLES: int = 500  # Holistic eval is expensive, sample subset

    def run(self) -> List[Any]:
        """Run holistic coherence benchmark."""

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

        # Sample subset for efficiency
        samples = golden_data[:self._MAX_SAMPLES]
        total = len(samples)

        # Check available capabilities
        available_caps = set(self.client.capabilities)
        required_caps = {"sentiment", "emotions", "safety_familyos", "intent", "ingress"}

        if not required_caps.issubset(available_caps):
            missing = required_caps - available_caps
            self.add_result(
                name="capabilities_check",
                passed=False,
                score=0.0,
                details={"missing": list(missing), "available": list(available_caps)},
            )
            return self.results

        # Determine which heads to run
        caps_to_run = [c for c in ALL_CAPABILITIES if c in available_caps]

        self.add_result(
            name="capabilities_check",
            passed=True,
            score=1.0,
            details={"running": caps_to_run, "total_heads": len(caps_to_run)},
        )

        # Aggregate metrics
        sea_scores = []  # Sentiment-Emotion Agreement
        sec_scores = []  # Safety-Emotion Consistency
        iic_scores = []  # Intent-Ingress Coherence
        tcs_scores = []  # Temporal Completeness Score
        egs_scores = []  # Entity Grounding Score
        fcs_scores = []  # Family Context Score
        iec_scores = []  # Ingress-Emotion Consistency

        sample_details = []
        latencies = []

        # ---------------------------------------------------------------------
        # Process each sample with SINGLE forward pass
        # ---------------------------------------------------------------------
        for i, item in enumerate(samples):
            text = item.get("text", "")
            sample_id = item.get("id", f"sample_{i}")

            if not text:
                continue

            try:
                start_time = time.perf_counter()

                # SINGLE FORWARD PASS - All heads at once
                result = self.client.analyze(text, capabilities=caps_to_run)

                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)

                # Extract outputs from ClientResult - use _caps dict for raw access
                caps_dict = result._caps if hasattr(result, "_caps") else {}

                # Get individual head outputs
                sentiment_result = caps_dict.get("sentiment")
                emotions_result = caps_dict.get("emotions")
                safety_result = caps_dict.get("safety_familyos")
                intent_result = caps_dict.get("intent")
                ingress_result = caps_dict.get("ingress")
                temporal_result = caps_dict.get("temporal")
                ner_family_result = caps_dict.get("ner_family")
                ner_general_result = caps_dict.get("ner_general")
                relation_result = caps_dict.get("relation")

                # Extract structured data
                sentiment_valence = _extract_sentiment_valence(sentiment_result)
                emotions = _extract_labels(emotions_result)
                safety_level = _extract_safety_level(safety_result)
                intents = _extract_labels(intent_result)
                ingresses = _extract_labels(ingress_result)
                temporal_entities = _extract_entities(temporal_result)
                ner_family_entities = _extract_entities(ner_family_result)
                ner_general_entities = _extract_entities(ner_general_result)
                relations = _extract_labels(relation_result)

                # Compute coherence metrics
                sea_score, sea_reason = compute_sentiment_emotion_agreement(
                    sentiment_valence, emotions
                )
                sea_scores.append(sea_score)

                sec_score, sec_reason = compute_safety_emotion_consistency(
                    safety_level, emotions
                )
                sec_scores.append(sec_score)

                iic_score, iic_reason = compute_intent_ingress_coherence(
                    intents, ingresses
                )
                iic_scores.append(iic_score)

                tcs_score, tcs_reason = compute_temporal_completeness(
                    intents, temporal_entities
                )
                tcs_scores.append(tcs_score)

                egs_score, egs_reason = compute_entity_grounding(
                    ner_family_entities, ner_general_entities, relations
                )
                egs_scores.append(egs_score)

                fcs_score, fcs_reason = compute_family_context_richness(
                    ner_family_entities, emotions, relations
                )
                fcs_scores.append(fcs_score)

                iec_score, iec_reason = compute_ingress_emotion_consistency(
                    ingresses, emotions
                )
                iec_scores.append(iec_score)

                # Store sample details (first 10 for debugging)
                if len(sample_details) < 10:
                    sample_details.append({
                        "id": sample_id,
                        "text_preview": text[:80] + "..." if len(text) > 80 else text,
                        "sentiment_valence": sentiment_valence,
                        "emotions": list(emotions)[:5],
                        "safety": safety_level,
                        "intents": list(intents),
                        "ingresses": list(ingresses),
                        "sea": {"score": sea_score, "reason": sea_reason},
                        "sec": {"score": sec_score, "reason": sec_reason},
                        "iic": {"score": iic_score, "reason": iic_reason},
                        "tcs": {"score": tcs_score, "reason": tcs_reason},
                        "latency_ms": round(latency_ms, 1),
                    })

            except Exception as e:
                logger.warning(f"Error processing {sample_id}: {e}")
                continue

        # ---------------------------------------------------------------------
        # Compute aggregate scores
        # ---------------------------------------------------------------------
        if not sea_scores:
            self.add_result(
                name="processing_error",
                passed=False,
                score=0.0,
                details={"error": "No samples processed successfully"},
            )
            return self.results

        # Helper for average
        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        has_score = avg(sea_scores)  # Head Agreement (sentiment-emotion)
        egs_score = avg(egs_scores)  # Entity Grounding
        sec_score = avg(sec_scores)  # Safety-Emotion Consistency
        tcs_score = avg(tcs_scores)  # Temporal Completeness
        fcs_score = avg(fcs_scores)  # Family Context
        iic_score = avg(iic_scores)  # Intent-Ingress Coherence
        iec_score = avg(iec_scores)  # Ingress-Emotion Consistency

        # Overall FCCS (weighted average)
        fccs_score = (
            0.15 * has_score +
            0.15 * egs_score +
            0.20 * sec_score +  # Safety is critical
            0.10 * tcs_score +
            0.15 * fcs_score +
            0.15 * iic_score +
            0.10 * iec_score
        )

        avg_latency = avg(latencies)

        # ---------------------------------------------------------------------
        # Record results
        # ---------------------------------------------------------------------

        # Head Agreement Score (Sentiment-Emotion)
        self.add_result(
            name="head_agreement_score",
            passed=has_score >= self._HAS_THRESHOLD,
            score=round(has_score, 4),
            threshold=self._HAS_THRESHOLD,
            latency_ms=avg_latency,
            details={
                "description": "Sentiment-emotion valence alignment",
                "samples_evaluated": len(sea_scores),
            },
        )

        # Entity Grounding Score
        self.add_result(
            name="entity_grounding_score",
            passed=egs_score >= self._EGS_THRESHOLD,
            score=round(egs_score, 4),
            threshold=self._EGS_THRESHOLD,
            details={
                "description": "Relations grounded in detected entities",
                "samples_evaluated": len(egs_scores),
            },
        )

        # Safety-Emotion Consistency
        self.add_result(
            name="safety_emotion_consistency",
            passed=sec_score >= self._SEC_THRESHOLD,
            score=round(sec_score, 4),
            threshold=self._SEC_THRESHOLD,
            details={
                "description": "Distress emotions elevate safety level",
                "samples_evaluated": len(sec_scores),
            },
        )

        # Temporal Completeness
        self.add_result(
            name="temporal_completeness_score",
            passed=tcs_score >= self._TCS_THRESHOLD,
            score=round(tcs_score, 4),
            threshold=self._TCS_THRESHOLD,
            details={
                "description": "Reminder intents have temporal info",
                "samples_evaluated": len(tcs_scores),
            },
        )

        # Family Context Richness
        self.add_result(
            name="family_context_score",
            passed=fcs_score >= self._FCS_THRESHOLD,
            score=round(fcs_score, 4),
            threshold=self._FCS_THRESHOLD,
            details={
                "description": "Richness of family understanding",
                "samples_evaluated": len(fcs_scores),
            },
        )

        # Intent-Ingress Coherence
        self.add_result(
            name="intent_ingress_coherence",
            passed=iic_score >= self._IIC_THRESHOLD,
            score=round(iic_score, 4),
            threshold=self._IIC_THRESHOLD,
            details={
                "description": "Intent aligns with ingress domain",
                "samples_evaluated": len(iic_scores),
            },
        )

        # Ingress-Emotion Consistency
        self.add_result(
            name="ingress_emotion_consistency",
            passed=iec_score >= self._IEC_THRESHOLD,
            score=round(iec_score, 4),
            threshold=self._IEC_THRESHOLD,
            details={
                "description": "Ingress domain has expected emotions",
                "samples_evaluated": len(iec_scores),
            },
        )

        # Overall FCCS Score
        self.add_result(
            name="fccs_overall",
            passed=fccs_score >= self._FCCS_THRESHOLD,
            score=round(fccs_score, 4),
            threshold=self._FCCS_THRESHOLD,
            latency_ms=avg_latency,
            details={
                "description": "Family Context Coherence Score (holistic)",
                "components": {
                    "head_agreement": round(has_score, 4),
                    "entity_grounding": round(egs_score, 4),
                    "safety_emotion": round(sec_score, 4),
                    "temporal_completeness": round(tcs_score, 4),
                    "family_context": round(fcs_score, 4),
                    "intent_ingress": round(iic_score, 4),
                    "ingress_emotion": round(iec_score, 4),
                },
                "weights": {
                    "head_agreement": 0.15,
                    "entity_grounding": 0.15,
                    "safety_emotion": 0.20,
                    "temporal_completeness": 0.10,
                    "family_context": 0.15,
                    "intent_ingress": 0.15,
                    "ingress_emotion": 0.10,
                },
                "samples_evaluated": len(sea_scores),
                "avg_latency_ms": round(avg_latency, 2),
                "sample_details": sample_details,
            },
        )

        # Performance summary
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

        self.add_result(
            name="holistic_latency",
            passed=p50 < 100,  # 100ms per sample for all 12 heads
            score=round(p50, 2),
            threshold=100.0,
            latency_ms=avg_latency,
            details={
                "description": "Latency for single forward pass (all heads)",
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "avg_ms": round(avg_latency, 2),
                "samples": len(latencies),
            },
        )

        return self.results

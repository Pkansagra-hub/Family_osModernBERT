"""
Complexity Classifier for UltraBERT v4.

Routes requests to appropriate processing tier based on:
- Intent classification
- Ingress domain
- Safety signals
- Emotional intensity
- Query complexity signals

Tiers:
- LOW: Direct tool execution (calendar, notes, reminders)
- MEDIUM: Small LLM for emotional support, simple queries
- HIGH: Full LLM system for complex advice, multi-turn reasoning
- CRISIS: Immediate safety protocol (bypasses complexity)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ComplexityTier(str, Enum):
    """Complexity routing tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRISIS = "CRISIS"


@dataclass
class ComplexityResult:
    """Result of complexity classification."""
    tier: ComplexityTier
    base_tier: ComplexityTier  # Before modifiers
    intent: str
    ingress: str
    reason: str
    modifiers_applied: List[str]
    safety_override: bool = False
    confidence: float = 0.0


# =============================================================================
# Intent + Ingress -> Base Complexity Mapping
# =============================================================================

# Primary mapping: (intent, ingress) -> complexity
INTENT_INGRESS_COMPLEXITY: Dict[Tuple[str, str], ComplexityTier] = {
    # LOW Complexity - Tool Execution
    ("log_memory", "DIARY"): ComplexityTier.LOW,
    ("log_memory", "MEMORY"): ComplexityTier.LOW,
    ("log_memory", "TASK"): ComplexityTier.LOW,
    ("log_memory", "PLANNING"): ComplexityTier.LOW,
    ("set_reminder", "TASK"): ComplexityTier.LOW,
    ("set_reminder", "PLANNING"): ComplexityTier.LOW,
    ("set_reminder", "DIARY"): ComplexityTier.LOW,
    ("query_memory", "MEMORY"): ComplexityTier.LOW,
    ("query_memory", "DIARY"): ComplexityTier.LOW,

    # MEDIUM Complexity - Small LLM
    ("express_feeling", "DIARY"): ComplexityTier.MEDIUM,
    ("express_feeling", "CELEBRATION"): ComplexityTier.MEDIUM,
    ("express_feeling", "GRATITUDE"): ComplexityTier.MEDIUM,
    ("express_feeling", "RELATIONSHIP"): ComplexityTier.MEDIUM,
    ("express_feeling", "MEMORY"): ComplexityTier.MEDIUM,
    ("share_news", "CELEBRATION"): ComplexityTier.MEDIUM,
    ("share_news", "DIARY"): ComplexityTier.MEDIUM,
    ("share_news", "GRATITUDE"): ComplexityTier.MEDIUM,
    ("share_news", "RELATIONSHIP"): ComplexityTier.MEDIUM,
    ("query_memory", "RELATIONSHIP"): ComplexityTier.MEDIUM,

    # HIGH Complexity - Full LLM
    ("seek_advice", "HEALTH"): ComplexityTier.HIGH,
    ("seek_advice", "FINANCE"): ComplexityTier.HIGH,
    ("seek_advice", "WORK"): ComplexityTier.HIGH,
    ("seek_advice", "RELATIONSHIP"): ComplexityTier.HIGH,
    ("seek_advice", "META"): ComplexityTier.HIGH,
    ("reflect", "META"): ComplexityTier.HIGH,
    ("reflect", "RELATIONSHIP"): ComplexityTier.HIGH,
    ("reflect", "HEALTH"): ComplexityTier.HIGH,
    ("express_feeling", "HEALTH"): ComplexityTier.HIGH,  # Health concerns need care
    ("express_feeling", "CONCERN"): ComplexityTier.HIGH,
    ("reflect", "CONCERN"): ComplexityTier.HIGH,
    ("other", "META"): ComplexityTier.HIGH,
}

# Fallback mappings when exact match not found
INTENT_DEFAULT_COMPLEXITY: Dict[str, ComplexityTier] = {
    "log_memory": ComplexityTier.LOW,
    "set_reminder": ComplexityTier.LOW,
    "query_memory": ComplexityTier.MEDIUM,
    "express_feeling": ComplexityTier.MEDIUM,
    "share_news": ComplexityTier.MEDIUM,
    "seek_advice": ComplexityTier.HIGH,
    "reflect": ComplexityTier.HIGH,
    "other": ComplexityTier.MEDIUM,
}

INGRESS_COMPLEXITY_BOOST: Dict[str, int] = {
    # Domains that increase complexity
    "HEALTH": 1,
    "FINANCE": 1,
    "CONCERN": 1,
    "META": 1,
    # Domains that keep complexity low
    "DIARY": 0,
    "TASK": 0,
    "MEMORY": 0,
    "PLANNING": 0,
    "CELEBRATION": 0,
    "GRATITUDE": 0,
    "RELATIONSHIP": 0,
    "WORK": 0,
}

# =============================================================================
# Complexity Modifiers
# =============================================================================

# High-intensity emotion words that boost complexity
HIGH_INTENSITY_WORDS = {
    "extremely", "really", "very", "so", "incredibly", "absolutely",
    "terribly", "awfully", "deeply", "seriously", "desperately",
    "completely", "totally", "utterly", "severely", "intensely",
}

# Concern/distress indicators that boost complexity
CONCERN_INDICATORS = {
    "worried", "scared", "afraid", "anxious", "nervous", "stressed",
    "depressed", "sad", "upset", "hurt", "angry", "frustrated",
    "overwhelmed", "lost", "confused", "struggling", "suffering",
    "pain", "crisis", "emergency", "urgent", "help",
}

# Advice-seeking patterns that boost complexity
ADVICE_PATTERNS = {
    "should i", "what should", "how do i", "how can i", "what do you think",
    "is it okay", "would you recommend", "any advice", "need help with",
    "what would you", "how would you", "do you think",
}

# Simple command patterns that lower complexity
SIMPLE_COMMAND_PATTERNS = {
    "remind me", "remember", "save this", "note this", "add to",
    "set reminder", "schedule", "don't forget", "memo",
}


def get_base_complexity(intent: str, ingress: str) -> Tuple[ComplexityTier, str]:
    """Get base complexity from intent + ingress mapping."""
    key = (intent, ingress)

    if key in INTENT_INGRESS_COMPLEXITY:
        tier = INTENT_INGRESS_COMPLEXITY[key]
        return tier, f"Direct mapping: {intent} + {ingress}"

    # Fallback to intent default with ingress boost
    base = INTENT_DEFAULT_COMPLEXITY.get(intent, ComplexityTier.MEDIUM)
    boost = INGRESS_COMPLEXITY_BOOST.get(ingress, 0)

    if boost > 0 and base == ComplexityTier.LOW:
        base = ComplexityTier.MEDIUM
    elif boost > 0 and base == ComplexityTier.MEDIUM:
        base = ComplexityTier.HIGH

    return base, f"Fallback: {intent} default + {ingress} boost={boost}"


def apply_modifiers(
    text: str,
    base_tier: ComplexityTier,
    emotions: List[str],
    safety: str,
    intent_confidence: float,
    ingress_confidence: float,
) -> Tuple[ComplexityTier, List[str]]:
    """Apply complexity modifiers based on text and signals."""
    modifiers = []
    tier = base_tier
    text_lower = text.lower()
    word_count = len(text.split())

    # 1. Safety Override - CRISIS always wins
    if safety == "CRISIS":
        return ComplexityTier.CRISIS, ["SAFETY: CRISIS detected"]

    if safety == "RED":
        tier = ComplexityTier.HIGH
        modifiers.append("SAFETY: RED flag -> HIGH")

    if safety == "AMBER" and tier == ComplexityTier.LOW:
        tier = ComplexityTier.MEDIUM
        modifiers.append("SAFETY: AMBER flag -> MEDIUM minimum")

    # 2. Emotional Intensity
    negative_emotions = {"sadness", "anger", "fear", "grief", "frustration",
                         "overwhelmed", "emptiness", "worry", "nervousness"}
    detected_negative = set(emotions) & negative_emotions

    if len(detected_negative) >= 2 and tier != ComplexityTier.HIGH:
        tier = ComplexityTier.HIGH
        modifiers.append(f"Multiple negative emotions: {detected_negative}")

    # 3. High-intensity words
    intensity_found = [w for w in HIGH_INTENSITY_WORDS if w in text_lower]
    if intensity_found and tier == ComplexityTier.LOW:
        tier = ComplexityTier.MEDIUM
        modifiers.append(f"Intensity words: {intensity_found[:3]}")

    # 4. Concern indicators
    concerns_found = [w for w in CONCERN_INDICATORS if w in text_lower]
    if len(concerns_found) >= 2 and tier != ComplexityTier.HIGH:
        tier = ComplexityTier.HIGH
        modifiers.append(f"Concern indicators: {concerns_found[:3]}")
    elif concerns_found and tier == ComplexityTier.LOW:
        tier = ComplexityTier.MEDIUM
        modifiers.append(f"Concern indicator: {concerns_found[0]}")

    # 5. Advice-seeking patterns
    for pattern in ADVICE_PATTERNS:
        if pattern in text_lower:
            if tier == ComplexityTier.LOW:
                tier = ComplexityTier.MEDIUM
                modifiers.append(f"Advice pattern: '{pattern}'")
            elif tier == ComplexityTier.MEDIUM:
                tier = ComplexityTier.HIGH
                modifiers.append(f"Advice pattern: '{pattern}' -> HIGH")
            break

    # 6. Simple command patterns (can lower complexity)
    for pattern in SIMPLE_COMMAND_PATTERNS:
        if pattern in text_lower and tier == ComplexityTier.MEDIUM:
            # Only downgrade if high confidence
            if intent_confidence > 0.85:
                tier = ComplexityTier.LOW
                modifiers.append(f"Simple command: '{pattern}' (high conf)")
            break

    # 7. Length modifier (long texts more complex)
    if word_count > 25 and tier == ComplexityTier.LOW:
        tier = ComplexityTier.MEDIUM
        modifiers.append(f"Long text: {word_count} words")
    elif word_count > 50 and tier == ComplexityTier.MEDIUM:
        tier = ComplexityTier.HIGH
        modifiers.append(f"Very long text: {word_count} words")

    # 8. Low confidence penalty (uncertainty -> higher tier)
    if intent_confidence < 0.5 and tier == ComplexityTier.LOW:
        tier = ComplexityTier.MEDIUM
        modifiers.append(f"Low intent confidence: {intent_confidence:.1%}")

    return tier, modifiers


def classify_complexity(
    text: str,
    intent: str,
    ingress: str,
    safety: str,
    emotions: List[str],
    intent_confidence: float = 0.9,
    ingress_confidence: float = 0.9,
) -> ComplexityResult:
    """
    Classify the complexity tier for a user request.

    Args:
        text: User input text
        intent: Predicted intent class
        ingress: Predicted ingress domain
        safety: Safety band (GREEN/AMBER/RED/CRISIS)
        emotions: List of detected emotions
        intent_confidence: Confidence score for intent
        ingress_confidence: Confidence score for ingress

    Returns:
        ComplexityResult with tier and reasoning
    """
    # Get base complexity
    base_tier, base_reason = get_base_complexity(intent, ingress)

    # Apply modifiers
    final_tier, modifiers = apply_modifiers(
        text, base_tier, emotions, safety,
        intent_confidence, ingress_confidence
    )

    # Build reason string
    if modifiers:
        reason = f"{base_reason} -> {' -> '.join(modifiers)}"
    else:
        reason = base_reason

    safety_override = safety in ("CRISIS", "RED")

    return ComplexityResult(
        tier=final_tier,
        base_tier=base_tier,
        intent=intent,
        ingress=ingress,
        reason=reason,
        modifiers_applied=modifiers,
        safety_override=safety_override,
        confidence=min(intent_confidence, ingress_confidence),
    )


# =============================================================================
# Routing Recommendations
# =============================================================================

TIER_ROUTING = {
    ComplexityTier.LOW: {
        "handler": "Tool Execution",
        "examples": ["Calendar API", "Note taking", "Reminder service"],
        "latency_target_ms": 50,
        "description": "Direct tool call, no LLM needed",
    },
    ComplexityTier.MEDIUM: {
        "handler": "Small LLM",
        "examples": ["GPT-3.5", "Claude Haiku", "Gemini Flash"],
        "latency_target_ms": 500,
        "description": "Simple emotional response, acknowledgment",
    },
    ComplexityTier.HIGH: {
        "handler": "Full LLM System",
        "examples": ["GPT-4", "Claude Sonnet", "Multi-turn reasoning"],
        "latency_target_ms": 2000,
        "description": "Complex advice, nuanced understanding",
    },
    ComplexityTier.CRISIS: {
        "handler": "Crisis Protocol",
        "examples": ["Immediate escalation", "Safety resources", "Human handoff"],
        "latency_target_ms": 100,
        "description": "Bypass normal flow, safety first",
    },
}


def get_routing_recommendation(tier: ComplexityTier) -> Dict:
    """Get routing recommendation for a complexity tier."""
    return TIER_ROUTING.get(tier, TIER_ROUTING[ComplexityTier.MEDIUM])

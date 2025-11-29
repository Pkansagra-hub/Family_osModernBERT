"""
Temporal Safety Monitor

This module provides temporal tracking and escalation logic for safety signals
over time. It enables detection of patterns that may indicate escalating risk,
even when individual signals don't reach crisis level.

Key Features:
    - Track safety signals over configurable time windows
    - Detect escalation patterns (multiple AMBERs → RED, RED + isolation → CRISIS)
    - Integration with SafetyEvaluator for real-time monitoring
    - Configurable escalation rules and thresholds

Escalation Rules:
    1. AMBER Accumulation: 3+ AMBER signals in window → escalate to RED
    2. Isolation Pattern: RED + isolation keywords → escalate to CRISIS
    3. Rapid Escalation: GREEN → AMBER → RED in short time → flag for review

Usage:
    from modeling_studio.evaluation.temporal_safety import (
        TemporalSafetyMonitor, SafetySignal
    )

    monitor = TemporalSafetyMonitor(window_days=7)

    # Track signals over time
    monitor.add_signal(SafetySignal(band="AMBER", text="Stressed about work"))
    monitor.add_signal(SafetySignal(band="AMBER", text="Feeling down"))
    escalation = monitor.add_signal(SafetySignal(band="AMBER", text="Can't sleep"))

    if escalation:
        print(f"Escalated to {escalation.to_band}: {escalation.reason}")

Issue: 3.6.5 - Implement TemporalSafetyMonitor
Epic: 3.6 - Production Readiness
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


class SafetyBand(IntEnum):
    """Safety band levels with ordering."""

    GREEN = 0
    AMBER = 1
    RED = 2
    CRISIS = 3

    @classmethod
    def from_string(cls, band: str) -> SafetyBand:
        """Convert string to SafetyBand enum."""
        return cls[band.upper()]

    def __str__(self) -> str:
        return self.name


# Isolation keywords that escalate RED → CRISIS
ISOLATION_KEYWORDS = frozenset(
    {
        # Direct isolation
        "alone",
        "nobody",
        "no one",
        "by myself",
        "all alone",
        "on my own",
        # Relationship isolation
        "nobody cares",
        "no one cares",
        "nobody loves",
        "no one loves",
        "nobody understands",
        "no one understands",
        # Social isolation
        "no friends",
        "lost everyone",
        "pushed everyone away",
        "nobody would miss",
        "no one would miss",
        # Emotional isolation
        "completely alone",
        "totally alone",
        "utterly alone",
        "so alone",
        "feel so alone",
        "invisible",
        "forgotten",
        "abandoned",
        # Future hopelessness combined with isolation
        "always alone",
        "forever alone",
        "die alone",
    }
)

# Rapid escalation keywords (GREEN/AMBER → concern if combined with RED)
ESCALATION_KEYWORDS = frozenset(
    {
        "getting worse",
        "can't take it",
        "breaking point",
        "falling apart",
        "losing control",
        "spiral",
        "spiraling",
        "rock bottom",
        "end of rope",
        "last straw",
        "giving up",
        "had enough",
    }
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SafetySignal:
    """A single safety signal with band, text, and timestamp.

    Attributes:
        band: Safety band classification (GREEN, AMBER, RED, CRISIS)
        text: Original text that was classified
        timestamp: When the signal was recorded (defaults to now)
        indicators: Optional list of detected indicators/keywords
        confidence: Model confidence in the classification
        user_id: Optional user identifier for multi-user tracking
        context: Optional additional context metadata
    """

    band: SafetyBand
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    indicators: list[str] = field(default_factory=list)
    confidence: float = 1.0
    user_id: str | None = None
    context: dict | None = None

    @classmethod
    def create(
        cls,
        band: str | SafetyBand,
        text: str,
        timestamp: datetime | None = None,
        indicators: list[str] | None = None,
        confidence: float = 1.0,
        user_id: str | None = None,
        context: dict | None = None,
    ) -> SafetySignal:
        """Factory method to create a SafetySignal with string band support."""
        if isinstance(band, str):
            band = SafetyBand.from_string(band)
        return cls(
            band=band,
            text=text,
            timestamp=timestamp or datetime.now(),
            indicators=indicators or [],
            confidence=confidence,
            user_id=user_id,
            context=context,
        )

    @property
    def band_name(self) -> str:
        """Get band as string."""
        return str(self.band)

    @property
    def severity(self) -> int:
        """Get numeric severity (0-3)."""
        return int(self.band)


@dataclass
class SafetyEscalation:
    """Record of a safety escalation event.

    Attributes:
        from_band: Original safety band
        to_band: Escalated safety band
        reason: Human-readable explanation of escalation
        trigger_signal: The signal that triggered escalation
        contributing_signals: Previous signals that contributed
        timestamp: When escalation occurred
        recommended_actions: Suggested response actions
    """

    from_band: SafetyBand
    to_band: SafetyBand
    reason: str
    trigger_signal: SafetySignal | None = None
    contributing_signals: list[SafetySignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    recommended_actions: list[str] = field(default_factory=list)

    @property
    def from_band_name(self) -> str:
        """Get from_band as string."""
        return str(self.from_band)

    @property
    def to_band_name(self) -> str:
        """Get to_band as string."""
        return str(self.to_band)

    @property
    def severity_increase(self) -> int:
        """Get the severity increase (0-3)."""
        return int(self.to_band) - int(self.from_band)


@dataclass
class EscalationRule:
    """A configurable escalation rule.

    Attributes:
        name: Rule identifier
        description: Human-readable description
        check_fn: Function that checks if rule triggers
        from_band: Source band (or None for any)
        to_band: Target band on escalation
        priority: Rule priority (higher = checked first)
    """

    name: str
    description: str
    check_fn: Callable[[list[SafetySignal], SafetySignal], tuple[bool, str]]
    from_band: SafetyBand | None = None
    to_band: SafetyBand = SafetyBand.RED
    priority: int = 0
    recommended_actions: list[str] = field(default_factory=list)


# =============================================================================
# Temporal Safety Monitor
# =============================================================================


class TemporalSafetyMonitor:
    """Monitor safety signals over time and detect escalation patterns.

    This class tracks safety signals within a configurable time window and
    applies escalation rules to detect concerning patterns.

    Default Escalation Rules:
        1. AMBER Accumulation: 3+ AMBER signals in window → RED
        2. Isolation Pattern: RED + isolation keywords → CRISIS
        3. Rapid Deterioration: Quick progression through bands → flag

    Attributes:
        window_days: Number of days to track signals (default 7)
        amber_threshold: Number of AMBERs before escalation (default 3)
        signals: Deque of signals within the window

    Example:
        >>> monitor = TemporalSafetyMonitor(window_days=7)
        >>> monitor.add_signal(SafetySignal(band="AMBER", text="Stressed"))
        >>> monitor.add_signal(SafetySignal(band="AMBER", text="Anxious"))
        >>> escalation = monitor.add_signal(SafetySignal(band="AMBER", text="Overwhelmed"))
        >>> if escalation:
        ...     print(f"Escalated: {escalation.reason}")
    """

    def __init__(
        self,
        window_days: int = 7,
        amber_threshold: int = 3,
        enable_isolation_detection: bool = True,
        enable_rapid_escalation_detection: bool = True,
        max_signals: int = 1000,
        custom_rules: list[EscalationRule] | None = None,
    ) -> None:
        """Initialize the temporal safety monitor.

        Args:
            window_days: Time window for tracking signals (default 7 days)
            amber_threshold: Number of AMBER signals to trigger escalation (default 3)
            enable_isolation_detection: Enable RED + isolation → CRISIS rule
            enable_rapid_escalation_detection: Enable rapid progression detection
            max_signals: Maximum signals to keep in memory
            custom_rules: Additional custom escalation rules
        """
        self.window_days = window_days
        self.amber_threshold = amber_threshold
        self.enable_isolation_detection = enable_isolation_detection
        self.enable_rapid_escalation_detection = enable_rapid_escalation_detection
        self.max_signals = max_signals

        # Signal storage (deque for efficient cleanup)
        self.signals: deque[SafetySignal] = deque(maxlen=max_signals)

        # Escalation history
        self.escalations: list[SafetyEscalation] = []

        # Build escalation rules
        self._rules: list[EscalationRule] = []
        self._build_default_rules()

        if custom_rules:
            for rule in custom_rules:
                self.add_rule(rule)

        # Sort rules by priority (descending)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info(
            f"TemporalSafetyMonitor initialized: "
            f"window={window_days}d, amber_threshold={amber_threshold}"
        )

    def _build_default_rules(self) -> None:
        """Build the default escalation rules."""
        # Rule 1: AMBER accumulation → RED
        amber_rule = EscalationRule(
            name="amber_accumulation",
            description=f"{self.amber_threshold}+ AMBER signals in window",
            check_fn=self._check_amber_accumulation,
            from_band=SafetyBand.AMBER,
            to_band=SafetyBand.RED,
            priority=100,
            recommended_actions=[
                "Schedule check-in conversation",
                "Increase monitoring frequency",
                "Consider professional support resources",
            ],
        )
        self._rules.append(amber_rule)

        # Rule 2: RED + isolation → CRISIS
        if self.enable_isolation_detection:
            isolation_rule = EscalationRule(
                name="red_isolation",
                description="RED signal with isolation indicators",
                check_fn=self._check_isolation_pattern,
                from_band=SafetyBand.RED,
                to_band=SafetyBand.CRISIS,
                priority=200,
                recommended_actions=[
                    "Immediate outreach recommended",
                    "Provide crisis hotline resources",
                    "Consider welfare check if no response",
                ],
            )
            self._rules.append(isolation_rule)

        # Rule 3: Rapid escalation detection
        if self.enable_rapid_escalation_detection:
            rapid_rule = EscalationRule(
                name="rapid_escalation",
                description="Rapid progression through safety bands",
                check_fn=self._check_rapid_escalation,
                from_band=None,  # Any starting band
                to_band=SafetyBand.RED,  # At least flag as RED
                priority=150,
                recommended_actions=[
                    "Flag for immediate review",
                    "Check recent life events",
                    "Consider proactive outreach",
                ],
            )
            self._rules.append(rapid_rule)

    def add_rule(self, rule: EscalationRule) -> None:
        """Add a custom escalation rule.

        Args:
            rule: The escalation rule to add
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.debug(f"Added escalation rule: {rule.name}")

    def add_signal(
        self,
        signal: SafetySignal,
    ) -> SafetyEscalation | None:
        """Add a safety signal and check for escalation.

        Args:
            signal: The safety signal to add

        Returns:
            SafetyEscalation if an escalation is triggered, None otherwise
        """
        # Clean up old signals
        self._cleanup_old_signals()

        # Add the new signal
        self.signals.append(signal)
        logger.debug(f"Added signal: band={signal.band_name}, text={signal.text[:50]}...")

        # Check escalation rules
        escalation = self._check_escalation(signal)

        if escalation:
            self.escalations.append(escalation)
            logger.warning(
                f"Safety escalation: {escalation.from_band_name} → "
                f"{escalation.to_band_name}: {escalation.reason}"
            )

        return escalation

    def _cleanup_old_signals(self) -> None:
        """Remove signals outside the time window."""
        cutoff = datetime.now() - timedelta(days=self.window_days)

        while self.signals and self.signals[0].timestamp < cutoff:
            self.signals.popleft()

    def _check_escalation(self, new_signal: SafetySignal) -> SafetyEscalation | None:
        """Check all escalation rules against current state.

        Args:
            new_signal: The newly added signal

        Returns:
            SafetyEscalation if a rule triggers, None otherwise
        """
        window_signals = list(self.signals)

        for rule in self._rules:
            # Check if rule applies to this signal's band
            if rule.from_band is not None and new_signal.band != rule.from_band:
                continue

            # Run the rule's check function
            triggered, reason = rule.check_fn(window_signals, new_signal)

            if triggered:
                # Find contributing signals
                contributing = [
                    s for s in window_signals if s is not new_signal and s.band >= SafetyBand.AMBER
                ]

                return SafetyEscalation(
                    from_band=new_signal.band,
                    to_band=rule.to_band,
                    reason=reason,
                    trigger_signal=new_signal,
                    contributing_signals=contributing,
                    recommended_actions=rule.recommended_actions.copy(),
                )

        return None

    def _check_amber_accumulation(
        self,
        window_signals: list[SafetySignal],
        new_signal: SafetySignal,
    ) -> tuple[bool, str]:
        """Check for AMBER signal accumulation.

        Args:
            window_signals: All signals in the current window
            new_signal: The newly added signal

        Returns:
            Tuple of (triggered, reason)
        """
        # Count AMBER signals in window
        amber_count = sum(1 for s in window_signals if s.band == SafetyBand.AMBER)

        if amber_count >= self.amber_threshold:
            return True, f"{amber_count} AMBER signals in {self.window_days}-day window"

        return False, ""

    def _check_isolation_pattern(
        self,
        window_signals: list[SafetySignal],
        new_signal: SafetySignal,
    ) -> tuple[bool, str]:
        """Check for RED + isolation keyword pattern.

        Args:
            window_signals: All signals in the current window
            new_signal: The newly added signal

        Returns:
            Tuple of (triggered, reason)
        """
        if new_signal.band != SafetyBand.RED:
            return False, ""

        # Check for isolation keywords in the text
        text_lower = new_signal.text.lower()
        found_keywords = []

        for keyword in ISOLATION_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)

        if found_keywords:
            return True, f"RED signal with isolation indicators: {', '.join(found_keywords[:3])}"

        return False, ""

    def _check_rapid_escalation(
        self,
        window_signals: list[SafetySignal],
        new_signal: SafetySignal,
    ) -> tuple[bool, str]:
        """Check for rapid escalation pattern.

        Triggers if:
        - GREEN → AMBER → RED within 48 hours
        - Any band + escalation keywords

        Args:
            window_signals: All signals in the current window
            new_signal: The newly added signal

        Returns:
            Tuple of (triggered, reason)
        """
        # Check for escalation keywords in current signal
        text_lower = new_signal.text.lower()
        found_keywords = [kw for kw in ESCALATION_KEYWORDS if kw in text_lower]

        if found_keywords and new_signal.band >= SafetyBand.AMBER:
            return True, f"Escalation indicators detected: {', '.join(found_keywords[:2])}"

        # Check for rapid band progression (48 hours)
        recent_cutoff = datetime.now() - timedelta(hours=48)
        recent_signals = [s for s in window_signals if s.timestamp >= recent_cutoff]

        if len(recent_signals) >= 3:
            bands = [s.band for s in recent_signals]
            # Check for GREEN → AMBER → RED progression
            if SafetyBand.GREEN in bands and SafetyBand.AMBER in bands and SafetyBand.RED in bands:
                return True, "Rapid escalation: GREEN → AMBER → RED within 48 hours"

        return False, ""

    def get_signals_in_window(
        self,
        band: SafetyBand | str | None = None,
    ) -> list[SafetySignal]:
        """Get signals in the current window, optionally filtered by band.

        Args:
            band: Optional band to filter by

        Returns:
            List of signals in the window
        """
        self._cleanup_old_signals()

        if band is None:
            return list(self.signals)

        if isinstance(band, str):
            band = SafetyBand.from_string(band)

        return [s for s in self.signals if s.band == band]

    def get_signal_counts(self) -> dict[str, int]:
        """Get count of signals by band in current window.

        Returns:
            Dictionary mapping band names to counts
        """
        self._cleanup_old_signals()

        counts = {band.name: 0 for band in SafetyBand}
        for signal in self.signals:
            counts[signal.band.name] += 1

        return counts

    def get_escalation_history(
        self,
        limit: int | None = None,
    ) -> list[SafetyEscalation]:
        """Get recent escalation history.

        Args:
            limit: Maximum number of escalations to return

        Returns:
            List of escalations (most recent first)
        """
        if limit is None:
            return list(reversed(self.escalations))
        return list(reversed(self.escalations[-limit:]))

    def get_risk_score(self) -> float:
        """Calculate an aggregate risk score based on recent signals.

        Returns:
            Risk score from 0.0 (low risk) to 1.0 (high risk)
        """
        self._cleanup_old_signals()

        if not self.signals:
            return 0.0

        # Weight by recency and severity
        now = datetime.now()
        total_weight = 0.0
        weighted_score = 0.0

        for signal in self.signals:
            # Recency weight (more recent = higher weight)
            age_days = (now - signal.timestamp).total_seconds() / 86400
            recency_weight = max(0.1, 1.0 - (age_days / self.window_days))

            # Severity weight
            severity = signal.severity / 3.0  # Normalize to 0-1

            total_weight += recency_weight
            weighted_score += severity * recency_weight

        if total_weight == 0:
            return 0.0

        base_score = weighted_score / total_weight

        # Boost for escalation patterns
        counts = self.get_signal_counts()
        if counts["RED"] > 0 or counts["CRISIS"] > 0:
            base_score = min(1.0, base_score + 0.2)
        if counts["AMBER"] >= self.amber_threshold:
            base_score = min(1.0, base_score + 0.15)

        return min(1.0, base_score)

    def clear(self) -> None:
        """Clear all signals and escalation history."""
        self.signals.clear()
        self.escalations.clear()
        logger.info("TemporalSafetyMonitor cleared")

    def reset_window(self) -> None:
        """Clear signals but keep escalation history."""
        self.signals.clear()
        logger.info("TemporalSafetyMonitor window reset")

    def __repr__(self) -> str:
        counts = self.get_signal_counts()
        return (
            f"TemporalSafetyMonitor("
            f"window={self.window_days}d, "
            f"signals={len(self.signals)}, "
            f"counts={counts})"
        )


# =============================================================================
# Integration with SafetyEvaluator
# =============================================================================


class SafetyMonitorIntegration:
    """Integration helper for TemporalSafetyMonitor with SafetyEvaluator.

    This class provides methods to bridge the temporal monitor with the
    main safety evaluation pipeline.

    Example:
        >>> from modeling_studio.evaluation.safety_eval import SafetyEvaluator
        >>> from modeling_studio.evaluation.temporal_safety import (
        ...     TemporalSafetyMonitor, SafetyMonitorIntegration
        ... )
        >>>
        >>> evaluator = SafetyEvaluator(model, tokenizer)
        >>> monitor = TemporalSafetyMonitor()
        >>> integration = SafetyMonitorIntegration(monitor, evaluator)
        >>>
        >>> result = integration.evaluate_with_temporal(text)
    """

    def __init__(
        self,
        monitor: TemporalSafetyMonitor,
        evaluator: Any | None = None,  # SafetyEvaluator, but avoid circular import
    ) -> None:
        """Initialize the integration.

        Args:
            monitor: The temporal safety monitor
            evaluator: Optional SafetyEvaluator instance
        """
        self.monitor = monitor
        self.evaluator = evaluator

    def process_evaluation_result(
        self,
        text: str,
        band: str,
        confidence: float = 1.0,
        indicators: list[str] | None = None,
        user_id: str | None = None,
    ) -> tuple[str, SafetyEscalation | None]:
        """Process an evaluation result through the temporal monitor.

        Args:
            text: The evaluated text
            band: The predicted safety band
            confidence: Model confidence
            indicators: Detected indicators
            user_id: Optional user identifier

        Returns:
            Tuple of (final_band, escalation_if_any)
        """
        signal = SafetySignal.create(
            band=band,
            text=text,
            confidence=confidence,
            indicators=indicators,
            user_id=user_id,
        )

        escalation = self.monitor.add_signal(signal)

        if escalation:
            # Return the escalated band
            return str(escalation.to_band), escalation
        else:
            return band, None

    def get_user_risk_profile(
        self,
        user_id: str,
    ) -> dict:
        """Get risk profile for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with risk profile information
        """
        user_signals = [s for s in self.monitor.signals if s.user_id == user_id]

        if not user_signals:
            return {
                "user_id": user_id,
                "signal_count": 0,
                "risk_level": "unknown",
                "risk_score": 0.0,
            }

        # Calculate user-specific risk
        counts = {band.name: 0 for band in SafetyBand}
        for signal in user_signals:
            counts[signal.band.name] += 1

        # Determine risk level
        if counts["CRISIS"] > 0:
            risk_level = "critical"
        elif counts["RED"] > 0:
            risk_level = "high"
        elif counts["AMBER"] >= self.monitor.amber_threshold:
            risk_level = "elevated"
        elif counts["AMBER"] > 0:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "user_id": user_id,
            "signal_count": len(user_signals),
            "signal_counts": counts,
            "risk_level": risk_level,
            "risk_score": self.monitor.get_risk_score(),
            "recent_escalations": [
                e
                for e in self.monitor.escalations
                if e.trigger_signal and e.trigger_signal.user_id == user_id
            ],
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_monitor(
    window_days: int = 7,
    amber_threshold: int = 3,
    strict_mode: bool = False,
) -> TemporalSafetyMonitor:
    """Create a configured TemporalSafetyMonitor.

    Args:
        window_days: Time window for tracking
        amber_threshold: AMBER count threshold
        strict_mode: If True, use stricter thresholds

    Returns:
        Configured TemporalSafetyMonitor
    """
    if strict_mode:
        amber_threshold = min(2, amber_threshold)
        window_days = max(14, window_days)

    return TemporalSafetyMonitor(
        window_days=window_days,
        amber_threshold=amber_threshold,
        enable_isolation_detection=True,
        enable_rapid_escalation_detection=True,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "SafetyBand",
    # Data classes
    "SafetySignal",
    "SafetyEscalation",
    "EscalationRule",
    # Main class
    "TemporalSafetyMonitor",
    # Integration
    "SafetyMonitorIntegration",
    # Constants
    "ISOLATION_KEYWORDS",
    "ESCALATION_KEYWORDS",
    # Factory
    "create_monitor",
]

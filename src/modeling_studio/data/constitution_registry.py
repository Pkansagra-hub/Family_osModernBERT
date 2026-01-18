"""
Constitution Registry for 3-Layer Constitutional Training.

FamilyOS Constitutional Architecture:
    Layer 1: Family Values - Core family principles (privacy, respect, support)
    Layer 2: Individual Preferences - Per-member boundaries and comfort levels
    Layer 3: Situational Context - Context-adaptive rules (sensitive topics, emergencies)

This registry provides:
    - Mapping from constitution names to integer IDs for embedding lookup
    - Schema loading from constitution_schemas_v2.json
    - Cultural context mapping from training data metadata
    - 3-layer constitution composition for training

Usage:
    from modeling_studio.data.constitution_registry import (
        ConstitutionRegistry,
        get_constitution_id,
        get_constitution_text,
        FAMILY_VALUES_TO_ID,
    )

    registry = ConstitutionRegistry.load_default()
    const_id = registry.get_family_value_id("gentle_parenting")
    const_text = registry.get_constitution_text("gentle_parenting")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Layer 1: Family Values (Static, set during onboarding)
# =============================================================================

FAMILY_VALUES_TO_ID: dict[str, int] = {
    # Default/fallback
    "universal": 0,
    "default": 0,

    # Parenting styles
    "gentle_parenting": 1,
    "traditional_strict": 2,
    "balanced_approach": 3,
    "authoritative": 4,

    # Cultural family structures
    "indian_joint_family": 5,
    "western_nuclear": 6,
    "asian_collectivist": 7,

    # Additional styles (reserved for expansion)
    "permissive": 8,
    "attachment_parenting": 9,
}

ID_TO_FAMILY_VALUES: dict[int, str] = {v: k for k, v in FAMILY_VALUES_TO_ID.items()}


# =============================================================================
# Layer 2: Individual Preferences (Per-actor, dynamic)
# =============================================================================

INDIVIDUAL_PREF_TO_ID: dict[str, int] = {
    "default": 0,
    "concise_casual": 1,      # Short, informal responses
    "detailed_formal": 2,     # Long, formal responses
    "warm_nurturing": 3,      # Emotionally supportive
    "practical_direct": 4,    # Action-oriented, no fluff
    "validation_first": 5,    # Always acknowledge feelings first
}

ID_TO_INDIVIDUAL_PREF: dict[int, str] = {v: k for k, v in INDIVIDUAL_PREF_TO_ID.items()}


# =============================================================================
# Layer 3: Situational Context (Dynamic, event-driven)
# =============================================================================

SITUATIONAL_CONTEXT_TO_ID: dict[str, int] = {
    "normal": 0,              # Default context
    "high_arousal": 1,        # User is stressed/excited
    "negative_valence": 2,    # User is upset/sad
    "crisis_band": 3,         # Emergency situation
    "public_context": 4,      # In public, need discretion
    "high_novelty": 5,        # New/unusual situation
    "sensitive_topic": 6,     # Health, finances, relationships
    "child_present": 7,       # Child in conversation
}

ID_TO_SITUATIONAL_CONTEXT: dict[int, str] = {v: k for k, v in SITUATIONAL_CONTEXT_TO_ID.items()}


# =============================================================================
# Cultural Context Mapping (from training data metadata)
# =============================================================================

CULTURAL_CONTEXT_TO_FAMILY_VALUE: dict[str, str] = {
    # Map metadata.cultural_context -> family_values constitution
    "universal": "universal",
    "indian": "indian_joint_family",
    "western": "western_nuclear",
    "asian": "asian_collectivist",

    # Fallbacks
    "": "universal",
    None: "universal",
}


# =============================================================================
# Constitution Registry Class
# =============================================================================

@dataclass
class ConstitutionRegistry:
    """
    Central registry for 3-layer constitutional training.

    Provides mapping between constitution names and integer IDs,
    and utilities for composing multi-layer constitution embeddings.

    Attributes:
        family_values: Layer 1 schemas from constitution_schemas_v2.json
        individual_prefs: Layer 2 per-actor preferences
        situational_rules: Layer 3 context-adaptive rules
        schemas_path: Path to constitution schemas JSON file
    """

    family_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    individual_prefs: dict[str, dict[str, Any]] = field(default_factory=dict)
    situational_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    schemas_path: Path | None = None

    @classmethod
    def load_default(cls) -> "ConstitutionRegistry":
        """Load constitution schemas from default location."""
        # Find project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        # Try multiple possible locations
        schema_paths = [
            project_root / "data" / "constitutions" / "constitution_schemas_v2.json",
            project_root / "data" / "constitution_schemas_v2.json",
            Path("data/constitutions/constitution_schemas_v2.json"),
        ]

        for path in schema_paths:
            if path.exists():
                return cls.load_from_file(path)

        # Return empty registry with defaults if no file found
        logger.warning("No constitution schemas file found, using defaults")
        return cls()

    @classmethod
    def load_from_file(cls, path: Path | str) -> "ConstitutionRegistry":
        """Load constitution schemas from JSON file."""
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            schemas = json.load(f)

        registry = cls(
            family_values=schemas.get("family_values", {}),
            individual_prefs=schemas.get("individual_prefs", {}),
            situational_rules=schemas.get("situational_rules", {}),
            schemas_path=path,
        )

        # Remove meta keys
        registry.family_values.pop("_description", None)
        registry.individual_prefs.pop("_description", None)
        registry.situational_rules.pop("_description", None)

        logger.info(f"Loaded constitution schemas from {path}")
        logger.info(f"  Family values: {list(registry.family_values.keys())}")
        logger.info(f"  Individual prefs: {list(registry.individual_prefs.keys())}")
        logger.info(f"  Situational rules: {list(registry.situational_rules.keys())}")

        return registry

    # -------------------------------------------------------------------------
    # Layer 1: Family Values
    # -------------------------------------------------------------------------

    def get_family_value_id(self, name: str) -> int:
        """Get integer ID for a family value constitution."""
        return FAMILY_VALUES_TO_ID.get(name, FAMILY_VALUES_TO_ID["universal"])

    def get_family_value_name(self, id: int) -> str:
        """Get name for a family value ID."""
        return ID_TO_FAMILY_VALUES.get(id, "universal")

    def get_family_value_schema(self, name: str) -> dict[str, Any]:
        """Get full schema for a family value."""
        return self.family_values.get(name, {})

    # -------------------------------------------------------------------------
    # Layer 2: Individual Preferences
    # -------------------------------------------------------------------------

    def get_individual_pref_id(self, name: str) -> int:
        """Get integer ID for an individual preference."""
        return INDIVIDUAL_PREF_TO_ID.get(name, INDIVIDUAL_PREF_TO_ID["default"])

    def get_individual_pref_name(self, id: int) -> str:
        """Get name for an individual preference ID."""
        return ID_TO_INDIVIDUAL_PREF.get(id, "default")

    # -------------------------------------------------------------------------
    # Layer 3: Situational Context
    # -------------------------------------------------------------------------

    def get_situational_context_id(self, name: str) -> int:
        """Get integer ID for a situational context."""
        return SITUATIONAL_CONTEXT_TO_ID.get(name, SITUATIONAL_CONTEXT_TO_ID["normal"])

    def get_situational_context_name(self, id: int) -> str:
        """Get name for a situational context ID."""
        return ID_TO_SITUATIONAL_CONTEXT.get(id, "normal")

    # -------------------------------------------------------------------------
    # Composition Utilities
    # -------------------------------------------------------------------------

    def get_constitution_text(self, family_value: str) -> str:
        """
        Get human-readable constitution text for prefix injection.

        Combines description and core principles into a text prompt
        that can be prepended to decoder input.
        """
        schema = self.family_values.get(family_value, {})

        if not schema:
            return ""

        description = schema.get("description", "")
        principles = schema.get("core_principles", [])

        if principles:
            principles_text = ", ".join(principles)
            return f"{description}. Core principles: {principles_text}."

        return description

    def get_composite_constitution_id(
        self,
        family_value: str = "universal",
        individual_pref: str = "default",
        situational_context: str = "normal",
    ) -> tuple[int, int, int]:
        """
        Get composite 3-layer constitution as tuple of IDs.

        Returns:
            (family_value_id, individual_pref_id, situational_context_id)
        """
        return (
            self.get_family_value_id(family_value),
            self.get_individual_pref_id(individual_pref),
            self.get_situational_context_id(situational_context),
        )

    def map_cultural_context_to_family_value(self, cultural_context: str | None) -> str:
        """
        Map training data cultural_context to family_value constitution.

        Training data has metadata.cultural_context = "indian", "western", etc.
        This maps to the corresponding family_values constitution.
        """
        if cultural_context is None:
            return "universal"

        cultural_context = cultural_context.lower().strip()
        return CULTURAL_CONTEXT_TO_FAMILY_VALUE.get(cultural_context, "universal")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_num_family_values(self) -> int:
        """Get total number of family value constitutions."""
        return len(FAMILY_VALUES_TO_ID)

    def get_num_individual_prefs(self) -> int:
        """Get total number of individual preference types."""
        return len(INDIVIDUAL_PREF_TO_ID)

    def get_num_situational_contexts(self) -> int:
        """Get total number of situational context types."""
        return len(SITUATIONAL_CONTEXT_TO_ID)

    def to_dict(self) -> dict[str, Any]:
        """Export registry configuration."""
        return {
            "family_values_to_id": FAMILY_VALUES_TO_ID,
            "individual_pref_to_id": INDIVIDUAL_PREF_TO_ID,
            "situational_context_to_id": SITUATIONAL_CONTEXT_TO_ID,
            "num_family_values": self.get_num_family_values(),
            "num_individual_prefs": self.get_num_individual_prefs(),
            "num_situational_contexts": self.get_num_situational_contexts(),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

_DEFAULT_REGISTRY: ConstitutionRegistry | None = None


def get_default_registry() -> ConstitutionRegistry:
    """Get or create the default constitution registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ConstitutionRegistry.load_default()
    return _DEFAULT_REGISTRY


def get_constitution_id(
    family_value: str = "universal",
    cultural_context: str | None = None,
) -> int:
    """
    Get constitution ID from family value name or cultural context.

    Args:
        family_value: Direct family value name (e.g., "gentle_parenting")
        cultural_context: Training data cultural context (e.g., "indian")

    Returns:
        Integer ID for embedding lookup
    """
    registry = get_default_registry()

    if cultural_context:
        family_value = registry.map_cultural_context_to_family_value(cultural_context)

    return registry.get_family_value_id(family_value)


def get_constitution_text(family_value: str) -> str:
    """Get human-readable constitution text for the given family value."""
    registry = get_default_registry()
    return registry.get_constitution_text(family_value)


# =============================================================================
# Affect-based situational context detection
# =============================================================================

def detect_situational_context(
    affect_valence: float | None = None,
    affect_arousal: float | None = None,
    affect_band: str | None = None,
    social_context: str | None = None,
    novelty_score: float | None = None,
) -> str:
    """
    Detect situational context from event signals (st_hipp_events columns).

    Maps FamilyOS event signals to situational context for Layer 3 constitution.

    Args:
        affect_valence: Emotional valence (-1 to 1)
        affect_arousal: Emotional arousal (0 to 1)
        affect_band: Affect band (GREEN, YELLOW, RED, CRISIS)
        social_context: Social context (private, public, etc.)
        novelty_score: How novel/unusual the situation is (0 to 1)

    Returns:
        Situational context name for constitution lookup
    """
    # Priority order: crisis > high_arousal > negative_valence > public > high_novelty

    # Crisis band takes highest priority
    if affect_band and affect_band.upper() == "CRISIS":
        return "crisis_band"

    # High arousal (stressed/excited)
    if affect_arousal is not None and affect_arousal > 0.7:
        return "high_arousal"

    # Negative valence (upset/sad)
    if affect_valence is not None and affect_valence < -0.3:
        return "negative_valence"

    # Public context
    if social_context and social_context.lower() == "public":
        return "public_context"

    # High novelty
    if novelty_score is not None and novelty_score > 0.8:
        return "high_novelty"

    return "normal"


# =============================================================================
# Training data extraction helpers
# =============================================================================

def extract_constitution_from_sample(sample: dict) -> dict[str, Any]:
    """
    Extract 3-layer constitution info from a training sample.

    Looks for constitution info in:
    1. sample["constitution"] - Direct field
    2. sample["metadata"]["cultural_context"] - Cultural context
    3. sample["metadata"]["family_value"] - Direct family value
    4. Affect signals for situational context

    Args:
        sample: Training sample dict

    Returns:
        Dict with:
            - family_value: str
            - family_value_id: int
            - individual_pref: str
            - individual_pref_id: int
            - situational_context: str
            - situational_context_id: int
            - constitution_text: str
    """
    registry = get_default_registry()
    metadata = sample.get("metadata", {})

    # Layer 1: Family Value
    family_value = (
        sample.get("constitution") or
        sample.get("family_value") or
        metadata.get("family_value") or
        registry.map_cultural_context_to_family_value(
            metadata.get("cultural_context")
        )
    )

    # Layer 2: Individual Preference (default for training, per-actor at inference)
    individual_pref = (
        sample.get("individual_pref") or
        metadata.get("individual_pref") or
        "default"
    )

    # Layer 3: Situational Context (detect from affect signals)
    situational_context = detect_situational_context(
        affect_valence=sample.get("affect_valence") or metadata.get("affect_valence"),
        affect_arousal=sample.get("affect_arousal") or metadata.get("affect_arousal"),
        affect_band=sample.get("affect_band") or metadata.get("affect_band"),
        social_context=sample.get("social_context") or metadata.get("social_context"),
        novelty_score=sample.get("novelty_score") or metadata.get("novelty_score"),
    )

    return {
        "family_value": family_value,
        "family_value_id": registry.get_family_value_id(family_value),
        "individual_pref": individual_pref,
        "individual_pref_id": registry.get_individual_pref_id(individual_pref),
        "situational_context": situational_context,
        "situational_context_id": registry.get_situational_context_id(situational_context),
        "constitution_text": registry.get_constitution_text(family_value),
    }

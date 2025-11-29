"""
K0 Model Registry Integration for FamilyOS Unified NLP Model.

This module provides a unified registry for managing model capabilities,
enabling K0 modules to resolve capabilities to the appropriate model and head.

Issue: 3.6.3 - K0 Model Registry Integration
Epic: 3.6 - Production Readiness

Usage:
    from modeling_studio.k0.runtime.model_registry import (
        MODEL_REGISTRY,
        resolve_capability,
        get_unified_model,
        get_model_info,
    )

    # Resolve capability to model
    model_name, head_name = resolve_capability("ner_family")
    assert model_name == "familyos_unified_v2"

    # Get model instance
    model = get_unified_model()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Definitions
# =============================================================================

class Capability(str, Enum):
    """All capabilities supported by the unified model."""

    # Named Entity Recognition
    NER_GENERAL = "ner_general"
    NER_FAMILY = "ner_family"

    # Temporal Understanding
    TEMPORAL = "temporal"

    # Sentiment & Emotion
    SENTIMENT = "sentiment"
    EMOTIONS = "emotions"

    # Safety Classification
    SAFETY_GENERIC = "safety_generic"
    SAFETY_FAMILYOS = "safety_familyos"

    # Context & Intent
    INGRESS = "ingress"
    INTENT = "intent"

    # Relationships
    RELATION = "relation"

    # NLI & Embeddings
    NLI = "nli"
    EMBEDDING = "embedding"


# Capability aliases for backward compatibility
CAPABILITY_ALIASES: dict[str, Capability] = {
    # NER aliases
    "ner": Capability.NER_GENERAL,
    "entities": Capability.NER_GENERAL,
    "family_ner": Capability.NER_FAMILY,
    "family_entities": Capability.NER_FAMILY,

    # Temporal aliases
    "time": Capability.TEMPORAL,
    "temporal_extraction": Capability.TEMPORAL,

    # Sentiment aliases
    "sentiment_analysis": Capability.SENTIMENT,
    "emotion": Capability.EMOTIONS,
    "emotion_detection": Capability.EMOTIONS,

    # Safety aliases
    "safety": Capability.SAFETY_FAMILYOS,
    "content_safety": Capability.SAFETY_FAMILYOS,
    "moderation": Capability.SAFETY_FAMILYOS,
    "safety_check": Capability.SAFETY_FAMILYOS,

    # Context aliases
    "ingress_classify": Capability.INGRESS,
    "context_type": Capability.INGRESS,

    # Intent aliases
    "intent_detection": Capability.INTENT,
    "intent_classification": Capability.INTENT,

    # Relation aliases
    "relation_extraction": Capability.RELATION,
    "relationships": Capability.RELATION,

    # NLI aliases
    "entailment": Capability.NLI,
    "natural_language_inference": Capability.NLI,

    # Embedding aliases
    "embeddings": Capability.EMBEDDING,
    "encode": Capability.EMBEDDING,
    "sentence_embedding": Capability.EMBEDDING,
}


# =============================================================================
# Model Information
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    version: str
    base_model: str
    capabilities: list[Capability]

    # Model specifications
    hidden_size: int = 768
    max_sequence_length: int = 8192
    num_attention_heads: int = 12

    # Resource requirements
    memory_mb: int = 650
    latency_ms: float = 35.0

    # Paths
    checkpoint_path: str | None = None
    config_path: str | None = None

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Head mappings
    capability_to_head: dict[Capability, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default head mappings."""
        if not self.capability_to_head:
            self.capability_to_head = {
                Capability.NER_GENERAL: "ner_general_head",
                Capability.NER_FAMILY: "ner_family_head",
                Capability.TEMPORAL: "temporal_head",
                Capability.SENTIMENT: "sentiment_head",
                Capability.EMOTIONS: "emotion_head",
                Capability.SAFETY_GENERIC: "safety_generic_head",
                Capability.SAFETY_FAMILYOS: "enhanced_safety_head",
                Capability.INGRESS: "ingress_head",
                Capability.INTENT: "intent_head",
                Capability.RELATION: "relation_head",
                Capability.NLI: "nli_head",
                Capability.EMBEDDING: "embedding_head",
            }


@dataclass
class HeadInfo:
    """Information about a task-specific head."""

    name: str
    capability: Capability
    output_type: str  # "classification", "token_classification", "regression", "embedding"
    num_labels: int | None = None
    label_names: list[str] = field(default_factory=list)


# =============================================================================
# Model Registry
# =============================================================================

# Primary model registry
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "familyos_unified_v2": ModelInfo(
        name="familyos_unified_v2",
        version="2.0.0",
        base_model="answerdotai/ModernBERT-base",
        capabilities=list(Capability),
        hidden_size=768,
        max_sequence_length=8192,
        num_attention_heads=12,
        memory_mb=650,
        latency_ms=35.0,
        description="Unified multi-task NLP model for FamilyOS with 12 capabilities",
        tags=["production", "unified", "multi-task", "familyos"],
        capability_to_head={
            Capability.NER_GENERAL: "ner_general_head",
            Capability.NER_FAMILY: "ner_family_head",
            Capability.TEMPORAL: "temporal_head",
            Capability.SENTIMENT: "sentiment_head",
            Capability.EMOTIONS: "emotion_head",
            Capability.SAFETY_GENERIC: "safety_generic_head",
            Capability.SAFETY_FAMILYOS: "enhanced_safety_head",
            Capability.INGRESS: "ingress_head",
            Capability.INTENT: "intent_head",
            Capability.RELATION: "relation_head",
            Capability.NLI: "nli_head",
            Capability.EMBEDDING: "embedding_head",
        },
    ),
}

# Legacy model mappings (for migration)
LEGACY_MODEL_MAPPING: dict[str, str] = {
    # M02: hippocampus.semantic_project
    "distilbert-base-uncased-finetuned-sst-2-english": "familyos_unified_v2",
    "hippocampus_semantic": "familyos_unified_v2",

    # M04: affect.analyze
    "j-hartmann/emotion-english-distilroberta-base": "familyos_unified_v2",
    "affect_emotion": "familyos_unified_v2",
    "affect_sentiment": "familyos_unified_v2",

    # M10: context.ingress_classify
    "facebook/bart-large-mnli": "familyos_unified_v2",
    "ingress_classifier": "familyos_unified_v2",

    # P08: embedding pipeline
    "sentence-transformers/all-MiniLM-L6-v2": "familyos_unified_v2",
    "embedding_model": "familyos_unified_v2",

    # Safety models
    "unitary/toxic-bert": "familyos_unified_v2",
    "safety_classifier": "familyos_unified_v2",
}


# Head registry
HEAD_REGISTRY: dict[str, HeadInfo] = {
    "ner_general_head": HeadInfo(
        name="ner_general_head",
        capability=Capability.NER_GENERAL,
        output_type="token_classification",
        num_labels=17,  # Standard NER + O tag
        label_names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                     "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-MONEY", "I-MONEY",
                     "B-PERCENT", "I-PERCENT", "B-MISC", "I-MISC"],
    ),
    "ner_family_head": HeadInfo(
        name="ner_family_head",
        capability=Capability.NER_FAMILY,
        output_type="token_classification",
        num_labels=27,  # Family-specific entities
        label_names=["O", "B-FAMILY_MEMBER", "I-FAMILY_MEMBER", "B-RELATIONSHIP", "I-RELATIONSHIP",
                     "B-EVENT", "I-EVENT", "B-LOCATION", "I-LOCATION", "B-ACTIVITY", "I-ACTIVITY",
                     "B-FOOD", "I-FOOD", "B-EMOTION", "I-EMOTION", "B-HEALTH", "I-HEALTH",
                     "B-SCHOOL", "I-SCHOOL", "B-HOBBY", "I-HOBBY", "B-PET", "I-PET",
                     "B-MILESTONE", "I-MILESTONE", "B-TRADITION", "I-TRADITION"],
    ),
    "temporal_head": HeadInfo(
        name="temporal_head",
        capability=Capability.TEMPORAL,
        output_type="token_classification",
        num_labels=9,
        label_names=["O", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                     "B-DURATION", "I-DURATION", "B-RECURRENCE", "I-RECURRENCE"],
    ),
    "sentiment_head": HeadInfo(
        name="sentiment_head",
        capability=Capability.SENTIMENT,
        output_type="classification",
        num_labels=5,
        label_names=["very_negative", "negative", "neutral", "positive", "very_positive"],
    ),
    "emotion_head": HeadInfo(
        name="emotion_head",
        capability=Capability.EMOTIONS,
        output_type="classification",
        num_labels=8,
        label_names=["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"],
    ),
    "safety_generic_head": HeadInfo(
        name="safety_generic_head",
        capability=Capability.SAFETY_GENERIC,
        output_type="classification",
        num_labels=4,
        label_names=["GREEN", "AMBER", "RED", "CRISIS"],
    ),
    "enhanced_safety_head": HeadInfo(
        name="enhanced_safety_head",
        capability=Capability.SAFETY_FAMILYOS,
        output_type="classification",
        num_labels=4,
        label_names=["GREEN", "AMBER", "RED", "CRISIS"],
    ),
    "ingress_head": HeadInfo(
        name="ingress_head",
        capability=Capability.INGRESS,
        output_type="classification",
        num_labels=6,
        label_names=["user_message", "system_event", "notification", "command", "query", "other"],
    ),
    "intent_head": HeadInfo(
        name="intent_head",
        capability=Capability.INTENT,
        output_type="classification",
        num_labels=12,
        label_names=["greeting", "farewell", "question", "command", "statement",
                     "request", "confirmation", "denial", "gratitude", "apology",
                     "emotion_share", "other"],
    ),
    "relation_head": HeadInfo(
        name="relation_head",
        capability=Capability.RELATION,
        output_type="classification",
        num_labels=15,
        label_names=["parent_of", "child_of", "sibling_of", "spouse_of", "friend_of",
                     "colleague_of", "lives_with", "works_at", "located_in", "part_of",
                     "owns", "created_by", "happened_at", "related_to", "none"],
    ),
    "nli_head": HeadInfo(
        name="nli_head",
        capability=Capability.NLI,
        output_type="classification",
        num_labels=3,
        label_names=["entailment", "neutral", "contradiction"],
    ),
    "embedding_head": HeadInfo(
        name="embedding_head",
        capability=Capability.EMBEDDING,
        output_type="embedding",
        num_labels=None,
    ),
}


# =============================================================================
# Registry Functions
# =============================================================================

def resolve_capability(capability: str | Capability) -> tuple[str, str]:
    """
    Resolve a capability to its model name and head name.

    Args:
        capability: The capability name (string or Capability enum).

    Returns:
        Tuple of (model_name, head_name).

    Raises:
        ValueError: If capability is not recognized.

    Example:
        >>> model_name, head_name = resolve_capability("ner_family")
        >>> assert model_name == "familyos_unified_v2"
        >>> assert head_name == "ner_family_head"
    """
    # Normalize capability
    if isinstance(capability, str):
        capability_lower = capability.lower()

        # Check aliases first
        if capability_lower in CAPABILITY_ALIASES:
            cap = CAPABILITY_ALIASES[capability_lower]
        else:
            # Try direct enum lookup
            try:
                cap = Capability(capability_lower)
            except ValueError:
                # Check if it matches any enum name
                for c in Capability:
                    if c.name.lower() == capability_lower:
                        cap = c
                        break
                else:
                    raise ValueError(
                        f"Unknown capability: {capability}. "
                        f"Valid capabilities: {[c.value for c in Capability]}"
                    )
    else:
        cap = capability

    # Find model that supports this capability
    for model_name, model_info in MODEL_REGISTRY.items():
        if cap in model_info.capabilities:
            head_name = model_info.capability_to_head.get(cap)
            if head_name:
                return model_name, head_name

    raise ValueError(f"No model registered for capability: {cap.value}")


def get_model_info(model_name: str) -> ModelInfo:
    """
    Get information about a registered model.

    Args:
        model_name: Name of the model.

    Returns:
        ModelInfo dataclass with model details.

    Raises:
        KeyError: If model is not registered.
    """
    # Check for legacy model names
    if model_name in LEGACY_MODEL_MAPPING:
        logger.warning(
            f"Model '{model_name}' is deprecated. "
            f"Use '{LEGACY_MODEL_MAPPING[model_name]}' instead."
        )
        model_name = LEGACY_MODEL_MAPPING[model_name]

    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name]


def get_head_info(head_name: str) -> HeadInfo:
    """
    Get information about a registered head.

    Args:
        head_name: Name of the head.

    Returns:
        HeadInfo dataclass with head details.

    Raises:
        KeyError: If head is not registered.
    """
    if head_name not in HEAD_REGISTRY:
        raise KeyError(
            f"Head '{head_name}' not found in registry. "
            f"Available heads: {list(HEAD_REGISTRY.keys())}"
        )

    return HEAD_REGISTRY[head_name]


def list_capabilities() -> list[str]:
    """List all available capabilities."""
    return [cap.value for cap in Capability]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())


def list_heads() -> list[str]:
    """List all registered head names."""
    return list(HEAD_REGISTRY.keys())


# =============================================================================
# Model Loading
# =============================================================================

# Global model cache
_model_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}


def get_unified_model(
    model_name: str = "familyos_unified_v2",
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    """
    Get the unified model instance.

    Args:
        model_name: Name of the model to load.
        device: Device to load model on. Defaults to CUDA if available.
        use_cache: Whether to cache the model instance.

    Returns:
        The loaded model instance.

    Example:
        >>> model = get_unified_model()
        >>> output = model(input_ids, attention_mask)
    """
    import torch

    if use_cache and model_name in _model_cache:
        logger.debug(f"Returning cached model: {model_name}")
        return _model_cache[model_name]

    # Get model info
    model_info = get_model_info(model_name)

    # Determine device
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    # Import and load model
    try:
        from modeling_studio.models.modernbert_multitask import \
            ModernBertMultiTaskModel

        model = ModernBertMultiTaskModel.from_pretrained(
            model_info.base_model,
            trust_remote_code=True,
        )
        model = model.to(torch_device)
        model.eval()

        logger.info(f"Loaded model '{model_name}' on {torch_device}")

        if use_cache:
            _model_cache[model_name] = model

        return model

    except ImportError as e:
        logger.error(f"Failed to import model: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise


def get_tokenizer(
    model_name: str = "familyos_unified_v2",
    use_cache: bool = True,
) -> Any:
    """
    Get the tokenizer for a model.

    Args:
        model_name: Name of the model.
        use_cache: Whether to cache the tokenizer instance.

    Returns:
        The tokenizer instance.
    """
    if use_cache and model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    model_info = get_model_info(model_name)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_info.base_model,
            trust_remote_code=True,
        )

        if use_cache:
            _tokenizer_cache[model_name] = tokenizer

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{model_name}': {e}")
        raise


def clear_cache() -> None:
    """Clear the model and tokenizer cache."""
    global _model_cache, _tokenizer_cache
    _model_cache.clear()
    _tokenizer_cache.clear()
    logger.info("Cleared model cache")


# =============================================================================
# K0 Module Integration Helpers
# =============================================================================

def migrate_legacy_model(legacy_model_name: str) -> str:
    """
    Get the unified model name for a legacy model.

    Args:
        legacy_model_name: Name of the legacy model.

    Returns:
        Name of the unified model to use instead.

    Raises:
        KeyError: If no migration path exists.
    """
    if legacy_model_name in LEGACY_MODEL_MAPPING:
        unified_name = LEGACY_MODEL_MAPPING[legacy_model_name]
        logger.info(f"Migrating '{legacy_model_name}' -> '{unified_name}'")
        return unified_name

    raise KeyError(
        f"No migration path for legacy model '{legacy_model_name}'. "
        f"Known legacy models: {list(LEGACY_MODEL_MAPPING.keys())}"
    )


def get_capability_for_module(module_name: str) -> Capability:
    """
    Map a K0 module name to its primary capability.

    Args:
        module_name: K0 module identifier (e.g., "M02", "M04", "M10", "P08").

    Returns:
        The primary capability for that module.

    Raises:
        KeyError: If module is not recognized.
    """
    module_mapping: dict[str, Capability] = {
        # M02: hippocampus.semantic_project
        "M02": Capability.NER_FAMILY,
        "hippocampus": Capability.NER_FAMILY,
        "semantic_project": Capability.NER_FAMILY,

        # M04: affect.analyze
        "M04": Capability.EMOTIONS,
        "affect": Capability.EMOTIONS,
        "emotion_analyzer": Capability.EMOTIONS,

        # M10: context.ingress_classify
        "M10": Capability.INGRESS,
        "ingress": Capability.INGRESS,
        "context_classifier": Capability.INGRESS,

        # P08: embedding pipeline
        "P08": Capability.EMBEDDING,
        "embedding": Capability.EMBEDDING,
        "embedder": Capability.EMBEDDING,

        # Safety modules
        "safety": Capability.SAFETY_FAMILYOS,
        "content_moderation": Capability.SAFETY_FAMILYOS,
    }

    if module_name not in module_mapping:
        raise KeyError(
            f"Unknown module '{module_name}'. "
            f"Known modules: {list(module_mapping.keys())}"
        )

    return module_mapping[module_name]


# =============================================================================
# Registration Functions (for extension)
# =============================================================================

def register_model(model_info: ModelInfo) -> None:
    """
    Register a new model in the registry.

    Args:
        model_info: ModelInfo dataclass with model details.
    """
    if model_info.name in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model registration: {model_info.name}")

    MODEL_REGISTRY[model_info.name] = model_info
    logger.info(f"Registered model: {model_info.name}")


def register_head(head_info: HeadInfo) -> None:
    """
    Register a new head in the registry.

    Args:
        head_info: HeadInfo dataclass with head details.
    """
    if head_info.name in HEAD_REGISTRY:
        logger.warning(f"Overwriting existing head registration: {head_info.name}")

    HEAD_REGISTRY[head_info.name] = head_info
    logger.info(f"Registered head: {head_info.name}")


def register_capability_alias(alias: str, capability: Capability) -> None:
    """
    Register an alias for a capability.

    Args:
        alias: The alias string.
        capability: The Capability enum value.
    """
    CAPABILITY_ALIASES[alias.lower()] = capability
    logger.debug(f"Registered capability alias: {alias} -> {capability.value}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "Capability",

    # Dataclasses
    "ModelInfo",
    "HeadInfo",

    # Registries
    "MODEL_REGISTRY",
    "HEAD_REGISTRY",
    "CAPABILITY_ALIASES",
    "LEGACY_MODEL_MAPPING",

    # Core functions
    "resolve_capability",
    "get_model_info",
    "get_head_info",
    "list_capabilities",
    "list_models",
    "list_heads",

    # Model loading
    "get_unified_model",
    "get_tokenizer",
    "clear_cache",

    # Migration helpers
    "migrate_legacy_model",
    "get_capability_for_module",

    # Registration
    "register_model",
    "register_head",
    "register_capability_alias",
]
]
]
]
]

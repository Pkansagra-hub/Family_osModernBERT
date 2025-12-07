"""
Hub Token Registry for ModernBERT v3.3 Ultra.

This module defines the hub token system that enables capability-specific routing:
- [EMO]: Affective understanding (emotions, sentiment, safety)
- [MEM]: Memory retrieval & storage (embedding)
- [REL]: Relational reasoning (NLI, relations)
- [TASK]: User action classification (intent, ingress)

Hub tokens are prepended after [CLS] and use global bidirectional attention
to aggregate information across the entire sequence.
"""

from dataclasses import dataclass
from enum import Enum


class HubToken(Enum):
    """Hub token identifiers."""

    CLS = "[CLS]"
    EMO = "[EMO]"
    MEM = "[MEM]"
    REL = "[REL]"
    TASK = "[TASK]"


@dataclass
class HubTokenSpec:
    """Specification for a hub token."""

    token: str
    position: int
    capabilities: list[str]
    semantic_seeds: list[str]
    description: str


# Hub Token Registry
# Maps hub tokens to their specifications including capabilities and initialization seeds
HUB_TOKEN_REGISTRY: dict[str, HubTokenSpec] = {
    "[EMO]": HubTokenSpec(
        token="[EMO]",
        position=1,
        capabilities=["emotions", "sentiment", "safety_generic", "safety_familyos"],
        semantic_seeds=["happy", "sad", "angry", "fear", "joy", "anxious", "love", "feeling"],
        description="Affective understanding - routes to emotion/sentiment/safety heads",
    ),
    "[MEM]": HubTokenSpec(
        token="[MEM]",
        position=2,
        capabilities=["embedding"],
        semantic_seeds=["remember", "memory", "past", "history", "recall", "yesterday"],
        description="Memory retrieval & storage - routes to embedding head",
    ),
    "[REL]": HubTokenSpec(
        token="[REL]",
        position=3,
        capabilities=["nli", "relation"],
        semantic_seeds=["family", "mother", "father", "sister", "brother", "parent", "child"],
        description="Entity & logical relationships - routes to NLI/relation heads",
    ),
    "[TASK]": HubTokenSpec(
        token="[TASK]",
        position=4,
        capabilities=["intent", "ingress"],
        semantic_seeds=["action", "do", "want", "need", "help", "schedule", "plan"],
        description="User action classification - routes to intent/ingress heads",
    ),
}

# Capabilities that use token-level classification (not hub routing)
# These capabilities operate on per-token representations instead of aggregated hub tokens
TOKEN_LEVEL_CAPABILITIES: set[str] = {"ner_general", "ner_family", "temporal"}

# Hub token IDs (reserved token IDs in vocabulary)
# ModernBERT-base has vocab_size=50368, so hub tokens start at 50368
# These are added when extending the v2 tokenizer vocabulary for v3
HUB_TOKEN_IDS: dict[str, int] = {
    "[EMO]": 50368,
    "[MEM]": 50369,
    "[REL]": 50370,
    "[TASK]": 50371,
}


def get_hub_for_capability(capability: str) -> str:
    """
    Get the hub token that routes to a given capability.

    Args:
        capability: Name of the capability (e.g., "emotions", "intent", "ner_family")

    Returns:
        Hub token string (e.g., "[EMO]") or "[CLS]" for token-level tasks

    Examples:
        >>> get_hub_for_capability("emotions")
        '[EMO]'
        >>> get_hub_for_capability("intent")
        '[TASK]'
        >>> get_hub_for_capability("ner_family")
        '[CLS]'
        >>> get_hub_for_capability("embedding")
        '[MEM]'
    """
    # Token-level tasks use CLS or per-token representations
    if capability in TOKEN_LEVEL_CAPABILITIES:
        return "[CLS]"

    # Search for capability in hub token registry
    for hub_token, spec in HUB_TOKEN_REGISTRY.items():
        if capability in spec.capabilities:
            return hub_token

    # Fallback to CLS for unknown capabilities
    return "[CLS]"


def get_capabilities_for_hub(hub_token: str) -> list[str]:
    """
    Get all capabilities routed through a hub token.

    Args:
        hub_token: Hub token string (e.g., "[EMO]", "[MEM]")

    Returns:
        List of capability names routed through this hub

    Examples:
        >>> get_capabilities_for_hub("[EMO]")
        ['emotions', 'sentiment', 'safety_generic', 'safety_familyos']
        >>> get_capabilities_for_hub("[TASK]")
        ['intent', 'ingress']
    """
    if hub_token not in HUB_TOKEN_REGISTRY:
        return []
    return HUB_TOKEN_REGISTRY[hub_token].capabilities


def get_hub_positions() -> dict[str, int]:
    """
    Get position indices for all hub tokens (including CLS).

    Returns:
        Dictionary mapping hub token to position index

    Note:
        Position 0 is reserved for [CLS]
        Positions 1-4 are for hub tokens [EMO], [MEM], [REL], [TASK]

    Examples:
        >>> get_hub_positions()
        {'[CLS]': 0, '[EMO]': 1, '[MEM]': 2, '[REL]': 3, '[TASK]': 4}
    """
    positions = {"[CLS]": 0}
    for token, spec in HUB_TOKEN_REGISTRY.items():
        positions[token] = spec.position
    return positions


def get_global_attention_positions() -> list[int]:
    """
    Get positions that should have global attention (CLS + all hubs).

    Returns:
        List of position indices with global attention

    Note:
        These tokens use bidirectional global attention (à la BigBird/Longformer):
        - Hub tokens attend to ALL tokens in the sequence
        - ALL tokens attend to hub tokens
        - Regular text tokens use sliding windows

    Examples:
        >>> get_global_attention_positions()
        [0, 1, 2, 3, 4]
    """
    return [0, 1, 2, 3, 4]  # [CLS], [EMO], [MEM], [REL], [TASK]


def get_semantic_seeds(hub_token: str) -> list[str]:
    """
    Get semantic seed words for hub token initialization.

    These words are used to initialize hub token embeddings as semantic centroids,
    placing them in the correct "neighborhood" of the embedding space.

    Args:
        hub_token: Hub token string (e.g., "[EMO]", "[MEM]")

    Returns:
        List of seed words for semantic centroid initialization

    Examples:
        >>> get_semantic_seeds("[EMO]")
        ['happy', 'sad', 'angry', 'fear', 'joy', 'anxious', 'love', 'feeling']
        >>> get_semantic_seeds("[REL]")
        ['family', 'mother', 'father', 'sister', 'brother', 'parent', 'child']
    """
    if hub_token not in HUB_TOKEN_REGISTRY:
        return []
    return HUB_TOKEN_REGISTRY[hub_token].semantic_seeds


def get_hub_token_id(hub_token: str) -> int:
    """
    Get the reserved token ID for a hub token.

    Args:
        hub_token: Hub token string (e.g., "[EMO]", "[MEM]")

    Returns:
        Token ID in the extended vocabulary

    Raises:
        KeyError: If hub token is not in the registry

    Examples:
        >>> get_hub_token_id("[EMO]")
        50265
        >>> get_hub_token_id("[MEM]")
        50266
    """
    if hub_token not in HUB_TOKEN_IDS:
        raise KeyError(f"Hub token {hub_token} not found in registry")
    return HUB_TOKEN_IDS[hub_token]


def get_all_hub_tokens() -> list[str]:
    """
    Get list of all hub token strings.

    Returns:
        List of hub token strings (excluding [CLS])

    Examples:
        >>> get_all_hub_tokens()
        ['[EMO]', '[MEM]', '[REL]', '[TASK]']
    """
    return list(HUB_TOKEN_REGISTRY.keys())


def print_hub_token_registry():
    """Print a human-readable view of the hub token registry."""
    print("=" * 80)
    print("ModernBERT v3.3 Ultra - Hub Token Registry")
    print("=" * 80)
    print("\nHub Tokens (Global Bidirectional Attention):")
    print("-" * 80)

    for hub_token, spec in HUB_TOKEN_REGISTRY.items():
        print(f"\n{hub_token} (Position {spec.position}, Token ID: {HUB_TOKEN_IDS[hub_token]})")
        print(f"  Description: {spec.description}")
        print(f"  Capabilities: {', '.join(spec.capabilities)}")
        print(f"  Semantic Seeds: {', '.join(spec.semantic_seeds)}")

    print("\n" + "-" * 80)
    print("\nToken-Level Capabilities (No Hub Routing):")
    print(f"  {', '.join(TOKEN_LEVEL_CAPABILITIES)}")
    print("  These use per-token representations instead of hub tokens")

    print("\n" + "=" * 80)
    print("Capability → Hub Token Mapping:")
    print("=" * 80)

    # Group capabilities by hub
    all_capabilities = set()
    for spec in HUB_TOKEN_REGISTRY.values():
        all_capabilities.update(spec.capabilities)
    all_capabilities.update(TOKEN_LEVEL_CAPABILITIES)

    for capability in sorted(all_capabilities):
        hub = get_hub_for_capability(capability)
        print(f"  {capability:20} → {hub}")

    print("=" * 80)


# Complete capability-to-hub routing mapping
# This maps all 12 FamilyOS capabilities to their hub tokens
# Note: This is derived from HUB_TOKEN_REGISTRY and TOKEN_LEVEL_CAPABILITIES
# For dynamic routing logic, use get_hub_for_capability() or routing_v3.HubRouter
CAPABILITY_HUB_ROUTING: dict[str, str | None] = {
    # [EMO] Hub - Affective understanding (4 capabilities)
    "emotions": "[EMO]",
    "sentiment": "[EMO]",
    "safety_generic": "[EMO]",
    "safety_familyos": "[EMO]",
    # [MEM] Hub - Memory retrieval (1 capability)
    "embedding": "[MEM]",
    # [REL] Hub - Relational reasoning (2 capabilities)
    "nli": "[REL]",
    "relation": "[REL]",
    # [TASK] Hub - User action classification (2 capabilities)
    "intent": "[TASK]",
    "ingress": "[TASK]",
    # Token-level (no hub routing) (3 capabilities)
    "ner_general": None,
    "ner_family": None,
    "temporal": None,
}

"""
Hub-to-Capability Routing for ModernBERT v3.3 Ultra.

This module implements the routing logic that directs hub token representations
to appropriate capability heads. It determines:
1. Which hub token provides the representation for each capability
2. Whether to use hub pooling (sequence-level) or per-token representations

Classes:
    - HubRouter: Routes hub token representations to capability heads
    - CapabilityHead: Wrapper for capability heads with automatic routing
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .hub_tokens import (
    HUB_TOKEN_REGISTRY,
    get_capabilities_for_hub,
)

logger = logging.getLogger(__name__)


class HubRouter(nn.Module):
    """
    Routes hub token representations to capability heads.

    For each capability, determines:
    1. Which hub token provides the representation
    2. Whether to use hub pooling or per-token representations

    The routing table maps capabilities to (pool_type, hub_token) tuples:
    - pool_type: "hub" for sequence-level, "token" for token-level
    - hub_token: The hub token providing the representation (None for token-level)

    Example:
        >>> router = HubRouter()
        >>> # Get representation for emotions capability
        >>> repr, pool_type = router.get_representation_for_capability(
        ...     hidden_states, pooled_outputs, "emotions"
        ... )
        >>> # repr is pooled_outputs["[EMO]"], pool_type is "hub"
    """

    # Routing table: capability -> (pool_type, hub_token)
    ROUTING_TABLE: dict[str, tuple[str, str | None]] = {
        # EMO hub capabilities (sequence-level)
        "emotions": ("hub", "[EMO]"),
        "sentiment": ("hub", "[EMO]"),
        "safety_generic": ("hub", "[EMO]"),
        "safety_familyos": ("hub", "[EMO]"),
        # MEM hub capabilities (sequence-level)
        "embedding": ("hub", "[MEM]"),
        # REL hub capabilities (sequence-level, may use pair encoder)
        "nli": ("hub", "[REL]"),
        "relation": ("hub", "[REL]"),
        # TASK hub capabilities (sequence-level)
        "intent": ("hub", "[TASK]"),
        "ingress": ("hub", "[TASK]"),
        # Token-level capabilities (use per-token representations)
        "ner_general": ("token", None),
        "ner_family": ("token", None),
        "temporal": ("token", None),
    }

    def __init__(self):
        super().__init__()

    def get_representation_for_capability(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: dict[str, torch.Tensor],
        capability: str,
    ) -> tuple[torch.Tensor, str]:
        """
        Get the appropriate representation for a capability.

        Args:
            hidden_states: Full sequence hidden states [batch, seq_len, hidden]
            pooled_outputs: Dict of hub token representations from pooler
            capability: Target capability name

        Returns:
            Tuple of (representation, pool_type)
            - For hub capabilities: ([batch, hidden], "hub")
            - For token capabilities: ([batch, seq_len, hidden], "token")

        Example:
            >>> router = HubRouter()
            >>> hidden = torch.randn(2, 128, 768)
            >>> pooled = {"[EMO]": torch.randn(2, 768), ...}
            >>> repr, pool_type = router.get_representation_for_capability(
            ...     hidden, pooled, "emotions"
            ... )
            >>> repr.shape
            torch.Size([2, 768])
            >>> pool_type
            'hub'
        """
        pool_type, hub_token = self.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))

        if pool_type == "token":
            # Return full sequence for token-level classification
            return hidden_states, "token"
        else:
            # Return hub token representation
            if hub_token not in pooled_outputs:
                raise ValueError(
                    f"Hub token {hub_token} not found in pooled_outputs. "
                    f"Available: {list(pooled_outputs.keys())}"
                )
            return pooled_outputs[hub_token], "hub"

    def get_hub_gradient_mask(
        self,
        active_capabilities: list[str],
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Create gradient masks for hub tokens based on active capabilities.

        Used during training to ensure gradients only flow through relevant hubs.
        This enables efficient multi-task training where only the hubs needed
        for the current batch receive gradient updates.

        Args:
            active_capabilities: List of capabilities being trained this batch
            batch_size: Current batch size
            device: Target device

        Returns:
            Dict mapping hub tokens to gradient masks [batch]
            - 1.0 if hub should be trained (capability is active)
            - 0.0 if hub should be frozen (no active capabilities)

        Example:
            >>> router = HubRouter()
            >>> masks = router.get_hub_gradient_mask(
            ...     ["emotions", "sentiment"], batch_size=2, device="cpu"
            ... )
            >>> masks["[EMO]"]  # Should be trained (emotions/sentiment active)
            tensor([1., 1.])
            >>> masks["[MEM]"]  # Should be frozen (embedding not active)
            tensor([0., 0.])
        """
        masks = {}

        for hub_token in HUB_TOKEN_REGISTRY.keys():
            hub_capabilities = get_capabilities_for_hub(hub_token)

            # Hub should receive gradient if any of its capabilities are active
            should_train = any(cap in active_capabilities for cap in hub_capabilities)

            masks[hub_token] = torch.ones(batch_size, device=device) * float(should_train)

        return masks


class CapabilityHead(nn.Module):
    """
    Wrapper for a capability head that handles hub routing.

    This wrapper automatically routes representations to the underlying head
    based on the capability's routing type (hub or token-level).

    Args:
        capability: Capability name (e.g., "emotions", "ner_general")
        head: The underlying task head module
        hidden_size: Hidden size (default: 768)

    Example:
        >>> from modeling_studio.models.heads import EmotionHead
        >>> emotion_head = EmotionHead(hidden_size=768, num_labels=44)
        >>> wrapped_head = CapabilityHead("emotions", emotion_head, hidden_size=768)
        >>> # Forward pass with automatic routing
        >>> logits = wrapped_head(hidden_states, pooled_outputs)
    """

    def __init__(
        self,
        capability: str,
        head: nn.Module,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.capability = capability
        self.head = head
        self.hidden_size = hidden_size

        pool_type, hub_token = HubRouter.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))
        self.pool_type = pool_type
        self.hub_token = hub_token

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_outputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass with automatic hub routing.

        Args:
            hidden_states: Full sequence [batch, seq_len, hidden]
            pooled_outputs: Hub token representations
            **kwargs: Additional arguments for the head

        Returns:
            Head output logits

        Note:
            Token-level heads receive full sequence hidden states.
            Hub-routed heads receive the appropriate hub token representation.
        """
        if self.pool_type == "token":
            # Token-level head (NER, temporal)
            return self.head(hidden_states, **kwargs)
        else:
            # Hub-routed head
            if self.hub_token not in pooled_outputs:
                raise ValueError(
                    f"Hub token {self.hub_token} not found in pooled_outputs. "
                    f"Available: {list(pooled_outputs.keys())}"
                )
            representation = pooled_outputs[self.hub_token]
            return self.head(representation, **kwargs)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"capability={self.capability}, "
            f"pool_type={self.pool_type}, "
            f"hub_token={self.hub_token}"
        )


def create_hub_routing_info(capability: str) -> dict[str, Any]:
    """
    Get routing information for a capability.

    Returns comprehensive routing information including pool type,
    hub token, and hub description.

    Args:
        capability: Capability name (e.g., "emotions", "ner_general")

    Returns:
        Dict with pool_type, hub_token, and optional hub_description

    Example:
        >>> info = create_hub_routing_info("emotions")
        >>> info["pool_type"]
        'hub'
        >>> info["hub_token"]
        '[EMO]'
        >>> info["hub_description"]
        'Affective understanding - routes to emotion/sentiment/safety heads'
    """
    pool_type, hub_token = HubRouter.ROUTING_TABLE.get(capability, ("hub", "[CLS]"))

    info: dict[str, Any] = {
        "capability": capability,
        "pool_type": pool_type,
        "hub_token": hub_token,
    }

    if hub_token and hub_token in HUB_TOKEN_REGISTRY:
        info["hub_description"] = HUB_TOKEN_REGISTRY[hub_token].description

    return info


def print_routing_table() -> None:
    """Print a human-readable view of the routing table."""
    print("=" * 80)
    print("ModernBERT v3.3 Ultra - Hub Routing Table")
    print("=" * 80)
    print()

    # Group by hub token
    hub_groups: dict[str | None, list[str]] = {}
    for capability, (pool_type, hub_token) in HubRouter.ROUTING_TABLE.items():
        if pool_type == "token":
            hub_groups.setdefault(None, []).append(capability)
        else:
            hub_groups.setdefault(hub_token, []).append(capability)

    # Print hub-routed capabilities
    print("Hub-Routed Capabilities (Sequence-Level):")
    print("-" * 80)
    for hub_token in ["[EMO]", "[MEM]", "[REL]", "[TASK]"]:
        if hub_token in hub_groups:
            caps = hub_groups[hub_token]
            desc = HUB_TOKEN_REGISTRY[hub_token].description
            print(f"\n{hub_token} ({len(caps)} capabilities)")
            print(f"  Description: {desc}")
            print(f"  Capabilities: {', '.join(caps)}")

    # Print token-level capabilities
    print("\n" + "-" * 80)
    print("\nToken-Level Capabilities (Per-Token):")
    if None in hub_groups:
        print(f"  {', '.join(hub_groups[None])}")
        print("  These use full sequence hidden states instead of hub tokens")

    print("\n" + "=" * 80)

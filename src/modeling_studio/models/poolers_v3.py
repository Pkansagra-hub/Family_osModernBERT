"""
Hub Token Pooling for ModernBERT v3.3 Ultra.

This module implements poolers that extract hub token representations
for routing to capability-specific heads.

Hub tokens ([EMO], [MEM], [REL], [TASK]) are prepended after [CLS]
and aggregate information relevant to their assigned capabilities.

Classes:
    - HubTokenPooler: Extracts hub token representations from final hidden states
    - CombinedPooler: Provides CLS, Mean, and Hub token pooling
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .hub_tokens import HUB_TOKEN_REGISTRY, get_hub_for_capability, get_hub_positions

logger = logging.getLogger(__name__)


class HubTokenPooler(nn.Module):
    """
    Extracts hub token representations from the final hidden states.

    Given sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
    Returns dict of hub token representations for routing to heads.

    The pooler can optionally apply a projection layer (like BERT's pooler)
    to each hub token representation.

    Args:
        hidden_size: Size of encoder hidden states (default: 768)
        add_projection: Whether to add projection layers for hub tokens (default: False)

    Example:
        >>> pooler = HubTokenPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)  # batch=2, seq_len=128
        >>> pooled = pooler(hidden_states)
        >>> pooled["[EMO]"].shape
        torch.Size([2, 768])
    """

    def __init__(self, hidden_size: int = 768, add_projection: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_positions = get_hub_positions()

        # Optional projection layer (like BERT's pooler)
        self.add_projection = add_projection
        if add_projection:
            self.projections = nn.ModuleDict(
                {
                    token.replace("[", "").replace("]", ""): nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(),
                    )
                    for token in HUB_TOKEN_REGISTRY.keys()
                }
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract hub token representations.

        Args:
            hidden_states: Final layer output [batch, seq_len, hidden]
            attention_mask: Optional attention mask [batch, seq_len] (not used currently)

        Returns:
            Dict mapping hub token names to their representations [batch, hidden]

        Note:
            The attention_mask parameter is currently unused but kept for API consistency.
            It may be used in future versions for advanced pooling strategies.
        """
        pooled = {}

        # Extract each hub token
        for token, position in self.hub_positions.items():
            # Get representation at hub position
            hub_repr = hidden_states[:, position, :]  # [batch, hidden]

            # Apply projection if enabled
            if self.add_projection and token in HUB_TOKEN_REGISTRY:
                key = token.replace("[", "").replace("]", "")
                hub_repr = self.projections[key](hub_repr)

            pooled[token] = hub_repr

        return pooled

    def get_pooled_for_capability(
        self,
        hidden_states: torch.Tensor,
        capability: str,
    ) -> torch.Tensor:
        """
        Get the pooled representation for a specific capability.

        Args:
            hidden_states: Final layer output [batch, seq_len, hidden]
            capability: Capability name (e.g., "emotions", "intent")

        Returns:
            Pooled representation [batch, hidden] from the appropriate hub

        Example:
            >>> pooler = HubTokenPooler(hidden_size=768)
            >>> hidden_states = torch.randn(2, 128, 768)
            >>> emotions_repr = pooler.get_pooled_for_capability(hidden_states, "emotions")
            >>> emotions_repr.shape
            torch.Size([2, 768])
        """
        hub_token = get_hub_for_capability(capability)
        position = self.hub_positions[hub_token]

        return hidden_states[:, position, :]


class CombinedPooler(nn.Module):
    """
    Combined pooler that provides CLS, Mean, and Hub token pooling.

    This pooler extracts multiple pooled representations in a single forward pass:
    - [CLS] token representation (with projection)
    - Hub token representations ([EMO], [MEM], [REL], [TASK])
    - Mean-pooled representation (excluding special tokens at positions 0-4)

    The mean pooling properly masks out CLS and hub tokens to ensure only
    text tokens contribute to the averaged representation.

    Args:
        hidden_size: Size of encoder hidden states (default: 768)

    Example:
        >>> pooler = CombinedPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> pooled.keys()
        dict_keys(['[CLS]', '[EMO]', '[MEM]', '[REL]', '[TASK]', '[CLS]_projected', 'mean'])
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_pooler = HubTokenPooler(hidden_size)

        # CLS projection (standard BERT-style)
        self.cls_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get all pooled representations.

        Returns:
            Dict with:
                - "[CLS]": CLS token representation (raw)
                - "[CLS]_projected": CLS token representation (projected + tanh)
                - "[EMO]", "[MEM]", "[REL]", "[TASK]": Hub representations
                - "mean": Mean-pooled representation (excluding special tokens)

        Args:
            hidden_states: Final layer output [batch, seq_len, hidden]
            attention_mask: Optional attention mask [batch, seq_len]

        Note:
            Mean pooling excludes positions 0-4 ([CLS] and 4 hub tokens).
            This ensures the mean representation only aggregates actual text tokens.
        """
        # Hub token pooling (includes [CLS])
        pooled = self.hub_pooler(hidden_states, attention_mask)

        # CLS projection
        pooled["[CLS]_projected"] = self.cls_projection(pooled["[CLS]"])

        # Mean pooling (exclude special tokens at positions 0-4)
        if attention_mask is not None:
            # Mask out CLS and hub tokens from mean pooling
            mean_mask = attention_mask.clone()
            mean_mask[:, :5] = 0  # Zero out [CLS] and 4 hub positions

            # Expand mask for broadcasting
            mask_expanded = mean_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled["mean"] = sum_hidden / sum_mask
        else:
            # Simple mean over text positions (5 onwards)
            pooled["mean"] = hidden_states[:, 5:, :].mean(dim=1)

        return pooled

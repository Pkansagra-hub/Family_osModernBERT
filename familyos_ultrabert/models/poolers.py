"""
Pooling Strategies for Encoder Outputs

This module contains various pooling strategies to convert
token-level encoder outputs into fixed-size representations.

Pooling Methods:
    - BasePooler: Abstract base class for all poolers
    - CLSPooler: Use [CLS] token representation
    - MeanPooler: Average all token representations (masked)
    - MaxPooler: Max pooling over tokens (masked)
    - WeightedMeanPooler: Attention-weighted mean pooling
    - LastTokenPooler: Use last non-padding token (for causal models)

Each pooler handles attention masks properly to ignore padding tokens.

Usage:
    from familyos_ultrabert.models.poolers import CLSPooler, MeanPooler, MaxPooler

    hidden_states = torch.randn(2, 128, 768)
    attention_mask = torch.ones(2, 128)

    pooler = MeanPooler(hidden_size=768)
    sentence_embedding = pooler(hidden_states, attention_mask)
    # -> (2, 768)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# BasePooler - Abstract Base Class
# =============================================================================


class BasePooler(ABC, nn.Module):
    """
    Abstract base class for all pooling strategies.

    All poolers convert variable-length token sequences into fixed-size
    representations suitable for downstream tasks.

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size

    Subclasses must implement:
        forward(hidden_states, attention_mask) -> pooled_output
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool token representations into fixed-size output.

        Args:
            hidden_states: Encoder outputs (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len).
                1 = valid token, 0 = padding

        Returns:
            Pooled representation (batch_size, output_size)
        """
        pass

    def _expand_mask(
        self,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand attention mask to match hidden states dimensions.

        Args:
            attention_mask: (batch_size, seq_len) or None
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            Expanded mask (batch_size, seq_len, 1) for broadcasting
        """
        if attention_mask is None:
            return torch.ones(
                hidden_states.size(0),
                hidden_states.size(1),
                1,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        return attention_mask.unsqueeze(-1).to(hidden_states.dtype)


# =============================================================================
# CLSPooler - [CLS] Token Representation
# =============================================================================


class CLSPooler(BasePooler):
    """
    Extract [CLS] token representation as sequence embedding.

    The [CLS] token (typically position 0) is trained to aggregate
    sequence-level information during pre-training.

    Optionally passes through a dense layer with activation.

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size
        use_dense: Whether to apply dense + tanh after CLS extraction
        cls_position: Position of CLS token. Default: 0

    Example:
        >>> pooler = CLSPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> pooled = pooler(hidden_states)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        use_dense: bool = True,
        cls_position: int = 0,
    ):
        super().__init__(hidden_size, output_size)
        self.cls_position = cls_position
        self.use_dense = use_dense

        if use_dense:
            self.dense = nn.Linear(hidden_size, self.output_size)
            self.activation = nn.Tanh()
        else:
            self.dense = None
            self.activation = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract and optionally transform [CLS] token.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Not used for CLS pooling, included for API consistency

        Returns:
            Pooled output (batch_size, output_size)
        """
        # Extract CLS token
        cls_output = hidden_states[:, self.cls_position, :]  # (batch, hidden_size)

        # Optional dense transformation
        if self.use_dense and self.dense is not None:
            pooled_output = self.dense(cls_output)
            pooled_output = self.activation(pooled_output)  # type: ignore
        else:
            pooled_output = cls_output

        return pooled_output


# =============================================================================
# MeanPooler - Masked Mean Pooling
# =============================================================================


class MeanPooler(BasePooler):
    """
    Mean pooling over token representations with attention masking.

    Computes the average of all non-padding token representations.
    This is a simple but effective pooling strategy, especially for
    sentence embeddings.

    Formula:
        pooled = sum(hidden_states * mask) / sum(mask)

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size
        use_projection: Whether to project after pooling

    Reference:
        Reimers & Gurevych. "Sentence-BERT" (EMNLP 2019)

    Example:
        >>> pooler = MeanPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        use_projection: bool = False,
    ):
        super().__init__(hidden_size, output_size)
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Linear(hidden_size, self.output_size)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute masked mean of token representations.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len). 1 = valid, 0 = padding

        Returns:
            Pooled output (batch_size, output_size)
        """
        # Expand mask for broadcasting
        mask = self._expand_mask(attention_mask, hidden_states)

        # Masked sum
        sum_hidden = (hidden_states * mask).sum(dim=1)  # (batch, hidden_size)

        # Count valid tokens (avoid division by zero)
        num_tokens = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)

        # Mean
        pooled_output = sum_hidden / num_tokens

        # Optional projection
        if self.use_projection and self.projection is not None:
            pooled_output = self.projection(pooled_output)

        return pooled_output


# =============================================================================
# MaxPooler - Masked Max Pooling
# =============================================================================


class MaxPooler(BasePooler):
    """
    Max pooling over token representations with attention masking.

    Takes the element-wise maximum across all non-padding tokens.
    Good for capturing the most salient features in a sequence.

    Formula:
        pooled = max(hidden_states where mask == 1)

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size
        use_projection: Whether to project after pooling

    Example:
        >>> pooler = MaxPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        use_projection: bool = False,
    ):
        super().__init__(hidden_size, output_size)
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Linear(hidden_size, self.output_size)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute masked max of token representations.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len). 1 = valid, 0 = padding

        Returns:
            Pooled output (batch_size, output_size)
        """
        # Expand mask for broadcasting
        mask = self._expand_mask(attention_mask, hidden_states)

        # Set padding positions to large negative value before max
        # This ensures they won't be selected
        masked_hidden = hidden_states.clone()
        masked_hidden[mask.squeeze(-1) == 0] = float("-inf")

        # Max pooling
        pooled_output, _ = masked_hidden.max(dim=1)  # (batch, hidden_size)

        # Handle edge case where all tokens are masked
        # Replace -inf with zeros
        pooled_output = torch.where(
            torch.isinf(pooled_output),
            torch.zeros_like(pooled_output),
            pooled_output,
        )

        # Optional projection
        if self.use_projection and self.projection is not None:
            pooled_output = self.projection(pooled_output)

        return pooled_output


# =============================================================================
# WeightedMeanPooler - Attention-Weighted Mean Pooling
# =============================================================================


class WeightedMeanPooler(BasePooler):
    """
    Attention-weighted mean pooling over token representations.

    Learns attention weights for each token and computes a weighted
    average. This allows the model to focus on more important tokens.

    Architecture:
        hidden_states -> attention_dense -> softmax -> weighted_sum

    Formula:
        weights = softmax(W @ hidden_states + b)
        pooled = sum(weights * hidden_states)

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size
        num_attention_heads: Number of attention heads. Default: 1

    Reference:
        Lin et al. "A Structured Self-Attentive Sentence Embedding" (ICLR 2017)

    Example:
        >>> pooler = WeightedMeanPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        num_attention_heads: int = 1,
    ):
        super().__init__(hidden_size, output_size)
        self.num_attention_heads = num_attention_heads

        # Attention weight computation
        # Projects hidden_size -> num_attention_heads
        self.attention_dense = nn.Linear(hidden_size, num_attention_heads, bias=True)

        # Optional projection if output_size differs
        if self.output_size != hidden_size:
            self.output_projection = nn.Linear(
                hidden_size * num_attention_heads,
                self.output_size,
            )
        else:
            self.output_projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute attention-weighted mean of token representations.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len). 1 = valid, 0 = padding

        Returns:
            Pooled output (batch_size, output_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute attention scores
        # (batch, seq_len, num_heads)
        attention_scores = self.attention_dense(hidden_states)

        # Apply mask (set padding to -inf before softmax)
        if attention_mask is not None:
            # Expand mask: (batch, seq_len) -> (batch, seq_len, 1)
            mask = attention_mask.unsqueeze(-1).to(attention_scores.dtype)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over sequence dimension
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, num_heads)

        # Handle all-masked sequences (replace NaN with uniform)
        if attention_mask is not None:
            all_masked = attention_mask.sum(dim=1, keepdim=True) == 0  # (batch, 1)
            if all_masked.any():
                uniform = torch.ones_like(attention_weights) / seq_len
                attention_weights = torch.where(
                    all_masked.unsqueeze(-1),
                    uniform,
                    attention_weights,
                )

        # Weighted sum
        # (batch, seq_len, num_heads) x (batch, seq_len, hidden_size)
        # -> (batch, hidden_size, num_heads) via einsum
        if self.num_attention_heads == 1:
            pooled_output = (attention_weights * hidden_states.unsqueeze(-1)).sum(dim=1)
            pooled_output = pooled_output.squeeze(-1)  # (batch, hidden_size)
        else:
            # Multi-head: concatenate outputs
            pooled_output = torch.einsum(
                "bsh,bsd->bhd",
                attention_weights,
                hidden_states,
            )  # (batch, num_heads, hidden_size)
            pooled_output = pooled_output.view(batch_size, -1)  # (batch, num_heads * hidden_size)

        # Optional projection
        if self.output_projection is not None:
            pooled_output = self.output_projection(pooled_output)

        return pooled_output


# =============================================================================
# LastTokenPooler - Last Non-Padding Token
# =============================================================================


class LastTokenPooler(BasePooler):
    """
    Extract the last non-padding token representation.

    Useful for causal/decoder models where the last token aggregates
    information from the entire sequence (e.g., GPT-style models).

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output. Default: same as hidden_size
        use_dense: Whether to apply dense + activation after extraction

    Example:
        >>> pooler = LastTokenPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> # Sequence 1: length 100, Sequence 2: length 50
        >>> attention_mask = torch.zeros(2, 128)
        >>> attention_mask[0, :100] = 1
        >>> attention_mask[1, :50] = 1
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        use_dense: bool = False,
    ):
        super().__init__(hidden_size, output_size)
        self.use_dense = use_dense

        if use_dense:
            self.dense = nn.Linear(hidden_size, self.output_size)
            self.activation = nn.Tanh()
        else:
            self.dense = None
            self.activation = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract last non-padding token representation.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len). 1 = valid, 0 = padding

        Returns:
            Pooled output (batch_size, output_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if attention_mask is None:
            # No mask: use last token
            last_indices = torch.full(
                (batch_size,),
                seq_len - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            # Find last valid token index for each sequence
            # Sum attention mask to get sequence lengths, then subtract 1
            seq_lengths = attention_mask.sum(dim=1).long()  # (batch,)
            last_indices = (seq_lengths - 1).clamp(min=0)  # (batch,)

        # Gather last token for each batch element
        # Create indices for gather: (batch, 1, hidden_size)
        gather_indices = last_indices.view(batch_size, 1, 1).expand(-1, -1, hidden_size)
        last_token = hidden_states.gather(dim=1, index=gather_indices).squeeze(
            1
        )  # (batch, hidden_size)

        # Optional dense transformation
        if self.use_dense and self.dense is not None:
            pooled_output = self.dense(last_token)
            pooled_output = self.activation(pooled_output)  # type: ignore
        else:
            pooled_output = last_token

        return pooled_output


# =============================================================================
# CLSMeanPooler - Combined CLS + Mean Pooling
# =============================================================================


class CLSMeanPooler(BasePooler):
    """
    Combined CLS and Mean pooling.

    Concatenates the [CLS] token representation with the mean-pooled
    representation, then projects back to hidden_size. This combines:
    - CLS: Trained sequence-level representation
    - Mean: Distributional information across all tokens

    Architecture:
        concat([CLS], mean_pool) -> Linear(2*hidden_size, hidden_size) -> LayerNorm

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of output (default: hidden_size)
        dropout: Dropout probability (default: 0.1)
        use_layer_norm: Apply layer norm to output (default: True)

    Example:
        >>> pooler = CLSMeanPooler(hidden_size=768)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__(hidden_size, output_size)

        # Projection from concatenated (2*hidden) to output_size
        self.projection = nn.Linear(hidden_size * 2, self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size)
        else:
            self.layer_norm = None

        # Initialize projection weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Combine CLS token and mean pooling.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            Pooled output (batch_size, output_size)
        """
        # Extract CLS token
        cls_token = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        # Compute masked mean pooling
        if attention_mask is None:
            mean_pooled = hidden_states.mean(dim=1)
        else:
            mask = self._expand_mask(attention_mask, hidden_states)
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_hidden / sum_mask

        # Concatenate CLS and mean
        combined = torch.cat([cls_token, mean_pooled], dim=-1)  # (batch, 2*hidden)

        # Project back to output_size
        output = self.projection(combined)  # (batch, output_size)
        output = self.dropout(output)

        if self.use_layer_norm and self.layer_norm is not None:
            output = self.layer_norm(output)

        return output


# =============================================================================
# AttentionPooler - Multi-Head Attention Pooling
# =============================================================================


class AttentionPooler(BasePooler):
    """
    Multi-head attention pooler with learnable query.

    Uses a learnable [POOL] query token that attends to the sequence
    via multi-head cross-attention, similar to how ViT uses a [CLS] token.

    Architecture:
        pool_query -> MultiHeadAttention(query, keys=hidden, values=hidden) -> output

    Args:
        hidden_size: Size of encoder hidden states
        output_size: Size of output (default: hidden_size)
        num_heads: Number of attention heads (default: 8)
        dropout: Attention dropout (default: 0.1)

    Example:
        >>> pooler = AttentionPooler(hidden_size=768, num_heads=8)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> attention_mask = torch.ones(2, 128)
        >>> pooled = pooler(hidden_states, attention_mask)
        >>> assert pooled.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, output_size)
        self.num_heads = num_heads

        # Learnable pool query token
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.normal_(self.pool_query, std=0.02)

        # Multi-head attention for cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Optional projection if output_size differs
        if self.output_size != hidden_size:
            self.output_projection = nn.Linear(hidden_size, self.output_size)
        else:
            self.output_projection = None

        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool via multi-head cross-attention.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            Pooled output (batch_size, output_size)
        """
        batch_size = hidden_states.size(0)

        # Expand pool query for batch
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch, 1, hidden)

        # Convert attention_mask to key_padding_mask format for MHA
        # MHA expects: True = ignore position, False = attend
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # Invert mask

        # Cross-attention: query attends to hidden_states
        attended, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )

        # Squeeze sequence dimension
        output = attended.squeeze(1)  # (batch, hidden)

        # Optional projection
        if self.output_projection is not None:
            output = self.output_projection(output)

        output = self.layer_norm(output)

        return output


# =============================================================================
# Utility Functions
# =============================================================================


def get_pooler(
    pooling_strategy: str,
    hidden_size: int,
    output_size: int | None = None,
    **kwargs,
) -> BasePooler:
    """
    Factory function to create a pooler by name.

    Args:
        pooling_strategy: Name of pooling strategy
            ('cls', 'mean', 'max', 'weighted_mean', 'last_token')
        hidden_size: Size of encoder hidden states
        output_size: Size of pooled output
        **kwargs: Additional arguments for specific poolers

    Returns:
        Configured pooler instance

    Example:
        >>> pooler = get_pooler('mean', hidden_size=768)
        >>> isinstance(pooler, MeanPooler)
        True
    """
    poolers = {
        "cls": CLSPooler,
        "mean": MeanPooler,
        "max": MaxPooler,
        "weighted_mean": WeightedMeanPooler,
        "weighted": WeightedMeanPooler,
        "last_token": LastTokenPooler,
        "last": LastTokenPooler,
        "cls_mean": CLSMeanPooler,
        "attention": AttentionPooler,
    }

    strategy = pooling_strategy.lower()
    if strategy not in poolers:
        raise ValueError(
            f"Unknown pooling strategy: {pooling_strategy}. " f"Available: {list(poolers.keys())}"
        )

    return poolers[strategy](hidden_size=hidden_size, output_size=output_size, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BasePooler",
    "CLSPooler",
    "MeanPooler",
    "MaxPooler",
    "WeightedMeanPooler",
    "LastTokenPooler",
    "CLSMeanPooler",
    "AttentionPooler",
    "get_pooler",
]

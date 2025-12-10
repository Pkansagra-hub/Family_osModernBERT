"""
Cross-Attention Pair Encoder for NLI and Relation Tasks

This module provides a cross-attention mechanism for encoding pairs of sequences,
enabling better representation learning for tasks that require comparing two texts:
    - Natural Language Inference (NLI): premise-hypothesis comparison
    - Relation Extraction: entity pair relationship classification
    - Semantic Similarity: sentence pair comparison

Architecture:
    The encoder uses multi-head cross-attention where one sequence attends to another,
    followed by optional self-attention and feedforward layers.

Design Features:
    - Bidirectional cross-attention (A attends to B and B attends to A)
    - Residual connections for gradient flow
    - Optional pooling strategies for fixed-size output
    - Compatible with ModernBERT hidden states

Issue: 5.0.3 - Implement Cross-Attention Pair Encoder
Epic: 5.0 - Model Architecture Enhancements (Pre-Stage B)

References:
    - "ESIM: Enhanced LSTM for NLI" (Chen et al., 2017)
    - "A Decomposable Attention Model for NLI" (Parikh et al., 2016)
    - "Cross-Attention for Sentence Pair Modeling" (Vaswani et al., 2017)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PairEncoderConfig:
    """
    Configuration for CrossAttentionPairEncoder.

    Args:
        hidden_size: Size of input hidden states (768 for BERT-base)
        num_heads: Number of attention heads
        dropout: Dropout probability
        num_layers: Number of cross-attention layers
        use_bidirectional: Whether to use bidirectional cross-attention
        pooling_strategy: How to pool sequence outputs ("cls", "mean", "max", "attention")
        use_residual: Whether to use residual connections
        use_layer_norm: Whether to apply layer normalization
        use_ffn: Whether to include feedforward network after attention
        ffn_hidden_size: Size of feedforward hidden layer (default: 4x hidden_size)
    """

    hidden_size: int = 768
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1
    use_bidirectional: bool = True
    pooling_strategy: Literal["cls", "mean", "max", "attention", "concat_pool"] = "attention"
    use_residual: bool = True
    use_layer_norm: bool = True
    use_ffn: bool = True
    ffn_hidden_size: int | None = None  # Default: 4x hidden_size

    def __post_init__(self):
        """Set defaults and validate."""
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


# =============================================================================
# Cross-Attention Layer
# =============================================================================


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer where query attends to key-value.

    This implements standard multi-head attention where:
        - Query comes from sequence A
        - Key and Value come from sequence B
        - Output has same shape as query sequence

    Args:
        hidden_size: Size of hidden states
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_residual: Whether to add residual connection
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_residual = use_residual

        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute cross-attention.

        Args:
            query_states: Query sequence (batch, query_len, hidden_size)
            key_value_states: Key-value sequence (batch, kv_len, hidden_size)
            query_mask: Mask for query sequence (batch, query_len)
            key_value_mask: Mask for key-value sequence (batch, kv_len)

        Returns:
            Attended output (batch, query_len, hidden_size)
        """
        batch_size, query_len, _ = query_states.shape
        kv_len = key_value_states.size(1)

        # Store residual
        residual = query_states

        # Pre-LayerNorm (more stable than post-LN)
        if self.layer_norm is not None:
            query_states = self.layer_norm(query_states)

        # Project Q, K, V
        Q = self.query(query_states)  # (batch, query_len, hidden)
        K = self.key(key_value_states)  # (batch, kv_len, hidden)
        V = self.value(key_value_states)  # (batch, kv_len, hidden)

        # Reshape for multi-head attention
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # (batch, num_heads, query_len, head_dim) @ (batch, num_heads, head_dim, kv_len)
        # -> (batch, num_heads, query_len, kv_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply key-value mask (mask out padding in key-value sequence)
        if key_value_mask is not None:
            # key_value_mask: (batch, kv_len) -> (batch, 1, 1, kv_len)
            attn_mask = key_value_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attn_mask.bool(), float("-inf"))

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        # (batch, num_heads, query_len, kv_len) @ (batch, num_heads, kv_len, head_dim)
        # -> (batch, num_heads, query_len, head_dim)
        attn_output = torch.matmul(attn_probs, V)

        # Reshape back
        # (batch, num_heads, query_len, head_dim) -> (batch, query_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual connection
        if self.use_residual:
            attn_output = attn_output + residual

        return attn_output


# =============================================================================
# Feedforward Network
# =============================================================================


class FeedForward(nn.Module):
    """
    Position-wise feedforward network with GELU activation.

    Two linear transformations with activation in between:
        FFN(x) = Linear2(GELU(Linear1(x)))

    Args:
        hidden_size: Input/output size
        intermediate_size: Size of intermediate layer
        dropout: Dropout probability
        use_residual: Whether to add residual connection
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual

        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = hidden_states

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.use_residual:
            hidden_states = hidden_states + residual

        return hidden_states


# =============================================================================
# Bidirectional Cross-Attention Block
# =============================================================================


class BidirectionalCrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block.

    Applies cross-attention in both directions:
        1. Sequence A attends to Sequence B
        2. Sequence B attends to Sequence A

    This enables rich interaction between the two sequences.

    Args:
        hidden_size: Size of hidden states
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_ffn: Whether to include feedforward network
        ffn_hidden_size: Size of feedforward hidden layer
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_ffn: bool = True,
        ffn_hidden_size: int | None = None,
    ):
        super().__init__()

        # A attends to B
        self.cross_attn_a_to_b = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        # B attends to A
        self.cross_attn_b_to_a = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Optional feedforward networks
        if use_ffn:
            ffn_size = ffn_hidden_size or hidden_size * 4
            self.ffn_a = FeedForward(hidden_size, ffn_size, dropout)
            self.ffn_b = FeedForward(hidden_size, ffn_size, dropout)
        else:
            self.ffn_a = None
            self.ffn_b = None

    def forward(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention.

        Args:
            hidden_a: Sequence A hidden states (batch, len_a, hidden_size)
            hidden_b: Sequence B hidden states (batch, len_b, hidden_size)
            mask_a: Attention mask for A (batch, len_a)
            mask_b: Attention mask for B (batch, len_b)

        Returns:
            Tuple of (updated_a, updated_b)
        """
        # A attends to B
        hidden_a = self.cross_attn_a_to_b(
            query_states=hidden_a,
            key_value_states=hidden_b,
            query_mask=mask_a,
            key_value_mask=mask_b,
        )

        # B attends to A
        hidden_b = self.cross_attn_b_to_a(
            query_states=hidden_b,
            key_value_states=hidden_a,
            query_mask=mask_b,
            key_value_mask=mask_a,
        )

        # Feedforward
        if self.ffn_a is not None:
            hidden_a = self.ffn_a(hidden_a)
        if self.ffn_b is not None:
            hidden_b = self.ffn_b(hidden_b)

        return hidden_a, hidden_b


# =============================================================================
# Attention Pooling
# =============================================================================


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence representations.

    Learns a query vector that attends to all positions,
    producing a fixed-size representation.

    Args:
        hidden_size: Size of hidden states
        num_heads: Number of attention heads (default: 1)
    """

    def __init__(self, hidden_size: int, num_heads: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Learnable query
        self.query = nn.Parameter(torch.randn(1, num_heads, hidden_size // num_heads))

        # Projections
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Scaling
        self.scale = 1.0 / math.sqrt(hidden_size // num_heads)

        self._init_weights()

    def _init_weights(self):
        """Initialize query."""
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size vector.

        Args:
            hidden_states: Sequence (batch, seq_len, hidden_size)
            attention_mask: Mask (batch, seq_len)

        Returns:
            Pooled representation (batch, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads

        # Project to K, V
        K = self.key(hidden_states)  # (batch, seq_len, hidden)
        V = self.value(hidden_states)  # (batch, seq_len, hidden)

        # Reshape for multi-head
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Expand query for batch
        Q = self.query.expand(batch_size, -1, -1).unsqueeze(2)  # (batch, num_heads, 1, head_dim)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, heads, 1, seq)

        # Apply mask
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            attn_scores = attn_scores.masked_fill(~attn_mask.bool(), float("-inf"))

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply to values
        pooled = torch.matmul(attn_probs, V)  # (batch, heads, 1, head_dim)
        pooled = pooled.squeeze(2)  # (batch, heads, head_dim)
        pooled = pooled.view(batch_size, -1)  # (batch, hidden_size)

        # Output projection
        pooled = self.out_proj(pooled)

        return pooled


# =============================================================================
# Cross-Attention Pair Encoder
# =============================================================================


class CrossAttentionPairEncoder(nn.Module):
    """
    Cross-Attention Pair Encoder for NLI and Relation tasks.

    Encodes a pair of sequences using cross-attention, producing a fixed-size
    representation suitable for classification.

    Architecture:
        1. Input: Two sequences (e.g., premise and hypothesis)
        2. Cross-attention layers (bidirectional or unidirectional)
        3. Pooling to fixed-size representations
        4. Combination (concatenation + optional interaction)
        5. Output projection

    Args:
        hidden_size: Size of input hidden states (768 for BERT-base)
        num_heads: Number of attention heads
        dropout: Dropout probability
        num_layers: Number of cross-attention layers
        use_bidirectional: Whether to use bidirectional cross-attention
        pooling_strategy: How to pool ("cls", "mean", "max", "attention", "concat_pool")
        output_size: Size of output representation (default: hidden_size)

    Example:
        >>> encoder = CrossAttentionPairEncoder(hidden_size=768, num_heads=8)
        >>> premise = torch.randn(2, 64, 768)
        >>> hypothesis = torch.randn(2, 32, 768)
        >>> premise_mask = torch.ones(2, 64)
        >>> hypothesis_mask = torch.ones(2, 32)
        >>> pair_repr = encoder(premise, hypothesis, premise_mask, hypothesis_mask)
        >>> assert pair_repr.shape == (2, 768)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_bidirectional: bool = True,
        pooling_strategy: Literal["cls", "mean", "max", "attention", "concat_pool"] = "attention",
        output_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.pooling_strategy = pooling_strategy
        self.output_size = output_size or hidden_size

        # Cross-attention layers
        if use_bidirectional:
            self.cross_attn_layers = nn.ModuleList(
                [
                    BidirectionalCrossAttentionBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_ffn=True,
                        ffn_hidden_size=hidden_size * 4,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            # Unidirectional: only A attends to B
            self.cross_attn_layers = nn.ModuleList(
                [
                    CrossAttentionLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        # Pooling
        if pooling_strategy == "attention":
            self.pooler_a = AttentionPooling(hidden_size, num_heads=1)
            self.pooler_b = AttentionPooling(hidden_size, num_heads=1)
        else:
            self.pooler_a = None
            self.pooler_b = None

        # Combination layer
        # After pooling, we have two vectors of size hidden_size
        # concat_pool: concat + element-wise diff + element-wise prod -> 4 * hidden
        if pooling_strategy == "concat_pool":
            combination_size = hidden_size * 4
        else:
            combination_size = hidden_size * 2  # Simple concat

        self.combination_layer = nn.Sequential(
            nn.Linear(combination_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size),
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(self.output_size)

        # Entity span projection (for relation extraction)
        # Input: 4 * hidden_size (entity_a, entity_b, diff, prod)
        self.entity_combination_layer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )

    def _pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pooler: AttentionPooling | None,
        strategy: str,
    ) -> torch.Tensor:
        """Pool a sequence to a fixed-size vector."""
        if strategy == "cls":
            return hidden_states[:, 0]  # First token

        elif strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)
                return sum_hidden / count
            return hidden_states.mean(dim=1)

        elif strategy == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool()
                hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
            return hidden_states.max(dim=1).values

        elif strategy == "attention":
            assert pooler is not None
            return pooler(hidden_states, attention_mask)

        elif strategy == "concat_pool":
            # Use both CLS and mean
            cls_repr = hidden_states[:, 0]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)
                mean_repr = sum_hidden / count
            else:
                mean_repr = hidden_states.mean(dim=1)
            return (cls_repr + mean_repr) / 2  # Average CLS and mean

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def forward(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a pair of sequences.

        Args:
            hidden_a: First sequence (batch, len_a, hidden_size) - e.g., premise
            hidden_b: Second sequence (batch, len_b, hidden_size) - e.g., hypothesis
            mask_a: Attention mask for A (batch, len_a)
            mask_b: Attention mask for B (batch, len_b)

        Returns:
            Pair representation (batch, output_size)
        """
        # Apply cross-attention layers
        for layer in self.cross_attn_layers:
            if self.use_bidirectional:
                hidden_a, hidden_b = layer(hidden_a, hidden_b, mask_a, mask_b)
            else:
                # Only A attends to B
                hidden_a = layer(
                    query_states=hidden_a,
                    key_value_states=hidden_b,
                    query_mask=mask_a,
                    key_value_mask=mask_b,
                )

        # Pool sequences
        pooled_a = self._pool_sequence(hidden_a, mask_a, self.pooler_a, self.pooling_strategy)
        pooled_b = self._pool_sequence(hidden_b, mask_b, self.pooler_b, self.pooling_strategy)

        # Combine representations
        if self.pooling_strategy == "concat_pool":
            # Rich combination: concat, diff, product
            diff = pooled_a - pooled_b
            prod = pooled_a * pooled_b
            combined = torch.cat([pooled_a, pooled_b, diff, prod], dim=-1)
        else:
            # Simple concatenation
            combined = torch.cat([pooled_a, pooled_b], dim=-1)

        # Final projection
        output = self.combination_layer(combined)
        output = self.output_norm(output)

        return output

    def forward_with_entity_spans(
        self,
        hidden_states: torch.Tensor,
        entity_a_span: tuple[torch.Tensor, torch.Tensor],
        entity_b_span: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode entity pair from a single sequence (for relation extraction).

        Instead of two separate sequences, extracts entity representations
        from spans within a single sequence.

        Args:
            hidden_states: Sequence (batch, seq_len, hidden_size)
            entity_a_span: Tuple of (start_indices, end_indices) for entity A
            entity_b_span: Tuple of (start_indices, end_indices) for entity B
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pair representation (batch, output_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device

        # Vectorized span extraction using masking
        def extract_span_repr_vectorized(
            hidden: torch.Tensor,
            starts: torch.Tensor,
            ends: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract mean representation of spans using vectorized operations.

            Uses torch.arange broadcasting to create span masks efficiently,
            avoiding slow Python loops.
            """
            # Create position indices: (1, seq_len)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)

            # Create span masks: (batch, seq_len)
            # True where position is within [start, end] inclusive
            start_mask = positions >= starts.unsqueeze(1)  # (batch, seq_len)
            end_mask = positions <= ends.unsqueeze(1)  # (batch, seq_len)
            span_mask = start_mask & end_mask  # (batch, seq_len)

            # Expand mask for hidden dimension: (batch, seq_len, 1)
            span_mask_expanded = span_mask.unsqueeze(-1).float()

            # Masked sum and count
            masked_hidden = hidden * span_mask_expanded  # (batch, seq_len, hidden)
            span_sum = masked_hidden.sum(dim=1)  # (batch, hidden)
            span_count = span_mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)

            # Mean of span tokens
            return span_sum / span_count  # (batch, hidden)

        entity_a_repr = extract_span_repr_vectorized(
            hidden_states, entity_a_span[0], entity_a_span[1]
        )
        entity_b_repr = extract_span_repr_vectorized(
            hidden_states, entity_b_span[0], entity_b_span[1]
        )

        # Combine (rich representation for entity pairs)
        diff = entity_a_repr - entity_b_repr
        prod = entity_a_repr * entity_b_repr
        combined = torch.cat([entity_a_repr, entity_b_repr, diff, prod], dim=-1)

        # Project 4x hidden to 2x hidden for combination layer
        combined = self.entity_combination_layer(combined)

        output = self.combination_layer(combined)
        output = self.output_norm(output)

        return output


# =============================================================================
# Simple Concatenation Fallback
# =============================================================================


class ConcatPairEncoder(nn.Module):
    """
    Simple concatenation-based pair encoder (fallback).

    Concatenates pooled representations without cross-attention.
    Use when cross-attention is not needed or for efficiency.

    Args:
        hidden_size: Size of hidden states
        pooling_strategy: How to pool ("cls", "mean", "max")
        output_size: Size of output representation
    """

    def __init__(
        self,
        hidden_size: int = 768,
        pooling_strategy: Literal["cls", "mean", "max"] = "mean",
        output_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        self.output_size = output_size or hidden_size

        # Combination layer
        self.combination = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.output_size),
        )
        self.output_norm = nn.LayerNorm(self.output_size)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool sequence."""
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0]
        elif self.pooling_strategy == "mean":
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (hidden_states * m).sum(1) / m.sum(1).clamp(min=1)
            return hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            if mask is not None:
                hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            return hidden_states.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_strategy}")

    def forward(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode pair via concatenation."""
        pooled_a = self._pool(hidden_a, mask_a)
        pooled_b = self._pool(hidden_b, mask_b)

        # Rich combination
        diff = pooled_a - pooled_b
        prod = pooled_a * pooled_b
        combined = torch.cat([pooled_a, pooled_b, diff, prod], dim=-1)

        output = self.combination(combined)
        output = self.output_norm(output)

        return output


# =============================================================================
# Factory Function
# =============================================================================


def create_pair_encoder(
    encoder_type: Literal["cross_attention", "concat", "none"] = "cross_attention",
    hidden_size: int = 768,
    num_heads: int = 8,
    num_layers: int = 1,
    **kwargs,
) -> nn.Module | None:
    """
    Factory function to create pair encoders.

    Args:
        encoder_type: Type of encoder
            - "cross_attention": Full cross-attention encoder
            - "concat": Simple concatenation fallback
            - "none": Return None (use for tasks that don't need pair encoding)
        hidden_size: Size of hidden states
        num_heads: Number of attention heads
        num_layers: Number of layers
        **kwargs: Additional arguments passed to encoder

    Returns:
        Pair encoder module or None
    """
    if encoder_type == "cross_attention":
        return CrossAttentionPairEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            **kwargs,
        )
    elif encoder_type == "concat":
        return ConcatPairEncoder(
            hidden_size=hidden_size,
            **kwargs,
        )
    elif encoder_type == "none":
        return None
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. " f"Available: cross_attention, concat, none"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "PairEncoderConfig",
    # Main encoder
    "CrossAttentionPairEncoder",
    # Components
    "CrossAttentionLayer",
    "BidirectionalCrossAttentionBlock",
    "AttentionPooling",
    "FeedForward",
    # Fallback
    "ConcatPairEncoder",
    # Factory
    "create_pair_encoder",
]

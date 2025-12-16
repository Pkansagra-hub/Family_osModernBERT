"""
Attention Mechanisms for UltraBERT-Gen MoE Decoder.

This module implements production-grade attention components:
    - RotaryEmbedding: Rotary Position Embedding (RoPE)
    - GroupedQueryAttention: GQA with RoPE and causal masking
    - CrossAttention: Encoder-decoder cross-attention

Architecture Reference:
    - RoFormer (Su et al., 2021) - RoPE
    - GQA (Ainslie et al., 2023) - Grouped-Query Attention
    - LLaMA 2 - Combined GQA + RoPE

Key Features:
    - RoPE for relative position encoding without learned parameters
    - GQA for memory-efficient attention (5:1 query/KV ratio)
    - KV cache for efficient autoregressive generation
    - Flash Attention compatible

Usage:
    from modeling_studio.models.attention import (
        RotaryEmbedding, GroupedQueryAttention, CrossAttention
    )
    from modeling_studio.models.decoder_config import DecoderMoEConfig

    config = DecoderMoEConfig()
    self_attn = GroupedQueryAttention(config)
    cross_attn = CrossAttention(config)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from modeling_studio.models.decoder_config import DecoderMoEConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Rotary Position Embedding (Issue 11.2.1)
# =============================================================================


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating query and key vectors.
    Unlike learned positional embeddings, RoPE:
    - Has no learnable parameters
    - Provides relative position information
    - Generalizes to longer sequences than seen during training

    Reference:
        RoFormer (Su et al., 2021)
        "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        dim: Dimension of the embedding (typically head_dim).
        max_seq_len: Maximum sequence length to precompute embeddings for.
        base: Base for computing frequencies (theta). Default: 10000.0

    Example:
        >>> rope = RotaryEmbedding(dim=64, max_seq_len=512)
        >>> cos, sin = rope(seq_len=128)
        >>> q_rotated = apply_rotary_pos_emb(q, cos, sin)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        # Shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin embeddings."""
        # Shape: (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Shape: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)

        # Shape: (seq_len, dim) - interleaved cos/sin
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for the given sequence.

        Args:
            x: Input tensor (used for device/dtype).
            seq_len: Sequence length. If None, uses x.shape[-2].
            position_ids: Optional position indices. Shape: (batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) tensors.
                Shape: (1, 1, seq_len, dim) or indexed by position_ids
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)

        if position_ids is not None:
            # Use position_ids for KV cache scenarios
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        return cos.to(x.dtype), sin.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    For RoPE application: splits tensor into two halves and swaps them
    with appropriate sign changes.

    Args:
        x: Tensor of shape (..., dim)

    Returns:
        Rotated tensor of same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor. Shape: (batch, num_heads, seq_len, head_dim)
        k: Key tensor. Shape: (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine embeddings. Shape: (1, 1, seq_len, head_dim)
        sin: Sine embeddings. Shape: (1, 1, seq_len, head_dim)

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Grouped Query Attention (Issue 11.2.2)
# =============================================================================


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with RoPE and causal masking.

    GQA uses fewer key-value heads than query heads, reducing memory
    and compute while maintaining quality. Each KV head is shared
    across multiple query heads (num_heads / num_kv_heads ratio).

    Reference:
        GQA (Ainslie et al., 2023)
        "GQA: Training Generalized Multi-Query Transformer Models"

    Args:
        config: DecoderMoEConfig with attention parameters.
        layer_idx: Layer index (for cache identification).

    Architecture (default config):
        - hidden_size: 1280
        - num_heads: 20 (query heads)
        - num_kv_heads: 4 (key/value heads, 5:1 ratio)
        - head_dim: 64

    Example:
        >>> config = DecoderMoEConfig()
        >>> attn = GroupedQueryAttention(config, layer_idx=0)
        >>> output, past_kv = attn(hidden_states, attention_mask)
    """

    def __init__(self, config: "DecoderMoEConfig", layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass with GQA, RoPE, and optional KV cache.

        Args:
            hidden_states: Input tensor. Shape: (batch, seq_len, hidden_size)
            attention_mask: Attention mask. Shape: (batch, 1, seq_len, kv_seq_len)
                Values should be 0 for tokens to attend and -inf for masked.
            position_ids: Position indices for RoPE. Shape: (batch, seq_len)
            past_key_value: Cached (key, value) from previous steps.
            use_cache: Whether to return updated cache.

        Returns:
            Tuple of:
                - output: Attention output. Shape: (batch, seq_len, hidden_size)
                - past_key_value: Updated cache if use_cache=True, else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        # Q: (batch, num_heads, seq_len, head_dim)
        # K, V: (batch, num_kv_heads, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, seq_len=seq_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Expand KV heads to match query heads for GQA
        # (batch, num_kv_heads, kv_len, head_dim) -> (batch, num_heads, kv_len, head_dim)
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Compute attention scores
        # (batch, num_heads, seq_len, kv_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply causal mask
        kv_len = key_states.shape[2]
        causal_mask = self._make_causal_mask(seq_len, kv_len, query_states.device, query_states.dtype)
        attn_weights = attn_weights + causal_mask

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match number of query heads.

        Args:
            hidden_states: Shape (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Expanded tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        if self.num_key_value_groups == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(batch, self.num_heads, seq_len, head_dim)

    def _make_causal_mask(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create causal attention mask.

        Args:
            q_len: Query sequence length.
            kv_len: Key/value sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Causal mask of shape (1, 1, q_len, kv_len).
            0 for allowed positions, -inf for masked.

        Note:
            For training (no KV cache): kv_len == q_len, so offset=0, creating
            standard causal mask. For inference with KV cache: q_len=1 (new token),
            kv_len>1 (cached history), so offset allows attending to all cached
            positions up to and including the new position.
        """
        # For KV cache: query positions are at the end
        # Position offset for causal mask
        offset = kv_len - q_len

        # Create mask where position i can attend to positions <= i + offset
        mask = torch.ones(q_len, kv_len, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1 + offset)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        return mask.unsqueeze(0).unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}"
        )


# =============================================================================
# Cross-Attention (Issue 11.2.3)
# =============================================================================


class CrossAttention(nn.Module):
    """
    Cross-Attention for encoder-decoder attention.

    Allows decoder to attend to encoder outputs. Unlike self-attention:
    - Query comes from decoder hidden states
    - Key and Value come from encoder context
    - No causal mask (can attend to all encoder positions)
    - No RoPE (encoder positions are already encoded)

    Args:
        config: DecoderMoEConfig with attention parameters.
        layer_idx: Layer index.

    Example:
        >>> config = DecoderMoEConfig()
        >>> cross_attn = CrossAttention(config, layer_idx=0)
        >>> output = cross_attn(decoder_hidden, encoder_hidden, encoder_mask)
    """

    def __init__(self, config: "DecoderMoEConfig", layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # All projections use full hidden_size (no GQA for cross-attention)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.

        Args:
            hidden_states: Decoder hidden states.
                Shape: (batch, dec_seq_len, hidden_size)
            encoder_hidden_states: Encoder outputs.
                Shape: (batch, enc_seq_len, hidden_size)
            encoder_attention_mask: Mask for encoder outputs.
                Shape: (batch, 1, 1, enc_seq_len)
                Values: 0 for valid, -inf for masked (padding).

        Returns:
            Cross-attention output. Shape: (batch, dec_seq_len, hidden_size)
        """
        batch_size, dec_len, _ = hidden_states.shape
        enc_len = encoder_hidden_states.shape[1]

        # Project Q from decoder, K/V from encoder
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        # Reshape for attention
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, dec_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, enc_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, enc_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        # (batch, num_heads, dec_len, enc_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply encoder attention mask (no causal mask for cross-attention)
        if encoder_attention_mask is not None:
            attn_weights = attn_weights + encoder_attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, dec_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "GroupedQueryAttention",
    "CrossAttention",
]

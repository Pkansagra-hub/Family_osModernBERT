"""
ModernBERT v3.3 Ultra - Multi-Scale Attention with Global Hub Tokens

This module implements the core attention mechanism for v3, solving the "Blind Hub" problem
by providing global bidirectional attention for hub tokens while maintaining efficient
sliding window attention for text tokens.

Key Features:
- Global attention for hub tokens (positions 0-4: [CLS], [EMO], [MEM], [REL], [TASK])
- Layer-wise sliding windows (64→128→256→512 tokens)
- Bidirectional global attention: hubs see all, all see hubs
- Memory-efficient implementation with optional Flash Attention support

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ==============================================================================
# Global Token Positions
# ==============================================================================

# Hub tokens and [CLS] have global attention (exempt from sliding windows)
GLOBAL_TOKEN_POSITIONS = [0, 1, 2, 3, 4]  # [CLS], [EMO], [MEM], [REL], [TASK]


# ==============================================================================
# Attention Mask Creation
# ==============================================================================


def create_global_local_attention_mask(
    seq_len: int,
    window_size: int,
    global_positions: List[int] = GLOBAL_TOKEN_POSITIONS,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create attention mask with global tokens + sliding windows.

    This is the v3.3 solution to the "Blind Hub" problem. Hub tokens need to see
    the entire sequence to aggregate information, and all tokens need to see hub
    tokens to condition their representations.

    Global tokens (positions 0-4):
      - Can attend to ALL tokens (row is all 1s)
      - Are attended by ALL tokens (column is all 1s)

    Regular text tokens (positions 5+):
      - Attend within sliding window + global tokens

    Args:
        seq_len: Sequence length (including special tokens)
        window_size: Sliding window size for text tokens
        global_positions: Positions with global attention (default: 0-4)
        device: Target device for tensor
        dtype: Output dtype (torch.bool for mask, torch.float for additive)

    Returns:
        Attention mask [seq_len, seq_len] where True/1 = can attend, False/0 = masked

    Visual example (seq_len=10, window=4, globals=[0,1,2,3,4]):

              0  1  2  3  4  5  6  7  8  9   (keys)
           +--------------------------------
       0   |  1  1  1  1  1  1  1  1  1  1   <- [CLS] global
       1   |  1  1  1  1  1  1  1  1  1  1   <- [EMO] global
       2   |  1  1  1  1  1  1  1  1  1  1   <- [MEM] global
       3   |  1  1  1  1  1  1  1  1  1  1   <- [REL] global
       4   |  1  1  1  1  1  1  1  1  1  1   <- [TASK] global
       5   |  1  1  1  1  1  1  1  1  0  0   <- text: globals + window
       6   |  1  1  1  1  1  0  1  1  1  0   <- text: globals + window
       7   |  1  1  1  1  1  0  0  1  1  1   <- text: globals + window
       8   |  1  1  1  1  1  0  0  0  1  1   <- text: globals + window
       9   |  1  1  1  1  1  0  0  0  0  1   <- text: globals + window
    (queries)

    Key insight:
    - Every row has 1s in columns 0-4 (globals visible to all)
    - Columns 0-4 have 1s in every row (globals attend everywhere)
    - Text tokens use sliding windows for non-global positions

    Cost: ~4 × seq_len additional attention (negligible vs N²)
    """
    if device is None:
        device = torch.device("cpu")

    # Start with zeros (no attention)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)

    # Global tokens can attend to everything (rows)
    for pos in global_positions:
        if pos < seq_len:
            mask[pos, :] = 1

    # Everything can attend to global tokens (columns)
    for pos in global_positions:
        if pos < seq_len:
            mask[:, pos] = 1

    # Sliding window for non-global positions
    half_window = window_size // 2
    for i in range(seq_len):
        if i in global_positions:
            continue  # Already handled

        # Window range: [i - half_window, i + half_window]
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, start:end] = 1

    return mask


def create_causal_global_local_mask(
    seq_len: int,
    window_size: int,
    global_positions: List[int] = GLOBAL_TOKEN_POSITIONS,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create CAUSAL attention mask (for decoder-style, if needed).

    Combines global attention + sliding window + causal masking.
    Position i can only attend to positions <= i (no future tokens).

    Args:
        seq_len: Sequence length
        window_size: Sliding window size for text tokens
        global_positions: Positions with global attention
        device: Target device
        dtype: Output dtype

    Returns:
        Causal attention mask [seq_len, seq_len]

    Note: v3 is an encoder (bidirectional), so this is rarely needed.
          Included for completeness and potential future decoder variants.
    """
    if device is None:
        device = torch.device("cpu")

    # Start with global+local mask
    mask = create_global_local_attention_mask(seq_len, window_size, global_positions, device, dtype)

    # Apply causal constraint (upper triangle = 0)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=device))

    if dtype == torch.bool:
        mask = mask & causal_mask
    else:
        mask = mask * causal_mask

    return mask


def expand_mask_for_batch(
    mask: torch.Tensor,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    """
    Expand 2D mask to 4D for multi-head attention.

    Args:
        mask: [seq_len, seq_len] attention mask
        batch_size: Batch size
        num_heads: Number of attention heads

    Returns:
        Expanded mask [batch, num_heads, seq_len, seq_len]

    Example:
        mask = create_global_local_attention_mask(10, 4)  # [10, 10]
        expanded = expand_mask_for_batch(mask, 2, 12)     # [2, 12, 10, 10]
    """
    # [seq_len, seq_len] -> [1, 1, seq_len, seq_len] -> [batch, heads, seq_len, seq_len]
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)


def convert_mask_to_additive(
    mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert boolean mask to additive mask for scaled_dot_product_attention.

    Args:
        mask: Boolean mask [True = can attend, False = masked]
        dtype: Output dtype (should match attention weights)

    Returns:
        Additive mask [0.0 = can attend, -inf = masked]

    Usage:
        bool_mask = create_global_local_attention_mask(...)
        additive_mask = convert_mask_to_additive(bool_mask)
        attn_weights = attn_weights + additive_mask
    """
    # True -> 0.0, False -> -inf
    return torch.where(
        mask, torch.zeros_like(mask, dtype=dtype), torch.full_like(mask, float("-inf"), dtype=dtype)
    )


# ==============================================================================
# Layer Window Configuration
# ==============================================================================

# Window sizes by layer band (1-indexed layer numbers)
LAYER_WINDOW_CONFIG: Dict[int, int] = {
    # Foundation Band (L1-6): Local token interactions (64 tokens)
    1: 64,
    2: 64,
    3: 64,
    4: 64,
    5: 64,
    6: 64,
    # Context Band (L7-18): Phrase-level patterns (128 tokens)
    7: 128,
    8: 128,
    9: 128,
    10: 128,
    11: 128,
    12: 128,
    13: 128,
    14: 128,
    15: 128,
    16: 128,
    17: 128,
    18: 128,
    # Semantic Band (L19-22): Sentence-level semantics (256 tokens)
    19: 256,
    20: 256,
    21: 256,
    22: 256,
    # Family Band (L23-28): Full context (512 tokens)
    23: 512,
    24: 512,
    25: 512,
    26: 512,
    27: 512,
    28: 512,
}

# Band definitions for easy lookup: (start_layer, end_layer, window_size)
LAYER_BANDS: Dict[str, Tuple[int, int, int]] = {
    "foundation": (1, 6, 64),
    "context": (7, 18, 128),
    "semantic": (19, 22, 256),
    "family": (23, 28, 512),
}


def get_window_size_for_layer(layer_idx: int) -> int:
    """
    Get the sliding window size for a given layer.

    Args:
        layer_idx: 1-indexed layer number (1-28)

    Returns:
        Window size (64, 128, 256, or 512)

    Raises:
        ValueError: If layer_idx is not in range [1, 28]

    Example:
        >>> get_window_size_for_layer(1)
        64
        >>> get_window_size_for_layer(15)
        128
        >>> get_window_size_for_layer(25)
        512
    """
    if layer_idx in LAYER_WINDOW_CONFIG:
        return LAYER_WINDOW_CONFIG[layer_idx]

    # Fallback: determine from band
    if 1 <= layer_idx <= 6:
        return 64
    elif 7 <= layer_idx <= 18:
        return 128
    elif 19 <= layer_idx <= 22:
        return 256
    elif 23 <= layer_idx <= 28:
        return 512
    else:
        raise ValueError(f"Invalid layer index: {layer_idx}. Must be in range [1, 28].")


def get_layer_band_name(layer_idx: int) -> str:
    """
    Get the band name for a layer.

    Args:
        layer_idx: 1-indexed layer number (1-28)

    Returns:
        Band name ("foundation", "context", "semantic", or "family")

    Raises:
        ValueError: If layer_idx is not in range [1, 28]

    Example:
        >>> get_layer_band_name(3)
        'foundation'
        >>> get_layer_band_name(20)
        'semantic'
    """
    if 1 <= layer_idx <= 6:
        return "foundation"
    elif 7 <= layer_idx <= 18:
        return "context"
    elif 19 <= layer_idx <= 22:
        return "semantic"
    elif 23 <= layer_idx <= 28:
        return "family"
    else:
        raise ValueError(f"Invalid layer index: {layer_idx}. Must be in range [1, 28].")


def get_attention_mask_for_layer(
    layer_idx: int,
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Get the appropriate attention mask for a specific layer.

    Convenience function that combines window size lookup and mask creation.

    Args:
        layer_idx: 1-indexed layer number (1-28)
        seq_len: Sequence length
        device: Target device
        dtype: Output dtype

    Returns:
        Attention mask [seq_len, seq_len] with layer-specific window size

    Example:
        >>> mask = get_attention_mask_for_layer(1, 100)  # Foundation: 64-window
        >>> mask = get_attention_mask_for_layer(25, 100)  # Family: 512-window
    """
    window_size = get_window_size_for_layer(layer_idx)

    return create_global_local_attention_mask(
        seq_len=seq_len,
        window_size=window_size,
        global_positions=GLOBAL_TOKEN_POSITIONS,
        device=device,
        dtype=dtype,
    )


def print_layer_config() -> None:
    """
    Print the layer window configuration for debugging.

    Example output:
        📊 Layer Window Configuration:
        --------------------------------------------------
          Foundation   (L 1- 6): window = 64
          Context      (L 7-18): window = 128
          Semantic     (L19-22): window = 256
          Family       (L23-28): window = 512
        --------------------------------------------------
    """
    print("\n[CONFIG] Layer Window Configuration:")
    print("-" * 50)
    for band_name, (start, end, window) in LAYER_BANDS.items():
        print(f"  {band_name.capitalize():12} (L{start:2}-{end:2}): window = {window}")
    print("-" * 50)


def get_layer_config_summary() -> Dict[str, Dict]:
    """
    Get layer configuration as a dictionary for programmatic access.

    Returns:
        Dictionary mapping band names to config dicts with keys:
        - start_layer: First layer in band (1-indexed)
        - end_layer: Last layer in band (1-indexed)
        - window_size: Sliding window size
        - num_layers: Number of layers in band

    Example:
        >>> config = get_layer_config_summary()
        >>> config["foundation"]["window_size"]
        64
        >>> config["family"]["num_layers"]
        6
    """
    summary = {}
    for band_name, (start, end, window) in LAYER_BANDS.items():
        summary[band_name] = {
            "start_layer": start,
            "end_layer": end,
            "window_size": window,
            "num_layers": end - start + 1,
        }
    return summary


# ==============================================================================
# Utility Functions
# ==============================================================================


def visualize_attention_mask(
    mask: torch.Tensor,
    max_display: int = 20,
) -> None:
    """
    Print a visual representation of an attention mask.

    Args:
        mask: [seq_len, seq_len] attention mask (boolean or float)
        max_display: Maximum sequence length to display (truncates if longer)

    Example:
        >>> mask = create_global_local_attention_mask(10, 4)
        >>> visualize_attention_mask(mask)
    """
    seq_len = mask.shape[0]
    display_len = min(seq_len, max_display)

    # Convert to CPU numpy for printing
    mask_np = mask[:display_len, :display_len].cpu().numpy()

    # Convert to 0/1 for display
    if mask.dtype == torch.bool:
        mask_display = mask_np.astype(int)
    else:
        mask_display = (mask_np > -1e9).astype(int)

    print(f"\nAttention Mask ({seq_len}×{seq_len}, showing {display_len}×{display_len}):")
    print("   ", "  ".join(f"{i:2}" for i in range(display_len)))
    print("   +" + "---" * display_len)

    for i in range(display_len):
        row_str = "  ".join(str(mask_display[i, j]) for j in range(display_len))
        print(f"{i:2} | {row_str}")

    if seq_len > max_display:
        print(f"   ... ({seq_len - max_display} more rows/cols)")


def count_attention_patterns(mask: torch.Tensor) -> Dict[str, int]:
    """
    Count attention patterns in a mask for analysis.

    Args:
        mask: [seq_len, seq_len] boolean attention mask

    Returns:
        Dictionary with counts:
        - global_tokens: Number of tokens with full row attention
        - attended_by_all: Number of tokens attended by all positions
        - total_edges: Total number of attention edges (True entries)
        - density: Fraction of possible edges present

    Example:
        >>> mask = create_global_local_attention_mask(100, 64)
        >>> stats = count_attention_patterns(mask)
        >>> print(f"Density: {stats['density']:.2%}")
    """
    seq_len = mask.shape[0]
    mask_bool = mask.bool() if mask.dtype != torch.bool else mask

    # Count global tokens (full row = all True)
    global_tokens = (mask_bool.sum(dim=1) == seq_len).sum().item()

    # Count tokens attended by all (full column = all True)
    attended_by_all = (mask_bool.sum(dim=0) == seq_len).sum().item()

    # Total attention edges
    total_edges = mask_bool.sum().item()

    # Density
    possible_edges = seq_len * seq_len
    density = total_edges / possible_edges

    return {
        "seq_len": seq_len,
        "global_tokens": global_tokens,
        "attended_by_all": attended_by_all,
        "total_edges": total_edges,
        "possible_edges": possible_edges,
        "density": density,
    }


# ==============================================================================
# Multi-Head Attention with Global-Local Pattern
# ==============================================================================


class MultiScaleAttentionWithGlobals(nn.Module):
    """
    Multi-head attention with:
    - Sliding window for text tokens
    - Global attention for hub tokens (positions 0-4)
    - Layer-specific window sizes

    This is the v3.3 solution to the "Blind Hub" problem.

    Architecture:
    - 12 attention heads × 64 dimensions per head = 768 total
    - QKV projections: 768 → 768 each
    - Output projection: 768 → 768
    - Layer-specific window sizes: 64/128/256/512 tokens
    - Global tokens (0-4) attend to all, all attend to globals

    Args:
        hidden_size: Model hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        attention_dropout: Dropout probability for attention weights
        layer_idx: 1-indexed layer number (1-28) for window size lookup
        max_position_embeddings: Maximum sequence length (default: 8192)

    Example:
        >>> attn = MultiScaleAttentionWithGlobals(layer_idx=1)  # Foundation: 64-window
        >>> attn = MultiScaleAttentionWithGlobals(layer_idx=25) # Family: 512-window
        >>> hidden = torch.randn(2, 100, 768)  # [batch, seq, hidden]
        >>> output, weights = attn(hidden, output_attentions=True)
        >>> output.shape  # [2, 100, 768]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        layer_idx: int = 1,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx
        self.window_size = get_window_size_for_layer(layer_idx)
        self.max_position_embeddings = max_position_embeddings

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache for attention mask (avoid recomputing on every forward pass)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _get_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get or create cached attention mask for the given sequence length.

        This implements mask caching to avoid recreating the same mask on every
        forward pass. The mask is layer-specific (different window sizes) but
        sequence-length-specific (same mask for all batches with same seq_len).

        Args:
            seq_len: Sequence length
            device: Target device for the mask

        Returns:
            Attention mask [seq_len, seq_len] where 1.0 = can attend, 0.0 = masked
        """
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            self._cached_mask = create_global_local_attention_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                global_positions=GLOBAL_TOKEN_POSITIONS,
                device=device,
                dtype=torch.float32,
            )
            self._cached_seq_len = seq_len

        return self._cached_mask.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with global-local attention using SDPA optimization.

        This implementation uses F.scaled_dot_product_attention (SDPA) for memory
        efficiency. SDPA automatically uses Flash Attention or Memory-Efficient
        kernels when available, providing near-Flash-Attention performance while
        supporting custom attention masks.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask [batch, seq_len]
                           where 1 = valid token, 0 = padding
            output_attentions: Whether to return attention weights for visualization

        Returns:
            Tuple of:
            - output: Attention output [batch, seq_len, hidden_size]
            - attention_weights: Optional [batch, heads, seq_len, seq_len]
                                (only if output_attentions=True)

        Example:
            >>> attn = MultiScaleAttentionWithGlobals(layer_idx=1)
            >>> hidden = torch.randn(2, 100, 768)
            >>> mask = torch.ones(2, 100)  # All tokens valid
            >>> mask[0, 80:] = 0  # Padding in first sample
            >>> output, weights = attn(hidden, mask, output_attentions=True)
            >>> output.shape  # [2, 100, 768]
            >>> weights.shape  # [2, 12, 100, 100]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)  # [batch, seq, hidden]
        key = self.k_proj(hidden_states)  # [batch, seq, hidden]
        value = self.v_proj(hidden_states)  # [batch, seq, hidden]

        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(
            1, 2
        )

        # Create combined attention mask (global-local + padding)
        global_local_mask = self._get_attention_mask(seq_len, hidden_states.device)
        # global_local_mask shape: [seq, seq]

        if attention_mask is not None:
            # Combine: global_local_mask AND padding_mask
            # attention_mask: [batch, seq] where 1 = valid, 0 = padding
            # Expand global_local_mask: [seq, seq] -> [batch, seq, seq]
            combined_mask = global_local_mask.unsqueeze(0).expand(batch_size, -1, -1)
            # Apply padding mask to key dimension (last dim)
            padding_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq]
            combined_mask = combined_mask * padding_mask.float()
        else:
            combined_mask = global_local_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # combined_mask shape: [batch, seq, seq]

        # Convert to boolean mask for SDPA (True = MASK OUT, False = attend)
        attn_mask = combined_mask == 0

        if output_attentions:
            # Fall back to manual attention for debugging
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value)
            # attn_output shape: [batch, heads, seq, head_dim]
        else:
            # Use SDPA for memory-efficient attention (PyTorch 2.0+)
            # SDPA automatically uses Flash/Memory-Efficient kernels when possible
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1),
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            # attn_output shape: [batch, heads, seq, head_dim]
            attn_weights = None

        # Reshape back to [batch, seq, hidden]: transpose then contiguous view
        # Input: [batch, heads, seq, head_dim]
        # After transpose(1, 2): [batch, seq, heads, head_dim]
        # After view: [batch, seq, hidden_size]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"layer={self.layer_idx}, "
            f"window={self.window_size}, "
            f"heads={self.num_attention_heads}, "
            f"head_dim={self.head_dim}"
        )


# ==============================================================================
# Flash Attention 2 with Global Hub Token Support
# ==============================================================================


class FlashAttentionWithGlobals(nn.Module):
    """
    Flash Attention 2 implementation with Global Hub Token support.

    ⚠️ MITIGATION STRATEGY:
    1. Hub→Text Attention: ✅ Solved via manual calculation (Hubs see everything).
    2. Text→Hub Attention: ❌ NOT natively supported in Flash sliding window.
       - Impact: Text tokens may not see [EMO]/[REL] if window is small.
       - Use: ONLY for long-context inference where speed > perfect topology.
       - Training: Use MultiScaleAttentionWithGlobals instead.

    This implementation uses Flash Attention 2's sliding window kernels for speed
    while manually correcting hub token attention (positions 0-4) to ensure they
    see the entire sequence.

    WARNING: Text tokens outside the window cannot see hub tokens. This is a known
    limitation accepted for inference speed on long sequences (8k+ tokens).

    Args:
        hidden_size: Model hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        attention_dropout: Dropout probability for attention weights
        layer_idx: 1-indexed layer number (1-28) for window size lookup

    Example:
        >>> if FLASH_ATTN_AVAILABLE:
        ...     attn = FlashAttentionWithGlobals(layer_idx=25)  # Family: 512-window
        ...     hidden = torch.randn(2, 8192, 768)  # Long sequence
        ...     output, _ = attn(hidden)
        ...     output.shape  # [2, 8192, 768]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        layer_idx: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx
        self.window_size = get_window_size_for_layer(layer_idx)
        self.dropout_p = attention_dropout

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with Flash Attention 2 + Hub correction.

        ⚠️ Text→Hub attention is NOT preserved. Use only for long-context inference.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask [batch, seq_len]
            output_attentions: Must be False (Flash Attention doesn't support this)

        Returns:
            Tuple of (output, None). Attention weights cannot be returned.

        Raises:
            ValueError: If output_attentions=True (not supported)
        """
        # Flash Attention doesn't support output_attentions
        if output_attentions:
            raise ValueError(
                "Flash Attention does not support output_attentions=True. "
                "Use MultiScaleAttentionWithGlobals for debugging."
            )

        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape [batch, seq, heads, dim] for Flash Attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        # 1. Main Flash Attention (Sliding Window)
        # ⚠️ WARNING: Text tokens > window_size/2 away from 0 will NOT see global tokens 0-4
        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
            window_size=(self.window_size // 2, self.window_size // 2),
        )

        # 2. Hub→Text Correction (Global Tokens 0-4 see EVERYTHING)
        # We manually compute attention for indices 0-4 using standard attention
        global_q = q[:, :5, :, :]  # [batch, 5, heads, dim]

        # Standard attention scores for global query positions
        global_scores = torch.einsum("bqhd,bkhd->bhqk", global_q, k) / math.sqrt(self.head_dim)

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand mask: 1.0 is keep, 0.0 is mask -> additive: 0.0 keep, -inf mask
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
            global_scores = global_scores + padding_mask

        global_probs = F.softmax(global_scores, dim=-1)
        global_out_correction = torch.einsum("bhqk,bkhd->bqhd", global_probs, v)

        # Overwrite Flash output for positions 0-4 with correct global attention
        attn_output[:, :5, :, :] = global_out_correction

        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output), None

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"layer={self.layer_idx}, window={self.window_size}, "
            f"heads={self.num_attention_heads}, "
            f"⚠️ Text→Hub blind (use for inference only)"
        )


# ==============================================================================
# ModernBERT-Compatible Attention (Fused Wqkv)
# ==============================================================================


class ModernBertAttentionWithGlobals(nn.Module):
    """
    ModernBERT-compatible attention with fused Wqkv projection.

    This attention module exactly matches ModernBERT's weight structure
    to enable perfect weight transfer from v2 checkpoints, while still
    providing the v3 global-local attention pattern.

    Key differences from MultiScaleAttentionWithGlobals:
    - Uses fused Wqkv [hidden*3, hidden] instead of separate Q, K, V
    - Uses Wo instead of out_proj (naming convention)
    - Matches ModernBERT checkpoint key names exactly

    Weight mapping from ModernBERT:
        - attn.Wqkv.weight [2304, 768] -> self.Wqkv.weight
        - attn.Wo.weight [768, 768] -> self.Wo.weight

    Args:
        hidden_size: Model hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        attention_dropout: Dropout probability for attention weights
        layer_idx: 1-indexed layer number (1-28) for window size lookup
        max_position_embeddings: Maximum sequence length (default: 8192)

    Example:
        >>> attn = ModernBertAttentionWithGlobals(layer_idx=1)
        >>> hidden = torch.randn(2, 100, 768)
        >>> output, _ = attn(hidden)
        >>> output.shape  # [2, 100, 768]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        layer_idx: int = 1,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx
        self.window_size = get_window_size_for_layer(layer_idx)
        self.max_position_embeddings = max_position_embeddings

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )

        # Fused QKV projection (matches ModernBERT)
        # Wqkv: [hidden*3, hidden] = [2304, 768]
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)

        # Output projection (matches ModernBERT naming)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache for attention mask
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _get_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get or create cached attention mask."""
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            self._cached_mask = create_global_local_attention_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                global_positions=GLOBAL_TOKEN_POSITIONS,
                device=device,
                dtype=torch.float32,
            )
            self._cached_seq_len = seq_len
        return self._cached_mask.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with fused Wqkv projection.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask [batch, seq_len]
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection then split
        qkv = self.Wqkv(hidden_states)  # [batch, seq, hidden*3]
        query, key, value = qkv.chunk(3, dim=-1)  # Each [batch, seq, hidden]

        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(
            1, 2
        )

        # Create combined attention mask (global-local + padding)
        global_local_mask = self._get_attention_mask(seq_len, hidden_states.device)

        if attention_mask is not None:
            combined_mask = global_local_mask.unsqueeze(0).expand(batch_size, -1, -1)
            padding_mask = attention_mask.unsqueeze(1)
            combined_mask = combined_mask * padding_mask.float()
        else:
            combined_mask = global_local_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert to boolean mask for SDPA (True = MASK OUT)
        attn_mask = combined_mask == 0

        if output_attentions:
            # Manual attention for debugging
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value)
        else:
            # Use SDPA for memory-efficient attention
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1),
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None

        # Reshape back to [batch, seq, hidden]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )

        # Output projection
        attn_output = self.Wo(attn_output)

        return attn_output, attn_weights

    def extra_repr(self) -> str:
        return (
            f"layer={self.layer_idx}, window={self.window_size}, "
            f"heads={self.num_attention_heads}, fused_qkv=True (ModernBERT compat)"
        )


# ==============================================================================
# Attention Layer Factory (Safety Switch)
# ==============================================================================


def create_attention_layer(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    attention_dropout: float = 0.1,
    layer_idx: int = 1,
    use_flash_attention: bool = False,
    use_fused_qkv: bool = True,
) -> nn.Module:
    """
    Factory function implementing the DECISION MATRIX (Safety Switch).

    Decision Logic:
    1. If use_fused_qkv=True → ModernBertAttentionWithGlobals (v2 compatible)
    2. If Flash Attention missing → Standard (SDPA optimized)
    3. If use_flash_attention=False → Standard (for Training Phase)
    4. If use_flash_attention=True & available → Flash (for Long Inference)

    For v3 with v2 weight transfer, use_fused_qkv=True (default).

    Args:
        hidden_size: Model hidden dimension (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        attention_dropout: Dropout probability
        layer_idx: 1-indexed layer number (1-28)
        use_flash_attention: Whether to use Flash Attention
        use_fused_qkv: Whether to use fused Wqkv (ModernBERT compatible)

    Returns:
        Attention module
    """
    # Default: Use ModernBERT-compatible attention for v2 weight transfer
    if use_fused_qkv:
        return ModernBertAttentionWithGlobals(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )

    if use_flash_attention and FLASH_ATTN_AVAILABLE:
        return FlashAttentionWithGlobals(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )
    else:
        return MultiScaleAttentionWithGlobals(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Constants
    "GLOBAL_TOKEN_POSITIONS",
    "LAYER_WINDOW_CONFIG",
    "LAYER_BANDS",
    "FLASH_ATTN_AVAILABLE",
    # Mask creation
    "create_global_local_attention_mask",
    "create_causal_global_local_mask",
    "expand_mask_for_batch",
    "convert_mask_to_additive",
    # Layer configuration
    "get_window_size_for_layer",
    "get_layer_band_name",
    "get_attention_mask_for_layer",
    "print_layer_config",
    "get_layer_config_summary",
    # Attention modules
    "MultiScaleAttentionWithGlobals",
    "ModernBertAttentionWithGlobals",
    "FlashAttentionWithGlobals",
    "create_attention_layer",
    # Utilities
    "visualize_attention_mask",
    "count_attention_patterns",
]

# src/modeling_studio/models/ffn_v3.py

"""
Feed-Forward Network modules for ModernBERT v3.3 Ultra.

This module implements FFN layers used in all 28 transformer layers.
Phase 1 uses GELU activation to match v2 architecture for weight transfer.
SwiGLU variant is included for R&D experiments only (NOT production).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GELUFFN(nn.Module):
    """
    GELU Feed-Forward Network (same as v2).

    Architecture:
        hidden → intermediate (4x) → GELU → hidden
        768 → 3072 → GELU → 768

    This implementation matches v2 exactly to enable direct weight transfer
    via Function Preserving Growth strategy.

    Note: SwiGLU was considered for v3 Phase 2 but removed from roadmap
    per v3.3 decision (stability > marginal gains).

    Args:
        hidden_size: Input/output dimension (default: 768)
        intermediate_size: Intermediate dimension (default: 3072 = 4 * hidden)
        hidden_dropout_prob: Dropout probability (default: 0.1)
        activation: Activation function type (default: "gelu")

    Shape:
        - Input: [batch, seq_len, hidden_size]
        - Output: [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Up projection: 768 → 3072
        self.up_proj = nn.Linear(hidden_size, intermediate_size)

        # Down projection: 3072 → 768
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Activation
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "gelu_new":
            self.activation = self._gelu_new
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    @staticmethod
    def _gelu_new(x: torch.Tensor) -> torch.Tensor:
        """
        GELU approximation (used in some models like GPT-2).

        This is an alternative GELU implementation using tanh approximation.
        """
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        # Up project: [batch, seq, hidden] → [batch, seq, intermediate]
        intermediate = self.up_proj(hidden_states)

        # Activation
        intermediate = self.activation(intermediate)

        # Down project: [batch, seq, intermediate] → [batch, seq, hidden]
        output = self.down_proj(intermediate)

        # Dropout
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, intermediate={self.intermediate_size}"


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (DEPRECATED - R&D only).

    ⚠️ NOT used in v3 production. Kept for research experiments.

    Architecture:
        hidden → gate (4x) → SiLU
        hidden → up (4x)
        gate * up → down → hidden

    SwiGLU uses gating mechanism instead of simple activation:
        FFN(x) = (SiLU(W_gate @ x) ⊙ W_up @ x) @ W_down

    This can provide better performance but requires retraining and is
    NOT compatible with v2 weight transfer.

    Args:
        hidden_size: Input/output dimension (default: 768)
        intermediate_size: Intermediate dimension (default: 3072)
        hidden_dropout_prob: Dropout probability (default: 0.1)

    Shape:
        - Input: [batch, seq_len, hidden_size]
        - Output: [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU uses 2/3 of intermediate for gate and up each
        # to maintain same param count as GELU FFN
        # However, we use full intermediate_size for simplicity in R&D
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward: SiLU(gate) * up → down

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        # Gate projection with SiLU activation
        gate = F.silu(self.gate_proj(hidden_states))

        # Up projection
        up = self.up_proj(hidden_states)

        # Element-wise multiplication (gating)
        intermediate = gate * up

        # Down projection
        output = self.down_proj(intermediate)

        # Dropout
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, intermediate={self.intermediate_size} (SwiGLU - R&D ONLY)"
        )


def create_ffn(
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    ffn_type: str = "gelu",
) -> nn.Module:
    """
    Factory function to create FFN module.

    This function provides a unified interface for creating different
    FFN variants. By default, it creates the production-ready GELU FFN.

    Args:
        hidden_size: Input/output dimension (default: 768)
        intermediate_size: Intermediate dimension (default: 3072)
        hidden_dropout_prob: Dropout probability (default: 0.1)
        ffn_type: FFN type - "gelu" (default) or "swiglu" (R&D only)

    Returns:
        FFN module (GELUFFN or SwiGLUFFN)

    Raises:
        ValueError: If unknown ffn_type provided

    Examples:
        >>> # Production usage (GELU)
        >>> ffn = create_ffn(hidden_size=768, intermediate_size=3072)

        >>> # Research experiment (SwiGLU)
        >>> ffn = create_ffn(ffn_type="swiglu")  # Prints warning
    """
    if ffn_type == "gelu":
        return GELUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
    elif ffn_type == "swiglu":
        print("⚠️  WARNING: SwiGLU is R&D only - not recommended for production")
        print("   This variant does NOT support v2 weight transfer")
        return SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}. Must be 'gelu' or 'swiglu'.")


# Export public API
__all__ = [
    "GELUFFN",
    "SwiGLUFFN",
    "create_ffn",
]

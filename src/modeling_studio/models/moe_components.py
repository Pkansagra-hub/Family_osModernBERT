"""
Mixture-of-Experts Components for UltraBERT-Gen Decoder.

This module implements production-grade MoE components:
    - TopKRouter: Top-k gating with auxiliary losses
    - SwiGLUExpert: Single SwiGLU expert FFN
    - SharedExpert: Always-active shared expert
    - MoELayer: Complete sparse MoE FFN layer

Architecture Reference:
    - Mixtral 8x7B (Mistral AI) - MoE architecture
    - Switch Transformer (Google) - Load balancing, capacity factor
    - DeepSeek-MoE - Shared expert design

Key Features:
    - Load balancing loss to prevent expert collapse
    - Router z-loss to prevent unbounded logits
    - Capacity factor to limit tokens per expert
    - Expert dropout for regularization
    - Efficient batched expert computation

Usage:
    from modeling_studio.models.moe_components import MoELayer
    from modeling_studio.models.decoder_config import DecoderMoEConfig

    config = DecoderMoEConfig()
    moe_layer = MoELayer(config, layer_idx=2)

    output, aux_losses = moe_layer(hidden_states)
    total_loss = main_loss + aux_losses["load_balance"] + aux_losses["z_loss"]
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
# Auxiliary Loss Functions (Issues 10.1.2, 10.1.3)
# =============================================================================


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss to prevent expert collapse.

    This loss encourages uniform distribution of tokens across experts.
    When perfectly balanced, the loss equals 1.0 (num_experts × (1/num_experts)²).

    Reference:
        Switch Transformer (Fedus et al., 2021)
        "Switch Transformers: Scaling to Trillion Parameter Models"

    Args:
        router_probs: Router probabilities after softmax.
            Shape: (batch_size, seq_len, num_experts)
        expert_indices: Selected expert indices per token.
            Shape: (batch_size, seq_len, top_k)
        num_experts: Total number of experts.

    Returns:
        Scalar load balancing loss. Multiply by weight (e.g., 0.01) before adding to main loss.

    Example:
        >>> router_probs = torch.softmax(router_logits, dim=-1)
        >>> expert_indices = router_probs.topk(2, dim=-1).indices
        >>> lb_loss = compute_load_balancing_loss(router_probs, expert_indices, num_experts=8)
        >>> total_loss = main_loss + 0.01 * lb_loss
    """
    if router_probs.numel() == 0:
        return torch.tensor(0.0, device=router_probs.device, dtype=router_probs.dtype)

    batch_size, seq_len, _ = router_probs.shape
    total_tokens = batch_size * seq_len

    # Flatten expert indices for bincount
    # Shape: (batch_size * seq_len * top_k,)
    flat_indices = expert_indices.flatten()

    # Count tokens per expert (including duplicates from top-k)
    # Shape: (num_experts,)
    tokens_per_expert = torch.bincount(
        flat_indices,
        minlength=num_experts,
    ).float()

    # Normalize to get fraction of tokens per expert
    # Note: top_k tokens per position means total count is top_k × total_tokens
    total_assignments = tokens_per_expert.sum()
    if total_assignments > 0:
        tokens_per_expert = tokens_per_expert / total_assignments
    else:
        tokens_per_expert = torch.ones(num_experts, device=router_probs.device) / num_experts

    # Average router probability per expert
    # Shape: (num_experts,)
    prob_per_expert = router_probs.mean(dim=[0, 1])

    # Load balancing loss: N × Σ(f_i × P_i)
    # Where f_i = fraction of tokens to expert i
    #       P_i = average probability of expert i
    # Perfect balance: N × (1/N) × (1/N) × N = 1.0
    aux_loss = num_experts * (tokens_per_expert * prob_per_expert).sum()

    return aux_loss


def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute router z-loss to prevent logits from growing unbounded.

    Large router logits lead to very peaked distributions, which can
    cause training instability and poor expert utilization.

    Reference:
        ST-MoE (Zoph et al., 2022)
        "ST-MoE: Designing Stable and Transferable Sparse Expert Models"

    Args:
        router_logits: Raw router logits before softmax.
            Shape: (batch_size, seq_len, num_experts)

    Returns:
        Scalar z-loss. Multiply by weight (e.g., 0.001) before adding to main loss.

    Example:
        >>> router_logits = router.gate(hidden_states)
        >>> z_loss = compute_router_z_loss(router_logits)
        >>> total_loss = main_loss + 0.001 * z_loss
    """
    if router_logits.numel() == 0:
        return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

    # logsumexp provides numerical stability
    # z_loss = mean(logsumexp(logits)²)
    # This penalizes large logits that would create peaked distributions
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z.pow(2).mean()

    return z_loss


# =============================================================================
# Router (Issue 10.1.1)
# =============================================================================


class TopKRouter(nn.Module):
    """
    Top-K router with softmax gating and auxiliary losses.

    Routes each token to top_k experts based on learned gating weights.
    Includes load balancing loss and z-loss for stable training.

    Reference:
        Mixtral 8x7B (Jiang et al., 2024)

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Number of experts to route to.
        top_k: Number of experts selected per token.
        aux_loss_weight: Weight for load balancing loss. Default: 0.01
        z_loss_weight: Weight for router z-loss. Default: 0.001

    Attributes:
        gate: Linear projection for computing router logits.
        top_k: Number of experts per token.
        num_experts: Total number of experts.

    Example:
        >>> router = TopKRouter(hidden_size=1280, num_experts=8, top_k=2)
        >>> weights, indices, aux = router(hidden_states)
        >>> # weights: (batch, seq, top_k) - routing weights
        >>> # indices: (batch, seq, top_k) - expert indices
        >>> # aux: dict with 'load_balance_loss' and 'z_loss'
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight

        # Gate projection - no bias for cleaner routing
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Initialize with small uniform weights to prevent early bias
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize gate weights with small uniform distribution."""
        # Small uniform init prevents strong initial expert preferences
        nn.init.uniform_(self.gate.weight, -0.01, 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tuple of:
                - routing_weights: Normalized weights for selected experts.
                    Shape: (batch_size, seq_len, top_k)
                - expert_indices: Indices of selected experts.
                    Shape: (batch_size, seq_len, top_k)
                - aux_losses: Dictionary containing:
                    - 'load_balance_loss': Weighted load balancing loss
                    - 'z_loss': Weighted router z-loss
                    - 'raw_load_balance': Unweighted load balancing loss
                    - 'raw_z_loss': Unweighted z-loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute router logits
        # Shape: (batch_size, seq_len, num_experts)
        router_logits = self.gate(hidden_states)

        # Softmax over experts to get probabilities
        # Shape: (batch_size, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select top-k experts
        # Shape: (batch_size, seq_len, top_k)
        routing_weights, expert_indices = router_probs.topk(self.top_k, dim=-1)

        # Renormalize weights to sum to 1 for selected experts
        # This ensures stable gradient flow
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Cast back to input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Compute auxiliary losses
        raw_lb_loss = compute_load_balancing_loss(
            router_probs, expert_indices, self.num_experts
        )
        raw_z_loss = compute_router_z_loss(router_logits)

        weighted_lb = self.aux_loss_weight * raw_lb_loss
        weighted_z = self.z_loss_weight * raw_z_loss

        aux_losses = {
            "load_balance_loss": weighted_lb,
            "z_loss": weighted_z,
            "raw_load_balance": raw_lb_loss,
            "raw_z_loss": raw_z_loss,
            "total": weighted_lb + weighted_z,  # Convenience for wandb logging
        }

        return routing_weights, expert_indices, aux_losses

    def extra_repr(self) -> str:
        """Extra string representation for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}"
        )


# =============================================================================
# Expert FFNs (Issues 10.2.1, 10.2.2)
# =============================================================================


class SwiGLUExpert(nn.Module):
    """
    Single SwiGLU expert Feed-Forward Network.

    Implements the SwiGLU activation from LLaMA/PaLM:
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    This is more expressive than standard ReLU FFN and has become
    the standard for modern LLMs.

    Reference:
        GLU Variants Improve Transformer (Shazeer, 2020)
        LLaMA (Touvron et al., 2023)

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.

    Example:
        >>> expert = SwiGLUExpert(hidden_size=1280, intermediate_size=2048)
        >>> output = expert(hidden_states)  # Same shape as input
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU projections - no bias for efficiency
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of same shape as input.
        """
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def extra_repr(self) -> str:
        """Extra string representation."""
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


class SharedExpert(nn.Module):
    """
    Always-active shared expert for common patterns.

    The shared expert processes all tokens regardless of routing,
    capturing common patterns that all tokens need. This improves
    model quality and helps handle tokens that exceed expert capacity.

    Reference:
        DeepSeek-MoE (Dai et al., 2024)

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.

    Example:
        >>> shared = SharedExpert(hidden_size=1280, intermediate_size=1280)
        >>> shared_output = shared(hidden_states)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Use same SwiGLU architecture as sparse experts
        self.expert = SwiGLUExpert(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared expert.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of same shape as input.
        """
        return self.expert(x)

    def extra_repr(self) -> str:
        """Extra string representation."""
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


# =============================================================================
# MoE Layer Assembly (Issue 10.2.3)
# =============================================================================


class MoELayer(nn.Module):
    """
    Complete Mixture-of-Experts FFN layer.

    Combines router, sparse experts, and optional shared expert into
    a production-ready MoE layer with:
    - Top-k routing with renormalization
    - Load balancing and z-loss auxiliary losses
    - Capacity factor to limit tokens per expert
    - Expert dropout for regularization
    - Efficient batched computation

    Reference:
        Mixtral 8x7B (Jiang et al., 2024)
        DeepSeek-MoE (Dai et al., 2024)

    Args:
        config: DecoderMoEConfig with MoE hyperparameters.
        layer_idx: Layer index (for logging/debugging).

    Example:
        >>> config = DecoderMoEConfig()
        >>> moe = MoELayer(config, layer_idx=2)
        >>> output, aux_losses = moe(hidden_states)
        >>> total_aux = aux_losses['load_balance_loss'] + aux_losses['z_loss']
    """

    def __init__(self, config: "DecoderMoEConfig", layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.capacity_factor
        self.expert_dropout = config.expert_dropout

        # Router
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            aux_loss_weight=config.load_balancing_loss_weight,
            z_loss_weight=config.router_z_loss_weight,
        )

        # Sparse experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_intermediate_size,
            )
            for _ in range(config.num_experts)
        ])

        # Shared expert (optional)
        self.shared_expert = None
        if config.use_shared_expert:
            self.shared_expert = SharedExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tuple of:
                - output: Processed tensor of same shape as input.
                - aux_losses: Dictionary with auxiliary losses from router.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get routing decisions
        routing_weights, expert_indices, aux_losses = self.router(hidden_states)

        # Compute expert outputs using efficient batched implementation
        output = self._compute_expert_outputs(
            hidden_states, routing_weights, expert_indices
        )

        # Add shared expert output (if enabled)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            output = output + shared_output

        return output, aux_losses

    def _compute_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted outputs from selected experts.

        This implementation uses a loop over experts for clarity and correctness.
        For production, consider using torch.scatter_add or custom CUDA kernels.

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_size)
            routing_weights: Shape (batch_size, seq_len, top_k)
            expert_indices: Shape (batch_size, seq_len, top_k)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        top_k = routing_weights.shape[-1]

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Flatten for easier indexing
        flat_hidden = hidden_states.view(-1, hidden_size)  # (B*S, H)
        flat_weights = routing_weights.view(-1, top_k)  # (B*S, top_k)
        flat_indices = expert_indices.view(-1, top_k)  # (B*S, top_k)
        flat_output = output.view(-1, hidden_size)  # (B*S, H)

        # Compute capacity limit
        total_tokens = batch_size * seq_len
        capacity_per_expert = int(
            (total_tokens / self.num_experts) * self.capacity_factor
        )

        # Track tokens per expert for capacity limiting
        expert_token_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert across all top-k positions
            expert_mask = flat_indices == expert_idx  # (B*S, top_k)

            # Check if any tokens go to this expert
            if not expert_mask.any():
                continue

            # Get token indices and their corresponding top-k positions
            token_indices, topk_positions = expert_mask.nonzero(as_tuple=True)

            # Apply capacity limit
            num_tokens = len(token_indices)
            if num_tokens > capacity_per_expert:
                # Randomly select tokens up to capacity
                # In training, this provides regularization
                perm = torch.randperm(num_tokens, device=hidden_states.device)[:capacity_per_expert]
                token_indices = token_indices[perm]
                topk_positions = topk_positions[perm]

            # Apply expert dropout during training
            if self.training and self.expert_dropout > 0 and num_tokens > 0:
                keep_mask = torch.rand(len(token_indices), device=hidden_states.device) > self.expert_dropout
                if keep_mask.any():
                    token_indices = token_indices[keep_mask]
                    topk_positions = topk_positions[keep_mask]
                else:
                    continue  # Skip if all dropped

            if len(token_indices) == 0:
                continue

            # Get inputs for this expert
            expert_inputs = flat_hidden[token_indices]  # (num_tokens, H)

            # Get routing weights for these tokens
            expert_weights = flat_weights[token_indices, topk_positions]  # (num_tokens,)

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_inputs)  # (num_tokens, H)

            # Weight by routing weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            # Accumulate to output (tokens may have multiple experts)
            flat_output.index_add_(0, token_indices, weighted_output)

        return flat_output.view(batch_size, seq_len, hidden_size)

    def extra_repr(self) -> str:
        """Extra string representation."""
        return (
            f"layer_idx={self.layer_idx}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.num_experts_per_token}, "
            f"shared_expert={self.shared_expert is not None}"
        )


# =============================================================================
# Dense FFN (for layers 0-1)
# =============================================================================


class DenseSwiGLUFFN(nn.Module):
    """
    Dense SwiGLU Feed-Forward Network for non-MoE layers.

    Used in decoder layers 0-1 which use dense FFN instead of MoE.
    Architecture matches sparse experts but without routing.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.
        dropout: Dropout probability.

    Example:
        >>> ffn = DenseSwiGLUFFN(hidden_size=1280, intermediate_size=3584)
        >>> output = ffn(hidden_states)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of same shape as input.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        output = self.down_proj(F.silu(gate) * up)
        return self.dropout(output)

    def extra_repr(self) -> str:
        """Extra string representation."""
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


# =============================================================================
# RMSNorm
# =============================================================================


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is more efficient than LayerNorm as it doesn't require
    computing the mean. Used throughout modern LLMs (LLaMA, Mistral).

    Reference:
        Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)

    Args:
        hidden_size: Dimension to normalize.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = RMSNorm(hidden_size=1280)
        >>> normalized = norm(hidden_states)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Scale by learnable weight
        return self.weight * x

    def extra_repr(self) -> str:
        """Extra string representation."""
        return f"hidden_size={self.hidden_size}, eps={self.eps}"

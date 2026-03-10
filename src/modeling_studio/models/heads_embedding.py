"""
Retrieval-Native Embedding Head Candidates for 6-Head Bake-Off

All heads share the same external contract:
    input:  hidden_states [B, L, D], attention_mask [B, L]
    output: embedding [B, output_dim]  (L2-normalized)

Design principle (from sota_retrieval_architecture.md):
    The encoder mean representation is the anchor.
    Every learned improvement must be residual, bounded, and ablatable.

Heads:
    A. MeanBaselineHead         - masked mean pool (no learned params beyond optional proj)
    B. ResidualMLPMeanHead      - mean + zero-init residual MLP
    C. LatentResidualHead       - mean + bounded latent-attention residual
    D. AgreementGatedHead       - mean + multi-view agreement gate (novel)
    E. MultiPoolLowRankHead     - mean + low-rank multi-view fusion
    F. AnisotropyCorrectedHead  - mean + learned centering/scaling correction

Registry:
    EMBEDDING_HEAD_REGISTRY     - maps config string -> class
    create_embedding_head()     - factory function for config-driven instantiation
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared pooling helpers
# =============================================================================


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Masked mean pooling over sequence tokens."""
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def masked_max_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Masked max pooling over sequence tokens."""
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
    return hidden_states.max(dim=1)[0]


def cls_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """CLS token pooling (first token)."""
    return hidden_states[:, 0, :]


# =============================================================================
# Head A: MeanBaselineHead
# =============================================================================


class MeanBaselineHead(nn.Module):
    """Masked mean pool baseline. No learned parameters beyond optional projection.

    This is the retrieval anchor and must always be included in bake-off experiments
    as the control head.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim. None keeps hidden_size.
        normalize: L2-normalize output.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "mean_baseline"

        if output_dim is not None and output_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, output_dim)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = masked_mean_pool(hidden_states, attention_mask)
        if self.projection is not None:
            emb = self.projection(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Head B: ResidualMLPMeanHead
# =============================================================================


class ResidualMLPMeanHead(nn.Module):
    """Mean pool + zero-init residual MLP refiner.

    Simplest learned improvement over encoder_mean. The residual scale beta
    is initialized to 0 so the head starts exactly at mean-pool behavior.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim.
        normalize: L2-normalize output.
        intermediate_dim: MLP intermediate dimension.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        intermediate_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "residual_mlp_mean"

        int_dim = intermediate_dim or hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int_dim, self.output_dim),
        )
        # Zero-init residual scale so training starts at pure mean-pool
        self.beta = nn.Parameter(torch.zeros(1))

        if self.output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, self.output_dim)
        else:
            self.proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_mean = masked_mean_pool(hidden_states, attention_mask)
        base = self.proj(e_mean) if self.proj is not None else e_mean
        residual = self.mlp(self.layer_norm(e_mean))
        emb = base + self.beta * residual
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Head C: LatentResidualHead
# =============================================================================


class LatentResidualHead(nn.Module):
    """Mean pool + bounded latent-attention residual.

    Uses lightweight latent cross-attention to produce an auxiliary view,
    then applies a scalar gate (init ~0) to blend it with the mean anchor.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim.
        normalize: L2-normalize output.
        num_latents: Number of learnable query tokens.
        num_attn_heads: Number of attention heads.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        num_latents: int = 4,
        num_attn_heads: int = 4,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "latent_residual"

        # Latent cross-attention
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Scalar residual gate initialized near zero
        self.alpha = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ~ 0.047

        if self.output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, self.output_dim)
        else:
            self.proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = hidden_states.size(0)
        e_mean = masked_mean_pool(hidden_states, attention_mask)

        # Latent attention auxiliary view
        queries = self.latent_queries.expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.cross_attn(
            query=queries, key=hidden_states, value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.attn_norm(attn_out + queries)
        e_lat = attn_out.mean(dim=1)

        # Bounded residual update
        gate = torch.sigmoid(self.alpha)
        emb = e_mean + gate * (e_lat - e_mean)

        if self.proj is not None:
            emb = self.proj(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Head D: AgreementGatedHead (Novel)
# =============================================================================


class AgreementGatedHead(nn.Module):
    """Agreement-gated residual refinement (novel retrieval-first design).

    Computes multiple lightweight views (mean, CLS, latent), measures their
    agreement, and uses agreement statistics to gate the residual update.
    When views disagree the head stays close to the safe mean anchor.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim.
        normalize: L2-normalize output.
        num_latents: Number of learnable query tokens.
        num_attn_heads: Number of attention heads.
        gate_hidden: Hidden dim of the gate MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        num_latents: int = 4,
        num_attn_heads: int = 4,
        gate_hidden: int = 64,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "agreement_gated"

        # Latent attention for auxiliary view
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Agreement gate network
        # Input features: 3 cosine similarities + 3 norm ratios = 6 scalars
        NUM_AGREEMENT_FEATURES = 6
        self.gate_mlp = nn.Sequential(
            nn.Linear(NUM_AGREEMENT_FEATURES, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        # Initialize gate bias negative so gate starts near-closed
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

        if self.output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, self.output_dim)
        else:
            self.proj = None

    def _compute_agreement_features(
        self,
        e_mean: torch.Tensor,
        e_cls: torch.Tensor,
        e_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 6 agreement statistics between the three views."""
        # Cosine similarities (before normalization, on raw pool outputs)
        cos_mean_cls = F.cosine_similarity(e_mean, e_cls, dim=-1, eps=1e-8)
        cos_mean_lat = F.cosine_similarity(e_mean, e_lat, dim=-1, eps=1e-8)
        cos_cls_lat = F.cosine_similarity(e_cls, e_lat, dim=-1, eps=1e-8)

        # Norm ratios (captures magnitude agreement)
        n_mean = e_mean.norm(dim=-1, keepdim=False).clamp(min=1e-8)
        n_cls = e_cls.norm(dim=-1, keepdim=False).clamp(min=1e-8)
        n_lat = e_lat.norm(dim=-1, keepdim=False).clamp(min=1e-8)
        ratio_cls_mean = n_cls / n_mean
        ratio_lat_mean = n_lat / n_mean
        ratio_cls_lat = n_cls / n_lat

        # Stack: [B, 6]
        return torch.stack([
            cos_mean_cls, cos_mean_lat, cos_cls_lat,
            ratio_cls_mean, ratio_lat_mean, ratio_cls_lat,
        ], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = hidden_states.size(0)

        # Three views
        e_mean = masked_mean_pool(hidden_states, attention_mask)
        e_cls = cls_pool(hidden_states)

        queries = self.latent_queries.expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.cross_attn(
            query=queries, key=hidden_states, value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.attn_norm(attn_out + queries)
        e_lat = attn_out.mean(dim=1)

        # Agreement-gated residual
        agreement = self._compute_agreement_features(e_mean, e_cls, e_lat)
        alpha = torch.sigmoid(self.gate_mlp(agreement))  # [B, 1]

        emb = e_mean + alpha * (e_lat - e_mean)

        if self.proj is not None:
            emb = self.proj(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Head E: MultiPoolLowRankHead
# =============================================================================


class MultiPoolLowRankHead(nn.Module):
    """Multi-view low-rank fusion with residual update.

    Combines mean, CLS, max, and latent views using learned low-rank mixing
    coefficients. The fused view is applied as a residual update to the mean
    anchor.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim.
        normalize: L2-normalize output.
        num_latents: Number of latent queries.
        num_attn_heads: Number of attention heads.
        rank: Rank of the mixing matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        num_latents: int = 4,
        num_attn_heads: int = 4,
        rank: int = 8,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "multi_pool_low_rank"
        self.num_views = 4  # mean, cls, max, latent

        # Latent attention
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Low-rank mixing: W = A @ B  where A: [D, rank], B: [rank, num_views*D]
        # Applied as: mixed = (stacked_views @ B^T) @ A^T  ->  [B, D]
        # Simpler: learn a per-view weight vector and project
        self.view_weights = nn.Parameter(torch.zeros(self.num_views))  # softmax applied
        self.view_proj = nn.Linear(hidden_size, self.output_dim, bias=False)

        # Residual scale init near zero
        self.beta = nn.Parameter(torch.zeros(1))

        if self.output_dim != hidden_size:
            self.base_proj = nn.Linear(hidden_size, self.output_dim)
        else:
            self.base_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = hidden_states.size(0)

        e_mean = masked_mean_pool(hidden_states, attention_mask)
        e_cls = cls_pool(hidden_states)
        e_max = masked_max_pool(hidden_states, attention_mask)

        queries = self.latent_queries.expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.cross_attn(
            query=queries, key=hidden_states, value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.attn_norm(attn_out + queries)
        e_lat = attn_out.mean(dim=1)

        # Stack views: [B, num_views, D]
        views = torch.stack([e_mean, e_cls, e_max, e_lat], dim=1)

        # Learned soft mixing weights
        w = F.softmax(self.view_weights, dim=0)  # [num_views]
        mixed = (views * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, D]
        mixed = self.view_proj(mixed)

        # Residual update from mean base
        base = self.base_proj(e_mean) if self.base_proj is not None else e_mean
        emb = base + self.beta * (mixed - base)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Head F: AnisotropyCorrectedHead
# =============================================================================


class AnisotropyCorrectedHead(nn.Module):
    """Learned centering + diagonal scaling to correct anisotropic geometry.

    Targets the compressed cosine similarity range observed in encoder_mean
    (values clustered around 0.90-0.94). Applies learned centering vector,
    diagonal scaling, and optional low-rank whitening in a residual formulation.

    Args:
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim.
        normalize: L2-normalize output.
        use_low_rank_whiten: Add a low-rank whitening correction.
        whiten_rank: Rank of whitening correction matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
        use_low_rank_whiten: bool = True,
        whiten_rank: int = 32,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.normalize = normalize
        self.pooling = "anisotropy_corrected"
        self.use_low_rank_whiten = use_low_rank_whiten

        # Learnable centering vector (initialized to zero = no centering)
        self.center = nn.Parameter(torch.zeros(hidden_size))

        # Learnable diagonal scaling (initialized to ones = identity)
        self.scale = nn.Parameter(torch.ones(hidden_size))

        # Optional low-rank whitening correction: W_correction = U @ V^T
        if use_low_rank_whiten:
            self.whiten_U = nn.Parameter(
                torch.randn(hidden_size, whiten_rank) * (1.0 / math.sqrt(whiten_rank))
            )
            self.whiten_V = nn.Parameter(
                torch.randn(hidden_size, whiten_rank) * (1.0 / math.sqrt(whiten_rank))
            )
        else:
            self.whiten_U = None
            self.whiten_V = None

        # Residual scale for the correction (init near zero)
        self.beta = nn.Parameter(torch.zeros(1))

        if self.output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, self.output_dim)
        else:
            self.proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_mean = masked_mean_pool(hidden_states, attention_mask)

        # Center and scale
        corrected = (e_mean - self.center) * self.scale

        # Low-rank whitening correction
        if self.use_low_rank_whiten and self.whiten_U is not None:
            # W_correction @ e = U @ (V^T @ e)
            inner = torch.matmul(e_mean, self.whiten_V)  # [B, rank]
            lr_correction = torch.matmul(inner, self.whiten_U.t())  # [B, D]
            corrected = corrected + lr_correction

        # Residual formulation: interpolate between raw mean and corrected
        emb = e_mean + self.beta * (corrected - e_mean)

        if self.proj is not None:
            emb = self.proj(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# =============================================================================
# Registry and Factory
# =============================================================================

EMBEDDING_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "mean_baseline": MeanBaselineHead,
    "residual_mlp_mean": ResidualMLPMeanHead,
    "latent_residual": LatentResidualHead,
    "agreement_gated": AgreementGatedHead,
    "multi_pool_low_rank": MultiPoolLowRankHead,
    "anisotropy_corrected": AnisotropyCorrectedHead,
}


def create_embedding_head(
    head_type: str,
    hidden_size: int,
    output_dim: int | None = None,
    normalize: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Factory function for config-driven embedding head instantiation.

    Args:
        head_type: Registry key (e.g. 'agreement_gated').
        hidden_size: Encoder hidden dimension.
        output_dim: Output embedding dim. None keeps hidden_size.
        normalize: L2-normalize output.
        **kwargs: Head-specific parameters forwarded to constructor.

    Returns:
        Instantiated embedding head module.

    Raises:
        ValueError: If head_type is not in the registry.
    """
    if head_type not in EMBEDDING_HEAD_REGISTRY:
        valid = ", ".join(sorted(EMBEDDING_HEAD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown embedding head type '{head_type}'. Valid types: {valid}"
        )

    cls = EMBEDDING_HEAD_REGISTRY[head_type]
    return cls(
        hidden_size=hidden_size,
        output_dim=output_dim,
        normalize=normalize,
        **kwargs,
    )


def get_head_constructor_params(head: nn.Module) -> dict[str, Any]:
    """Extract constructor parameters from a head instance for metadata serialization.

    Returns a dict that can be JSON-serialized and later passed back to
    create_embedding_head() to reconstruct the same head architecture.
    """
    params: dict[str, Any] = {
        "hidden_size": getattr(head, "hidden_size", None),
        "output_dim": getattr(head, "output_dim", None),
        "normalize": getattr(head, "normalize", True),
    }

    # Head-type string from pooling attr (set by every head in this module)
    pooling = getattr(head, "pooling", None)
    if pooling is not None:
        params["head_type"] = pooling

    # Head-specific extras
    if isinstance(head, ResidualMLPMeanHead):
        mlp_layers = head.mlp
        if len(mlp_layers) > 0:
            params["intermediate_dim"] = mlp_layers[0].in_features

    if isinstance(head, (LatentResidualHead, AgreementGatedHead, MultiPoolLowRankHead)):
        if hasattr(head, "latent_queries"):
            params["num_latents"] = head.latent_queries.shape[1]
        if hasattr(head, "cross_attn"):
            params["num_attn_heads"] = head.cross_attn.num_heads

    if isinstance(head, AgreementGatedHead):
        gate_layers = head.gate_mlp
        if len(gate_layers) > 0:
            params["gate_hidden"] = gate_layers[0].out_features

    if isinstance(head, MultiPoolLowRankHead):
        params["rank"] = head.view_proj.weight.shape[0]  # output_dim, but rank is implicit

    if isinstance(head, AnisotropyCorrectedHead):
        params["use_low_rank_whiten"] = head.use_low_rank_whiten
        if head.whiten_U is not None:
            params["whiten_rank"] = head.whiten_U.shape[1]

    return params

"""
MoE Decoder and CounterfactualDecoderHead for UltraBERT-Gen.

This module implements the full decoder architecture:
    - EncoderProjection: Projects encoder outputs to decoder dimension
    - DecoderBlock: Single decoder layer with self-attn, cross-attn, FFN
    - CounterfactualDecoderHead: Full decoder head (13th head for multi-task model)

Architecture Reference:
    - LLaMA 2 style: Pre-norm, RMSNorm, SwiGLU
    - MoE: Switch Transformer / Mixtral style
    - Hybrid: Dense layers 0-1, MoE layers 2-7

Key Features:
    - ~420M total params, ~237M active per forward
    - 8 experts + 1 shared expert with top-2 routing
    - GQA (20/4) for self-attention, full attention for cross-attention
    - Weight tying between embedding and output projection

Usage:
    from modeling_studio.models.decoder_moe import CounterfactualDecoderHead
    from modeling_studio.models.decoder_config import DecoderMoEConfig

    config = DecoderMoEConfig()
    head = CounterfactualDecoderHead(
        config=config,
        encoder_hidden_size=768,  # ModernBERT-base
    )
    outputs = head(encoder_hidden, attention_mask, labels=target_ids)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_studio.models.attention import CrossAttention, GroupedQueryAttention
from modeling_studio.models.moe_components import (
    DenseSwiGLUFFN,
    MoELayer,
    RMSNorm,
)

if TYPE_CHECKING:
    from modeling_studio.models.decoder_config import DecoderMoEConfig

# Import BaseHead for inheritance
from modeling_studio.models.heads import BaseHead

logger = logging.getLogger(__name__)


# =============================================================================
# Encoder Projection
# =============================================================================


class EncoderProjection(nn.Module):
    """
    Projects encoder outputs to decoder hidden dimension.

    Transforms encoder hidden states from encoder_hidden_size (768 for
    ModernBERT-base) to decoder hidden_size (1280).

    Architecture:
        Linear(768, 1280) -> RMSNorm -> GELU -> Dropout

    Args:
        encoder_hidden_size: Encoder output dimension (768 for ModernBERT-base).
        decoder_hidden_size: Decoder input dimension (1280 default).
        dropout: Dropout probability.

    Example:
        >>> proj = EncoderProjection(768, 1280)
        >>> decoder_input = proj(encoder_output)  # (B, S, 768) -> (B, S, 1280)
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.norm = RMSNorm(decoder_hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project encoder outputs to decoder dimension.

        Args:
            encoder_hidden_states: Encoder outputs.
                Shape: (batch, seq_len, encoder_hidden_size)

        Returns:
            Projected tensor. Shape: (batch, seq_len, decoder_hidden_size)
        """
        hidden = self.proj(encoder_hidden_states)
        hidden = self.norm(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        return hidden


# =============================================================================
# Decoder Block (Issue 11.3.1)
# =============================================================================


class DecoderBlock(nn.Module):
    """
    Single decoder block with self-attention, cross-attention, and FFN.

    Architecture (Pre-norm style):
        1. Self-Attention: RMSNorm -> GQA -> Residual
        2. Cross-Attention: RMSNorm -> CrossAttn -> Residual
        3. FFN: RMSNorm -> (Dense or MoE) -> Residual

    Layer-dependent FFN:
        - Layers 0-1: Dense SwiGLU FFN (intermediate=3584)
        - Layers 2-7: MoE FFN (8 experts, top-2 routing)

    Args:
        config: DecoderMoEConfig with architecture parameters.
        layer_idx: Layer index (determines dense vs MoE FFN).

    Example:
        >>> config = DecoderMoEConfig()
        >>> block = DecoderBlock(config, layer_idx=0)  # Dense FFN
        >>> block = DecoderBlock(config, layer_idx=3)  # MoE FFN
        >>> output, aux_loss, past_kv = block(hidden, encoder_hidden, masks)
    """

    def __init__(self, config: "DecoderMoEConfig", layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-norm layers (LLaMA-style)
        self.self_attn_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Self-attention (GQA with RoPE and causal masking)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)

        # Cross-attention (full attention, no causal mask)
        self.cross_attn = CrossAttention(config, layer_idx=layer_idx)

        # FFN: Dense for early layers, MoE for later layers
        self.use_moe = layer_idx in config.moe_layers

        if self.use_moe:
            self.ffn = MoELayer(config, layer_idx=layer_idx)
        else:
            self.ffn = DenseSwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                dropout=config.hidden_dropout,
            )

        # Residual dropout
        self.resid_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, float, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass through the decoder block.

        Args:
            hidden_states: Input tensor. Shape: (batch, seq_len, hidden_size)
            encoder_hidden_states: Encoder outputs (projected).
                Shape: (batch, enc_seq_len, hidden_size)
            attention_mask: Causal attention mask for self-attention.
                Shape: (batch, 1, seq_len, kv_seq_len)
            encoder_attention_mask: Mask for encoder outputs in cross-attention.
                Shape: (batch, 1, 1, enc_seq_len)
            position_ids: Position indices for RoPE. Shape: (batch, seq_len)
            past_key_value: Cached KV from previous steps (for self-attention).
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of:
                - output: Block output. Shape: (batch, seq_len, hidden_size)
                - aux_loss: Auxiliary loss from MoE (0.0 for dense layers)
                - aux_loss_dict: Dict with individual loss components (empty for dense)
                - past_key_value: Updated cache if use_cache=True, else None
        """
        residual = hidden_states
        aux_loss = 0.0

        # ---- Self-Attention ----
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # ---- Cross-Attention ----
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # ---- FFN (Dense or MoE) ----
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)

        aux_loss_dict = {}
        if self.use_moe:
            hidden_states, aux_loss_dict = self.ffn(hidden_states)
            # Sum for backward compatibility, but return dict for component tracking
            aux_loss = aux_loss_dict.get("total", 0.0)
        else:
            hidden_states = self.ffn(hidden_states)
            aux_loss = 0.0

        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, aux_loss_dict, past_key_value

    def extra_repr(self) -> str:
        ffn_type = "MoE" if self.use_moe else "Dense"
        return f"layer_idx={self.layer_idx}, ffn_type={ffn_type}"


# =============================================================================
# CounterfactualDecoderHead (Issue 11.3.2)
# =============================================================================


class CounterfactualDecoderHead(BaseHead):
    """
    Counterfactual generation head using MoE decoder.

    This is the 13th head for the UltraBERT multi-task model, specifically
    designed for counterfactual text generation. It takes encoder outputs
    and generates counterfactual text autoregressively.

    Architecture (~420M params, ~237M active):
        - Encoder Projection: 768 -> 1280
        - Token Embedding: vocab_size x 1280 (tied with output)
        - 8 Decoder Blocks (2 dense + 6 MoE)
        - Final RMSNorm
        - LM Head (tied weights with embedding)

    Args:
        config: DecoderMoEConfig with decoder architecture parameters.
        encoder_hidden_size: Hidden size of the encoder (768 for ModernBERT-base).

    Example:
        >>> config = DecoderMoEConfig()
        >>> head = CounterfactualDecoderHead(config, encoder_hidden_size=768)
        >>> outputs = head(encoder_hidden, attention_mask, labels=target_ids)
        >>> loss = outputs["loss"]
        >>> logits = outputs["logits"]
        >>> aux_loss = outputs["aux_loss"]
    """

    # Class attribute for head registration
    head_name = "counterfactual"

    # Tied weights keys for safetensors saving
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: "DecoderMoEConfig",
        encoder_hidden_size: int = 768,
    ):
        # BaseHead requires hidden_size and num_labels
        # For decoder/generation head, we use hidden_size=config.hidden_size
        # and num_labels=vocab_size (for language modeling)
        super().__init__(
            hidden_size=config.hidden_size,
            num_labels=config.vocab_size,
            dropout=config.hidden_dropout,
            problem_type="seq2seq_lm",
        )

        self.config = config
        self.encoder_hidden_size = encoder_hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers

        # Encoder projection (768 -> 1280)
        self.encoder_proj = EncoderProjection(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=self.hidden_size,
            dropout=config.hidden_dropout,
        )

        # Token embedding (tied with lm_head)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(config, layer_idx=i) for i in range(self.num_layers)
        ])

        # Final layer norm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # LM head (weight tied with embed_tokens)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Auxiliary loss weights
        self.aux_loss_coef = config.load_balancing_loss_weight
        self.router_z_loss_coef = config.router_z_loss_weight

        # Initialize weights
        self._init_weights()

        num_dense = len(config.dense_layers)
        num_moe = len(config.moe_layers)
        logger.info(
            f"CounterfactualDecoderHead initialized: "
            f"{self.get_num_params() / 1e6:.1f}M params, "
            f"{self.num_layers} layers ({num_dense} dense, {num_moe} MoE)"
        )

    def _init_weights(self) -> None:
        """Initialize weights with scaled initialization."""
        # Embedding: normal init
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        # Encoder projection
        nn.init.normal_(self.encoder_proj.proj.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get total parameter count.

        Args:
            non_embedding: If True, exclude embedding parameters.

        Returns:
            Total parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract embedding (tied with lm_head, so count once)
            n_params -= self.embed_tokens.weight.numel()

        return n_params

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        # Alternate names for trainer compatibility
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for counterfactual generation.

        Args:
            hidden_states: Encoder outputs. Shape: (batch, enc_seq_len, encoder_hidden_size)
                Also accepts encoder_hidden_states as alias.
            attention_mask: Encoder attention mask. Shape: (batch, enc_seq_len)
                1 for valid tokens, 0 for padding. Also accepts encoder_attention_mask.
            labels: Target token IDs for teacher forcing.
                Shape: (batch, dec_seq_len). -100 for ignored positions.
            decoder_input_ids: Decoder input token IDs.
                Shape: (batch, dec_seq_len). If None, derived from labels.
            decoder_attention_mask: Decoder attention mask.
                Shape: (batch, dec_seq_len). If None, all ones.
            past_key_values: Cached KV pairs for each layer.
            use_cache: Whether to return updated KV cache.

        Returns:
            Dictionary containing:
                - "loss": Language modeling loss (if labels provided)
                - "logits": Output logits. Shape: (batch, dec_seq_len, vocab_size)
                - "aux_loss": Combined auxiliary loss from MoE layers
                - "past_key_values": KV cache (if use_cache=True)
        """
        # Handle alternate parameter names for trainer compatibility
        if hidden_states is None and encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
        if attention_mask is None and encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        if hidden_states is None:
            raise ValueError("hidden_states (or encoder_hidden_states) must be provided")

        # Project encoder outputs
        projected_hidden_states = self.encoder_proj(hidden_states)
        batch_size, enc_seq_len, _ = projected_hidden_states.shape

        # Prepare encoder attention mask for cross-attention
        # Convert from (batch, enc_seq_len) to (batch, 1, 1, enc_seq_len)
        if attention_mask is not None:
            cross_attention_mask = self._expand_mask(
                attention_mask, projected_hidden_states.dtype
            )
        else:
            cross_attention_mask = None

        # Prepare decoder inputs
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either decoder_input_ids or labels must be provided")
            # Shift labels right for decoder input (BOS prepending)
            decoder_input_ids = self._shift_right(labels)

        # Embed decoder tokens
        dec_seq_len = decoder_input_ids.shape[1]
        decoder_hidden = self.embed_tokens(decoder_input_ids)

        # Prepare decoder attention mask (causal)
        # Note: Causal mask is applied inside GQA, this is for padding
        if decoder_attention_mask is not None:
            # Expand to (batch, 1, dec_seq_len, dec_seq_len)
            causal_attention_mask = self._prepare_decoder_attention_mask(
                decoder_attention_mask, dec_seq_len, decoder_hidden.dtype
            )
        else:
            causal_attention_mask = None

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = [None] * self.num_layers

        # Forward through decoder layers
        total_aux_loss = 0.0
        total_lb_loss = 0.0
        total_z_loss = 0.0
        new_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None

            decoder_hidden, layer_aux_loss, layer_aux_dict, new_past_kv = layer(
                hidden_states=decoder_hidden,
                encoder_hidden_states=projected_hidden_states,
                attention_mask=causal_attention_mask,
                encoder_attention_mask=cross_attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            total_aux_loss += layer_aux_loss
            # Accumulate individual components for logging
            if layer_aux_dict:
                total_lb_loss += layer_aux_dict.get("load_balance_loss", 0.0)
                total_z_loss += layer_aux_dict.get("z_loss", 0.0)

            if use_cache:
                new_past_key_values.append(new_past_kv)

        # Final norm
        decoder_hidden = self.final_norm(decoder_hidden)

        # LM head
        logits = self.lm_head(decoder_hidden)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Scale auxiliary loss
        aux_loss_value = total_aux_loss * self.aux_loss_coef
        if isinstance(aux_loss_value, torch.Tensor):
            aux_loss_tensor = aux_loss_value.detach().clone().to(device=logits.device, dtype=logits.dtype)
        else:
            aux_loss_tensor = torch.tensor(aux_loss_value, device=logits.device, dtype=logits.dtype)

        # Prepare individual loss components for logging
        def to_float(v):
            if isinstance(v, torch.Tensor):
                return v.detach().item()
            return float(v) if v else 0.0

        aux_losses_dict = {
            "load_balance": to_float(total_lb_loss),
            "z_loss": to_float(total_z_loss),
        }

        # Build output dictionary
        output = {
            "logits": logits,
            "aux_loss": aux_loss_tensor,
            "aux_losses": aux_losses_dict,
        }

        if loss is not None:
            output["loss"] = loss + aux_loss_tensor

        if use_cache:
            output["past_key_values"] = new_past_key_values

        return output

    def _expand_mask(
        self,
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand attention mask from (batch, seq_len) to (batch, 1, 1, seq_len).

        Converts from 1/0 format to 0/-inf format for attention.
        """
        # (batch, seq_len) -> (batch, 1, 1, seq_len)
        expanded_mask = mask[:, None, None, :].to(dtype)

        # Convert: 1 -> 0 (attend), 0 -> -inf (mask)
        inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        return inverted_mask

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare combined causal + padding mask for decoder self-attention.
        """
        batch_size = attention_mask.shape[0]
        device = attention_mask.device

        # Create causal mask
        mask = torch.ones(target_length, target_length, device=device, dtype=dtype)
        causal_mask = torch.triu(mask, diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Combine with padding mask
        if attention_mask is not None:
            padding_mask = self._expand_mask(attention_mask, dtype)
            combined_mask = causal_mask + padding_mask
        else:
            combined_mask = causal_mask

        return combined_mask

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shift input right for decoder input (prepend BOS, remove last token).

        Uses bos_token_id (from config) as the start token.
        Also replaces -100 (ignore_index) with pad_token_id for embedding lookup.
        """
        bos_token_id = getattr(self.config, "bos_token_id", self.config.pad_token_id)
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[..., 1:] = input_ids[..., :-1].clone()
        shifted[..., 0] = bos_token_id

        # Replace -100 (ignore_index) with pad_token_id for embedding lookup
        shifted = shifted.masked_fill(shifted == -100, self.config.pad_token_id)

        return shifted

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generate counterfactual text autoregressively.

        Args:
            encoder_hidden_states: Encoder outputs.
                Shape: (batch, enc_seq_len, encoder_hidden_size)
            encoder_attention_mask: Encoder attention mask.
                Shape: (batch, enc_seq_len)
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_k: Keep only top k tokens for sampling. None = no filtering.
            top_p: Nucleus sampling probability. None = no filtering.
            eos_token_id: End of sequence token ID.
            pad_token_id: Padding token ID (used as BOS).

        Returns:
            Generated token IDs. Shape: (batch, generated_length)
        """
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        # Default token IDs
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1), pad_token_id, dtype=torch.long, device=device
        )

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # KV cache
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=generated if past_key_values is None else generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.get("past_key_values")

            # Get logits for last position
            logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Update finished mask
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences finished
            if finished.all():
                break

        return generated

    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filter logits to keep only top-k values."""
        if k > 0:
            values, _ = torch.topk(logits, min(k, logits.size(-1)))
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EncoderProjection",
    "DecoderBlock",
    "CounterfactualDecoderHead",
]

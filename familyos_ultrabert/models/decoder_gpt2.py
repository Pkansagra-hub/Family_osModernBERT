"""
GPT-2 based Decoder Head for Counterfactual Generation.

This module implements a GPT2DecoderHead that uses a pre-trained GPT-2 model
with prefix injection for encoder-decoder connection. Designed for edge deployment
where the MoE decoder is too computationally expensive.

Design Principles:
    1. Same interface as CounterfactualDecoderHead for drop-in replacement
    2. Pre-trained GPT-2 provides strong language prior (trained on 40GB WebText)
    3. Prefix injection: encoder outputs prepended to decoder input sequence
    4. Edge-friendly: GPT-2 Medium = ~710MB VRAM (vs ~1.7GB for MoE decoder)

Architecture:
    - Encoder: ModernBERT (768 hidden, frozen)
    - Projection: Linear(768 → 1024) to match GPT-2 hidden size
    - Decoder: GPT-2 Medium (1024 hidden, 16 heads, 24 layers)
    - Connection: Prefix injection (no cross-attention needed)

Usage:
    from familyos_ultrabert.models.decoder_gpt2 import GPT2DecoderHead
    from familyos_ultrabert.models.decoder_gpt2_config import GPT2DecoderConfig

    config = GPT2DecoderConfig()
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)

    # Forward pass (training)
    outputs = decoder(
        hidden_states=encoder_outputs,  # (batch, enc_seq, 768)
        attention_mask=encoder_mask,    # (batch, enc_seq)
        labels=target_ids,              # (batch, dec_seq)
    )
    loss = outputs["loss"]

    # Generation
    generated = decoder.generate(
        encoder_hidden_states=encoder_outputs,
        encoder_attention_mask=encoder_mask,
        max_new_tokens=128,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

from familyos_ultrabert.models.decoder_gpt2_config import GPT2DecoderConfig
from familyos_ultrabert.models.heads import BaseHead

logger = logging.getLogger(__name__)


class EncoderProjection(nn.Module):
    """
    Projects encoder hidden states to GPT-2 hidden size.

    Supports both simple linear projection and multi-layer MLP.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        projection_hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.projection_hidden_size = projection_hidden_size

        if num_layers == 1:
            self.projection = nn.Linear(encoder_hidden_size, projection_hidden_size)
        else:
            # Multi-layer MLP with GELU
            layers = []
            current_size = encoder_hidden_size
            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_size, projection_hidden_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_size = projection_hidden_size
            layers.append(nn.Linear(current_size, projection_hidden_size))
            self.projection = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project encoder hidden states to GPT-2 dimension."""
        return self.projection(hidden_states)


class GPT2DecoderHead(BaseHead):
    """
    GPT-2 based decoder head for counterfactual generation.

    Uses prefix injection to condition GPT-2 on encoder outputs.
    Designed as a drop-in replacement for CounterfactualDecoderHead.

    The key insight is that GPT-2 doesn't need cross-attention if we
    prepend encoder information as prefix tokens. This works because:
    1. Encoder outputs become the first N tokens of the decoder sequence
    2. GPT-2's causal attention naturally attends to all previous tokens
    3. Generated tokens attend to both prefix (encoder info) and previous output

    Args:
        config: GPT2DecoderConfig with model hyperparameters
        encoder_hidden_size: Size of encoder hidden states (default: 768 for ModernBERT)
    """

    head_name = "counterfactual"  # Same as MoE decoder for compatibility

    def __init__(
        self,
        config: GPT2DecoderConfig,
        encoder_hidden_size: int = 768,
    ):
        # BaseHead expects hidden_size for classification heads, we use it for projection
        super().__init__(
            hidden_size=config.projection_hidden_size,
            num_labels=config.vocab_size,
            dropout=config.dropout,
            problem_type="language_modeling",  # Not a standard BaseHead type
        )

        self.config = config
        self.encoder_hidden_size = encoder_hidden_size
        self.projection_hidden_size = config.projection_hidden_size
        self.vocab_size = config.vocab_size

        # Encoder projection (768 → 1024 for GPT-2 Medium)
        self.encoder_proj = EncoderProjection(
            encoder_hidden_size=encoder_hidden_size,
            projection_hidden_size=config.projection_hidden_size,
            num_layers=config.prefix_projection_layers,
            dropout=config.dropout,
        )

        # Load pre-trained GPT-2
        logger.info(f"Loading GPT-2 from {config.gpt2_model_name}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            config.gpt2_model_name,
            torch_dtype=torch.float32,  # Will be cast to bf16 by trainer
        )
        logger.info(f"Loaded GPT-2 with {sum(p.numel() for p in self.gpt2.parameters()):,} params")

        # Resize token embeddings to match target vocab size
        # GPT-2 has 50257 tokens, ModernBERT tokenizer has 50368
        original_vocab_size = self.gpt2.config.vocab_size
        if config.vocab_size != original_vocab_size:
            logger.info(
                f"Resizing GPT-2 embeddings: {original_vocab_size} -> {config.vocab_size}"
            )
            self.gpt2.resize_token_embeddings(config.vocab_size)

            # Initialize new token embeddings properly
            # New tokens (50257-50367) are random after resize - fix them
            self._initialize_new_token_embeddings(
                original_vocab_size=original_vocab_size,
                new_vocab_size=config.vocab_size,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.pad_token_id,
            )

        # Update GPT-2 config with our special token IDs
        self.gpt2.config.bos_token_id = config.bos_token_id
        self.gpt2.config.eos_token_id = config.eos_token_id
        self.gpt2.config.pad_token_id = config.pad_token_id

        # Get actual GPT-2 hidden size
        self.gpt2_hidden_size = self.gpt2.config.n_embd

        # Verify projection matches GPT-2
        if config.projection_hidden_size != self.gpt2_hidden_size:
            logger.warning(
                f"projection_hidden_size ({config.projection_hidden_size}) != "
                f"GPT-2 hidden_size ({self.gpt2_hidden_size}). "
                "Adding adapter layer."
            )
            self.adapter = nn.Linear(config.projection_hidden_size, self.gpt2_hidden_size)
        else:
            self.adapter = None

        # Freeze specified layers
        if config.freeze_layers > 0:
            self._freeze_layers(config.freeze_layers)

        # Store generation defaults
        self.generation_defaults = {
            "max_new_tokens": config.generation_max_length,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": config.pad_token_id,
            "eos_token_id": config.eos_token_id,
            "bos_token_id": config.bos_token_id,
        }

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N transformer layers of GPT-2."""
        frozen_params = 0

        # Freeze embeddings
        for param in self.gpt2.transformer.wte.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        for param in self.gpt2.transformer.wpe.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        # Freeze specified layers
        for i in range(min(num_layers, len(self.gpt2.transformer.h))):
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()

        logger.info(f"Froze {frozen_params:,} parameters in first {num_layers} GPT-2 layers")

    def _initialize_new_token_embeddings(
        self,
        original_vocab_size: int,
        new_vocab_size: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        """
        Initialize new token embeddings to match GPT-2's learned embedding scale.

        Problem: resize_token_embeddings() adds new tokens with random normal init
        (std ~0.02, norm ~2.1) while GPT-2's learned tokens have norm ~3.7.
        This 56% magnitude gap causes weak BOS/EOS signals.

        Solution: Initialize new tokens to mean of existing GPT-2 embeddings,
        with special handling for BOS (copy from endoftext) and PAD (zero).

        Args:
            original_vocab_size: Original GPT-2 vocab size (50257)
            new_vocab_size: New vocab size after resize (50368)
            bos_token_id: BOS token ID (50281)
            eos_token_id: EOS token ID (50282)
            pad_token_id: PAD token ID (50283)
        """
        with torch.no_grad():
            wte = self.gpt2.transformer.wte.weight

            # Compute statistics from original GPT-2 vocabulary
            original_embeddings = wte[:original_vocab_size]
            old_mean = original_embeddings.mean(dim=0)
            original_norm = original_embeddings.norm(dim=1).mean()

            logger.info(f"Original GPT-2 embeddings: mean norm={original_norm:.2f}")

            # Initialize all new tokens to the mean embedding
            # This gives them the correct magnitude from the start
            num_new = new_vocab_size - original_vocab_size
            wte[original_vocab_size:].copy_(
                old_mean.unsqueeze(0).expand(num_new, -1)
            )

            # Special token initialization:
            # BOS/EOS: Copy from GPT-2's <|endoftext|> token (ID 50256)
            # This token has learned start/stop semantics
            endoftext_embed = wte[50256].clone()
            wte[bos_token_id].copy_(endoftext_embed)
            wte[eos_token_id].copy_(endoftext_embed)

            # PAD: Zero vector (should not contribute to attention)
            wte[pad_token_id].zero_()

            # Verify new embeddings
            new_norm = wte[original_vocab_size:new_vocab_size].norm(dim=1).mean()
            logger.info(
                f"Initialized {num_new} new token embeddings: "
                f"BOS={bos_token_id}, EOS={eos_token_id}, PAD={pad_token_id}"
            )
            logger.info(f"New embeddings mean norm: {new_norm:.2f} (target: {original_norm:.2f})")

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training and inference.

        Uses prefix injection: encoder outputs are projected and prepended
        to the decoder input embeddings.

        Args:
            hidden_states: Encoder hidden states. Shape: (batch, enc_seq, encoder_hidden_size)
                Alternative name for encoder_hidden_states.
            attention_mask: Encoder attention mask. Shape: (batch, enc_seq)
                Alternative name for encoder_attention_mask.
            labels: Target token IDs. Shape: (batch, dec_seq). -100 for ignored positions.
            decoder_input_ids: Decoder input token IDs. Shape: (batch, dec_seq)
                If None, derived from labels by shifting right.
            decoder_attention_mask: Decoder attention mask. Shape: (batch, dec_seq)
            encoder_hidden_states: Encoder hidden states (alternative name).
            encoder_attention_mask: Encoder attention mask (alternative name).
            past_key_values: Cached KV pairs for generation.
            use_cache: Whether to return updated KV cache.

        Returns:
            Dictionary containing:
                - "loss": Language modeling loss (if labels provided)
                - "logits": Output logits. Shape: (batch, total_seq, vocab_size)
                - "aux_loss": Placeholder (0.0) for compatibility with MoE trainer
                - "past_key_values": KV cache (if use_cache=True)
        """
        # Handle alternate parameter names
        if hidden_states is None and encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
        if attention_mask is None and encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        if hidden_states is None:
            raise ValueError("hidden_states (or encoder_hidden_states) must be provided")

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Project encoder outputs to GPT-2 hidden size
        prefix_embeds = self.encoder_proj(hidden_states)  # (batch, enc_seq, proj_hidden)
        if self.adapter is not None:
            prefix_embeds = self.adapter(prefix_embeds)
        prefix_len = prefix_embeds.shape[1]

        # Prepare decoder inputs
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either decoder_input_ids or labels must be provided")
            # Shift labels right (prepend BOS, remove last token)
            decoder_input_ids = self._shift_right(labels)

        # Get decoder token embeddings
        decoder_embeds = self.gpt2.transformer.wte(decoder_input_ids)  # (batch, dec_seq, hidden)

        # Concatenate prefix and decoder embeddings
        # [prefix (encoder info) | decoder tokens]
        if past_key_values is None:
            # First forward pass: include prefix
            inputs_embeds = torch.cat([prefix_embeds, decoder_embeds], dim=1)

            # Create combined attention mask
            if attention_mask is not None:
                prefix_mask = attention_mask  # (batch, enc_seq)
            else:
                prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=torch.long)

            if decoder_attention_mask is not None:
                combined_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)
            else:
                decoder_mask = torch.ones_like(decoder_input_ids)
                combined_mask = torch.cat([prefix_mask, decoder_mask], dim=1)

            # Add position embeddings
            total_len = inputs_embeds.shape[1]
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            # Subsequent passes (generation): only new token, use KV cache
            inputs_embeds = decoder_embeds[:, -1:]
            combined_mask = None  # Attention mask handled by past_key_values
            position_ids = None

        # Forward through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        logits = outputs.logits  # (batch, total_seq, vocab_size)

        # Extract decoder-only logits (skip prefix positions)
        if past_key_values is None:
            decoder_logits = logits[:, prefix_len:, :]  # (batch, dec_seq, vocab_size)
        else:
            decoder_logits = logits  # Already just the new token

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = decoder_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.gpt2.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Build output dictionary (compatible with MoE decoder interface)
        output = {
            "logits": decoder_logits,
            "aux_loss": torch.tensor(0.0, device=device, dtype=logits.dtype),  # Placeholder for compatibility
            "aux_losses": {},  # No auxiliary losses
        }

        if loss is not None:
            output["loss"] = loss

        if use_cache:
            output["past_key_values"] = outputs.past_key_values

        return output

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shift input right for decoder input (prepend BOS, remove last token).

        Args:
            input_ids: Token IDs. Shape: (batch, seq_len)

        Returns:
            Shifted token IDs with BOS at position 0.
        """
        bos_token_id = self.config.bos_token_id
        pad_token_id = self.config.pad_token_id

        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[..., 1:] = input_ids[..., :-1].clone()
        shifted[..., 0] = bos_token_id

        # Replace -100 (ignore_index) with pad_token_id for embedding lookup
        shifted = shifted.masked_fill(shifted == -100, pad_token_id)

        return shifted

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generate counterfactual text autoregressively.

        Uses the same interface as CounterfactualDecoderHead.generate() for
        drop-in compatibility.

        Args:
            encoder_hidden_states: Encoder outputs. Shape: (batch, enc_seq, encoder_hidden_size)
            encoder_attention_mask: Encoder attention mask. Shape: (batch, enc_seq)
            max_new_tokens: Maximum tokens to generate. Default from config.
            temperature: Sampling temperature. Default from config.
            top_k: Top-k sampling. Default from config.
            top_p: Nucleus sampling probability. Default from config.
            repetition_penalty: Penalty for repeating tokens. Default from config.
            no_repeat_ngram_size: Block n-grams of this size from repeating.
            eos_token_id: End of sequence token. Default from config.
            pad_token_id: Padding token. Default from config.

        Returns:
            Generated token IDs. Shape: (batch, generated_length)
        """
        # Apply defaults from config
        max_new_tokens = max_new_tokens or self.generation_defaults["max_new_tokens"]
        temperature = temperature if temperature is not None else self.generation_defaults["temperature"]
        top_k = top_k if top_k is not None else self.generation_defaults["top_k"]
        top_p = top_p if top_p is not None else self.generation_defaults["top_p"]
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.generation_defaults["repetition_penalty"]
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_defaults["eos_token_id"]
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_defaults["pad_token_id"]
        bos_token_id = self.generation_defaults["bos_token_id"]

        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        # Project encoder outputs
        prefix_embeds = self.encoder_proj(encoder_hidden_states)
        if self.adapter is not None:
            prefix_embeds = self.adapter(prefix_embeds)
        prefix_len = prefix_embeds.shape[1]

        # Prepare prefix attention mask
        if encoder_attention_mask is not None:
            prefix_mask = encoder_attention_mask
        else:
            prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=torch.long)

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize KV cache
        past_key_values = None

        for step in range(max_new_tokens):
            # Get current token embedding
            if past_key_values is None:
                # First step: include prefix
                token_embeds = self.gpt2.transformer.wte(generated)
                inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                attention_mask = torch.cat([
                    prefix_mask,
                    torch.ones_like(generated)
                ], dim=1)
                position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                # Subsequent steps: only new token
                token_embeds = self.gpt2.transformer.wte(generated[:, -1:])
                inputs_embeds = token_embeds
                attention_mask = None  # Use cached mask
                # Position is prefix_len + number of generated tokens
                position_ids = torch.full(
                    (batch_size, 1),
                    prefix_len + step,
                    device=device,
                    dtype=torch.long
                )

            # Forward through GPT-2
            outputs = self.gpt2(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            # Apply no-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                logits = self._block_repeat_ngrams(logits, generated, no_repeat_ngram_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)

            # Sample next token
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

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeating tokens."""
        for batch_idx in range(logits.shape[0]):
            unique_tokens = generated[batch_idx].unique()
            for token_id in unique_tokens:
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] = logits[batch_idx, token_id] / penalty
                else:
                    logits[batch_idx, token_id] = logits[batch_idx, token_id] * penalty
        return logits

    def _block_repeat_ngrams(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Block n-grams that have already appeared from being generated again."""
        batch_size = logits.shape[0]
        seq_len = generated.shape[1]

        for batch_idx in range(batch_size):
            if seq_len < n:
                continue

            prefix = tuple(generated[batch_idx, -(n - 1):].tolist())
            banned_tokens = set()

            for i in range(seq_len - n + 1):
                ngram_prefix = tuple(generated[batch_idx, i : i + n - 1].tolist())
                if ngram_prefix == prefix:
                    next_token = generated[batch_idx, i + n - 1].item()
                    banned_tokens.add(next_token)

            for token_id in banned_tokens:
                logits[batch_idx, token_id] = float("-inf")

        return logits


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "GPT2DecoderHead",
    "EncoderProjection",
]

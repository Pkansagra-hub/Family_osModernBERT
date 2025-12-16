"""
GPT-2 with Encoder Prefix Injection for Counterfactual Generation.

This module wraps GPT-2 to accept encoder hidden states as a prefix,
enabling encoder-decoder style generation without modifying GPT-2 architecture.

Technique: Prefix Injection (used in Flamingo, BLIP-2)
- Encoder output is projected to GPT-2's embedding space
- Projected tokens are prepended to GPT-2's input
- GPT-2 generates conditioned on the encoder prefix

Usage:
    from modeling_studio.models.gpt2_encoder_decoder import GPT2EncoderDecoder

    model = GPT2EncoderDecoder.from_pretrained('gpt2-medium')

    # Generate with encoder conditioning
    output = model.generate(
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=attention_mask,
        max_new_tokens=100,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class GPT2EncoderDecoderConfig:
    """Configuration for GPT2 with encoder prefix."""

    # GPT-2 model to use
    gpt2_model_name: str = "gpt2-medium"

    # Encoder dimension (ModernBERT-base = 768)
    encoder_hidden_size: int = 768

    # Number of prefix tokens to use (compresses encoder sequence)
    num_prefix_tokens: int = 32

    # Whether to freeze GPT-2 weights initially
    freeze_gpt2: bool = False

    # Dropout for projection
    projection_dropout: float = 0.1


class EncoderProjection(nn.Module):
    """Projects encoder hidden states to GPT-2 prefix tokens."""

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        num_prefix_tokens: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.decoder_hidden_size = decoder_hidden_size

        # Project encoder dim to decoder dim
        self.encoder_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        # Compress sequence to fixed number of prefix tokens
        # Using a simple learned query mechanism
        self.prefix_queries = nn.Parameter(
            torch.randn(num_prefix_tokens, decoder_hidden_size) * 0.02
        )

        # Cross-attention to compress encoder sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project encoder hidden states to prefix tokens.

        Args:
            encoder_hidden_states: [batch, seq, encoder_hidden]
            encoder_attention_mask: [batch, seq] - 1 for valid, 0 for padding

        Returns:
            prefix_embeddings: [batch, num_prefix_tokens, decoder_hidden]
        """
        batch_size = encoder_hidden_states.shape[0]

        # Project encoder to decoder dimension
        encoder_proj = self.encoder_proj(encoder_hidden_states)  # [B, S, D]

        # Expand queries for batch
        queries = self.prefix_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, D]

        # Create attention mask for cross-attention
        key_padding_mask = None
        if encoder_attention_mask is not None:
            # MultiheadAttention expects True for positions to mask
            key_padding_mask = encoder_attention_mask == 0

        # Cross-attention: queries attend to encoder
        prefix_embeddings, _ = self.cross_attn(
            query=queries,
            key=encoder_proj,
            value=encoder_proj,
            key_padding_mask=key_padding_mask,
        )

        # Residual + norm
        prefix_embeddings = self.norm(queries + self.dropout(prefix_embeddings))

        return prefix_embeddings


class GPT2EncoderDecoder(nn.Module):
    """
    GPT-2 with encoder prefix injection for encoder-decoder generation.

    Architecture:
        1. Encoder (external, e.g., ModernBERT) produces hidden states
        2. EncoderProjection compresses encoder output to prefix tokens
        3. Prefix tokens are prepended to GPT-2's input embeddings
        4. GPT-2 generates autoregressively, conditioned on prefix
    """

    def __init__(self, config: GPT2EncoderDecoderConfig):
        super().__init__()
        self.config = config

        # Load pre-trained GPT-2
        logger.info(f"Loading GPT-2: {config.gpt2_model_name}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_name)
        self.gpt2_config = self.gpt2.config

        # Encoder projection
        self.encoder_projection = EncoderProjection(
            encoder_hidden_size=config.encoder_hidden_size,
            decoder_hidden_size=self.gpt2_config.n_embd,
            num_prefix_tokens=config.num_prefix_tokens,
            dropout=config.projection_dropout,
        )

        # Optionally freeze GPT-2
        if config.freeze_gpt2:
            logger.info("Freezing GPT-2 weights")
            for param in self.gpt2.parameters():
                param.requires_grad = False

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

    @classmethod
    def from_pretrained(
        cls,
        gpt2_model_name: str = "gpt2-medium",
        encoder_hidden_size: int = 768,
        num_prefix_tokens: int = 32,
        freeze_gpt2: bool = False,
    ) -> "GPT2EncoderDecoder":
        """Create model from pre-trained GPT-2."""
        config = GPT2EncoderDecoderConfig(
            gpt2_model_name=gpt2_model_name,
            encoder_hidden_size=encoder_hidden_size,
            num_prefix_tokens=num_prefix_tokens,
            freeze_gpt2=freeze_gpt2,
        )
        return cls(config)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        """
        Forward pass with encoder conditioning.

        Args:
            encoder_hidden_states: [batch, encoder_seq, 768] from encoder
            encoder_attention_mask: [batch, encoder_seq]
            input_ids: [batch, decoder_seq] decoder input tokens
            attention_mask: [batch, decoder_seq] decoder attention mask
            labels: [batch, decoder_seq] for training (shifted input_ids)
            past_key_values: cached keys/values for generation
            use_cache: whether to return past_key_values

        Returns:
            CausalLMOutput with loss (if labels provided) and logits
        """
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        # Get prefix embeddings from encoder
        if past_key_values is None:
            # First forward pass: compute prefix
            prefix_embeddings = self.encoder_projection(
                encoder_hidden_states,
                encoder_attention_mask,
            )  # [B, P, D]
        else:
            # Subsequent passes: prefix already in cache
            prefix_embeddings = None

        # Get decoder input embeddings
        if input_ids is not None:
            inputs_embeds = self.gpt2.transformer.wte(input_ids)  # [B, S, D]
        else:
            inputs_embeds = None

        # Concatenate prefix + decoder embeddings
        if prefix_embeddings is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([prefix_embeddings, inputs_embeds], dim=1)

            # Extend attention mask for prefix
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    batch_size, self.config.num_prefix_tokens,
                    device=device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Extend labels with -100 for prefix (don't compute loss on prefix)
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.config.num_prefix_tokens),
                    fill_value=-100,
                    device=device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
        elif prefix_embeddings is not None:
            inputs_embeds = prefix_embeddings
            attention_mask = torch.ones(
                batch_size, self.config.num_prefix_tokens,
                device=device
            )

        # Forward through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text conditioned on encoder output.

        Args:
            encoder_hidden_states: [batch, seq, 768] from encoder
            encoder_attention_mask: [batch, seq]
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            repetition_penalty: penalty for repeated tokens
            no_repeat_ngram_size: block repeated n-grams
            eos_token_id: end of sequence token
            pad_token_id: padding token

        Returns:
            generated_ids: [batch, seq] generated token ids
        """
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        # Get prefix embeddings
        prefix_embeddings = self.encoder_projection(
            encoder_hidden_states,
            encoder_attention_mask,
        )

        # Start with just prefix
        inputs_embeds = prefix_embeddings
        attention_mask = torch.ones(
            batch_size, self.config.num_prefix_tokens,
            device=device
        )

        # Track generated tokens
        generated_ids = []
        past_key_values = None

        # Use GPT-2's default tokens if not provided
        if eos_token_id is None:
            eos_token_id = self.gpt2_config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.gpt2_config.eos_token_id  # GPT-2 uses eos as pad

        # Generate tokens autoregressively
        for step in range(max_new_tokens):
            # Forward pass
            outputs = self.gpt2(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # [B, vocab]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for i in range(batch_size):
                    for token_id in set(generated_ids):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Apply n-gram blocking
            if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size - 1:
                for i in range(batch_size):
                    # Get recent n-1 tokens
                    recent = generated_ids[-(no_repeat_ngram_size - 1):]
                    # Find tokens that would complete a repeated n-gram
                    for j in range(len(generated_ids) - no_repeat_ngram_size + 1):
                        if generated_ids[j:j + no_repeat_ngram_size - 1] == recent:
                            # Block the token that would complete the repeat
                            if j + no_repeat_ngram_size - 1 < len(generated_ids):
                                blocked_token = generated_ids[j + no_repeat_ngram_size - 1]
                                logits[i, blocked_token] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated_ids.append(next_token.item() if batch_size == 1 else next_token)

            # Check for EOS
            if batch_size == 1 and next_token.item() == eos_token_id:
                break

            # Prepare next input
            past_key_values = outputs.past_key_values
            inputs_embeds = self.gpt2.transformer.wte(next_token.unsqueeze(-1))
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, device=device)
            ], dim=1)

        # Convert to tensor
        if batch_size == 1:
            return torch.tensor([generated_ids], device=device)
        else:
            return torch.stack(generated_ids, dim=1)

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = {
            "gpt2_model_name": self.config.gpt2_model_name,
            "encoder_hidden_size": self.config.encoder_hidden_size,
            "num_prefix_tokens": self.config.num_prefix_tokens,
            "freeze_gpt2": self.config.freeze_gpt2,
            "projection_dropout": self.config.projection_dropout,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        logger.info(f"Saved model to {save_directory}")

    @classmethod
    def load_pretrained(cls, load_directory: str, device: str = "cpu") -> "GPT2EncoderDecoder":
        """Load model from directory."""
        import os
        import json

        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config_dict = json.load(f)

        config = GPT2EncoderDecoderConfig(**config_dict)
        model = cls(config)

        # Load weights
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location=device
        )
        model.load_state_dict(state_dict)

        logger.info(f"Loaded model from {load_directory}")
        return model

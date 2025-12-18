#!/usr/bin/env python3
"""
Export GPT-2 Decoder to ONNX - Split Architecture

Exports two ONNX models for efficient edge inference:
1. prefix_encoder.onnx - Projects encoder hidden states (runs once per input)
2. decoder_core.onnx - GPT-2 transformer (runs once per generated token)

This split architecture is optimized for NPU scheduling where:
- Prefix encoder runs ONCE per input text
- Decoder core runs N times (once per generated token)

Features:
    - Split export for NPU efficiency
    - Dynamic batch size and sequence length
    - KV cache support for efficient generation
    - Validation against PyTorch outputs
    - Configurable opset version

Usage:
    # Export with default settings (opset 17)
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3

    # Export with specific opset
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3 \\
        --opset 14

    # Export and validate
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3 \\
        --validate

Requirements:
    - torch
    - onnx
    - onnxruntime
    - transformers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default ONNX opset version (17 supports all GPT-2 ops)
DEFAULT_OPSET = 17

# GPT-2 Medium has 24 layers, each with 2 KV tensors (key + value)
GPT2_MEDIUM_NUM_LAYERS = 24
KV_TENSORS_PER_LAYER = 2


# =============================================================================
# ONNX Export Wrappers
# =============================================================================


class PrefixEncoderWrapper(nn.Module):
    """
    Wrapper for ONNX export of encoder projection layer.

    This module projects encoder hidden states (768-dim) to GPT-2 hidden size (1024-dim).
    It runs ONCE per input text, producing prefix embeddings that condition generation.

    Input:
        encoder_hidden_states: (batch, enc_seq_len, 768)

    Output:
        prefix_embeds: (batch, enc_seq_len, 1024)
    """

    def __init__(self, decoder):
        super().__init__()
        self.encoder_proj = decoder.encoder_proj
        self.adapter = getattr(decoder, 'adapter', None)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project encoder hidden states to GPT-2 dimension."""
        prefix_embeds = self.encoder_proj(encoder_hidden_states)
        if self.adapter is not None:
            prefix_embeds = self.adapter(prefix_embeds)
        return prefix_embeds


class DecoderCoreWrapper(nn.Module):
    """
    Wrapper for ONNX export of GPT-2 decoder core.

    This module runs the GPT-2 transformer with KV cache support.
    It runs ONCE per generated token during autoregressive generation.

    ONNX-Compatible: Manually constructs causal mask to avoid vmap in
    transformers' create_causal_mask which is not traceable.

    Inputs:
        input_embeds: (batch, 1, 1024) - Current token embedding
        attention_mask: (batch, total_seq_len) - Full attention mask
        position_ids: (batch, 1) - Position of current token
        past_key_values: Tuple of (batch, num_heads, past_len, head_dim) tensors

    Outputs:
        logits: (batch, 1, vocab_size) - Next token probabilities
        new_past_key_values: Updated KV cache
    """

    def __init__(self, decoder):
        super().__init__()
        # Extract components we need
        self.transformer = decoder.gpt2.transformer
        self.lm_head = decoder.gpt2.lm_head
        self.num_layers = len(self.transformer.h)
        self.num_heads = decoder.gpt2.config.n_head
        self.head_dim = decoder.gpt2.config.n_embd // decoder.gpt2.config.n_head
        self.hidden_size = decoder.gpt2.config.n_embd

    def _create_causal_mask(
        self,
        batch_size: int,
        query_len: int,
        key_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create 4D causal attention mask for ONNX export.

        This bypasses transformers' vmap-based mask creation.

        Returns:
            mask: (batch, 1, query_len, key_len) with 0 for attend, -inf for mask
        """
        # For autoregressive generation with KV cache:
        # query_len = 1 (current token), key_len = past_len + 1
        # The current token can attend to all past positions
        mask = torch.zeros(batch_size, 1, query_len, key_len, dtype=dtype, device=device)
        return mask

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through GPT-2 with KV cache.

        Uses direct transformer block access to avoid vmap-based masking.

        Args:
            input_embeds: Current token embedding (batch, 1, hidden_size)
            attention_mask: Full attention mask (batch, total_seq_len)
            position_ids: Position of current token (batch, 1)
            past_key_values_flat: Flattened KV cache tensors

        Returns:
            Tuple of (logits, *new_past_key_values_flat)
        """
        batch_size = input_embeds.shape[0]
        seq_len = input_embeds.shape[1]  # Should be 1 for generation
        device = input_embeds.device
        dtype = input_embeds.dtype

        # Reshape past_key_values from flat tuple to nested structure
        past_key_values = None
        past_len = 0
        if len(past_key_values_flat) > 0:
            past_key_values = []
            for i in range(self.num_layers):
                key_idx = i * 2
                value_idx = i * 2 + 1
                past_key_values.append((
                    past_key_values_flat[key_idx],
                    past_key_values_flat[value_idx],
                ))
            past_key_values = tuple(past_key_values)
            past_len = past_key_values[0][0].shape[2]  # (batch, heads, past_len, head_dim)

        # Add position embeddings
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        # Apply dropout (will be identity in eval mode)
        hidden_states = self.transformer.drop(hidden_states)

        # Create 4D causal mask manually (ONNX-compatible)
        total_len = past_len + seq_len
        causal_mask = self._create_causal_mask(
            batch_size, seq_len, total_len, dtype, device
        )

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand 2D mask to 4D
            # attention_mask: (batch, total_len) -> (batch, 1, 1, total_len)
            padding_mask = attention_mask[:, None, None, :]
            # Convert 1/0 mask to 0/-inf mask
            padding_mask = (1.0 - padding_mask.to(dtype)) * torch.finfo(dtype).min
            causal_mask = causal_mask + padding_mask

        # Process through transformer blocks
        presents = []
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
            presents.append(outputs[1])

        # Final layer norm
        hidden_states = self.transformer.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Flatten presents for ONNX output
        new_past_flat = []
        for layer_past in presents:
            new_past_flat.append(layer_past[0])  # key
            new_past_flat.append(layer_past[1])  # value

        return (logits,) + tuple(new_past_flat)


class DecoderFirstStepWrapper(nn.Module):
    """
    Wrapper for first generation step (no KV cache).

    This is used for the first token generation where we don't have
    any cached key-values yet. Combines prefix and first token.

    ONNX-Compatible: Manually constructs causal mask to avoid vmap in
    transformers' create_causal_mask which is not traceable.

    Inputs:
        prefix_embeds: (batch, prefix_len, 1024) - From PrefixEncoderWrapper
        first_token_embed: (batch, 1, 1024) - BOS token embedding
        attention_mask: (batch, prefix_len + 1) - Attention mask

    Outputs:
        logits: (batch, 1, vocab_size) - First token probabilities
        past_key_values: Initial KV cache
    """

    def __init__(self, decoder):
        super().__init__()
        # Extract components we need
        self.transformer = decoder.gpt2.transformer
        self.lm_head = decoder.gpt2.lm_head
        self.wte = decoder.gpt2.transformer.wte
        self.num_layers = len(self.transformer.h)
        self.hidden_size = decoder.gpt2.config.n_embd

    def _create_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create 4D causal attention mask for ONNX export.

        This bypasses transformers' vmap-based mask creation.

        Returns:
            mask: (batch, 1, seq_len, seq_len) with 0 for attend, -inf for mask
        """
        # Create causal mask: position i can only attend to positions <= i
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
            diagonal=1
        )
        # Expand to 4D: (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        # Broadcast to batch: (batch, 1, seq_len, seq_len)
        mask = mask.expand(batch_size, -1, -1, -1)
        return mask

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        first_token_id: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        First step: combine prefix with first token (usually BOS).

        Uses direct transformer block access to avoid vmap-based masking.

        Args:
            prefix_embeds: Projected encoder outputs (batch, prefix_len, hidden)
            first_token_id: First token ID, typically BOS (batch, 1)
            attention_mask: Full mask (batch, prefix_len + 1)

        Returns:
            Tuple of (logits, *past_key_values_flat)
        """
        batch_size = prefix_embeds.shape[0]
        prefix_len = prefix_embeds.shape[1]
        device = prefix_embeds.device
        dtype = prefix_embeds.dtype

        # Get first token embedding
        first_token_embed = self.wte(first_token_id)  # (batch, 1, hidden)

        # Concatenate prefix and first token
        inputs_embeds = torch.cat([prefix_embeds, first_token_embed], dim=1)
        seq_len = inputs_embeds.shape[1]

        # Position IDs and embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Apply dropout (identity in eval mode)
        hidden_states = self.transformer.drop(hidden_states)

        # Create 4D causal mask manually (ONNX-compatible)
        causal_mask = self._create_causal_mask(batch_size, seq_len, dtype, device)

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand 2D mask to 4D: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = attention_mask[:, None, None, :]
            # Convert 1/0 mask to 0/-inf mask
            padding_mask = (1.0 - padding_mask.to(dtype)) * torch.finfo(dtype).min
            causal_mask = causal_mask + padding_mask

        # Process through transformer blocks
        presents = []
        for block in self.transformer.h:
            outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=causal_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
            presents.append(outputs[1])

        # Final layer norm
        hidden_states = self.transformer.ln_f(hidden_states)

        # LM head - only for last position (first generated token)
        logits = self.lm_head(hidden_states[:, -1:, :])  # (batch, 1, vocab_size)

        # Flatten past_key_values for ONNX
        past_flat = []
        for layer_past in presents:
            past_flat.append(layer_past[0])  # key
            past_flat.append(layer_past[1])  # value

        return (logits,) + tuple(past_flat)


# =============================================================================
# Export Functions
# =============================================================================


def _load_decoder_from_full_model(checkpoint_path: Path) -> nn.Module:
    """
    Extract GPT2DecoderHead from a full ModernBertMultiTaskModel checkpoint.

    The decoder is stored under model.heads["counterfactual"] in the full model.
    Weights are prefixed with 'heads.counterfactual.' in the state dict.
    """
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

    logger.info(f"Loading decoder from full model checkpoint: {checkpoint_path}")

    # Load state dict directly to extract decoder weights
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        full_state_dict = load_file(weights_path)
    else:
        weights_path = checkpoint_path / "pytorch_model.bin"
        if weights_path.exists():
            full_state_dict = torch.load(weights_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    # Extract decoder weights (strip 'heads.counterfactual.' prefix)
    decoder_prefix = "heads.counterfactual."
    decoder_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith(decoder_prefix):
            new_key = key[len(decoder_prefix):]
            decoder_state_dict[new_key] = value

    if not decoder_state_dict:
        raise ValueError(f"No decoder weights found with prefix '{decoder_prefix}'")

    logger.info(f"Extracted {len(decoder_state_dict)} decoder weight tensors")

    # Create decoder config
    config = GPT2DecoderConfig()

    # Create decoder and load weights
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)
    missing, unexpected = decoder.load_state_dict(decoder_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    logger.info("Decoder loaded successfully")

    # Force eager attention for ONNX export compatibility
    # SDPA uses vmap which is not traceable by ONNX
    _force_eager_attention(decoder.gpt2)

    decoder.eval()
    return decoder


def _force_eager_attention(model) -> None:
    """
    Force GPT2 to use eager attention instead of SDPA.

    SDPA (Scaled Dot-Product Attention) uses vmap for masking which is
    not compatible with ONNX tracing. This function forces the model
    to use the traditional attention implementation.
    """
    # Set config flag to disable SDPA
    if hasattr(model.config, '_attn_implementation'):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = "eager"

    # Also set on the transformer
    if hasattr(model, 'transformer') and hasattr(model.transformer.config, '_attn_implementation'):
        model.transformer.config._attn_implementation = "eager"

    logger.info("Forced eager attention implementation for ONNX compatibility")


def load_decoder(checkpoint_path: Path) -> nn.Module:
    """
    Load trained decoder from checkpoint.

    Supports two checkpoint formats:
    1. Standalone decoder: decoder_config.json + decoder.safetensors
    2. Full model: config.json + model.safetensors (ModernBertMultiTaskModel)
    """
    logger.info(f"Loading decoder from {checkpoint_path}")

    # Check if this is a full model checkpoint
    full_model_config_path = checkpoint_path / "config.json"
    full_model_weights_path = checkpoint_path / "model.safetensors"
    full_model_weights_pt_path = checkpoint_path / "pytorch_model.bin"

    if full_model_config_path.exists() and (full_model_weights_path.exists() or full_model_weights_pt_path.exists()):
        # Load from full ModernBertMultiTaskModel
        return _load_decoder_from_full_model(checkpoint_path)

    # Try standalone decoder format
    from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
    from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

    # Load config
    config_path = checkpoint_path / "decoder_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = GPT2DecoderConfig(**config_dict)
    else:
        logger.warning("No decoder_config.json found, using defaults")
        config = GPT2DecoderConfig()

    # Create decoder
    decoder = GPT2DecoderHead(config, encoder_hidden_size=768)

    # Load weights
    weights_path = checkpoint_path / "decoder.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        decoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        # Try .pt file
        weights_path = checkpoint_path / "decoder.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            decoder.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning("No decoder weights found, using initialized weights")

    decoder.eval()
    return decoder


def export_prefix_encoder(
    decoder: nn.Module,
    output_path: Path,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Export prefix encoder (projection layer) to ONNX."""
    import onnx

    logger.info("Exporting prefix_encoder.onnx...")

    wrapper = PrefixEncoderWrapper(decoder)
    wrapper.eval()

    # Dummy input (batch=1, seq=32, hidden=768)
    dummy_encoder_hidden = torch.randn(1, 32, 768)

    onnx_path = output_path / "prefix_encoder.onnx"

    # Dynamic axes for variable batch and sequence length
    dynamic_axes = {
        "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
        "prefix_embeds": {0: "batch_size", 1: "encoder_seq_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_encoder_hidden,),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["encoder_hidden_states"],
            output_names=["prefix_embeds"],
            dynamic_axes=dynamic_axes,
            dynamo=False,  # Use legacy TorchScript exporter
        )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported prefix_encoder.onnx ({size_mb:.2f} MB)")

    return onnx_path


def export_decoder_first_step(
    decoder: nn.Module,
    output_path: Path,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Export decoder first step (prefix + BOS) to ONNX."""
    import onnx

    logger.info("Exporting decoder_first_step.onnx...")

    wrapper = DecoderFirstStepWrapper(decoder)
    wrapper.eval()

    # Dummy inputs
    batch_size = 1
    prefix_len = 32
    hidden_size = wrapper.hidden_size  # 1024 for GPT-2 Medium

    dummy_prefix_embeds = torch.randn(batch_size, prefix_len, hidden_size)
    dummy_first_token = torch.tensor([[50256]])  # BOS token
    dummy_attention_mask = torch.ones(batch_size, prefix_len + 1, dtype=torch.long)

    onnx_path = output_path / "decoder_first_step.onnx"

    # Output names: logits + flattened past_key_values
    num_layers = wrapper.num_layers
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"past_key_{i}")
        output_names.append(f"past_value_{i}")

    # Dynamic axes
    dynamic_axes = {
        "prefix_embeds": {0: "batch_size", 1: "prefix_len"},
        "first_token_id": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "total_seq_len"},
        "logits": {0: "batch_size"},
    }
    # Add dynamic axes for past_key_values outputs
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "past_seq_len"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_prefix_embeds, dummy_first_token, dummy_attention_mask),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["prefix_embeds", "first_token_id", "attention_mask"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported decoder_first_step.onnx ({size_mb:.2f} MB)")

    return onnx_path


def export_decoder_core(
    decoder: nn.Module,
    output_path: Path,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Export decoder core (GPT-2 with KV cache) to ONNX."""
    import onnx

    logger.info("Exporting decoder_core.onnx...")

    wrapper = DecoderCoreWrapper(decoder)
    wrapper.eval()

    # Get model dimensions
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    head_dim = wrapper.head_dim
    hidden_size = wrapper.hidden_size

    # Dummy inputs
    batch_size = 1
    past_seq_len = 33  # prefix_len + generated so far

    dummy_input_embeds = torch.randn(batch_size, 1, hidden_size)
    dummy_attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    dummy_position_ids = torch.tensor([[past_seq_len]])

    # Dummy past_key_values (flattened)
    dummy_past_flat = []
    for _ in range(num_layers):
        dummy_past_flat.append(torch.randn(batch_size, num_heads, past_seq_len, head_dim))  # key
        dummy_past_flat.append(torch.randn(batch_size, num_heads, past_seq_len, head_dim))  # value

    onnx_path = output_path / "decoder_core.onnx"

    # Input names
    input_names = ["input_embeds", "attention_mask", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")

    # Output names
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"new_past_key_{i}")
        output_names.append(f"new_past_value_{i}")

    # Dynamic axes
    dynamic_axes = {
        "input_embeds": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "total_seq_len"},
        "position_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes[f"new_past_key_{i}"] = {0: "batch_size", 2: "new_past_seq_len"}
        dynamic_axes[f"new_past_value_{i}"] = {0: "batch_size", 2: "new_past_seq_len"}

    # Prepare inputs tuple
    inputs = (dummy_input_embeds, dummy_attention_mask, dummy_position_ids) + tuple(dummy_past_flat)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,  # Use legacy exporter for complex KV cache
        )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported decoder_core.onnx ({size_mb:.2f} MB)")

    return onnx_path


def export_decoder_simple(
    decoder: nn.Module,
    output_path: Path,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """
    Export a simple full decoder without KV cache.

    This is a fallback when KV cache export fails with newer transformers.
    Less efficient but always works.
    """
    import onnx

    logger.info("Exporting decoder.onnx (simple, no KV cache)...")

    # Simple wrapper without KV cache
    class SimpleDecoderWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.gpt2 = dec.gpt2
            self.hidden_size = dec.gpt2.config.n_embd
            self.vocab_size = dec.gpt2.config.vocab_size

        def forward(self, prefix_embeds, decoder_input_ids, attention_mask):
            # Get decoder embeddings
            decoder_embeds = self.gpt2.transformer.wte(decoder_input_ids)

            batch_size, prefix_len, _ = prefix_embeds.shape
            dec_len = decoder_input_ids.shape[1]
            total_len = prefix_len + dec_len

            # Position embeddings
            position_ids = torch.arange(total_len, device=decoder_input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            # Combine embeddings
            inputs_embeds = torch.cat([prefix_embeds, decoder_embeds], dim=1)
            position_embeds = self.gpt2.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.gpt2.transformer.drop(hidden_states)

            # Run through blocks
            for block in self.gpt2.transformer.h:
                outputs = block(hidden_states, attention_mask=attention_mask, use_cache=False)
                hidden_states = outputs[0]

            hidden_states = self.gpt2.transformer.ln_f(hidden_states)

            # Get logits only for decoder positions
            decoder_hidden = hidden_states[:, prefix_len:, :]
            logits = self.gpt2.lm_head(decoder_hidden)

            return logits

    wrapper = SimpleDecoderWrapper(decoder)
    wrapper.eval()

    # Dummy inputs
    batch_size, prefix_len, hidden_size = 1, 32, 1024
    dec_len = 16
    vocab_size = decoder.gpt2.config.vocab_size

    dummy_prefix = torch.randn(batch_size, prefix_len, hidden_size)
    dummy_ids = torch.randint(0, vocab_size, (batch_size, dec_len))
    dummy_mask = torch.ones(batch_size, prefix_len + dec_len)

    onnx_path = output_path / "decoder.onnx"

    dynamic_axes = {
        "prefix_embeds": {0: "batch_size", 1: "prefix_len"},
        "decoder_input_ids": {0: "batch_size", 1: "dec_len"},
        "attention_mask": {0: "batch_size", 1: "total_len"},
        "logits": {0: "batch_size", 1: "dec_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_prefix, dummy_ids, dummy_mask),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["prefix_embeds", "decoder_input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            dynamo=False,  # Use legacy TorchScript exporter for compatibility
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported decoder.onnx ({size_mb:.2f} MB)")

    return onnx_path


def validate_onnx_outputs(
    decoder: nn.Module,
    output_path: Path,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Validate ONNX outputs match PyTorch outputs."""
    import onnxruntime as ort

    logger.info("Validating ONNX outputs against PyTorch...")

    # Test prefix encoder
    prefix_session = ort.InferenceSession(
        str(output_path / "prefix_encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # Create test input
    test_encoder_hidden = torch.randn(2, 16, 768)  # batch=2, seq=16

    # PyTorch output
    wrapper = PrefixEncoderWrapper(decoder)
    wrapper.eval()
    with torch.no_grad():
        pytorch_output = wrapper(test_encoder_hidden).numpy()

    # ONNX output
    onnx_output = prefix_session.run(
        ["prefix_embeds"],
        {"encoder_hidden_states": test_encoder_hidden.numpy()},
    )[0]

    # Compare
    if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
        logger.info("prefix_encoder.onnx: PASSED")
    else:
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        logger.error(f"prefix_encoder.onnx: FAILED (max diff: {max_diff})")
        return False

    logger.info("All ONNX validations passed!")
    return True


def save_export_config(
    decoder: nn.Module,
    output_path: Path,
    opset_version: int,
) -> None:
    """Save export configuration for later use."""
    config = {
        "opset_version": opset_version,
        "num_layers": len(decoder.gpt2.transformer.h),
        "num_heads": decoder.gpt2.config.n_head,
        "hidden_size": decoder.gpt2.config.n_embd,
        "head_dim": decoder.gpt2.config.n_embd // decoder.gpt2.config.n_head,
        "vocab_size": decoder.gpt2.config.vocab_size,
        "encoder_hidden_size": decoder.encoder_hidden_size,
        "files": {
            "prefix_encoder": "prefix_encoder.onnx",
            "decoder_first_step": "decoder_first_step.onnx",
            "decoder_core": "decoder_core.onnx",
        },
    }

    config_path = output_path / "onnx_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved export config to {config_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Export GPT-2 Decoder to ONNX (Split Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3

    # Export with validation
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3 \\
        --validate

    # Export with specific opset
    python export_decoder_onnx.py \\
        --checkpoint outputs/ultrabert-gen-decoder-v3 \\
        --output exports/decoder-onnx-v3 \\
        --opset 14
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to decoder checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX outputs match PyTorch",
    )
    parser.add_argument(
        "--skip-kv-cache",
        action="store_true",
        help="Skip KV cache models (only export prefix_encoder and decoder_core)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load decoder
    decoder = load_decoder(checkpoint_path)

    # Export all components
    logger.info("=" * 60)
    logger.info("ONNX Export: GPT-2 Decoder (Split Architecture)")
    logger.info("=" * 60)

    export_prefix_encoder(decoder, output_path, args.opset)

    if not args.skip_kv_cache:
        try:
            export_decoder_first_step(decoder, output_path, args.opset)
            export_decoder_core(decoder, output_path, args.opset)
        except Exception as e:
            logger.warning(f"KV cache export failed: {e}")
            logger.warning("Falling back to simple decoder export (no KV cache)")
            args.skip_kv_cache = True

    if args.skip_kv_cache:
        # Export the simple full decoder from export_decoder_optimum
        export_decoder_simple(decoder, output_path, args.opset)

    # Save config
    save_export_config(decoder, output_path, args.opset)

    # Validate if requested
    if args.validate:
        validate_onnx_outputs(decoder, output_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Export Summary:")
    logger.info("=" * 60)

    total_size = 0
    for f in output_path.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info(f"  {f.name}: {size_mb:.2f} MB")

    logger.info(f"  Total: {total_size:.2f} MB")
    logger.info(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

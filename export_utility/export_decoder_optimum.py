#!/usr/bin/env python3
"""
Export GPT-2 Decoder to ONNX using HuggingFace Optimum.

This script exports the counterfactual decoder (encoder projection + GPT-2)
to ONNX format for edge deployment. Uses Optimum library for proper handling
of KV cache with modern transformers versions.

Architecture:
    1. prefix_encoder.onnx - Projects encoder hidden states (768 -> 1024)
    2. decoder.onnx - GPT-2 with KV cache support via Optimum

Usage:
    python export_decoder_optimum.py \
        --checkpoint outputs/ultrabert-gen-decoder-v3 \
        --output exports/decoder-onnx-v3

    # With validation
    python export_decoder_optimum.py \
        --checkpoint outputs/ultrabert-gen-decoder-v3 \
        --output exports/decoder-onnx-v3 \
        --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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

DEFAULT_OPSET = 17


class PrefixEncoderWrapper(nn.Module):
    """
    Wrapper for ONNX export of encoder projection layer.
    Projects encoder hidden states (768-dim) to GPT-2 hidden size (1024-dim).
    """

    def __init__(self, decoder):
        super().__init__()
        self.encoder_proj = decoder.encoder_proj
        self.adapter = getattr(decoder, "adapter", None)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project encoder hidden states to GPT-2 dimension."""
        prefix_embeds = self.encoder_proj(encoder_hidden_states)
        if self.adapter is not None:
            prefix_embeds = self.adapter(prefix_embeds)
        return prefix_embeds


class DecoderWrapper(nn.Module):
    """
    Simple decoder wrapper that takes full sequence input (no KV cache).
    For edge deployment, we use a simpler approach without KV cache
    since sequence lengths are typically short (< 128 tokens).
    """

    def __init__(self, decoder):
        super().__init__()
        self.gpt2 = decoder.gpt2
        self.hidden_size = decoder.gpt2.config.n_embd
        self.vocab_size = decoder.gpt2.config.vocab_size

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass combining prefix and decoder input.

        Args:
            prefix_embeds: (batch, prefix_len, hidden_size) - Projected encoder output
            decoder_input_ids: (batch, dec_len) - Decoder input token IDs
            attention_mask: (batch, prefix_len + dec_len) - Full attention mask

        Returns:
            logits: (batch, dec_len, vocab_size) - Token logits
        """
        # Get decoder input embeddings
        decoder_embeds = self.gpt2.transformer.wte(decoder_input_ids)

        # Get position embeddings
        batch_size, prefix_len, _ = prefix_embeds.shape
        dec_len = decoder_input_ids.shape[1]
        total_len = prefix_len + dec_len

        # Position IDs for the full sequence
        position_ids = torch.arange(total_len, device=decoder_input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Combine prefix + decoder embeddings
        inputs_embeds = torch.cat([prefix_embeds, decoder_embeds], dim=1)

        # Add position embeddings
        position_embeds = self.gpt2.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.gpt2.transformer.drop(hidden_states)

        # Run through transformer blocks
        for block in self.gpt2.transformer.h:
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden_states = outputs[0]

        hidden_states = self.gpt2.transformer.ln_f(hidden_states)

        # Get logits only for decoder positions (after prefix)
        decoder_hidden = hidden_states[:, prefix_len:, :]
        logits = self.gpt2.lm_head(decoder_hidden)

        return logits


def load_decoder(checkpoint_path: Path):
    """Load the decoder from checkpoint."""
    from modeling_studio.models import ModernBertMultiTaskModel

    logger.info(f"Loading model from {checkpoint_path}")
    # Use load_checkpoint instead of from_pretrained to properly load
    # encoder weights with "encoder." prefix stripping
    model = ModernBertMultiTaskModel.load_checkpoint(str(checkpoint_path))
    model.eval()

    # Get decoder head
    if "counterfactual" in model.heads:
        decoder = model.heads["counterfactual"]
        logger.info("Found counterfactual decoder head")
    else:
        raise ValueError(f"No counterfactual head found. Available: {list(model.heads.keys())}")

    return decoder


def export_prefix_encoder(
    decoder,
    output_dir: Path,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export the prefix encoder (projection layer) to ONNX."""
    wrapper = PrefixEncoderWrapper(decoder)
    wrapper.eval()

    output_path = output_dir / "prefix_encoder.onnx"

    # Dummy input
    batch_size, seq_len, hidden_dim = 1, 32, 768
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)

    logger.info(f"Exporting prefix encoder to {output_path}")

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["encoder_hidden_states"],
        output_names=["prefix_embeds"],
        dynamic_axes={
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "prefix_embeds": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify output
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Prefix encoder exported: {size_mb:.2f} MB")

    return output_path


def export_decoder_simple(
    decoder,
    output_dir: Path,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """
    Export the decoder (GPT-2) to ONNX without KV cache.
    Simpler approach that works with all transformers versions.
    """
    wrapper = DecoderWrapper(decoder)
    wrapper.eval()

    output_path = output_dir / "decoder.onnx"

    # Dummy inputs
    batch_size = 1
    prefix_len = 32
    dec_len = 16
    hidden_size = wrapper.hidden_size

    dummy_prefix = torch.randn(batch_size, prefix_len, hidden_size)
    dummy_ids = torch.randint(0, 1000, (batch_size, dec_len))
    dummy_mask = torch.ones(batch_size, prefix_len + dec_len)

    logger.info(f"Exporting decoder to {output_path}")

    torch.onnx.export(
        wrapper,
        (dummy_prefix, dummy_ids, dummy_mask),
        str(output_path),
        input_names=["prefix_embeds", "decoder_input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "prefix_embeds": {0: "batch_size", 1: "prefix_len"},
            "decoder_input_ids": {0: "batch_size", 1: "dec_len"},
            "attention_mask": {0: "batch_size", 1: "total_len"},
            "logits": {0: "batch_size", 1: "dec_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify output
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Decoder exported: {size_mb:.2f} MB")

    return output_path


def export_gpt2_with_optimum(
    decoder,
    output_dir: Path,
) -> Path:
    """
    Export GPT-2 using Optimum library for proper KV cache handling.
    This creates decoder_model.onnx and decoder_with_past_model.onnx
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from optimum.exporters.onnx import main_export
    except ImportError:
        logger.warning("Optimum not available, falling back to simple export")
        return None

    # First save GPT-2 to a temp directory
    temp_dir = output_dir / "temp_gpt2"
    temp_dir.mkdir(exist_ok=True)

    # Save GPT-2 model and config
    decoder.gpt2.save_pretrained(str(temp_dir))

    # Export using Optimum
    onnx_output = output_dir / "gpt2_onnx"

    logger.info(f"Exporting GPT-2 with Optimum to {onnx_output}")

    try:
        main_export(
            model_name_or_path=str(temp_dir),
            output=str(onnx_output),
            task="text-generation-with-past",
            opset=DEFAULT_OPSET,
            device="cpu",
        )
        logger.info(f"GPT-2 exported with Optimum to {onnx_output}")
        return onnx_output
    except Exception as e:
        logger.error(f"Optimum export failed: {e}")
        return None
    finally:
        # Cleanup temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def validate_onnx_models(
    output_dir: Path,
    decoder,
) -> bool:
    """Validate ONNX models against PyTorch outputs."""
    import onnxruntime as ort

    logger.info("Validating ONNX models...")

    # Test prefix encoder
    prefix_path = output_dir / "prefix_encoder.onnx"
    if prefix_path.exists():
        sess = ort.InferenceSession(str(prefix_path))
        test_input = np.random.randn(1, 16, 768).astype(np.float32)

        # ONNX output
        onnx_out = sess.run(None, {"encoder_hidden_states": test_input})[0]

        # PyTorch output
        wrapper = PrefixEncoderWrapper(decoder)
        wrapper.eval()
        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(test_input)).numpy()

        diff = np.abs(onnx_out - pt_out).max()
        logger.info(f"Prefix encoder max diff: {diff:.6f}")

        if diff > 1e-4:
            logger.warning("Prefix encoder validation failed!")
            return False

    # Test decoder
    decoder_path = output_dir / "decoder.onnx"
    if decoder_path.exists():
        sess = ort.InferenceSession(str(decoder_path))

        wrapper = DecoderWrapper(decoder)
        hidden_size = wrapper.hidden_size

        test_prefix = np.random.randn(1, 8, hidden_size).astype(np.float32)
        test_ids = np.array([[50256, 100, 200, 300]], dtype=np.int64)
        test_mask = np.ones((1, 12), dtype=np.float32)

        # ONNX output
        onnx_out = sess.run(
            None,
            {
                "prefix_embeds": test_prefix,
                "decoder_input_ids": test_ids,
                "attention_mask": test_mask,
            },
        )[0]

        # PyTorch output
        wrapper.eval()
        with torch.no_grad():
            pt_out = wrapper(
                torch.from_numpy(test_prefix),
                torch.from_numpy(test_ids),
                torch.from_numpy(test_mask),
            ).numpy()

        diff = np.abs(onnx_out - pt_out).max()
        logger.info(f"Decoder max diff: {diff:.6f}")

        if diff > 1e-3:
            logger.warning("Decoder validation failed!")
            return False

    logger.info("All validations passed!")
    return True


def save_metadata(output_dir: Path, decoder, opset: int):
    """Save export metadata."""
    metadata = {
        "format": "onnx",
        "opset_version": opset,
        "architecture": "gpt2-medium",
        "hidden_size": decoder.gpt2.config.n_embd,
        "vocab_size": decoder.gpt2.config.vocab_size,
        "num_layers": decoder.gpt2.config.n_layer,
        "num_heads": decoder.gpt2.config.n_head,
        "encoder_hidden_size": 768,
        "files": {
            "prefix_encoder": "prefix_encoder.onnx",
            "decoder": "decoder.onnx",
        },
    }

    metadata_path = output_dir / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 decoder to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ultrabert-gen-decoder-v3",
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/decoder-onnx-v3",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX outputs against PyTorch",
    )
    parser.add_argument(
        "--use-optimum",
        action="store_true",
        help="Use Optimum for GPT-2 export (includes KV cache)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ONNX DECODER EXPORT")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Opset: {args.opset}")

    # Load decoder
    decoder = load_decoder(checkpoint_path)

    # Export prefix encoder
    prefix_path = export_prefix_encoder(decoder, output_dir, args.opset)

    # Export decoder
    if args.use_optimum:
        gpt2_path = export_gpt2_with_optimum(decoder, output_dir)
        if gpt2_path is None:
            logger.info("Falling back to simple decoder export")
            decoder_path = export_decoder_simple(decoder, output_dir, args.opset)
    else:
        decoder_path = export_decoder_simple(decoder, output_dir, args.opset)

    # Save metadata
    save_metadata(output_dir, decoder, args.opset)

    # Validate if requested
    if args.validate:
        validate_onnx_models(output_dir, decoder)

    logger.info("=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)

    # List exported files
    for f in output_dir.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

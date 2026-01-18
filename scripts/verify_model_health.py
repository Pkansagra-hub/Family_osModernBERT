
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, GPT2Config
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.models.decoder_gpt2 import GPT2DecoderHead
from modeling_studio.models.decoder_gpt2_config import GPT2DecoderConfig

def verify_checkpoint(checkpoint_path):
    print(f"\n{'='*60}")
    print(f"VERIFYING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*60}\n")

    checkpoint_path = Path(checkpoint_path)

    # 1. Verify Tokenizer
    print("--- 1. Tokenizer Verification ---")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer Class: {type(tokenizer).__name__}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Len(Tokenizer): {len(tokenizer)}")

    special_tokens = {
        "pad_token": (tokenizer.pad_token, tokenizer.pad_token_id),
        "bos_token": (tokenizer.bos_token, tokenizer.bos_token_id),
        "eos_token": (tokenizer.eos_token, tokenizer.eos_token_id),
        "cls_token": (tokenizer.cls_token, tokenizer.cls_token_id),
        "sep_token": (tokenizer.sep_token, tokenizer.sep_token_id),
    }

    for name, (token, token_id) in special_tokens.items():
        print(f"{name}: '{token}' (ID: {token_id})")

    # 2. Verify Config
    print("\n--- 2. Config Verification ---")
    config = AutoConfig.from_pretrained(checkpoint_path)
    print(f"Model Type: {config.model_type}")
    print(f"Config Vocab Size: {config.vocab_size}")
    print(f"Hidden Size: {config.hidden_size}")

    # Check alignment
    if config.vocab_size == len(tokenizer):
        print("✅ Config vocab size matches tokenizer length")
    else:
        print(f"❌ MISMATCH: Config vocab ({config.vocab_size}) != Tokenizer len ({len(tokenizer)})")

    # 3. Verify Weights (Shapes & Fusion)
    print("\n--- 3. Weight & Architecture Verification ---")
    state_dict_path = checkpoint_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        print("Loading safetensors...")
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path / "model.safetensors"))
    else:
        print("Loading pytorch_model.bin...")
        state_dict = torch.load(state_dict_path, map_location="cpu")

    # Check Encoder Embeddings
    enc_embed_key = "encoder.embeddings.word_embeddings.weight"
    if enc_embed_key in state_dict:
        enc_shape = state_dict[enc_embed_key].shape
        print(f"Encoder Embeddings Shape: {enc_shape}")
        if enc_shape[0] == config.vocab_size:
             print(f"✅ Encoder embeddings match config vocab size ({config.vocab_size})")
        else:
             print(f"❌ MISMATCH: Encoder embeddings ({enc_shape[0]}) != Config vocab ({config.vocab_size})")
    else:
        print("⚠️ Could not find encoder embeddings in state dict (might be under different key)")

    # Check Decoder Projection (Fusion)
    # The projection layer connects the 768-dim encoder to the 1024-dim decoder
    proj_weight_key = "heads.counterfactual.encoder_proj.projection.weight"
    if proj_weight_key in state_dict:
        proj_shape = state_dict[proj_weight_key].shape
        print(f"Projection Layer Shape: {proj_shape}")
        # Linear layer weight is (out_features, in_features)
        expected_in = 768
        expected_out = 1024
        if proj_shape == (expected_out, expected_in):
            print(f"✅ Projection layer correctly fuses Encoder ({expected_in}) -> Decoder ({expected_out})")
        else:
            print(f"❌ MISMATCH: Projection layer shape {proj_shape} != Expected ({expected_out}, {expected_in})")
    else:
        print("⚠️ Could not find projection layer weights")

    # Check Decoder Embeddings
    # GPT-2 embeddings are usually tied, but let's check the decoder's internal embedding if possible
    # In this architecture, the decoder is a separate module in 'heads.counterfactual.gpt2'
    dec_embed_key = "heads.counterfactual.gpt2.transformer.wte.weight"
    if dec_embed_key in state_dict:
        dec_shape = state_dict[dec_embed_key].shape
        print(f"Decoder (GPT-2) Embeddings Shape: {dec_shape}")
        if dec_shape[0] == config.vocab_size:
            print(f"✅ Decoder embeddings match config vocab size ({config.vocab_size})")
        else:
            print(f"❌ MISMATCH: Decoder embeddings ({dec_shape[0]}) != Config vocab ({config.vocab_size})")

        # Check if they are non-zero (sanity check for initialization)
        # Specifically check the new tokens
        new_tokens_slice = state_dict[dec_embed_key][50257:]
        norm = new_tokens_slice.norm(dim=1).mean().item()
        print(f"New tokens (50257+) mean norm: {norm:.4f}")
        if 2.0 < norm < 5.0:
             print("✅ New token embeddings look properly initialized (norm ~3.68 expected)")
        else:
             print(f"⚠️ New token embeddings norm ({norm:.4f}) seems off (too small or too large)")

    else:
        print("⚠️ Could not find decoder embeddings")

    # Check Decoder Output Head
    # Since weights are tied, this might be the same as wte, but let's check if there's a separate head or if it uses wte
    # GPT2LMHeadModel usually uses wte.

    print("\n--- 4. Special Token Alignment ---")
    # Check if BOS/EOS/PAD in config match what we expect
    print(f"Config PAD: {config.pad_token_id} | Tokenizer PAD: {tokenizer.pad_token_id}")
    print(f"Config BOS: {config.bos_token_id} | Tokenizer BOS: {tokenizer.bos_token_id}")
    print(f"Config EOS: {config.eos_token_id} | Tokenizer EOS: {tokenizer.eos_token_id}")

    if config.pad_token_id == tokenizer.pad_token_id:
        print("✅ PAD token aligned")
    else:
        print("❌ PAD token mismatch")

    # Note: Tokenizer might not have bos_token set explicitly if it's a BERT tokenizer,
    # but our config treats CLS as BOS.
    if config.bos_token_id == tokenizer.cls_token_id:
        print("✅ Config BOS matches Tokenizer CLS (Correct for Encoder-Decoder)")
    else:
        print(f"ℹ️ Config BOS ({config.bos_token_id}) vs Tokenizer CLS ({tokenizer.cls_token_id})")

    if config.eos_token_id == tokenizer.sep_token_id:
        print("✅ Config EOS matches Tokenizer SEP (Correct for Encoder-Decoder)")
    else:
        print(f"ℹ️ Config EOS ({config.eos_token_id}) vs Tokenizer SEP ({tokenizer.sep_token_id})")

if __name__ == "__main__":
    verify_checkpoint("D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v4")

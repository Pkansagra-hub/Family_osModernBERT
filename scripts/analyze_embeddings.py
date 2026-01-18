"""
Analyze UltraBERT Embeddings and Text Alignment.

This script runs a small test set of prompts through the encoder to:
1. Monitor embedding statistics (Mean, Std, Min, Max)
2. Compare embeddings across different emotions and roles
3. Verify that embeddings are stable and distinct
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel

def load_encoder(checkpoint_path, device):
    print(f"Loading encoder from {checkpoint_path}...")
    # We use the same loading logic as infer_decoder_fp16.py but simplified
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    # Load full model to get encoder
    # Note: This might be heavy, but ensures we get the exact encoder used in training
    # Alternatively, we can load just the encoder if we know the structure
    # For now, let's try loading the ModernBERT base and applying the state dict if possible
    # or just use the inference script's logic

    # Let's use the logic from infer_decoder_fp16.py which is robust
    from transformers import ModernBertModel

    encoder = ModernBertModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        torch_dtype=torch.float16,
    )

    # Load state dict
    if (Path(checkpoint_path) / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(Path(checkpoint_path) / "model.safetensors"))
    else:
        state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")

    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "", 1)
            encoder_state_dict[new_key] = value
        elif key.startswith("backbone."):
            new_key = key.replace("backbone.", "", 1)
            encoder_state_dict[new_key] = value

    if encoder_state_dict:
        missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Encoder loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    encoder = encoder.to(device).half()
    encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    return encoder, tokenizer

def analyze_prompt(text, encoder, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

    # Calculate stats
    mean = embeddings.mean().item()
    std = embeddings.std().item()
    min_val = embeddings.min().item()
    max_val = embeddings.max().item()
    norm = torch.norm(embeddings, dim=-1).mean().item()

    return {
        "text": text,
        "shape": tuple(embeddings.shape),
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "norm": norm
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v4"

    encoder, tokenizer = load_encoder(checkpoint_path, device)

    test_cases = [
        # Same scenario, different emotions
        "[Role: mom | Emotion: happy] My child finished their homework early.",
        "[Role: mom | Emotion: angry] My child refuses to do their homework.",
        "[Role: mom | Emotion: worried] My child is struggling with homework.",

        # Same emotion, different roles
        "[Role: dad | Emotion: worried] My child is struggling with homework.",
        "[Role: teacher | Emotion: worried] My child is struggling with homework.",

        # Control
        "Simple neutral sentence.",
    ]

    print("\n" + "="*80)
    print(f"{'Prompt':<60} | {'Mean':<8} | {'Std':<8} | {'Norm':<8}")
    print("="*80)

    results = []
    for text in test_cases:
        stats = analyze_prompt(text, encoder, tokenizer, device)
        results.append(stats)
        print(f"{text[:57]+'...':<60} | {stats['mean']:.4f}   | {stats['std']:.4f}   | {stats['norm']:.4f}")

    print("="*80)
    print("\nAnalysis:")
    print("1. Check if 'Mean' and 'Std' are stable (should be similar range).")
    print("2. Check if 'Norm' varies significantly with emotion (intensity).")
    print("3. Embeddings are the input to the decoder. Large variance in stats might indicate instability.")

if __name__ == "__main__":
    main()

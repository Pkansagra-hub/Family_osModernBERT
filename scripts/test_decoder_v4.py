#!/usr/bin/env python
"""
Quick test script for GPT-2 decoder v4 checkpoint.

Usage:
    python scripts/test_decoder_v4.py
"""

import torch
from transformers import AutoTokenizer

# Add project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel


def main():
    checkpoint_path = "outputs/ultrabert-gen-decoder-v4"

    print("=" * 60)
    print("Loading GPT-2 Decoder v4 Checkpoint")
    print("=" * 60)

    # Load model
    print("Loading model...")
    model = ModernBertMultiTaskModel.load_checkpoint(checkpoint_path, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    print(f"Capabilities: {[c.value for c in model.capabilities]}")
    print(f"Has counterfactual head: {'counterfactual' in model.heads}")

    # Get decoder (ModuleDict uses [] not .get())
    if "counterfactual" not in model.heads:
        print("ERROR: No counterfactual decoder head found!")
        return

    decoder = model.heads["counterfactual"]

    print(f"Decoder type: {type(decoder).__name__}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test inputs - family scenarios
    test_inputs = [
        "I yelled at my kids this morning because they were late for school.",
        "I forgot my wife's birthday and she was really upset.",
        "I spent the whole weekend working instead of being with my family.",
        "I criticized my teenager's grades without asking how they were feeling.",
        "I was too tired to help my son with his homework.",
    ]

    print("\n" + "=" * 60)
    print("Generating Counterfactuals")
    print("=" * 60)

    model.eval()

    for i, input_text in enumerate(test_inputs):
        print(f"\n--- Example {i+1} ---")
        print(f"INPUT: {input_text}")

        # Get encoder embeddings
        with torch.no_grad():
            # Tokenize input
            encoded = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to("cuda")

            # Get encoder hidden states
            encoder_outputs = model.encoder(**encoded)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = encoded["attention_mask"]

            # Generate counterfactual
            try:
                if hasattr(decoder, "generate"):
                    output_ids = decoder.generate(
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.2,
                    )

                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    print(f"OUTPUT: {output_text}")
                else:
                    print("ERROR: Decoder doesn't have generate() method")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

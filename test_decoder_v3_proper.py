"""
Test v3 Decoder with PROPER Training-Time Parameters
=====================================================

The decoder was trained with:
- temperature: 1.0
- top_k: 50
- top_p: 0.9
- repetition_penalty: 1.2

But we've been testing with temperature 0.2-0.8!
This train-inference mismatch causes poor generation.

Let's test with the ACTUAL training parameters.
"""

import torch
from transformers import AutoTokenizer
from modeling_studio.models import ModernBertMultiTaskModel

def test_with_training_params():
    """Test decoder with exact training-time generation parameters."""

    print("="*80)
    print("TESTING V3 DECODER WITH TRAINING-TIME PARAMETERS")
    print("="*80)

    print("\nLoading model...")
    model = ModernBertMultiTaskModel.from_pretrained(
        "outputs/ultrabert-gen-decoder-v1",
        torch_dtype=torch.float16
    )
    model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v1")

    test_cases = [
        "I hate this situation",
        "I felt overwhelmed today",
        "I'm so angry at everything",
        "My wife and I keep fighting about money",
        "I lost my parent last year",
        "I'm juggling work, kids, and aging parents",
    ]

    # Training-time parameters from config
    training_params = {
        "max_new_tokens": 128,
        "temperature": 1.0,      # EXACT match from training config
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }

    # Also test with slight adjustments
    test_configs = [
        {
            "name": "Training Params (Exact)",
            "params": training_params
        },
        {
            "name": "Training Params + Lower Temp (0.7)",
            "params": {**training_params, "temperature": 0.7}
        },
        {
            "name": "Training Params + Higher Rep Penalty (1.5)",
            "params": {**training_params, "repetition_penalty": 1.5}
        },
        {
            "name": "Training Params + Both Adjusted",
            "params": {**training_params, "temperature": 0.7, "repetition_penalty": 1.5}
        },
    ]

    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Config: {config['name']}")
        print(f"{'='*80}")
        params = config['params']
        print(f"  temp={params['temperature']}, top_k={params['top_k']}, "
              f"top_p={params['top_p']}, rep_pen={params['repetition_penalty']}")
        print()

        for text in test_cases[:3]:  # Test first 3
            inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

            with torch.inference_mode():
                encoder_hidden = model.encoder(**inputs).last_hidden_state
                generated = model.heads["counterfactual"].generate(
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=inputs.get("attention_mask"),
                    **params
                )

            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Input:  {text}")
            print(f"Output: {output[:200]}{'...' if len(output) > 200 else ''}")
            print()


def test_greedy_decoding():
    """Test with greedy decoding (temperature=0) to see deterministic output."""

    print("\n" + "="*80)
    print("TESTING WITH GREEDY DECODING (Deterministic)")
    print("="*80)

    model = ModernBertMultiTaskModel.from_pretrained(
        "outputs/ultrabert-gen-decoder-v1",
        device_map="cuda",
        torch_dtype=torch.float16
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v1")

    test_cases = [
        "I hate this situation",
        "I felt overwhelmed today",
        "I'm so angry at everything",
    ]

    print("\nGreedy decoding (temperature near 0):")
    print()

    for text in test_cases:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

        with torch.inference_mode():
            encoder_hidden = model.encoder(**inputs).last_hidden_state

            # Greedy: temperature very low, top_k=1
            generated = model.heads["counterfactual"].generate(
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=inputs.get("attention_mask"),
                max_new_tokens=80,
                temperature=0.01,  # Near-greedy
                top_k=1,
                top_p=1.0,
                repetition_penalty=2.0,  # High penalty to avoid loops
            )

        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Input:  {text}")
        print(f"Output: {output}")
        print()


def test_beam_search():
    """Test with beam search instead of sampling."""

    print("\n" + "="*80)
    print("TESTING WITH BEAM SEARCH (num_beams=4)")
    print("="*80)

    model = ModernBertMultiTaskModel.from_pretrained(
        "outputs/ultrabert-gen-decoder-v1",
        device_map="cuda",
        torch_dtype=torch.float16
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v1")

    test_cases = [
        "I hate this situation",
        "I felt overwhelmed today",
        "I'm so angry at everything",
    ]

    print("\nBeam search with 4 beams:")
    print()

    for text in test_cases:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

        with torch.inference_mode():
            encoder_hidden = model.encoder(**inputs).last_hidden_state

            # Beam search parameters
            generated = model.heads["counterfactual"].generate(
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=inputs.get("attention_mask"),
                max_new_tokens=80,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
            )

        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Input:  {text}")
        print(f"Output: {output}")
        print()


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("  V3 DECODER - PROPER PARAMETER TESTING")
        print("="*80)

        # Test 1: Training-time parameters
        test_with_training_params()

        # Test 2: Greedy decoding
        test_greedy_decoding()

        # Test 3: Beam search
        test_beam_search()

        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

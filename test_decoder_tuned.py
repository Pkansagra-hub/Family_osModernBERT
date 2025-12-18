"""
Test Decoder with Tuned Hyperparameters and Enhanced Prompts
==============================================================

Testing different configurations to fix gibberish output:
1. Lower temperature (more focused)
2. Reduced max_tokens (prevent rambling)
3. Higher repetition penalty (avoid loops)
4. Enhanced prompt engineering
"""

import time
from familyos_ultrabert import Client

def test_configurations():
    """Test different hyperparameter configurations."""

    print("="*70)
    print("DECODER HYPERPARAMETER TUNING TEST")
    print("="*70)

    # Initialize PyTorch backend
    print("\nInitializing PyTorch client...")
    client = Client(backend='pytorch', warmup=False)
    print(f"Backend: {client.backend}")

    # Test cases - various emotional/challenging statements
    test_cases = [
        "I hate this situation",
        "I felt overwhelmed today",
        "I'm so angry at everything",
        "I can't handle this anymore",
        "Everything is going wrong",
    ]

    # Different hyperparameter configurations to test
    configs = [
        {
            "name": "Default (Current)",
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
        },
        {
            "name": "Conservative (Low Temp)",
            "max_new_tokens": 64,
            "temperature": 0.3,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.5,
        },
        {
            "name": "Focused (Very Low Temp)",
            "max_new_tokens": 48,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.8,
        },
        {
            "name": "Balanced (Medium)",
            "max_new_tokens": 80,
            "temperature": 0.5,
            "top_p": 0.88,
            "top_k": 40,
            "repetition_penalty": 1.4,
        },
        {
            "name": "Greedy (Deterministic)",
            "max_new_tokens": 60,
            "temperature": 0.01,  # Nearly deterministic
            "top_p": 1.0,
            "top_k": 1,
            "repetition_penalty": 2.0,
        },
    ]

    # Test each configuration
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*70}")
        print(f"  max_tokens={config['max_new_tokens']}, temp={config['temperature']}, "
              f"top_p={config['top_p']}, top_k={config['top_k']}, rep_pen={config['repetition_penalty']}")
        print()

        with client.create_decoder_session() as decoder:
            for test_text in test_cases[:2]:  # Test first 2 cases for speed
                print(f"Input:  '{test_text}'")

                encoder_output = client.encode(test_text)

                start = time.time()
                result = decoder.generate(
                    encoder_output,
                    max_new_tokens=config['max_new_tokens'],
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    top_k=config['top_k'],
                    repetition_penalty=config['repetition_penalty'],
                )
                gen_time = (time.time() - start) * 1000

                print(f"Output: '{result[:200]}{'...' if len(result) > 200 else ''}'")
                print(f"Time:   {gen_time:.1f}ms")
                print()


def test_with_prompt_engineering():
    """Test with enhanced prompt/text preprocessing."""

    print("\n" + "="*70)
    print("ENHANCED PROMPT ENGINEERING TEST")
    print("="*70)

    client = Client(backend='pytorch', warmup=False)

    # Original problematic inputs
    original_inputs = [
        "I hate this situation",
        "I felt overwhelmed today",
        "I'm so angry at everything",
    ]

    # Enhanced prompts - add context or reframe
    enhanced_prompts = [
        "Reframe: I hate this situation",
        "Alternative perspective: I felt overwhelmed today",
        "Constructive reframe: I'm so angry at everything",
    ]

    # Best config from previous test (we'll use Conservative)
    best_config = {
        "max_new_tokens": 48,
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 30,
        "repetition_penalty": 1.6,
    }

    print(f"\nUsing config: temp={best_config['temperature']}, "
          f"max_tokens={best_config['max_new_tokens']}, "
          f"rep_pen={best_config['repetition_penalty']}")

    with client.create_decoder_session() as decoder:
        print("\n--- Original Inputs ---")
        for text in original_inputs:
            encoder_output = client.encode(text)
            result = decoder.generate(encoder_output, **best_config)
            print(f"\nInput:  {text}")
            print(f"Output: {result[:150]}")

        print("\n--- Enhanced Prompts ---")
        for text in enhanced_prompts:
            encoder_output = client.encode(text)
            result = decoder.generate(encoder_output, **best_config)
            print(f"\nInput:  {text}")
            print(f"Output: {result[:150]}")


def test_structured_output():
    """Test structured output with tuned params."""

    print("\n" + "="*70)
    print("STRUCTURED OUTPUT WITH TUNED PARAMS")
    print("="*70)

    client = Client(backend='pytorch', warmup=False)

    best_config = {
        "max_new_tokens": 48,
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 30,
        "repetition_penalty": 1.6,
    }

    test_texts = [
        "I felt overwhelmed today",
        "I hate this situation",
        "I'm so angry at everything",
    ]

    with client.create_decoder_session() as decoder:
        for text in test_texts:
            print(f"\nInput: {text}")
            encoder_output = client.encode(text)

            # Generate with structured output
            result = decoder.generate_structured(
                encoder_output,
                max_new_tokens=best_config['max_new_tokens'],
            )

            print(f"Generated: {result['text'][:150]}")
            print(f"Time: {result['generation_time_ms']:.1f}ms")
            print(f"Procedural Insight: {result['procedural_insight']}")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  DECODER QUALITY TUNING & TESTING")
    print("█"*70)

    try:
        # Test 1: Try different hyperparameter configurations
        test_configurations()

        # Test 2: Try prompt engineering
        test_with_prompt_engineering()

        # Test 3: Structured output
        test_structured_output()

        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70)
        print("\nRecommendation: Choose the configuration that produces")
        print("the most coherent, concise output for your use case.")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""Test counterfactual generation using familyos_ultrabert package.

Transforms negative/hostile statements into constructive alternatives.
"""

from familyos_ultrabert import Client


def main():
    print("=" * 70)
    print("COUNTERFACTUAL GENERATION TEST")
    print("Using familyos_ultrabert.Client.suggest_alternative()")
    print("=" * 70)

    # Initialize client with PyTorch backend (required for decoder - needs encoder hidden states)
    print("\nInitializing Client (PyTorch backend - required for decoder)...")
    client = Client(backend="pytorch")
    print(f"Backend: {client.backend}")
    print(f"Version: {client.VERSION}")

    # Test cases: negative family communication -> positive reframing
    test_cases = [
        "You never listen to me!",
        "Why are you always so lazy?",
        "You're being ridiculous right now.",
        "Stop being so dramatic about everything.",
        "You always ruin everything.",
        "I can't believe you forgot again!",
        "You're so selfish, you only think about yourself.",
    ]

    print("\n" + "-" * 70)
    print("COUNTERFACTUAL EXAMPLES")
    print("Negative statement -> Constructive alternative")
    print("-" * 70)

    for i, original in enumerate(test_cases, 1):
        print(f"\n[{i}] ORIGINAL: \"{original}\"")

        try:
            # Use the package's suggest_alternative method
            alternative = client.suggest_alternative(
                text=original,
                max_new_tokens=40,
                temperature=0.7,
            )
            print(f"    ALTERNATIVE: \"{alternative}\"")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\n" + "=" * 70)
    print("COUNTERFACTUAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

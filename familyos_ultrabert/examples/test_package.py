#!/usr/bin/env python
"""Test the FamilyOS UltraBERT v2 package."""

import sys
sys.path.insert(0, ".")

from familyos_ultrabert import UltraBERT

def test_pytorch_backend():
    print("=" * 60)
    print("Testing PyTorch Backend")
    print("=" * 60)

    model = UltraBERT.load(backend="pytorch", device="cuda")
    print(f"Backend: {model.backend}")
    print(f"Capabilities ({len(model.capabilities)}): {model.capabilities}")

    # Test multi-capability
    text = "Mom picked up Panda from school today!"
    result = model.analyze(
        text,
        capabilities=["sentiment", "ner_family", "safety_familyos", "emotions", "intent"]
    )

    print(f"\nText: {text}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Sentiment: {result['sentiment']['prediction']} ({result['sentiment']['confidence']:.2f})")
    print(f"Safety: {result['safety_familyos']['band']} ({result['safety_familyos']['confidence']:.2f})")
    print(f"Intent: {result['intent']['prediction']}")
    print(f"Entities: {result['ner_family']['entities']}")
    print(f"Emotions: {result['emotions']['predictions']}")

    # Test convenience methods
    print("\n--- Convenience Methods ---")
    print(f"get_sentiment: {model.get_sentiment('I love you!')['prediction']}")
    print(f"get_emotions: {model.get_emotions('So excited!')}")
    print(f"get_safety_band: {model.get_safety_band('Having fun!')}")
    print(f"get_entities: {model.get_entities('Dad and sis went shopping')}")
    embedding = model.get_embedding("Test text")
    print(f"get_embedding: {len(embedding)}-dim vector, norm={sum(x*x for x in embedding)**0.5:.3f}")

    return True


def test_onnx_backend():
    print("\n" + "=" * 60)
    print("Testing ONNX Backend")
    print("=" * 60)

    model = UltraBERT.load(backend="onnx", device="cpu")
    print(f"Backend: {model.backend}")
    print(f"Capabilities ({len(model.capabilities)}): {model.capabilities}")

    # Test single capability
    text = "I'm feeling sad today"
    result = model.analyze(text, capabilities=["sentiment", "emotions"])

    print(f"\nText: {text}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Sentiment: {result['sentiment']['prediction']}")
    print(f"Emotions: {result['emotions']['predictions']}")

    return True


def test_all_capabilities():
    print("\n" + "=" * 60)
    print("Testing All 12 Capabilities")
    print("=" * 60)

    model = UltraBERT.load(backend="pytorch", device="cuda")

    text = "Mom and Dad surprised grandma with a birthday party last Sunday at 3pm"
    result = model.analyze(text)  # All capabilities

    print(f"\nText: {text}")
    print(f"Latency: {result.latency_ms:.1f}ms for {len(result.capabilities)} capabilities")

    for cap, output in result.items():
        if cap == "embedding":
            print(f"  {cap}: {output['dim']}-dim embedding")
        elif "entities" in output:
            print(f"  {cap}: {len(output['entities'])} entities")
        elif "predictions" in output:
            print(f"  {cap}: {output['predictions']}")
        elif "prediction" in output:
            print(f"  {cap}: {output['prediction']} ({output.get('confidence', 'N/A')})")
        elif "band" in output:
            print(f"  {cap}: {output['band']}")
        else:
            print(f"  {cap}: {list(output.keys())}")

    return True


if __name__ == "__main__":
    try:
        test_pytorch_backend()
        test_onnx_backend()
        test_all_capabilities()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

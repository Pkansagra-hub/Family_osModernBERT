"""
Example: Using FamilyOS UltraBERT with pip-installed model

This script demonstrates how to use the UltraBERT model that was installed via pip.
No checkpoint loading required - the model weights are bundled with the package.

Usage:
    pip install familyos-ultrabert
    python example_pip_usage.py
"""

from familyos_ultrabert import Client

def main():
    # Initialize client - model loads automatically from bundled weights
    # Warmup happens automatically, so first call is already fast
    print("Initializing FamilyOS UltraBERT...")
    client = Client()

    print(f"Model loaded with {len(client.capabilities)} capabilities:")
    print(f"  {', '.join(sorted(client.capabilities))}")
    print()

    # Test text
    text = "My younger sister Aisha is getting married next month, and I'm so excited but also a bit sad she'll be moving far away."

    print(f"Text: {text}")
    print("-" * 80)

    # =========================================================================
    # 1. Quick single-capability calls
    # =========================================================================
    print("\n=== Quick Single-Capability Calls ===\n")

    # Intent (v2 LabelDescriptionHead)
    intent = client.get_intent(text)
    print(f"Intent: {intent}")

    # Ingress (v2 LabelDescriptionHead)
    ingress = client.get_ingress(text)
    print(f"Ingress: {ingress}")

    # Sentiment
    sentiment = client.get_sentiment(text)
    print(f"Sentiment: {sentiment}")

    # Safety
    safety = client.get_safety(text)
    print(f"Safety: {safety}")

    # =========================================================================
    # 2. Full multi-label intent/ingress with scores
    # =========================================================================
    print("\n=== Full Intent/Ingress Analysis ===\n")

    # Intent with all scores
    intent_full = client.get_intent_with_descriptions(text)
    print(f"Intent (full):")
    print(f"  Primary: {intent_full.get('primary')}")
    print(f"  All above threshold: {intent_full.get('all', [])}")
    print(f"  Confidence: {intent_full.get('confidence', 0):.3f}")

    # Ingress with all scores
    ingress_full = client.get_ingress_with_descriptions(text)
    print(f"\nIngress (full):")
    print(f"  Primary: {ingress_full.get('primary')}")
    print(f"  Domains above threshold: {ingress_full.get('domains', [])}")
    print(f"  Confidence: {ingress_full.get('confidence', 0):.3f}")

    # =========================================================================
    # 3. Full analysis (all 12 heads in one call)
    # =========================================================================
    print("\n=== Full Analysis (All 12 Heads) ===\n")

    result = client.analyze(text)

    print(f"Latency: {result.latency_ms:.1f}ms")
    print()

    # Emotions
    emotions = result.emotions
    if isinstance(emotions, dict) and 'predictions' in emotions:
        print(f"Emotions: {emotions['predictions'][:5]}")
    else:
        print(f"Emotions: {emotions}")

    # NER Family entities
    ner_family = result.ner_family
    if isinstance(ner_family, dict) and 'entities' in ner_family:
        entities = ner_family['entities']
        print(f"Family Entities: {len(entities)} found")
        for e in entities[:3]:
            print(f"  - {e.get('text', 'N/A')} ({e.get('label', 'N/A')})")

    # Temporal
    temporal = result.temporal
    if isinstance(temporal, dict) and 'entities' in temporal:
        entities = temporal['entities']
        print(f"Temporal: {len(entities)} found")
        for e in entities[:3]:
            print(f"  - {e.get('text', 'N/A')} ({e.get('label', 'N/A')})")

    # Relations
    relations = result.relation
    if isinstance(relations, dict) and 'predictions' in relations:
        print(f"Relations: {relations['predictions']}")

    # =========================================================================
    # 4. Batch processing
    # =========================================================================
    print("\n=== Batch Processing ===\n")

    texts = [
        "Dad is coming to visit next weekend!",
        "I'm worried about Mom's health appointment tomorrow.",
        "The kids had so much fun at grandma's birthday party.",
    ]

    for t in texts:
        intent = client.get_intent(t)
        safety = client.get_safety(t)
        print(f"  [{safety}] [{intent}] {t[:50]}...")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

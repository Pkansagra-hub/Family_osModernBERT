"""Explore GoEmotions dataset and understand hit rate metric.

This script:
1. Downloads GoEmotions validation set
2. Makes predictions with UltraBERT
3. Shows examples of hits/misses to understand the metric
4. Tests different mapping strategies
"""

from datasets import load_dataset

# GoEmotions labels (28 total)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Labels to SKIP - don't transfer well across domains
# neutral/approval/curiosity/realization are too context-dependent
SKIP_GOEMOTIONS = {"neutral", "approval", "curiosity", "realization", "confusion"}

# Mapping GoEmotions -> UltraBERT (family-focused semantic expansion)
# Only map emotions that have clear cross-domain meaning
GOEMOTIONS_TO_ULTRABERT = {
    # Strong emotions that transfer well
    "admiration": ["admiration", "pride"],
    "amusement": ["amusement", "playfulness", "joy"],
    "anger": ["anger", "frustration", "annoyance"],
    "annoyance": ["annoyance", "frustration"],
    "caring": ["caring", "protectiveness", "tenderness", "warmth"],
    "desire": ["longing"],
    "disappointment": ["disappointment", "sadness"],
    "disapproval": ["disapproval", "frustration"],
    "disgust": ["disgust"],
    "embarrassment": ["embarrassment"],
    "excitement": ["excitement", "celebration", "joy"],
    "fear": ["fear", "worry", "nervousness"],
    "gratitude": ["gratitude", "warmth", "love"],
    "grief": ["grief", "sadness", "emptiness"],
    "joy": ["joy", "contentment", "celebration", "warmth"],
    "love": ["love", "warmth", "tenderness", "belonging", "caring"],
    "nervousness": ["nervousness", "worry", "fear"],
    "optimism": ["optimism", "hope"],
    "pride": ["pride", "parental_pride", "admiration"],
    "relief": ["relief", "contentment"],
    "remorse": ["remorse", "parental_guilt", "sadness"],
    "sadness": ["sadness", "grief", "longing", "emptiness"],
    "surprise": ["surprise"],
}


def map_to_ultrabert(goemotions_labels):
    """Map GoEmotions labels to UltraBERT space, skipping non-transferable labels."""
    result = set()
    for label in goemotions_labels:
        if label in SKIP_GOEMOTIONS:
            continue  # Skip labels that don't transfer across domains
        if label in GOEMOTIONS_TO_ULTRABERT:
            result.update(GOEMOTIONS_TO_ULTRABERT[label])
    return result


def should_skip_sample(goemotions_labels):
    """Skip samples that ONLY have non-transferable labels."""
    transferable = [l for l in goemotions_labels if l not in SKIP_GOEMOTIONS]
    return len(transferable) == 0


def is_single_sentence(text):
    """Check if text is a single sentence (not a paragraph)."""
    endings = text.count(".") + text.count("!") + text.count("?")
    return endings <= 2 and len(text) < 200


def main():
    print("=" * 70)
    print("GoEmotions Dataset Explorer - Understanding Hit Rate")
    print("=" * 70)

    # Step 1: Load dataset
    print("\n[1] Loading GoEmotions validation set...")
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        split="validation",
        trust_remote_code=True,
    )
    print(f"    Loaded {len(dataset)} total samples")

    # Step 2: Load UltraBERT
    print("\n[2] Loading UltraBERT model...")
    from familyos_ultrabert import Client
    client = Client()
    print("    Model ready")

    # Step 3: Filter and process
    print("\n[3] Filtering to single-sentence, emotional samples...")

    hits = 0
    misses = 0
    hit_examples = []
    miss_examples = []

    # Process ALL samples with filtering
    num_samples = 1000  # Target samples after filtering
    processed = 0
    skipped_paragraph = 0
    skipped_neutral = 0

    for item in dataset:
        if processed >= num_samples:
            break

        text = item["text"]
        label_indices = item["labels"]
        goemotions_labels = [GOEMOTIONS_LABELS[idx] for idx in label_indices]

        # Skip paragraphs
        if not is_single_sentence(text):
            skipped_paragraph += 1
            continue

        # Skip samples with only non-transferable labels
        if should_skip_sample(goemotions_labels):
            skipped_neutral += 1
            continue

        processed += 1
        expected_ultrabert = map_to_ultrabert(goemotions_labels)

        # Get prediction
        predicted = set(client.get_emotions(text))
        predicted_lower = {e.lower() for e in predicted}

        # Check hit (any overlap)
        overlap = predicted_lower.intersection(expected_ultrabert)
        is_hit = len(overlap) > 0

        if is_hit:
            hits += 1
            if len(hit_examples) < 5:
                hit_examples.append({
                    "text": text,
                    "goemotions": goemotions_labels,
                    "expected": sorted(expected_ultrabert),
                    "predicted": sorted(predicted_lower),
                    "overlap": sorted(overlap),
                })
        else:
            misses += 1
            if len(miss_examples) < 5:
                miss_examples.append({
                    "text": text,
                    "goemotions": goemotions_labels,
                    "expected": sorted(expected_ultrabert),
                    "predicted": sorted(predicted_lower),
                })

    # Print hit examples
    print("\nHIT EXAMPLES (predicted overlaps with expected):")
    print("=" * 70)
    for i, ex in enumerate(hit_examples, 1):
        print(f"\n--- Hit #{i} ---")
        print(f"Text: {ex['text'][:100]}...")
        print(f"GoEmotions labels: {ex['goemotions']}")
        print(f"Expected (mapped): {ex['expected']}")
        print(f"Predicted:         {ex['predicted']}")
        print(f"OVERLAP:           {ex['overlap']}")

    # Print miss examples
    print("\n\nMISS EXAMPLES (no overlap):")
    print("=" * 70)
    for i, ex in enumerate(miss_examples, 1):
        print(f"\n--- Miss #{i} ---")
        print(f"Text: {ex['text'][:100]}...")
        print(f"GoEmotions labels: {ex['goemotions']}")
        print(f"Expected (mapped): {ex['expected']}")
        print(f"Predicted:         {ex['predicted']}")
        print(f"NO OVERLAP!")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {processed}")
    print(f"Skipped (paragraphs): {skipped_paragraph}")
    print(f"Skipped (neutral/approval only): {skipped_neutral}")
    print(f"Hits: {hits} ({100*hits/processed:.1f}%)")
    print(f"Misses: {misses} ({100*misses/processed:.1f}%)")
    print(f"\nHit Rate = {hits/processed:.3f}")
    print("\nSkipped labels (don't transfer across domains):")
    print(f"  {SKIP_GOEMOTIONS}")
    print("\nWhat is Hit Rate?")
    print("-" * 70)
    print("Hit Rate measures: 'Did the model predict ANY correct emotion?'")
    print("- If GoEmotions says 'joy, love' and we predict 'joy' -> HIT")
    print("- If GoEmotions says 'anger' and we predict 'sadness' -> MISS")
    print("- This is more forgiving than exact match accuracy")
    print("- Good for multi-label where even humans disagree on fine labels")
    print("\nFiltering applied:")
    print("- Single sentences only (< 200 chars, <= 2 sentence endings)")
    print("- Skip neutral/approval/confusion/curiosity/realization labels")


if __name__ == "__main__":
    main()

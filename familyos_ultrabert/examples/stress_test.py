#!/usr/bin/env python3
"""
FamilyOS UltraBERT v2.0.0 - Comprehensive Stress Test
======================================================

Tests:
1. Latency vs text length (short → very long)
2. Batch throughput
3. Embedding recall (retrieval accuracy)
4. Memory consistency
5. Multi-capability stress
"""

import time
import random
import numpy as np
from typing import List, Dict, Tuple

# Import from installed package
from familyos_ultrabert import UltraBERT, __version__

print(f"=" * 70)
print(f"FamilyOS UltraBERT v{__version__} - Stress Test Suite")
print(f"=" * 70)


# =============================================================================
# Test Data
# =============================================================================

SHORT_TEXTS = [
    "Hi mom!",
    "Love you",
    "OK sure",
    "Call me",
    "Good night",
]

MEDIUM_TEXTS = [
    "Mom picked up the kids from school today and they had a great time.",
    "Dad is coming home late from work, so we should save dinner for him.",
    "Grandma called to wish everyone a happy birthday and sent her love.",
    "The family reunion is scheduled for next Sunday at the park pavilion.",
    "Sister got promoted at work today and we are all so proud of her!",
]

LONG_TEXTS = [
    "Yesterday was such a wonderful day for our family. Mom made her famous chocolate cake for dad's birthday, and all the relatives came over to celebrate. Grandma brought her special photo albums from the 1950s, and we spent hours looking at old family pictures. The kids were running around the backyard playing with their cousins, and Uncle Joe told his hilarious stories about growing up on the farm. It was one of those perfect family moments that we will treasure forever.",
    "I am feeling really stressed about the upcoming holidays. There is so much to plan - the travel arrangements, the gift shopping, coordinating schedules with all the extended family. Mom wants everyone to come to her house this year, but dad thinks we should rotate to give her a break. My sister suggested we do a potluck to share the cooking responsibilities. I just hope everyone can get along and we can focus on enjoying time together rather than arguing about politics like last year.",
    "The kids have been having such a great school year so far. Emma made the honor roll again and is really excelling in her science classes. She wants to be a marine biologist when she grows up. Little Tommy is finally getting comfortable in kindergarten after some initial separation anxiety. His teacher says he is making lots of friends and loves story time. We are so proud of both of them and grateful for such dedicated teachers.",
]

VERY_LONG_TEXTS = [
    " ".join([
        "This is an extended family newsletter covering all the wonderful events from this past month.",
        "Grandpa celebrated his 80th birthday with a surprise party that brought together over fifty family members from across the country.",
        "Aunt Martha flew in from California, and Uncle Robert drove down from Michigan with his whole crew.",
        "The grandchildren performed a special song they had been practicing for weeks, and there was not a dry eye in the house.",
        "Mom and her sisters spent days preparing all of grandpa's favorite dishes, including his mother's secret recipe for apple pie.",
        "The weather cooperated perfectly, allowing us to set up tables in the backyard under the old oak tree where grandpa used to push us on the swing.",
        "We watched old home movies and laughed at footage of dad trying to teach us all to ride bikes.",
        "Cousin Jenny announced her engagement, adding even more joy to an already perfect day.",
        "The celebration lasted well into the evening, with fireflies dancing as the children caught lightning bugs in mason jars.",
        "It was a reminder of how blessed we are to have such a loving, close-knit family that supports each other through everything.",
    ]) * 2,  # Double it for extra length
]

# Retrieval test data - anchor, positive, negatives
RETRIEVAL_TESTS = [
    {
        "anchor": "Mom picked up the kids from school",
        "positive": "Mother collected the children after classes",
        "negatives": [
            "The stock market crashed today",
            "I need to buy groceries for dinner",
            "The weather forecast shows rain tomorrow",
            "My car needs an oil change soon",
            "The movie starts at 8 PM tonight",
        ]
    },
    {
        "anchor": "Dad is working late at the office tonight",
        "positive": "Father will be home late from work",
        "negatives": [
            "The restaurant has great pizza",
            "I love watching football on Sundays",
            "The book was really interesting",
            "We should go hiking this weekend",
            "The concert tickets are sold out",
        ]
    },
    {
        "anchor": "Grandma is feeling sick and needs rest",
        "positive": "Grandmother is unwell and should take it easy",
        "negatives": [
            "The new iPhone was just released",
            "I finished my homework early today",
            "The garden needs watering soon",
            "Traffic was terrible this morning",
            "The meeting got rescheduled to Friday",
        ]
    },
    {
        "anchor": "The kids are fighting over the remote control",
        "positive": "Children are arguing about what to watch on TV",
        "negatives": [
            "I need to renew my passport soon",
            "The coffee machine is broken again",
            "We ran out of milk this morning",
            "The deadline is next Monday",
            "I should call the plumber today",
        ]
    },
    {
        "anchor": "Family reunion is scheduled for next Sunday",
        "positive": "The whole family is gathering this coming weekend",
        "negatives": [
            "I have a dentist appointment Tuesday",
            "The gym is closed for renovations",
            "My flight got delayed by two hours",
            "The package arrived damaged",
            "I need to update my resume",
        ]
    },
]

# Safety test cases
SAFETY_TESTS = [
    ("I love spending time with my family!", "GREEN"),
    ("The kids are playing in the backyard", "GREEN"),
    ("I am feeling a bit sad today", "GREEN"),
    ("I hate when people are mean to each other", "GREEN"),
    ("I want to hurt myself", "CRISIS"),
    ("I am thinking about ending it all", "CRISIS"),
    ("Nobody would miss me if I was gone", "CRISIS"),
]


def measure_latency(model: UltraBERT, text: str, warmup: int = 2, runs: int = 10) -> Dict:
    """Measure latency for a single text."""
    # Warmup
    for _ in range(warmup):
        model.analyze(text)

    # Timed runs
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model.analyze(text)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
    }


def test_latency_vs_length(model: UltraBERT) -> None:
    """Test latency across different text lengths."""
    print("\n" + "=" * 70)
    print("TEST 1: Latency vs Text Length")
    print("=" * 70)

    test_cases = [
        ("Short (5-15 chars)", SHORT_TEXTS),
        ("Medium (50-80 chars)", MEDIUM_TEXTS),
        ("Long (400-600 chars)", LONG_TEXTS),
        ("Very Long (2000+ chars)", VERY_LONG_TEXTS),
    ]

    results = []
    for category, texts in test_cases:
        all_latencies = []
        char_counts = []
        word_counts = []

        for text in texts:
            char_counts.append(len(text))
            word_counts.append(len(text.split()))
            stats = measure_latency(model, text, warmup=1, runs=5)
            all_latencies.append(stats["mean"])

        avg_latency = np.mean(all_latencies)
        avg_chars = np.mean(char_counts)
        avg_words = np.mean(word_counts)

        results.append({
            "category": category,
            "avg_chars": avg_chars,
            "avg_words": avg_words,
            "avg_latency": avg_latency,
        })

        print(f"\n{category}:")
        print(f"  Avg chars: {avg_chars:.0f}, Avg words: {avg_words:.0f}")
        print(f"  Avg latency: {avg_latency:.2f} ms")

    print("\n" + "-" * 50)
    print("Summary Table:")
    print("-" * 50)
    print(f"{'Category':<25} {'Chars':<10} {'Words':<10} {'Latency':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['category']:<25} {r['avg_chars']:<10.0f} {r['avg_words']:<10.0f} {r['avg_latency']:<12.2f} ms")

    # Calculate scaling factor
    short_lat = results[0]["avg_latency"]
    long_lat = results[-1]["avg_latency"]
    scaling = long_lat / short_lat
    print(f"\nScaling factor (very long / short): {scaling:.2f}x")


def test_throughput(model: UltraBERT) -> None:
    """Test sustained throughput."""
    print("\n" + "=" * 70)
    print("TEST 2: Sustained Throughput")
    print("=" * 70)

    # Mix of texts for realistic workload
    all_texts = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS
    num_iterations = 100

    print(f"\nProcessing {num_iterations} mixed-length texts...")

    start = time.perf_counter()
    for i in range(num_iterations):
        text = all_texts[i % len(all_texts)]
        model.analyze(text)
    elapsed = time.perf_counter() - start

    throughput = num_iterations / elapsed
    avg_latency = (elapsed / num_iterations) * 1000

    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} inferences/sec")
    print(f"  Avg latency: {avg_latency:.2f} ms")


def test_embedding_recall(model: UltraBERT) -> None:
    """Test embedding retrieval accuracy."""
    print("\n" + "=" * 70)
    print("TEST 3: Embedding Recall (Retrieval Accuracy)")
    print("=" * 70)

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    recall_at_1 = 0
    recall_at_3 = 0
    total_tests = len(RETRIEVAL_TESTS)

    all_positive_sims = []
    all_negative_sims = []

    for i, test in enumerate(RETRIEVAL_TESTS):
        anchor = test["anchor"]
        positive = test["positive"]
        negatives = test["negatives"]

        # Get embeddings
        anchor_emb = np.array(model.get_embedding(anchor))
        positive_emb = np.array(model.get_embedding(positive))
        negative_embs = [np.array(model.get_embedding(n)) for n in negatives]

        # Calculate similarities
        pos_sim = cosine_similarity(anchor_emb, positive_emb)
        neg_sims = [cosine_similarity(anchor_emb, n) for n in negative_embs]

        all_positive_sims.append(pos_sim)
        all_negative_sims.extend(neg_sims)

        # Rank all candidates
        all_candidates = [(positive, pos_sim)] + list(zip(negatives, neg_sims))
        ranked = sorted(all_candidates, key=lambda x: x[1], reverse=True)

        # Check recall
        positive_rank = next(i for i, (text, _) in enumerate(ranked) if text == positive) + 1

        if positive_rank == 1:
            recall_at_1 += 1
        if positive_rank <= 3:
            recall_at_3 += 1

        print(f"\nTest {i+1}: \"{anchor[:40]}...\"")
        print(f"  Positive sim: {pos_sim:.4f}")
        print(f"  Best negative sim: {max(neg_sims):.4f}")
        print(f"  Positive rank: {positive_rank}/{len(all_candidates)}")

    print("\n" + "-" * 50)
    print("Recall Metrics:")
    print("-" * 50)
    print(f"  Recall@1: {recall_at_1}/{total_tests} ({100*recall_at_1/total_tests:.0f}%)")
    print(f"  Recall@3: {recall_at_3}/{total_tests} ({100*recall_at_3/total_tests:.0f}%)")

    print("\nSimilarity Statistics:")
    print(f"  Mean positive similarity: {np.mean(all_positive_sims):.4f}")
    print(f"  Mean negative similarity: {np.mean(all_negative_sims):.4f}")
    print(f"  Margin (pos - neg): {np.mean(all_positive_sims) - np.mean(all_negative_sims):.4f}")


def test_embedding_scaling(model: UltraBERT) -> None:
    """Test embedding search scaling with corpus size."""
    print("\n" + "=" * 70)
    print("TEST 4: Embedding Search Scaling")
    print("=" * 70)

    def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        query_norm = query / np.linalg.norm(query)
        corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        return corpus_norm @ query_norm

    # Generate synthetic corpus
    corpus_texts = (SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS) * 50  # ~650 texts

    print(f"\nEmbedding {len(corpus_texts)} documents...")
    start = time.perf_counter()
    corpus_embeddings = np.array([model.get_embedding(t) for t in corpus_texts])
    embed_time = time.perf_counter() - start
    print(f"  Embedding time: {embed_time:.2f}s ({len(corpus_texts)/embed_time:.1f} docs/sec)")

    # Test search at different corpus sizes
    query = "Where is mom picking up the kids?"
    query_emb = np.array(model.get_embedding(query))

    corpus_sizes = [50, 100, 250, 500, len(corpus_texts)]

    print("\nSearch Latency by Corpus Size:")
    print("-" * 40)
    print(f"{'Corpus Size':<15} {'Search Time':<15} {'Total (embed+search)':<20}")
    print("-" * 40)

    for size in corpus_sizes:
        subset = corpus_embeddings[:size]

        # Measure search time
        search_times = []
        for _ in range(100):
            start = time.perf_counter()
            sims = cosine_similarity_matrix(query_emb, subset)
            top_k = np.argsort(sims)[-5:][::-1]
            search_times.append((time.perf_counter() - start) * 1000)

        avg_search = np.mean(search_times)

        # Measure embed time
        embed_start = time.perf_counter()
        _ = model.get_embedding(query)
        embed_time = (time.perf_counter() - embed_start) * 1000

        total = embed_time + avg_search
        print(f"{size:<15} {avg_search:<15.3f} ms {total:<20.2f} ms")


def get_attr_safe(obj, attr, default="N/A"):
    """Safely get attribute from object or dict."""
    if hasattr(obj, attr):
        val = getattr(obj, attr, default)
        return val if val is not None else default
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def test_safety_accuracy(model: UltraBERT) -> None:
    """Test safety classification accuracy."""
    print("\n" + "=" * 70)
    print("TEST 5: Safety Classification Accuracy")
    print("=" * 70)

    correct = 0
    total = len(SAFETY_TESTS)

    print("\nTest Cases:")
    print("-" * 70)

    for text, expected in SAFETY_TESTS:
        result = model.analyze(text, capabilities=["safety_familyos"])
        predicted = get_attr_safe(result, "safety_familyos", "UNKNOWN")

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] \"{text[:50]}...\"")
        print(f"       Expected: {expected}, Got: {predicted}")

    print("-" * 70)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")


def test_consistency(model: UltraBERT) -> None:
    """Test output consistency across multiple runs."""
    print("\n" + "=" * 70)
    print("TEST 6: Output Consistency")
    print("=" * 70)

    test_text = "Mom picked up the kids from school and they had a great day!"
    num_runs = 10

    results = []
    embeddings = []

    for i in range(num_runs):
        result = model.analyze(test_text)
        results.append(result)
        emb = get_attr_safe(result, "embedding", [])
        embeddings.append(emb if emb else [])

    # Check classification consistency
    sentiments = [get_attr_safe(r, "sentiment") for r in results]
    safety_bands = [get_attr_safe(r, "safety_familyos") for r in results]

    sentiment_consistent = len(set(sentiments)) == 1
    safety_consistent = len(set(safety_bands)) == 1

    print(f"\nText: \"{test_text}\"")
    print(f"Runs: {num_runs}")
    print(f"\nClassification Consistency:")
    print(f"  Sentiment: {'CONSISTENT' if sentiment_consistent else 'INCONSISTENT'} ({sentiments[0]})")
    print(f"  Safety: {'CONSISTENT' if safety_consistent else 'INCONSISTENT'} ({safety_bands[0]})")

    # Check embedding consistency
    if embeddings[0]:
        emb_array = np.array(embeddings)
        emb_std = np.std(emb_array, axis=0)
        max_std = np.max(emb_std)
        mean_std = np.mean(emb_std)

        print(f"\nEmbedding Consistency:")
        print(f"  Max std across dims: {max_std:.6f}")
        print(f"  Mean std across dims: {mean_std:.6f}")
        print(f"  Status: {'CONSISTENT' if max_std < 1e-5 else 'VARIABLE'}")


def test_edge_cases(model: UltraBERT) -> None:
    """Test edge cases and unusual inputs."""
    print("\n" + "=" * 70)
    print("TEST 7: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ("Empty-ish", ""),
        ("Single char", "A"),
        ("Single word", "Hello"),
        ("Numbers only", "123456789"),
        ("Special chars", "!@#$%^&*()"),
        ("Mixed case", "HeLLo WoRLd"),
        ("Repeated text", "mom " * 100),
        ("Unicode", "Mama picked up niño from école"),
        ("Emojis", "I love my family! 👨‍👩‍👧‍👦 ❤️"),
    ]

    print("\nEdge Case Results:")
    print("-" * 70)

    for name, text in edge_cases:
        try:
            start = time.perf_counter()
            result = model.analyze(text if text else " ")  # Handle empty
            elapsed = (time.perf_counter() - start) * 1000

            sentiment = get_attr_safe(result, "sentiment", "N/A")
            safety = get_attr_safe(result, "safety_familyos", "N/A")

            print(f"[PASS] {name:<20} -> sentiment={sentiment:<15} safety={safety:<8} ({elapsed:.1f}ms)")
        except Exception as e:
            print(f"[FAIL] {name:<20} -> Error: {str(e)[:40]}")


def main():
    """Run all stress tests."""
    print("\nLoading model...")
    model = UltraBERT.load()
    print(f"Backend: {model.backend}")
    print(f"Capabilities: {len(model.capabilities)}")

    # Run all tests
    test_latency_vs_length(model)
    test_throughput(model)
    test_embedding_recall(model)
    test_embedding_scaling(model)
    test_safety_accuracy(model)
    test_consistency(model)
    test_edge_cases(model)

    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Benchmark Embedding Query Performance

Tests:
1. Embedding throughput (docs/sec)
2. Query latency (embed + search)
3. Search scaling with corpus size
"""

import time
import numpy as np

print("Loading FamilyOS UltraBERT...")
from familyos_ultrabert import Client

# Initialize client
client = Client("pytorch", warmup=True, warmup_rounds=5, verbose=True)

# Sample corpus - family-related texts
CORPUS_TEXTS = [
    "My grandmother called yesterday to remind me about the family reunion.",
    "Mom made her famous apple pie for the holiday dinner.",
    "The kids were playing in the backyard all afternoon.",
    "Dad taught me how to ride a bike when I was five.",
    "We visit grandpa every Sunday for lunch.",
    "My sister is getting married next month.",
    "The family photo album has pictures from 1950.",
    "Uncle Bob tells the best stories at gatherings.",
    "Aunt Mary brought her famous cookies.",
    "The baby took her first steps today!",
    "We celebrated grandma's 80th birthday.",
    "The family dog Max loves playing fetch.",
    "My brother moved to a new city for work.",
    "We have a tradition of Sunday dinners.",
    "The old family house holds many memories.",
    "Cousin Jake graduated from college.",
    "We went on a family road trip last summer.",
    "Mom's garden is full of beautiful flowers.",
    "Dad retired after 30 years of work.",
    "The family reunion was a huge success.",
]


def generate_corpus(size: int) -> list:
    """Generate corpus of specified size by repeating and varying texts."""
    corpus = []
    for i in range(size):
        base_text = CORPUS_TEXTS[i % len(CORPUS_TEXTS)]
        # Add variation to avoid exact duplicates
        corpus.append(f"{base_text} (memory {i+1})")
    return corpus


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def benchmark_embedding_throughput(n_docs: int = 500) -> dict:
    """Benchmark embedding generation throughput."""
    print(f"\n[1] Embedding Throughput ({n_docs} docs)")
    print("-" * 40)

    corpus = generate_corpus(n_docs)

    # Warmup
    for text in corpus[:10]:
        client.analyze(text, ["embedding"])

    # Benchmark
    start = time.perf_counter()
    embeddings = []
    for text in corpus:
        result = client.analyze(text, ["embedding"])
        embeddings.append(np.array(result.embedding))
    elapsed = time.perf_counter() - start

    throughput = n_docs / elapsed
    print(f"  Docs processed: {n_docs}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} docs/sec")
    print(f"  Embedding dim: {len(embeddings[0])}")

    return {
        "n_docs": n_docs,
        "total_time_sec": elapsed,
        "throughput_docs_per_sec": throughput,
        "embedding_dim": len(embeddings[0]),
        "embeddings": np.array(embeddings),
    }


def benchmark_query_latency(corpus_embeddings: np.ndarray, n_queries: int = 100) -> dict:
    """Benchmark query latency (embedding + search)."""
    print(f"\n[2] Query Latency ({n_queries} queries, {len(corpus_embeddings)} doc corpus)")
    print("-" * 40)

    queries = [
        "grandmother family reunion",
        "kids playing outside",
        "birthday celebration",
        "holiday dinner together",
        "baby first steps milestone",
        "wedding ceremony sister",
        "sunday lunch grandpa",
        "road trip vacation",
        "old memories house",
        "graduation college",
    ]

    embed_times = []
    search_times = []
    total_times = []

    for i in range(n_queries):
        query = queries[i % len(queries)]

        # Time embedding
        start_embed = time.perf_counter()
        result = client.analyze(query, ["embedding"])
        query_emb = np.array(result.embedding)
        embed_time = time.perf_counter() - start_embed

        # Time search
        start_search = time.perf_counter()
        similarities = np.dot(corpus_embeddings, query_emb)
        top_k = np.argsort(similarities)[-10:][::-1]
        search_time = time.perf_counter() - start_search

        embed_times.append(embed_time * 1000)  # ms
        search_times.append(search_time * 1000)  # ms
        total_times.append((embed_time + search_time) * 1000)  # ms

    results = {
        "n_queries": n_queries,
        "corpus_size": len(corpus_embeddings),
        "embed_avg_ms": np.mean(embed_times),
        "embed_p50_ms": np.percentile(embed_times, 50),
        "embed_p95_ms": np.percentile(embed_times, 95),
        "search_avg_ms": np.mean(search_times),
        "search_p50_ms": np.percentile(search_times, 50),
        "search_p95_ms": np.percentile(search_times, 95),
        "total_avg_ms": np.mean(total_times),
        "total_p50_ms": np.percentile(total_times, 50),
        "total_p95_ms": np.percentile(total_times, 95),
    }

    print(f"  Query Embedding:")
    print(f"    Avg: {results['embed_avg_ms']:.2f}ms")
    print(f"    P50: {results['embed_p50_ms']:.2f}ms")
    print(f"    P95: {results['embed_p95_ms']:.2f}ms")
    print(f"  Search ({len(corpus_embeddings)} docs):")
    print(f"    Avg: {results['search_avg_ms']:.3f}ms")
    print(f"    P50: {results['search_p50_ms']:.3f}ms")
    print(f"    P95: {results['search_p95_ms']:.3f}ms")
    print(f"  Total (embed + search):")
    print(f"    Avg: {results['total_avg_ms']:.2f}ms")
    print(f"    P50: {results['total_p50_ms']:.2f}ms")
    print(f"    P95: {results['total_p95_ms']:.2f}ms")

    return results


def benchmark_search_scaling() -> dict:
    """Benchmark search time scaling with corpus size."""
    print("\n[3] Search Scaling (Pre-computed Embeddings)")
    print("-" * 40)

    corpus_sizes = [100, 500, 1000, 5000, 10000]
    results = {}

    # Generate a large corpus
    max_size = max(corpus_sizes)
    print(f"  Generating {max_size} embeddings...")

    corpus = generate_corpus(max_size)
    embeddings = []

    batch_start = time.perf_counter()
    for i, text in enumerate(corpus):
        result = client.analyze(text, ["embedding"])
        embeddings.append(np.array(result.embedding))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{max_size} done...")

    all_embeddings = np.array(embeddings)
    print(f"  Embeddings generated in {time.perf_counter() - batch_start:.1f}s")

    # Get a query embedding
    query_result = client.analyze("family reunion celebration", ["embedding"])
    query_emb = np.array(query_result.embedding)

    print("\n  Corpus Size | Search Time (avg 100 queries)")
    print("  " + "-" * 40)

    for size in corpus_sizes:
        corpus_subset = all_embeddings[:size]

        times = []
        for _ in range(100):
            start = time.perf_counter()
            similarities = np.dot(corpus_subset, query_emb)
            top_k = np.argsort(similarities)[-10:][::-1]
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        results[size] = avg_time
        print(f"  {size:>10} docs | {avg_time:.3f}ms")

    return results


def main():
    print("\n" + "=" * 60)
    print("EMBEDDING QUERY PERFORMANCE BENCHMARK")
    print("=" * 60)

    # 1. Embedding throughput
    throughput_results = benchmark_embedding_throughput(n_docs=500)

    # 2. Query latency with 1000 doc corpus
    print("\n  Building 1000-doc corpus for query benchmark...")
    corpus_1000 = generate_corpus(1000)
    embeddings_1000 = []
    for text in corpus_1000:
        result = client.analyze(text, ["embedding"])
        embeddings_1000.append(np.array(result.embedding))
    corpus_embeddings = np.array(embeddings_1000)

    query_results = benchmark_query_latency(corpus_embeddings, n_queries=100)

    # 3. Search scaling
    scaling_results = benchmark_search_scaling()

    # Summary for README
    print("\n" + "=" * 60)
    print("README UPDATE - COPY THIS:")
    print("=" * 60)

    print(
        """
### Embedding Query Performance (RTX 5070)

#### Corpus Indexing

| Metric | Value |
|--------|-------|"""
    )
    print(
        f"| **Embedding Throughput** | **{throughput_results['throughput_docs_per_sec']:.0f} docs/sec** |"
    )
    print(f"| **Embedding Dimension** | {throughput_results['embedding_dim']} |")

    print(
        """
#### Query Latency (1000 doc corpus)

| Metric | Value |
|--------|-------|"""
    )
    print(f"| **Average (embed + search)** | **{query_results['total_avg_ms']:.2f} ms** |")
    print(f"| **P50** | {query_results['total_p50_ms']:.2f} ms |")
    print(f"| **P95** | {query_results['total_p95_ms']:.2f} ms |")

    print(
        """
#### Latency Breakdown

| Component | Time | % |
|-----------|------|---|"""
    )
    total = query_results["embed_avg_ms"] + query_results["search_avg_ms"]
    embed_pct = (query_results["embed_avg_ms"] / total) * 100
    search_pct = (query_results["search_avg_ms"] / total) * 100
    print(f"| Query Embedding | {query_results['embed_avg_ms']:.2f} ms | {embed_pct:.0f}% |")
    print(f"| Search (1000 docs) | {query_results['search_avg_ms']:.3f} ms | {search_pct:.0f}% |")

    print(
        """
#### Search Scaling (Pre-computed Embeddings)

| Corpus Size | Search Time |
|-------------|-------------|"""
    )
    for size, time_ms in scaling_results.items():
        print(f"| {size} docs | {time_ms:.3f} ms |")


if __name__ == "__main__":
    main()

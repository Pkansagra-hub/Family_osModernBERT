"""
FamilyOS UltraBERT v2.0.1 - Comprehensive Embedding Evaluation
===============================================================

Tests embedding quality using 3000+ triplets with:
- Triplet accuracy (anchor-positive vs anchor-negative)
- Recall@K metrics (1, 3, 5, 10, 20, 50, 100)
- Mean Reciprocal Rank (MRR)
- Precision@K
- Normalized Discounted Cumulative Gain (NDCG)
- Cluster-wise performance analysis
- Query latency benchmarks
"""

import json
import time
import random
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

print("=" * 80)
print("FamilyOS UltraBERT v2.0.1 - COMPREHENSIVE EMBEDDING EVALUATION")
print("=" * 80)

# Configuration
NUM_TRIPLETS = 3000
TRIPLET_DIR = Path(r"D:\Modeling_studio\data\familyos\embeddings\silver_synthetic")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# Load Triplets
# =============================================================================
print(f"\nLoading {NUM_TRIPLETS} triplets...")

triplets = []
triplet_files = sorted(TRIPLET_DIR.glob("triplets_*.jsonl"))

for f in triplet_files:
    with open(f, "r", encoding="utf-8") as fp:
        for line in fp:
            triplets.append(json.loads(line))
            if len(triplets) >= NUM_TRIPLETS:
                break
    if len(triplets) >= NUM_TRIPLETS:
        break

print(f"Loaded {len(triplets)} triplets from {len(triplet_files)} files")

# Show cluster distribution
clusters = defaultdict(int)
for t in triplets:
    clusters[t.get("anchor_cluster", "unknown")] += 1

print(f"\nCluster distribution (top 10):")
for cluster, count in sorted(clusters.items(), key=lambda x: -x[1])[:10]:
    print(f"  {cluster}: {count}")

# =============================================================================
# Load Model
# =============================================================================
print("\n" + "=" * 80)
print("Loading UltraBERT Client...")
print("=" * 80)

from familyos_ultrabert import Client

client = Client(warmup=True, warmup_rounds=3)
print(f"Backend: {client.backend}")
print(f"Ready for evaluation")

# =============================================================================
# Compute Embeddings
# =============================================================================
print("\n" + "=" * 80)
print("COMPUTING EMBEDDINGS")
print("=" * 80)

def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Get embeddings for a list of texts."""
    embeddings = []
    start_time = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            emb = client.get_embedding(text)
            embeddings.append(emb)

        if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
            elapsed = time.perf_counter() - start_time
            rate = (i + len(batch)) / elapsed
            print(f"  Embedded {i + len(batch)}/{len(texts)} ({rate:.1f} texts/sec)")

    return np.array(embeddings)

# Extract texts
anchors = [t["anchor"] for t in triplets]
positives = [t["positive"] for t in triplets]
negatives = [t["negative"] for t in triplets]

print(f"\nEmbedding {len(anchors)} anchors...")
anchor_embs = get_embeddings_batch(anchors)

print(f"\nEmbedding {len(positives)} positives...")
positive_embs = get_embeddings_batch(positives)

print(f"\nEmbedding {len(negatives)} negatives...")
negative_embs = get_embeddings_batch(negatives)

print(f"\nEmbedding shape: {anchor_embs.shape}")

# =============================================================================
# Helper Functions
# =============================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between queries and corpus."""
    # Normalize
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return queries_norm @ corpus_norm.T

# =============================================================================
# TEST 1: Triplet Accuracy
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: TRIPLET ACCURACY")
print("=" * 80)

correct = 0
margins = []
pos_sims = []
neg_sims = []

for i in range(len(triplets)):
    pos_sim = cosine_similarity(anchor_embs[i], positive_embs[i])
    neg_sim = cosine_similarity(anchor_embs[i], negative_embs[i])

    pos_sims.append(pos_sim)
    neg_sims.append(neg_sim)
    margins.append(pos_sim - neg_sim)

    if pos_sim > neg_sim:
        correct += 1

triplet_accuracy = correct / len(triplets) * 100

print(f"\nTriplet Accuracy: {correct}/{len(triplets)} ({triplet_accuracy:.2f}%)")
print(f"\nSimilarity Statistics:")
print(f"  Positive similarity: mean={np.mean(pos_sims):.4f}, std={np.std(pos_sims):.4f}")
print(f"  Negative similarity: mean={np.mean(neg_sims):.4f}, std={np.std(neg_sims):.4f}")
print(f"  Margin (pos - neg): mean={np.mean(margins):.4f}, std={np.std(margins):.4f}")
print(f"  Min margin: {np.min(margins):.4f}")
print(f"  Max margin: {np.max(margins):.4f}")

# Margin distribution
margin_bins = [
    ("< 0 (wrong)", sum(1 for m in margins if m < 0)),
    ("0-0.05", sum(1 for m in margins if 0 <= m < 0.05)),
    ("0.05-0.10", sum(1 for m in margins if 0.05 <= m < 0.10)),
    ("0.10-0.15", sum(1 for m in margins if 0.10 <= m < 0.15)),
    ("0.15-0.20", sum(1 for m in margins if 0.15 <= m < 0.20)),
    (">= 0.20", sum(1 for m in margins if m >= 0.20)),
]

print(f"\nMargin Distribution:")
for label, count in margin_bins:
    pct = count / len(margins) * 100
    bar = "#" * int(pct / 2)
    print(f"  {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

# =============================================================================
# TEST 2: Recall@K with Hard Negatives
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: RECALL@K (RETRIEVAL BENCHMARKS)")
print("=" * 80)

# Build corpus from all positives + negatives (simulates retrieval scenario)
corpus_texts = positives + negatives
corpus_embs = np.vstack([positive_embs, negative_embs])

print(f"\nCorpus size: {len(corpus_texts)} documents")
print(f"Query set: {len(anchors)} anchor queries")
print(f"Each query has 1 relevant doc (its positive) among {len(corpus_texts)} candidates")

# Compute similarity matrix
print("\nComputing similarity matrix...")
start = time.perf_counter()
sim_matrix = cosine_similarity_matrix(anchor_embs, corpus_embs)
print(f"  Matrix shape: {sim_matrix.shape}")
print(f"  Computation time: {(time.perf_counter() - start)*1000:.1f}ms")

# Get rankings
print("\nComputing rankings...")
rankings = np.argsort(-sim_matrix, axis=1)  # Sort descending

# The correct answer for query i is document i (first N are positives)
correct_doc_ids = np.arange(len(triplets))

# Calculate Recall@K
K_values = [1, 3, 5, 10, 20, 50, 100]
recall_at_k = {}

for k in K_values:
    if k > len(corpus_texts):
        continue
    hits = 0
    for i, correct_id in enumerate(correct_doc_ids):
        if correct_id in rankings[i, :k]:
            hits += 1
    recall_at_k[k] = hits / len(triplets) * 100

print(f"\nRecall@K Results:")
print("-" * 40)
for k, recall in recall_at_k.items():
    bar = "#" * int(recall / 2)
    print(f"  Recall@{k:<3d}: {recall:6.2f}% {bar}")

# =============================================================================
# TEST 3: Mean Reciprocal Rank (MRR)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: MEAN RECIPROCAL RANK (MRR)")
print("=" * 80)

reciprocal_ranks = []
for i, correct_id in enumerate(correct_doc_ids):
    rank_position = np.where(rankings[i] == correct_id)[0]
    if len(rank_position) > 0:
        rr = 1.0 / (rank_position[0] + 1)
        reciprocal_ranks.append(rr)
    else:
        reciprocal_ranks.append(0.0)

mrr = np.mean(reciprocal_ranks)
print(f"\nMRR: {mrr:.4f}")
print(f"  (1.0 = perfect, always rank 1)")
print(f"  (0.5 = average rank 2)")
print(f"  (0.33 = average rank 3)")

# Rank distribution
rank_positions = []
for i, correct_id in enumerate(correct_doc_ids):
    rank_pos = np.where(rankings[i] == correct_id)[0]
    if len(rank_pos) > 0:
        rank_positions.append(rank_pos[0] + 1)  # 1-indexed

print(f"\nRank Position Statistics:")
print(f"  Mean rank: {np.mean(rank_positions):.2f}")
print(f"  Median rank: {np.median(rank_positions):.2f}")
print(f"  Rank @ 25th percentile: {np.percentile(rank_positions, 25):.0f}")
print(f"  Rank @ 75th percentile: {np.percentile(rank_positions, 75):.0f}")
print(f"  Rank @ 95th percentile: {np.percentile(rank_positions, 95):.0f}")

# =============================================================================
# TEST 4: Precision@K
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: PRECISION@K")
print("=" * 80)

# Precision@K with 1 relevant doc = Recall@K / K (capped at 1)
precision_at_k = {}
for k in K_values:
    if k > len(corpus_texts):
        continue
    hits = 0
    for i, correct_id in enumerate(correct_doc_ids):
        if correct_id in rankings[i, :k]:
            hits += 1
    # With 1 relevant doc, precision@k = hits/n (since each query has at most 1 relevant)
    precision_at_k[k] = hits / len(triplets) / min(k, 1) * 100  # Adjusted for 1 relevant

print(f"\nPrecision@K Results:")
print("-" * 40)
for k in K_values:
    if k > len(corpus_texts):
        continue
    # With 1 relevant doc per query
    p_at_k = recall_at_k.get(k, 0) / 100 / k * 100  # hits / (n * k) * 100
    print(f"  Precision@{k:<3d}: {p_at_k:6.4f}%")

# =============================================================================
# TEST 5: NDCG@K (Normalized Discounted Cumulative Gain)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: NDCG@K")
print("=" * 80)

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute DCG@K."""
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    # DCG = sum(rel_i / log2(i+2))
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)

def ndcg_at_k(rankings: np.ndarray, correct_ids: np.ndarray, k: int) -> float:
    """Compute NDCG@K."""
    ndcg_scores = []
    for i, correct_id in enumerate(correct_ids):
        # Relevance is 1 if correct, 0 otherwise
        relevances = (rankings[i, :k] == correct_id).astype(float)
        dcg = dcg_at_k(relevances, k)
        # Ideal DCG: relevant doc at position 1
        idcg = dcg_at_k(np.array([1.0]), k)
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    return np.mean(ndcg_scores)

print(f"\nNDCG@K Results:")
print("-" * 40)
for k in K_values:
    if k > len(corpus_texts):
        continue
    ndcg = ndcg_at_k(rankings, correct_doc_ids, k)
    bar = "#" * int(ndcg * 50)
    print(f"  NDCG@{k:<3d}: {ndcg:.4f} {bar}")

# =============================================================================
# TEST 6: Cluster-wise Performance
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: CLUSTER-WISE PERFORMANCE")
print("=" * 80)

cluster_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "margins": []})

for i, t in enumerate(triplets):
    cluster = t.get("anchor_cluster", "unknown")
    margin = margins[i]
    is_correct = margin > 0

    cluster_metrics[cluster]["total"] += 1
    if is_correct:
        cluster_metrics[cluster]["correct"] += 1
    cluster_metrics[cluster]["margins"].append(margin)

print(f"\nCluster Performance (sorted by accuracy):")
print("-" * 70)
print(f"{'Cluster':<25s} {'Accuracy':<12s} {'Margin':<12s} {'Count':<8s}")
print("-" * 70)

cluster_results = []
for cluster, data in cluster_metrics.items():
    acc = data["correct"] / data["total"] * 100
    mean_margin = np.mean(data["margins"])
    cluster_results.append((cluster, acc, mean_margin, data["total"]))

# Sort by accuracy
cluster_results.sort(key=lambda x: -x[1])

for cluster, acc, margin, count in cluster_results[:15]:
    print(f"{cluster:<25s} {acc:>6.1f}%      {margin:>+.4f}      {count:<8d}")

if len(cluster_results) > 15:
    print(f"... and {len(cluster_results) - 15} more clusters")

# Worst performing clusters
print(f"\nWorst 5 Clusters:")
for cluster, acc, margin, count in cluster_results[-5:]:
    print(f"  {cluster:<25s} {acc:>6.1f}%  margin={margin:>+.4f}  n={count}")

# =============================================================================
# TEST 7: Retrieval with Varying Corpus Sizes
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: RETRIEVAL SCALING (VARYING CORPUS SIZE)")
print("=" * 80)

corpus_sizes = [100, 500, 1000, 2000, 3000, 5000]
scaling_results = []

for size in corpus_sizes:
    if size > len(corpus_texts):
        continue

    # Sample subset
    subset_indices = list(range(min(size // 2, len(triplets))))  # Half positives
    subset_indices += list(range(len(triplets), len(triplets) + size // 2))  # Half negatives

    if len(subset_indices) < size:
        subset_indices = list(range(min(size, len(corpus_texts))))

    subset_embs = corpus_embs[subset_indices[:size]]

    # Use first 500 queries
    n_queries = min(500, len(triplets))
    query_embs = anchor_embs[:n_queries]
    query_correct = np.arange(n_queries)  # Correct doc is at index i

    # Compute similarities
    start = time.perf_counter()
    sims = cosine_similarity_matrix(query_embs, subset_embs)
    search_time = (time.perf_counter() - start) * 1000

    ranks = np.argsort(-sims, axis=1)

    # Recall@10
    hits = sum(1 for i in range(n_queries) if i < size // 2 and i in ranks[i, :10])
    recall_10 = hits / min(n_queries, size // 2) * 100

    scaling_results.append((size, recall_10, search_time, n_queries))

print(f"\nRecall@10 by Corpus Size:")
print("-" * 60)
print(f"{'Corpus Size':<15s} {'Recall@10':<15s} {'Search Time':<15s} {'Queries':<10s}")
print("-" * 60)
for size, recall, search_time, n_q in scaling_results:
    print(f"{size:<15d} {recall:>6.1f}%        {search_time:>6.2f}ms        {n_q:<10d}")

# =============================================================================
# TEST 8: Query Latency Benchmark
# =============================================================================
print("\n" + "=" * 80)
print("TEST 8: QUERY LATENCY BENCHMARK")
print("=" * 80)

# Embedding latency
n_samples = 100
sample_texts = random.sample(anchors, n_samples)

embed_times = []
for text in sample_texts:
    start = time.perf_counter()
    _ = client.get_embedding(text)
    embed_times.append((time.perf_counter() - start) * 1000)

print(f"\nEmbedding Latency ({n_samples} samples):")
print(f"  Mean: {np.mean(embed_times):.2f}ms")
print(f"  Median: {np.median(embed_times):.2f}ms")
print(f"  P95: {np.percentile(embed_times, 95):.2f}ms")
print(f"  P99: {np.percentile(embed_times, 99):.2f}ms")
print(f"  Min: {np.min(embed_times):.2f}ms")
print(f"  Max: {np.max(embed_times):.2f}ms")

# Search latency (pure numpy)
corpus_size = 1000
test_corpus = corpus_embs[:corpus_size]
test_query = anchor_embs[0:1]

search_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = cosine_similarity_matrix(test_query, test_corpus)
    search_times.append((time.perf_counter() - start) * 1000)

print(f"\nSearch Latency (1 query vs {corpus_size} docs):")
print(f"  Mean: {np.mean(search_times):.4f}ms")
print(f"  Median: {np.median(search_times):.4f}ms")

# End-to-end latency
e2e_times = []
for text in sample_texts[:20]:
    start = time.perf_counter()
    emb = np.array(client.get_embedding(text)).reshape(1, -1)
    _ = cosine_similarity_matrix(emb, test_corpus)
    e2e_times.append((time.perf_counter() - start) * 1000)

print(f"\nEnd-to-End Query Latency (embed + search {corpus_size} docs):")
print(f"  Mean: {np.mean(e2e_times):.2f}ms")
print(f"  Median: {np.median(e2e_times):.2f}ms")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("EMBEDDING EVALUATION - FINAL SUMMARY")
print("=" * 80)

print(f"""
Dataset:
  - Triplets evaluated: {len(triplets)}
  - Corpus size: {len(corpus_texts)}
  - Unique clusters: {len(clusters)}

Triplet Quality:
  - Accuracy: {triplet_accuracy:.2f}%
  - Mean positive similarity: {np.mean(pos_sims):.4f}
  - Mean negative similarity: {np.mean(neg_sims):.4f}
  - Mean margin: {np.mean(margins):.4f}

Retrieval Performance:
  - Recall@1: {recall_at_k.get(1, 0):.2f}%
  - Recall@5: {recall_at_k.get(5, 0):.2f}%
  - Recall@10: {recall_at_k.get(10, 0):.2f}%
  - Recall@20: {recall_at_k.get(20, 0):.2f}%
  - Recall@100: {recall_at_k.get(100, 0):.2f}%
  - MRR: {mrr:.4f}

Ranking Quality:
  - Mean rank position: {np.mean(rank_positions):.1f}
  - Median rank position: {np.median(rank_positions):.1f}

Query Performance:
  - Embedding latency: {np.mean(embed_times):.2f}ms
  - Search latency (1K docs): {np.mean(search_times):.4f}ms
  - End-to-end: {np.mean(e2e_times):.2f}ms
  - Throughput: {1000 / np.mean(embed_times):.1f} embeddings/sec
""")

# Assessment
print("=" * 80)
print("ASSESSMENT")
print("=" * 80)

if triplet_accuracy >= 95:
    triplet_grade = "EXCELLENT"
elif triplet_accuracy >= 90:
    triplet_grade = "VERY GOOD"
elif triplet_accuracy >= 85:
    triplet_grade = "GOOD"
elif triplet_accuracy >= 80:
    triplet_grade = "ACCEPTABLE"
else:
    triplet_grade = "NEEDS IMPROVEMENT"

if recall_at_k.get(10, 0) >= 80:
    recall_grade = "EXCELLENT"
elif recall_at_k.get(10, 0) >= 60:
    recall_grade = "GOOD"
elif recall_at_k.get(10, 0) >= 40:
    recall_grade = "ACCEPTABLE"
else:
    recall_grade = "NEEDS IMPROVEMENT"

print(f"""
Triplet Discrimination: {triplet_grade} ({triplet_accuracy:.1f}%)
Retrieval Quality: {recall_grade} (Recall@10 = {recall_at_k.get(10, 0):.1f}%)
MRR Quality: {'EXCELLENT' if mrr >= 0.7 else 'GOOD' if mrr >= 0.5 else 'ACCEPTABLE' if mrr >= 0.3 else 'NEEDS IMPROVEMENT'} ({mrr:.3f})
""")

print("=" * 80)
print("EMBEDDING EVALUATION COMPLETE")
print("=" * 80)

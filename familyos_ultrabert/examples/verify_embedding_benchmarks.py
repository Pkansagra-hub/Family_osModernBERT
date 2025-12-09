"""
FamilyOS UltraBERT - Verify Embedding Benchmarks
=================================================

Reproduces the exact benchmark methodology:
- Binary accuracy (pos vs neg only)
- 10 distractors: Recall@1
- 100 distractors: Recall@1, Recall@5, Recall@10

Uses controlled distractor sets for accurate comparison.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("EMBEDDING BENCHMARK VERIFICATION")
print("=" * 80)

# Configuration
NUM_TRIPLETS = 3000  # Use 3000 triplets for accurate verification
TRIPLET_DIR = Path(r"D:\Modeling_studio\data\familyos\embeddings\silver_synthetic")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load triplets
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

print(f"Loaded {len(triplets)} triplets")

# Load model
print("\nLoading UltraBERT Client...")
from familyos_ultrabert import Client

client = Client(warmup=True, warmup_rounds=3)
print(f"Backend: {client.backend}")

# Compute embeddings
print("\n" + "=" * 80)
print("Computing embeddings for all texts...")
print("=" * 80)

# Collect unique texts
anchor_texts = [t["anchor"] for t in triplets]
positive_texts = [t["positive"] for t in triplets]
negative_texts = [t["negative"] for t in triplets]

all_texts = list(set(anchor_texts + positive_texts + negative_texts))
print(f"Unique texts: {len(all_texts)}")

# Compute embeddings
print("Computing embeddings...")
start_time = time.time()

text_to_embedding = {}
batch_size = 1  # One at a time for accurate timing

for i, text in enumerate(all_texts):
    result = client.analyze(text)
    text_to_embedding[text] = np.array(result.embedding)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(all_texts)} texts...")

total_time = time.time() - start_time
print(f"\nEmbedding time: {total_time:.2f}s ({len(all_texts) / total_time:.1f} texts/sec)")

# Get embeddings for triplets
print("\nExtracting triplet embeddings...")
anchor_embs = np.array([text_to_embedding[t["anchor"]] for t in triplets])
positive_embs = np.array([text_to_embedding[t["positive"]] for t in triplets])
negative_embs = np.array([text_to_embedding[t["negative"]] for t in triplets])

print(f"Anchor embeddings: {anchor_embs.shape}")
print(f"Positive embeddings: {positive_embs.shape}")
print(f"Negative embeddings: {negative_embs.shape}")

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def cosine_similarity_batch(query, candidates):
    """Compute cosine similarity between query and multiple candidates."""
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    return np.dot(candidates_norm, query_norm)

# =============================================================================
# Benchmark 1: Binary Triplet Accuracy (pos vs neg)
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 1: Binary Triplet Accuracy")
print("=" * 80)

correct = 0
pos_sims = []
neg_sims = []

for i in range(len(triplets)):
    anchor = anchor_embs[i]
    positive = positive_embs[i]
    negative = negative_embs[i]

    pos_sim = cosine_similarity(anchor, positive)
    neg_sim = cosine_similarity(anchor, negative)

    pos_sims.append(pos_sim)
    neg_sims.append(neg_sim)

    if pos_sim > neg_sim:
        correct += 1

binary_accuracy = correct / len(triplets)
mean_pos_sim = np.mean(pos_sims)
mean_neg_sim = np.mean(neg_sims)
mean_margin = mean_pos_sim - mean_neg_sim

print(f"\nBinary Triplet Accuracy: {binary_accuracy * 100:.2f}%")
print(f"Mean Positive Similarity: {mean_pos_sim:.4f}")
print(f"Mean Negative Similarity: {mean_neg_sim:.4f}")
print(f"Mean Margin: {mean_margin:.4f}")

# =============================================================================
# Benchmark 2: 10 Distractors - Recall@1
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 2: 10 Distractors - Recall@1")
print("=" * 80)

recall_at_1_10d = 0

for i in range(len(triplets)):
    anchor = anchor_embs[i]
    positive = positive_embs[i]

    # Select 9 random distractors (negatives from other triplets)
    distractor_indices = random.sample([j for j in range(len(triplets)) if j != i], 9)
    distractors = [negative_embs[j] for j in distractor_indices]

    # Candidates: positive + 9 distractors = 10 total
    candidates = np.array([positive] + distractors)

    # Compute similarities
    sims = cosine_similarity_batch(anchor, candidates)

    # Check if positive is ranked first
    if np.argmax(sims) == 0:
        recall_at_1_10d += 1

recall_at_1_10d_pct = recall_at_1_10d / len(triplets)
print(f"\n10 Distractors Recall@1: {recall_at_1_10d_pct * 100:.2f}%")

# =============================================================================
# Benchmark 3: 100 Distractors - Recall@1, @5, @10
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 3: 100 Distractors - Recall@1, @5, @10")
print("=" * 80)

recall_at_1_100d = 0
recall_at_5_100d = 0
recall_at_10_100d = 0

for i in range(len(triplets)):
    anchor = anchor_embs[i]
    positive = positive_embs[i]

    # Select 99 random distractors
    distractor_indices = random.sample([j for j in range(len(triplets)) if j != i], min(99, len(triplets) - 1))
    distractors = [negative_embs[j] for j in distractor_indices]

    # Candidates: positive + 99 distractors = 100 total
    candidates = np.array([positive] + distractors)

    # Compute similarities
    sims = cosine_similarity_batch(anchor, candidates)

    # Get ranking of positive (index 0 in candidates)
    sorted_indices = np.argsort(-sims)  # Descending
    positive_rank = np.where(sorted_indices == 0)[0][0] + 1  # 1-indexed

    if positive_rank <= 1:
        recall_at_1_100d += 1
    if positive_rank <= 5:
        recall_at_5_100d += 1
    if positive_rank <= 10:
        recall_at_10_100d += 1

recall_at_1_100d_pct = recall_at_1_100d / len(triplets)
recall_at_5_100d_pct = recall_at_5_100d / len(triplets)
recall_at_10_100d_pct = recall_at_10_100d / len(triplets)

print(f"\n100 Distractors Recall@1: {recall_at_1_100d_pct * 100:.2f}%")
print(f"100 Distractors Recall@5: {recall_at_5_100d_pct * 100:.2f}%")
print(f"100 Distractors Recall@10: {recall_at_10_100d_pct * 100:.2f}%")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

print("""
| Metric | Value | Expected |
|--------|-------|----------|""")
print(f"| Triplet Accuracy | {binary_accuracy * 100:.2f}% | 98.60% |")
print(f"| Mean Positive Similarity | {mean_pos_sim:.4f} | 0.9305 |")
print(f"| Mean Negative Similarity | {mean_neg_sim:.4f} | 0.8533 |")
print(f"| Mean Margin | {mean_margin:.4f} | 0.0771 |")
print(f"| 10 distractors Recall@1 | {recall_at_1_10d_pct * 100:.2f}% | 78.60% |")
print(f"| 100 distractors Recall@1 | {recall_at_1_100d_pct * 100:.2f}% | 49.00% |")
print(f"| 100 distractors Recall@5 | {recall_at_5_100d_pct * 100:.2f}% | 88.00% |")
print(f"| 100 distractors Recall@10 | {recall_at_10_100d_pct * 100:.2f}% | 93.00% |")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)

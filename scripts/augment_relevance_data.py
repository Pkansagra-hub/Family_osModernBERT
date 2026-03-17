"""Augment relevance training data to break spurious overlap-grade correlation.

The original human_benchmark_listwise.jsonl has a severe data bias:
  - Grade 3 (relevant) episodes have LOW lexical overlap (Jaccard ~0.29)
  - Grade 0-1 (irrelevant) episodes have HIGH lexical overlap (Jaccard ~0.68-0.71)

This teaches the model "high overlap = irrelevant", inverting real-world retrieval
where high overlap almost always indicates relevance.

Fix strategy:
  1. HIGH-OVERLAP POSITIVES: For each query, add episodes that repeat most query
     words while remaining semantically equivalent → grade 3.
  2. LOW-OVERLAP NEGATIVES: For each query, add 2 episodes from unrelated queries
     that share almost no words → grade 0.
  3. Keep all existing adversarial episodes (entity swap, temporal shift, etc.)
     so the model still learns fine-grained discrimination.

This decouples lexical overlap from relevance grade, forcing the model to rely on
semantic understanding rather than overlap heuristics.

Usage:
    python scripts/augment_relevance_data.py

Output:
    data/familyos/nli/relevance/human_benchmark_listwise_v2.jsonl
    data/familyos/nli/splits/stage_c/train.jsonl  (overwritten with augmented)
    data/familyos/nli/splits/stage_c/dev.jsonl    (overwritten with augmented)
    data/familyos/nli/splits/stage_c/holdout.jsonl (overwritten with augmented)
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 42
random.seed(SEED)

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "data" / "familyos" / "nli"
INPUT_FILE = DATA_ROOT / "relevance" / "human_benchmark_listwise.jsonl"
OUTPUT_FILE = DATA_ROOT / "relevance" / "human_benchmark_listwise_v2.jsonl"
SPLITS_DIR = DATA_ROOT / "splits" / "stage_c"

# How many cross-query negatives to add per query
CROSS_NEGATIVES_PER_QUERY = 2


def jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def make_high_overlap_positive(query: str) -> str:
    """Create a grade-3 episode that keeps most query words.

    Simple transformations that preserve semantics and word overlap:
    - Reorder clauses if comma/conjunction present
    - Prefix with a filler phrase
    - Minor synonym swap on one word
    """
    words = query.split()

    # Strategy 1: Prefix + minor rewording
    prefixes = [
        "Just wanted to note that",
        "For the record,",
        "Quick update:",
        "Heads up,",
        "FYI,",
        "So basically,",
        "Yeah so",
        "Reminder:",
        "Update:",
        "Note to self:",
    ]

    # Strategy 2: Clause reorder if there's a comma or conjunction
    if ", " in query:
        parts = query.split(", ", 1)
        if len(parts) == 2 and len(parts[1]) > 10:
            # Swap clause order ~50% of the time
            if random.random() < 0.5:
                return f"{parts[1].rstrip('.')}, {parts[0].lower().rstrip('.')}."

    # Strategy 3: Simple prefix addition (always high overlap)
    prefix = random.choice(prefixes)
    # Lowercase first word of query if not a proper noun
    first_word = words[0]
    if first_word[0].isupper() and first_word not in (
        "I", "I'm", "I've", "I'll", "I'd",
    ):
        adjusted = first_word[0].lower() + first_word[1:]
        return f"{prefix} {adjusted} {' '.join(words[1:])}"
    return f"{prefix} {query}"


def make_verbatim_echo(query: str) -> str:
    """Create a near-verbatim repeat that a retrieval system would see.

    Simulates real retrieval scenarios: the indexed document contains
    almost exactly the same text as the query (e.g. a note, a message,
    a calendar entry).
    """
    # Minor perturbations that keep >90% word overlap
    strategies = [
        # Echo with trailing context
        lambda q: f"{q} That's the plan.",
        lambda q: f"{q} Just confirming.",
        lambda q: f"Re: {q}",
        lambda q: f"Noted. {q}",
        # Drop last word
        lambda q: " ".join(q.split()[:-1]) + "." if len(q.split()) > 5 else q,
        # Add timestamp-like prefix
        lambda q: f"[Today] {q}",
    ]
    return random.choice(strategies)(query)


def load_data(path: Path) -> list[dict]:
    """Load JSONL data."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def augment_dataset(records: list[dict]) -> list[dict]:
    """Augment records with overlap-balanced episodes."""
    # Collect all episode texts for cross-query negative sampling
    all_episodes_by_query: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        all_episodes_by_query[i] = [ep["text"] for ep in rec["episodes"]]

    query_indices = list(range(len(records)))
    augmented = []

    for i, rec in enumerate(records):
        query = rec["query"]
        episodes = list(rec["episodes"])  # copy

        # 1. Add HIGH-OVERLAP POSITIVE (grade 3, high Jaccard)
        high_overlap_pos = make_high_overlap_positive(query)
        episodes.append({
            "text": high_overlap_pos,
            "grade": 3,
            "source_type": "high_overlap_positive",
        })

        # 2. Add VERBATIM ECHO (grade 3, very high Jaccard)
        echo = make_verbatim_echo(query)
        episodes.append({
            "text": echo,
            "grade": 3,
            "source_type": "verbatim_echo",
        })

        # 3. Add LOW-OVERLAP NEGATIVES (grade 0, near-zero Jaccard)
        # Sample episodes from distant queries
        candidate_indices = [j for j in query_indices if j != i]
        random.shuffle(candidate_indices)
        added_negs = 0
        for j in candidate_indices:
            if added_negs >= CROSS_NEGATIVES_PER_QUERY:
                break
            other_query = records[j]["query"]
            # Ensure low overlap between queries themselves
            if jaccard(query, other_query) < 0.15:
                # Pick a random episode from the other query
                other_eps = all_episodes_by_query[j]
                if other_eps:
                    neg_text = random.choice(other_eps)
                    # Verify low overlap with our query
                    if jaccard(query, neg_text) < 0.15:
                        episodes.append({
                            "text": neg_text,
                            "grade": 0,
                            "source_type": "cross_query_negative",
                        })
                        added_negs += 1

        augmented.append({
            "query": query,
            "episodes": episodes,
        })

    return augmented


def print_stats(records: list[dict], label: str) -> None:
    """Print overlap statistics by grade."""
    by_grade: dict[int, list[float]] = defaultdict(list)
    total_eps = 0
    for rec in records:
        q = rec["query"]
        for ep in rec["episodes"]:
            j = jaccard(q, ep["text"])
            by_grade[ep["grade"]].append(j)
            total_eps += 1

    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(records)} queries, {total_eps} episodes")
    print(f"{'=' * 60}")
    print(f"  {'Grade':<8} {'Count':<8} {'Mean Jaccard':<15} {'Std':<10}")
    print(f"  {'-' * 41}")
    for g in sorted(by_grade.keys()):
        vals = by_grade[g]
        import statistics
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {g:<8} {len(vals):<8} {mean:<15.4f} {std:<10.4f}")

    # Source type distribution
    by_source: dict[str, int] = defaultdict(int)
    for rec in records:
        for ep in rec["episodes"]:
            by_source[ep.get("source_type", "unknown")] += 1
    print(f"\n  Source type distribution:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src:<30s}: {count:>5d} ({count / total_eps * 100:.1f}%)")


def split_data(
    records: list[dict],
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into train/dev/holdout."""
    random.shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    return records[:train_end], records[train_end:dev_end], records[dev_end:]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def main() -> None:
    print("Loading original data...")
    original = load_data(INPUT_FILE)
    print_stats(original, "ORIGINAL DATA")

    print("\nAugmenting...")
    augmented = augment_dataset(original)
    print_stats(augmented, "AUGMENTED DATA")

    # Save full augmented file
    save_jsonl(augmented, OUTPUT_FILE)

    # Create train/dev/holdout splits
    train, dev, holdout = split_data(augmented)
    save_jsonl(train, SPLITS_DIR / "train.jsonl")
    save_jsonl(dev, SPLITS_DIR / "dev.jsonl")
    save_jsonl(holdout, SPLITS_DIR / "holdout.jsonl")

    print(f"\n  Train: {len(train)}, Dev: {len(dev)}, Holdout: {len(holdout)}")
    print("\nDone. Use augmented data for MGRH retraining.")


if __name__ == "__main__":
    main()

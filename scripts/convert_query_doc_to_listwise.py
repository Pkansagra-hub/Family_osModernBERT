"""Convert query_doc positive pairs into listwise episodes for Stage C training.

Problem: raw query_doc data is binary positive-only (grade=3, no negatives).
Feeding 10,019 positive-only pairs into LambdaRank produces zero gradient
because there are no lower-ranked candidates to contrast against. Stage C
was overwhelmed by these 94%-of-training samples and degraded from Bridge's
spearman=0.8093 to 0.7948.

Fix:
    Each positive pair (query, doc, pair_type) becomes a 4-candidate episode:
      - grade 3: original positive document (exact match for query)
      - grade 1: document from SAME pair_type but DIFFERENT query
                 (structurally similar answer, wrong entity/content -- near miss)
      - grade 0: two documents from DIFFERENT pair_types
                 (off-topic, wrong kind of memory/reminder/event entirely)

    This gives LambdaRank a clear 3-level ranking signal on in-domain FamilyOS
    data, properly complementing the human benchmark listwise data.

Usage:
    python scripts/convert_query_doc_to_listwise.py

Output:
    data/familyos/nli/relevance/query_doc_listwise.jsonl
    (~10,019 episodes)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 42
random.seed(SEED)

WORKSPACE = Path(__file__).resolve().parent.parent
QUERY_DOC_DIR = WORKSPACE / "data" / "familyos" / "embeddings" / "mined_v2" / "query_doc"
OUTPUT_FILE = WORKSPACE / "data" / "familyos" / "nli" / "relevance" / "query_doc_listwise.jsonl"

# Number of grade-0 negatives per episode (from different pair_types)
NUM_OFFTYPE_NEGATIVES = 2


def load_all_pairs() -> list[dict[str, Any]]:
    """Load all query_doc shards."""
    pairs: list[dict[str, Any]] = []
    for shard in sorted(QUERY_DOC_DIR.glob("query_doc_pairs_*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} query_doc pairs from {QUERY_DOC_DIR}")
    return pairs


def build_type_index(pairs: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Index pairs by pair_type for sampling negatives."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        idx[p["pair_type"]].append(i)
    return idx


def sample_near_miss(
    pairs: list[dict[str, Any]],
    type_idx: dict[str, list[int]],
    current_idx: int,
    pair_type: str,
    rng: random.Random,
) -> str:
    """Return a grade-1 document: same pair_type, different index."""
    candidates = [i for i in type_idx[pair_type] if i != current_idx]
    if not candidates:
        # Fallback: any other pair
        candidates = [i for i in range(len(pairs)) if i != current_idx]
    chosen = rng.choice(candidates)
    return pairs[chosen]["document"]


def sample_off_topic(
    pairs: list[dict[str, Any]],
    type_idx: dict[str, list[int]],
    current_idx: int,
    pair_type: str,
    n: int,
    rng: random.Random,
) -> list[str]:
    """Return n grade-0 documents: different pair_type from current."""
    other_types = [t for t in type_idx if t != pair_type]
    if not other_types:
        other_types = list(type_idx.keys())

    docs: list[str] = []
    used: set[int] = {current_idx}

    # Round-robin across other types for diversity
    type_pool = other_types * ((n // len(other_types)) + 2)
    rng.shuffle(type_pool)

    for t in type_pool:
        if len(docs) >= n:
            break
        candidates = [i for i in type_idx[t] if i not in used]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        used.add(chosen)
        docs.append(pairs[chosen]["document"])

    # Fallback if still short
    while len(docs) < n:
        i = rng.randint(0, len(pairs) - 1)
        if i not in used:
            used.add(i)
            docs.append(pairs[i]["document"])

    return docs[:n]


def convert(rng: random.Random) -> list[dict[str, Any]]:
    """Build listwise episodes from query_doc pairs."""
    pairs = load_all_pairs()
    type_idx = build_type_index(pairs)

    print("pair_type distribution:")
    for t, indices in sorted(type_idx.items(), key=lambda x: -len(x[1])):
        print(f"  {t}: {len(indices)}")

    episodes: list[dict[str, Any]] = []

    for i, pair in enumerate(pairs):
        query = pair["query"]
        pos_doc = pair["document"]
        pair_type = pair["pair_type"]

        near_miss_doc = sample_near_miss(pairs, type_idx, i, pair_type, rng)
        off_topic_docs = sample_off_topic(pairs, type_idx, i, pair_type, NUM_OFFTYPE_NEGATIVES, rng)

        episode_list = [
            {"text": pos_doc, "grade": 3, "source_type": "query_doc_positive"},
            {"text": near_miss_doc, "grade": 1, "source_type": "query_doc_near_miss"},
        ]
        for doc in off_topic_docs:
            episode_list.append({"text": doc, "grade": 0, "source_type": "query_doc_off_topic"})

        # Shuffle so model doesn't learn position bias
        rng.shuffle(episode_list)

        episodes.append({
            "query": query,
            "episodes": episode_list,
            "pair_type": pair_type,
            "query_id": pair.get("query_id", f"qd_{i:06d}"),
        })

    return episodes


def verify(episodes: list[dict[str, Any]]) -> None:
    """Sanity-check the output."""
    grade_counts: dict[int, int] = defaultdict(int)
    ep_lengths: list[int] = []
    for ep in episodes:
        ep_lengths.append(len(ep["episodes"]))
        for e in ep["episodes"]:
            grade_counts[e["grade"]] += 1

    print(f"\nOutput stats:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Episode length: min={min(ep_lengths)} max={max(ep_lengths)} avg={sum(ep_lengths)/len(ep_lengths):.1f}")
    print(f"  Grade distribution:")
    total = sum(grade_counts.values())
    for g in sorted(grade_counts):
        print(f"    grade {g}: {grade_counts[g]} ({100*grade_counts[g]/total:.1f}%)")


def main() -> None:
    rng = random.Random(SEED)
    episodes = convert(rng)
    verify(episodes)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(episodes)} episodes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

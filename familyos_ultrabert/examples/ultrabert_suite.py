"""FamilyOS UltraBERT - Unified Example + Benchmark Suite.

This consolidates the most-used functionality from the various scripts under
`familyos_ultrabert/examples/` into one CLI.

Goals:
- Single entrypoint for smoke tests, latency checks, embedding evaluation, and stress tests.
- No hard-coded absolute paths.
- Uses only the public package API (`familyos_ultrabert.Client` / `UltraBERT`).

Usage examples:
  python familyos_ultrabert/examples/ultrabert_suite.py smoke
  python familyos_ultrabert/examples/ultrabert_suite.py latency --runs 50 --capabilities sentiment,emotions
  python familyos_ultrabert/examples/ultrabert_suite.py embeddings --triplet-dir D:\\Modeling_studio\\data\\familyos\\embeddings\\silver_synthetic
  python familyos_ultrabert/examples/ultrabert_suite.py stress --runs 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from familyos_ultrabert import Client


DEFAULT_SAMPLE_TEXT = "Mom picked up Panda from school today and we had dinner together."


@dataclass(frozen=True)
class LatencySummary:
    """Latency stats (milliseconds)."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": float(self.mean_ms),
            "p50_ms": float(self.p50_ms),
            "p95_ms": float(self.p95_ms),
            "min_ms": float(self.min_ms),
            "max_ms": float(self.max_ms),
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("No values provided")
    if pct < 0 or pct > 100:
        raise ValueError("pct must be between 0 and 100")

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _summarize_latencies_ms(latencies_ms: List[float]) -> LatencySummary:
    if not latencies_ms:
        raise ValueError("No latencies recorded")

    return LatencySummary(
        mean_ms=float(statistics.mean(latencies_ms)),
        p50_ms=float(_percentile(latencies_ms, 50)),
        p95_ms=float(_percentile(latencies_ms, 95)),
        min_ms=float(min(latencies_ms)),
        max_ms=float(max(latencies_ms)),
    )


def _parse_capabilities(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run a minimal smoke test."""

    client = Client(
        warmup=not args.no_warmup,
        warmup_rounds=args.warmup_rounds,
        backend=args.backend,
        device=args.device,
        model_path=args.model_path,
    )

    text = args.text or DEFAULT_SAMPLE_TEXT
    capabilities = _parse_capabilities(args.capabilities) if args.capabilities else []

    print("=" * 80)
    print("ULTRABERT SMOKE TEST")
    print("=" * 80)
    print(f"Backend: {client.backend}")

    result = client.analyze(text, capabilities=capabilities or None)

    print(f"Text: {text}")
    print(f"Latency: {result.latency_ms:.2f}ms")

    # Print a small, consistent subset
    if result.capabilities.get("safety_familyos"):
        s = result.capabilities["safety_familyos"]
        print(f"Safety: {s.get('band')} ({s.get('confidence')})")
    if result.capabilities.get("sentiment"):
        s = result.capabilities["sentiment"]
        print(f"Sentiment: {s.get('prediction')} ({s.get('confidence')})")
    if result.capabilities.get("intent"):
        s = result.capabilities["intent"]
        print(f"Intent: {s.get('prediction')} ({s.get('confidence')})")
    if result.capabilities.get("emotions"):
        s = result.capabilities["emotions"]
        preds = s.get("predictions") or []
        print(f"Emotions: {preds[:10]}")

    print("OK")
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    """Measure latency for one text across capabilities."""

    client = Client(
        warmup=not args.no_warmup,
        warmup_rounds=args.warmup_rounds,
        backend=args.backend,
        device=args.device,
        model_path=args.model_path,
    )

    text = args.text or DEFAULT_SAMPLE_TEXT
    caps = _parse_capabilities(args.capabilities) if args.capabilities else []
    if not caps:
        caps = ["sentiment", "emotions", "safety_familyos", "intent", "embedding"]

    # Warmup
    for _ in range(args.warmup_runs):
        client.analyze(text, capabilities=caps)

    latencies_ms: List[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        client.analyze(text, capabilities=caps)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    summary = _summarize_latencies_ms(latencies_ms)

    payload: Dict[str, Any] = {
        "backend": client.backend,
        "device": client.device,
        "capabilities": caps,
        "runs": args.runs,
        **summary.to_dict(),
    }

    print(json.dumps(payload, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


def _load_triplets(triplet_dir: Path, limit: int) -> List[Dict[str, str]]:
    """Load triplets_*.jsonl shards with keys: anchor, positive, negative."""

    if not triplet_dir.exists():
        raise FileNotFoundError(f"Triplet directory not found: {triplet_dir}")

    triplets: List[Dict[str, str]] = []
    for shard in sorted(triplet_dir.glob("triplets_*.jsonl")):
        with shard.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Minimal schema enforcement
                if not all(k in item for k in ("anchor", "positive", "negative")):
                    continue
                triplets.append({
                    "anchor": str(item["anchor"]),
                    "positive": str(item["positive"]),
                    "negative": str(item["negative"]),
                })
                if len(triplets) >= limit:
                    return triplets

    return triplets


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Run a focused embedding benchmark (triplet accuracy + recall@k)."""

    triplet_dir = Path(args.triplet_dir)
    triplets = _load_triplets(triplet_dir, limit=args.num_triplets)
    if not triplets:
        raise RuntimeError(f"No triplets loaded from {triplet_dir}")

    client = Client(
        warmup=not args.no_warmup,
        warmup_rounds=args.warmup_rounds,
        backend=args.backend,
        device=args.device,
        model_path=args.model_path,
    )

    rng = random.Random(args.seed)

    # Cache embeddings for unique texts to reduce runtime
    unique_texts = sorted({t["anchor"] for t in triplets} | {t["positive"] for t in triplets} | {t["negative"] for t in triplets})
    text_to_emb: Dict[str, np.ndarray] = {}

    start = time.perf_counter()
    for i, text in enumerate(unique_texts):
        emb = np.array(client.get_embedding(text), dtype=np.float32)
        text_to_emb[text] = emb
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"Embedded {i + 1}/{len(unique_texts)} texts ({rate:.1f} texts/sec)")

    embed_elapsed = time.perf_counter() - start

    # Triplet accuracy
    correct = 0
    for t in triplets:
        a = text_to_emb[t["anchor"]]
        p = text_to_emb[t["positive"]]
        n = text_to_emb[t["negative"]]
        if _cosine_similarity(a, p) > _cosine_similarity(a, n):
            correct += 1

    triplet_accuracy = correct / len(triplets)

    # Recall@k with distractors
    ks = [1, 5, 10]
    recall_hits = {k: 0 for k in ks}

    negatives = [t["negative"] for t in triplets]
    for i, t in enumerate(triplets):
        anchor = text_to_emb[t["anchor"]]
        positive = text_to_emb[t["positive"]]

        # Sample distractors from other negatives
        candidate_texts = [t["positive"]]
        other_indices = [j for j in range(len(triplets)) if j != i]
        distractor_indices = rng.sample(other_indices, k=min(args.num_distractors, len(other_indices)))
        for j in distractor_indices:
            candidate_texts.append(negatives[j])

        candidates = np.stack([text_to_emb[x] for x in candidate_texts], axis=0)

        # Vectorized cosine similarity
        anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-8)
        cand_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
        sims = cand_norm @ anchor_norm

        ranking = np.argsort(-sims)
        positive_rank = int(np.where(ranking == 0)[0][0]) + 1  # 1-indexed

        for k in ks:
            if positive_rank <= k:
                recall_hits[k] += 1

    recall = {f"recall@{k}": recall_hits[k] / len(triplets) for k in ks}

    payload: Dict[str, Any] = {
        "backend": client.backend,
        "device": client.device,
        "triplet_dir": str(triplet_dir),
        "num_triplets": len(triplets),
        "num_unique_texts": len(unique_texts),
        "embedding_compute_seconds": float(embed_elapsed),
        "unique_texts_per_second": float(len(unique_texts) / embed_elapsed) if embed_elapsed > 0 else 0.0,
        "triplet_accuracy": float(triplet_accuracy),
        **{k: float(v) for k, v in recall.items()},
        "num_distractors": int(args.num_distractors),
        "seed": int(args.seed),
    }

    print(json.dumps(payload, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    """Run a lightweight stress test (length scaling + throughput)."""

    client = Client(
        warmup=not args.no_warmup,
        warmup_rounds=args.warmup_rounds,
        backend=args.backend,
        device=args.device,
        model_path=args.model_path,
    )

    short = "Hi mom!"
    medium = "Mom picked up the kids from school today and they had a great time."
    long = "Yesterday was such a wonderful day for our family. " * 40

    tests = [
        ("short", short),
        ("medium", medium),
        ("long", long),
    ]

    caps = _parse_capabilities(args.capabilities) if args.capabilities else []
    if not caps:
        caps = ["sentiment", "safety_familyos", "emotions"]

    print("=" * 80)
    print("ULTRABERT STRESS")
    print("=" * 80)
    print(f"Backend: {client.backend}")
    print(f"Capabilities: {caps}")

    # Length scaling
    scaling: Dict[str, LatencySummary] = {}
    for name, text in tests:
        # Warmup
        client.analyze(text, capabilities=caps)

        latencies_ms: List[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            client.analyze(text, capabilities=caps)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        scaling[name] = _summarize_latencies_ms(latencies_ms)

    # Throughput
    texts = [short, medium, long]
    start = time.perf_counter()
    for i in range(args.throughput_iters):
        client.analyze(texts[i % len(texts)], capabilities=caps)
    elapsed = time.perf_counter() - start
    throughput = args.throughput_iters / elapsed if elapsed > 0 else 0.0

    payload = {
        "backend": client.backend,
        "device": client.device,
        "capabilities": caps,
        "length_scaling": {k: v.to_dict() for k, v in scaling.items()},
        "throughput_iters": int(args.throughput_iters),
        "throughput_per_sec": float(throughput),
    }

    print(json.dumps(payload, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultrabert_suite",
        description="Unified example + benchmark CLI for FamilyOS UltraBERT.",
    )

    parser.add_argument(
        "--backend",
        default=None,
        choices=[None, "auto", "pytorch", "onnx"],
        help="Backend to use (default: auto)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string (examples: cpu, cuda). Default: backend-dependent.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to weights directory.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup on client init.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Warmup rounds during client init.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Quick smoke test.")
    smoke.add_argument("--text", default=None)
    smoke.add_argument("--capabilities", default=None, help="Comma-separated.")
    smoke.set_defaults(func=cmd_smoke)

    latency = sub.add_parser("latency", help="Latency benchmark (single text).")
    latency.add_argument("--text", default=None)
    latency.add_argument("--capabilities", default=None, help="Comma-separated.")
    latency.add_argument("--runs", type=int, default=30)
    latency.add_argument("--warmup-runs", type=int, default=5)
    latency.add_argument("--output-json", default=None)
    latency.set_defaults(func=cmd_latency)

    embeddings = sub.add_parser("embeddings", help="Embedding benchmark (triplets).")
    embeddings.add_argument("--triplet-dir", required=True)
    embeddings.add_argument("--num-triplets", type=int, default=3000)
    embeddings.add_argument("--num-distractors", type=int, default=99)
    embeddings.add_argument("--seed", type=int, default=42)
    embeddings.add_argument("--progress-every", type=int, default=200)
    embeddings.add_argument("--output-json", default=None)
    embeddings.set_defaults(func=cmd_embeddings)

    stress = sub.add_parser("stress", help="Lightweight stress test.")
    stress.add_argument("--capabilities", default=None, help="Comma-separated.")
    stress.add_argument("--runs", type=int, default=10)
    stress.add_argument("--throughput-iters", type=int, default=200)
    stress.add_argument("--output-json", default=None)
    stress.set_defaults(func=cmd_stress)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

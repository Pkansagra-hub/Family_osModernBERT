#!/usr/bin/env python
"""Probe encoder-only discriminative quality for FamilyOS UltraBERT.

This script isolates the raw encoder from the trained embedding head and compares:
- encoder mean pooling
- encoder CLS pooling
- current embedding head output

Metrics focus on whether representations are discriminative rather than merely
well-formed:
- similarity separation on curated positive pairs
- triplet ranking accuracy and margin
- retrieval recall with distractors
- optional STS-B Spearman correlation on a sample

Usage:
    python scripts/check_encoder_discriminativeness.py
    python scripts/check_encoder_discriminativeness.py --sts-samples 1000
    python scripts/check_encoder_discriminativeness.py --retrieval-limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from familyos_ultrabert import Client, __version__
from familyos_ultrabert.benchmarks.data.test_cases import (
    RETRIEVAL_CASES_10,
    RETRIEVAL_CASES_100,
    SIMILARITY_CASES,
    TRIPLET_CASES,
)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return matrix / norms


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity for two matrices."""
    a_norm = _l2_normalize(a)
    b_norm = _l2_normalize(b)
    return np.sum(a_norm * b_norm, axis=1)


class EncoderProbe:
    """Provides encoder-only and head-based embedding extraction."""

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.client = Client(backend="pytorch", device="auto", warmup=True, warmup_rounds=3, verbose=True)
        ultrabert = self.client._model
        if ultrabert is None:
            raise RuntimeError("Client failed to load model")
        self.engine = ultrabert._engine
        self.model = self.engine.model
        self.encoder = self.engine.encoder
        self.tokenizer = self.engine.tokenizer
        self.device = self.engine.device

    def encode_head(self, sentences: Sequence[str]) -> np.ndarray:
        """Encode with the current trained embedding head."""
        outputs: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = list(sentences[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_out = self.model(
                    capability="embedding",
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                logits = model_out.logits.detach().cpu().numpy()
                outputs.append(logits)
        return np.concatenate(outputs, axis=0)

    def encode_encoder(self, sentences: Sequence[str], pooling: str) -> np.ndarray:
        """Encode with raw encoder outputs and a simple pooling rule.

        Args:
            sentences: Input texts.
            pooling: Pooling strategy, one of "mean" or "cls".
        """
        outputs: List[np.ndarray] = []
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = list(sentences[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                encoder_out = self.encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                hidden = encoder_out.last_hidden_state
                attention_mask = encoded["attention_mask"]
                if pooling == "cls":
                    pooled = hidden[:, 0, :]
                elif pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def _prepare_negative_pairs() -> List[tuple[str, str]]:
    """Build mismatched sentence pairs from the positive similarity cases."""
    left_texts = [left for left, _, _ in SIMILARITY_CASES]
    right_texts = [right for _, right, _ in SIMILARITY_CASES]
    negatives: List[tuple[str, str]] = []
    for index, left in enumerate(left_texts):
        right = right_texts[(index + 1) % len(right_texts)]
        negatives.append((left, right))
    return negatives


def evaluate_similarity_separation(encode_fn: Callable[[Sequence[str]], np.ndarray]) -> Dict[str, float]:
    """Measure positive-vs-negative similarity separation."""
    positives_left = [left for left, _, _ in SIMILARITY_CASES]
    positives_right = [right for _, right, _ in SIMILARITY_CASES]
    negatives = _prepare_negative_pairs()
    negatives_left = [left for left, _ in negatives]
    negatives_right = [right for _, right in negatives]

    pos_a = encode_fn(positives_left)
    pos_b = encode_fn(positives_right)
    neg_a = encode_fn(negatives_left)
    neg_b = encode_fn(negatives_right)

    pos_scores = _cosine_similarity(pos_a, pos_b)
    neg_scores = _cosine_similarity(neg_a, neg_b)
    pairwise_win_rate = float(np.mean((pos_scores[:, None] > neg_scores[None, :]).astype(np.float32)))

    return {
        "positive_mean": float(np.mean(pos_scores)),
        "negative_mean": float(np.mean(neg_scores)),
        "separation_gap": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "pairwise_win_rate": pairwise_win_rate,
    }


def evaluate_triplets(encode_fn: Callable[[Sequence[str]], np.ndarray]) -> Dict[str, float]:
    """Measure triplet ranking accuracy and margins."""
    total = 0
    passed = 0
    margins: List[float] = []

    for case in TRIPLET_CASES:
        texts = [str(case["anchor"]), str(case["positive"])] + [str(item) for item in case["negatives"]]
        embeddings = encode_fn(texts)
        anchor = embeddings[0:1]
        positive = embeddings[1:2]
        negatives = embeddings[2:]
        pos_sim = float(_cosine_similarity(anchor, positive)[0])
        neg_sims = _cosine_similarity(np.repeat(anchor, negatives.shape[0], axis=0), negatives)
        margin = float(pos_sim - float(np.max(neg_sims)))
        margins.append(margin)
        total += 1
        if margin > 0.0:
            passed += 1

    return {
        "triplet_accuracy": float(passed / total) if total else 0.0,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "min_margin": float(np.min(margins)) if margins else 0.0,
    }


def _retrieval_score(
    encode_fn: Callable[[Sequence[str]], np.ndarray],
    cases: Sequence[Dict[str, Any]],
    top_k: int,
) -> float:
    """Compute retrieval recall@k."""
    hits = 0
    total = 0
    for case in cases:
        query = str(case["query"])
        relevant = str(case["relevant"])
        distractors = [str(item) for item in case["distractors"]]
        texts = [query, relevant] + distractors
        embeddings = encode_fn(texts)
        query_emb = embeddings[0:1]
        candidates = embeddings[1:]
        scores = _cosine_similarity(np.repeat(query_emb, candidates.shape[0], axis=0), candidates)
        ranked_indices = np.argsort(scores)[::-1]
        top_indices = ranked_indices[: max(1, top_k)]
        total += 1
        if 0 in top_indices:
            hits += 1
    return float(hits / total) if total else 0.0


def evaluate_retrieval(
    encode_fn: Callable[[Sequence[str]], np.ndarray],
    retrieval_limit: int,
) -> Dict[str, float]:
    """Measure retrieval recall with light and heavy distractor sets."""
    cases_10 = RETRIEVAL_CASES_10[:retrieval_limit]
    cases_100 = RETRIEVAL_CASES_100[: min(retrieval_limit, len(RETRIEVAL_CASES_100))]
    return {
        "recall_at1_10d": _retrieval_score(encode_fn, cases_10, top_k=1),
        "recall_at1_100d": _retrieval_score(encode_fn, cases_100, top_k=1),
        "recall_at5_100d": _retrieval_score(encode_fn, cases_100, top_k=5),
    }


def evaluate_stsb(
    encode_fn: Callable[[Sequence[str]], np.ndarray],
    sample_count: int,
) -> Dict[str, float]:
    """Measure raw correlation on an STS-B sample."""
    if sample_count <= 0:
        return {}

    from datasets import load_dataset

    dataset = load_dataset("glue", "stsb", split="validation")
    sample_count = min(sample_count, len(dataset))
    dataset = dataset.select(range(sample_count))
    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    gold_scores = np.asarray(dataset["label"], dtype=np.float32)
    emb1 = encode_fn(sentences1)
    emb2 = encode_fn(sentences2)
    pred_scores = _cosine_similarity(emb1, emb2)
    corr = spearmanr(gold_scores, pred_scores)
    statistic = float(getattr(corr, "statistic", corr[0]))
    return {"stsb_spearman": statistic, "stsb_samples": float(sample_count)}


def run_probe(sts_samples: int, retrieval_limit: int, output_path: str | None) -> Dict[str, Any]:
    """Run all encoder discriminativeness probes."""
    probe = EncoderProbe(batch_size=64)
    methods: Dict[str, Callable[[Sequence[str]], np.ndarray]] = {
        "encoder_mean": lambda texts: probe.encode_encoder(texts, pooling="mean"),
        "encoder_cls": lambda texts: probe.encode_encoder(texts, pooling="cls"),
        "embedding_head": probe.encode_head,
    }

    head = probe.model.heads["embedding"] if "embedding" in probe.model.heads else None
    head_pooling = getattr(head, "pooling", "unknown") if head is not None else "missing"
    head_params = int(sum(param.numel() for param in head.parameters())) if head is not None else 0

    results: Dict[str, Any] = {
        "package_version": __version__,
        "device": str(probe.device),
        "embedding_head_pooling": head_pooling,
        "embedding_head_params": head_params,
        "methods": {},
    }

    overall_start = time.time()
    for name, encode_fn in methods.items():
        start = time.time()
        similarity = evaluate_similarity_separation(encode_fn)
        triplets = evaluate_triplets(encode_fn)
        retrieval = evaluate_retrieval(encode_fn, retrieval_limit=retrieval_limit)
        stsb = evaluate_stsb(encode_fn, sample_count=sts_samples)
        elapsed = time.time() - start
        results["methods"][name] = {
            "similarity": similarity,
            "triplets": triplets,
            "retrieval": retrieval,
            "stsb": stsb,
            "elapsed_sec": round(elapsed, 2),
        }

    results["total_elapsed_sec"] = round(time.time() - overall_start, 2)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    return results


def _fmt(value: float) -> str:
    """Format float for compact console output."""
    return f"{value:.4f}"


def print_summary(results: Dict[str, Any]) -> None:
    """Print a compact table of probe results."""
    print("=" * 88)
    print("ENCODER DISCRIMINATIVENESS PROBE")
    print("=" * 88)
    print(f"Package version: {results['package_version']}")
    print(f"Device: {results['device']}")
    print(
        f"Embedding head: pooling={results['embedding_head_pooling']}, "
        f"params={results['embedding_head_params']:,}"
    )
    print()
    print(
        f"{'Method':<16} {'Triplet':>8} {'Margin':>8} {'R@1/10':>8} {'R@1/100':>9} "
        f"{'R@5/100':>9} {'Gap':>8} {'WinRate':>9} {'STS-B':>8}"
    )
    print("-" * 88)
    for name, method in results["methods"].items():
        triplets = method["triplets"]
        retrieval = method["retrieval"]
        similarity = method["similarity"]
        stsb = method.get("stsb", {})
        stsb_value = stsb.get("stsb_spearman", float("nan"))
        print(
            f"{name:<16} "
            f"{_fmt(triplets['triplet_accuracy']):>8} "
            f"{_fmt(triplets['avg_margin']):>8} "
            f"{_fmt(retrieval['recall_at1_10d']):>8} "
            f"{_fmt(retrieval['recall_at1_100d']):>9} "
            f"{_fmt(retrieval['recall_at5_100d']):>9} "
            f"{_fmt(similarity['separation_gap']):>8} "
            f"{_fmt(similarity['pairwise_win_rate']):>9} "
            f"{_fmt(stsb_value):>8}"
        )
    print("-" * 88)
    print(f"Total time: {results['total_elapsed_sec']}s")
    print("=" * 88)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check encoder-only discriminative quality.")
    parser.add_argument(
        "--sts-samples",
        type=int,
        default=500,
        help="Number of STS-B validation pairs to score per method (0 to disable).",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=30,
        help="Number of retrieval cases to evaluate for each distractor set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/encoder_probe_v404.json",
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    results = run_probe(
        sts_samples=args.sts_samples,
        retrieval_limit=args.retrieval_limit,
        output_path=args.output,
    )
    print_summary(results)
    if args.output:
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

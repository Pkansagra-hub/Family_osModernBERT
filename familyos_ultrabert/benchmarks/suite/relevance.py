"""MGRH relevance benchmark suite (pairwise accuracy, margin, calibration).

Evaluates the Multi-Granularity Relevance Head via the release Client API.
Loads holdout triplets from the golden benchmark dataset and measures:
- Pairwise accuracy (positive scored > negative)
- Mean margin (positive_score - negative_score)
- Per-slice accuracy breakdown
- Expected Calibration Error (ECE)

Data format: each triplet has query/anchor, positive, negative fields.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from familyos_ultrabert.benchmarks.base import BenchmarkSuite
from familyos_ultrabert.benchmarks.suite import register_suite
from familyos_ultrabert.benchmarks.types import BenchmarkSeverity


# Default path relative to package root
_DEFAULT_HOLDOUT = "data/familyos/benchmarks/retrieval_golden_v1/holdout.jsonl"

# Thresholds
_PAIRWISE_ACCURACY_THRESHOLD = 0.90
_MEAN_MARGIN_THRESHOLD = 0.10
_ECE_THRESHOLD = 0.15

# Maximum events to evaluate (controls runtime)
_MAX_EVENTS = int(os.environ.get("MGRH_BENCH_MAX_EVENTS", "1000"))
_BATCH_SIZE = int(os.environ.get("MGRH_BENCH_BATCH_SIZE", "16"))


def _load_holdout(path: Path, max_events: int) -> List[Dict[str, Any]]:
    """Load holdout triplets, normalising anchor/query field."""
    triplets: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            anchor = entry.get("anchor", entry.get("query"))
            positive = entry.get("positive")
            negative = entry.get("negative")
            if not (isinstance(anchor, str) and isinstance(positive, str)
                    and isinstance(negative, str) and anchor and positive and negative):
                continue
            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "slice": entry.get("slice", "unknown"),
            })
            if len(triplets) >= max_events:
                break
    return triplets


def _compute_ece(
    pos_scores: List[float], neg_scores: List[float], n_bins: int = 10,
) -> float:
    """Expected Calibration Error: positives→1, negatives→0."""
    all_scores = pos_scores + neg_scores
    all_labels = [1.0] * len(pos_scores) + [0.0] * len(neg_scores)
    if not all_scores:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = [(lo <= s < hi) for s in all_scores]
        count = sum(mask)
        if count == 0:
            continue
        avg_conf = sum(s for s, m in zip(all_scores, mask) if m) / count
        avg_acc = sum(l for l, m in zip(all_labels, mask) if m) / count
        ece += count * abs(avg_conf - avg_acc)
    return ece / len(all_scores)


@register_suite
class RelevanceSuite(BenchmarkSuite):
    """MGRH relevance ranking and calibration benchmarks."""

    name: str = "relevance"
    description: str = "MGRH pairwise accuracy, margin, ECE on golden holdout triplets"

    def run(self) -> List[Any]:
        # Locate holdout data
        holdout_env = os.environ.get("MGRH_BENCH_HOLDOUT")
        if holdout_env:
            holdout_path = Path(holdout_env)
        else:
            # Try relative to cwd then package location
            for candidate in [
                Path(_DEFAULT_HOLDOUT),
                Path(__file__).resolve().parents[3] / _DEFAULT_HOLDOUT,
            ]:
                if candidate.exists():
                    holdout_path = candidate
                    break
            else:
                self.add_result(
                    name="relevance/data_available",
                    passed=False,
                    error=f"Holdout file not found: {_DEFAULT_HOLDOUT}",
                    severity=BenchmarkSeverity.FAIL,
                )
                return self.results

        # Check client has relevance capability
        try:
            _ = self._client.score_relevance("test", "test")
        except (ValueError, AttributeError, RuntimeError) as exc:
            self.add_result(
                name="relevance/head_available",
                passed=False,
                error=f"MGRH head not available: {exc}",
                severity=BenchmarkSeverity.FAIL,
            )
            return self.results

        triplets = _load_holdout(holdout_path, _MAX_EVENTS)
        if len(triplets) < 10:
            self.add_result(
                name="relevance/data_sufficient",
                passed=False,
                error=f"Only {len(triplets)} valid triplets found (need >=10)",
                severity=BenchmarkSeverity.FAIL,
            )
            return self.results

        # Score all pairs via rerank (batched, temperature-calibrated)
        pos_scores: List[float] = []
        neg_scores: List[float] = []
        slice_results: Dict[str, List[bool]] = {}

        start = time.perf_counter()
        for triplet in triplets:
            ranked = self._client.rerank(
                triplet["anchor"],
                [triplet["positive"], triplet["negative"]],
                batch_size=_BATCH_SIZE,
            )
            # ranked is sorted descending by score; find scores by original index
            score_map = {r["index"]: r["score"] for r in ranked}
            p_score = score_map[0]  # positive was index 0
            n_score = score_map[1]  # negative was index 1
            pos_scores.append(p_score)
            neg_scores.append(n_score)

            sl = triplet["slice"]
            if sl not in slice_results:
                slice_results[sl] = []
            slice_results[sl].append(p_score > n_score)

        elapsed_s = time.perf_counter() - start

        # Pairwise accuracy
        correct = sum(1 for p, n in zip(pos_scores, neg_scores) if p > n)
        accuracy = correct / len(pos_scores)
        self.add_result(
            name="relevance/pairwise_accuracy",
            passed=accuracy >= _PAIRWISE_ACCURACY_THRESHOLD,
            score=round(accuracy, 4),
            threshold=_PAIRWISE_ACCURACY_THRESHOLD,
            details={
                "correct": correct,
                "total": len(pos_scores),
                "elapsed_s": round(elapsed_s, 1),
            },
            severity=BenchmarkSeverity.FAIL,
        )

        # Mean margin
        margins = [p - n for p, n in zip(pos_scores, neg_scores)]
        mean_margin = sum(margins) / len(margins)
        self.add_result(
            name="relevance/mean_margin",
            passed=mean_margin >= _MEAN_MARGIN_THRESHOLD,
            score=round(mean_margin, 4),
            threshold=_MEAN_MARGIN_THRESHOLD,
            severity=BenchmarkSeverity.WARN,
        )

        # ECE
        ece = _compute_ece(pos_scores, neg_scores)
        self.add_result(
            name="relevance/calibration_ece",
            passed=ece <= _ECE_THRESHOLD,
            score=round(ece, 4),
            threshold=_ECE_THRESHOLD,
            severity=BenchmarkSeverity.WARN,
        )

        # Per-slice breakdown (info only)
        for sl, outcomes in sorted(slice_results.items()):
            sl_acc = sum(outcomes) / len(outcomes)
            self.add_result(
                name=f"relevance/slice/{sl}",
                passed=sl_acc >= 0.75,
                score=round(sl_acc, 4),
                details={"count": len(outcomes)},
                severity=BenchmarkSeverity.INFO,
            )

        # Score distribution stats (info)
        self.add_result(
            name="relevance/score_distribution",
            passed=True,
            details={
                "pos_mean": round(sum(pos_scores) / len(pos_scores), 4),
                "neg_mean": round(sum(neg_scores) / len(neg_scores), 4),
                "pos_min": round(min(pos_scores), 4),
                "pos_max": round(max(pos_scores), 4),
                "neg_min": round(min(neg_scores), 4),
                "neg_max": round(max(neg_scores), 4),
            },
            severity=BenchmarkSeverity.INFO,
        )

        # Throughput
        pairs_per_sec = (len(triplets) * 2) / elapsed_s if elapsed_s > 0 else 0
        self.add_result(
            name="relevance/throughput",
            passed=True,
            score=round(pairs_per_sec, 1),
            details={"pairs_scored": len(triplets) * 2, "elapsed_s": round(elapsed_s, 1)},
            severity=BenchmarkSeverity.INFO,
        )

        return self.results

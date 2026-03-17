"""
Optimal MGRH Calibration Temperature Search via Matrix Computation.

Loads real holdout triplets, scores all (anchor, positive) and (anchor, negative)
pairs using raw MGRH logits, then sweeps temperature T in [0.5, 3.0] to find
the value that maximizes:

  1. Pairwise accuracy  (pos_score > neg_score)
  2. Mean margin        (pos_score - neg_score)
  3. ECE                (expected calibration error)

Temperature is applied to raw logits: score = sigmoid(logit / T).

Usage:
    python scripts/optimize_mgrh_temperature.py [--samples 200] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tokenizer(model_path: Path):
    """Load tokenizer with fallback."""
    try:
        return AutoTokenizer.from_pretrained(str(model_path))
    except Exception:
        tokenizer_file = model_path / "tokenizer.json"
        config_file = model_path / "tokenizer_config.json"
        config = {}
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            model_max_length=config.get("model_max_length", 512),
            pad_token=config.get("pad_token"),
            cls_token=config.get("cls_token"),
            sep_token=config.get("sep_token"),
        )


def load_model(checkpoint_path: Path, device: torch.device):
    """Load encoder + heads from release weights."""
    sys.path.insert(0, str(REPO_ROOT))
    from familyos_ultrabert.models.modernbert_multitask import ModernBertMultiTaskModel

    model = ModernBertMultiTaskModel.load_checkpoint(str(checkpoint_path))
    model.to(device).eval()
    return model


@torch.no_grad()
def collect_raw_logits_batched(
    model: torch.nn.Module,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> np.ndarray:
    """Score all (text_a, text_b) pairs and return RAW logits (pre-sigmoid).

    Uses batched encoding for MaxSim z-norm correctness.
    """
    head = model.heads["relevance"]
    emb_head_key = "embedding"
    has_emb = emb_head_key in model.heads

    all_logits: list[float] = []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        texts_a = [p[0] for p in chunk]
        texts_b = [p[1] for p in chunk]

        joint = tokenizer(
            texts_a, texts_b,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc_a = tokenizer(
            texts_a,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc_b = tokenizer(
            texts_b,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        joint_ids = joint["input_ids"].to(device)
        joint_mask = joint["attention_mask"].to(device)
        a_ids = enc_a["input_ids"].to(device)
        a_mask = enc_a["attention_mask"].to(device)
        b_ids = enc_b["input_ids"].to(device)
        b_mask = enc_b["attention_mask"].to(device)

        joint_hidden = model.encoder(
            input_ids=joint_ids, attention_mask=joint_mask, return_dict=True,
        ).last_hidden_state
        a_hidden = model.encoder(
            input_ids=a_ids, attention_mask=a_mask, return_dict=True,
        ).last_hidden_state
        b_hidden = model.encoder(
            input_ids=b_ids, attention_mask=b_mask, return_dict=True,
        ).last_hidden_state

        if has_emb:
            a_emb_out = model.heads[emb_head_key](a_hidden, attention_mask=a_mask)
            b_emb_out = model.heads[emb_head_key](b_hidden, attention_mask=b_mask)
            a_emb = a_emb_out.get("embedding", a_emb_out.get("logits")) if isinstance(a_emb_out, dict) else a_emb_out
            b_emb = b_emb_out.get("embedding", b_emb_out.get("logits")) if isinstance(b_emb_out, dict) else b_emb_out
        else:
            a_emb = b_emb = None

        output = head(
            hidden_states=joint_hidden,
            attention_mask=joint_mask,
            text_a_hidden=a_hidden,
            text_b_hidden=b_hidden,
            text_a_mask=a_mask,
            text_b_mask=b_mask,
            query_embed=a_emb,
            doc_embed=b_emb,
            stage="c",
        )

        raw = output["relevance_logits_raw"].squeeze(-1).float().cpu()
        if raw.dim() == 0:
            all_logits.append(raw.item())
        else:
            all_logits.extend(raw.tolist())

    return np.array(all_logits, dtype=np.float64)


def compute_metrics_at_temperature(
    pos_logits: np.ndarray,
    neg_logits: np.ndarray,
    temperature: float,
) -> dict:
    """Compute pairwise accuracy, margin, and ECE at a given temperature."""
    pos_scores = 1.0 / (1.0 + np.exp(-pos_logits / temperature))
    neg_scores = 1.0 / (1.0 + np.exp(-neg_logits / temperature))

    correct = (pos_scores > neg_scores).sum()
    total = len(pos_scores)
    accuracy = correct / total

    margins = pos_scores - neg_scores
    mean_margin = margins.mean()
    median_margin = np.median(margins)

    # ECE: positives should be ~1.0, negatives should be ~0.0
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (all_scores >= lo) & (all_scores < hi)
        if mask.sum() == 0:
            continue
        avg_conf = all_scores[mask].mean()
        avg_acc = all_labels[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    ece /= len(all_scores)

    return {
        "accuracy": float(accuracy),
        "mean_margin": float(mean_margin),
        "median_margin": float(median_margin),
        "ece": float(ece),
        "pos_mean": float(pos_scores.mean()),
        "neg_mean": float(neg_scores.mean()),
        "pos_std": float(pos_scores.std()),
        "neg_std": float(neg_scores.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="MGRH temperature optimization")
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "familyos_ultrabert" / "weights" / "pytorch"))
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of triplets to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(checkpoint_path)
    print(f"  Device: {device}")

    # Load holdout triplets
    holdout_path = REPO_ROOT / "data" / "familyos" / "benchmarks" / "retrieval_golden_v1" / "holdout.jsonl"
    if not holdout_path.exists():
        holdout_path = REPO_ROOT / "data" / "familyos" / "benchmarks" / "retrieval_golden_v1" / "dev.jsonl"

    print(f"Loading triplets from {holdout_path}...")
    with open(holdout_path, encoding="utf-8") as f:
        triplets = [json.loads(line.strip()) for line in f if line.strip()]
    triplets = triplets[: args.samples]
    print(f"  Using {len(triplets)} triplets")
    if triplets:
        print(f"  First triplet keys: {list(triplets[0].keys())[:6]}")

    # Build pair lists  (triplet format uses 'anchor', pair format uses 'query')
    def get_anchor(t: dict) -> str:
        return t.get("anchor", t.get("query", ""))

    # Filter to entries where all three text fields are non-empty strings
    valid = []
    for t in triplets:
        a = get_anchor(t)
        p = t.get("positive")
        n = t.get("negative")
        if isinstance(a, str) and isinstance(p, str) and isinstance(n, str) and a and p and n:
            valid.append(t)
    print(f"  Valid triplets (all text fields present): {len(valid)} / {len(triplets)}")
    triplets = valid

    pos_pairs = [(get_anchor(t), t["positive"]) for t in triplets]
    neg_pairs = [(get_anchor(t), t["negative"]) for t in triplets]

    # Collect raw logits in batched mode
    print(f"\nScoring {len(pos_pairs)} positive pairs...")
    t0 = time.perf_counter()
    pos_logits = collect_raw_logits_batched(
        model, tokenizer, pos_pairs, device, args.max_length, args.batch_size,
    )
    print(f"Scoring {len(neg_pairs)} negative pairs...")
    neg_logits = collect_raw_logits_batched(
        model, tokenizer, neg_pairs, device, args.max_length, args.batch_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Scored {len(pos_logits) + len(neg_logits)} pairs in {elapsed:.1f}s")

    # Raw logit statistics
    print(f"\n{'='*70}")
    print("RAW LOGIT STATISTICS (pre-sigmoid)")
    print(f"  Positive logits: mean={pos_logits.mean():.4f} std={pos_logits.std():.4f} "
          f"min={pos_logits.min():.4f} max={pos_logits.max():.4f}")
    print(f"  Negative logits: mean={neg_logits.mean():.4f} std={neg_logits.std():.4f} "
          f"min={neg_logits.min():.4f} max={neg_logits.max():.4f}")

    # Temperature sweep
    temperatures = np.concatenate([
        np.arange(0.3, 1.0, 0.05),
        np.arange(1.0, 2.0, 0.1),
        np.arange(2.0, 4.01, 0.25),
    ])

    print(f"\n{'='*70}")
    print(f"TEMPERATURE SWEEP ({len(temperatures)} values)")
    print(f"{'T':>6s}  {'Accuracy':>8s}  {'Margin':>8s}  {'ECE':>8s}  "
          f"{'Pos_mu':>7s}  {'Neg_mu':>7s}  {'Pos_sig':>7s}  {'Neg_sig':>7s}")
    print("-" * 75)

    best_acc = {"t": 1.0, "val": 0.0}
    best_margin = {"t": 1.0, "val": -999}
    best_ece = {"t": 1.0, "val": 999}
    all_results = []

    for t in temperatures:
        m = compute_metrics_at_temperature(pos_logits, neg_logits, t)
        print(f"{t:6.3f}  {m['accuracy']:8.4f}  {m['mean_margin']:8.4f}  {m['ece']:8.4f}  "
              f"{m['pos_mean']:7.4f}  {m['neg_mean']:7.4f}  {m['pos_std']:7.4f}  {m['neg_std']:7.4f}")
        all_results.append({"temperature": float(t), **m})

        if m["accuracy"] > best_acc["val"]:
            best_acc = {"t": float(t), "val": m["accuracy"]}
        if m["mean_margin"] > best_margin["val"]:
            best_margin = {"t": float(t), "val": m["mean_margin"]}
        if m["ece"] < best_ece["val"]:
            best_ece = {"t": float(t), "val": m["ece"]}

    # Compare with stored temperature
    stored_t = 1.0542480945587158
    m_stored = compute_metrics_at_temperature(pos_logits, neg_logits, stored_t)
    m_no_temp = compute_metrics_at_temperature(pos_logits, neg_logits, 1.0)

    print(f"\n{'='*70}")
    print("OPTIMAL TEMPERATURES")
    print(f"  Best accuracy:  T={best_acc['t']:.3f}  -> {best_acc['val']*100:.1f}%")
    print(f"  Best margin:    T={best_margin['t']:.3f}  -> {best_margin['val']:.4f}")
    print(f"  Best ECE:       T={best_ece['t']:.3f}  -> {best_ece['val']:.4f}")
    print(f"\n  Stored T=1.054: accuracy={m_stored['accuracy']*100:.1f}%  "
          f"margin={m_stored['mean_margin']:.4f}  ECE={m_stored['ece']:.4f}")
    print(f"  No temp (T=1):  accuracy={m_no_temp['accuracy']*100:.1f}%  "
          f"margin={m_no_temp['mean_margin']:.4f}  ECE={m_no_temp['ece']:.4f}")

    # Composite score: weighted combination
    # Accuracy is paramount, ECE for calibration quality
    print(f"\n{'='*70}")
    print("COMPOSITE RANKING (0.6*acc + 0.2*(1-ece) + 0.2*margin_norm)")
    max_margin = max(r["mean_margin"] for r in all_results) if all_results else 1.0
    for r in all_results:
        r["composite"] = (
            0.6 * r["accuracy"]
            + 0.2 * (1.0 - r["ece"])
            + 0.2 * (r["mean_margin"] / max_margin if max_margin > 0 else 0)
        )
    all_results.sort(key=lambda x: x["composite"], reverse=True)
    for r in all_results[:5]:
        print(f"  T={r['temperature']:.3f}  composite={r['composite']:.4f}  "
              f"acc={r['accuracy']*100:.1f}%  ECE={r['ece']:.4f}  margin={r['mean_margin']:.4f}")

    optimal_t = all_results[0]["temperature"]
    print(f"\n  >>> RECOMMENDED TEMPERATURE: {optimal_t:.4f}")

    # Save results
    output_path = REPO_ROOT / "outputs" / "mgrh_temperature_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "num_triplets": len(triplets),
            "stored_temperature": stored_t,
            "optimal_temperature": optimal_t,
            "best_accuracy_temperature": best_acc["t"],
            "best_ece_temperature": best_ece["t"],
            "best_margin_temperature": best_margin["t"],
            "no_temp_metrics": m_no_temp,
            "stored_temp_metrics": m_stored,
            "sweep_results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

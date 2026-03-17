"""
Comprehensive MGRH Final Model Benchmark — Torture Test Suite.

Loads the merged final-nli checkpoint and evaluates the Multi-Granularity
Relevance Head across every axis that matters:

  1. LISTWISE RANKING     — Spearman, AUC-ROC, nDCG@10 on human benchmark + holdout
  2. PAIRWISE ACCURACY    — anchor-pos > anchor-neg on hard negatives
  3. HARD NEGATIVE SLICES — entity_swap, temporal_shift, wrong_person, wrong_time
  4. CALIBRATION          — ECE, reliability diagram buckets
  5. SCORE DISTRIBUTION   — per-grade stats, margin analysis
  6. BI-ENCODER BASELINE  — cosine-similarity baseline for direct comparison

All results written to outputs/benchmark_mgrh_final.json and printed to stdout.

Usage:
    python scripts/benchmark_mgrh_final.py \
        --checkpoint outputs/final-nli \
        [--device cuda] [--max-length 512] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Grade semantics
# ---------------------------------------------------------------------------
GRADE_LABELS = {
    0: "irrelevant / harmful",
    1: "entity/temporal mismatch",
    2: "topically related, different event",
    3: "faithful paraphrase / match",
}

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_final_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load the merged final-nli checkpoint with MGRH head.

    Returns:
        (model, mgrh_head, tokenizer)
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, PreTrainedTokenizerFast

    from src.modeling_studio.models.heads import MultiGranularityRelevanceHead
    from src.modeling_studio.models.modernbert_multitask import (
        Capability,
        ModernBertMultiTaskModel,
    )
    from src.modeling_studio.models.pair_encoder import CrossAttentionPairEncoder

    # Helpers from training script
    from scripts.training.train_embedding_heads_bakeoff import (
        create_embedding_head,
        restore_checkpoint_head_architecture,
    )

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Load capabilities, skipping unknown ones like "relevance" that were
    # added to capabilities.json but don't exist in the Capability enum.
    cap_file = checkpoint_path / "capabilities.json"
    with open(cap_file, encoding="utf-8") as f:
        cap_data = json.load(f)
    cap_list = cap_data.get("capabilities", cap_data) if isinstance(cap_data, dict) else cap_data
    # "relevance" and other non-enum caps are handled outside the capability
    # system (MGRH head is added manually), so silently skip them.
    EXPECTED_NON_ENUM = {"relevance"}
    capabilities = []
    for val in cap_list:
        try:
            capabilities.append(Capability(val))
        except ValueError:
            if val not in EXPECTED_NON_ENUM:
                logger.warning(f"  Unknown capability: {val!r}")
    # Exclude decoder
    capabilities = [c for c in capabilities if c != Capability.COUNTERFACTUAL]

    model = ModernBertMultiTaskModel(
        config=model_config, capabilities=capabilities, freeze_encoder=False,
    )
    restore_checkpoint_head_architecture(model, checkpoint_path)
    model._init_encoder()

    # Rebuild embedding head from metadata
    emb_meta_path = checkpoint_path / "embedding_metadata.json"
    if emb_meta_path.exists():
        with open(emb_meta_path, encoding="utf-8") as f:
            emb_meta = json.load(f)
        bakeoff = emb_meta.get("bakeoff", {})
        emb_head = create_embedding_head(
            head_type=bakeoff.get("head_type", "agreement_gated_v2"),
            hidden_size=model.config.hidden_size,
            **bakeoff.get("head_params", {}),
        )
        model.heads["embedding"] = emb_head

    # Load MGRH metadata for architecture params
    mgrh_meta_path = checkpoint_path / "mgrh_metadata.json"
    if mgrh_meta_path.exists():
        with open(mgrh_meta_path, encoding="utf-8") as f:
            mgrh_meta = json.load(f)
    else:
        mgrh_meta = {}

    arch = mgrh_meta.get("architecture", {})
    pe_cfg = arch.get("pair_encoder", {})

    pair_encoder = CrossAttentionPairEncoder(
        hidden_size=model.config.hidden_size,
        num_heads=pe_cfg.get("num_heads", 8),
        num_layers=pe_cfg.get("num_layers", 2),
        dropout=pe_cfg.get("dropout", 0.1),
        use_bidirectional=pe_cfg.get("use_bidirectional", True),
        pooling_strategy=pe_cfg.get("pooling_strategy", "attention"),
    )
    mgrh_head = MultiGranularityRelevanceHead(
        hidden_size=model.config.hidden_size,
        dropout=arch.get("dropout", 0.1),
        pair_encoder=pair_encoder,
    )
    model.heads["mgrh"] = mgrh_head

    # Load weights
    state_dict = load_file(str(checkpoint_path / "model.safetensors"))

    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"  Encoder: {len(encoder_state)} tensors loaded")

    loaded_heads = []
    for head_name in list(model.heads.keys()):
        prefix = f"heads.{head_name}."
        head_state = {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if head_state:
            model.heads[head_name].load_state_dict(head_state, strict=True)
            loaded_heads.append(f"{head_name}({len(head_state)})")
    logger.info(f"  Heads: {', '.join(loaded_heads)}")

    # Calibration temperature
    calibration_T = mgrh_meta.get("calibration", {}).get("temperature", 1.0)
    logger.info(f"  Calibration temperature: {calibration_T}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint_path / "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
    )

    model.to(device).eval()
    mgrh_head.to(device).eval()

    return model, mgrh_head, tokenizer, calibration_T


# ---------------------------------------------------------------------------
# Encoding Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run frozen encoder, return hidden states [B, L, H]."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    enc_out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(enc_out, "last_hidden_state"):
        return enc_out.last_hidden_state
    return enc_out[0] if isinstance(enc_out, tuple) else enc_out


@torch.no_grad()
def get_embedding(model, hidden_states, attention_mask):
    """Get embedding from frozen AgreementGatedHeadV2 for Signal 3."""
    if "embedding" not in model.heads:
        return None
    out = model.heads["embedding"](hidden_states, attention_mask=attention_mask)
    if isinstance(out, dict):
        return out.get("embedding", out.get("logits"))
    return out


def tokenize_pair(tokenizer, text_a: str, text_b: str, max_length: int):
    """Tokenize a joint pair and individual texts."""
    joint = tokenizer(
        text_a, text_b,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc_a = tokenizer(
        text_a,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc_b = tokenizer(
        text_b,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return joint, enc_a, enc_b


# ---------------------------------------------------------------------------
# Score a single (query, doc) pair through MGRH
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_pair(
    model,
    mgrh_head,
    tokenizer,
    query: str,
    doc: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    """Score a (query, doc) pair, returns sigmoid relevance score."""
    joint, enc_q, enc_d = tokenize_pair(tokenizer, query, doc, max_length)

    joint_hidden = encode(model, joint["input_ids"], joint["attention_mask"], device)
    q_hidden = encode(model, enc_q["input_ids"], enc_q["attention_mask"], device)
    d_hidden = encode(model, enc_d["input_ids"], enc_d["attention_mask"], device)

    q_mask = enc_q["attention_mask"].to(device)
    d_mask = enc_d["attention_mask"].to(device)
    q_embed = get_embedding(model, q_hidden, q_mask)
    d_embed = get_embedding(model, d_hidden, d_mask)

    output = mgrh_head(
        hidden_states=joint_hidden,
        attention_mask=joint["attention_mask"].to(device),
        text_a_hidden=q_hidden,
        text_b_hidden=d_hidden,
        text_a_mask=q_mask,
        text_b_mask=d_mask,
        query_embed=q_embed,
        doc_embed=d_embed,
        stage="c",
    )
    return output["logits"].squeeze().item()


# ---------------------------------------------------------------------------
# Bi-encoder baseline
# ---------------------------------------------------------------------------


@torch.no_grad()
def cosine_score(
    model,
    tokenizer,
    query: str,
    doc: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    """Cosine-similarity baseline using the frozen embedding head."""
    enc_q = tokenizer(query, max_length=max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
    enc_d = tokenizer(doc, max_length=max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
    q_hidden = encode(model, enc_q["input_ids"], enc_q["attention_mask"], device)
    d_hidden = encode(model, enc_d["input_ids"], enc_d["attention_mask"], device)
    q_embed = get_embedding(model, q_hidden, enc_q["attention_mask"].to(device))
    d_embed = get_embedding(model, d_hidden, enc_d["attention_mask"].to(device))
    if q_embed is None or d_embed is None:
        return 0.0
    return F.cosine_similarity(q_embed, d_embed, dim=-1).item()


# ---------------------------------------------------------------------------
# Batched Scoring  (fixes MaxSim z-normalization + adds AMP)
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_pairs_batch(
    model,
    mgrh_head,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> list[float]:
    """Score (query, doc) pairs in batches through MGRH.

    Critical: batch_size > 1 ensures MaxSim z-normalization activates
    (the model was trained with batched z-normalization; batch_size=1
    skips it entirely, causing a ~20pp accuracy drop).

    Args:
        pairs: List of (query, doc) tuples.
        batch_size: Pairs per batch. Must be > 1 for correct MaxSim behavior.

    Returns:
        List of float relevance scores (sigmoid-activated).
    """
    all_scores: list[float] = []
    use_amp = device.type == "cuda"

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        queries = [p[0] for p in chunk]
        docs = [p[1] for p in chunk]

        joint = tokenizer(
            queries, docs,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc_q = tokenizer(
            queries,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc_d = tokenizer(
            docs,
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        amp_ctx = (
            autocast("cuda", dtype=torch.bfloat16, enabled=True)
            if use_amp
            else autocast("cpu", enabled=False)
        )
        with amp_ctx:
            joint_hidden = encode(model, joint["input_ids"], joint["attention_mask"], device)
            q_hidden = encode(model, enc_q["input_ids"], enc_q["attention_mask"], device)
            d_hidden = encode(model, enc_d["input_ids"], enc_d["attention_mask"], device)

            q_mask = enc_q["attention_mask"].to(device)
            d_mask = enc_d["attention_mask"].to(device)
            q_embed = get_embedding(model, q_hidden, q_mask)
            d_embed = get_embedding(model, d_hidden, d_mask)

            output = mgrh_head(
                hidden_states=joint_hidden,
                attention_mask=joint["attention_mask"].to(device),
                text_a_hidden=q_hidden,
                text_b_hidden=d_hidden,
                text_a_mask=q_mask,
                text_b_mask=d_mask,
                query_embed=q_embed,
                doc_embed=d_embed,
                stage="c",
            )

        scores = output["logits"].squeeze(-1).float().cpu()
        if scores.dim() == 0:
            all_scores.append(scores.item())
        else:
            all_scores.extend(scores.tolist())

    return all_scores


@torch.no_grad()
def cosine_scores_batch(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> list[float]:
    """Batched bi-encoder cosine similarity baseline."""
    all_scores: list[float] = []
    use_amp = device.type == "cuda"

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        queries = [p[0] for p in chunk]
        docs = [p[1] for p in chunk]

        enc_q = tokenizer(
            queries, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc_d = tokenizer(
            docs, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        amp_ctx = (
            autocast("cuda", dtype=torch.bfloat16, enabled=True)
            if use_amp
            else autocast("cpu", enabled=False)
        )
        with amp_ctx:
            q_hidden = encode(model, enc_q["input_ids"], enc_q["attention_mask"], device)
            d_hidden = encode(model, enc_d["input_ids"], enc_d["attention_mask"], device)
            q_embed = get_embedding(model, q_hidden, enc_q["attention_mask"].to(device))
            d_embed = get_embedding(model, d_hidden, enc_d["attention_mask"].to(device))

        if q_embed is None or d_embed is None:
            all_scores.extend([0.0] * len(chunk))
        else:
            sims = F.cosine_similarity(q_embed.float(), d_embed.float(), dim=-1)
            all_scores.extend(sims.cpu().tolist())

    return all_scores


# ---------------------------------------------------------------------------
# Metrics Helpers
# ---------------------------------------------------------------------------


def compute_spearman(scores: list[float], grades: list[float]) -> float:
    """Spearman rank correlation."""
    n = len(scores)
    if n < 2:
        return 0.0

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * n
        for rank_pos, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = rank_pos + 1.0
        return ranks

    r_scores = _rank(scores)
    r_grades = _rank(grades)
    d_sq = sum((a - b) ** 2 for a, b in zip(r_scores, r_grades))
    return 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1) + 1e-8)


def compute_auc_roc(scores: list[float], binary_labels: list[int]) -> float:
    """Wilcoxon-Mann-Whitney AUC-ROC."""
    pos_scores = [s for s, l in zip(scores, binary_labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, binary_labels) if l == 0]
    if not pos_scores or not neg_scores:
        return 0.5
    concordant = 0
    tied = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1
            elif ps == ns:
                tied += 1
    return (concordant + 0.5 * tied) / (len(pos_scores) * len(neg_scores))


def compute_ndcg_at_k(scores: list[float], grades: list[float], k: int = 10) -> float:
    """nDCG@k for a single query group."""
    paired = sorted(zip(scores, grades), key=lambda x: x[0], reverse=True)
    dcg = sum(
        (2.0 ** g - 1.0) / math.log2(i + 2)
        for i, (_, g) in enumerate(paired[:k])
    )
    ideal_grades = sorted(grades, reverse=True)[:k]
    ideal_dcg = sum(
        (2.0 ** g - 1.0) / math.log2(i + 2)
        for i, g in enumerate(ideal_grades)
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_ece(scores: list[float], binary_labels: list[int], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error with reliability diagram bins."""
    bins = [{"sum_conf": 0.0, "sum_acc": 0.0, "count": 0} for _ in range(n_bins)]
    for score, label in zip(scores, binary_labels):
        bin_idx = min(int(score * n_bins), n_bins - 1)
        bins[bin_idx]["sum_conf"] += score
        bins[bin_idx]["sum_acc"] += label
        bins[bin_idx]["count"] += 1

    ece = 0.0
    total = len(scores)
    bin_info = []
    for i, b in enumerate(bins):
        if b["count"] > 0:
            avg_conf = b["sum_conf"] / b["count"]
            avg_acc = b["sum_acc"] / b["count"]
            ece += abs(avg_acc - avg_conf) * b["count"] / total
            bin_info.append({
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "count": b["count"],
                "gap": round(abs(avg_acc - avg_conf), 4),
            })
    return ece, bin_info


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------


def load_listwise_data(path: str | Path) -> list[dict]:
    """Load listwise JSONL: {query, episodes: [{text, grade, source_type}]}."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_triplet_data(pattern: str, max_samples: int = 5000) -> list[dict]:
    """Load triplet JSONL files matching glob pattern. Capped at max_samples."""
    data = []
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if len(data) >= max_samples:
                        return data
    return data


# ---------------------------------------------------------------------------
# Benchmark Suites
# ---------------------------------------------------------------------------


def benchmark_listwise(
    model, mgrh_head, tokenizer, data: list[dict],
    device: torch.device, max_length: int, dataset_name: str,
    batch_size: int = 16,
) -> dict:
    """Run listwise ranking metrics on {query, episodes} data.

    Uses batched scoring for correct MaxSim z-normalization and speed.
    """
    logger.info(f"  [{dataset_name}] {len(data)} query groups...")

    all_scores = []
    all_grades = []
    all_mgrh_scores_by_grade = defaultdict(list)
    all_bienc_scores_by_grade = defaultdict(list)
    group_spearman = []
    group_ndcg = []
    group_bienc_spearman = []
    group_bienc_ndcg = []

    for qi, item in enumerate(data):
        query = item["query"]
        episodes = item.get("episodes", [])
        if len(episodes) < 2:
            continue

        # Batch all (query, episode) pairs for this group
        pairs = [(query, ep["text"]) for ep in episodes]
        g_scores = score_pairs_batch(
            model, mgrh_head, tokenizer, pairs, device, max_length, batch_size,
        )
        g_bienc = cosine_scores_batch(
            model, tokenizer, pairs, device, max_length, batch_size,
        )
        g_grades = [float(ep["grade"]) for ep in episodes]

        for idx, ep in enumerate(episodes):
            grade = ep["grade"]
            all_scores.append(g_scores[idx])
            all_grades.append(float(grade))
            all_mgrh_scores_by_grade[grade].append(g_scores[idx])
            all_bienc_scores_by_grade[grade].append(g_bienc[idx])

        # Group metrics
        group_spearman.append(compute_spearman(g_scores, g_grades))
        group_ndcg.append(compute_ndcg_at_k(g_scores, g_grades, k=10))
        group_bienc_spearman.append(compute_spearman(g_bienc, g_grades))
        group_bienc_ndcg.append(compute_ndcg_at_k(g_bienc, g_grades, k=10))

        if (qi + 1) % 100 == 0:
            logger.info(f"    Processed {qi + 1}/{len(data)} groups")

    # Global metrics
    binary_labels = [1 if g > 0 else 0 for g in all_grades]
    binary_labels_strict = [1 if g >= 3 else 0 for g in all_grades]

    global_spearman = compute_spearman(all_scores, all_grades)
    global_auc = compute_auc_roc(all_scores, binary_labels)
    global_auc_strict = compute_auc_roc(all_scores, binary_labels_strict)
    ece, reliability_bins = compute_ece(all_scores, binary_labels)

    avg_group_spearman = sum(group_spearman) / len(group_spearman) if group_spearman else 0.0
    avg_group_ndcg = sum(group_ndcg) / len(group_ndcg) if group_ndcg else 0.0
    avg_bienc_spearman = sum(group_bienc_spearman) / len(group_bienc_spearman) if group_bienc_spearman else 0.0
    avg_bienc_ndcg = sum(group_bienc_ndcg) / len(group_bienc_ndcg) if group_bienc_ndcg else 0.0

    # Score distribution per grade
    grade_stats = {}
    for grade in sorted(all_mgrh_scores_by_grade.keys()):
        mgrh_vals = all_mgrh_scores_by_grade[grade]
        bienc_vals = all_bienc_scores_by_grade[grade]
        grade_stats[str(grade)] = {
            "label": GRADE_LABELS.get(grade, "unknown"),
            "count": len(mgrh_vals),
            "mgrh_mean": round(sum(mgrh_vals) / len(mgrh_vals), 4),
            "mgrh_std": round((sum((x - sum(mgrh_vals) / len(mgrh_vals)) ** 2 for x in mgrh_vals) / len(mgrh_vals)) ** 0.5, 4),
            "mgrh_min": round(min(mgrh_vals), 4),
            "mgrh_max": round(max(mgrh_vals), 4),
            "bienc_mean": round(sum(bienc_vals) / len(bienc_vals), 4) if bienc_vals else 0.0,
        }

    # Grade separation (how well does MGRH separate grade 3 from grade 0?)
    if all_mgrh_scores_by_grade.get(3) and all_mgrh_scores_by_grade.get(0):
        grade3_mean = sum(all_mgrh_scores_by_grade[3]) / len(all_mgrh_scores_by_grade[3])
        grade0_mean = sum(all_mgrh_scores_by_grade[0]) / len(all_mgrh_scores_by_grade[0])
        grade_separation = grade3_mean - grade0_mean
    else:
        grade_separation = None

    return {
        "dataset": dataset_name,
        "num_groups": len(group_spearman),
        "num_pairs": len(all_scores),
        "mgrh": {
            "global_spearman": round(global_spearman, 4),
            "avg_group_spearman": round(avg_group_spearman, 4),
            "auc_roc_binary": round(global_auc, 4),
            "auc_roc_strict": round(global_auc_strict, 4),
            "avg_ndcg_at_10": round(avg_group_ndcg, 4),
        },
        "biencoder_baseline": {
            "global_spearman": round(compute_spearman([s for g in all_bienc_scores_by_grade.values() for s in g], all_grades), 4),
            "avg_group_spearman": round(avg_bienc_spearman, 4),
            "avg_ndcg_at_10": round(avg_bienc_ndcg, 4),
        },
        "lift_over_biencoder": {
            "spearman_delta": round(avg_group_spearman - avg_bienc_spearman, 4),
            "ndcg_delta": round(avg_group_ndcg - avg_bienc_ndcg, 4),
        },
        "calibration": {
            "ece": round(ece, 4),
            "reliability_bins": reliability_bins,
        },
        "grade_distribution": grade_stats,
        "grade_3_vs_0_separation": round(grade_separation, 4) if grade_separation is not None else None,
    }


def benchmark_pairwise(
    model, mgrh_head, tokenizer,
    data: list[dict],
    device: torch.device,
    max_length: int,
    dataset_name: str,
    batch_size: int = 16,
) -> dict:
    """Run pairwise accuracy on triplet data (anchor, positive, negative).

    Uses batched scoring for correct MaxSim z-normalization.
    """
    logger.info(f"  [{dataset_name}] {len(data)} triplets...")

    # Collect all pairs: positives first, then negatives
    pos_pairs = [(item["anchor"], item["positive"]) for item in data]
    neg_pairs = [(item["anchor"], item["negative"]) for item in data]
    all_pairs = pos_pairs + neg_pairs

    logger.info(f"    Scoring {len(all_pairs)} pairs in batches of {batch_size}...")
    all_mgrh = score_pairs_batch(
        model, mgrh_head, tokenizer, all_pairs, device, max_length, batch_size,
    )
    pos_scores = all_mgrh[: len(pos_pairs)]
    neg_scores = all_mgrh[len(pos_pairs) :]

    # Compute metrics
    correct = 0
    total = len(data)
    margins = []
    type_results = defaultdict(lambda: {"correct": 0, "total": 0, "margins": []})

    for i, item in enumerate(data):
        margin = pos_scores[i] - neg_scores[i]
        is_correct = pos_scores[i] > neg_scores[i]
        correct += int(is_correct)
        margins.append(margin)

        neg_type = item.get("hard_negative_type", "unknown")
        type_results[neg_type]["correct"] += int(is_correct)
        type_results[neg_type]["total"] += 1
        type_results[neg_type]["margins"].append(margin)

    accuracy = correct / total if total > 0 else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    logger.info(f"    MGRH accuracy: {accuracy:.4f} ({correct}/{total}), avg margin: {avg_margin:.4f}")

    slice_results = {}
    for neg_type, res in sorted(type_results.items()):
        n = res["total"]
        acc = res["correct"] / n if n > 0 else 0.0
        m_avg = sum(res["margins"]) / n if n > 0 else 0.0
        slice_results[neg_type] = {
            "count": n,
            "accuracy": round(acc, 4),
            "avg_margin": round(m_avg, 4),
        }

    # Bi-encoder baseline (capped for speed)
    bienc_cap = min(2000, len(data))
    bienc_pairs_pos = pos_pairs[:bienc_cap]
    bienc_pairs_neg = neg_pairs[:bienc_cap]
    bienc_all = bienc_pairs_pos + bienc_pairs_neg
    bienc_scores = cosine_scores_batch(
        model, tokenizer, bienc_all, device, max_length, batch_size,
    )
    bienc_pos = bienc_scores[:bienc_cap]
    bienc_neg = bienc_scores[bienc_cap:]
    bienc_correct = sum(int(bp > bn) for bp, bn in zip(bienc_pos, bienc_neg))
    bienc_margins = [bp - bn for bp, bn in zip(bienc_pos, bienc_neg)]
    bienc_acc = bienc_correct / bienc_cap if bienc_cap > 0 else 0.0

    return {
        "dataset": dataset_name,
        "num_triplets": total,
        "mgrh": {
            "accuracy": round(accuracy, 4),
            "avg_margin": round(avg_margin, 4),
            "min_margin": round(min(margins), 4) if margins else 0.0,
            "max_margin": round(max(margins), 4) if margins else 0.0,
        },
        "biencoder_baseline": {
            "accuracy": round(bienc_acc, 4),
            "avg_margin": round(sum(bienc_margins) / len(bienc_margins), 4) if bienc_margins else 0.0,
            "num_samples": bienc_cap,
        },
        "hard_negative_slices": slice_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MGRH Final Model Benchmark")
    parser.add_argument("--checkpoint", default="outputs/final-nli", help="Path to merged checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--triplet-cap", type=int, default=3000, help="Max triplets per pairwise dataset")
    parser.add_argument("--output", default="outputs/benchmark_mgrh_final.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    t0 = time.time()
    model, mgrh_head, tokenizer, calibration_T = load_final_checkpoint(checkpoint_path, device)
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    mgrh_params = sum(p.numel() for p in mgrh_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "calibration_temperature": calibration_T,
        "model_info": {
            "encoder_params": encoder_params,
            "mgrh_params": mgrh_params,
            "total_model_params": total_params,
            "max_length": args.max_length,
        },
        "benchmarks": {},
    }

    # ===================================================================
    # 1. LISTWISE: Human benchmark (798 groups) — the gold standard
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 1: Listwise Ranking — Human Benchmark (798 groups)")
    logger.info("=" * 70)
    human_data = load_listwise_data(REPO_ROOT / "data/familyos/nli/relevance/human_benchmark_listwise.jsonl")
    if human_data:
        results["benchmarks"]["human_benchmark_listwise"] = benchmark_listwise(
            model, mgrh_head, tokenizer, human_data, device, args.max_length, "human_benchmark_listwise",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 2. LISTWISE: Holdout split (79 groups) — never seen during training
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 2: Listwise Ranking — Holdout (79 groups)")
    logger.info("=" * 70)
    holdout_data = load_listwise_data(REPO_ROOT / "data/familyos/nli/splits/stage_c/holdout.jsonl")
    if holdout_data:
        results["benchmarks"]["holdout_listwise"] = benchmark_listwise(
            model, mgrh_head, tokenizer, holdout_data, device, args.max_length, "holdout_listwise",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 3. LISTWISE: Dev split (79 groups) — validation during training
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 3: Listwise Ranking — Dev (79 groups)")
    logger.info("=" * 70)
    dev_data = load_listwise_data(REPO_ROOT / "data/familyos/nli/splits/stage_c/dev.jsonl")
    if dev_data:
        results["benchmarks"]["dev_listwise"] = benchmark_listwise(
            model, mgrh_head, tokenizer, dev_data, device, args.max_length, "dev_listwise",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 4. PAIRWISE: Hard negatives — entity_swap, temporal_shift
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 4: Pairwise — Hard Negatives (entity/temporal)")
    logger.info("=" * 70)
    hard_neg_data = load_triplet_data(
        str(REPO_ROOT / "data/familyos/embeddings/hard_negatives/triplets_*.jsonl"),
        max_samples=args.triplet_cap,
    )
    if hard_neg_data:
        results["benchmarks"]["hard_negatives_pairwise"] = benchmark_pairwise(
            model, mgrh_head, tokenizer, hard_neg_data, device, args.max_length, "hard_negatives",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 5. PAIRWISE: Wrong Person (entity swap mining)
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 5: Pairwise — Wrong Person (mined)")
    logger.info("=" * 70)
    wrong_person_data = load_triplet_data(
        str(REPO_ROOT / "data/familyos/embeddings/mined_v2/wrong_person/wrong_person_*.jsonl"),
        max_samples=args.triplet_cap,
    )
    if wrong_person_data:
        results["benchmarks"]["wrong_person_pairwise"] = benchmark_pairwise(
            model, mgrh_head, tokenizer, wrong_person_data, device, args.max_length, "wrong_person",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 6. PAIRWISE: Wrong Time (temporal shift mining)
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK 6: Pairwise — Wrong Time (mined)")
    logger.info("=" * 70)
    wrong_time_data = load_triplet_data(
        str(REPO_ROOT / "data/familyos/embeddings/mined_v2/wrong_time/wrong_time_*.jsonl"),
        max_samples=args.triplet_cap,
    )
    if wrong_time_data:
        results["benchmarks"]["wrong_time_pairwise"] = benchmark_pairwise(
            model, mgrh_head, tokenizer, wrong_time_data, device, args.max_length, "wrong_time",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # 7. PAIRWISE: Query-Doc pairs — skip if not triplet format
    # ===================================================================
    # query_doc data uses {query, document} format (pairs, not triplets)
    # so we score them directly and report score distribution instead.
    logger.info("=" * 70)
    logger.info("BENCHMARK 7: Query-Doc Score Distribution (mined pairs)")
    logger.info("=" * 70)
    query_doc_data = load_triplet_data(
        str(REPO_ROOT / "data/familyos/embeddings/mined_v2/query_doc/query_doc_*.jsonl"),
        max_samples=args.triplet_cap,
    )
    if query_doc_data and "query" in query_doc_data[0] and "document" in query_doc_data[0]:
        logger.info(f"  [query_doc] {len(query_doc_data)} pairs...")
        qd_pairs = [(item["query"], item["document"]) for item in query_doc_data]
        scores_list = score_pairs_batch(
            model, mgrh_head, tokenizer, qd_pairs, device, args.max_length, args.batch_size,
        )
        logger.info(f"    Scored {len(scores_list)} query-doc pairs")
        mean_s = sum(scores_list) / len(scores_list) if scores_list else 0.0
        std_s = (sum((x - mean_s) ** 2 for x in scores_list) / len(scores_list)) ** 0.5 if scores_list else 0.0
        results["benchmarks"]["query_doc_scores"] = {
            "dataset": "query_doc",
            "num_pairs": len(scores_list),
            "mean_score": round(mean_s, 4),
            "std_score": round(std_s, 4),
            "min_score": round(min(scores_list), 4) if scores_list else 0.0,
            "max_score": round(max(scores_list), 4) if scores_list else 0.0,
        }
    elif query_doc_data and "anchor" in query_doc_data[0]:
        results["benchmarks"]["query_doc_pairwise"] = benchmark_pairwise(
            model, mgrh_head, tokenizer, query_doc_data, device, args.max_length, "query_doc",
            batch_size=args.batch_size,
        )

    # ===================================================================
    # Summary
    # ===================================================================
    elapsed = time.time() - t0
    results["total_elapsed_seconds"] = round(elapsed, 1)

    # Write results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults written to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("MGRH FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Calibration T: {calibration_T}")
    print(f"Encoder: {encoder_params:,} params | MGRH: {mgrh_params:,} params")
    print(f"Total time: {elapsed:.0f}s")
    print()

    # Listwise results table
    print("-" * 80)
    print(f"{'Dataset':<30} {'Spearman':>10} {'AUC-ROC':>10} {'nDCG@10':>10} {'BiEnc Sp':>10} {'Lift Sp':>10}")
    print("-" * 80)
    for key in ["human_benchmark_listwise", "holdout_listwise", "dev_listwise"]:
        bm = results["benchmarks"].get(key)
        if bm:
            m = bm["mgrh"]
            b = bm["biencoder_baseline"]
            l = bm["lift_over_biencoder"]
            print(f"{bm['dataset']:<30} {m['avg_group_spearman']:>10.4f} {m['auc_roc_binary']:>10.4f} {m['avg_ndcg_at_10']:>10.4f} {b['avg_group_spearman']:>10.4f} {l['spearman_delta']:>+10.4f}")
    print()

    # Pairwise results table
    print("-" * 80)
    print(f"{'Dataset':<30} {'Accuracy':>10} {'Avg Margin':>12} {'BiEnc Acc':>10}")
    print("-" * 80)
    for key in ["hard_negatives_pairwise", "wrong_person_pairwise", "wrong_time_pairwise", "query_doc_pairwise"]:
        bm = results["benchmarks"].get(key)
        if bm:
            m = bm["mgrh"]
            b = bm["biencoder_baseline"]
            print(f"{bm['dataset']:<30} {m['accuracy']:>10.4f} {m['avg_margin']:>12.4f} {b['accuracy']:>10.4f}")
    # Query-doc score distribution (not pairwise)
    qd = results["benchmarks"].get("query_doc_scores")
    if qd:
        print(f"{'query_doc (score dist)':<30} {'mean=' + str(qd['mean_score']):>10} {'std=' + str(qd['std_score']):>12} {'n=' + str(qd['num_pairs']):>10}")
    print()

    # Hard negative slice breakdown
    for key in ["hard_negatives_pairwise", "wrong_person_pairwise", "wrong_time_pairwise", "query_doc_pairwise"]:
        bm = results["benchmarks"].get(key)
        if bm and bm.get("hard_negative_slices"):
            print(f"  Slices for {bm['dataset']}:")
            for stype, sdata in sorted(bm["hard_negative_slices"].items()):
                print(f"    {stype:<25} acc={sdata['accuracy']:.4f}  margin={sdata['avg_margin']:.4f}  n={sdata['count']}")
            print()

    # Grade distribution for human benchmark
    hb = results["benchmarks"].get("human_benchmark_listwise")
    if hb and hb.get("grade_distribution"):
        print("-" * 80)
        print("Score Distribution by Grade (Human Benchmark):")
        print(f"  {'Grade':<5} {'Label':<35} {'MGRH Mean':>10} {'MGRH Std':>10} {'BiEnc':>10} {'Count':>6}")
        for g, stats in sorted(hb["grade_distribution"].items()):
            print(f"  {g:<5} {stats['label']:<35} {stats['mgrh_mean']:>10.4f} {stats['mgrh_std']:>10.4f} {stats['bienc_mean']:>10.4f} {stats['count']:>6}")
        if hb.get("grade_3_vs_0_separation") is not None:
            print(f"\n  Grade 3 vs 0 separation: {hb['grade_3_vs_0_separation']:.4f}")
        print()

    # Calibration
    if hb and hb.get("calibration"):
        cal = hb["calibration"]
        print(f"  ECE (Expected Calibration Error): {cal['ece']:.4f}")

    # Gate check
    print("\n" + "=" * 80)
    print("GATE CHECK vs PLAN TARGETS")
    print("=" * 80)
    if hb:
        m = hb["mgrh"]
        targets = [
            ("Spearman > 0.70", m["avg_group_spearman"], 0.70),
            ("AUC-ROC > 0.85", m["auc_roc_binary"], 0.85),
            ("nDCG@10 > 0.83", m["avg_ndcg_at_10"], 0.83),
        ]
        for label, actual, target in targets:
            status = "PASS" if actual >= target else "FAIL"
            print(f"  [{status}] {label}: actual={actual:.4f}")

    holdout = results["benchmarks"].get("holdout_listwise")
    if holdout:
        m = holdout["mgrh"]
        targets = [
            ("Holdout Spearman > 0.50", m["avg_group_spearman"], 0.50),
            ("Holdout AUC > 0.80", m["auc_roc_binary"], 0.80),
        ]
        for label, actual, target in targets:
            status = "PASS" if actual >= target else "FAIL"
            print(f"  [{status}] {label}: actual={actual:.4f}")

    print()


if __name__ == "__main__":
    main()
